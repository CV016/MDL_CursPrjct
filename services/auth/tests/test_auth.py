"""
services/auth/tests/test_auth.py
=================================
Unit tests for the auth service.

These tests exercise the JWT helper functions and the rate-limiting logic
entirely in-process — no database or Redis connection is required.

Run with:
    pytest services/auth/tests/ -v
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest

# We import from the service module directly
# (the service is not installed as a package; add its parent to sys.path via conftest or pytest.ini)
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set required environment variables before importing the module
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET", "test_secret_key_for_unit_tests_only")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("JWT_EXPIRE_MINUTES", "60")

import main as auth_main  # noqa: E402  (must come after env setup)


# ---------------------------------------------------------------------------
# Token generation tests
# ---------------------------------------------------------------------------


class TestCreateAccessToken:
    """Verify that create_access_token produces well-formed JWTs."""

    def test_token_contains_expected_claims(self) -> None:
        token, jti = auth_main.create_access_token(
            user_id="user-123",
            username="alice",
            email="alice@example.com",
        )
        payload = jwt.decode(token, auth_main.JWT_SECRET, algorithms=[auth_main.JWT_ALGORITHM])

        assert payload["sub"] == "user-123"
        assert payload["username"] == "alice"
        assert payload["email"] == "alice@example.com"
        assert payload["jti"] == jti

    def test_jti_is_unique_across_calls(self) -> None:
        _, jti_a = auth_main.create_access_token("u1", "alice", "a@x.com")
        _, jti_b = auth_main.create_access_token("u1", "alice", "a@x.com")
        assert jti_a != jti_b, "Every token must carry a unique jti"

    def test_token_algorithm_is_hs256(self) -> None:
        token, _ = auth_main.create_access_token("u1", "alice", "a@x.com")
        header = jwt.get_unverified_header(token)
        assert header["alg"] == "HS256"


# ---------------------------------------------------------------------------
# Token expiry tests
# ---------------------------------------------------------------------------


class TestTokenExpiry:
    """Verify that expired tokens are rejected by decode_access_token."""

    def test_expired_token_raises_401(self) -> None:
        """Forge a token whose exp is in the past and confirm it is rejected."""
        from datetime import datetime, timedelta, timezone

        payload = {
            "sub": "user-999",
            "username": "bob",
            "email": "bob@example.com",
            "jti": "test-jti",
            "iat": datetime.now(tz=timezone.utc) - timedelta(hours=2),
            # exp is 1 hour ago
            "exp": datetime.now(tz=timezone.utc) - timedelta(hours=1),
        }
        expired_token: str = jwt.encode(payload, auth_main.JWT_SECRET, algorithm="HS256")

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            auth_main.decode_access_token(expired_token)

        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_valid_token_does_not_raise(self) -> None:
        token, _ = auth_main.create_access_token("u2", "carol", "carol@example.com")
        payload = auth_main.decode_access_token(token)
        assert payload["username"] == "carol"


# ---------------------------------------------------------------------------
# Invalid credential tests
# ---------------------------------------------------------------------------


class TestDecodeInvalidToken:
    """Verify that tampered or malformed tokens are rejected."""

    def test_wrong_secret_raises_401(self) -> None:
        import jwt as pyjwt
        from fastapi import HTTPException

        token = pyjwt.encode({"sub": "x"}, "wrong_secret", algorithm="HS256")
        with pytest.raises(HTTPException) as exc_info:
            auth_main.decode_access_token(token)
        assert exc_info.value.status_code == 401

    def test_malformed_token_raises_401(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            auth_main.decode_access_token("not.a.valid.jwt")
        assert exc_info.value.status_code == 401

    def test_empty_token_raises_401(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            auth_main.decode_access_token("")


# ---------------------------------------------------------------------------
# Password hashing tests
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    """Verify bcrypt hash / verify round-trips."""

    def test_hash_is_not_plaintext(self) -> None:
        hashed = auth_main.hash_password("secret123")
        assert hashed != "secret123"

    def test_correct_password_verifies(self) -> None:
        hashed = auth_main.hash_password("hunter2")
        assert auth_main.verify_password("hunter2", hashed) is True

    def test_wrong_password_fails(self) -> None:
        hashed = auth_main.hash_password("hunter2")
        assert auth_main.verify_password("wrong", hashed) is False

    def test_each_hash_is_unique_due_to_salt(self) -> None:
        h1 = auth_main.hash_password("same_password")
        h2 = auth_main.hash_password("same_password")
        assert h1 != h2, "bcrypt should produce distinct hashes due to random salt"


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


class TestRateLimit:
    """Verify the Redis-based rate limiter raises HTTP 429 after threshold."""

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self) -> None:
        """First RATE_LIMIT_MAX_ATTEMPTS calls must all pass."""
        mock_redis = AsyncMock()
        mock_redis.incr = AsyncMock(side_effect=range(1, auth_main.RATE_LIMIT_MAX_ATTEMPTS + 1))
        mock_redis.expire = AsyncMock()

        for _ in range(auth_main.RATE_LIMIT_MAX_ATTEMPTS):
            # Should not raise
            await auth_main.check_rate_limit("1.2.3.4", mock_redis)

    @pytest.mark.asyncio
    async def test_blocks_request_over_limit(self) -> None:
        from fastapi import HTTPException

        mock_redis = AsyncMock()
        # Simulate counter already at the max + 1
        mock_redis.incr = AsyncMock(return_value=auth_main.RATE_LIMIT_MAX_ATTEMPTS + 1)
        mock_redis.expire = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await auth_main.check_rate_limit("1.2.3.4", mock_redis)

        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_expire_called_only_on_first_increment(self) -> None:
        """
        The TTL must only be set when the counter is first created (incr returns 1).
        Subsequent increments must not reset the window.
        """
        mock_redis = AsyncMock()
        mock_redis.incr = AsyncMock(return_value=2)  # Not the first increment
        mock_redis.expire = AsyncMock()

        await auth_main.check_rate_limit("5.6.7.8", mock_redis)

        mock_redis.expire.assert_not_called()
