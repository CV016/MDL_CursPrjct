"""
services/auth/main.py
=====================
AI-DOC INTERACT — Authentication Service

Responsibilities:
  - POST /auth/register  : Create a new user account (bcrypt-hashed password).
  - POST /auth/token     : Validate credentials and issue a signed JWT.
  - GET  /auth/me        : Decode Bearer token and return the current user.

Rate limiting:
  - Login endpoint is capped at 5 attempts per minute per IP using a
    Redis sliding-window counter.  Exceeding the limit returns HTTP 429.

Storage:
  - User records are persisted in PostgreSQL via asyncpg.
  - Rate-limit counters live in Redis (TTL 60 s).

Security:
  - Passwords are hashed with bcrypt (passlib, cost factor 12).
  - JWTs are signed with HS256; expiry defaults to 60 minutes.
  - Each token carries a unique `jti` claim for future revocation support.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import asyncpg
import jwt
import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

# ---------------------------------------------------------------------------
# Configuration — resolved from environment variables set by docker-compose
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.environ["DATABASE_URL"]
REDIS_URL: str = os.environ["REDIS_URL"]
JWT_SECRET: str = os.environ["JWT_SECRET"]
JWT_ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES: int = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))

# Maximum login attempts per IP per 60-second window
RATE_LIMIT_MAX_ATTEMPTS: int = 5
RATE_LIMIT_WINDOW_SECONDS: int = 60

# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-DOC INTERACT — Auth Service",
    version="1.0.0",
    description="JWT-based authentication service with bcrypt password hashing and Redis rate limiting.",
)

# ---------------------------------------------------------------------------
# Shared clients — initialised at startup, closed at shutdown
# ---------------------------------------------------------------------------

_db_pool: asyncpg.Pool | None = None
_redis: aioredis.Redis | None = None  # type: ignore[type-arg]


@app.on_event("startup")
async def on_startup() -> None:
    global _db_pool, _redis
    _db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    _redis = aioredis.from_url(REDIS_URL, decode_responses=True)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    if _db_pool:
        await _db_pool.close()
    if _redis:
        await _redis.aclose()


def get_db() -> asyncpg.Pool:
    """FastAPI dependency that returns the active connection pool."""
    if _db_pool is None:
        raise RuntimeError("Database pool not initialised")
    return _db_pool


def get_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    """FastAPI dependency that returns the active Redis client."""
    if _redis is None:
        raise RuntimeError("Redis client not initialised")
    return _redis


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


def hash_password(plain: str) -> str:
    """Return bcrypt hash of `plain`."""
    return str(_pwd_context.hash(plain))


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if `plain` matches `hashed`."""
    return bool(_pwd_context.verify(plain, hashed))


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def create_access_token(user_id: str, username: str, email: str) -> tuple[str, str]:
    """
    Mint a new HS256 JWT.

    Returns:
        (encoded_token, jti) — caller should persist jti in the sessions table.
    """
    jti = str(uuid.uuid4())
    now = datetime.now(tz=timezone.utc)
    payload: dict[str, Any] = {
        "sub": user_id,
        "username": username,
        "email": email,
        "jti": jti,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    token: str = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token, jti


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT.

    Raises:
        HTTPException(401) if the token is expired or has an invalid signature.
    """
    try:
        payload: dict[str, Any] = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


# ---------------------------------------------------------------------------
# Rate limiting helper
# ---------------------------------------------------------------------------


async def check_rate_limit(ip: str, redis_client: aioredis.Redis) -> None:  # type: ignore[type-arg]
    """
    Enforce at most RATE_LIMIT_MAX_ATTEMPTS login attempts per IP per window.

    Uses a Redis INCR + EXPIRE pattern:
      - On the first attempt the key is created with a 60 s TTL.
      - On subsequent attempts within the window the counter is incremented.
      - When the counter exceeds the limit HTTP 429 is raised.
    """
    key = f"rate_limit:login:{ip}"
    count: int = await redis_client.incr(key)
    if count == 1:
        # First request in this window — set the expiry
        await redis_client.expire(key, RATE_LIMIT_WINDOW_SECONDS)
    if count > RATE_LIMIT_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many login attempts. Try again in {RATE_LIMIT_WINDOW_SECONDS} seconds.",
        )


# ---------------------------------------------------------------------------
# Request / response Pydantic models (service-local, mirroring shared.schemas)
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class RegisterResponse(BaseModel):
    user_id: str
    username: str
    email: str
    message: str = "User registered successfully."


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class MeResponse(BaseModel):
    user_id: str
    username: str
    email: str


# ---------------------------------------------------------------------------
# HTTP Bearer security scheme used by GET /auth/me
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(_bearer_scheme)],
) -> dict[str, Any]:
    """FastAPI dependency — validates Bearer token and returns its payload."""
    return decode_access_token(credentials.credentials)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/auth/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    db: Annotated[asyncpg.Pool, Depends(get_db)],
) -> RegisterResponse:
    """
    Create a new user account.

    - Checks that username and email are not already taken.
    - Stores bcrypt-hashed password.
    """
    async with db.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT user_id FROM users WHERE username = $1 OR email = $2",
            body.username,
            body.email,
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username or email already registered.",
            )

        new_id = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO users (user_id, username, email, password_hash)
            VALUES ($1, $2, $3, $4)
            """,
            new_id,
            body.username,
            body.email,
            hash_password(body.password),
        )

    return RegisterResponse(user_id=new_id, username=body.username, email=body.email)


@app.post("/auth/token", response_model=TokenResponse)
async def login(
    request: Request,
    body: LoginRequest,
    db: Annotated[asyncpg.Pool, Depends(get_db)],
    redis_client: Annotated[aioredis.Redis, Depends(get_redis)],  # type: ignore[type-arg]
) -> TokenResponse:
    """
    Authenticate credentials and return a signed JWT.

    Rate-limited to RATE_LIMIT_MAX_ATTEMPTS attempts per IP per minute.
    """
    # Resolve client IP — honour X-Forwarded-For when behind a reverse proxy
    client_ip: str = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    await check_rate_limit(client_ip, redis_client)

    async with db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT user_id, password_hash, email FROM users WHERE username = $1 AND is_active = TRUE",
            body.username,
        )

    if not row or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    token, jti = create_access_token(
        user_id=str(row["user_id"]),
        username=body.username,
        email=row["email"],
    )

    # Persist session record for audit purposes
    async with db.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO sessions (user_id, jti, expires_at)
            VALUES ($1, $2, NOW() + INTERVAL '1 hour')
            """,
            row["user_id"],
            jti,
        )

    return TokenResponse(
        access_token=token,
        expires_in=JWT_EXPIRE_MINUTES * 60,
    )


@app.get("/auth/me", response_model=MeResponse)
async def me(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> MeResponse:
    """Return the authenticated user's profile decoded from the Bearer token."""
    return MeResponse(
        user_id=current_user["sub"],
        username=current_user["username"],
        email=current_user["email"],
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Kubernetes / Docker liveness probe endpoint."""
    return {"status": "ok"}
