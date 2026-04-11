"""
services/gateway/main.py
========================
AI-DOC INTERACT — API Gateway Service

Responsibilities:
  - Single entry point for all client traffic.
  - Validates Bearer tokens by forwarding to the auth service.
  - Proxies document upload to the parser service.
  - Implements A/B routing: routes inference requests to summarizer or
    question_gen with configurable traffic splits (env: AB_VARIANT_B_PERCENT).
  - Persists every inference request to the experiment_logs table.
  - Caches inference results in Redis (TTL 10 minutes).
  - POST /feedback — stores thumbs-up/thumbs-down ratings.
  - Exposes /health for container readiness probes.

A/B routing strategy:
  - A random number in [0, 100) is drawn per request.
  - If the number < AB_VARIANT_B_PERCENT, variant B is used; otherwise A.
  - Both variants call the same inference worker endpoints; the variant label
    is passed inside InferenceRequest so the worker can select the correct
    model checkpoint / configuration.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from typing import Annotated, Any

import asyncpg
import httpx
import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, Request, Security, UploadFile, File, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUTH_SERVICE_URL: str = os.environ.get("AUTH_SERVICE_URL", "http://auth:8001")
PARSER_SERVICE_URL: str = os.environ.get("PARSER_SERVICE_URL", "http://parser:8002")
SUMMARIZER_SERVICE_URL: str = os.environ.get("SUMMARIZER_SERVICE_URL", "http://summarizer:8003")
QUESTION_GEN_SERVICE_URL: str = os.environ.get("QUESTION_GEN_SERVICE_URL", "http://question_gen:8004")
REDIS_URL: str = os.environ.get("REDIS_URL", "redis://redis:6379/4")
DATABASE_URL: str = os.environ.get("DATABASE_URL", "postgresql://aidoc:aidoc_secret@postgres:5432/aidoc")
AB_VARIANT_B_PERCENT: int = int(os.environ.get("AB_VARIANT_B_PERCENT", "20"))

# Redis TTL for cached inference results: 10 minutes
CACHE_TTL_SECONDS: int = 600

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-DOC INTERACT — API Gateway",
    version="1.0.0",
    description="Central routing layer with A/B inference splitting and feedback ingestion.",
)

# ---------------------------------------------------------------------------
# Shared clients
# ---------------------------------------------------------------------------

_db_pool: asyncpg.Pool | None = None
_redis: aioredis.Redis | None = None  # type: ignore[type-arg]
_http: httpx.AsyncClient | None = None


@app.on_event("startup")
async def on_startup() -> None:
    global _db_pool, _redis, _http
    _db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    _http = httpx.AsyncClient(timeout=120.0)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    if _db_pool:
        await _db_pool.close()
    if _redis:
        await _redis.aclose()
    if _http:
        await _http.aclose()


def get_db() -> asyncpg.Pool:
    if _db_pool is None:
        raise RuntimeError("DB pool not ready")
    return _db_pool


def get_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    if _redis is None:
        raise RuntimeError("Redis not ready")
    return _redis


def get_http() -> httpx.AsyncClient:
    if _http is None:
        raise RuntimeError("HTTP client not ready")
    return _http


# ---------------------------------------------------------------------------
# Auth token validation
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer()


async def validate_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(_bearer_scheme)],
    http: Annotated[httpx.AsyncClient, Depends(get_http)],
) -> dict[str, Any]:
    """
    Delegate token validation to the auth service GET /auth/me.
    Returns the decoded user payload on success.
    Raises HTTP 401 if the token is invalid or expired.
    """
    try:
        resp = await http.get(
            f"{AUTH_SERVICE_URL}/auth/me",
            headers={"Authorization": f"Bearer {credentials.credentials}"},
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Auth service unavailable: {exc}",
        ) from exc

    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )
    return dict(resp.json())


# ---------------------------------------------------------------------------
# A/B routing helper
# ---------------------------------------------------------------------------


def select_variant() -> str:
    """
    Return 'B' with probability AB_VARIANT_B_PERCENT / 100, else 'A'.
    Uses Python's random module which is sufficient for traffic splitting
    (cryptographic randomness is not required here).
    """
    return "B" if random.randint(0, 99) < AB_VARIANT_B_PERCENT else "A"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class InferenceResult(BaseModel):
    doc_id: str
    variant: str
    summary: str | None = None
    questions: list[str] | None = None
    latency_ms: float


class FeedbackRequest(BaseModel):
    doc_id: str
    variant: str = Field(..., pattern="^[AB]$")
    score: int = Field(..., ge=-1, le=1)
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str = "Feedback recorded successfully."


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/upload", response_model=InferenceResult)
async def upload_and_infer(
    file: Annotated[UploadFile, File(description="PDF, DOCX, or PPTX document.")],
    current_user: Annotated[dict[str, Any], Depends(validate_token)],
    db: Annotated[asyncpg.Pool, Depends(get_db)],
    redis_client: Annotated[aioredis.Redis, Depends(get_redis)],  # type: ignore[type-arg]
    http: Annotated[httpx.AsyncClient, Depends(get_http)],
) -> InferenceResult:
    """
    Full inference pipeline:
      1. Forward file to parser service.
      2. Check Redis cache for a previous result on this document.
      3. If cache miss: select A/B variant, call summarizer and question_gen
         workers concurrently, persist experiment log.
      4. Cache the result and return it.
    """
    doc_id = str(uuid.uuid4())
    variant = select_variant()

    # --- Step 1: Parse ---
    file_bytes = await file.read()
    try:
        parse_resp = await http.post(
            f"{PARSER_SERVICE_URL}/parse",
            files={"file": (file.filename, file_bytes, file.content_type)},
        )
        parse_resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail="Parser service error.") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    parsed = parse_resp.json()

    # --- Step 2: Cache lookup ---
    cache_key = f"inference:{hash(str(parsed['chunks']))}"
    cached = await redis_client.get(cache_key)
    if cached:
        cached_result: dict[str, Any] = json.loads(cached)
        return InferenceResult(**cached_result)

    # --- Step 3: Concurrent inference ---
    inference_payload = {
        "doc_id": doc_id,
        "variant": variant,
        "chunks": parsed["chunks"],
    }

    start_ts = time.monotonic()
    summarizer_task = asyncio.create_task(
        http.post(f"{SUMMARIZER_SERVICE_URL}/summarize", json=inference_payload)
    )
    qgen_task = asyncio.create_task(
        http.post(f"{QUESTION_GEN_SERVICE_URL}/generate", json=inference_payload)
    )
    summarizer_resp, qgen_resp = await asyncio.gather(summarizer_task, qgen_task, return_exceptions=True)
    latency_ms = (time.monotonic() - start_ts) * 1000

    # Parse responses tolerantly — inference workers may fail independently
    summary: str | None = None
    questions: list[str] | None = None

    if isinstance(summarizer_resp, httpx.Response) and summarizer_resp.status_code == 200:
        summary = summarizer_resp.json().get("summary")

    if isinstance(qgen_resp, httpx.Response) and qgen_resp.status_code == 200:
        questions = qgen_resp.json().get("questions")

    # --- Persist experiment log ---
    async with db.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO experiment_logs
                (doc_id, user_id, variant, service, input_tokens, latency_ms)
            VALUES ($1, $2, $3, 'gateway', $4, $5)
            """,
            uuid.UUID(doc_id),
            uuid.UUID(current_user["user_id"]),
            variant,
            sum(c["token_count"] for c in parsed["chunks"]),
            latency_ms,
        )

    result = InferenceResult(
        doc_id=doc_id,
        variant=variant,
        summary=summary,
        questions=questions,
        latency_ms=round(latency_ms, 2),
    )

    # --- Step 4: Cache result ---
    await redis_client.setex(cache_key, CACHE_TTL_SECONDS, result.model_dump_json())

    return result


@app.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    body: FeedbackRequest,
    current_user: Annotated[dict[str, Any], Depends(validate_token)],
    db: Annotated[asyncpg.Pool, Depends(get_db)],
) -> FeedbackResponse:
    """Persist a thumbs-up / thumbs-down rating from the frontend."""
    feedback_id = str(uuid.uuid4())
    async with db.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO feedback (feedback_id, doc_id, user_id, variant, score, comment)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            uuid.UUID(feedback_id),
            uuid.UUID(body.doc_id),
            uuid.UUID(current_user["user_id"]),
            body.variant,
            body.score,
            body.comment,
        )
    return FeedbackResponse(feedback_id=feedback_id)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
