"""
services/question_gen/main.py
==============================
AI-DOC INTERACT — Question Generation Inference Worker

Responsibilities:
  - POST /generate : Accept an InferenceRequest (doc_id, variant, chunks)
    and return a list of generated questions.
  - Load a HuggingFace text-to-text (T5-based) question-generation model.
  - Variant A uses MODEL_NAME; variant B uses MODEL_NAME_B.
  - Log every inference run to MLflow.
  - Expose Prometheus metrics.

The question-generation model (e.g. valhalla/t5-base-qg-hl) expects the
input format:
    "generate question: <context>"
and returns one or more question strings.
"""

from __future__ import annotations

import os
import time
from typing import Any

import mlflow
import torch
from fastapi import FastAPI, HTTPException, status
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field
from transformers import pipeline, Pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME_A: str = os.environ.get("MODEL_NAME", "valhalla/t5-base-qg-hl")
MODEL_NAME_B: str = os.environ.get("MODEL_NAME_B", "mrm8488/t5-base-finetuned-question-generation-ap")
MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT: str = os.environ.get("MLFLOW_EXPERIMENT", "question_gen")

MAX_QUESTIONS_PER_CHUNK: int = int(os.environ.get("MAX_QUESTIONS_PER_CHUNK", "3"))
MAX_OUTPUT_LENGTH: int = int(os.environ.get("MAX_OUTPUT_LENGTH", "128"))

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

QGEN_REQUESTS = Counter(
    "question_gen_requests_total",
    "Total question-generation requests.",
    ["variant", "status"],
)

QGEN_LATENCY = Histogram(
    "question_gen_latency_seconds",
    "End-to-end question-generation latency in seconds.",
    ["variant"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

QGEN_QUESTIONS_GENERATED = Counter(
    "question_gen_questions_total",
    "Total number of questions generated.",
    ["variant"],
)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-DOC INTERACT — Question Generator",
    version="1.0.0",
    description="T5-based question generation worker with A/B model switching.",
)

app.mount("/metrics", make_asgi_app())

# ---------------------------------------------------------------------------
# Model pipelines
# ---------------------------------------------------------------------------

_device: str = "cuda" if torch.cuda.is_available() else "cpu"
_pipelines: dict[str, Pipeline] = {}


@app.on_event("startup")
async def on_startup() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    _pipelines["A"] = pipeline(
        "text2text-generation",
        model=MODEL_NAME_A,
        device=0 if _device == "cuda" else -1,
    )
    _pipelines["B"] = pipeline(
        "text2text-generation",
        model=MODEL_NAME_B,
        device=0 if _device == "cuda" else -1,
    )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChunkIn(BaseModel):
    chunk_id: int
    text: str
    token_count: int


class GenerateRequest(BaseModel):
    doc_id: str
    variant: str = Field(..., pattern="^[AB]$")
    chunks: list[ChunkIn] = Field(..., min_length=1)
    hint: str | None = None


class GenerateResponse(BaseModel):
    doc_id: str
    variant: str
    questions: list[str]
    latency_ms: float


# ---------------------------------------------------------------------------
# Helper: generate questions for a single chunk
# ---------------------------------------------------------------------------


def _generate_for_chunk(qgen_pipeline: Pipeline, chunk_text: str) -> list[str]:
    """
    Produce up to MAX_QUESTIONS_PER_CHUNK questions for a single chunk.

    The T5 QG model expects the prefix "generate question: " prepended to
    the context; some fine-tuned variants use a different prompt format,
    but this prefix is the most common convention.
    """
    prompt = f"generate question: {chunk_text}"
    outputs: list[dict[str, Any]] = qgen_pipeline(
        prompt,
        max_length=MAX_OUTPUT_LENGTH,
        num_return_sequences=MAX_QUESTIONS_PER_CHUNK,
        num_beams=MAX_QUESTIONS_PER_CHUNK,
        do_sample=False,
        truncation=True,
    )
    # Each output dict has key "generated_text"
    return [str(out["generated_text"]).strip() for out in outputs if out.get("generated_text")]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    """
    Generate questions for each document chunk and return the deduplicated list.
    """
    if body.variant not in _pipelines:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown variant '{body.variant}'.",
        )

    qgen_pipeline = _pipelines[body.variant]
    all_questions: list[str] = []

    start_ts = time.monotonic()
    try:
        with mlflow.start_run(run_name=f"qgen-{body.doc_id[:8]}-{body.variant}"):
            mlflow.log_params({
                "doc_id": body.doc_id,
                "variant": body.variant,
                "model": MODEL_NAME_A if body.variant == "A" else MODEL_NAME_B,
                "num_chunks": len(body.chunks),
            })

            for chunk in body.chunks:
                chunk_questions = _generate_for_chunk(qgen_pipeline, chunk.text)
                all_questions.extend(chunk_questions)

            # Deduplicate while preserving order
            seen: set[str] = set()
            unique_questions: list[str] = []
            for q in all_questions:
                if q not in seen:
                    seen.add(q)
                    unique_questions.append(q)

            latency_ms = (time.monotonic() - start_ts) * 1000

            mlflow.log_metrics({
                "latency_ms": latency_ms,
                "num_questions": len(unique_questions),
            })

    except Exception as exc:
        QGEN_REQUESTS.labels(variant=body.variant, status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question generation failed: {exc}",
        ) from exc

    QGEN_REQUESTS.labels(variant=body.variant, status="success").inc()
    QGEN_LATENCY.labels(variant=body.variant).observe(latency_ms / 1000)
    QGEN_QUESTIONS_GENERATED.labels(variant=body.variant).inc(len(unique_questions))

    return GenerateResponse(
        doc_id=body.doc_id,
        variant=body.variant,
        questions=unique_questions,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "device": _device}
