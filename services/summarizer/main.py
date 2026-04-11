"""
services/summarizer/main.py
============================
AI-DOC INTERACT — Summarization Inference Worker

Responsibilities:
  - POST /summarize : Accept an InferenceRequest (doc_id, variant, chunks)
    and return a combined summary.
  - Load a HuggingFace summarization model at startup.
  - Variant A uses the primary model; variant B uses an alternate checkpoint
    identified by the MODEL_NAME_B environment variable.
  - Log every inference run to MLflow (latency, token counts, model name).
  - Expose Prometheus metrics: inference latency histogram, request counter.

GPU:
  - The container is granted GPU access via NVIDIA Container Toolkit.
  - PyTorch automatically uses CUDA if available; falls back to CPU.
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

MODEL_NAME_A: str = os.environ.get("MODEL_NAME", "facebook/bart-large-cnn")
MODEL_NAME_B: str = os.environ.get("MODEL_NAME_B", "sshleifer/distilbart-cnn-12-6")
MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT: str = os.environ.get("MLFLOW_EXPERIMENT", "summarizer")

# Maximum tokens generated in the summary
MAX_SUMMARY_LENGTH: int = int(os.environ.get("MAX_SUMMARY_LENGTH", "256"))
MIN_SUMMARY_LENGTH: int = int(os.environ.get("MIN_SUMMARY_LENGTH", "64"))

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

SUMMARIZER_REQUESTS = Counter(
    "summarizer_requests_total",
    "Total summarization requests.",
    ["variant", "status"],
)

SUMMARIZER_LATENCY = Histogram(
    "summarizer_latency_seconds",
    "End-to-end summarization latency in seconds.",
    ["variant"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-DOC INTERACT — Summarizer",
    version="1.0.0",
    description="HuggingFace summarization inference worker with A/B model switching.",
)

app.mount("/metrics", make_asgi_app())

# ---------------------------------------------------------------------------
# Model registry — loaded lazily at startup
# ---------------------------------------------------------------------------

_device: str = "cuda" if torch.cuda.is_available() else "cpu"
_pipelines: dict[str, Pipeline] = {}


@app.on_event("startup")
async def on_startup() -> None:
    """
    Load both model variants at startup so the first request is not penalised
    by a cold model load.  Models are loaded onto GPU when available.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    _pipelines["A"] = pipeline(
        "summarization",
        model=MODEL_NAME_A,
        device=0 if _device == "cuda" else -1,
    )
    _pipelines["B"] = pipeline(
        "summarization",
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


class SummarizeRequest(BaseModel):
    doc_id: str
    variant: str = Field(..., pattern="^[AB]$")
    chunks: list[ChunkIn] = Field(..., min_length=1)
    hint: str | None = None


class SummarizeResponse(BaseModel):
    doc_id: str
    variant: str
    summary: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(body: SummarizeRequest) -> SummarizeResponse:
    """
    Summarize the provided document chunks.

    - Concatenates all chunks into a single input string (within model limits).
    - Routes to the model checkpoint corresponding to the requested variant.
    - Logs the run to MLflow for experiment tracking.
    """
    if body.variant not in _pipelines:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown variant '{body.variant}'. Must be 'A' or 'B'.",
        )

    # Concatenate chunks; the summarization model will handle long inputs via
    # its own truncation; we join with a separator to preserve paragraph breaks.
    full_text = " ".join(chunk.text for chunk in body.chunks)
    summarizer_pipeline = _pipelines[body.variant]

    start_ts = time.monotonic()
    try:
        with mlflow.start_run(run_name=f"summarize-{body.doc_id[:8]}-{body.variant}"):
            mlflow.log_params({
                "doc_id": body.doc_id,
                "variant": body.variant,
                "model": MODEL_NAME_A if body.variant == "A" else MODEL_NAME_B,
                "num_chunks": len(body.chunks),
            })

            output: list[dict[str, Any]] = summarizer_pipeline(
                full_text,
                max_length=MAX_SUMMARY_LENGTH,
                min_length=MIN_SUMMARY_LENGTH,
                do_sample=False,
                truncation=True,
            )
            summary_text: str = output[0]["summary_text"]
            latency_ms = (time.monotonic() - start_ts) * 1000

            mlflow.log_metrics({
                "latency_ms": latency_ms,
                "input_chars": len(full_text),
                "output_chars": len(summary_text),
            })

    except Exception as exc:
        SUMMARIZER_REQUESTS.labels(variant=body.variant, status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {exc}",
        ) from exc

    SUMMARIZER_REQUESTS.labels(variant=body.variant, status="success").inc()
    SUMMARIZER_LATENCY.labels(variant=body.variant).observe(latency_ms / 1000)

    return SummarizeResponse(
        doc_id=body.doc_id,
        variant=body.variant,
        summary=summary_text,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "device": _device}
