"""
backend/main.py
===============
AI-DOC INTERACT — Backend API Service

This is the single FastAPI application that implements all backend logic for
the academic MLOps pipeline:

  - Document parsing: PDF via PyPDF2, DOCX via python-docx, PPTX via python-pptx.
  - Non-parametric drift detection: word count and Flesch-Kincaid Grade Level
    (computed with textstat) are tested against a hardcoded reference population
    using the Kolmogorov-Smirnov percentile rank and Wasserstein (Earth Mover's)
    distance. Drift is flagged when either metric exceeds its threshold.
  - Thompson Sampling A/B model routing: maintains a Beta(alpha, beta)
    posterior per model from MLflow feedback history; samples from each
    posterior and routes to the model whose sample is higher. Exploration
    and exploitation are balanced automatically — no fixed epsilon required.
  - Inference: facebook/bart-large-cnn for summarisation; google/flan-t5-base
    for question generation. Both models are loaded once at startup.
  - MLflow experiment tracking: each /process call opens a new run, logs
    parameters, metrics, a drift tag, and saves a text artifact. The run_id
    is returned to the frontend so the user's feedback can be appended.
  - Feedback ingestion: /feedback resumes the MLflow run by run_id and logs
    the user_satisfaction_score metric (1 = thumbs-up, 0 = thumbs-down).
  - Automated judge (optional): after each /process, a background task scores
    source vs summary with an NLI cross-encoder; logs to MLflow and binarizes
    the entailment probability to update Thompson Sampling (same path as 👍/👎).
  - Prometheus: prometheus-fastapi-instrumentator exposes HTTP traffic metrics on
    /metrics together with custom ai_doc_* counters (single registry).

Endpoints:
  POST /process  — accepts a multipart file upload; returns summary, questions,
                   run_id, drift status, and inference latency.
  POST /feedback — accepts run_id + boolean score; logs to MLflow.
  GET  /health   — liveness probe for Docker health checks.

Authentication:
  A hardcoded mock Bearer token is used for all protected endpoints.
  The Streamlit frontend sends MOCK_JWT_TOKEN from the environment. This
  satisfies the "JWT token in header" requirement without a live identity
  provider, which is appropriate for an academic project.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import threading
import time
from collections import deque
from typing import Any

import mlflow
import mlflow.tracking
import numpy as np
from prometheus_client import Counter, Gauge, Histogram
from scipy.stats import percentileofscore, wasserstein_distance
import textstat
import torch
from docx import Document as DocxDocument
from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, UploadFile, status
from pptx import Presentation
from pydantic import BaseModel, ConfigDict
from PyPDF2 import PdfReader
from prometheus_fastapi_instrumentator import Instrumentator
from sentence_transformers import CrossEncoder
from transformers import Pipeline, pipeline

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all values read from environment variables with sensible
# defaults so the service starts correctly both inside Docker and locally.
# ---------------------------------------------------------------------------

# MLflow tracking server URI (container name resolved by Docker's DNS)
MLFLOW_TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI", "http://mlflow-server:5000"
)

# Name of the MLflow experiment to log runs under
MLFLOW_EXPERIMENT_NAME: str = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "ai_doc_interact"
)

# HuggingFace model identifiers — downloaded from the Hub on first startup
MODEL_BART: str = os.environ.get("MODEL_BART", "facebook/bart-large-cnn")
MODEL_FLAN: str = os.environ.get("MODEL_FLAN", "google/flan-t5-base")

# Shared mock JWT token — must match what the frontend sends
MOCK_JWT_TOKEN: str = os.environ.get(
    "MOCK_JWT_TOKEN", "mock-jwt-token-for-academic-project"
)

# LLM-as-a-judge: NLI cross-encoder scores premise/hypothesis entailment; runs in BackgroundTasks
MODEL_JUDGE: str = os.environ.get(
    "MODEL_JUDGE", "cross-encoder/nli-deberta-base"
)
JUDGE_ENABLED: bool = os.environ.get("JUDGE_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
JUDGE_THRESHOLD: float = float(os.environ.get("JUDGE_THRESHOLD", "0.65"))

# ---------------------------------------------------------------------------
# Drift detection — non-parametric reference population
#
# 50 sample points per feature, representative of real academic and
# professional documents.  Word counts are right-skewed (most documents are
# short; a long tail of large reports exists).  FK grades are centred around
# the college-level reading range (grade 12-16).
#
# Drift is flagged when EITHER condition holds for any feature:
#   1. Percentile rank falls outside [DRIFT_PERCENTILE_LOW, DRIFT_PERCENTILE_HIGH]
#      — the incoming document is a statistical outlier relative to the reference CDF.
#   2. Wasserstein distance exceeds DRIFT_WASSERSTEIN_THRESHOLDS[feature]
#      — the "cost" of transforming the live distribution to the baseline is too high.
# ---------------------------------------------------------------------------

DRIFT_REFERENCE: dict[str, np.ndarray] = {
    "word_count": np.array([
        180,  220,  260,  300,  340,  380,  420,  460,  500,  550,
        600,  650,  700,  750,  800,  880,  960,  1050, 1150, 1250,
        1380, 1520, 1680, 1850, 2050, 2250, 2500, 2750, 3050, 3400,
        3800, 4200, 4700, 5200, 5800, 6500, 7200, 8000, 8900, 9900,
        320,  480,  640,  920,  1100, 1450, 1750, 2100, 2600, 3200,
    ], dtype=float),
    "flesch_kincaid_grade": np.array([
        8.0,  8.5,  9.0,  9.5,  10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
        13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5,
        18.0, 18.5, 19.0, 19.5, 20.0, 9.2,  9.8,  10.2, 10.8, 11.2,
        11.8, 12.2, 12.8, 13.2, 13.8, 14.2, 14.8, 15.2, 15.8, 16.2,
        8.3,  9.3,  10.3, 11.3, 12.3, 13.3, 14.3, 15.3, 16.3, 17.3,
    ], dtype=float),
}

# Percentile bounds for the point-anomaly test (per-request)
DRIFT_PERCENTILE_LOW: float = 2.5
DRIFT_PERCENTILE_HIGH: float = 97.5

# Wasserstein distance thresholds — compared against the rolling window, not a
# single point. Expressed in each feature's own units.
DRIFT_WASSERSTEIN_THRESHOLDS: dict[str, float] = {
    "word_count": 2000.0,          # raw word-count units
    "flesch_kincaid_grade": 5.0,   # grade-level units
}

# Rolling window: retain the last N observed feature values.
# Wasserstein is only computed once the window contains at least
# DRIFT_WINDOW_MIN_SAMPLES entries; before that only the point-anomaly
# check runs.
DRIFT_WINDOW_SIZE: int = 10
DRIFT_WINDOW_MIN_SAMPLES: int = 5

# Module-level stateful buffers — populated on every /process request.
# A lock ensures safe concurrent access under Uvicorn's async workers.
_drift_windows: dict[str, deque[float]] = {
    "word_count": deque(maxlen=DRIFT_WINDOW_SIZE),
    "flesch_kincaid_grade": deque(maxlen=DRIFT_WINDOW_SIZE),
}
_drift_window_lock: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# Prometheus metrics definitions
#
# All metric names are prefixed with "ai_doc_" to avoid collision with
# default process/platform metrics that prometheus-client registers.
# Labels allow Prometheus and Grafana to slice data by model, file type, etc.
# ---------------------------------------------------------------------------

# Counts every completed /process request
REQUEST_COUNT: Counter = Counter(
    "ai_doc_requests_total",
    "Total number of document analysis requests completed.",
    ["model_name", "file_type", "drift_status"],
)

# Measures wall-clock inference time per request (in seconds)
REQUEST_LATENCY: Histogram = Histogram(
    "ai_doc_inference_latency_seconds",
    "End-to-end inference latency in seconds.",
    ["model_name"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 120.0],
)

# Counts drift events by type
DRIFT_COUNTER: Counter = Counter(
    "ai_doc_drift_total",
    "Number of requests that triggered each drift status.",
    ["drift_status"],
)

# Counts feedback submissions
FEEDBACK_COUNTER: Counter = Counter(
    "ai_doc_feedback_total",
    "Number of feedback submissions by model and outcome.",
    ["model_name", "outcome"],  # outcome: "thumbs_up" or "thumbs_down"
)

AUTOMATED_JUDGE_COUNTER: Counter = Counter(
    "ai_doc_automated_judge_total",
    "NLI automated judge verdicts after binarization.",
    ["model_name", "verdict"],  # verdict: thumbs_up or thumbs_down
)

# Live Thompson Sampling posterior parameters — updated on every /feedback call
THOMPSON_ALPHA: Gauge = Gauge(
    "ai_doc_thompson_alpha",
    "Current alpha (success count + prior) of each model's Beta posterior.",
    ["model_name"],
)

THOMPSON_BETA: Gauge = Gauge(
    "ai_doc_thompson_beta",
    "Current beta (failure count + prior) of each model's Beta posterior.",
    ["model_name"],
)

# Current size of the rolling window used for Wasserstein drift detection
DRIFT_WINDOW_GAUGE: Gauge = Gauge(
    "ai_doc_drift_window_size",
    "Number of observations currently in the rolling drift window.",
)

# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-DOC INTERACT — Backend API",
    version="1.0.0",
    description=(
        "Single backend service implementing document parsing, non-parametric "
        "drift detection (KS percentile rank + Wasserstein distance), Thompson "
        "Sampling A/B model routing, MLflow experiment tracking, and feedback ingestion."
    ),
)

# ---------------------------------------------------------------------------
# Model registry
#
# Both pipelines are stored in this module-level dict, populated during the
# startup event. Using a dict avoids global variables while keeping the
# singleton pattern clean.
# ---------------------------------------------------------------------------

# Device selection: prefer GPU when available for faster inference
_device: str = "cuda" if torch.cuda.is_available() else "cpu"
_device_idx: int = 0 if _device == "cuda" else -1

# Keyed by model short-name ("bart" or "flan") for clear lookup
_models: dict[str, Pipeline] = {}

# NLI cross-encoder for automated grading — kept on CPU so BART/FLAN keep the GPU
_judge_ce: CrossEncoder | None = None


@app.on_event("startup")
async def on_startup() -> None:
    """
    Load both HuggingFace model pipelines and configure MLflow.

    Running this at startup instead of lazily on the first request ensures
    every user request benefits from pre-loaded model weights — a key MLOps
    practice to eliminate cold-start latency from the critical path.
    """
    logger.info("Configuring MLflow: tracking_uri=%s", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # If the experiment was soft-deleted through the MLflow UI, restore it
    # automatically so set_experiment never crashes on startup.  MLflow's
    # soft-delete keeps the record in SQLite but marks it DELETED; calling
    # set_experiment on a DELETED experiment raises MlflowException.
    _client = mlflow.tracking.MlflowClient()
    _existing = _client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if _existing is not None and _existing.lifecycle_stage == "deleted":
        _client.restore_experiment(_existing.experiment_id)
        logger.warning(
            "MLflow experiment '%s' was soft-deleted; restored automatically.",
            MLFLOW_EXPERIMENT_NAME,
        )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("Loading summarisation model: %s (device=%s)", MODEL_BART, _device)
    _models["bart"] = pipeline(
        "summarization",
        model=MODEL_BART,
        device=_device_idx,
    )

    logger.info("Loading question-generation model: %s (device=%s)", MODEL_FLAN, _device)
    _models["flan"] = pipeline(
        "text2text-generation",
        model=MODEL_FLAN,
        device=_device_idx,
    )

    global _judge_ce
    if JUDGE_ENABLED:
        logger.info(
            "Loading NLI judge model: %s (device=cpu; avoids competing with summarisers)",
            MODEL_JUDGE,
        )
        _judge_ce = CrossEncoder(MODEL_JUDGE, device="cpu")
    else:
        logger.info("Automated judge disabled (JUDGE_ENABLED=false).")
        _judge_ce = None

    # Seed the Thompson Sampling gauges with the initial Beta(1,1) prior so
    # Grafana has non-null values from the first Prometheus scrape.
    for _m in ("bart", "flan"):
        THOMPSON_ALPHA.labels(model_name=_m).set(1)
        THOMPSON_BETA.labels(model_name=_m).set(1)

    logger.info("Startup complete. Both models are loaded and ready.")


# ---------------------------------------------------------------------------
# Authentication helper
# ---------------------------------------------------------------------------


def _validate_token(authorization: str | None) -> None:
    """
    Validate the Bearer token in the Authorization header.

    Checks that the header follows the "Bearer <token>" format and that the
    token matches MOCK_JWT_TOKEN. In a production system this function would
    cryptographically verify a signed JWT instead of doing a string comparison.

    Raises:
        HTTPException(401) on any authentication failure.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must use the Bearer scheme.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if parts[1] != MOCK_JWT_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Document parsing helpers
# ---------------------------------------------------------------------------


def _extract_text_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF using PyPDF2.

    Iterates through every page and concatenates the extracted text.
    Returns an empty string if no text layer is present (e.g. scanned images).
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            pages.append(extracted)
    return "\n".join(pages)


def _extract_text_docx(file_bytes: bytes) -> str:
    """
    Extract paragraph text from a DOCX file using python-docx.

    Only non-empty paragraphs are included; purely whitespace paragraphs
    (used as visual spacing in Word) are filtered out.
    """
    doc = DocxDocument(io.BytesIO(file_bytes))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def _extract_text_pptx(file_bytes: bytes) -> str:
    """
    Extract text from every slide of a PPTX file using python-pptx.

    Iterates through all shapes on each slide and collects text frame content.
    Each slide's text is joined with newlines; slides are separated by blank
    lines for readability.
    """
    prs = Presentation(io.BytesIO(file_bytes))
    slides: list[str] = []
    for slide in prs.slides:
        slide_lines: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs).strip()
                    if line:
                        slide_lines.append(line)
        if slide_lines:
            slides.append("\n".join(slide_lines))
    return "\n\n".join(slides)


def _parse_document(file_bytes: bytes, filename: str) -> str:
    """
    Route a file to the appropriate text extractor based on its extension.

    Returns:
        The full extracted text as a single string.
    Raises:
        HTTPException(415) for unsupported file types.
        HTTPException(422) if parsing fails.
    """
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    try:
        if extension == "pdf":
            return _extract_text_pdf(file_bytes)
        elif extension == "docx":
            return _extract_text_docx(file_bytes)
        elif extension == "pptx":
            return _extract_text_pptx(file_bytes)
        else:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"Unsupported file type '{extension}'. "
                    "Accepted types: pdf, docx, pptx."
                ),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse document: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def _point_anomaly(feature: str, value: float) -> tuple[float, bool]:
    """
    Point-anomaly test: locate a single value in the reference CDF.

    Uses scipy.stats.percentileofscore — a non-parametric test that makes no
    assumption about the shape of the underlying distribution.

    Returns:
        (percentile_rank, is_anomaly) where is_anomaly is True when the value
        falls outside [DRIFT_PERCENTILE_LOW, DRIFT_PERCENTILE_HIGH].
    """
    pct = float(percentileofscore(DRIFT_REFERENCE[feature], value, kind="rank"))
    is_anomaly = pct < DRIFT_PERCENTILE_LOW or pct > DRIFT_PERCENTILE_HIGH
    return round(pct, 2), is_anomaly


def _window_drift(feature: str, window: list[float]) -> tuple[float, bool]:
    """
    Distributional-drift test: compare a rolling window of live observations
    against the reference population using the Wasserstein (Earth Mover's)
    Distance.

    wasserstein_distance(window, reference) measures the minimum "work" needed
    to transform the live distribution into the reference shape.  Unlike the
    single-point case, comparing two real distributions here produces a
    statistically meaningful result — the metric rises when traffic
    systematically shifts (e.g. users consistently uploading larger documents).

    Returns:
        (wasserstein_distance, is_drift) where is_drift is True when the
        distance exceeds DRIFT_WASSERSTEIN_THRESHOLDS[feature].
    """
    w_dist = float(wasserstein_distance(window, DRIFT_REFERENCE[feature]))
    is_drift = w_dist > DRIFT_WASSERSTEIN_THRESHOLDS[feature]
    return round(w_dist, 4), is_drift


def _detect_drift(text: str) -> dict[str, Any]:
    """
    Two-tier drift assessment for the incoming document.

    Tier 1 — Point anomaly (every request):
        percentileofscore(reference, value) flags documents that are
        outliers relative to the baseline CDF.  Catches a single abnormal
        request immediately.

    Tier 2 — Distributional drift (once window has >= DRIFT_WINDOW_MIN_SAMPLES):
        wasserstein_distance(rolling_window, reference) detects systematic
        shifts in the traffic profile over time.  Catches the case where
        individual documents are within bounds but the overall population
        is slowly moving — the mathematically correct use of Wasserstein.

    drift_status encoding (logged as an MLflow tag):
        "normal"                — neither test triggered.
        "point_anomaly"         — this document is an outlier; window is fine.
        "window_drift"          — rolling window has drifted; this document is normal.
        "point_anomaly|window_drift" — both conditions are active simultaneously.

    Returns a flat dict of all feature values and test statistics suitable for
    direct logging to MLflow as metrics and tags.
    """
    word_count: int = len(text.split())
    fk_grade: float = textstat.flesch_kincaid_grade(text)

    # --- Tier 1: point-anomaly test ---
    wc_pct, wc_anomaly = _point_anomaly("word_count", float(word_count))
    fk_pct, fk_anomaly = _point_anomaly("flesch_kincaid_grade", fk_grade)
    is_point_anomaly: bool = wc_anomaly or fk_anomaly

    # --- Rolling window update (thread-safe) ---
    with _drift_window_lock:
        _drift_windows["word_count"].append(float(word_count))
        _drift_windows["flesch_kincaid_grade"].append(fk_grade)
        wc_window = list(_drift_windows["word_count"])
        fk_window = list(_drift_windows["flesch_kincaid_grade"])

    # --- Tier 2: distributional-drift test ---
    wc_wass: float = 0.0
    fk_wass: float = 0.0
    is_window_drift: bool = False
    window_size: int = len(wc_window)

    if window_size >= DRIFT_WINDOW_MIN_SAMPLES:
        wc_wass, wc_wdrift = _window_drift("word_count", wc_window)
        fk_wass, fk_wdrift = _window_drift("flesch_kincaid_grade", fk_window)
        is_window_drift = wc_wdrift or fk_wdrift

    # --- Compose drift_status tag ---
    flags: list[str] = []
    if is_point_anomaly:
        flags.append("point_anomaly")
    if is_window_drift:
        flags.append("window_drift")
    drift_status: str = "|".join(flags) if flags else "normal"

    logger.info(
        "Drift — wc=%d (pct=%.1f anomaly=%s wass=%.1f) "
        "fk=%.2f (pct=%.1f anomaly=%s wass=%.2f) "
        "window=%d/%d status=%s",
        word_count, wc_pct, wc_anomaly, wc_wass,
        fk_grade, fk_pct, fk_anomaly, fk_wass,
        window_size, DRIFT_WINDOW_SIZE, drift_status,
    )

    return {
        "word_count": word_count,
        "flesch_kincaid_grade": round(fk_grade, 2),
        "wc_percentile": wc_pct,
        "fk_percentile": fk_pct,
        "wc_wasserstein": wc_wass,
        "fk_wasserstein": fk_wass,
        "window_size": window_size,
        "drift_status": drift_status,
    }


# ---------------------------------------------------------------------------
# Thompson Sampling A/B model router — in-memory state
# ---------------------------------------------------------------------------
#
# State is kept entirely in memory and updated when /feedback is called or when
# the automated NLI judge finishes (BackgroundTasks).
# This eliminates an MLflow database read on every /process request, which
# would otherwise become the dominant source of inference latency.
#
# Cold-start / crash safety: initialising with alpha=1, beta=1 (the uniform
# Beta prior) means np.random.beta is always called with valid parameters
# (both strictly > 0) even before any feedback has been collected.  A
# Beta(1, 1) posterior represents complete uncertainty — both models are
# equally likely to be selected until evidence accumulates.

_thompson_state: dict[str, dict[str, int]] = {
    "bart": {"alpha": 1, "beta": 1},
    "flan": {"alpha": 1, "beta": 1},
}

# Protects _thompson_state and _run_model_map from concurrent writes under
# Uvicorn's threaded workers.
_thompson_lock: threading.Lock = threading.Lock()

# Maps MLflow run_id → model short-name so the /feedback endpoint can update
# the correct posterior without querying MLflow.  Capped at 2 000 entries to
# bound memory usage on a long-running server.
_run_model_map: dict[str, str] = {}
_RUN_MAP_MAX: int = 2_000


def _register_run(run_id: str, model: str) -> None:
    """Store the run_id → model mapping; evict oldest entry when the cap is hit."""
    with _thompson_lock:
        if len(_run_model_map) >= _RUN_MAP_MAX:
            # dict preserves insertion order in Python 3.7+; pop the oldest key
            oldest = next(iter(_run_model_map))
            del _run_model_map[oldest]
        _run_model_map[run_id] = model


def _select_model() -> str:
    """
    Select a model using Thompson Sampling against the in-memory Beta posteriors.

    One value is sampled from Beta(alpha, beta) for each model.  The model
    with the higher draw wins the request.  No database query is made here —
    the posteriors are updated when /feedback runs or the automated judge finishes.
    """
    with _thompson_lock:
        samples: dict[str, float] = {
            model: float(np.random.beta(state["alpha"], state["beta"]))
            for model, state in _thompson_state.items()
        }

    chosen: str = max(samples, key=lambda m: samples[m])

    with _thompson_lock:
        s = _thompson_state

    logger.info(
        "Thompson Sampling: selected '%s' "
        "(bart sample=%.4f α=%d β=%d | flan sample=%.4f α=%d β=%d)",
        chosen,
        samples["bart"],
        s["bart"]["alpha"], s["bart"]["beta"],
        samples["flan"],
        s["flan"]["alpha"], s["flan"]["beta"],
    )
    return chosen


def _update_thompson(model: str, thumbs_up: bool) -> None:
    """
    Apply one feedback observation to the model's Beta posterior.

    thumbs_up=True  → increment alpha (success count).
    thumbs_up=False → increment beta  (failure count).
    """
    key = "alpha" if thumbs_up else "beta"
    with _thompson_lock:
        _thompson_state[model][key] += 1
        new_alpha = _thompson_state[model]["alpha"]
        new_beta = _thompson_state[model]["beta"]

    # Mirror the updated posterior into Prometheus gauges so Grafana always
    # shows the current state of the A/B router.
    THOMPSON_ALPHA.labels(model_name=model).set(new_alpha)
    THOMPSON_BETA.labels(model_name=model).set(new_beta)

    logger.info(
        "Thompson posterior updated — model='%s' %s "
        "(α=%d β=%d)",
        model,
        "thumbs_up" if thumbs_up else "thumbs_down",
        new_alpha,
        new_beta,
    )


def _entailment_score(premise: str, hypothesis: str) -> float:
    """
    Return softmax probability of the **entailment** class for an NLI pair.

    The cross-encoder was trained on (premise, hypothesis) pairs; here the
    source document is the premise and the generated summary is the hypothesis.
    """
    if _judge_ce is None:
        return 0.0
    p_trunc = premise[:12000]
    h_trunc = hypothesis[:2000]
    probs = _judge_ce.predict([[p_trunc, h_trunc]], apply_softmax=True)
    row = np.asarray(probs[0]).flatten()
    id2label = getattr(_judge_ce.model.config, "id2label", None) or {}
    for idx_key, lab in id2label.items():
        if str(lab).lower() == "entailment":
            return float(row[int(idx_key)])
    return float(np.max(row))


def automated_judge_worker(
    source_text: str,
    generated_summary: str,
    model_used: str,
    run_id: str,
) -> None:
    """
    Runs after the HTTP response is returned. Logs NLI scores to MLflow and
    applies the binarized reward to the Thompson router (same update path as 👍/👎).
    """
    if _judge_ce is None or not source_text.strip() or not generated_summary.strip():
        return
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        score = _entailment_score(source_text, generated_summary)
        is_thumbs_up = score > JUDGE_THRESHOLD
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("automated_reward_score", float(score))
            mlflow.log_metric("automated_is_thumbs_up", float(int(is_thumbs_up)))

        if model_used in _thompson_state:
            _update_thompson(model_used, thumbs_up=is_thumbs_up)
            AUTOMATED_JUDGE_COUNTER.labels(
                model_name=model_used,
                verdict="thumbs_up" if is_thumbs_up else "thumbs_down",
            ).inc()

        logger.info(
            "Automated judge — run_id=%s  model=%s  score=%.4f  thumbs_up=%s",
            run_id,
            model_used,
            score,
            is_thumbs_up,
        )
    except Exception as exc:
        logger.exception("Automated judge failed for run_id=%s: %s", run_id, exc)


# ---------------------------------------------------------------------------
# Inference helpers — tokenizer utilities
# ---------------------------------------------------------------------------

# Safe chunk sizes: headroom below each model's hard position-embedding limit
# to accommodate the special tokens (BOS, EOS, padding) that the pipeline
# adds automatically around every input.
BART_CHUNK_TOKENS: int = 900   # BART limit = 1024; 124-token headroom
FLAN_CHUNK_TOKENS: int = 435   # FLAN-T5-base limit = 512; 77-token headroom


def _encode(pipe: Pipeline, text: str) -> list[int]:
    """
    Encode `text` to a list of token IDs using the pipeline's tokenizer.
    Special tokens (BOS/EOS) are excluded so the count reflects only the
    content tokens — the pipeline re-adds them when it calls the model.
    """
    return list(pipe.tokenizer.encode(text, add_special_tokens=False))


def _decode(pipe: Pipeline, token_ids: list[int]) -> str:
    """Decode a list of token IDs back to a plain-text string."""
    return str(pipe.tokenizer.decode(token_ids, skip_special_tokens=True))


def _split_into_chunks(pipe: Pipeline, text: str, chunk_size: int) -> list[str]:
    """
    Split `text` into a list of non-overlapping chunks, each containing at
    most `chunk_size` tokens as measured by the pipeline's tokenizer.

    Splitting is done in token space (not word space) so each chunk is
    guaranteed to be within the model's position-embedding limit regardless
    of how densely the words tokenise.
    """
    token_ids = _encode(pipe, text)

    # Fast path: the entire text already fits in one chunk
    if len(token_ids) <= chunk_size:
        return [text]

    chunks: list[str] = []
    for start in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[start: start + chunk_size]
        chunks.append(_decode(pipe, chunk_ids))
    return chunks


def _group_in_batches(items: list[str], batch_size: int) -> list[list[str]]:
    """Group a flat list into fixed-size batches for UI presentation."""
    if batch_size <= 0:
        return [items] if items else []
    return [
        items[start: start + batch_size]
        for start in range(0, len(items), batch_size)
    ]


# ---------------------------------------------------------------------------
# Inference helpers — Map-Reduce summarisation
# ---------------------------------------------------------------------------


def _bart_single_pass(
    text: str,
    max_length: int = 220,
    min_length: int = 40,
    length_penalty: float = 1.0,
    num_beams: int = 4,
    early_stopping: bool = False,
) -> str:
    """
    Run one BART summarisation pass.  The caller is responsible for ensuring
    `text` is within BART_CHUNK_TOKENS before calling this function.

    length_penalty > 1.0 rewards the model during beam search for producing
    longer, more detailed sequences.  It should only be applied on the final
    summary pass; intermediate map-phase passes use the default (1.0) so their
    outputs stay compact enough for the reduce step to process in one chunk.
    """
    output: list[dict[str, Any]] = _models["bart"](
        text,
        max_length=max_length,
        min_length=min_length,
        length_penalty=length_penalty,
        num_beams=num_beams,
        early_stopping=early_stopping,
        do_sample=False,
    )
    return str(output[0]["summary_text"])


def _bart_map_reduce(
    text: str,
    depth: int = 0,
    collect_top_level_points: bool = False,
) -> tuple[str, list[str]]:
    """
    Summarise `text` using the Map-Reduce strategy to handle documents longer
    than BART's 1024-token position-embedding limit.

    Algorithm:
      Base case  — document fits in one BART pass (≤ BART_CHUNK_TOKENS):
                   call BART directly and return.

      Map phase  — split the document into BART_CHUNK_TOKENS-sized chunks;
                   summarise each chunk independently with BART.

      Reduce phase — concatenate the mini-summaries and call _bart_map_reduce
                     recursively.  The combined mini-summaries are much shorter
                     than the original document, so recursion terminates quickly.

    depth is an internal guard against infinite recursion in pathological cases
    (e.g. a model that always generates near-max-length output).  At depth 3 the
    input is hard-truncated and a single pass is forced.

    Example for a 3 000-token document (chunk size 900):
      Map:
        Chunk 1 (tokens   1  900) → BART → Mini-summary 1 (~150 tokens)
        Chunk 2 (tokens 901 1800) → BART → Mini-summary 2 (~150 tokens)
        Chunk 3 (tokens 1801 2700)→ BART → Mini-summary 3 (~150 tokens)
        Chunk 4 (tokens 2701 3000)→ BART → Mini-summary 4 (~100 tokens)
      Reduce:
        Combined mini-summaries (~550 tokens) → BART → Final summary

        collect_top_level_points is used by /process to expose the first map pass
        mini-summaries in the UI (grouped in sets of 5). Recursive calls always
        return only the reduced summary.
    """
    bart_pipe = _models["bart"]
    token_ids = _encode(bart_pipe, text)

    # Base case: document fits within one safe BART pass.
    # Use the full generation budget and length_penalty so the final output
    # is detailed rather than truncated by BART's conservative defaults.
    if len(token_ids) <= BART_CHUNK_TOKENS:
        return _bart_single_pass(
            text,
            max_length=600,
            min_length=200,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        ), []

    # Safety valve: at max recursion depth force a single truncated pass
    if depth >= 3:
        logger.warning(
            "Map-Reduce hit max depth (%d); forcing single truncated pass "
            "on %d tokens.",
            depth,
            len(token_ids),
        )
        safe_text = _decode(bart_pipe, token_ids[:BART_CHUNK_TOKENS])
        return _bart_single_pass(
            safe_text,
            max_length=600,
            min_length=200,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        ), []

    chunks = _split_into_chunks(bart_pipe, text, BART_CHUNK_TOKENS)
    logger.info(
        "Map-Reduce BART (depth=%d): %d tokens → %d chunks of ≤%d tokens",
        depth,
        len(token_ids),
        len(chunks),
        BART_CHUNK_TOKENS,
    )

    # MAP: summarise each chunk independently; use shorter outputs so the
    # combined reduce input stays well below BART_CHUNK_TOKENS.
    mini_summaries: list[str] = []
    for idx, chunk in enumerate(chunks):
        logger.info("Map-Reduce BART: chunk %d/%d", idx + 1, len(chunks))
        mini = _bart_single_pass(chunk, max_length=170, min_length=28)
        mini_summaries.append(mini)

    # REDUCE: join mini-summaries and recurse; the combined text is much
    # shorter than the original so the next level almost always hits the
    # base case.
    combined = " ".join(mini_summaries)
    logger.info(
        "Map-Reduce BART (depth=%d): reduce pass — %d mini-summaries → "
        "%d combined tokens",
        depth,
        len(mini_summaries),
        len(_encode(bart_pipe, combined)),
    )
    reduced_summary, _ = _bart_map_reduce(combined, depth=depth + 1)
    if collect_top_level_points:
        return reduced_summary, mini_summaries
    return reduced_summary, []


def _flan_map_reduce(text: str) -> str:
    """
    Summarise `text` using FLAN-T5 with a Map-Reduce strategy to handle
    documents longer than FLAN-T5-base's 512-token limit.

        Map  : split into FLAN_CHUNK_TOKENS-sized chunks; summarise each with an
            explicit instruction prefix.
    Reduce: concatenate mini-summaries; if the combined text still exceeds
           FLAN_CHUNK_TOKENS, truncate it before the final pass (FLAN-T5 is
           less suited to deep recursion than BART because its outputs can be
           verbose).
    """
    flan_pipe = _models["flan"]
    token_ids = _encode(flan_pipe, text)

    # Base case: fits in a single pass.
    # max_length=300 / min_length=60 give FLAN-T5 a larger budget so it
    # produces a fully formed multi-sentence summary rather than a single
    # short sentence (its default tendency on short token budgets).
    if len(token_ids) <= FLAN_CHUNK_TOKENS:
        prompt = (
            "Summarize the following article in a few paragraphs:"
            f"\n\n{text}"
        )
        output: list[dict[str, Any]] = flan_pipe(
            prompt,
            max_length=300,
            min_length=60,
            do_sample=False,
        )
        return str(output[0]["generated_text"])

    chunks = _split_into_chunks(flan_pipe, text, FLAN_CHUNK_TOKENS)
    logger.info(
        "Map-Reduce FLAN (1 level): %d tokens → %d chunks of ≤%d tokens",
        len(token_ids),
        len(chunks),
        FLAN_CHUNK_TOKENS,
    )

    # MAP: one short summary per chunk
    mini_summaries: list[str] = []
    for idx, chunk in enumerate(chunks):
        logger.info("Map-Reduce FLAN: chunk %d/%d", idx + 1, len(chunks))
        prompt = (
            "Summarize the following article chunk in 2-3 clear sentences:"
            f"\n\n{chunk}"
        )
        out: list[dict[str, Any]] = flan_pipe(
            prompt,
            max_length=120,
            min_length=20,
            do_sample=False,
        )
        mini_summaries.append(str(out[0]["generated_text"]))

    # REDUCE: combine and run a final summarisation pass
    combined = " ".join(mini_summaries)
    combined_ids = _encode(flan_pipe, combined)

    # If the combined mini-summaries are still too long, truncate them
    if len(combined_ids) > FLAN_CHUNK_TOKENS:
        combined = _decode(flan_pipe, combined_ids[:FLAN_CHUNK_TOKENS])

    final_prompt = (
        "Summarize the following article in a few paragraphs:"
        f"\n\n{combined}"
    )
    # Use the same expanded budget as the base case so the final reduce
    # pass produces a detailed summary rather than a truncated one.
    final_out: list[dict[str, Any]] = flan_pipe(
        final_prompt,
        max_length=300,
        min_length=60,
        do_sample=False,
    )
    return str(final_out[0]["generated_text"])


def _summarise(
    text: str,
    model_name: str,
) -> tuple[str, list[list[str]] | None, int]:
    """
    Entry point for document summarisation.

    Routes to the correct Map-Reduce implementation based on the model
    selected by the Thompson Sampling A/B router.  Both implementations handle
    documents of arbitrary length by chunking, summarising each chunk, and
    recursively (BART) or single-level (FLAN-T5) reducing the results.

    The chunk count is computed by tokenising the document once before
    inference so it can be surfaced in the API response and displayed on the
    Streamlit UI without any additional inference overhead.

    Returns:
      summary            : final condensed text.
      summary_point_groups: for BART only, top-level map mini-summaries grouped
                            in batches of 5 for UI display; None for FLAN.
      chunk_count        : number of token-accurate chunks the document was
                           split into at depth-0 of the map phase.
    """
    if model_name == "bart":
        bart_pipe = _models["bart"]
        # Pre-compute the top-level chunk count using the same tokeniser and
        # chunk size used inside _bart_map_reduce so the value is exact.
        chunk_count: int = len(
            _split_into_chunks(bart_pipe, text, BART_CHUNK_TOKENS)
        )
        summary, mini_summaries = _bart_map_reduce(
            text,
            collect_top_level_points=True,
        )
        point_groups = _group_in_batches(mini_summaries, batch_size=5)
        return summary, (point_groups or None), chunk_count

    # FLAN path
    flan_pipe = _models["flan"]
    chunk_count = len(_split_into_chunks(flan_pipe, text, FLAN_CHUNK_TOKENS))
    return _flan_map_reduce(text), None, chunk_count


# ---------------------------------------------------------------------------
# Inference helpers — question generation
# ---------------------------------------------------------------------------


def _extract_questions(generated_text: str) -> list[str]:
    """
    Parse one or more questions from a FLAN generation.

    FLAN may return numbered lists, bullet lists, or a single line with
    multiple question marks. This helper normalises those variants.
    """
    raw = generated_text.strip()
    if not raw:
        return []

    candidates: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r"^(?:\d+[\).:-]\s*|[-*]\s*)", "", stripped).strip()
        if cleaned:
            candidates.append(cleaned)

    parsed: list[str] = []
    for candidate in candidates:
        parts = [part.strip() for part in candidate.split("?") if part.strip()]
        if len(parts) > 1 or candidate.endswith("?"):
            parsed.extend(f"{part}?" for part in parts)
        else:
            parsed.append(candidate)

    if parsed:
        return parsed

    fallback_parts = [part.strip() for part in raw.split("?") if part.strip()]
    if fallback_parts:
        return [f"{part}?" for part in fallback_parts]
    return [raw]


def _generate_questions(text: str) -> list[str]:
    """
    Generate comprehension questions for the document using FLAN-T5.

    The document is split into FLAN_CHUNK_TOKENS-sized chunks (token-accurate,
    not word-approximate).  One question is generated per chunk; a maximum of
    5 chunks are processed to keep inference time practical.  Duplicate
    questions are removed while preserving insertion order.
    """
    flan_pipe = _models["flan"]

    # Split using the same token-accurate chunking used by _flan_map_reduce
    # so question coverage matches the summarisation coverage.
    max_chunks = 5
    max_questions = 9
    all_chunks = _split_into_chunks(flan_pipe, text, FLAN_CHUNK_TOKENS)
    chunks = all_chunks[:max_chunks]

    questions: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        # Explicit instruction prefix asks FLAN to produce three numbered
        # questions rather than a single question per call.  This is the
        # second leg of the two-call sequential pipeline (summary + questions)
        # that FLAN-T5 needs for reliable task separation.
        prompt = (
            "Generate 3 reading comprehension questions based on this text:"
            f"\n\n{chunk}"
        )
        output: list[dict[str, Any]] = flan_pipe(
            prompt,
            max_length=150,
            min_length=24,
            do_sample=False,
        )
        generated_text = str(output[0]["generated_text"]).strip()
        for question in _extract_questions(generated_text):
            question = question.strip()
            if question and question not in seen:
                seen.add(question)
                questions.append(question)
            if len(questions) >= max_questions:
                return questions

    return questions


# ---------------------------------------------------------------------------
# Request / response Pydantic models
# ---------------------------------------------------------------------------


class ProcessResponse(BaseModel):
    """Response body returned by POST /process."""

    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    model_name: str
    summary: str
    summary_point_groups: list[list[str]] | None = None
    questions: list[str]
    drift_status: str
    inference_latency: float
    # Number of token-accurate chunks the document was split into at the
    # top-level map phase.  A value of 1 means the whole document fitted
    # within one model pass and no splitting was required.
    chunk_count: int


class FeedbackRequest(BaseModel):
    """Request body accepted by POST /feedback."""

    run_id: str
    # 1 = thumbs-up (positive feedback), 0 = thumbs-down (negative feedback)
    score: int


class FeedbackResponse(BaseModel):
    """Response body returned by POST /feedback."""

    run_id: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/process", response_model=ProcessResponse)
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(description="PDF, DOCX, or PPTX document to analyse."),
    authorization: str | None = Header(default=None),
) -> ProcessResponse:
    """
    Full document analysis pipeline.

    Steps:
      1. Validate the Bearer token.
      2. Extract text from the uploaded file (PDF / DOCX / PPTX).
      3. Run non-parametric drift detection on word count and readability grade.
      4. Select a model via Thompson Sampling.
      5. Generate a summary (BART or FLAN-T5 depending on the selected model).
      6. Generate comprehension questions (always FLAN-T5).
      7. Open an MLflow run; log parameters, metrics, drift tag, and a text
         artifact containing the summary and questions.
      8. Return the summary, questions, run_id, drift status, and latency.
      9. Optionally schedule automated_judge_worker (NLI score → MLflow + Thompson).
    """
    _validate_token(authorization)

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename was provided with the upload.",
        )

    file_bytes: bytes = await file.read()

    # Step 1: Parse the document to plain text
    text: str = _parse_document(file_bytes, file.filename)

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the uploaded document.",
        )

    # Step 2: Drift detection — runs before inference so the result can be
    # logged as a tag on the same MLflow run as the inference metrics.
    # Logging is handled inside _detect_drift itself.
    drift_info: dict[str, Any] = _detect_drift(text)

    # Step 3: Model selection
    selected_model: str = _select_model()

    # Step 4: Run inference; measure wall-clock time for latency logging
    start_ts: float = time.monotonic()

    try:
        summary, summary_point_groups, chunk_count = _summarise(text, selected_model)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarisation failed: {exc}",
        ) from exc

    try:
        questions: list[str] = _generate_questions(text)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question generation failed: {exc}",
        ) from exc

    inference_latency: float = time.monotonic() - start_ts

    # Step 5: Log everything to MLflow
    file_extension: str = (
        file.filename.rsplit(".", 1)[-1].lower()
        if "." in file.filename
        else "unknown"
    )

    with mlflow.start_run() as active_run:
        run_id: str = active_run.info.run_id

        # Parameters: inputs / configuration choices for this run
        mlflow.log_params({
            "model_name": selected_model,
            "document_length": drift_info["word_count"],
            "filename": file.filename,
        })

        # Metrics: quantitative measurements produced by this run
        mlflow.log_metrics({
            "inference_latency": round(inference_latency, 4),
            "word_count": float(drift_info["word_count"]),
            "flesch_kincaid_grade": drift_info["flesch_kincaid_grade"],
            "wc_percentile": drift_info["wc_percentile"],
            "fk_percentile": drift_info["fk_percentile"],
            "wc_wasserstein": drift_info["wc_wasserstein"],
            "fk_wasserstein": drift_info["fk_wasserstein"],
            "drift_window_size": float(drift_info["window_size"]),
            # Number of map-phase chunks; useful for understanding latency
            # scaling and is surfaced on the Streamlit UI.
            "chunk_count": float(chunk_count),
        })

        # Tags: categorical metadata for filtering in the MLflow UI
        mlflow.set_tags({
            "drift_status": drift_info["drift_status"],
            "file_type": file_extension,
        })

        # Artifact: persist the generated text so results are reproducible
        bart_points_block = ""
        if summary_point_groups:
            grouped_lines: list[str] = []
            for group_idx, group in enumerate(summary_point_groups, start=1):
                grouped_lines.append(f"Group {group_idx}")
                grouped_lines.extend(f"- {point}" for point in group)
                grouped_lines.append("")
            bart_points_block = (
                "=== BART CHUNK POINTS (GROUPED BY 5) ===\n"
                + "\n".join(grouped_lines).rstrip()
                + "\n\n"
            )

        artifact_text: str = (
            "=== SUMMARY ===\n"
            f"{summary}\n\n"
            + bart_points_block
            + "=== COMPREHENSION QUESTIONS ===\n"
            + "\n".join(
                f"{idx + 1}. {q}" for idx, q in enumerate(questions)
            )
        )
        # Write to a temporary file; MLflow copies it into the artifact store
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix=f"result_{run_id[:8]}_",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(artifact_text)
            tmp_path: str = tmp.name

        mlflow.log_artifact(tmp_path, artifact_path="results")

    # Register the run→model mapping so /feedback can update the correct
    # Thompson posterior without querying MLflow.
    _register_run(run_id, selected_model)

    # Update Prometheus metrics
    REQUEST_COUNT.labels(
        model_name=selected_model,
        file_type=file_extension,
        drift_status=drift_info["drift_status"],
    ).inc()
    REQUEST_LATENCY.labels(model_name=selected_model).observe(inference_latency)
    DRIFT_COUNTER.labels(drift_status=drift_info["drift_status"]).inc()
    DRIFT_WINDOW_GAUGE.set(drift_info["window_size"])

    logger.info(
        "Inference complete — run_id=%s  model=%s  latency=%.3fs  drift=%s",
        run_id,
        selected_model,
        inference_latency,
        drift_info["drift_status"],
    )
    # Full summary text is written here so operators can inspect results in
    # container logs (docker logs ai_doc_backend) without opening MLflow or the UI.
    logger.info("Generated summary (run_id=%s):\n%s", run_id, summary)

    if JUDGE_ENABLED and _judge_ce is not None:
        background_tasks.add_task(
            automated_judge_worker,
            text,
            summary,
            selected_model,
            run_id,
        )

    return ProcessResponse(
        run_id=run_id,
        model_name=selected_model,
        summary=summary,
        summary_point_groups=summary_point_groups,
        questions=questions,
        drift_status=drift_info["drift_status"],
        inference_latency=round(inference_latency, 4),
        chunk_count=chunk_count,
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    authorization: str | None = Header(default=None),
) -> FeedbackResponse:
    """
    Record the user's satisfaction score for a completed inference run.

    Two actions are performed:
      1. MLflow — resumes the run by run_id and appends user_satisfaction_score
         for permanent audit and dashboard visibility.
      2. In-memory — updates the Thompson Sampling Beta posterior for the model
         that served this run, so the next routing decision is immediately
         informed by the feedback without any database read.

    score values:
        1 — thumbs-up  (the result was helpful)
        0 — thumbs-down (the result was not helpful)
    """
    _validate_token(authorization)

    if body.score not in (0, 1):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="score must be 0 (thumbs-down) or 1 (thumbs-up).",
        )

    # Persist to MLflow for dashboards and long-term audit trail
    try:
        with mlflow.start_run(run_id=body.run_id):
            mlflow.log_metric("user_satisfaction_score", float(body.score))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"MLflow run '{body.run_id}' could not be found or resumed: {exc}"
            ),
        ) from exc

    # Update the in-memory Thompson posterior — no MLflow read required
    with _thompson_lock:
        model = _run_model_map.get(body.run_id)

    if model and model in _thompson_state:
        _update_thompson(model, thumbs_up=body.score == 1)
        FEEDBACK_COUNTER.labels(
            model_name=model,
            outcome="thumbs_up" if body.score == 1 else "thumbs_down",
        ).inc()
    else:
        logger.warning(
            "Feedback for run_id=%s: model not found in run map "
            "(server may have restarted); MLflow record saved but "
            "in-memory posterior not updated.",
            body.run_id,
        )

    logger.info(
        "Feedback recorded — run_id=%s  model=%s  score=%d",
        body.run_id,
        model or "unknown",
        body.score,
    )

    return FeedbackResponse(
        run_id=body.run_id,
        message="Feedback recorded successfully.",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """
    Liveness probe used by Docker Compose health checks and monitoring tools.
    Returns the compute device so operators can confirm GPU/CPU mode at a glance.
    """
    return {"status": "ok", "device": _device}


# HTTP traffic metrics (http_requests_total, request latency, etc.) plus the default
# registry that holds all ai_doc_* custom metrics — single /metrics endpoint.
Instrumentator().instrument(app).expose(app)
