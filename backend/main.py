"""
backend/main.py
===============
AI-DOC INTERACT — Backend API Service

This is the single FastAPI application that implements all backend logic for
the academic MLOps pipeline:

  - Document parsing: PDF via PyPDF2, DOCX via python-docx, PPTX via python-pptx.
  - Z-score data drift detection: word count and Flesch-Kincaid Grade Level
    (computed with textstat) are compared against a hardcoded statistical
    baseline. Any feature whose |Z-score| exceeds the threshold is flagged.
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
import tempfile
import time
from typing import Any

import mlflow
import mlflow.tracking
import numpy as np
import textstat
import torch
from docx import Document as DocxDocument
from fastapi import FastAPI, File, Header, HTTPException, UploadFile, status
from pptx import Presentation
from pydantic import BaseModel
from PyPDF2 import PdfReader
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

# ---------------------------------------------------------------------------
# Drift detection — hardcoded baseline statistics
#
# These values represent the expected distribution of a "normal" academic or
# professional document based on empirical observation:
#   word_count            : documents typically contain 200–800 words
#   flesch_kincaid_grade  : readability grade level typically 7–13
#
# A document is flagged as drift_detected when the absolute Z-score of any
# feature exceeds DRIFT_Z_THRESHOLD (3.0 standard deviations from the mean).
# ---------------------------------------------------------------------------

DRIFT_BASELINE: dict[str, dict[str, float]] = {
    "word_count": {"mu": 500.0, "sigma": 200.0},
    "flesch_kincaid_grade": {"mu": 10.0, "sigma": 3.0},
}

DRIFT_Z_THRESHOLD: float = 3.0

# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-DOC INTERACT — Backend API",
    version="1.0.0",
    description=(
        "Single backend service implementing document parsing, Z-score drift "
        "detection, epsilon-greedy A/B model routing, MLflow experiment "
        "tracking, and feedback ingestion."
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


def _z_score(value: float, mu: float, sigma: float) -> float:
    """
    Compute the standard Z-score: (value - mean) / std_dev.

    Returns 0.0 when sigma is zero to avoid division by zero on degenerate
    baselines.
    """
    if sigma == 0.0:
        return 0.0
    return (value - mu) / sigma


def _detect_drift(text: str) -> dict[str, Any]:
    """
    Assess whether the input document differs significantly from the baseline.

    Two features are computed using textstat:
      1. word_count            — total whitespace-delimited tokens
      2. flesch_kincaid_grade  — readability grade level (higher = harder)

    A Z-score is calculated for each feature against the hardcoded baseline.
    The document is tagged drift_detected when |Z| > DRIFT_Z_THRESHOLD for
    at least one feature; otherwise it is tagged normal.

    Returns a dict containing raw feature values, Z-scores, and drift_status.
    """
    word_count: int = len(text.split())
    fk_grade: float = textstat.flesch_kincaid_grade(text)

    z_wc: float = _z_score(
        word_count,
        DRIFT_BASELINE["word_count"]["mu"],
        DRIFT_BASELINE["word_count"]["sigma"],
    )
    z_fk: float = _z_score(
        fk_grade,
        DRIFT_BASELINE["flesch_kincaid_grade"]["mu"],
        DRIFT_BASELINE["flesch_kincaid_grade"]["sigma"],
    )

    is_drift: bool = abs(z_wc) > DRIFT_Z_THRESHOLD or abs(z_fk) > DRIFT_Z_THRESHOLD
    drift_status: str = "drift_detected" if is_drift else "normal"

    return {
        "word_count": word_count,
        "flesch_kincaid_grade": round(fk_grade, 2),
        "z_word_count": round(z_wc, 4),
        "z_fk_grade": round(z_fk, 4),
        "drift_status": drift_status,
    }


# ---------------------------------------------------------------------------
# Thompson Sampling A/B model router
# ---------------------------------------------------------------------------

# Uniform Beta prior applied to every model with no feedback history yet.
# Beta(1, 1) is equivalent to a uniform distribution over [0, 1], meaning
# the algorithm has no initial preference between the two models.
_BETA_PRIOR_ALPHA: int = 1
_BETA_PRIOR_BETA: int = 1


def _fetch_beta_params() -> dict[str, dict[str, int]]:
    """
    Query MLflow for the thumbs-up and thumbs-down counts per model and
    return the corresponding Beta posterior parameters.

    For each model:
        alpha = thumbs_up_count  + _BETA_PRIOR_ALPHA
        beta  = thumbs_down_count + _BETA_PRIOR_BETA

    Falls back to the uniform prior (alpha=1, beta=1) for any model whose
    history cannot be retrieved, so routing remains functional even when
    MLflow is unavailable or the experiment has no feedback yet.

    Returns:
        {
            "bart": {"alpha": int, "beta": int},
            "flan": {"alpha": int, "beta": int},
        }
    """
    prior: dict[str, dict[str, int]] = {
        "bart": {"alpha": _BETA_PRIOR_ALPHA, "beta": _BETA_PRIOR_BETA},
        "flan": {"alpha": _BETA_PRIOR_ALPHA, "beta": _BETA_PRIOR_BETA},
    }

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

        if experiment is None:
            logger.debug("Experiment not found yet; using Beta prior for both models.")
            return prior

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.user_satisfaction_score >= 0",
            max_results=500,
        )

    except Exception as exc:
        logger.warning(
            "MLflow query failed — falling back to Beta prior for routing: %s", exc
        )
        return prior

    counts: dict[str, dict[str, int]] = {
        "bart": {"thumbs_up": 0, "thumbs_down": 0},
        "flan": {"thumbs_up": 0, "thumbs_down": 0},
    }

    for run in runs:
        model: str | None = run.data.params.get("model_name")
        score: float | None = run.data.metrics.get("user_satisfaction_score")
        if model in counts and score is not None:
            if score >= 1.0:
                counts[model]["thumbs_up"] += 1
            else:
                counts[model]["thumbs_down"] += 1

    return {
        model: {
            "alpha": counts[model]["thumbs_up"] + _BETA_PRIOR_ALPHA,
            "beta":  counts[model]["thumbs_down"] + _BETA_PRIOR_BETA,
        }
        for model in counts
    }


def _select_model() -> str:
    """
    Select a model using Thompson Sampling.

    For each model a single value is drawn from its Beta posterior
    Beta(alpha, beta). The model whose draw is higher wins the request.

    This naturally balances exploration and exploitation:
    - A model with few observations has a wide, flat posterior, so it
      occasionally draws high values and receives exploratory traffic.
    - A model with many observations has a narrow, peaked posterior centred
      on its true success rate, so it wins consistently once its superiority
      is established.
    - Unlike epsilon-greedy, no fixed exploration rate is required and
      cumulative regret is minimised asymptotically.

    Returns:
        "bart" or "flan"
    """
    params = _fetch_beta_params()

    samples: dict[str, float] = {
        model: float(np.random.beta(p["alpha"], p["beta"]))
        for model, p in params.items()
    }

    chosen: str = max(samples, key=lambda m: samples[m])

    logger.info(
        "Thompson Sampling: selected '%s' "
        "(bart sample=%.4f α=%d β=%d | flan sample=%.4f α=%d β=%d)",
        chosen,
        samples["bart"],
        params["bart"]["alpha"],
        params["bart"]["beta"],
        samples["flan"],
        params["flan"]["alpha"],
        params["flan"]["beta"],
    )
    return chosen


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


# ---------------------------------------------------------------------------
# Inference helpers — Map-Reduce summarisation
# ---------------------------------------------------------------------------


def _bart_single_pass(text: str, max_length: int = 200, min_length: int = 30) -> str:
    """
    Run one BART summarisation pass.  The caller is responsible for ensuring
    `text` is within BART_CHUNK_TOKENS before calling this function.
    """
    output: list[dict[str, Any]] = _models["bart"](
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    return str(output[0]["summary_text"])


def _bart_map_reduce(text: str, depth: int = 0) -> str:
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
    """
    bart_pipe = _models["bart"]
    token_ids = _encode(bart_pipe, text)

    # Base case: document fits within one safe BART pass
    if len(token_ids) <= BART_CHUNK_TOKENS:
        return _bart_single_pass(text)

    # Safety valve: at max recursion depth force a single truncated pass
    if depth >= 3:
        logger.warning(
            "Map-Reduce hit max depth (%d); forcing single truncated pass "
            "on %d tokens.",
            depth,
            len(token_ids),
        )
        safe_text = _decode(bart_pipe, token_ids[:BART_CHUNK_TOKENS])
        return _bart_single_pass(safe_text)

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
        mini = _bart_single_pass(chunk, max_length=150, min_length=20)
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
    return _bart_map_reduce(combined, depth=depth + 1)


def _flan_map_reduce(text: str) -> str:
    """
    Summarise `text` using FLAN-T5 with a Map-Reduce strategy to handle
    documents longer than FLAN-T5-base's 512-token limit.

    Map  : split into FLAN_CHUNK_TOKENS-sized chunks; summarise each with a
           short instruction prompt ("Summarize in 1-2 sentences: …").
    Reduce: concatenate mini-summaries; if the combined text still exceeds
           FLAN_CHUNK_TOKENS, truncate it before the final pass (FLAN-T5 is
           less suited to deep recursion than BART because its outputs can be
           verbose).
    """
    flan_pipe = _models["flan"]
    token_ids = _encode(flan_pipe, text)

    # Base case: fits in a single pass
    if len(token_ids) <= FLAN_CHUNK_TOKENS:
        prompt = (
            "Summarize the following document in a few clear sentences:"
            f"\n\n{text}"
        )
        output: list[dict[str, Any]] = flan_pipe(
            prompt,
            max_length=200,
            min_length=20,
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
        prompt = f"Summarize the following text in 1-2 sentences:\n\n{chunk}"
        out: list[dict[str, Any]] = flan_pipe(
            prompt,
            max_length=100,
            min_length=15,
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
        "Summarize the following document in a few clear sentences:"
        f"\n\n{combined}"
    )
    final_out: list[dict[str, Any]] = flan_pipe(
        final_prompt,
        max_length=200,
        min_length=20,
        do_sample=False,
    )
    return str(final_out[0]["generated_text"])


def _summarise(text: str, model_name: str) -> str:
    """
    Entry point for document summarisation.

    Routes to the correct Map-Reduce implementation based on the model
    selected by the epsilon-greedy A/B router.  Both implementations handle
    documents of arbitrary length by chunking, summarising each chunk, and
    recursively (BART) or single-level (FLAN-T5) reducing the results.
    """
    if model_name == "bart":
        return _bart_map_reduce(text)
    return _flan_map_reduce(text)


# ---------------------------------------------------------------------------
# Inference helpers — question generation
# ---------------------------------------------------------------------------


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
    all_chunks = _split_into_chunks(flan_pipe, text, FLAN_CHUNK_TOKENS)
    chunks = all_chunks[:max_chunks]

    questions: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        prompt = (
            "Generate a comprehension question based on the following text:"
            f"\n\n{chunk}"
        )
        output: list[dict[str, Any]] = flan_pipe(
            prompt,
            max_length=128,
            do_sample=False,
        )
        question = str(output[0]["generated_text"]).strip()
        if question and question not in seen:
            seen.add(question)
            questions.append(question)

    return questions


# ---------------------------------------------------------------------------
# Request / response Pydantic models
# ---------------------------------------------------------------------------


class ProcessResponse(BaseModel):
    """Response body returned by POST /process."""

    run_id: str
    model_name: str
    summary: str
    questions: list[str]
    drift_status: str
    inference_latency: float


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
    file: UploadFile = File(description="PDF, DOCX, or PPTX document to analyse."),
    authorization: str | None = Header(default=None),
) -> ProcessResponse:
    """
    Full document analysis pipeline.

    Steps:
      1. Validate the Bearer token.
      2. Extract text from the uploaded file (PDF / DOCX / PPTX).
      3. Run Z-score drift detection on word count and readability grade.
      4. Select a model via the epsilon-greedy A/B router.
      5. Generate a summary (BART or FLAN-T5 depending on the selected model).
      6. Generate comprehension questions (always FLAN-T5).
      7. Open an MLflow run; log parameters, metrics, drift tag, and a text
         artifact containing the summary and questions.
      8. Return the summary, questions, run_id, drift status, and latency.
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
    drift_info: dict[str, Any] = _detect_drift(text)
    logger.info(
        "Drift check — word_count=%d  fk_grade=%.2f  "
        "z_wc=%.4f  z_fk=%.4f  status=%s",
        drift_info["word_count"],
        drift_info["flesch_kincaid_grade"],
        drift_info["z_word_count"],
        drift_info["z_fk_grade"],
        drift_info["drift_status"],
    )

    # Step 3: Model selection
    selected_model: str = _select_model()

    # Step 4: Run inference; measure wall-clock time for latency logging
    start_ts: float = time.monotonic()

    try:
        summary: str = _summarise(text, selected_model)
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
            "z_word_count": drift_info["z_word_count"],
            "z_fk_grade": drift_info["z_fk_grade"],
        })

        # Tags: categorical metadata for filtering in the MLflow UI
        mlflow.set_tags({
            "drift_status": drift_info["drift_status"],
            "file_type": file_extension,
        })

        # Artifact: persist the generated text so results are reproducible
        artifact_text: str = (
            "=== SUMMARY ===\n"
            f"{summary}\n\n"
            "=== COMPREHENSION QUESTIONS ===\n"
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

    logger.info(
        "Inference complete — run_id=%s  model=%s  latency=%.3fs  drift=%s",
        run_id,
        selected_model,
        inference_latency,
        drift_info["drift_status"],
    )

    return ProcessResponse(
        run_id=run_id,
        model_name=selected_model,
        summary=summary,
        questions=questions,
        drift_status=drift_info["drift_status"],
        inference_latency=round(inference_latency, 4),
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    authorization: str | None = Header(default=None),
) -> FeedbackResponse:
    """
    Record the user's satisfaction score for a completed inference run.

    Resumes the MLflow run identified by run_id and appends the
    user_satisfaction_score metric. The epsilon-greedy router reads these
    scores on subsequent requests to decide which model to exploit.

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

    try:
        # mlflow.start_run with an existing run_id resumes that run rather
        # than creating a new one, allowing metrics to be appended.
        with mlflow.start_run(run_id=body.run_id):
            mlflow.log_metric("user_satisfaction_score", float(body.score))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"MLflow run '{body.run_id}' could not be found or resumed: {exc}"
            ),
        ) from exc

    logger.info(
        "Feedback recorded — run_id=%s  score=%d", body.run_id, body.score
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
