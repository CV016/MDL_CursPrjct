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
  - Epsilon-greedy A/B model routing: queries MLflow for the historical
    thumbs-up ratio of each model and exploits the winner 90% of the time;
    explores randomly the remaining 10%.
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
import random
import tempfile
import time
from typing import Any

import mlflow
import mlflow.tracking
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

# Epsilon-greedy exploration probability (0.1 = 10% random exploration)
EPSILON: float = float(os.environ.get("EPSILON", "0.1"))

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
# Epsilon-greedy A/B model router
# ---------------------------------------------------------------------------


def _fetch_model_satisfaction_rates() -> dict[str, float]:
    """
    Query MLflow for the historical mean user_satisfaction_score per model.

    Searches all runs in the current experiment that have a logged
    user_satisfaction_score metric and groups the scores by the model_name
    parameter. Falls back to 0.5 for any model with no history, which
    represents a neutral prior and gives both models equal initial weight.

    Returns:
        {"bart": <float>, "flan": <float>} with values in [0.0, 1.0].
    """
    default_rates: dict[str, float] = {"bart": 0.5, "flan": 0.5}

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

        if experiment is None:
            logger.debug("Experiment not found yet; using default satisfaction rates.")
            return default_rates

        # Retrieve the most recent 500 runs that have a satisfaction score
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.user_satisfaction_score >= 0",
            max_results=500,
        )

    except Exception as exc:
        # Network or MLflow server errors must not crash the routing logic;
        # fall back to the neutral prior so inference continues uninterrupted.
        logger.warning(
            "MLflow query failed — defaulting to equal satisfaction rates: %s", exc
        )
        return default_rates

    scores: dict[str, list[float]] = {"bart": [], "flan": []}

    for run in runs:
        model_name: str | None = run.data.params.get("model_name")
        score: float | None = run.data.metrics.get("user_satisfaction_score")
        if model_name in scores and score is not None:
            scores[model_name].append(score)

    return {
        model: (sum(vals) / len(vals)) if vals else 0.5
        for model, vals in scores.items()
    }


def _select_model() -> str:
    """
    Choose a model using the epsilon-greedy strategy.

    Exploration (probability = EPSILON):
        Pick a model uniformly at random. This ensures that the less-favoured
        model still receives traffic and can accumulate fresh feedback.

    Exploitation (probability = 1 - EPSILON):
        Pick the model with the highest historical mean satisfaction rate from
        MLflow. This directs most traffic to the currently best-performing model.

    Returns:
        "bart" or "flan"
    """
    if random.random() < EPSILON:
        chosen: str = random.choice(["bart", "flan"])
        logger.info("A/B router: EXPLORE — randomly selected '%s'", chosen)
        return chosen

    rates = _fetch_model_satisfaction_rates()
    chosen = max(rates, key=lambda m: rates[m])
    logger.info(
        "A/B router: EXPLOIT — selected '%s' "
        "(bart satisfaction=%.3f, flan satisfaction=%.3f)",
        chosen,
        rates.get("bart", 0.5),
        rates.get("flan", 0.5),
    )
    return chosen


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _summarise(text: str, model_name: str) -> str:
    """
    Generate a summary of the document text.

    When model_name is "bart": uses the BART summarisation pipeline which
        was fine-tuned specifically on CNN/DailyMail for extractive-abstractive
        summarisation.
    When model_name is "flan": uses FLAN-T5 with an instruction prompt,
        leveraging its general instruction-following capability for summarisation.

    Input text is truncated to the first 1 024 whitespace tokens before being
    passed to either model to stay within input length constraints.
    """
    # Truncate to the first 1 024 words to stay within model input limits
    truncated = " ".join(text.split()[:1024])

    if model_name == "bart":
        bart_pipe = _models["bart"]
        output: list[dict[str, Any]] = bart_pipe(
            truncated,
            max_length=256,
            min_length=64,
            do_sample=False,
            truncation=True,
        )
        return str(output[0]["summary_text"])

    # FLAN-T5 summarisation via instruction prompt
    flan_pipe = _models["flan"]
    prompt = f"Summarize the following document in a few clear sentences:\n\n{truncated}"
    output = flan_pipe(
        prompt,
        max_length=256,
        min_length=32,
        do_sample=False,
        truncation=True,
    )
    return str(output[0]["generated_text"])


def _generate_questions(text: str) -> list[str]:
    """
    Generate comprehension questions for the document using FLAN-T5.

    The text is split into segments of approximately 80 words each (roughly
    500 characters). One question is generated per segment; duplicates are
    removed while preserving insertion order. A maximum of 5 segments are
    processed to keep inference time practical on CPU.
    """
    flan_pipe = _models["flan"]
    words = text.split()

    # Build segments of ~80 words each; cap at 5 segments to bound latency
    segment_size = 80
    max_segments = 5
    segments: list[str] = [
        " ".join(words[i: i + segment_size])
        for i in range(0, min(len(words), segment_size * max_segments), segment_size)
        if words[i: i + segment_size]
    ]

    questions: list[str] = []
    seen: set[str] = set()

    for segment in segments:
        prompt = (
            "Generate a comprehension question based on the following text:\n\n"
            f"{segment}"
        )
        output: list[dict[str, Any]] = flan_pipe(
            prompt,
            max_length=128,
            do_sample=False,
            truncation=True,
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
