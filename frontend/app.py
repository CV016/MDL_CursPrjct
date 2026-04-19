"""
frontend/app.py
===============
AI-DOC INTERACT — Streamlit Frontend

User workflow:
  1. Upload a PDF, DOCX, or PPTX file via the Streamlit file uploader.
  2. Click "Analyse Document" — the file is sent to the backend via
     POST /process with a hardcoded Bearer token in the Authorization header.
  3. A loading spinner is displayed while the backend parses the document,
     runs drift detection, selects a model, and generates the output.
  4. The summary and comprehension questions are displayed on screen, along
     with metadata: model used, inference time, and drift status.
  5. The user clicks thumbs-up or thumbs-down. The score and the MLflow
     run_id (returned in step 2) are sent to POST /feedback so the backend
     can log user_satisfaction_score to the correct MLflow run.

Session state keys:
  result             — dict returned by POST /process; persists across reruns.
                       Includes chunk_count (int) from the updated API response.
  feedback_submitted — bool; prevents duplicate feedback submissions.
"""

import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration — read from environment variables injected by docker-compose
# ---------------------------------------------------------------------------

# Internal URL of the backend-api service (container-to-container DNS)
BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://backend-api:8000")

# Mock JWT token — must match MOCK_JWT_TOKEN on the backend
MOCK_JWT_TOKEN: str = os.environ.get(
    "MOCK_JWT_TOKEN", "mock-jwt-token-for-academic-project"
)

# Pre-built headers sent with every request to the backend
_AUTH_HEADERS: dict[str, str] = {"Authorization": f"Bearer {MOCK_JWT_TOKEN}"}

# Maximum time (seconds) to wait for the backend to respond.
# Inference on CPU for large documents can take up to two minutes.
_REQUEST_TIMEOUT: int = 300

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call in the script
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI-DOC INTERACT",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _call_process(file_name: str, file_bytes: bytes, file_type: str) -> dict:
    """
    Send the uploaded document to POST /process and return the JSON response.

    Raises:
        requests.exceptions.RequestException on network errors.
        ValueError if the server returns a non-200 status code.
    """
    response = requests.post(
        f"{BACKEND_URL}/process",
        headers=_AUTH_HEADERS,
        files={"file": (file_name, file_bytes, file_type)},
        timeout=_REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        # Surface the backend's error detail if available, fall back to status
        detail = response.json().get("detail", response.text) if response.content else response.text
        raise ValueError(f"Backend returned {response.status_code}: {detail}")

    return response.json()


def _call_feedback(run_id: str, score: int) -> None:
    """
    Send the user's satisfaction score to POST /feedback.

    Raises:
        requests.exceptions.RequestException on network errors.
        ValueError if the server returns a non-200 status code.
    """
    response = requests.post(
        f"{BACKEND_URL}/feedback",
        headers=_AUTH_HEADERS,
        json={"run_id": run_id, "score": score},
        timeout=30,
    )

    if response.status_code != 200:
        detail = response.json().get("detail", response.text) if response.content else response.text
        raise ValueError(f"Feedback endpoint returned {response.status_code}: {detail}")


# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------

st.title("AI-DOC INTERACT")
st.caption(
    "Upload a PDF, Word document, or PowerPoint presentation to generate "
    "an AI-powered summary and comprehension questions."
)

st.divider()

# ---------------------------------------------------------------------------
# UI — File uploader
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a document to analyse",
    type=["pdf", "docx", "pptx"],
    help="Accepted formats: PDF, DOCX, PPTX. Maximum recommended size: 10 MB.",
)

# Show which file is loaded
if uploaded_file is not None:
    st.info(
        f"Loaded: **{uploaded_file.name}** "
        f"({uploaded_file.size / 1024:.1f} KB)"
    )

# ---------------------------------------------------------------------------
# UI — Analyse button and inference call
# ---------------------------------------------------------------------------

if st.button(
    "Analyse Document",
    type="primary",
    disabled=(uploaded_file is None),
):
    # Clear any previous result and feedback state when a new analysis starts
    st.session_state.pop("result", None)
    st.session_state.pop("feedback_submitted", None)

    with st.spinner("Analysing document — this may take a minute on CPU..."):
        try:
            result = _call_process(
                file_name=uploaded_file.name,
                file_bytes=uploaded_file.getvalue(),
                file_type=uploaded_file.type or "application/octet-stream",
            )
            st.session_state["result"] = result
        except (requests.exceptions.RequestException, ValueError) as exc:
            st.error(f"Analysis failed: {exc}")

# ---------------------------------------------------------------------------
# UI — Results display
# Rendered whenever session_state["result"] is present (persists on rerun)
# ---------------------------------------------------------------------------

if "result" in st.session_state:
    result: dict = st.session_state["result"]

    st.divider()

    # Metadata row — four equal columns for the key run metadata
    col_model, col_latency, col_drift, col_chunks = st.columns(4)

    with col_model:
        st.metric(
            label="Model Selected",
            value=result.get("model_name", "—").upper(),
            help=(
                "bart = facebook/bart-large-cnn (summarisation-tuned). "
                "flan = google/flan-t5-base (instruction-tuned). "
                "Selected by the Thompson Sampling A/B router."
            ),
        )

    with col_latency:
        latency_val = result.get("inference_latency", 0.0)
        st.metric(
            label="Inference Time",
            value=f"{latency_val:.2f} s",
            help="Wall-clock time measured inside the backend for this inference run.",
        )

    with col_chunks:
        chunk_count_val: int = result.get("chunk_count", 1)
        st.metric(
            label="Chunks Processed",
            value=str(chunk_count_val),
            help=(
                "Number of token-accurate chunks the document was split into "
                "during the map phase of inference. "
                "A value of 1 means the entire document fitted within a single "
                "model pass and no splitting was required. "
                "Higher values indicate a longer document that required "
                "Map-Reduce processing."
            ),
        )

    with col_drift:
        drift_raw: str = result.get("drift_status", "normal")

        # Map the compound status tag to a concise display label
        _drift_labels: dict[str, str] = {
            "normal": "Normal",
            "point_anomaly": "Point Anomaly",
            "window_drift": "Window Drift",
            "point_anomaly|window_drift": "Anomaly + Drift",
        }
        drift_label: str = _drift_labels.get(drift_raw, drift_raw.replace("|", " + "))

        st.metric(
            label="Data Drift",
            value=drift_label,
            help=(
                "Two-tier non-parametric drift detection. "
                "Point Anomaly: this document's percentile rank is outside the "
                "[2.5, 97.5] interval of the reference CDF. "
                "Window Drift: the Wasserstein distance between the last "
                "10 requests and the reference distribution exceeds the threshold."
            ),
        )

    # Drift warning banners — separate messages for each tier
    if "point_anomaly" in drift_raw:
        st.warning(
            "Point anomaly detected: this document's features fall outside the "
            "expected range of the reference population. "
            "It may be unusually long, short, or at an atypical reading level."
        )
    if "window_drift" in drift_raw:
        st.error(
            "Distributional drift detected: the recent batch of uploaded documents "
            "has systematically shifted away from the baseline distribution. "
            "Model performance may be degraded for the current traffic pattern."
        )

    st.divider()

    # Summary section
    st.subheader("Summary")
    summary_text: str = result.get("summary", "")
    if summary_text:
        st.write(summary_text)
    else:
        st.write("No summary was generated.")

    # Optional BART map-phase points grouped in batches of 5.
    point_groups_raw = result.get("summary_point_groups")
    if (
        result.get("model_name") == "bart"
        and isinstance(point_groups_raw, list)
        and point_groups_raw
    ):
        st.subheader("BART Chunk Points")
        st.caption("Top-level BART mini-summaries grouped in batches of 5 chunks.")
        for group_idx, group in enumerate(point_groups_raw, start=1):
            if not isinstance(group, list):
                continue
            st.markdown(f"**Group {group_idx}**")
            for point in group:
                point_text = str(point).strip()
                if point_text:
                    st.write(f"- {point_text}")

    # Questions section
    st.subheader("Comprehension Questions")
    questions: list[str] = result.get("questions", [])
    if questions:
        for idx, question in enumerate(questions, start=1):
            st.write(f"{idx}. {question}")
    else:
        st.write("No questions were generated.")

    st.divider()

    # MLflow run ID — shown in a collapsed expander for transparency
    with st.expander("MLflow Run Details"):
        st.code(result.get("run_id", ""), language=None)
        st.caption(
            "Open the MLflow UI at http://localhost:5000 to inspect the full "
            "experiment log for this run."
        )

    # ---------------------------------------------------------------------------
    # UI — Feedback buttons
    # ---------------------------------------------------------------------------

    st.subheader("Was this result helpful?")

    if st.session_state.get("feedback_submitted", False):
        st.success("Thank you — your feedback has been recorded in MLflow.")
    else:
        col_up, col_down, col_spacer = st.columns([1, 1, 5])

        with col_up:
            if st.button("Thumbs Up", key="btn_thumbs_up"):
                try:
                    _call_feedback(
                        run_id=result["run_id"],
                        score=1,
                    )
                    st.session_state["feedback_submitted"] = True
                    st.rerun()
                except (requests.exceptions.RequestException, ValueError) as exc:
                    st.error(f"Could not submit feedback: {exc}")

        with col_down:
            if st.button("Thumbs Down", key="btn_thumbs_down"):
                try:
                    _call_feedback(
                        run_id=result["run_id"],
                        score=0,
                    )
                    st.session_state["feedback_submitted"] = True
                    st.rerun()
                except (requests.exceptions.RequestException, ValueError) as exc:
                    st.error(f"Could not submit feedback: {exc}")
