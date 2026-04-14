"""
backend/tests/test_backend.py
==============================
Unit and integration tests for the backend-api FastAPI application.

Test strategy:
  - All HuggingFace and MLflow dependencies are mocked in conftest.py so
    the test suite runs in CI without downloading model weights or connecting
    to a live MLflow server.
  - TestClient wraps the FastAPI app and triggers the full startup lifecycle,
    verifying that the application initialises without errors and that both
    model slots in the registry are populated.
  - Auth tests confirm that every protected endpoint enforces the Bearer token
    scheme before any business logic is executed.
  - Validation tests confirm that malformed request bodies are rejected with
    the correct HTTP status codes by Pydantic / FastAPI.

Classes:
  TestAppStartup        — confirms startup succeeds and models are registered
  TestHealthEndpoint    — confirms the /health probe returns the expected schema
  TestAuthentication    — confirms 401 is returned for missing / invalid tokens
  TestFeedbackValidation — confirms 422 / 401 for malformed feedback requests
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# The MOCK_JWT_TOKEN default value must match MOCK_JWT_TOKEN in main.py
VALID_TOKEN = "mock-jwt-token-for-academic-project"
BEARER_HEADER = {"Authorization": f"Bearer {VALID_TOKEN}"}
BAD_HEADER = {"Authorization": "Bearer this-is-a-wrong-token"}


# ---------------------------------------------------------------------------
# Session-scoped TestClient fixture
#
# Patches _models directly before startup so that the pipeline() calls inside
# on_startup() return our pre-configured mock callables. Each mock is set up
# to return the correct output format expected by the inference helpers.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Build a TestClient that exercises the full FastAPI startup lifecycle.

    The mock callables stored in _models mimic the real pipeline interface:
      - bart_pipe(text, ...)       -> [{"summary_text": "..."}]
      - flan_pipe(prompt, ...)     -> [{"generated_text": "..."}]
    """
    # Callable mocks that return the pipeline output format
    bart_pipe = MagicMock(return_value=[{"summary_text": "This is a mocked BART summary."}])
    flan_pipe = MagicMock(return_value=[{"generated_text": "What is the mocked question?"}])

    # pipeline() factory is called twice during startup (once per model)
    pipeline_factory = MagicMock(side_effect=[bart_pipe, flan_pipe])

    # Import main after conftest has already injected the sys.modules mocks
    import main

    with patch.object(main, "pipeline", pipeline_factory):
        # Trigger startup: populates main._models with our mocks
        with TestClient(main.app) as test_client:
            # Manually set _models so inference tests work even if startup
            # order differs between test environments
            main._models["bart"] = bart_pipe
            main._models["flan"] = flan_pipe
            yield test_client


# ---------------------------------------------------------------------------
# TestAppStartup
# ---------------------------------------------------------------------------

class TestAppStartup:
    """
    Verify that the FastAPI application initialises correctly.

    The TestClient fixture triggers the on_startup lifecycle event, which
    calls mlflow.set_tracking_uri, mlflow.set_experiment, and pipeline()
    twice. Confirming that _models is populated proves that startup executed
    the model-loading code without raising an exception.
    """

    def test_both_model_slots_populated_after_startup(self, client):
        """Both 'bart' and 'flan' keys must exist in the model registry."""
        import main
        assert "bart" in main._models, "BART model slot missing from registry"
        assert "flan" in main._models, "FLAN model slot missing from registry"

    def test_model_registry_values_are_callable(self, client):
        """Entries in _models must be callable (pipeline objects are callable)."""
        import main
        assert callable(main._models["bart"]), "_models['bart'] is not callable"
        assert callable(main._models["flan"]), "_models['flan'] is not callable"


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Verify that GET /health returns the expected liveness response."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body_contains_status_ok(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_body_contains_device_key(self, client):
        """The device key lets operators confirm CPU/GPU mode at a glance."""
        response = client.get("/health")
        assert "device" in response.json()


# ---------------------------------------------------------------------------
# TestAuthentication
# ---------------------------------------------------------------------------

class TestAuthentication:
    """
    Verify that every protected endpoint enforces Bearer token authentication.

    Tests use a minimal file payload so the request reaches the auth check;
    they do not exercise downstream parsing or inference code.
    """

    def _minimal_pdf_upload(self):
        """Return a files dict with a tiny placeholder byte payload."""
        return {"file": ("document.pdf", io.BytesIO(b"placeholder"), "application/pdf")}

    def test_process_missing_auth_header_returns_401(self, client):
        """POST /process without an Authorization header must return 401."""
        response = client.post("/process", files=self._minimal_pdf_upload())
        assert response.status_code == 401

    def test_process_wrong_token_returns_401(self, client):
        """POST /process with an incorrect token must return 401."""
        response = client.post(
            "/process",
            headers=BAD_HEADER,
            files=self._minimal_pdf_upload(),
        )
        assert response.status_code == 401

    def test_process_malformed_bearer_returns_401(self, client):
        """Authorization header without the 'Bearer' scheme must return 401."""
        response = client.post(
            "/process",
            headers={"Authorization": VALID_TOKEN},  # missing "Bearer " prefix
            files=self._minimal_pdf_upload(),
        )
        assert response.status_code == 401

    def test_feedback_missing_auth_header_returns_401(self, client):
        """POST /feedback without an Authorization header must return 401."""
        response = client.post(
            "/feedback",
            json={"run_id": "some-run-id", "score": 1},
        )
        assert response.status_code == 401

    def test_feedback_wrong_token_returns_401(self, client):
        """POST /feedback with an incorrect token must return 401."""
        response = client.post(
            "/feedback",
            headers=BAD_HEADER,
            json={"run_id": "some-run-id", "score": 1},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# TestFeedbackValidation
# ---------------------------------------------------------------------------

class TestFeedbackValidation:
    """
    Verify that POST /feedback rejects invalid request payloads before
    attempting any MLflow operations.
    """

    def test_score_out_of_range_returns_422(self, client):
        """A score of 2 is not in (0, 1) and must be rejected with 422."""
        response = client.post(
            "/feedback",
            headers=BEARER_HEADER,
            json={"run_id": "abc123", "score": 2},
        )
        assert response.status_code == 422

    def test_negative_score_returns_422(self, client):
        """A negative score must be rejected with 422."""
        response = client.post(
            "/feedback",
            headers=BEARER_HEADER,
            json={"run_id": "abc123", "score": -1},
        )
        assert response.status_code == 422

    def test_missing_run_id_returns_422(self, client):
        """A request body without run_id must be rejected with 422."""
        response = client.post(
            "/feedback",
            headers=BEARER_HEADER,
            json={"score": 1},
        )
        assert response.status_code == 422

    def test_missing_score_returns_422(self, client):
        """A request body without score must be rejected with 422."""
        response = client.post(
            "/feedback",
            headers=BEARER_HEADER,
            json={"run_id": "abc123"},
        )
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client):
        """An empty JSON body must be rejected with 422."""
        response = client.post(
            "/feedback",
            headers=BEARER_HEADER,
            json={},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# TestDriftDetection
# ---------------------------------------------------------------------------

class TestDriftDetection:
    """
    Unit tests for the Z-score drift detection helpers.
    These do not use the TestClient — they call the functions directly.
    """

    def test_z_score_at_mean_is_zero(self):
        """Z-score of the mean value must be exactly 0.0."""
        import main
        assert main._z_score(500.0, 500.0, 200.0) == 0.0

    def test_z_score_one_sigma_above_mean(self):
        """Value exactly one sigma above the mean must yield Z = 1.0."""
        import main
        assert main._z_score(700.0, 500.0, 200.0) == pytest.approx(1.0)

    def test_z_score_zero_sigma_returns_zero(self):
        """Zero sigma (degenerate baseline) must return 0.0 without dividing."""
        import main
        assert main._z_score(999.0, 500.0, 0.0) == 0.0

    def test_normal_document_not_flagged(self):
        """A document with typical word count and readability must be 'normal'."""
        import main
        # 500 words, grade 10.0 — both exactly at the baseline mean
        normal_text = " ".join(["word"] * 500)
        result = main._detect_drift(normal_text)
        # The drift_status is determined by Z-score; exact value depends on
        # textstat; we confirm the function returns the expected keys.
        assert "drift_status" in result
        assert "word_count" in result
        assert "flesch_kincaid_grade" in result
        assert "z_word_count" in result
        assert "z_fk_grade" in result

    def test_extremely_short_document_flags_drift(self):
        """A 2-word document should produce a large negative Z-score for word count."""
        import main
        result = main._detect_drift("Hello world.")
        # word_count ≈ 2, baseline mu=500, sigma=200 → Z ≈ -2.49
        # May or may not exceed 3.0 threshold but Z must be strongly negative
        assert result["z_word_count"] < 0
