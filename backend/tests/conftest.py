"""
backend/tests/conftest.py
=========================
Pytest session-level fixtures and path setup for the backend test suite.

Heavy ML dependencies (torch, transformers) are replaced with lightweight
MagicMock objects before any test module imports backend/main.py. This
approach means:
  - CI does not need to download multi-gigabyte model weights.
  - The full FastAPI application lifecycle (startup, endpoint routing,
    Pydantic validation) is still exercised.
  - MLflow is also mocked so no live tracking server is required.

The sys.path manipulation ensures that `import main` resolves to
backend/main.py regardless of the working directory pytest is invoked from.
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Mock heavy ML dependencies BEFORE any test module imports main.py.
# Setting entries in sys.modules intercepts the import machinery so that
# `import torch` and `from transformers import pipeline` inside main.py
# receive MagicMock instances without touching the real packages.
# ---------------------------------------------------------------------------

# torch mock — force CPU mode so _device_idx = -1 inside main.py
torch_mock = MagicMock()
torch_mock.cuda.is_available.return_value = False
sys.modules["torch"] = torch_mock

# transformers mock — pipeline() calls during startup return callable mocks
transformers_mock = MagicMock()
sys.modules["transformers"] = transformers_mock

# mlflow mock — prevent any real HTTP connections to a tracking server
sys.modules["mlflow"] = MagicMock()
sys.modules["mlflow.tracking"] = MagicMock()

# sentence_transformers — CrossEncoder judge (startup loads a tiny mock)
_judge_instance = MagicMock()
_judge_instance.predict.return_value = np.array([[0.1, 0.8, 0.1]])
_judge_instance.model.config.id2label = {
    0: "contradiction",
    1: "entailment",
    2: "neutral",
}
_st = MagicMock()
_st.CrossEncoder = MagicMock(return_value=_judge_instance)
sys.modules["sentence_transformers"] = _st

# prometheus_fastapi_instrumentator — chained .instrument().expose() at import time
_inst_chain = MagicMock()
_inst_chain.instrument.return_value = _inst_chain
_inst_chain.expose.return_value = None
_pinstrument = MagicMock()
_pinstrument.Instrumentator.return_value = _inst_chain
sys.modules["prometheus_fastapi_instrumentator"] = _pinstrument

# ---------------------------------------------------------------------------
# Add the backend/ directory to sys.path so `import main` works when pytest
# is executed from the repository root (e.g. `pytest backend/tests/`).
# ---------------------------------------------------------------------------
_backend_dir = os.path.join(os.path.dirname(__file__), "..")
if _backend_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_backend_dir))
