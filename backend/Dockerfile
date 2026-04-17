# =============================================================================
# backend/Dockerfile
#
# Builds the FastAPI backend-api service image with CUDA GPU support.
#
# PyTorch is installed from the official PyTorch CUDA 12.1 wheel index so
# the container can use the host GPUs via the NVIDIA Container Toolkit.
# Model weights are NOT baked into the image — they are downloaded from
# Hugging Face on first startup and read from a persistent host-mounted
# cache volume on all subsequent starts.
# =============================================================================

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Point the HuggingFace cache at the directory that is bind-mounted
    # from the host so model weights survive container restarts.
    HF_HOME=/hf_cache

WORKDIR /app

# System packages:
#   libgomp1  — OpenMP runtime required by PyTorch CPU/GPU kernels
#   curl      — used by the Docker healthcheck in docker-compose.yml
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CUDA-enabled PyTorch (CUDA 12.1 build) from the official wheel
# index first, then install the remaining packages from PyPI.
# If your server runs a different CUDA version, change cu121 to match:
#   CUDA 11.8 -> cu118   |   CUDA 12.1 -> cu121   |   CUDA 12.4 -> cu124
# The +cu121 build tag is required. Without it, pip resolves "torch==2.3.0"
# to the PyPI meta-package which lists nvidia-cusparse-cu12 and similar CUDA
# library packages as separate dependencies; those exact versions do not exist
# on the PyTorch wheel index so the build fails. The +cu121 tag selects the
# self-contained CUDA wheel directly and avoids that dependency chain entirely.
RUN pip install --no-cache-dir "torch==2.3.0+cu121" \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--timeout-keep-alive", "120"]
