# =============================================================================
# Summarizer Dockerfile
# Uses the official PyTorch CUDA base image so GPU drivers are pre-installed.
# =============================================================================

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime AS runtime

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Install Python dependencies (torch is already present in the base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

USER appuser
EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
