# =============================================================================
# Question-generation Dockerfile
# Uses the official PyTorch CUDA base image.
# =============================================================================

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime AS runtime

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

USER appuser
EXPOSE 8004

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]
