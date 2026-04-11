# =============================================================================
# Parser service Dockerfile
# =============================================================================

FROM python:3.11-slim AS builder

WORKDIR /install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/deps -r requirements.txt

# ---------------------------------------------------------------------------

FROM python:3.11-slim AS runtime

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

COPY --from=builder /install/deps /usr/local
COPY main.py .

# Prometheus multiprocess directory — required when running multiple uvicorn workers
RUN mkdir -p /tmp/prometheus_multiproc && chown appuser /tmp/prometheus_multiproc
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc

USER appuser
EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]
