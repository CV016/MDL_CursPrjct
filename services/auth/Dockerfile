# =============================================================================
# Auth service Dockerfile
# Multi-stage build: dependency installation separated from runtime image
# to keep the final layer lean and avoid shipping build tools.
# =============================================================================

# ---- Stage 1: dependency installation ----
FROM python:3.11-slim AS builder

WORKDIR /install

# Install only production dependencies into an isolated prefix so they can
# be copied cleanly into the runtime stage without __pycache__ pollution.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/deps -r requirements.txt


# ---- Stage 2: runtime image ----
FROM python:3.11-slim AS runtime

# Non-root user for security hardening
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install/deps /usr/local

# Copy application source
COPY main.py .

# Switch to non-root user
USER appuser

EXPOSE 8001

# Start uvicorn on container port 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
