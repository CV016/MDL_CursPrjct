# =============================================================================
# frontend/Dockerfile
#
# Builds the Streamlit frontend-ui service image.
# The image is intentionally minimal — only Streamlit and the requests library
# are installed; all ML and data processing happens in the backend-api service.
# =============================================================================

FROM python:3.11-slim AS runtime

# Prevent .pyc files; enable unbuffered logs for docker compose logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

# Streamlit configuration flags:
#   --server.headless=true   — disables browser auto-open in container context
#   --server.fileWatcherType=none — disables the file watcher (unnecessary in prod)
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
