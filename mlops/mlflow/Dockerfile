# =============================================================================
# MLflow tracking server Dockerfile
#
# Uses a local SQLite file as the backend store so no external database is
# required. Artifacts are written to a directory inside the same named volume,
# keeping the entire MLflow state in a single Docker volume mount.
#
# Environment variables consumed at runtime:
#   MLFLOW_BACKEND_STORE_URI   — SQLite URI, e.g. sqlite:////mlflow/mlflow.db
#   MLFLOW_DEFAULT_ARTIFACT_ROOT — directory inside the volume, e.g. /mlflow/artifacts
# =============================================================================

FROM python:3.11-slim AS runtime

# Install MLflow only — no database driver needed for SQLite
RUN pip install --no-cache-dir mlflow==2.13.0

# Create the directory that will hold mlflow.db and the artifacts sub-tree.
# This directory is overlaid by the named Docker volume at runtime.
RUN mkdir -p /mlflow/artifacts

EXPOSE 5000

# The CMD reads MLFLOW_BACKEND_STORE_URI and MLFLOW_DEFAULT_ARTIFACT_ROOT
# from the docker-compose environment block so the Dockerfile stays generic.
CMD ["sh", "-c", \
    "mlflow server \
     --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
     --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT} \
     --host 0.0.0.0 \
     --port 5000"]
