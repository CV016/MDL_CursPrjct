# =============================================================================
# MLflow tracking server Dockerfile
# Packages the MLflow server with PostgreSQL + psycopg2 driver support.
# =============================================================================

FROM python:3.11-slim AS runtime

# psycopg2-binary provides the PostgreSQL driver for MLflow's backend store
RUN pip install --no-cache-dir \
    mlflow==2.13.0 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.110

# Create artifact storage directory
RUN mkdir -p /mlflow/artifacts

EXPOSE 5000

# Start the MLflow tracking server.
# MLFLOW_BACKEND_STORE_URI and MLFLOW_DEFAULT_ARTIFACT_ROOT are supplied
# via docker-compose environment variables.
CMD ["sh", "-c", \
    "mlflow server \
     --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
     --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT} \
     --host 0.0.0.0 \
     --port 5000 \
     --serve-artifacts"]
