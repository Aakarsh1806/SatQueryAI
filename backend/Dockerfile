FROM python:3.11-slim
ARG INSTALL_ML=false
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 SATQUERY_DEVICE=cpu \
    SATQUERY_STORAGE_DIR=/var/lib/satquery/uploads SATQUERY_TRACE_DIR=/var/lib/satquery/traces \
    DATABASE_URL=sqlite:////var/lib/satquery/satquery.db HF_HOME=/var/cache/huggingface
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements*.txt /app/backend/
RUN pip install --no-cache-dir -r backend/requirements.txt -r backend/requirements-postgres.txt \
    && if [ "$INSTALL_ML" = "true" ]; then pip install --no-cache-dir -r backend/requirements-ml.txt; fi
COPY backend/app /app/backend/app
COPY backend/models.yaml /app/backend/models.yaml
# No weights, sample imagery, .env, uploads or local Hugging Face cache are copied.
RUN useradd --uid 10001 --create-home satquery \
    && mkdir -p /var/lib/satquery/uploads /var/lib/satquery/traces /var/cache/huggingface /app/weights /app/demo \
    && chown -R satquery:satquery /var/lib/satquery /var/cache/huggingface
USER satquery
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
