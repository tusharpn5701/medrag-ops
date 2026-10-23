# Serving image: API + the champion bundle exported from the MLflow registry.
# Build:  medrag export --role champion && docker build -t medrag-ops .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MEDRAG_ROOT=/app FASTEMBED_CACHE_PATH=/app/.cache/fastembed OMP_NUM_THREADS=1
WORKDIR /app

# Dependencies first, so code changes don't reinstall them
COPY pyproject.toml README.md ./
COPY src/medrag/__init__.py src/medrag/__init__.py
ARG WITH_DENSE=auto
COPY artifacts/bundle/manifest.json /tmp/manifest.json
RUN if [ "$WITH_DENSE" = "1" ] || { [ "$WITH_DENSE" = "auto" ] && grep -q '"fastembed"' /tmp/manifest.json; }; then \
      pip install ".[dense]"; else pip install .; fi

COPY src ./src
COPY prompts ./prompts
RUN pip install --no-deps . && useradd --create-home --uid 10001 app
COPY artifacts/bundle ./artifacts/bundle
# Fetch any pretrained embedding model at build time so the container starts offline
RUN python -c "from medrag.retrieval import Bundle; Bundle.load('artifacts/bundle').search('warm up', 1)" \
    && mkdir -p /app/logs /app/.cache && chown -R app /app/logs /app/.cache

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"
CMD ["sh", "-c", "uvicorn medrag.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers"]
