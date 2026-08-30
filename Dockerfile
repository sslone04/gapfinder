# Cloud Run Job image for the gapfinder daily cycle.
# Secrets are injected at runtime from Secret Manager -- never baked in (.dockerignore
# excludes .env). Mutable state lives in GCS, not the image; only frozen inputs are copied.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pipeline modules
COPY agent_loop.py probe_v3.py build_grid.py diff_engine.py build_corpus.py ./
COPY grid_config.py ./

# frozen inputs: the query, the base corpus, the held-out replay set, cached PubMed XML
COPY corpus_config.json corpus.csv corpus_future.csv ./
COPY raw_cache/ ./raw_cache/

ENTRYPOINT ["python", "agent_loop.py", "--replay-batch", "10"]
