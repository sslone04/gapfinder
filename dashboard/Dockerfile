# Cloud Run SERVICE image for the read-only dashboard.
# No state is baked in -- everything is read live from gs://$STATE_BUCKET/state/.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py index.html grid_config.py ./
# frozen corpus: abstracts + species MeSH for the paper panel. Immutable by design;
# papers ingested since live in corpus_live.csv in the state bucket.
COPY corpus.csv ./
EXPOSE 8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
