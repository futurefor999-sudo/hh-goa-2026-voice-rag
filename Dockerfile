# Deployable container for the voice RAG demo.
# Builds a real index from the bundled sample corpus at image-build time
# so the container is immediately queryable on first boot — swap the
# ingest command below (or run it at deploy time via a build hook) to
# point at the real MSMARCO-XI corpus once you're ready; see README.md.
FROM python:3.11-slim

WORKDIR /app

# System deps for scientific python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Full requirements.txt includes optional extras (sentence-transformers,
# faiss-cpu, datasets) that aren't needed to run the default TF-IDF /
# in-memory / mock-or-API-backed configuration. Installing the minimal
# set keeps the image small and the build fast; add the optional
# packages back in if you switch EMBEDDING_PROVIDER or ingest from HF.
RUN pip install --no-cache-dir \
    numpy scipy scikit-learn requests python-dotenv \
    anthropic openai \
    fastapi "uvicorn[standard]" python-multipart

COPY . .

# Build the demo index from the bundled sample corpus so the container
# is queryable immediately. GEN_PROVIDER=mock here only affects this
# build step (chunk embedding doesn't need a generation key); the real
# GEN_PROVIDER/keys are supplied as runtime environment variables by
# your hosting platform, never baked into the image.
RUN GEN_PROVIDER=mock python -m scripts.ingest --source data/sample_corpus.jsonl --out data/index.pkl --strategy hybrid

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
