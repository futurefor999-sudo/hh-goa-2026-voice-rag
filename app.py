"""
Live demo server.

    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /                          -> mobile-friendly web UI (static/index.html)
  GET  /health                    -> liveness check
  POST /query/text  {"query": str}       -> runs text straight through the pipeline
  POST /query/voice  (multipart file=..) -> runs audio through Sarvam STT, then the pipeline

Both /query/* endpoints return the same PipelineResponse JSON shape
(see src/pipeline.py), including the per-stage latency breakdown, which
the web UI's stage rail renders live.
"""
from __future__ import annotations
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import CONFIG
from src.retrieval.embeddings import load_embedder
from src.retrieval.vector_store import InMemoryVectorStore
from src.generation.generator import get_generator
from src.stt.sarvam_stt import get_stt_client, SarvamSTTError
from src.pipeline import RAGPipeline

app = FastAPI(title="Voice-Enabled RAG — HH Goa 2026 Task 2")

INDEX_PATH = "data/index.pkl"
EMBEDDER_PATH = str(Path(INDEX_PATH).with_suffix(".embedder.pkl"))
STATIC_DIR = Path(__file__).parent / "static"

_state = {}


@app.on_event("startup")
def load_pipeline():
    if not Path(INDEX_PATH).exists():
        print(
            f"[startup warning] No index found at {INDEX_PATH}. "
            f"Run `python -m scripts.ingest` first. The server will start, "
            f"the web UI will load, but /query/* will return 503 until an index exists."
        )
        _state["pipeline"] = None
    else:
        embedder = load_embedder(EMBEDDER_PATH)
        store = InMemoryVectorStore.load(INDEX_PATH)
        generator = get_generator(
            CONFIG.gen_provider, CONFIG.anthropic_api_key, CONFIG.anthropic_model,
            CONFIG.openai_api_key, CONFIG.openai_model,
        )
        _state["pipeline"] = RAGPipeline(
            embedder=embedder, vector_store=store, generator=generator,
            top_k=CONFIG.top_k, offtopic_min_sim=CONFIG.offtopic_min_sim,
            grounding_min_overlap=CONFIG.grounding_min_overlap,
        )

    try:
        _state["stt"] = get_stt_client(CONFIG.stt_provider, CONFIG.sarvam_api_key, CONFIG.sarvam_stt_model)
    except SarvamSTTError as e:
        # Don't crash the whole server if the key isn't set yet — text
        # queries should still work; /query/voice will error clearly.
        print(f"[startup warning] STT client not initialized: {e}")
        _state["stt"] = None


def _require_pipeline() -> RAGPipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            503, f"No index loaded — run `python -m scripts.ingest --out {INDEX_PATH}` "
                 f"on the server and restart."
        )
    return pipeline


@app.get("/")
def root():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(404, "static/index.html not found")
    return FileResponse(index_file)


@app.get("/health")
def health():
    pipeline = _state.get("pipeline")
    return {
        "status": "ok",
        "index_loaded": pipeline is not None,
        "index_size": len(pipeline.vector_store) if pipeline is not None else 0,
        "stt_configured": _state.get("stt") is not None,
        "gen_provider": CONFIG.gen_provider,
    }


class TextQuery(BaseModel):
    query: str


@app.post("/query/text")
def query_text(body: TextQuery):
    if not body.query.strip():
        raise HTTPException(400, "query must not be empty")
    pipeline = _require_pipeline()
    response = pipeline.answer(body.query)
    return response.to_dict()


@app.post("/query/voice")
async def query_voice(file: UploadFile = File(...)):
    pipeline = _require_pipeline()
    if _state.get("stt") is None:
        raise HTTPException(
            503, "STT is not configured on the server — set SARVAM_API_KEY and restart."
        )
    audio_bytes = await file.read()
    t0 = time.perf_counter()
    try:
        query = _state["stt"].transcribe(audio_bytes, filename=file.filename or "audio.webm")
    except SarvamSTTError as e:
        raise HTTPException(502, f"STT failed: {e}") from e
    stt_ms = (time.perf_counter() - t0) * 1000

    response = pipeline.answer(query, stt_ms=stt_ms)
    out = response.to_dict()
    out["transcript"] = query
    return out


# Mounted last so it doesn't shadow the API routes above — anything not
# matched by /, /health, /query/* falls through to static assets (none
# beyond index.html today, but this keeps room for e.g. a favicon).
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
