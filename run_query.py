"""
Run one query through the full pipeline from the command line.

Text-only by default (skips STT) so you can sanity-check retrieval and
generation without recording audio:
    python -m scripts.run_query --index data/index.pkl --query "What is RAG?"

Pass --audio to exercise the real Sarvam STT stage too:
    python -m scripts.run_query --index data/index.pkl --audio path/to/clip.wav
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG
from src.retrieval.embeddings import load_embedder
from src.retrieval.vector_store import InMemoryVectorStore
from src.generation.generator import get_generator
from src.stt.sarvam_stt import get_stt_client
from src.pipeline import RAGPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.pkl")
    ap.add_argument("--query", default=None, help="text query (skips STT)")
    ap.add_argument("--audio", default=None, help="path to a .wav file (goes through Sarvam STT)")
    args = ap.parse_args()

    if not args.query and not args.audio:
        ap.error("provide either --query or --audio")

    embedder_path = str(Path(args.index).with_suffix(".embedder.pkl"))
    embedder = load_embedder(embedder_path)
    store = InMemoryVectorStore.load(args.index)

    generator = get_generator(
        CONFIG.gen_provider, CONFIG.anthropic_api_key, CONFIG.anthropic_model,
        CONFIG.openai_api_key, CONFIG.openai_model,
    )

    pipeline = RAGPipeline(
        embedder=embedder, vector_store=store, generator=generator,
        top_k=CONFIG.top_k, offtopic_min_sim=CONFIG.offtopic_min_sim,
        grounding_min_overlap=CONFIG.grounding_min_overlap,
    )

    stt_ms = 0.0
    if args.audio:
        stt_client = get_stt_client(CONFIG.stt_provider, CONFIG.sarvam_api_key, CONFIG.sarvam_stt_model)
        audio_bytes = Path(args.audio).read_bytes()
        t0 = time.perf_counter()
        query = stt_client.transcribe(audio_bytes, filename=Path(args.audio).name)
        stt_ms = (time.perf_counter() - t0) * 1000
        print(f"[STT] transcript: {query!r}  ({stt_ms:.1f} ms)")
    else:
        query = args.query

    response = pipeline.answer(query, stt_ms=stt_ms)
    print(json.dumps(response.to_dict(), indent=2))


if __name__ == "__main__":
    main()
