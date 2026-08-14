"""
Latency benchmark: runs a set of test queries through the pipeline and
reports P50/P70/P100 per stage, per the submission brief's requirement
to measure "across a reasonable number of test queries — not a single
best-case run."

Usage:
    python -m scripts.benchmark --index data/index.pkl --queries data/test_queries.txt
    # or generate synthetic queries from the corpus itself for a quick smoke test:
    python -m scripts.benchmark --index data/index.pkl --synthetic 20
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG
from src.retrieval.embeddings import load_embedder
from src.retrieval.vector_store import InMemoryVectorStore
from src.generation.generator import get_generator
from src.pipeline import RAGPipeline
from src.latency import LatencyTracker


def load_queries(path: str) -> list[str]:
    """Accepts either a plain newline-delimited text file of queries, or
    a .jsonl file of {"query": ...} objects (e.g. the eval_queries file
    scripts/ingest.py writes out when ingesting from HuggingFace)."""
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.endswith(".jsonl") or line.startswith("{"):
                obj = json.loads(line)
                q = obj.get("query", "").strip()
                if q:
                    queries.append(q)
            else:
                queries.append(line)
    return queries


def synthetic_queries(store: InMemoryVectorStore, n: int) -> list[str]:
    """Quick smoke-test query set built from the indexed chunks
    themselves (first sentence of every Nth chunk, phrased as a
    question-ish fragment). Fine for exercising the pipeline's latency
    profile; not a substitute for a real held-out query set at
    submission time."""
    chunks = store._chunks  # noqa: SLF001 - internal use within same codebase for a test-only helper
    step = max(len(chunks) // n, 1) if chunks else 1
    out = []
    for c in chunks[::step][:n]:
        first_sentence = c.text.split(".")[0]
        out.append(first_sentence[:120])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.pkl")
    ap.add_argument("--queries", default=None, help="path to newline-delimited query file")
    ap.add_argument("--synthetic", type=int, default=None, help="generate N synthetic queries from the index instead")
    ap.add_argument("--out", default="data/latency_report.json")
    args = ap.parse_args()

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

    if args.queries:
        queries = load_queries(args.queries)
    elif args.synthetic:
        queries = synthetic_queries(store, args.synthetic)
    else:
        raise SystemExit("provide --queries <file> or --synthetic <N>")

    print(f"Running {len(queries)} queries...")
    tracker = LatencyTracker()
    statuses = {"answered": 0, "abstained": 0, "blocked": 0}
    for i, q in enumerate(queries):
        resp = pipeline.answer(q)
        tracker.add(resp.timing)
        statuses[resp.status] += 1
        print(f"[{i+1}/{len(queries)}] status={resp.status:<10} total={resp.timing.total_ms:.1f}ms  query={q[:60]!r}")

    tracker.print_report()
    print(f"\nStatus breakdown: {statuses}")

    report = {"n_queries": len(queries), "status_breakdown": statuses, "latency": tracker.report()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
