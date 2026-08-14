"""
Runs the demo test cases (data/demo_test_cases.jsonl) through the full
pipeline and reports whether each one behaved as expected — this is the
"demo test cases for normal, off-topic, unsafe, and unsupported
questions" deliverable, and doubles as a quick sanity check that the
guardrails are actually doing something on a real index (not just in
the offline unit tests).

Usage:
    python -m scripts.ingest --source data/sample_corpus.jsonl --out data/index.pkl   # if not already done
    python -m scripts.demo_cases --index data/index.pkl
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


def load_cases(path: str) -> list[dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.pkl")
    ap.add_argument("--cases", default="data/demo_test_cases.jsonl")
    ap.add_argument("--out", default="data/demo_results.json")
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

    cases = load_cases(args.cases)
    results = []
    n_match = 0
    print(f"{'category':<14}{'expected':<20}{'actual':<12}{'match':<7}query")
    for case in cases:
        resp = pipeline.answer(case["query"])
        expected = case["expected_status"]
        accepted = expected if isinstance(expected, list) else [expected]
        match = resp.status in accepted
        n_match += match
        results.append({
            **case,
            "actual_status": resp.status,
            "actual_reason": resp.reason,
            "answer": resp.answer,
            "match": match,
            "total_ms": resp.timing.total_ms,
        })
        print(f"{case['category']:<14}{'/'.join(accepted):<24}{resp.status:<12}"
              f"{'YES' if match else 'no':<7}{case['query'][:55]}")

    print(f"\n{n_match}/{len(cases)} matched expected status.")
    print(
        "Note: a mismatch is not automatically a bug — the guardrail thresholds "
        "(OFFTOPIC_MIN_SIM, GROUNDING_MIN_OVERLAP) and the generation backend "
        "both affect where exactly a case lands, especially with the offline "
        "mock generator, which is heuristic rather than a real LLM. Use this "
        "report to *tune* thresholds against your real corpus, not as a pass/fail gate."
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
