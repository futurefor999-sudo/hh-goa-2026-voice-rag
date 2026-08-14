"""
Ingestion: dataset -> chunks -> embeddings -> vector store on disk.

Usage:
    # Fast local smoke test on the small bundled sample corpus:
    python -m scripts.ingest --source data/sample_corpus.jsonl --out data/index.pkl

    # Real submission run against MS MARCO-XL from HuggingFace:
    python -m scripts.ingest --source hf --hf-config <language-config> \
        --limit 5000 --out data/index.pkl

Run `python -m scripts.ingest --list-hf-configs` to see the available
MS MARCO-XL language configs before picking one — ai4bharat/MSMARCO-XL
is a multilingual set, and this repo doesn't hardcode which language(s)
the team should build against; that's a project decision, not ours.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG
from src.chunking.strategies import chunk_document
from src.retrieval.embeddings import get_embedder, save_embedder
from src.retrieval.vector_store import get_vector_store


def load_local_jsonl(path: str) -> list[dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


MSMARCO_XI_LANGUAGES = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
    "or": "Odia", "pa": "Punjabi", "sa": "Sanskrit", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu",
}


def _transform_msmarco_row(row: dict, row_index: int, passage_field: str, seen_texts: set[str]) -> tuple[list[dict], dict]:
    """Pure transform: one ai4bharat/MSMARCO-XI row -> (new_docs, eval_query).
    Pulled out of load_hf_msmarco_xl so it can be unit-tested against the
    documented schema without needing network access or the `datasets`
    package at all (see tests/test_ingest.py) — this is the part of the
    HF loading path that's actually our logic and can go wrong; the
    `load_dataset(...)` call itself is just HuggingFace's client."""
    passages = row.get("passages") or {}
    texts = passages.get(passage_field) or []
    is_selected = passages.get("is_selected") or []
    query_id = row.get("query_id", row_index)

    new_docs = []
    relevant_doc_ids = []
    for j, text in enumerate(texts):
        if not text or not text.strip():
            continue
        doc_id = f"{query_id}_{j}"
        # MS MARCO passages repeat heavily across queries (same passage
        # backs many questions) — skip exact dupes so we don't index
        # (and pay embedding cost for) the same text thousands of times.
        key = text.strip()
        if key not in seen_texts:
            seen_texts.add(key)
            new_docs.append({"doc_id": doc_id, "title": row.get("query", "")[:80], "text": text})
        if j < len(is_selected) and is_selected[j] == 1:
            relevant_doc_ids.append(doc_id)

    eval_query = {
        "query": row.get("query", ""),
        "query_id": query_id,
        "query_type": row.get("query_type", ""),
        "relevant_doc_ids": relevant_doc_ids,  # empty list = no passage in this row answers it
    }
    return new_docs, eval_query


def load_hf_msmarco_xl(
    language: str,
    limit_rows: int | None,
    passage_field: str = "Translated_passages",
    split: str = "train",
) -> tuple[list[dict], list[dict]]:
    """Loads ai4bharat/MSMARCO-XI (the dataset the task brief links to —
    HuggingFace renders the dataset's capital 'I' in a way that's easy to
    misread as a lowercase 'l', hence "MSMARCO-XL" in the task doc).

    Confirmed schema (https://huggingface.co/datasets/ai4bharat/MSMARCO-XI):
      query (str), Answer (str), query_id (int), query_type (str),
      passages: {is_selected: list[int], English_passages: list[str],
                 Translated_passages: list[str]},
      Eng_Query (str), Eng_Answer (str), source_lang, target_lang, meta

    14 language configs: as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, te, ur.
    Requires network access + the `datasets` package — not available in a
    sandboxed/offline dev environment, which is why sample_corpus.jsonl
    exists for local iteration without hitting HuggingFace at all, and
    why the row-transform logic above is factored out into a function
    that's unit-tested against the documented schema independently of
    this network-calling wrapper (see tests/test_ingest.py).

    Returns (docs, eval_queries):
      docs         — one dict per passage: {doc_id, title, text} — this is
                     what gets chunked and indexed, same shape the rest of
                     ingest.py already expects.
      eval_queries — one dict per row: {query, query_id, query_type,
                     relevant_doc_ids} using the dataset's own
                     is_selected flags as ground truth. This is what
                     scripts/benchmark.py's --queries can consume for
                     a *real* (not synthetic) latency benchmark, and what
                     demo test cases pull "normal" and "unsupported"
                     examples from (see scripts/demo_cases.py).
    """
    if language not in MSMARCO_XI_LANGUAGES:
        raise ValueError(
            f"unknown language config '{language}'. Available: {sorted(MSMARCO_XI_LANGUAGES)}"
        )
    if passage_field not in ("Translated_passages", "English_passages"):
        raise ValueError("passage_field must be 'Translated_passages' or 'English_passages'")

    from datasets import load_dataset

    streaming = limit_rows is not None
    ds = load_dataset("ai4bharat/MSMARCO-XI", language, split=split, streaming=streaming)

    docs: list[dict] = []
    eval_queries: list[dict] = []
    seen_texts: set[str] = set()

    for i, row in enumerate(ds):
        if limit_rows is not None and i >= limit_rows:
            break
        new_docs, eval_query = _transform_msmarco_row(row, i, passage_field, seen_texts)
        docs.extend(new_docs)
        eval_queries.append(eval_query)

    return docs, eval_queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/sample_corpus.jsonl",
                     help="'hf' to pull ai4bharat/MSMARCO-XI, or a path to a local .jsonl "
                          "with {doc_id, title, text} per line")
    ap.add_argument("--language", default="hi", choices=sorted(MSMARCO_XI_LANGUAGES),
                     help="MSMARCO-XI language config (only used with --source hf)")
    ap.add_argument("--passage-field", default="Translated_passages",
                     choices=["Translated_passages", "English_passages"],
                     help="which passage text to index (only used with --source hf)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of dataset rows (esp. for --source hf)")
    ap.add_argument("--out", default="data/index.pkl")
    ap.add_argument("--strategy", default=CONFIG.chunk_strategy, choices=["fixed", "sentence", "semantic", "hybrid"])
    ap.add_argument("--list-hf-languages", action="store_true")
    args = ap.parse_args()

    if args.list_hf_languages:
        for code, name in sorted(MSMARCO_XI_LANGUAGES.items()):
            print(f"{code}: {name}")
        return

    print(f"Loading documents from: {args.source}")
    eval_queries = []
    if args.source == "hf":
        docs, eval_queries = load_hf_msmarco_xl(args.language, args.limit, args.passage_field)
    else:
        docs = load_local_jsonl(args.source)
        if args.limit:
            docs = docs[: args.limit]
    print(f"Loaded {len(docs)} documents.")

    embedder = get_embedder(CONFIG.embedding_provider, CONFIG.st_model)

    # TF-IDF needs to see the corpus before it can encode anything; fit
    # it up front. Semantic chunking (below) needs a *sentence-level*
    # embed_fn, which we build from the same fitted embedder.
    print(f"Fitting embedder ({CONFIG.embedding_provider}) on corpus...")
    embedder.fit([d["text"] for d in docs])

    def embed_fn(sentences: list[str]):
        return embedder.encode(sentences)

    print(f"Chunking with strategy='{args.strategy}'...")
    all_chunks = []
    for d in docs:
        chunks = chunk_document(
            text=d["text"],
            doc_id=d["doc_id"],
            strategy=args.strategy,
            embed_fn=embed_fn if args.strategy in ("semantic", "hybrid") else None,
            metadata={"title": d.get("title", ""), "source": args.source},
            fixed_size=CONFIG.fixed_chunk_size,
            fixed_overlap=CONFIG.fixed_chunk_overlap,
        )
        all_chunks.extend(chunks)
    print(f"Produced {len(all_chunks)} chunks from {len(docs)} documents "
          f"({len(all_chunks)/max(len(docs),1):.1f} chunks/doc).")

    print("Embedding chunks...")
    t0 = time.perf_counter()
    vectors = embedder.encode([c.text for c in all_chunks])
    print(f"Embedded {len(all_chunks)} chunks in {time.perf_counter()-t0:.2f}s")

    store = get_vector_store("memory")
    store.add(all_chunks, vectors)
    store.save(args.out)

    embedder_path = str(Path(args.out).with_suffix(".embedder.pkl"))
    save_embedder(embedder, embedder_path)

    print(f"Saved index with {len(store)} chunks to {args.out}")
    print(f"Saved fitted embedder to {embedder_path} "
          f"(required at query time — TF-IDF's vocabulary is corpus-specific)")

    if eval_queries:
        eval_path = str(Path(args.out).with_suffix("")) + ".eval_queries.jsonl"
        with open(eval_path, "w", encoding="utf-8") as f:
            for q in eval_queries:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        n_answerable = sum(1 for q in eval_queries if q["relevant_doc_ids"])
        print(f"Saved {len(eval_queries)} dataset queries ({n_answerable} with a known relevant "
              f"passage, {len(eval_queries) - n_answerable} unsupported-by-this-row) to {eval_path} "
              f"— feed this to scripts/benchmark.py for a real (non-synthetic) latency run, and to "
              f"scripts/demo_cases.py for real 'normal'/'unsupported' demo examples.")


if __name__ == "__main__":
    main()
