"""
Tests the ai4bharat/MSMARCO-XI row-transform logic against the schema
documented at https://huggingface.co/datasets/ai4bharat/MSMARCO-XI —
without calling `datasets.load_dataset` or touching the network.

This sandbox has no network access, so the actual HTTP fetch from
HuggingFace can't be exercised here. What CAN and should be verified
without network is: given a row shaped exactly like HuggingFace's own
documented example, does our transform logic (_transform_msmarco_row)
produce correct docs + eval_queries? That's the part of the ingestion
path that's our code and can have real bugs; `load_dataset(...)` itself
is HuggingFace's client, not ours to test.

Run with real network + credentials to exercise the actual fetch:
    python -m scripts.ingest --source hf --language hi --limit 50 --out data/index.pkl
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingest import _transform_msmarco_row, MSMARCO_XI_LANGUAGES


# Exact shape from the dataset card's own worked example (Assamese),
# https://huggingface.co/datasets/ai4bharat/MSMARCO-XI — field names,
# nesting, and types copied verbatim rather than approximated.
SAMPLE_ROW = {
    "source_lang": "eng_Latn",
    "target_lang": "asm_Beng",
    "meta": {
        "model_name": "ckpt-3epochs-sft-then-400k-kd",
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    "query": "মেনহাটন প্ৰকল্পৰ সফলতাৰ তাৎক্ষণিক প্ৰভাৱ কি আছিল?",
    "Answer": "মেনহাটন প্ৰকল্পৰ সফলতাৰ তাৎক্ষণিক প্ৰভাৱ আছিল...",
    "query_id": 1185869,
    "query_type": "DESCRIPTION",
    "passages": {
        "is_selected": [1, 0, 0],
        "English_passages": [
            "The presence of communication amid scientific minds was equally important...",
            "Some unrelated passage about something else entirely.",
            "Another unrelated passage.",
        ],
        "Translated_passages": [
            "বৈজ্ঞানিক মনৰ মাজত যোগাযোগৰ উপস্থিতি...",
            "অন্য এটা অসম্পৰ্কিত অনুচ্ছেদ।",
            "আৰু এটা অসম্পৰ্কিত অনুচ্ছেদ।",
        ],
    },
    "Eng_Query": ")what was the immediate impact of the success of the manhattan project?",
    "Eng_Answer": "The immediate impact of the success of the manhattan project was...",
}


def test_language_configs_match_dataset_card():
    # 14 languages listed on the dataset card's supported-languages table
    assert len(MSMARCO_XI_LANGUAGES) == 14
    assert MSMARCO_XI_LANGUAGES["hi"] == "Hindi"
    assert MSMARCO_XI_LANGUAGES["as"] == "Assamese"


def test_transform_produces_one_doc_per_nonempty_passage():
    seen = set()
    docs, eval_query = _transform_msmarco_row(SAMPLE_ROW, 0, "Translated_passages", seen)
    assert len(docs) == 3
    assert all(d["text"] for d in docs)
    assert docs[0]["doc_id"] == "1185869_0"
    assert docs[1]["doc_id"] == "1185869_1"


def test_transform_respects_passage_field_choice():
    seen = set()
    docs, _ = _transform_msmarco_row(SAMPLE_ROW, 0, "English_passages", seen)
    assert docs[0]["text"].startswith("The presence of communication")


def test_transform_captures_relevant_doc_ids_from_is_selected():
    seen = set()
    _, eval_query = _transform_msmarco_row(SAMPLE_ROW, 0, "Translated_passages", seen)
    # is_selected = [1, 0, 0] -> only the first passage is "relevant"
    assert eval_query["relevant_doc_ids"] == ["1185869_0"]
    assert eval_query["query_id"] == 1185869
    assert eval_query["query_type"] == "DESCRIPTION"
    assert eval_query["query"] == SAMPLE_ROW["query"]


def test_transform_dedupes_across_rows_via_seen_texts():
    seen = set()
    docs1, _ = _transform_msmarco_row(SAMPLE_ROW, 0, "Translated_passages", seen)
    # A second row that happens to share the first passage's exact text
    # (MS MARCO passages repeat heavily across queries) should not be
    # re-added to the corpus.
    row2 = {**SAMPLE_ROW, "query_id": 999}
    docs2, _ = _transform_msmarco_row(row2, 1, "Translated_passages", seen)
    assert len(docs1) == 3
    assert len(docs2) == 0  # all 3 texts already seen


def test_transform_handles_row_with_no_relevant_passage():
    # is_selected all zeros -> a genuinely "unsupported" query: retrieval
    # may still find passages, but none of them is the answer for this
    # query. This is exactly the shape scripts/demo_cases.py's
    # "unsupported" category is meant to be pulled from once real
    # MSMARCO-XI data is ingested.
    row = {**SAMPLE_ROW, "query_id": 42, "passages": {**SAMPLE_ROW["passages"], "is_selected": [0, 0, 0]}}
    seen = set()
    _, eval_query = _transform_msmarco_row(row, 0, "Translated_passages", seen)
    assert eval_query["relevant_doc_ids"] == []


def test_transform_handles_missing_or_short_is_selected_list():
    # Defensive: real-world rows can have passages shorter than expected,
    # or an is_selected list that doesn't cover every passage index.
    row = {
        **SAMPLE_ROW,
        "query_id": 7,
        "passages": {
            "is_selected": [1],  # shorter than English_passages/Translated_passages
            "English_passages": ["only one passage here"],
            "Translated_passages": ["এটা মাথোন অনুচ্ছেদ"],
        },
    }
    seen = set()
    docs, eval_query = _transform_msmarco_row(row, 0, "Translated_passages", seen)
    assert len(docs) == 1
    assert eval_query["relevant_doc_ids"] == ["7_0"]


def test_transform_skips_empty_passage_strings():
    row = {
        **SAMPLE_ROW,
        "query_id": 8,
        "passages": {
            "is_selected": [1, 0],
            "English_passages": ["real passage", ""],
            "Translated_passages": ["real passage", "   "],
        },
    }
    seen = set()
    docs, _ = _transform_msmarco_row(row, 0, "Translated_passages", seen)
    assert len(docs) == 1


if __name__ == "__main__":
    import traceback

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
