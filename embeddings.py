"""
Embedding backends.

Default is a TF-IDF vectorizer (scikit-learn) — it needs no model
download, works fully offline, and is enough to demo/benchmark the
whole pipeline. For submission-quality retrieval, switch
EMBEDDING_PROVIDER to "sentence-transformers" in .env, which uses a real
dense embedding model instead. Both implement the same `Embedder`
interface so nothing else in the pipeline needs to change.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def fit(self, corpus: list[str]) -> "Embedder": ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """Offline, dependency-light default. Must be `fit` once on the
    corpus (or a representative sample) before `encode` is used for
    query-time embedding, since TF-IDF vocabulary is corpus-relative."""

    def __init__(self, max_features: int = 50_000):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> "TfidfEmbedder":
        self._vectorizer.fit(corpus)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit(corpus) must be called before encode()")
        mat = self._vectorizer.transform(texts)
        return mat.toarray().astype(np.float32)


class SentenceTransformerEmbedder:
    """Real dense embeddings. Requires `sentence-transformers` installed
    and (on first run) network access to pull the model weights."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def fit(self, corpus: list[str]) -> "SentenceTransformerEmbedder":
        # Nothing to fit — pretrained model. Kept for interface parity.
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, show_progress_bar=False), dtype=np.float32)


def get_embedder(provider: str, st_model: str = "intfloat/multilingual-e5-base") -> Embedder:
    if provider == "tfidf":
        return TfidfEmbedder()
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(st_model)
    raise ValueError(f"unknown embedding provider: {provider}")


def save_embedder(embedder: Embedder, path: str | Path) -> None:
    """TF-IDF's vocabulary is fit from the corpus at ingest time, so the
    exact same fitted object must be reused at query time — a freshly
    constructed TfidfEmbedder would have a different (or empty)
    vocabulary. Sentence-transformer embedders don't strictly need this
    (the model is pretrained, not corpus-fit) but saving is cheap and
    keeps ingest/query symmetric regardless of backend."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embedder, f)


def load_embedder(path: str | Path) -> Embedder:
    with open(path, "rb") as f:
        return pickle.load(f)
