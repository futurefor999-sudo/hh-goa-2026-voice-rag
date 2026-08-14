"""
Vector store.

Default backend is a plain in-memory cosine-similarity index (numpy).
That's a deliberate choice for the hackathon build: it has zero external
dependencies, is trivially inspectable/debuggable, and is fast enough
for corpora up to roughly a few hundred thousand chunks — plenty for a
demo dataset slice.

A FaissVectorStore is also provided (flat IP index) for when the corpus
is large enough that brute-force numpy stops being the fastest option;
swap it in via `get_vector_store(backend="faiss")`. Both share the same
interface so the rest of the pipeline is backend-agnostic.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.chunking.strategies import Chunk


class InMemoryVectorStore:
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None  # (n, d), L2-normalized

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors must have the same length")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        normed = (vectors / norms).astype(np.float32)
        self._chunks.extend(chunks)
        self._vectors = normed if self._vectors is None else np.vstack([self._vectors, normed])

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        metadata_filter: Callable[[dict], bool] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if self._vectors is None or len(self._chunks) == 0:
            return []
        qv = query_vector.astype(np.float32)
        qn = np.linalg.norm(qv)
        if qn > 0:
            qv = qv / qn

        sims = self._vectors @ qv  # cosine sim since both sides are unit-normed

        if metadata_filter is not None:
            mask = np.array([metadata_filter(c.metadata) for c in self._chunks])
            sims = np.where(mask, sims, -np.inf)

        k = min(top_k, len(self._chunks))
        if k <= 0:
            return []
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(self._chunks[i], float(sims[i])) for i in top_idx if sims[i] != -np.inf]

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": [c.to_dict() for c in self._chunks], "vectors": self._vectors}, f)

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryVectorStore":
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls()
        store._chunks = [Chunk(**d) for d in data["chunks"]]
        store._vectors = data["vectors"]
        return store


class FaissVectorStore:
    """Optional faster backend for larger corpora. Requires faiss-cpu."""

    def __init__(self, dim: int):
        import faiss

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        normed = (vectors / norms).astype(np.float32)
        self._index.add(normed)
        self._chunks.extend(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        metadata_filter: Callable[[dict], bool] | None = None,
    ) -> list[tuple[Chunk, float]]:
        qv = query_vector.astype(np.float32).reshape(1, -1)
        qn = np.linalg.norm(qv)
        if qn > 0:
            qv = qv / qn
        # over-fetch when filtering, since faiss can't filter natively
        k = min(top_k * 5 if metadata_filter else top_k, len(self._chunks))
        if k <= 0:
            return []
        sims, idx = self._index.search(qv, k)
        results = []
        for i, s in zip(idx[0], sims[0]):
            if i < 0:
                continue
            chunk = self._chunks[i]
            if metadata_filter and not metadata_filter(chunk.metadata):
                continue
            results.append((chunk, float(s)))
            if len(results) >= top_k:
                break
        return results

    def __len__(self) -> int:
        return len(self._chunks)


def get_vector_store(backend: str = "memory", dim: int | None = None):
    if backend == "memory":
        return InMemoryVectorStore()
    if backend == "faiss":
        if dim is None:
            raise ValueError("dim is required for the faiss backend")
        return FaissVectorStore(dim)
    raise ValueError(f"unknown vector store backend: {backend}")
