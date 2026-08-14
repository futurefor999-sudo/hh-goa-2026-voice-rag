"""
Chunking strategies.

The brief explicitly penalizes a single naive fixed-size approach, so this
module implements several and a "hybrid" dispatcher that picks per-document:

  1. fixed        — fixed token/word window with overlap (baseline)
  2. sentence      — splits on sentence boundaries, packs sentences up to
                     a max size so we never cut mid-sentence
  3. semantic      — groups adjacent sentences by embedding similarity so
                     a chunk boundary falls where the topic actually shifts,
                     not at an arbitrary character count
  4. metadata_aware— wraps any of the above with document-level metadata
                     (source id, position, doc title/section if present)
                     propagated onto every chunk, so retrieval can filter
                     or re-rank using it later

All strategies return a list of Chunk objects with consistent shape so
downstream code (embeddings, vector store) doesn't care which strategy
produced them.
"""
from __future__ import annotations
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    strategy: str
    position: int  # order within the source document
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "doc_id": self.doc_id,
            "strategy": self.strategy,
            "position": self.position,
            "metadata": self.metadata,
        }


def _new_id(doc_id: str, position: int) -> str:
    return f"{doc_id}::{position}::{uuid.uuid4().hex[:8]}"


def chunk_fixed(
    text: str,
    doc_id: str,
    size: int = 250,
    overlap: int = 50,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Fixed-size word window with overlap. Simple, predictable, but can
    split mid-sentence — kept as a baseline / fallback strategy only."""
    words = text.split()
    if not words:
        return []
    metadata = metadata or {}
    chunks: list[Chunk] = []
    step = max(size - overlap, 1)
    pos = 0
    i = 0
    while i < len(words):
        window = words[i : i + size]
        if not window:
            break
        chunk_text = " ".join(window)
        chunks.append(
            Chunk(
                id=_new_id(doc_id, pos),
                text=chunk_text,
                doc_id=doc_id,
                strategy="fixed",
                position=pos,
                metadata={**metadata, "word_start": i, "word_end": i + len(window)},
            )
        )
        pos += 1
        i += step
    return chunks


def chunk_sentence(
    text: str,
    doc_id: str,
    max_words: int = 180,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Pack whole sentences into chunks up to max_words. Never splits a
    sentence, which fixed-size chunking can do."""
    metadata = metadata or {}
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_words = 0
    pos = 0
    for sent in sentences:
        w = len(sent.split())
        if buf and buf_words + w > max_words:
            chunk_text = " ".join(buf)
            chunks.append(
                Chunk(
                    id=_new_id(doc_id, pos),
                    text=chunk_text,
                    doc_id=doc_id,
                    strategy="sentence",
                    position=pos,
                    metadata=dict(metadata),
                )
            )
            pos += 1
            buf, buf_words = [], 0
        buf.append(sent)
        buf_words += w
    if buf:
        chunks.append(
            Chunk(
                id=_new_id(doc_id, pos),
                text=" ".join(buf),
                doc_id=doc_id,
                strategy="sentence",
                position=pos,
                metadata=dict(metadata),
            )
        )
    return chunks


def chunk_semantic(
    text: str,
    doc_id: str,
    embed_fn,
    similarity_drop: float = 0.35,
    max_words: int = 220,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Sentence-boundary chunking where the *break points* are chosen by
    embedding similarity between consecutive sentences: when similarity to
    the running chunk centroid drops below `similarity_drop`, that's a
    topic shift, so we cut there instead of at a fixed length.

    `embed_fn` is any callable: list[str] -> np.ndarray of shape (n, d).
    Passing it in (rather than importing an embedder here) keeps this
    module decoupled from whichever embedding backend is configured.
    """
    metadata = metadata or {}
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [
            Chunk(
                id=_new_id(doc_id, 0),
                text=sentences[0],
                doc_id=doc_id,
                strategy="semantic",
                position=0,
                metadata=dict(metadata),
            )
        ]

    vecs = embed_fn(sentences)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    unit = vecs / norms

    chunks: list[Chunk] = []
    buf = [sentences[0]]
    buf_vecs = [unit[0]]
    buf_words = len(sentences[0].split())
    pos = 0

    for i in range(1, len(sentences)):
        centroid = np.mean(buf_vecs, axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        sim = float(np.dot(centroid_norm, unit[i]))
        w = len(sentences[i].split())

        topic_shift = sim < similarity_drop
        too_long = buf_words + w > max_words

        if topic_shift or too_long:
            chunks.append(
                Chunk(
                    id=_new_id(doc_id, pos),
                    text=" ".join(buf),
                    doc_id=doc_id,
                    strategy="semantic",
                    position=pos,
                    metadata={**metadata, "break_reason": "topic_shift" if topic_shift else "max_len"},
                )
            )
            pos += 1
            buf, buf_vecs, buf_words = [], [], 0

        buf.append(sentences[i])
        buf_vecs.append(unit[i])
        buf_words += w

    if buf:
        chunks.append(
            Chunk(
                id=_new_id(doc_id, pos),
                text=" ".join(buf),
                doc_id=doc_id,
                strategy="semantic",
                position=pos,
                metadata=dict(metadata),
            )
        )
    return chunks


def chunk_document(
    text: str,
    doc_id: str,
    strategy: str = "hybrid",
    embed_fn=None,
    metadata: dict | None = None,
    fixed_size: int = 250,
    fixed_overlap: int = 50,
) -> list[Chunk]:
    """Single entry point used by ingestion. `strategy`:
      - "fixed"     -> chunk_fixed
      - "sentence"  -> chunk_sentence
      - "semantic"  -> chunk_semantic (requires embed_fn)
      - "hybrid"    -> semantic if embed_fn given and doc is long enough
                       to benefit, else sentence-packing; falls back to
                       fixed only if the text has no clean sentence breaks
                       (e.g. code, logs, tabular text dumped as one blob)
    All chunks get `metadata` merged in, so doc-level info (source, title,
    section, language, etc.) is queryable/filterable at retrieval time —
    this is the "metadata-aware chunking" requirement.
    """
    metadata = dict(metadata or {})
    metadata.setdefault("doc_id", doc_id)

    if strategy == "fixed":
        return chunk_fixed(text, doc_id, fixed_size, fixed_overlap, metadata)

    if strategy == "sentence":
        return chunk_sentence(text, doc_id, metadata=metadata)

    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("chunk_semantic requires an embed_fn")
        return chunk_semantic(text, doc_id, embed_fn, metadata=metadata)

    if strategy == "hybrid":
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            # No detectable sentence structure — fall back to fixed windows
            return chunk_fixed(text, doc_id, fixed_size, fixed_overlap, metadata)
        if embed_fn is not None and len(sentences) >= 4:
            return chunk_semantic(text, doc_id, embed_fn, metadata=metadata)
        return chunk_sentence(text, doc_id, metadata=metadata)

    raise ValueError(f"unknown chunk strategy: {strategy}")
