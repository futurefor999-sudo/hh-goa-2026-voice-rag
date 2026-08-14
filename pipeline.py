"""
The harness.

This is the "run inside a proper harness, not a single raw prompt-in
text-out call" piece of the brief: structured request/response objects,
per-stage retries with backoff, explicit error recovery paths (each
failure mode returns a typed, structured response instead of raising
past the caller), and stage-level timing fed into LatencyTracker.

Stage order: unsafe-input guardrail -> retrieval -> offtopic guardrail
-> generation -> grounding guardrail. STT happens before this function
is called (see app.py / scripts/run_query.py), timed separately, since
voice input is optional — text queries skip it entirely.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable

from src.chunking.strategies import Chunk
from src.generation.generator import GenerationResult
from src.guardrails.checks import (
    run_pre_generation_guardrails,
    run_post_generation_guardrails,
)
from src.latency import StageTiming


class PipelineError(Exception):
    pass


@dataclass
class RetrievedChunk:
    text: str
    score: float
    doc_id: str
    metadata: dict


@dataclass
class PipelineResponse:
    query: str
    status: str  # "answered" | "abstained" | "blocked"
    answer: str | None
    reason: str | None
    retrieved: list[RetrievedChunk]
    timing: StageTiming

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "status": self.status,
            "answer": self.answer,
            "reason": self.reason,
            "retrieved": [r.__dict__ for r in self.retrieved],
            "timing": {
                "stt_ms": self.timing.stt_ms,
                "retrieval_ms": self.timing.retrieval_ms,
                "generation_ms": self.timing.generation_ms,
                "guardrails_ms": self.timing.guardrails_ms,
                "total_ms": self.timing.total_ms,
            },
        }


def _with_retries(fn: Callable, max_attempts: int = 3, base_delay_s: float = 0.25):
    """Exponential backoff retry wrapper for calls to external services
    (embedding model, generation API). Retrieval against the local
    vector store doesn't need this — nothing external can fail there."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - intentionally broad: this is a generic retry wrapper
            last_exc = e
            if attempt < max_attempts - 1:
                time.sleep(base_delay_s * (2**attempt))
    raise PipelineError(f"failed after {max_attempts} attempts: {last_exc}") from last_exc


class RAGPipeline:
    def __init__(self, embedder, vector_store, generator, top_k: int, offtopic_min_sim: float, grounding_min_overlap: float):
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.top_k = top_k
        self.offtopic_min_sim = offtopic_min_sim
        self.grounding_min_overlap = grounding_min_overlap

    def answer(self, query: str, stt_ms: float = 0.0) -> PipelineResponse:
        timing = StageTiming(stt_ms=stt_ms)

        # --- retrieval ---
        t0 = time.perf_counter()
        try:
            query_vec = _with_retries(lambda: self.embedder.encode([query])[0])
            retrieved = self.vector_store.search(query_vec, top_k=self.top_k)
        except Exception as e:  # noqa: BLE001
            timing.retrieval_ms = (time.perf_counter() - t0) * 1000
            return PipelineResponse(
                query=query, status="blocked", answer=None,
                reason=f"retrieval_error: {e}", retrieved=[], timing=timing,
            )
        timing.retrieval_ms = (time.perf_counter() - t0) * 1000

        chunks = [c for c, _ in retrieved]
        retrieved_out = [
            RetrievedChunk(text=c.text, score=s, doc_id=c.doc_id, metadata=c.metadata)
            for c, s in retrieved
        ]

        # --- pre-generation guardrails (unsafe input + off-topic) ---
        t0 = time.perf_counter()
        pre = run_pre_generation_guardrails(query, retrieved, self.offtopic_min_sim)
        timing.guardrails_ms += (time.perf_counter() - t0) * 1000
        if not pre.passed:
            return PipelineResponse(
                query=query, status="blocked", answer=None,
                reason=pre.reason, retrieved=retrieved_out, timing=timing,
            )

        # --- generation ---
        t0 = time.perf_counter()
        try:
            result: GenerationResult = _with_retries(lambda: self.generator.generate(query, chunks))
        except Exception as e:  # noqa: BLE001
            timing.generation_ms = (time.perf_counter() - t0) * 1000
            return PipelineResponse(
                query=query, status="blocked", answer=None,
                reason=f"generation_error: {e}", retrieved=retrieved_out, timing=timing,
            )
        timing.generation_ms = (time.perf_counter() - t0) * 1000

        if result.abstained:
            return PipelineResponse(
                query=query, status="abstained", answer=None,
                reason="model_abstained", retrieved=retrieved_out, timing=timing,
            )

        # --- post-generation guardrail (grounding / hallucination check) ---
        t0 = time.perf_counter()
        post = run_post_generation_guardrails(result.answer, chunks, self.grounding_min_overlap)
        timing.guardrails_ms += (time.perf_counter() - t0) * 1000
        if not post.passed:
            return PipelineResponse(
                query=query, status="abstained", answer=None,
                reason=post.reason, retrieved=retrieved_out, timing=timing,
            )

        return PipelineResponse(
            query=query, status="answered", answer=result.answer,
            reason=None, retrieved=retrieved_out, timing=timing,
        )
