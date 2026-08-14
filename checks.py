"""
Guardrails.

Four checks run around the model, in order, each able to short-circuit
the pipeline before the (comparatively expensive) generation call:

  1. unsafe_input_check   — regex/keyword screen for clearly unsafe or
                             injection-style input, before we do anything else
  2. offtopic_check       — if best retrieval similarity is below a floor,
                             the query is probably outside the corpus's
                             domain; abstain instead of forcing an answer
  3. (generation happens in pipeline.py, using the ABSTAIN-aware prompt)
  4. grounding_check      — after generation, verify the answer's content
                             actually overlaps with the retrieved context.
                             This is the hallucination check: an LLM can
                             ignore "answer only from context" instructions,
                             so we verify post-hoc with a cheap lexical
                             overlap heuristic rather than trusting the
                             prompt alone.

These are heuristic and offline by design (no extra model call needed),
which matters for the pipeline's latency budget. They're intentionally
conservative — false "abstain" is a much cheaper mistake for a demo than
a confidently wrong answer.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

from src.chunking.strategies import Chunk, split_sentences

# Deliberately narrow: this is a lexical last-resort screen for obviously
# unsafe or prompt-injection-shaped input, not a substitute for a real
# moderation endpoint. Swap in a hosted moderation API for production.
_UNSAFE_PATTERNS = [
    r"\bignore (all|the) (previous|prior|above) instructions\b",
    r"\byou are now\b.{0,40}\bDAN\b",
    r"\bsystem prompt\b.{0,20}\b(reveal|leak|print|show)\b",
    r"\bhow (do|can) i (make|build|synthesi[sz]e)\b.{0,30}\b(bomb|explosive|nerve agent|pathogen)\b",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)


@dataclass
class GuardrailResult:
    passed: bool
    reason: str | None = None


def unsafe_input_check(query: str) -> GuardrailResult:
    if _UNSAFE_RE.search(query):
        return GuardrailResult(passed=False, reason="unsafe_or_injection_input")
    if len(query.strip()) == 0:
        return GuardrailResult(passed=False, reason="empty_input")
    return GuardrailResult(passed=True)


def offtopic_check(retrieved: list[tuple[Chunk, float]], min_sim: float) -> GuardrailResult:
    if not retrieved:
        return GuardrailResult(passed=False, reason="no_retrieval_results")
    best_score = max(score for _, score in retrieved)
    if best_score < min_sim:
        return GuardrailResult(passed=False, reason=f"offtopic_low_similarity({best_score:.3f}<{min_sim})")
    return GuardrailResult(passed=True)


def _lexical_overlap(answer: str, context: str) -> float:
    a_words = set(w.lower() for w in re.findall(r"[a-zA-Z0-9]+", answer) if len(w) > 2)
    c_words = set(w.lower() for w in re.findall(r"[a-zA-Z0-9]+", context) if len(w) > 2)
    if not a_words:
        return 0.0
    return len(a_words & c_words) / len(a_words)


def grounding_check(answer: str, retrieved_chunks: list[Chunk], min_overlap: float) -> GuardrailResult:
    """Post-hoc hallucination check: what fraction of the answer's
    content words also appear somewhere in the retrieved context?
    Low overlap => the model likely drew on outside knowledge instead
    of the retrieved passages, even if it didn't say ABSTAIN."""
    if not answer.strip():
        return GuardrailResult(passed=False, reason="empty_answer")

    context = " ".join(c.text for c in retrieved_chunks)
    overlap = _lexical_overlap(answer, context)
    if overlap < min_overlap:
        return GuardrailResult(passed=False, reason=f"low_grounding_overlap({overlap:.3f}<{min_overlap})")
    return GuardrailResult(passed=True)


def run_pre_generation_guardrails(query: str, retrieved, min_sim: float) -> GuardrailResult:
    r = unsafe_input_check(query)
    if not r.passed:
        return r
    return offtopic_check(retrieved, min_sim)


def run_post_generation_guardrails(answer: str, retrieved_chunks: list[Chunk], min_overlap: float) -> GuardrailResult:
    return grounding_check(answer, retrieved_chunks, min_overlap)
