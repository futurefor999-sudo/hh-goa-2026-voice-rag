"""
Central configuration. Loads from environment (.env if present).
Every other module reads its settings from here — no module should
call os.environ directly, so the whole pipeline stays configurable
from one place.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed — fine, we just rely on real env vars
    pass


def _get(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # STT
    stt_provider: str = field(default_factory=lambda: _get("STT_PROVIDER", "sarvam"))
    sarvam_api_key: str | None = field(default_factory=lambda: _get("SARVAM_API_KEY"))
    sarvam_stt_model: str = field(default_factory=lambda: _get("SARVAM_STT_MODEL", "saarika:v2"))

    # Generation
    gen_provider: str = field(default_factory=lambda: _get("GEN_PROVIDER", "mock"))
    anthropic_api_key: str | None = field(default_factory=lambda: _get("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _get("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    openai_api_key: str | None = field(default_factory=lambda: _get("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _get("OPENAI_MODEL", "gpt-4o-mini"))

    # Embeddings / retrieval
    embedding_provider: str = field(default_factory=lambda: _get("EMBEDDING_PROVIDER", "tfidf"))
    st_model: str = field(default_factory=lambda: _get("SENTENCE_TRANSFORMERS_MODEL", "intfloat/multilingual-e5-base"))
    top_k: int = field(default_factory=lambda: _get_int("TOP_K", 5))

    # Chunking
    chunk_strategy: str = field(default_factory=lambda: _get("CHUNK_STRATEGY", "hybrid"))
    fixed_chunk_size: int = field(default_factory=lambda: _get_int("FIXED_CHUNK_SIZE", 250))
    fixed_chunk_overlap: int = field(default_factory=lambda: _get_int("FIXED_CHUNK_OVERLAP", 50))

    # Guardrails
    grounding_min_overlap: float = field(default_factory=lambda: _get_float("GROUNDING_MIN_OVERLAP", 0.12))
    offtopic_min_sim: float = field(default_factory=lambda: _get_float("OFFTOPIC_MIN_SIM", 0.18))

    # Server
    host: str = field(default_factory=lambda: _get("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _get_int("PORT", 8000))


CONFIG = Config()
