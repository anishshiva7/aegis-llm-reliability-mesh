"""
Central configuration for the retrieval engine.

Values can be overridden with environment variables (handy for tuning the
chunking/embedding behaviour without touching code), all prefixed with
``AEGIS_`` — e.g. ``AEGIS_CHUNK_SIZE=400``. We keep this dependency-free
(plain stdlib + dataclass) so the config layer has no surprises.
"""

import os
from dataclasses import dataclass
from functools import lru_cache


def _env_str(key: str, default: str) -> str:
    return os.environ.get(f"AEGIS_{key}", default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(f"AEGIS_{key}")
    return int(raw) if raw is not None else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(f"AEGIS_{key}")
    return float(raw) if raw is not None else default


@dataclass(frozen=True)
class Settings:
    # ---- Embedding model -------------------------------------------------
    # all-MiniLM-L6-v2 is small (~80MB), fast on CPU, and produces 384-dim
    # vectors. A solid default for local development.
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # ---- Chunking --------------------------------------------------------
    # chunk_size / chunk_overlap are measured in *words* (whitespace tokens),
    # not characters or model tokens. This keeps the logic transparent and
    # framework-agnostic. Overlap preserves context across chunk boundaries.
    chunk_size: int = 200
    chunk_overlap: int = 40

    # ---- Search ----------------------------------------------------------
    default_top_k: int = 5

    # ---- Routing (Module 2) ----------------------------------------------
    # The router probes retrieval and routes on the top similarity score:
    #   score >= rag_score_threshold        -> RAG_ANSWER (strong match)
    #   score <  clarification_score_floor   -> too weak to ground on
    # These are cosine similarities in [-1, 1]; tune for your corpus.
    rag_score_threshold: float = 0.30
    clarification_score_floor: float = 0.10
    # Queries shorter than this (in words) that aren't greetings are treated
    # as underspecified and routed to NEEDS_CLARIFICATION.
    min_query_words: int = 2

    # ---- Logging ---------------------------------------------------------
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Build a Settings instance from the environment (cached per process)."""
    settings = Settings(
        embedding_model_name=_env_str("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
        chunk_size=_env_int("CHUNK_SIZE", 200),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 40),
        default_top_k=_env_int("DEFAULT_TOP_K", 5),
        rag_score_threshold=_env_float("RAG_SCORE_THRESHOLD", 0.30),
        clarification_score_floor=_env_float("CLARIFICATION_SCORE_FLOOR", 0.10),
        min_query_words=_env_int("MIN_QUERY_WORDS", 2),
        log_level=_env_str("LOG_LEVEL", "INFO"),
    )
    # Guard against a footgun: overlap >= chunk_size would never advance the
    # sliding window and would loop forever. Fail fast at startup instead.
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError(
            f"chunk_overlap ({settings.chunk_overlap}) must be smaller than "
            f"chunk_size ({settings.chunk_size})"
        )
    return settings
