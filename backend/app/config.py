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


def _env_raw(key: str, default: str) -> str:
    """Read an unprefixed env var (e.g. NEO4J_URI), falling back to AEGIS_<key>.

    Neo4j tooling conventionally uses bare ``NEO4J_*`` names, so we honour those
    first while still allowing the ``AEGIS_``-prefixed override for consistency.
    """
    return os.environ.get(key) or os.environ.get(f"AEGIS_{key}", default)


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

    # ---- Evaluation / Judge (Module 3) -----------------------------------
    # Thresholds that drive the should_retry flag computed by the evaluator.
    # An answer triggers a retry recommendation when ANY of these conditions holds:
    #   overall_score      < retry_threshold
    #   hallucination_risk > hallucination_threshold
    #   confidence         < confidence_floor
    retry_threshold: float = 0.60
    hallucination_threshold: float = 0.70
    confidence_floor: float = 0.40

    # ---- Adaptive retry / self-healing (Module 4) ------------------------
    # When an answer is flagged should_retry, the RetryManager runs up to
    # ``max_retries`` alternate strategies and keeps the best-scoring attempt.
    #   max_retries           — hard cap on alternate attempts per query
    #   retry_topk_increment  — how much to widen top_k in the breadth strategy
    #   min_score_improvement — a retry must beat the incumbent by at least this
    #                           much (overall_score) to displace it; keeps
    #                           selection stable and avoids flapping
    max_retries: int = 2
    retry_topk_increment: int = 2
    min_score_improvement: float = 0.05

    # ---- LLM provider / real generation (Module 5) -----------------------
    # The generation backend behind the RAGPipeline. ``mock`` is the offline
    # deterministic default (no network, no keys) used by tests and demos.
    #   provider          — mock | openai | anthropic | bedrock
    #   fallback_provider — provider to try if the primary raises (or "none").
    #                        Keeps the system answering when a vendor is down.
    #   model_name        — generation model; empty string => provider default.
    #   temperature       — sampling temperature for real providers.
    #   request_timeout    — per-request timeout (seconds) for real providers.
    #   max_tokens        — max output tokens (Anthropic requires this).
    # API keys are read from the environment and never logged.
    provider: str = "mock"
    fallback_provider: str = "mock"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    model_name: str = ""
    temperature: float = 0.0
    request_timeout: float = 30.0
    max_tokens: int = 1024

    # ---- AWS Bedrock (Module 6) ------------------------------------------
    # Used only when provider/fallback is "bedrock". Credentials are resolved
    # by boto3's standard chain (env vars, shared config, IAM role) — never
    # passed through here. ``bedrock_model_id`` empty => provider default.
    #   aws_region        — AWS region hosting Bedrock (e.g. "us-east-1").
    #   bedrock_model_id  — Bedrock model id (e.g. an Anthropic Claude id).
    aws_region: str = "us-east-1"
    bedrock_model_id: str = ""

    # ---- Observability / Ops (Module 7) ----------------------------------
    # metrics_db_path        — SQLite file for persistent request metrics. The
    #                          special value ":memory:" keeps it ephemeral.
    # Budget guardrail (<=0 disables a limit, so defaults are a no-op):
    #   max_request_cost_usd      — cap on estimated cost per single request
    #   max_daily_cost_usd        — cap on estimated cost accrued per UTC day
    #   max_retries_on_cost_limit — hard retry cap once any cost is incurred
    metrics_db_path: str = "aegis_metrics.db"
    max_request_cost_usd: float = 0.0
    max_daily_cost_usd: float = 0.0
    max_retries_on_cost_limit: int = 0

    # ---- GraphRAG / Knowledge graph (Module 10) --------------------------
    # graph_backend     — neo4j | memory. "neo4j" tries a real Neo4j instance
    #                     and gracefully falls back to the in-memory store if it
    #                     is unreachable; "memory" (default) is fully offline.
    # NEO4J_* credentials are used only when graph_backend == "neo4j". The bolt
    # URI/user/password follow Neo4j conventions; the password is never logged.
    graph_backend: str = "memory"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "neo4j"
    neo4j_database: str = "neo4j"
    # Default multi-hop traversal depth for graph retrieval (1–3 is sensible).
    graph_max_hops: int = 2

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
        retry_threshold=_env_float("RETRY_THRESHOLD", 0.60),
        hallucination_threshold=_env_float("HALLUCINATION_THRESHOLD", 0.70),
        confidence_floor=_env_float("CONFIDENCE_FLOOR", 0.40),
        max_retries=_env_int("MAX_RETRIES", 2),
        retry_topk_increment=_env_int("RETRY_TOPK_INCREMENT", 2),
        min_score_improvement=_env_float("MIN_SCORE_IMPROVEMENT", 0.05),
        provider=_env_str("PROVIDER", "mock"),
        fallback_provider=_env_str("FALLBACK_PROVIDER", "mock"),
        openai_api_key=_env_str("OPENAI_API_KEY", ""),
        anthropic_api_key=_env_str("ANTHROPIC_API_KEY", ""),
        model_name=_env_str("MODEL_NAME", ""),
        temperature=_env_float("TEMPERATURE", 0.0),
        request_timeout=_env_float("REQUEST_TIMEOUT", 30.0),
        max_tokens=_env_int("MAX_TOKENS", 1024),
        aws_region=_env_str("AWS_REGION", "us-east-1"),
        bedrock_model_id=_env_str("BEDROCK_MODEL_ID", ""),
        metrics_db_path=_env_str("METRICS_DB", "aegis_metrics.db"),
        max_request_cost_usd=_env_float("MAX_REQUEST_COST_USD", 0.0),
        max_daily_cost_usd=_env_float("MAX_DAILY_COST_USD", 0.0),
        max_retries_on_cost_limit=_env_int("MAX_RETRIES_ON_COST_LIMIT", 0),
        graph_backend=_env_str("GRAPH_BACKEND", "memory"),
        neo4j_uri=_env_raw("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=_env_raw("NEO4J_USERNAME", "neo4j"),
        neo4j_password=_env_raw("NEO4J_PASSWORD", "neo4j"),
        neo4j_database=_env_raw("NEO4J_DATABASE", "neo4j"),
        graph_max_hops=_env_int("GRAPH_MAX_HOPS", 2),
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
