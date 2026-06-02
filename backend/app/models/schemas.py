"""
Pydantic request/response models — the public contract of the API.

These models give us automatic validation, type coercion, and OpenAPI docs
(visible at /docs) for free. Keeping them in one place makes the API surface
easy to reason about.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
class IngestTextRequest(BaseModel):
    """Body for ingesting a raw text blob."""

    text: str = Field(..., min_length=1, description="Raw text to ingest.")
    source: Optional[str] = Field(
        default=None,
        description="Optional human-readable label for where this text came "
        "from (e.g. a filename or URL). Stored as metadata.",
    )
    # Per-request overrides for chunking. If omitted, the server defaults
    # (from config) are used. Lets callers tune granularity ad-hoc.
    chunk_size: Optional[int] = Field(default=None, ge=1)
    chunk_overlap: Optional[int] = Field(default=None, ge=0)


class IngestResponse(BaseModel):
    """Result of an ingest operation."""

    source: str = Field(..., description="The source label that was stored.")
    chunks_created: int = Field(..., description="How many chunks this text produced.")
    total_chunks_in_index: int = Field(
        ..., description="Total chunks now stored across all ingests."
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    """Body for a similarity search."""

    query: str = Field(..., min_length=1, description="Natural-language query.")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="How many chunks to return. Defaults to server config.",
    )


class SearchResult(BaseModel):
    """A single retrieved chunk plus its relevance score and provenance."""

    chunk_id: int = Field(..., description="Internal id of the chunk (FAISS row).")
    text: str = Field(..., description="The chunk text.")
    score: float = Field(
        ...,
        description="Cosine similarity in [-1, 1]; higher means more relevant.",
    )
    source: str = Field(..., description="Source label this chunk came from.")
    chunk_index: int = Field(
        ..., description="Position of this chunk within its source document."
    )


class SearchResponse(BaseModel):
    """Ranked search results for a query."""

    query: str
    results: list[SearchResult]


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
class StatsResponse(BaseModel):
    """Lightweight view of engine state — useful for debugging/tests."""

    total_chunks: int
    embedding_model: str
    embedding_dim: int


# ---------------------------------------------------------------------------
# Ask / RAG (Module 2)
# ---------------------------------------------------------------------------
class Route(str, Enum):
    """The three handling strategies the query router can choose."""

    DIRECT_ANSWER = "DIRECT_ANSWER"
    RAG_ANSWER = "RAG_ANSWER"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


class AskRequest(BaseModel):
    """Body for the unified /ask endpoint."""

    query: str = Field(..., min_length=1, description="The user's question.")
    top_k: Optional[int] = Field(
        default=None, ge=1, description="Chunks to retrieve for RAG. Defaults to config."
    )
    force_route: Optional[Route] = Field(
        default=None,
        description="Skip the router and force a specific route (debugging/evals).",
    )
    include_trace: bool = Field(
        default=True,
        description="Include routing/retrieval trace metadata in the response.",
    )


class RetrievedContext(BaseModel):
    """A single chunk used (or considered) as grounding context."""

    chunk_id: int
    text: str
    score: float = Field(..., description="Cosine similarity to the query, in [-1, 1].")
    source: str
    chunk_index: int


# ---------------------------------------------------------------------------
# Evaluation / Judge (Module 3)
# ---------------------------------------------------------------------------
class EvaluationScores(BaseModel):
    """Raw dimension scores produced by the judge, each in [0.0, 1.0]."""

    relevance: float = Field(..., ge=0.0, le=1.0, description="How relevant the answer is to the query.")
    groundedness: float = Field(..., ge=0.0, le=1.0, description="Degree to which the answer is grounded in retrieved context.")
    completeness: float = Field(..., ge=0.0, le=1.0, description="Whether the answer fully addresses the query.")
    hallucination_risk: float = Field(..., ge=0.0, le=1.0, description="Estimated probability the answer contains fabricated content.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Judge's overall confidence in answer quality.")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Weighted composite of all dimensions.")


class EvaluationResult(BaseModel):
    """Full output of the evaluation pass — scores plus retry recommendation."""

    scores: EvaluationScores
    should_retry: bool = Field(
        ...,
        description="True when quality is below acceptable thresholds (handled in Module 4).",
    )
    evaluation_reason: str = Field(
        ..., description="Human-readable explanation of the evaluation outcome."
    )


# ---------------------------------------------------------------------------
# Adaptive retry / self-healing (Module 4)
# ---------------------------------------------------------------------------
class AttemptTrace(BaseModel):
    """A single attempt in the retry loop — the initial try or one retry."""

    attempt: int = Field(..., ge=1, description="1-based attempt number.")
    strategy: str = Field(
        ..., description="Strategy that produced this attempt (e.g. 'initial', 'force_rag')."
    )
    route: Route = Field(..., description="Route taken on this attempt.")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Evaluation overall score.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Evaluation confidence.")
    should_retry: bool = Field(..., description="Whether this attempt was itself flagged for retry.")


class RetryTrace(BaseModel):
    """Observability for the self-healing loop — what was tried and what won."""

    attempts: list[AttemptTrace] = Field(
        ..., description="All attempts in order; index 0 is the initial try."
    )
    retry_count: int = Field(..., ge=0, description="Number of retries executed (excludes initial).")
    selected_best_attempt: int = Field(
        ..., ge=1, description="1-based attempt number that was returned as the answer."
    )
    retry_strategies_used: list[str] = Field(
        default_factory=list, description="Strategy names attempted, in order."
    )
    score_progression: list[float] = Field(
        default_factory=list, description="overall_score per attempt, in order."
    )
    degraded_response: bool = Field(
        default=False,
        description="True when the best answer still fails quality thresholds after all retries.",
    )
    budget_blocked: bool = Field(
        default=False,
        description="True when the cost/rate guardrail (Module 7) halted further "
        "retries before the budget was exceeded. Implies degraded_response.",
    )


class GenerationTrace(BaseModel):
    """
    Per-generation observability (Module 6).

    Captures *which* provider answered and *what it cost*, without leaking any
    vendor specifics into the pipeline. Token counts and cost are clearly
    labelled estimates (chars/4 heuristic + coarse per-family pricing) — useful
    for relative comparison and budget signals, not billing-grade accuracy.
    """

    provider_name: str = Field(..., description="Provider that produced the answer (e.g. 'bedrock:anthropic.claude-...').")
    model_name: str = Field(..., description="Model id behind that provider.")
    provider_latency_ms: float = Field(..., description="Time spent inside the provider call, in ms.")
    fallback_used: bool = Field(
        default=False,
        description="True when the primary provider failed and a fallback answered.",
    )
    fallback_chain: list[str] = Field(
        default_factory=list,
        description="Configured provider chain, in priority order.",
    )
    estimated_input_tokens: int = Field(..., ge=0, description="Prompt tokens — exact when from the provider, else chars/4 estimate.")
    estimated_output_tokens: int = Field(..., ge=0, description="Completion tokens — exact when from the provider, else chars/4 estimate.")
    estimated_cost_usd: float = Field(..., ge=0.0, description="USD cost for this generation (estimate when tokens are estimated).")
    token_usage_source: str = Field(
        default="estimated",
        description="Where token counts came from: 'provider' (exact, from the "
        "vendor response) or 'estimated' (chars/4 heuristic fallback).",
    )


# ---------------------------------------------------------------------------
# GraphRAG / hybrid retrieval (Module 10)
# ---------------------------------------------------------------------------
class RetrievalMode(str, Enum):
    """How context was gathered for this request."""

    VECTOR = "vector"  # FAISS semantic similarity only
    GRAPH = "graph"    # knowledge-graph traversal only
    HYBRID = "hybrid"  # FAISS + graph fused


class GraphEntityModel(BaseModel):
    """A single knowledge-graph entity surfaced in the trace."""

    name: str = Field(..., description="Canonical entity name (the MERGE key).")
    category: str = Field(..., description="Entity category, e.g. Component, Route, Provider.")
    description: str = Field(default="", description="Short human-readable description.")


class GraphRelationshipModel(BaseModel):
    """A directed, typed edge between two entities: (source)-[type]->(target)."""

    source: str
    type: str
    target: str


class GraphTrace(BaseModel):
    """Graph-retrieval observability — what the knowledge graph contributed."""

    graph_backend: str = Field(..., description="Backend that served the graph: neo4j | memory.")
    matched_entities: list[GraphEntityModel] = Field(
        default_factory=list,
        description="Entities the query referred to that anchor the traversal.",
    )
    traversed_entities: list[GraphEntityModel] = Field(
        default_factory=list,
        description="Entities reached within the hop budget (includes the anchors).",
    )
    traversed_relationships: list[GraphRelationshipModel] = Field(
        default_factory=list,
        description="Edges connecting the traversed neighbourhood.",
    )
    graph_chunks: list[RetrievedContext] = Field(
        default_factory=list,
        description="Document chunks linked to the matched entities.",
    )
    graph_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="0..1 confidence the graph had relevant structure to offer.",
    )
    hops: int = Field(default=0, ge=0, description="Traversal depth used.")
    graph_latency_ms: float = Field(default=0.0, description="Time spent in graph retrieval, in ms.")


class GraphSearchRequest(BaseModel):
    """Body for the standalone /graph/search endpoint (Module 10).

    Lets you inspect knowledge-graph retrieval in isolation — no LLM call, no
    vector store — to see exactly which entities/relationships a query anchors.
    """

    query: str = Field(..., min_length=1, description="Natural-language query.")
    max_hops: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Traversal depth. Defaults to server config (graph_max_hops).",
    )
    chunk_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Max document chunks to link back. Defaults to server config.",
    )


class GraphSearchResponse(BaseModel):
    """Result of a direct graph traversal — the GraphTrace, plus convenience flags."""

    query: str
    graph_used: bool = Field(
        ...,
        description="True when the query matched at least one entity and the graph "
        "had relevant structure to traverse.",
    )
    graph: GraphTrace = Field(..., description="Full graph-retrieval observability payload.")


class RouteTrace(BaseModel):
    """Observability payload — how and why the answer was produced."""

    route: Route = Field(..., description="The route that was selected.")
    reason: str = Field(..., description="Human-readable explanation of the choice.")
    retrieval_used: bool = Field(
        ...,
        description="Whether the vector store was touched at all during this "
        "request. Retained for backward compatibility; prefer the more precise "
        "retrieval_probe_used / answer_context_used below.",
    )
    retrieval_probe_used: bool = Field(
        default=False,
        description="True when the router searched the vector store to *decide* "
        "the route (a routing probe). May be true even for DIRECT_ANSWER.",
    )
    answer_context_used: bool = Field(
        default=False,
        description="True when retrieved chunks were actually injected into the "
        "answer prompt (grounded generation). False for direct/clarification "
        "answers even if a routing probe ran.",
    )
    generation_mode: str = Field(
        ..., description="grounded | direct | clarification."
    )
    retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.VECTOR,
        description="How context was gathered: vector | graph | hybrid (Module 10).",
    )
    graph_used: bool = Field(
        default=False,
        description="True when the knowledge graph contributed entities or chunks "
        "to this answer (Module 10).",
    )
    latency_ms: float = Field(..., description="End-to-end handling time in ms.")
    top_score: Optional[float] = Field(
        default=None, description="Best retrieval score seen (if retrieval ran)."
    )
    retrieved: list[RetrievedContext] = Field(
        default_factory=list, description="Chunks retrieved for grounding."
    )
    evaluation: Optional[EvaluationResult] = Field(
        default=None, description="LLM-as-a-Judge quality evaluation (Module 3+)."
    )
    retry: Optional[RetryTrace] = Field(
        default=None,
        description="Self-healing retry trace (Module 4); present whenever the "
        "loop engaged (evaluation flagged should_retry). retry_count may be 0 if "
        "no strategy was applicable, in which case degraded_response is set.",
    )
    generation_error: Optional[str] = Field(
        default=None,
        description="Set when the LLM provider failed and a degraded answer was "
        "returned (Module 5). None on the normal path.",
    )
    generation: Optional[GenerationTrace] = Field(
        default=None,
        description="Provider/cost observability for the winning attempt (Module 6). "
        "Absent on pure-clarification routes or when the generator predates the "
        "metadata interface.",
    )
    graph: Optional[GraphTrace] = Field(
        default=None,
        description="Knowledge-graph retrieval trace (Module 10). Present when "
        "retrieval_mode is graph or hybrid.",
    )


class AskResponse(BaseModel):
    """Response from /ask: the answer plus optional trace."""

    query: str
    route: Route
    answer: str
    trace: Optional[RouteTrace] = None


# ---------------------------------------------------------------------------
# Provider health (Module 7)
# ---------------------------------------------------------------------------
class ProviderHealth(BaseModel):
    """Rolling health snapshot for a single provider."""

    provider: str
    health_status: str = Field(..., description="healthy | degraded | unhealthy.")
    consecutive_failures: int = Field(..., ge=0)
    total_successes: int = Field(..., ge=0)
    total_failures: int = Field(..., ge=0)
    last_success_at: Optional[float] = Field(
        default=None, description="Unix timestamp of the last success, if any."
    )
    last_failure_at: Optional[float] = Field(
        default=None, description="Unix timestamp of the last failure, if any."
    )


class ProvidersHealthResponse(BaseModel):
    """All known providers' health, plus the health-aware fallback order."""

    providers: list[ProviderHealth]
    recommended_order: list[str] = Field(
        default_factory=list,
        description="Provider names sorted healthiest-first (mock kept last).",
    )
