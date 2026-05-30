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


class RouteTrace(BaseModel):
    """Observability payload — how and why the answer was produced."""

    route: Route = Field(..., description="The route that was selected.")
    reason: str = Field(..., description="Human-readable explanation of the choice.")
    retrieval_used: bool = Field(..., description="Whether the vector store was queried.")
    generation_mode: str = Field(
        ..., description="grounded | direct | clarification."
    )
    latency_ms: float = Field(..., description="End-to-end handling time in ms.")
    top_score: Optional[float] = Field(
        default=None, description="Best retrieval score seen (if retrieval ran)."
    )
    retrieved: list[RetrievedContext] = Field(
        default_factory=list, description="Chunks retrieved for grounding."
    )


class AskResponse(BaseModel):
    """Response from /ask: the answer plus optional trace."""

    query: str
    route: Route
    answer: str
    trace: Optional[RouteTrace] = None
