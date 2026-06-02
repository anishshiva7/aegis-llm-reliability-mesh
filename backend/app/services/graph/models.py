"""
Domain models for the knowledge graph (Module 10 — GraphRAG).

These are plain dataclasses — the storage-agnostic vocabulary every GraphStore
backend (Neo4j or in-memory) speaks. Pydantic trace models that cross the API
boundary live in ``app/models/schemas.py``; keeping the internal graph types
here means the storage layer has no web-framework dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class EntityCategory(str, Enum):
    """The node ontology Aegis reasons over.

    Deliberately small and domain-specific: these are the *kinds* of things a
    reliability-mesh architecture is made of, so graph answers stay grounded in
    Aegis's real components instead of arbitrary noun phrases.
    """

    COMPONENT = "Component"
    ROUTE = "Route"
    PROVIDER = "Provider"
    RETRIEVAL = "Retrieval"
    EVALUATION = "Evaluation"
    RETRY = "Retry"
    METRIC = "Metric"
    DASHBOARD = "Dashboard"
    DOCUMENT = "Document"


@dataclass(frozen=True)
class GraphNode:
    """A single entity in the knowledge graph.

    ``name`` is the canonical, unique identifier (we MERGE on it). ``aliases``
    are the surface forms used to detect the entity in free text / queries.
    """

    name: str
    category: EntityCategory
    description: str = ""
    aliases: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class GraphRelationship:
    """A directed, typed edge: ``(source)-[type]->(target)``."""

    source: str
    type: str
    target: str


@dataclass(frozen=True)
class LinkedChunk:
    """A document chunk linked to one or more entities it mentions.

    This is the bridge between the graph and the corpus: traversing to an
    entity surfaces the passages that actually talk about it.
    """

    text: str
    source: str
    chunk_index: int
    entities: tuple = field(default_factory=tuple)


@dataclass
class GraphSearchResult:
    """Outcome of a graph retrieval (Part D).

    Mirrors the trace the dashboard renders, but in internal types. ``graph_score``
    is a 0..1 confidence that the graph had something relevant to say.
    """

    matched_entities: List[GraphNode] = field(default_factory=list)
    traversed_entities: List[GraphNode] = field(default_factory=list)
    traversed_relationships: List[GraphRelationship] = field(default_factory=list)
    graph_chunks: List[LinkedChunk] = field(default_factory=list)
    graph_score: float = 0.0
    hops: int = 0

    @property
    def used(self) -> bool:
        """True when the graph contributed anything (entities or chunks)."""
        return bool(self.matched_entities or self.graph_chunks)

    def relationship_summary(self, limit: int = 12) -> str:
        """Human/LLM-readable one-line-per-edge summary of the neighbourhood."""
        lines = [
            f"({r.source}) -[{r.type}]-> ({r.target})"
            for r in self.traversed_relationships[:limit]
        ]
        return "\n".join(lines)
