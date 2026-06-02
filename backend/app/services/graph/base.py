"""
GraphStore abstraction (Module 10 — Part A).

Every backend (Neo4j or the offline in-memory store) implements this interface.
The retriever, builder, and pipeline only ever talk to ``GraphStore`` — the same
seam discipline used for ``LLMProvider``: swapping Neo4j for the in-memory store
(or, later, a different graph DB) touches exactly one factory, not the callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional

from .models import GraphNode, GraphRelationship, LinkedChunk


class GraphStore(ABC):
    """Storage-agnostic knowledge-graph operations."""

    #: Stable identifier surfaced in stats/logs (e.g. "neo4j", "memory").
    backend: str = "abstract"

    # -- writes ------------------------------------------------------------
    @abstractmethod
    def add_node(self, node: GraphNode) -> None:
        """Create or merge a node keyed by ``node.name``."""

    @abstractmethod
    def add_relationship(self, rel: GraphRelationship) -> None:
        """Create or merge a directed edge. Endpoints must already exist."""

    @abstractmethod
    def link_chunk(self, chunk: LinkedChunk) -> None:
        """Attach a document chunk to the entities it mentions."""

    # -- reads -------------------------------------------------------------
    @abstractmethod
    def get_node(self, name: str) -> Optional[GraphNode]:
        """Return the node with this canonical name, if present."""

    @abstractmethod
    def find_nodes(self, names: Iterable[str]) -> List[GraphNode]:
        """Return the subset of ``names`` that exist as nodes."""

    @abstractmethod
    def traverse(
        self, seeds: Iterable[str], max_hops: int = 2
    ) -> tuple[List[GraphNode], List[GraphRelationship]]:
        """Breadth-first multi-hop expansion from ``seeds``.

        Returns ``(entities, relationships)`` reachable within ``max_hops``
        (including the seeds themselves). Order is deterministic.
        """

    @abstractmethod
    def chunks_for_entities(
        self, names: Iterable[str], limit: int = 5
    ) -> List[LinkedChunk]:
        """Return document chunks linked to any of ``names`` (deduplicated)."""

    @abstractmethod
    def stats(self) -> Dict[str, int]:
        """Return ``{"graph_nodes", "graph_relationships", "linked_chunks"}``."""

    # -- lifecycle ---------------------------------------------------------
    def clear(self) -> None:  # pragma: no cover - optional for tests
        """Remove all data. Optional; in-memory implements it for test isolation."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional
        """Release any backend resources (driver connections)."""
        return None
