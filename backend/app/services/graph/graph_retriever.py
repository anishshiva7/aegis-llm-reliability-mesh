"""
GraphRetriever (Module 10 — Part D).

The read-side counterpart to the builder. Given a natural-language query it:

    1. extracts the entities the query refers to (same matcher as ingestion),
    2. matches them to nodes in the graph,
    3. traverses 1–``max_hops`` hops to gather the neighbourhood,
    4. collects document chunks linked to those entities,
    5. scores how confidently the graph had something relevant to say,

and returns a ``GraphSearchResult`` — the internal mirror of the trace the
dashboard renders.

It depends only on the ``GraphStore`` interface, so it works identically against
Neo4j and the in-memory store.
"""

from __future__ import annotations

import time
from typing import List, Optional

from ...logging_config import get_logger
from .base import GraphStore
from .extractor import DeterministicEntityExtractor, EntityExtractor
from .models import GraphSearchResult

logger = get_logger(__name__)


class GraphRetriever:
    """Entity-anchored, multi-hop retrieval over a ``GraphStore``."""

    def __init__(
        self,
        store: GraphStore,
        extractor: Optional[EntityExtractor] = None,
        max_hops: int = 2,
        chunk_limit: int = 5,
    ) -> None:
        self.store = store
        self.extractor = extractor or DeterministicEntityExtractor()
        self.max_hops = max_hops
        self.chunk_limit = chunk_limit
        self._last_latency_ms: float = 0.0

    def retrieve(
        self,
        query: str,
        max_hops: Optional[int] = None,
        chunk_limit: Optional[int] = None,
    ) -> GraphSearchResult:
        """Run graph retrieval for ``query``; always returns a result object."""
        start = time.perf_counter()
        hops = max_hops if max_hops is not None else self.max_hops
        limit = chunk_limit if chunk_limit is not None else self.chunk_limit

        # 1) Which ontology entities does the query mention?
        query_entities = self.extractor.match_query_entities(query)
        # Only seeds that actually exist as nodes anchor a traversal.
        seed_names = [
            n.name for n in query_entities if self.store.get_node(n.name) is not None
        ]
        matched = self.store.find_nodes(seed_names)

        if not seed_names:
            self._last_latency_ms = (time.perf_counter() - start) * 1000.0
            return GraphSearchResult(hops=0)

        # 2–3) Traverse the neighbourhood.
        traversed_entities, traversed_rels = self.store.traverse(
            seed_names, max_hops=hops
        )

        # 4) Linked document chunks for the matched (anchor) entities.
        graph_chunks = self.store.chunks_for_entities(seed_names, limit=limit)

        # 5) Confidence the graph was useful.
        score = _graph_score(
            matched_count=len(matched),
            relationship_count=len(traversed_rels),
            chunk_count=len(graph_chunks),
        )

        self._last_latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "Graph retrieve: query=%r -> %d matched, %d entities, %d rels, "
            "%d chunks, score=%.3f (%.1fms)",
            query,
            len(matched),
            len(traversed_entities),
            len(traversed_rels),
            len(graph_chunks),
            score,
            self._last_latency_ms,
        )
        return GraphSearchResult(
            matched_entities=matched,
            traversed_entities=traversed_entities,
            traversed_relationships=traversed_rels,
            graph_chunks=graph_chunks,
            graph_score=score,
            hops=hops,
        )

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms


def _graph_score(
    matched_count: int, relationship_count: int, chunk_count: int
) -> float:
    """Bounded 0..1 confidence the graph had relevant structure to offer.

    Intuition: matching at least one entity is the price of entry; richer
    neighbourhoods (more edges) and supporting chunks raise confidence with
    diminishing returns. Tuned to stay deterministic and explainable.
    """
    if matched_count == 0:
        return 0.0
    # Anchor confidence for matching entities (saturates quickly).
    anchor = min(1.0, 0.4 + 0.2 * matched_count)
    # Relationship richness: a connected neighbourhood is more useful.
    rel_bonus = min(0.4, 0.05 * relationship_count)
    # Supporting document chunks add grounding.
    chunk_bonus = min(0.2, 0.05 * chunk_count)
    return round(min(1.0, anchor * 0.6 + rel_bonus + chunk_bonus), 3)
