"""
GraphMetrics (Module 10 — Part I).

A small, thread-safe, process-wide collector for graph-retrieval operational
facts. Deliberately separate from the SQLite ``MetricsStore`` (which is keyed to
per-request LLM/route rows): graph metrics are cheap counters + a latency
running-average that the ``/metrics`` snapshot and Prometheus exposition merge
in. Keeping it in-memory mirrors the Module 6 ``MetricsCollector`` style and
keeps the hot path dependency-free.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional


class GraphMetrics:
    """In-memory counters for graph traversal + hybrid retrieval activity."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._graph_traversals = 0
        self._hybrid_queries = 0
        self._latency_total_ms = 0.0
        self._latency_samples = 0
        # Snapshot of the live store size, refreshed on record/seed.
        self._graph_nodes = 0
        self._graph_relationships = 0
        self._linked_chunks = 0

    def record_store_stats(self, stats: Dict[str, int]) -> None:
        """Capture the current graph size (nodes/relationships/linked chunks)."""
        with self._lock:
            self._graph_nodes = int(stats.get("graph_nodes", 0))
            self._graph_relationships = int(stats.get("graph_relationships", 0))
            self._linked_chunks = int(stats.get("linked_chunks", 0))

    def record_traversal(self, latency_ms: float, *, hybrid: bool = False) -> None:
        """Record one graph retrieval (and whether it was part of a hybrid query)."""
        with self._lock:
            self._graph_traversals += 1
            if hybrid:
                self._hybrid_queries += 1
            self._latency_total_ms += float(latency_ms)
            self._latency_samples += 1

    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            if self._latency_samples == 0:
                return 0.0
            return round(self._latency_total_ms / self._latency_samples, 2)

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            avg = (
                round(self._latency_total_ms / self._latency_samples, 2)
                if self._latency_samples
                else 0.0
            )
            return {
                "graph_nodes": self._graph_nodes,
                "graph_relationships": self._graph_relationships,
                "linked_chunks": self._linked_chunks,
                "graph_traversals": self._graph_traversals,
                "hybrid_queries": self._hybrid_queries,
                "graph_latency_ms": avg,
            }

    def reset(self) -> None:
        with self._lock:
            self._graph_traversals = 0
            self._hybrid_queries = 0
            self._latency_total_ms = 0.0
            self._latency_samples = 0


_metrics: Optional[GraphMetrics] = None


def get_graph_metrics() -> GraphMetrics:
    """Return the process-wide GraphMetrics singleton."""
    global _metrics
    if _metrics is None:
        _metrics = GraphMetrics()
    return _metrics
