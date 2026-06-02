"""
/metrics endpoints — operational observability (Modules 6–7).

``GET /metrics`` returns a rich JSON snapshot computed from the persistent
SQLite store (counts, percentiles, histograms, rates, cost). ``GET
/metrics/prometheus`` renders the same data as Prometheus text exposition with
no client-library dependency. Both read through the MetricsStore interface, so
the storage backend can be swapped without touching callers.
"""

from fastapi import APIRouter, Response

from ..logging_config import get_logger
from ..services.graph import get_graph_metrics
from ..services.metrics_store import get_metrics_store

logger = get_logger(__name__)

router = APIRouter(tags=["metrics"])


def _graph_prometheus_text() -> str:
    """Render graph metrics (Module 10) as Prometheus text exposition."""
    g = get_graph_metrics().snapshot()
    lines = []

    def metric(name: str, value, help_text: str, mtype: str = "gauge") -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")
        lines.append(f"{name} {value}")

    metric("aegis_graph_nodes", g["graph_nodes"], "Entities in the knowledge graph.")
    metric("aegis_graph_relationships", g["graph_relationships"], "Edges in the knowledge graph.")
    metric("aegis_graph_linked_chunks", g["linked_chunks"], "Document chunks linked to entities.")
    metric("aegis_graph_traversals", g["graph_traversals"], "Graph retrievals performed.", "counter")
    metric("aegis_hybrid_queries", g["hybrid_queries"], "Queries answered with hybrid retrieval.", "counter")
    metric("aegis_graph_latency_ms", g["graph_latency_ms"], "Mean graph-retrieval latency in ms.")
    return "\n".join(lines) + "\n"


@router.get("/metrics")
def metrics() -> dict:
    """Return a rich snapshot of persisted request metrics + graph metrics."""
    snapshot = get_metrics_store().snapshot()
    # Merge in Module 10 graph/hybrid telemetry under a dedicated key.
    snapshot["graph"] = get_graph_metrics().snapshot()
    return snapshot


@router.get("/metrics/prometheus")
def metrics_prometheus() -> Response:
    """Return Prometheus text-exposition metrics (text/plain)."""
    text = get_metrics_store().prometheus_text() + _graph_prometheus_text()
    # 0.0.4 is the Prometheus text exposition format version.
    return Response(content=text, media_type="text/plain; version=0.0.4")
