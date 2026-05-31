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
from ..services.metrics_store import get_metrics_store

logger = get_logger(__name__)

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def metrics() -> dict:
    """Return a rich snapshot of persisted request metrics."""
    return get_metrics_store().snapshot()


@router.get("/metrics/prometheus")
def metrics_prometheus() -> Response:
    """Return Prometheus text-exposition metrics (text/plain)."""
    text = get_metrics_store().prometheus_text()
    # 0.0.4 is the Prometheus text exposition format version.
    return Response(content=text, media_type="text/plain; version=0.0.4")
