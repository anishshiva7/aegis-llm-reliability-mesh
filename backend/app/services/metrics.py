"""
In-memory metrics collector (Module 6).

A tiny, thread-safe, process-local aggregator for operational visibility. No
database — this is intentionally ephemeral for a weekend MVP and resets on
restart. Swap the storage backend later without touching callers.

The pipeline records one RequestMetric per /ask; GET /metrics returns a snapshot.
"""

import threading
from dataclasses import dataclass
from typing import Dict, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RequestMetric:
    """One handled request's operational facts."""

    provider: str
    latency_ms: float
    overall_score: float
    fallback_used: bool = False
    degraded: bool = False
    retried: bool = False
    estimated_cost_usd: float = 0.0


class MetricsCollector:
    """Accumulates request metrics and emits a JSON-serializable snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_locked()

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()

    def _reset_locked(self) -> None:
        self._total = 0
        self._by_provider: Dict[str, int] = {}
        self._fallback = 0
        self._degraded = 0
        self._retried = 0
        self._latency_sum = 0.0
        self._score_sum = 0.0
        self._cost_sum = 0.0

    def record(self, m: RequestMetric) -> None:
        with self._lock:
            self._total += 1
            self._by_provider[m.provider] = self._by_provider.get(m.provider, 0) + 1
            self._fallback += int(m.fallback_used)
            self._degraded += int(m.degraded)
            self._retried += int(m.retried)
            self._latency_sum += m.latency_ms
            self._score_sum += m.overall_score
            self._cost_sum += m.estimated_cost_usd

    def snapshot(self) -> dict:
        with self._lock:
            total = self._total
            denom = total or 1  # avoid division by zero before any request
            return {
                "total_requests": total,
                "requests_by_provider": dict(self._by_provider),
                "fallback_count": self._fallback,
                "degraded_response_count": self._degraded,
                "average_latency_ms": round(self._latency_sum / denom, 2),
                "retry_rate": round(self._retried / denom, 4),
                "average_overall_score": round(self._score_sum / denom, 4),
                "estimated_cost_usd_total": round(self._cost_sum, 6),
            }


_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Return the process-wide MetricsCollector singleton."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
