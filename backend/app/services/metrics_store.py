"""
Persistent metrics store (Module 7).

A lightweight SQLite-backed sink for per-request operational facts. Unlike the
in-memory ``MetricsCollector`` (Module 6, kept for backward compatibility), this
store survives restarts and retains *raw rows*, which lets us compute
percentiles and histograms on demand — the building blocks of a production-grade
observability snapshot and a Prometheus exposition.

Design notes:
  - One process-wide connection (``check_same_thread=False``) guarded by a lock.
    FastAPI's threadpool is low-concurrency for a weekend MVP; this is plenty.
  - ``":memory:"`` is supported for tests (single connection => shared DB).
  - Aggregation is pure-Python and deterministic (nearest-rank percentiles,
    fixed histogram buckets) so tests are stable and dependency-free.
"""

import math
import sqlite3
import threading
import time
from typing import Dict, List, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS request_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    provider_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    route TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    overall_score REAL NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    fallback_used INTEGER NOT NULL DEFAULT 0,
    degraded_response INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0.0
);
"""

# Upper bounds (exclusive) for latency histogram buckets, in ms.
_LATENCY_BUCKETS_MS = [50, 100, 250, 500, 1000, 2000]
# Score histogram is fixed 0.2-wide bins over [0, 1].
_SCORE_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]


def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Nearest-rank percentile (deterministic). ``sorted_vals`` must be sorted."""
    if not sorted_vals:
        return 0.0
    rank = math.ceil(pct / 100.0 * len(sorted_vals))
    idx = min(max(rank, 1), len(sorted_vals)) - 1
    return round(sorted_vals[idx], 2)


def _latency_histogram(values: List[float]) -> Dict[str, int]:
    labels = []
    prev = 0
    for b in _LATENCY_BUCKETS_MS:
        labels.append(f"{prev}-{b}ms")
        prev = b
    labels.append(f"{prev}ms+")
    hist = {label: 0 for label in labels}
    for v in values:
        placed = False
        prev = 0
        for i, b in enumerate(_LATENCY_BUCKETS_MS):
            if v < b:
                hist[labels[i]] += 1
                placed = True
                break
            prev = b
        if not placed:
            hist[labels[-1]] += 1
    return hist


def _score_histogram(values: List[float]) -> Dict[str, int]:
    labels = [f"{lo:.1f}-{hi:.1f}" for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]]
    hist = {label: 0 for label in labels}
    for v in values:
        for i, (lo, hi) in enumerate(_SCORE_BINS):
            if lo <= v < hi:
                hist[labels[i]] += 1
                break
    return hist


class MetricsStore:
    """SQLite-backed persistent metrics with rich, deterministic aggregation."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        # check_same_thread=False: one shared connection across the threadpool.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("MetricsStore ready (db=%s).", db_path)

    def record(
        self,
        *,
        provider_name: str,
        model_name: str,
        route: str,
        latency_ms: float,
        overall_score: float,
        retry_count: int = 0,
        fallback_used: bool = False,
        degraded_response: bool = False,
        estimated_cost_usd: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO request_metrics (timestamp, provider_name, model_name, "
                "route, latency_ms, overall_score, retry_count, fallback_used, "
                "degraded_response, estimated_cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    timestamp if timestamp is not None else time.time(),
                    provider_name,
                    model_name,
                    route,
                    float(latency_ms),
                    float(overall_score),
                    int(retry_count),
                    int(bool(fallback_used)),
                    int(bool(degraded_response)),
                    float(estimated_cost_usd),
                ),
            )
            self._conn.commit()

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM request_metrics")
            self._conn.commit()

    def _rows(self) -> List[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute("SELECT * FROM request_metrics"))

    def snapshot(self) -> dict:
        """Full observability snapshot — superset of the Module 6 fields."""
        rows = self._rows()
        total = len(rows)
        denom = total or 1

        latencies = sorted(r["latency_ms"] for r in rows)
        scores = [r["overall_score"] for r in rows]
        costs = [r["estimated_cost_usd"] for r in rows]
        fallback = sum(1 for r in rows if r["fallback_used"])
        degraded = sum(1 for r in rows if r["degraded_response"])
        retried = sum(1 for r in rows if r["retry_count"] > 0)

        by_provider: Dict[str, int] = {}
        by_route: Dict[str, int] = {}
        for r in rows:
            by_provider[r["provider_name"]] = by_provider.get(r["provider_name"], 0) + 1
            by_route[r["route"]] = by_route.get(r["route"], 0) + 1

        return {
            "total_requests": total,
            "requests_by_provider": by_provider,
            "requests_by_route": by_route,
            "fallback_count": fallback,
            "degraded_response_count": degraded,
            "average_latency_ms": round(sum(latencies) / denom, 2),
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "p99_latency_ms": _percentile(latencies, 99),
            "average_overall_score": round(sum(scores) / denom, 4),
            "score_histogram": _score_histogram(scores),
            "latency_histogram": _latency_histogram(latencies),
            "retry_rate": round(retried / denom, 4),
            "fallback_rate": round(fallback / denom, 4),
            "degraded_response_rate": round(degraded / denom, 4),
            "cost_total": round(sum(costs), 6),
            "estimated_cost_usd_total": round(sum(costs), 6),
        }

    def prometheus_text(self) -> str:
        """Render a Prometheus text-exposition (no client library required)."""
        s = self.snapshot()
        lines: List[str] = []

        def metric(name: str, value, help_text: str, mtype: str = "gauge") -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {mtype}")
            lines.append(f"{name} {value}")

        metric("aegis_total_requests", s["total_requests"], "Total requests handled.", "counter")
        metric("aegis_fallback_count", s["fallback_count"], "Requests served by a fallback provider.", "counter")
        metric("aegis_degraded_response_count", s["degraded_response_count"], "Requests returned in a degraded state.", "counter")
        metric("aegis_average_latency_ms", s["average_latency_ms"], "Mean end-to-end latency in ms.")
        metric("aegis_p95_latency_ms", s["p95_latency_ms"], "95th percentile latency in ms.")
        metric("aegis_estimated_cost_usd_total", s["estimated_cost_usd_total"], "Cumulative estimated cost in USD.", "counter")
        metric("aegis_retry_rate", s["retry_rate"], "Fraction of requests that triggered a retry.")

        # Bonus labelled series — handy on a dashboard, still dependency-free.
        lines.append("# HELP aegis_requests_by_provider Requests per provider.")
        lines.append("# TYPE aegis_requests_by_provider counter")
        for provider, count in s["requests_by_provider"].items():
            safe = provider.replace('"', '\\"')
            lines.append(f'aegis_requests_by_provider{{provider="{safe}"}} {count}')

        return "\n".join(lines) + "\n"


_store: Optional[MetricsStore] = None


def get_metrics_store() -> MetricsStore:
    """Return the process-wide MetricsStore singleton (path from settings)."""
    global _store
    if _store is None:
        from ..config import get_settings

        _store = MetricsStore(get_settings().metrics_db_path)
    return _store
