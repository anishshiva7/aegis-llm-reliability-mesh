"""
Provider health tracking (Module 7).

A small, thread-safe registry of rolling per-provider health derived from recent
generate() outcomes. The fallback chain consults it to try the healthiest
providers first, so a flapping vendor is naturally de-prioritised without any
config change — while ``mock`` is always kept as the last resort.

Status is a simple, deterministic function of consecutive failures:
    consecutive_failures == 0      -> healthy
    1 <= consecutive_failures <= 2 -> degraded
    consecutive_failures >= 3      -> unhealthy

State is process-local and ephemeral (resets on restart) — matching the rest of
the weekend-MVP observability layer.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..logging_config import get_logger
from .cost import provider_family

logger = get_logger(__name__)

HEALTHY = "healthy"
DEGRADED = "degraded"
UNHEALTHY = "unhealthy"

_DEGRADED_AT = 1   # consecutive failures
_UNHEALTHY_AT = 3  # consecutive failures

# Sort rank for fallback ordering (lower = preferred).
_RANK = {HEALTHY: 0, DEGRADED: 1, UNHEALTHY: 2}


@dataclass
class _ProviderStat:
    consecutive_failures: int = 0
    total_successes: int = 0
    total_failures: int = 0
    last_success_at: Optional[float] = None
    last_failure_at: Optional[float] = None

    @property
    def status(self) -> str:
        if self.consecutive_failures >= _UNHEALTHY_AT:
            return UNHEALTHY
        if self.consecutive_failures >= _DEGRADED_AT:
            return DEGRADED
        return HEALTHY


class ProviderHealthRegistry:
    """Records provider outcomes and answers health/ordering questions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: Dict[str, _ProviderStat] = {}

    def _stat(self, name: str) -> _ProviderStat:
        st = self._stats.get(name)
        if st is None:
            st = _ProviderStat()
            self._stats[name] = st
        return st

    def record_success(self, name: str) -> None:
        with self._lock:
            st = self._stat(name)
            st.consecutive_failures = 0
            st.total_successes += 1
            st.last_success_at = time.time()

    def record_failure(self, name: str) -> None:
        with self._lock:
            st = self._stat(name)
            st.consecutive_failures += 1
            st.total_failures += 1
            st.last_failure_at = time.time()
            logger.warning(
                "Provider '%s' failure recorded (consecutive=%d, status=%s).",
                name, st.consecutive_failures, st.status,
            )

    def status(self, name: str) -> str:
        """Health of a provider; unknown providers are assumed healthy."""
        with self._lock:
            st = self._stats.get(name)
            return st.status if st is not None else HEALTHY

    def order(self, names: List[str]) -> List[str]:
        """
        Return ``names`` sorted healthiest-first (stable), with any ``mock``
        provider forced last so it stays the last-resort fallback.
        """
        def key(name: str):
            is_mock = provider_family(name) == "mock"
            return (1 if is_mock else 0, _RANK[self.status(name)])

        # Stable sort preserves the configured order within an equal rank.
        return sorted(names, key=key)

    def snapshot(self) -> List[dict]:
        with self._lock:
            out = []
            for name, st in self._stats.items():
                out.append({
                    "provider": name,
                    "health_status": st.status,
                    "consecutive_failures": st.consecutive_failures,
                    "total_successes": st.total_successes,
                    "total_failures": st.total_failures,
                    "last_success_at": st.last_success_at,
                    "last_failure_at": st.last_failure_at,
                })
            return out

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()


_registry: Optional[ProviderHealthRegistry] = None


def get_health_registry() -> ProviderHealthRegistry:
    """Return the process-wide ProviderHealthRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = ProviderHealthRegistry()
    return _registry
