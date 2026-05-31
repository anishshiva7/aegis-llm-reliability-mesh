"""
Budget / rate guardrail (Module 7).

A small, deterministic cost governor for the self-healing loop. Retries improve
quality but cost money; this guard stops the RetryManager from spending past a
configured ceiling. When it trips, the request is *not* crashed — the best
answer so far is returned, flagged ``degraded_response`` and ``budget_blocked``.

Three independent limits (any one trips the guard; ``<= 0`` disables a limit, so
the default config is a no-op and existing offline tests are unaffected):

  - AEGIS_MAX_REQUEST_COST_USD     — cap on estimated cost accrued *this request*
  - AEGIS_MAX_DAILY_COST_USD       — cap on estimated cost accrued *today*
  - AEGIS_MAX_RETRIES_ON_COST_LIMIT — hard cap on retries once any cost is incurred

The daily total is process-local and rolls over at UTC midnight. It is ephemeral
(resets on restart) — consistent with the weekend-MVP observability posture.
"""

import threading
import time
from typing import Optional, Tuple

from ..config import Settings, get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


def _utc_day(ts: float) -> int:
    """Whole-day bucket (UTC) for daily-rollover bookkeeping."""
    return int(ts // 86400)


class BudgetGuard:
    """Decides whether another (costly) retry is allowed under the budget."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        s = settings or get_settings()
        self.max_request_cost_usd = s.max_request_cost_usd
        self.max_daily_cost_usd = s.max_daily_cost_usd
        self.max_retries_on_cost_limit = s.max_retries_on_cost_limit
        self._lock = threading.Lock()
        self._day = _utc_day(time.time())
        self._daily_cost = 0.0

    # ------------------------------------------------------------------ daily
    def add_daily_cost(self, cost_usd: float) -> None:
        """Record realised cost toward today's running total (rolls at midnight)."""
        if cost_usd <= 0:
            return
        with self._lock:
            today = _utc_day(time.time())
            if today != self._day:
                self._day = today
                self._daily_cost = 0.0
            self._daily_cost += cost_usd

    @property
    def daily_cost_usd(self) -> float:
        with self._lock:
            return round(self._daily_cost, 6)

    def reset_daily(self) -> None:
        with self._lock:
            self._daily_cost = 0.0
            self._day = _utc_day(time.time())

    # ------------------------------------------------------------------ gate
    def should_block_retry(
        self, accumulated_request_cost_usd: float, retries_done: int
    ) -> Tuple[bool, str]:
        """
        Return ``(blocked, reason)`` for the *next* retry.

        ``accumulated_request_cost_usd`` is the estimated cost spent on this
        request so far; ``retries_done`` is how many retries already ran.
        """
        if self.max_request_cost_usd > 0 and (
            accumulated_request_cost_usd >= self.max_request_cost_usd
        ):
            return True, (
                f"request cost ${accumulated_request_cost_usd:.6f} >= limit "
                f"${self.max_request_cost_usd:.6f}"
            )

        if self.max_daily_cost_usd > 0:
            projected = self.daily_cost_usd + accumulated_request_cost_usd
            if projected >= self.max_daily_cost_usd:
                return True, (
                    f"daily cost ${projected:.6f} >= limit "
                    f"${self.max_daily_cost_usd:.6f}"
                )

        if (
            self.max_retries_on_cost_limit > 0
            and accumulated_request_cost_usd > 0
            and retries_done >= self.max_retries_on_cost_limit
        ):
            return True, (
                f"retries_done {retries_done} >= cost-limited cap "
                f"{self.max_retries_on_cost_limit}"
            )

        return False, ""


_guard: Optional[BudgetGuard] = None


def get_budget_guard() -> BudgetGuard:
    """Return the process-wide BudgetGuard singleton."""
    global _guard
    if _guard is None:
        _guard = BudgetGuard()
    return _guard
