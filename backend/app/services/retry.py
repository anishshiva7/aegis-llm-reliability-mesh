"""
Adaptive retry / self-healing loop (Module 4).

When an answer is flagged ``should_retry`` by the evaluator, the RetryManager
runs one or more *alternate strategies*, re-evaluates each attempt, and returns
the best-scoring one (best-of-N).  If every attempt still fails the quality
thresholds, it returns the best available answer flagged as ``degraded``.

Design goals:
  - Decoupled from RAGPipeline internals: the manager is handed an ``attempt_fn``
    callable, so it never imports the pipeline (no circular deps) and is trivial
    to unit-test with a fake attempt function.
  - Pluggable strategies: each RetryStrategy decides whether it applies and how
    to mutate the next attempt's parameters.  New policies drop in without
    touching the manager.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..config import Settings, get_settings
from ..logging_config import get_logger
from ..models.schemas import (
    AttemptTrace,
    EvaluationResult,
    GenerationTrace,
    RetrievedContext,
    RetryTrace,
    Route,
)
from .budget import BudgetGuard
from .router import RouteDecision

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data carriers shared with the pipeline
# ---------------------------------------------------------------------------
@dataclass
class AttemptParams:
    """Inputs for one pipeline attempt.  ``strategy`` is a human-readable label."""

    query: str
    top_k: Optional[int]
    force_route: Optional[Route]
    strategy: str


@dataclass
class AttemptResult:
    """Everything one attempt produced — answer plus its evaluation."""

    params: AttemptParams
    answer: str
    contexts: List[RetrievedContext]
    mode: str
    decision: RouteDecision
    evaluation: EvaluationResult
    # Set (Module 5) when the LLM provider raised and a degraded answer was used.
    generation_error: Optional[str] = None
    # Provider/cost observability for this attempt (Module 6). None for routes
    # that never call the generator (e.g. clarification) or legacy generators.
    generation: Optional[GenerationTrace] = None
    # Knowledge-graph retrieval result for this attempt (Module 10, hybrid mode).
    # Typed loosely (GraphSearchResult) to keep retry.py free of graph imports.
    graph_result: Optional[object] = None

    @property
    def score(self) -> float:
        return self.evaluation.scores.overall_score

    @property
    def confidence(self) -> float:
        return self.evaluation.scores.confidence


@dataclass
class RetryContext:
    """Read-only snapshot of the initial attempt that strategies reason about."""

    original_query: str
    effective_top_k: int
    initial_route: Route
    initial_top_score: Optional[float]
    index_size: int
    settings: Settings


@dataclass
class RetryOutcome:
    """Result of the retry loop: the winner, all attempts, and the trace."""

    best: AttemptResult
    attempts: List[AttemptResult]
    trace: RetryTrace


# ---------------------------------------------------------------------------
# Query expansion (deterministic — no LLM yet)
# ---------------------------------------------------------------------------
# Maps a keyword found in the query to a richer retrieval phrasing.  Deterministic
# so tests are stable; a real query-rewriter LLM can replace expand_query() later.
_EXPANSION_SYNONYMS = {
    "pricing": "pricing, subscription cost, fees, or payment structure",
    "price": "price, cost, fees, or payment structure",
    "cost": "cost, pricing, fees, or charges",
    "refund": "refund policy, returns, or money-back terms",
    "return": "return policy, refunds, or exchange terms",
    "cancel": "cancellation policy, termination, or account closure",
    "warranty": "warranty, guarantee, or coverage terms",
    "support": "customer support, help options, or contact channels",
    "security": "security, data protection, or privacy practices",
    "privacy": "privacy policy, data handling, or personal information use",
}


def expand_query(query: str) -> str:
    """
    Rewrite a vague/short query into a richer retrieval query (deterministic).

    Examples:
        "pricing" -> "What does the uploaded document say about pricing,
                      subscription cost, fees, or payment structure?"
        "refund"  -> "What does the uploaded document say about refund policy,
                      returns, or money-back terms?"
    """
    core = query.strip().rstrip("?.!").strip()
    lowered = core.lower()
    for keyword, expansion in _EXPANSION_SYNONYMS.items():
        if keyword in lowered:
            return f"What does the uploaded document say about {expansion}?"
    # Generic enrichment when no keyword matches: ask for specifics explicitly.
    return (
        f"What does the uploaded context say about {core}? "
        "Include relevant details, definitions, and specifics."
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
class RetryStrategy(ABC):
    """A pluggable alternate-attempt policy."""

    name: str = "abstract-strategy"

    @abstractmethod
    def applicable(self, ctx: RetryContext) -> bool:
        """Whether this strategy makes sense given the initial attempt."""
        raise NotImplementedError

    @abstractmethod
    def build(self, ctx: RetryContext) -> AttemptParams:
        """Produce the parameters for this strategy's attempt."""
        raise NotImplementedError


class ForceRagStrategy(RetryStrategy):
    """
    Strategy B — force grounding.

    If the first attempt answered directly (no grounding) but documents exist,
    force a RAG_ANSWER so the model is anchored to retrieved context.
    """

    name = "force_rag"

    def applicable(self, ctx: RetryContext) -> bool:
        return ctx.initial_route is Route.DIRECT_ANSWER and ctx.index_size > 0

    def build(self, ctx: RetryContext) -> AttemptParams:
        return AttemptParams(
            query=ctx.original_query,
            top_k=ctx.effective_top_k,
            force_route=Route.RAG_ANSWER,
            strategy=self.name,
        )


class QueryExpansionStrategy(RetryStrategy):
    """
    Strategy C — query expansion.

    Rewrite a vague/short query into a richer one to surface better passages.
    Let the router re-decide on the expanded query.
    """

    name = "query_expansion"

    def applicable(self, ctx: RetryContext) -> bool:
        return ctx.index_size > 0

    def build(self, ctx: RetryContext) -> AttemptParams:
        return AttemptParams(
            query=expand_query(ctx.original_query),
            top_k=ctx.effective_top_k,
            force_route=None,
            strategy=self.name,
        )


class IncreaseBreadthStrategy(RetryStrategy):
    """
    Strategy A — increase retrieval breadth.

    Widen top_k so more context is available, then let the router re-decide.
    """

    name = "increase_breadth"

    def applicable(self, ctx: RetryContext) -> bool:
        return ctx.index_size > 0

    def build(self, ctx: RetryContext) -> AttemptParams:
        return AttemptParams(
            query=ctx.original_query,
            top_k=ctx.effective_top_k + ctx.settings.retry_topk_increment,
            force_route=None,
            strategy=self.name,
        )


# Default order: highest-impact first, since the retry budget is small.
def _default_strategies() -> List[RetryStrategy]:
    return [ForceRagStrategy(), QueryExpansionStrategy(), IncreaseBreadthStrategy()]


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
AttemptFn = Callable[[AttemptParams], AttemptResult]

_EPS = 1e-9


class RetryManager:
    """
    Orchestrates the self-healing loop.

    Usage:
        manager = RetryManager()
        outcome = manager.run(initial_attempt, retry_context, attempt_fn)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        strategies: Optional[List[RetryStrategy]] = None,
        budget_guard: Optional[BudgetGuard] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.strategies = strategies if strategies is not None else _default_strategies()
        # Optional cost governor (Module 7). When None, retries are never blocked
        # on budget — preserving Module 4/5/6 behaviour and offline tests.
        self.budget_guard = budget_guard

    def run(
        self,
        initial: AttemptResult,
        ctx: RetryContext,
        attempt_fn: AttemptFn,
    ) -> RetryOutcome:
        attempts: List[AttemptResult] = [initial]
        strategies_used: List[str] = []
        best = initial

        candidates = [s for s in self.strategies if s.applicable(ctx)]
        budget = self.settings.max_retries
        logger.info(
            "RetryManager: initial_score=%.3f candidates=%s budget=%d",
            initial.score,
            [s.name for s in candidates],
            budget,
        )

        budget_blocked = False
        for strategy in candidates[:budget]:
            # Cost guardrail: stop before spending past the configured ceiling.
            if self.budget_guard is not None:
                accumulated = sum(
                    a.generation.estimated_cost_usd
                    for a in attempts
                    if a.generation is not None
                )
                blocked, reason = self.budget_guard.should_block_retry(
                    accumulated, len(attempts) - 1
                )
                if blocked:
                    budget_blocked = True
                    logger.warning("Budget guard blocked further retries: %s", reason)
                    break

            params = strategy.build(ctx)
            result = attempt_fn(params)
            attempts.append(result)
            strategies_used.append(strategy.name)
            logger.info(
                "Retry strategy=%s route=%s score=%.3f should_retry=%s",
                strategy.name,
                result.decision.route,
                result.score,
                result.evaluation.should_retry,
            )

            if self._is_better(result, best):
                best = result

            # Early stop: a satisfactory answer makes further retries pointless.
            if not result.evaluation.should_retry:
                logger.info("Retry early-stop: attempt cleared thresholds.")
                break

        trace = self._build_trace(attempts, best, strategies_used, budget_blocked)
        logger.info(
            "RetryManager done: best_attempt=%d best_score=%.3f degraded=%s budget_blocked=%s",
            trace.selected_best_attempt,
            best.score,
            trace.degraded_response,
            budget_blocked,
        )
        return RetryOutcome(best=best, attempts=attempts, trace=trace)

    # ----------------------------------------------------------------- helpers
    def _is_better(self, challenger: AttemptResult, incumbent: AttemptResult) -> bool:
        """
        Best-of-N comparison.

        A challenger displaces the incumbent only if it improves overall_score by
        at least ``min_score_improvement`` (keeps selection stable and prevents a
        marginally-different retry from flapping the result).  When scores are
        effectively tied, confidence breaks the tie — but never in favour of a
        strictly lower-scoring attempt.
        """
        gap = challenger.score - incumbent.score
        if gap >= self.settings.min_score_improvement:
            return True
        if abs(gap) <= _EPS:
            return challenger.confidence > incumbent.confidence + _EPS
        return False

    def _build_trace(
        self,
        attempts: List[AttemptResult],
        best: AttemptResult,
        strategies_used: List[str],
        budget_blocked: bool = False,
    ) -> RetryTrace:
        attempt_traces: List[AttemptTrace] = []
        best_index = 1
        for i, a in enumerate(attempts, start=1):
            if a is best:
                best_index = i
            attempt_traces.append(
                AttemptTrace(
                    attempt=i,
                    strategy=a.params.strategy,
                    route=a.decision.route,
                    overall_score=round(a.score, 3),
                    confidence=round(a.confidence, 3),
                    should_retry=a.evaluation.should_retry,
                )
            )

        return RetryTrace(
            attempts=attempt_traces,
            retry_count=len(attempts) - 1,
            selected_best_attempt=best_index,
            retry_strategies_used=strategies_used,
            score_progression=[round(a.score, 3) for a in attempts],
            # Circuit breaker: if the winner still fails thresholds (or the
            # budget guard halted us early), flag degraded.
            degraded_response=best.evaluation.should_retry or budget_blocked,
            budget_blocked=budget_blocked,
        )
