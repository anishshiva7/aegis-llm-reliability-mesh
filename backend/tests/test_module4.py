"""
Tests for Module 4 — Adaptive retry / self-healing loop.

Two layers, both deterministic:

  1. RetryManager unit tests drive a *scripted* attempt_fn that returns
     hand-built AttemptResults. This gives exact control over best-of-N
     selection, the retry-count cap, the circuit breaker, and the
     min_score_improvement tie-break — no engine/judge noise.

  2. Pipeline integration tests use a ScriptedEngine whose retrieval score
     depends on the query/route, proving the force-RAG and query-expansion
     strategies actually raise the evaluated score end-to-end.

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module4.py -v
    ./venv/bin/python tests/test_module4.py
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.models.schemas import (  # noqa: E402
    EvaluationResult,
    EvaluationScores,
    Route,
)
from app.services.generator import MockLLM  # noqa: E402
from app.services.rag import RAGPipeline  # noqa: E402
from app.services.retry import (  # noqa: E402
    AttemptParams,
    AttemptResult,
    ForceRagStrategy,
    QueryExpansionStrategy,
    RetryContext,
    RetryManager,
    expand_query,
)
from app.services.router import RouteDecision  # noqa: E402
from app.services.vector_store import ChunkRecord  # noqa: E402


# ---------------------------------------------------------------------------
# Builders for RetryManager unit tests
# ---------------------------------------------------------------------------

def _eval(overall: float, confidence: float = 0.5, should_retry: bool = False) -> EvaluationResult:
    scores = EvaluationScores(
        relevance=overall,
        groundedness=overall,
        completeness=overall,
        hallucination_risk=round(1.0 - overall, 3),
        confidence=confidence,
        overall_score=overall,
    )
    return EvaluationResult(scores=scores, should_retry=should_retry, evaluation_reason="test")


def _attempt(
    strategy: str,
    overall: float,
    *,
    route: Route = Route.RAG_ANSWER,
    confidence: float = 0.5,
    should_retry: bool = False,
) -> AttemptResult:
    params = AttemptParams(query="q", top_k=3, force_route=None, strategy=strategy)
    decision = RouteDecision(route, "reason", retrieval_used=True, top_score=overall, hits=[])
    return AttemptResult(
        params=params,
        answer=f"answer-{strategy}",
        contexts=[],
        mode="grounded",
        decision=decision,
        evaluation=_eval(overall, confidence, should_retry),
    )


def _ctx(initial_route: Route = Route.DIRECT_ANSWER) -> RetryContext:
    """RetryContext where all three default strategies are applicable."""
    return RetryContext(
        original_query="q",
        effective_top_k=3,
        initial_route=initial_route,
        initial_top_score=0.1,
        index_size=10,
        settings=get_settings(),
    )


def _scripted_fn(table: dict):
    """attempt_fn that returns a scripted AttemptResult keyed by strategy name."""

    def fn(params: AttemptParams) -> AttemptResult:
        return table[params.strategy]

    return fn


# ---------------------------------------------------------------------------
# RetryManager — best-of-N, counts, circuit breaker, hysteresis
# ---------------------------------------------------------------------------

def test_best_of_n_returns_highest_not_last():
    """The winner is the highest-scoring attempt, never blindly the final one."""
    initial = _attempt("initial", 0.40, route=Route.DIRECT_ANSWER, should_retry=True)
    table = {
        # force_rag is the best; query_expansion runs after but scores lower.
        "force_rag": _attempt("force_rag", 0.82, should_retry=True),
        "query_expansion": _attempt("query_expansion", 0.55, should_retry=True),
    }
    outcome = RetryManager().run(initial, _ctx(), _scripted_fn(table))

    assert outcome.best.params.strategy == "force_rag"
    assert outcome.trace.selected_best_attempt == 2  # attempt #2, not the last (#3)
    assert outcome.trace.score_progression == [0.40, 0.82, 0.55]


def test_retry_count_respects_max_retries():
    """No more than max_retries alternate attempts are executed."""
    settings = replace(get_settings(), max_retries=1)
    manager = RetryManager(settings=settings)
    initial = _attempt("initial", 0.40, route=Route.DIRECT_ANSWER, should_retry=True)
    table = {
        "force_rag": _attempt("force_rag", 0.50, should_retry=True),
        "query_expansion": _attempt("query_expansion", 0.60, should_retry=True),
        "increase_breadth": _attempt("increase_breadth", 0.70, should_retry=True),
    }
    outcome = manager.run(initial, _ctx(), _scripted_fn(table))

    assert outcome.trace.retry_count == 1
    assert len(outcome.trace.attempts) == 2  # initial + 1 retry


def test_early_stop_when_attempt_clears_thresholds():
    """A satisfactory retry stops the loop early — remaining budget is unused."""
    initial = _attempt("initial", 0.40, route=Route.DIRECT_ANSWER, should_retry=True)
    table = {
        "force_rag": _attempt("force_rag", 0.88, should_retry=False),  # clears
        "query_expansion": _attempt("query_expansion", 0.99, should_retry=False),
    }
    outcome = RetryManager().run(initial, _ctx(), _scripted_fn(table))

    assert outcome.trace.retry_count == 1  # stopped after force_rag, never tried expansion
    assert outcome.trace.degraded_response is False
    assert outcome.best.params.strategy == "force_rag"


def test_circuit_breaker_flags_degraded_when_all_fail():
    """When every attempt still fails thresholds, return the best but mark degraded."""
    initial = _attempt("initial", 0.30, route=Route.DIRECT_ANSWER, should_retry=True)
    table = {
        "force_rag": _attempt("force_rag", 0.45, should_retry=True),
        "query_expansion": _attempt("query_expansion", 0.38, should_retry=True),
    }
    outcome = RetryManager().run(initial, _ctx(), _scripted_fn(table))

    assert outcome.trace.degraded_response is True
    assert outcome.best.params.strategy == "force_rag"  # still the highest of the failures
    assert outcome.best.evaluation.should_retry is True  # signal preserved


def test_min_score_improvement_hysteresis_keeps_incumbent():
    """A marginal gain below min_score_improvement must not displace the incumbent."""
    settings = get_settings()
    # Single strategy so we test one challenger cleanly.
    manager = RetryManager(settings=settings, strategies=[QueryExpansionStrategy()])
    initial = _attempt("initial", 0.50, route=Route.RAG_ANSWER, should_retry=True)
    gain = settings.min_score_improvement / 2.0
    table = {"query_expansion": _attempt("query_expansion", 0.50 + gain, should_retry=True)}
    outcome = manager.run(initial, _ctx(initial_route=Route.RAG_ANSWER), _scripted_fn(table))

    assert outcome.best is initial
    assert outcome.trace.selected_best_attempt == 1


def test_confidence_breaks_ties_at_equal_score():
    """At equal overall_score, the higher-confidence attempt wins."""
    settings = get_settings()
    manager = RetryManager(settings=settings, strategies=[QueryExpansionStrategy()])
    initial = _attempt("initial", 0.50, confidence=0.40, should_retry=True)
    table = {"query_expansion": _attempt("query_expansion", 0.50, confidence=0.90, should_retry=True)}
    outcome = manager.run(initial, _ctx(initial_route=Route.RAG_ANSWER), _scripted_fn(table))

    assert outcome.best.params.strategy == "query_expansion"


def test_trace_integrity():
    """Trace fields stay internally consistent: counts, lengths, numbering."""
    initial = _attempt("initial", 0.40, route=Route.DIRECT_ANSWER, should_retry=True)
    table = {
        "force_rag": _attempt("force_rag", 0.55, should_retry=True),
        "query_expansion": _attempt("query_expansion", 0.60, should_retry=True),
    }
    outcome = RetryManager().run(initial, _ctx(), _scripted_fn(table))
    trace = outcome.trace

    assert len(trace.attempts) == trace.retry_count + 1
    assert len(trace.score_progression) == len(trace.attempts)
    assert [a.attempt for a in trace.attempts] == list(range(1, len(trace.attempts) + 1))
    assert 1 <= trace.selected_best_attempt <= len(trace.attempts)
    assert trace.attempts[0].strategy == "initial"
    assert trace.retry_strategies_used == ["force_rag", "query_expansion"]


# ---------------------------------------------------------------------------
# Query expansion — deterministic rewriting
# ---------------------------------------------------------------------------

def test_query_expansion_enriches_keywords():
    expanded = expand_query("pricing")
    assert "uploaded document" in expanded.lower()
    assert "subscription cost" in expanded.lower()

    refund = expand_query("refund")
    assert "refund policy" in refund.lower()


def test_query_expansion_generic_fallback():
    expanded = expand_query("widgets")
    assert "widgets" in expanded.lower()
    assert expanded != "widgets"  # actually enriched


# ---------------------------------------------------------------------------
# Pipeline integration — strategies improve the evaluated answer end-to-end
# ---------------------------------------------------------------------------

class ScriptedEngine:
    """RetrievalEngine stand-in whose top score is computed from the query."""

    def __init__(self, total_chunks: int, score_fn):
        self._total = total_chunks
        self._score_fn = score_fn

    def search(self, query, top_k=None):
        score = self._score_fn(query)
        if score is None:
            return []
        record = ChunkRecord(
            text="A detailed and relevant passage with enough content to ground a complete answer.",
            source="doc.txt",
            chunk_index=0,
        )
        return [(0, score, record)]

    @property
    def total_chunks(self):
        return self._total


def test_force_rag_strategy_targets_grounding():
    """Strategy B: applies when the initial route was DIRECT and forces RAG grounding."""
    strat = ForceRagStrategy()

    direct_ctx = _ctx(initial_route=Route.DIRECT_ANSWER)  # index_size=10
    assert strat.applicable(direct_ctx) is True
    params = strat.build(direct_ctx)
    assert params.force_route is Route.RAG_ANSWER

    # Not applicable once the initial attempt was already grounded.
    assert strat.applicable(_ctx(initial_route=Route.RAG_ANSWER)) is False
    # Not applicable with an empty index (nothing to ground on).
    empty = RetryContext(
        original_query="q",
        effective_top_k=3,
        initial_route=Route.DIRECT_ANSWER,
        initial_top_score=None,
        index_size=0,
        settings=get_settings(),
    )
    assert strat.applicable(empty) is False


def test_query_expansion_strategy_improves_score():
    """Vague query retrieves nothing useful until expansion adds richer terms."""

    def score_fn(query: str):
        # The bare query grounds weakly (routes RAG but below the retry
        # threshold); only the expanded phrasing ("...uploaded document...")
        # matches strongly. (Module 8: weak grounding is expressed as a passing
        # RAG score, since a DIRECT answer is no longer penalized for a weak
        # probe and so would not trigger the retry loop at all.)
        return 0.9 if "uploaded document" in query.lower() else 0.32

    engine = ScriptedEngine(total_chunks=5, score_fn=score_fn)
    pipe = RAGPipeline(engine=engine, generator=MockLLM())

    resp = pipe.ask("pricing info")

    assert resp.trace.retry is not None
    retry = resp.trace.retry
    assert "query_expansion" in retry.retry_strategies_used
    best = retry.attempts[retry.selected_best_attempt - 1]
    assert best.strategy == "query_expansion"
    assert best.overall_score > retry.score_progression[0]


def test_retry_triggered_on_weak_answer():
    """A weakly-grounded RAG answer engages the self-healing loop (retry sub-trace present)."""
    # Weak-but-passing retrieval score (>= rag_score_threshold) -> router picks
    # RAG_ANSWER, but the thin grounding keeps overall below the retry threshold,
    # so the self-healing loop engages. (Module 8: a DIRECT answer is no longer
    # penalized for low probe scores, so weak grounding must be expressed on the
    # RAG path to exercise retry.)
    engine = ScriptedEngine(total_chunks=5, score_fn=lambda q: 0.32)
    pipe = RAGPipeline(engine=engine, generator=MockLLM())

    resp = pipe.ask("Describe the overall approach in the document")

    assert resp.trace.route is Route.RAG_ANSWER
    assert resp.trace.retry is not None
    assert resp.trace.retry.retry_count >= 1


def test_force_route_disables_self_healing():
    """An explicit force_route is a manual override — the retry loop must not run."""
    # Strong chunk exists, but caller pins DIRECT for eval purposes.
    engine = ScriptedEngine(total_chunks=5, score_fn=lambda q: 0.9)
    pipe = RAGPipeline(engine=engine, generator=MockLLM())

    resp = pipe.ask("Explain the subject in detail", force_route=Route.DIRECT_ANSWER)

    assert resp.route is Route.DIRECT_ANSWER  # not healed into RAG
    assert resp.trace.retry is None
    assert resp.trace.retrieval_used is False


def test_circuit_breaker_end_to_end_marks_degraded():
    """When no strategy can clear thresholds, the response is degraded but still returned."""
    # Every query/route yields the same weak-but-passing RAG score: routes
    # RAG_ANSWER, but the thin grounding never clears the retry threshold and no
    # strategy can improve it (the score is constant), so the circuit breaker
    # returns a degraded response. (Module 8: weak grounding is now expressed on
    # the RAG path, since direct answers are no longer penalized.)
    engine = ScriptedEngine(total_chunks=5, score_fn=lambda q: 0.32)
    pipe = RAGPipeline(engine=engine, generator=MockLLM())

    resp = pipe.ask("Tell me about the configuration options in the document")

    assert resp.trace.route is Route.RAG_ANSWER
    assert resp.trace.retry is not None
    assert resp.trace.retry.degraded_response is True
    assert resp.trace.evaluation.should_retry is True  # degradation signal preserved
    assert resp.answer  # an answer is still returned


def test_strong_initial_answer_skips_retry():
    """A strong initial answer must NOT trigger the retry loop (retry trace is None)."""
    engine = ScriptedEngine(total_chunks=5, score_fn=lambda q: 0.92)
    pipe = RAGPipeline(engine=engine, generator=MockLLM())

    resp = pipe.ask("What does the document say about the topic?")

    assert resp.route is Route.RAG_ANSWER
    assert resp.trace.evaluation.should_retry is False
    assert resp.trace.retry is None, "no retries should run for a strong initial answer"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failures = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
