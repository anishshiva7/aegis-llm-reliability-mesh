"""
Tests for Module 7 — frontend/backend API contract.

The Next.js dashboard (frontend/lib/types.ts) consumes specific fields from
``/ask``, ``/metrics``, ``/metrics/prometheus`` and ``/providers/health``. These
tests pin the *shape* of those payloads so a backend rename can't silently break
the dashboard. They are fully offline: response models are constructed directly
and the metrics store uses an in-memory SQLite db — no embedder, network, or app
startup required.

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module7_frontend_contract.py -v
    ./venv/bin/python tests/test_module7_frontend_contract.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import (  # noqa: E402
    AskResponse,
    AttemptTrace,
    EvaluationResult,
    EvaluationScores,
    GenerationTrace,
    ProviderHealth,
    ProvidersHealthResponse,
    RetrievedContext,
    RetryTrace,
    Route,
    RouteTrace,
)
from app.services.metrics_store import MetricsStore  # noqa: E402


def _full_ask_response() -> AskResponse:
    """A maximal /ask payload exercising every panel the dashboard renders."""
    scores = EvaluationScores(
        relevance=0.9,
        groundedness=0.85,
        completeness=0.8,
        hallucination_risk=0.1,
        confidence=0.88,
        overall_score=0.86,
    )
    evaluation = EvaluationResult(
        scores=scores, should_retry=False, evaluation_reason="Grounded and complete."
    )
    retry = RetryTrace(
        attempts=[
            AttemptTrace(
                attempt=1,
                strategy="initial",
                route=Route.RAG_ANSWER,
                overall_score=0.6,
                confidence=0.7,
                should_retry=True,
            ),
            AttemptTrace(
                attempt=2,
                strategy="query_expansion",
                route=Route.RAG_ANSWER,
                overall_score=0.86,
                confidence=0.88,
                should_retry=False,
            ),
        ],
        retry_count=1,
        selected_best_attempt=2,
        retry_strategies_used=["query_expansion"],
        score_progression=[0.6, 0.86],
        degraded_response=False,
        budget_blocked=False,
    )
    generation = GenerationTrace(
        provider_name="openai:gpt-4o",
        model_name="gpt-4o",
        provider_latency_ms=123.4,
        fallback_used=False,
        fallback_chain=["openai:gpt-4o", "mock"],
        estimated_input_tokens=42,
        estimated_output_tokens=88,
        estimated_cost_usd=0.0012,
        token_usage_source="provider",
    )
    trace = RouteTrace(
        route=Route.RAG_ANSWER,
        reason="Strong retrieval match.",
        retrieval_used=True,
        generation_mode="grounded",
        latency_ms=130.0,
        top_score=0.71,
        retrieved=[
            RetrievedContext(
                chunk_id=1,
                text="Aegis is a reliability mesh.",
                score=0.71,
                source="aegis-overview",
                chunk_index=0,
            )
        ],
        evaluation=evaluation,
        retry=retry,
        generation_error=None,
        generation=generation,
    )
    return AskResponse(
        query="What is Aegis?",
        route=Route.RAG_ANSWER,
        answer="Aegis is a self-optimizing LLM reliability mesh.",
        trace=trace,
    )


def _require_keys(obj: dict, keys: set, where: str) -> None:
    missing = keys - set(obj.keys())
    assert not missing, f"{where} missing keys consumed by frontend: {sorted(missing)}"


def test_ask_response_contract() -> None:
    """AskResponse + nested traces expose every field frontend/lib/types.ts reads."""
    d = _full_ask_response().model_dump()
    _require_keys(d, {"query", "route", "answer", "trace"}, "AskResponse")

    trace = d["trace"]
    _require_keys(
        trace,
        {
            "route",
            "reason",
            "retrieval_used",
            "generation_mode",
            "latency_ms",
            "top_score",
            "retrieved",
            "evaluation",
            "retry",
            "generation_error",
            "generation",
        },
        "RouteTrace",
    )

    _require_keys(
        trace["retrieved"][0],
        {"chunk_id", "text", "score", "source", "chunk_index"},
        "RetrievedContext",
    )

    _require_keys(
        trace["evaluation"],
        {"scores", "should_retry", "evaluation_reason"},
        "EvaluationResult",
    )
    _require_keys(
        trace["evaluation"]["scores"],
        {
            "relevance",
            "groundedness",
            "completeness",
            "hallucination_risk",
            "confidence",
            "overall_score",
        },
        "EvaluationScores",
    )

    _require_keys(
        trace["retry"],
        {
            "attempts",
            "retry_count",
            "selected_best_attempt",
            "retry_strategies_used",
            "score_progression",
            "degraded_response",
            "budget_blocked",
        },
        "RetryTrace",
    )
    _require_keys(
        trace["retry"]["attempts"][0],
        {"attempt", "strategy", "route", "overall_score", "confidence", "should_retry"},
        "AttemptTrace",
    )

    _require_keys(
        trace["generation"],
        {
            "provider_name",
            "model_name",
            "provider_latency_ms",
            "fallback_used",
            "fallback_chain",
            "estimated_input_tokens",
            "estimated_output_tokens",
            "estimated_cost_usd",
            "token_usage_source",
        },
        "GenerationTrace",
    )
    # token_usage_source must be one of the two literals the UI badges on.
    assert trace["generation"]["token_usage_source"] in ("provider", "estimated")


def test_metrics_snapshot_contract() -> None:
    """/metrics snapshot exposes every key the Metrics page renders."""
    store = MetricsStore(db_path=":memory:")
    store.record(
        provider_name="openai:gpt-4o",
        model_name="gpt-4o",
        route="RAG_ANSWER",
        latency_ms=120.0,
        overall_score=0.82,
        retry_count=1,
        fallback_used=False,
        degraded_response=False,
        estimated_cost_usd=0.0012,
    )
    snap = store.snapshot()
    _require_keys(
        snap,
        {
            "total_requests",
            "requests_by_provider",
            "requests_by_route",
            "fallback_count",
            "degraded_response_count",
            "average_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "average_overall_score",
            "score_histogram",
            "latency_histogram",
            "retry_rate",
            "fallback_rate",
            "degraded_response_rate",
            "cost_total",
            "estimated_cost_usd_total",
        },
        "MetricsSnapshot",
    )
    assert isinstance(snap["requests_by_provider"], dict)
    assert isinstance(snap["score_histogram"], dict)
    assert isinstance(snap["latency_histogram"], dict)


def test_prometheus_contract() -> None:
    """/metrics/prometheus exposes the seven series the dashboard documents."""
    store = MetricsStore(db_path=":memory:")
    store.record(
        provider_name="mock",
        model_name="mock",
        route="DIRECT_ANSWER",
        latency_ms=8.0,
        overall_score=0.5,
        retry_count=0,
        fallback_used=True,
        degraded_response=True,
        estimated_cost_usd=0.0,
    )
    text = store.prometheus_text()
    for series in (
        "aegis_total_requests",
        "aegis_fallback_count",
        "aegis_degraded_response_count",
        "aegis_average_latency_ms",
        "aegis_p95_latency_ms",
        "aegis_estimated_cost_usd_total",
        "aegis_retry_rate",
    ):
        assert series in text, f"prometheus output missing series: {series}"
    # Exposition hygiene: HELP/TYPE comments present for tooling.
    assert "# HELP" in text and "# TYPE" in text


def test_providers_health_contract() -> None:
    """/providers/health exposes per-provider fields + recommended_order."""
    resp = ProvidersHealthResponse(
        providers=[
            ProviderHealth(
                provider="openai:gpt-4o",
                health_status="degraded",
                consecutive_failures=2,
                total_successes=5,
                total_failures=2,
                last_success_at=1.0,
                last_failure_at=2.0,
            )
        ],
        recommended_order=["openai:gpt-4o", "mock"],
    )
    d = resp.model_dump()
    _require_keys(d, {"providers", "recommended_order"}, "ProvidersHealthResponse")
    _require_keys(
        d["providers"][0],
        {
            "provider",
            "health_status",
            "consecutive_failures",
            "total_successes",
            "total_failures",
            "last_success_at",
            "last_failure_at",
        },
        "ProviderHealth",
    )
    assert d["providers"][0]["health_status"] in ("healthy", "degraded", "unhealthy")


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
