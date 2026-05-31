"""
Tests for Module 7 — production observability ops upgrade.

All offline and deterministic: SQLite uses ``:memory:`` / temp files, providers
are stubs, and the Bedrock usage path is exercised against a fake boto3 client.
No network, no AWS, no SDK installs.

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module7_ops.py -v
    ./venv/bin/python tests/test_module7_ops.py
"""

import json
import os
import sys
import tempfile
import types
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.models.schemas import (  # noqa: E402
    EvaluationResult,
    EvaluationScores,
    GenerationTrace,
    Route,
)
from app.services.budget import BudgetGuard  # noqa: E402
from app.services.cost import cost_from_tokens  # noqa: E402
from app.services.health import (  # noqa: E402
    DEGRADED,
    HEALTHY,
    UNHEALTHY,
    ProviderHealthRegistry,
)
from app.services.metrics_store import MetricsStore, _percentile  # noqa: E402
from app.services.providers import (  # noqa: E402
    FallbackProvider,
    LLMProvider,
    ProviderAPIError,
    ProviderLLMClient,
)
from app.services.providers.base import TokenUsage  # noqa: E402
from app.services.retry import (  # noqa: E402
    AttemptParams,
    AttemptResult,
    RetryContext,
    RetryManager,
)
from app.services.router import RouteDecision  # noqa: E402


def _settings(**overrides):
    return replace(get_settings(), **overrides)


# ---------------------------------------------------------------------------
# Provider stubs
# ---------------------------------------------------------------------------
class UsageProvider(LLMProvider):
    """Succeeds and reports exact provider usage."""

    name = "openai:gpt-4o"
    model = "gpt-4o"

    def __init__(self):
        self.last_usage = TokenUsage(input_tokens=1000, output_tokens=1000, source="provider")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "ok"


class PlainProvider(LLMProvider):
    """Succeeds but reports no usage => caller must estimate."""

    name = "openai:gpt-4o"
    model = "gpt-4o"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "hello world"


class NamedStub(LLMProvider):
    def __init__(self, name: str):
        self.name = name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return f"from {self.name}"


class FlakyProvider(LLMProvider):
    def __init__(self, name: str):
        self.name = name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise ProviderAPIError(f"{self.name} down")


# ---------------------------------------------------------------------------
# 1. SQLite metrics persistence
# ---------------------------------------------------------------------------
def test_metrics_store_persists_across_connections():
    path = tempfile.mktemp(suffix=".db")
    try:
        s1 = MetricsStore(path)
        s1.record(
            provider_name="bedrock:claude", model_name="claude", route="RAG_ANSWER",
            latency_ms=120.0, overall_score=0.8, retry_count=1, fallback_used=True,
            degraded_response=False, estimated_cost_usd=0.0012,
        )
        # A fresh store on the same file must see the persisted row.
        s2 = MetricsStore(path)
        snap = s2.snapshot()
        assert snap["total_requests"] == 1
        assert snap["requests_by_provider"] == {"bedrock:claude": 1}
        assert snap["requests_by_route"] == {"RAG_ANSWER": 1}
        assert snap["fallback_count"] == 1
        assert snap["estimated_cost_usd_total"] == 0.0012
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_metrics_store_rates_and_counts():
    store = MetricsStore(":memory:")
    store.record(provider_name="mock", model_name="mock", route="DIRECT_ANSWER",
                 latency_ms=10, overall_score=0.9, retry_count=0)
    store.record(provider_name="bedrock:c", model_name="c", route="RAG_ANSWER",
                 latency_ms=30, overall_score=0.3, retry_count=2,
                 fallback_used=True, degraded_response=True, estimated_cost_usd=0.01)
    snap = store.snapshot()
    assert snap["total_requests"] == 2
    assert snap["retry_rate"] == 0.5
    assert snap["fallback_rate"] == 0.5
    assert snap["degraded_response_rate"] == 0.5
    assert snap["cost_total"] == 0.01
    assert snap["requests_by_route"] == {"DIRECT_ANSWER": 1, "RAG_ANSWER": 1}


# ---------------------------------------------------------------------------
# 2. Percentile latency calculation
# ---------------------------------------------------------------------------
def test_percentile_nearest_rank():
    vals = list(range(10, 101, 10))  # 10..100, already sorted, n=10
    assert _percentile(vals, 50) == 50  # ceil(0.5*10)=5 -> idx4
    assert _percentile(vals, 95) == 100
    assert _percentile(vals, 99) == 100
    assert _percentile([], 95) == 0.0


def test_metrics_store_percentiles_in_snapshot():
    store = MetricsStore(":memory:")
    for lat in range(10, 101, 10):
        store.record(provider_name="mock", model_name="mock", route="RAG_ANSWER",
                     latency_ms=lat, overall_score=0.7)
    snap = store.snapshot()
    assert snap["p50_latency_ms"] == 50
    assert snap["p95_latency_ms"] == 100
    assert snap["p99_latency_ms"] == 100
    assert sum(snap["latency_histogram"].values()) == 10
    assert sum(snap["score_histogram"].values()) == 10


# ---------------------------------------------------------------------------
# 3. Prometheus text output
# ---------------------------------------------------------------------------
def test_prometheus_text_output():
    store = MetricsStore(":memory:")
    store.record(provider_name="bedrock:claude", model_name="claude", route="RAG_ANSWER",
                 latency_ms=200, overall_score=0.8, fallback_used=True,
                 estimated_cost_usd=0.002)
    text = store.prometheus_text()
    assert "# TYPE aegis_total_requests counter" in text
    assert "aegis_total_requests 1" in text
    assert "aegis_p95_latency_ms" in text
    assert "aegis_estimated_cost_usd_total" in text
    assert "aegis_retry_rate" in text
    assert 'aegis_requests_by_provider{provider="bedrock:claude"} 1' in text


# ---------------------------------------------------------------------------
# 4 & 5. Real usage extraction vs. estimated fallback
# ---------------------------------------------------------------------------
def test_client_uses_provider_usage_when_present():
    client = ProviderLLMClient(UsageProvider())
    _text, meta = client.complete_with_meta("a prompt", system="sys")
    assert meta.token_usage_source == "provider"
    assert meta.estimated_input_tokens == 1000
    assert meta.estimated_output_tokens == 1000
    # openai family pricing: 0.005 in + 0.015 out per 1K => 0.02
    assert meta.estimated_cost_usd == cost_from_tokens("openai:gpt-4o", 1000, 1000) == 0.02


def test_client_falls_back_to_estimated_usage():
    client = ProviderLLMClient(PlainProvider())
    _text, meta = client.complete_with_meta("a prompt", system="sys")
    assert meta.token_usage_source == "estimated"
    assert meta.estimated_input_tokens > 0
    assert meta.estimated_output_tokens > 0


def test_bedrock_parses_real_usage():
    payload = {
        "content": [{"type": "text", "text": "answer"}],
        "usage": {"input_tokens": 123, "output_tokens": 45},
    }
    with _fake_boto3(_FakeBedrockClient(response_payload=payload)) as client:
        from app.services.providers.bedrock_provider import BedrockProvider

        provider = BedrockProvider(region="us-east-1")
        text = provider.generate("", "hi")
    assert text == "answer"
    assert provider.last_usage is not None
    assert provider.last_usage.input_tokens == 123
    assert provider.last_usage.output_tokens == 45
    assert provider.last_usage.source == "provider"


# ---------------------------------------------------------------------------
# 6. Budget guardrail
# ---------------------------------------------------------------------------
def _attempt(strategy: str, score: float, should_retry: bool, cost: float) -> AttemptResult:
    scores = EvaluationScores(
        relevance=score, groundedness=score, completeness=score,
        hallucination_risk=0.1, confidence=score, overall_score=score,
    )
    ev = EvaluationResult(scores=scores, should_retry=should_retry, evaluation_reason="x")
    gen = GenerationTrace(
        provider_name="openai:gpt-4o", model_name="gpt-4o", provider_latency_ms=1.0,
        fallback_used=False, fallback_chain=["openai:gpt-4o"],
        estimated_input_tokens=1000, estimated_output_tokens=1000,
        estimated_cost_usd=cost, token_usage_source="provider",
    )
    return AttemptResult(
        params=AttemptParams(query="q", top_k=None, force_route=None, strategy=strategy),
        answer="a", contexts=[], mode="grounded",
        decision=RouteDecision(Route.RAG_ANSWER, "r"),
        evaluation=ev, generation=gen,
    )


def test_budget_guard_should_block_logic():
    guard = BudgetGuard(_settings(max_request_cost_usd=0.001))
    blocked, _reason = guard.should_block_retry(0.005, 0)
    assert blocked is True
    allowed, _ = BudgetGuard(_settings()).should_block_retry(0.005, 0)  # disabled
    assert allowed is False


def test_budget_guard_blocks_expensive_retry_in_manager():
    settings = _settings(max_request_cost_usd=0.001, max_retries=2)
    guard = BudgetGuard(settings)
    rm = RetryManager(settings, budget_guard=guard)
    ctx = RetryContext(
        original_query="q", effective_top_k=5, initial_route=Route.RAG_ANSWER,
        initial_top_score=0.5, index_size=3, settings=settings,
    )
    # Initial attempt already spent past the per-request cap.
    initial = _attempt("initial", 0.2, should_retry=True, cost=0.005)
    outcome = rm.run(initial, ctx, lambda p: _attempt(p.strategy, 0.9, False, 0.005))

    assert outcome.trace.budget_blocked is True
    assert outcome.trace.retry_count == 0  # blocked before any retry ran
    assert outcome.trace.degraded_response is True


def test_no_budget_guard_allows_retries():
    settings = _settings(max_retries=2)
    rm = RetryManager(settings)  # no guard
    ctx = RetryContext(
        original_query="q", effective_top_k=5, initial_route=Route.RAG_ANSWER,
        initial_top_score=0.5, index_size=3, settings=settings,
    )
    initial = _attempt("initial", 0.2, should_retry=True, cost=0.005)
    outcome = rm.run(initial, ctx, lambda p: _attempt(p.strategy, 0.9, False, 0.005))
    assert outcome.trace.budget_blocked is False
    assert outcome.trace.retry_count >= 1


def test_budget_daily_accumulation_and_rollover():
    guard = BudgetGuard(_settings(max_daily_cost_usd=0.01))
    guard.add_daily_cost(0.004)
    guard.add_daily_cost(0.004)
    assert guard.daily_cost_usd == 0.008
    # Next request projected over the daily cap should block.
    blocked, _ = guard.should_block_retry(0.003, 1)
    assert blocked is True


# ---------------------------------------------------------------------------
# 7. Provider health status transitions
# ---------------------------------------------------------------------------
def test_health_status_transitions():
    reg = ProviderHealthRegistry()
    name = "openai:gpt-4o"
    assert reg.status(name) == HEALTHY  # unknown == healthy
    reg.record_failure(name)
    assert reg.status(name) == DEGRADED
    reg.record_failure(name)
    assert reg.status(name) == DEGRADED  # 2 consecutive still degraded
    reg.record_failure(name)
    assert reg.status(name) == UNHEALTHY  # 3 consecutive
    reg.record_success(name)
    assert reg.status(name) == HEALTHY  # streak reset


def test_health_snapshot_fields():
    reg = ProviderHealthRegistry()
    reg.record_success("mock")
    reg.record_failure("bedrock:c")
    snap = {row["provider"]: row for row in reg.snapshot()}
    assert snap["mock"]["total_successes"] == 1
    assert snap["bedrock:c"]["total_failures"] == 1
    assert snap["bedrock:c"]["health_status"] == DEGRADED


# ---------------------------------------------------------------------------
# 8. Provider health influences fallback ordering
# ---------------------------------------------------------------------------
def test_health_order_prefers_healthy_keeps_mock_last():
    reg = ProviderHealthRegistry()
    for _ in range(3):
        reg.record_failure("openai:gpt-4o")  # -> unhealthy
    order = reg.order(["openai:gpt-4o", "anthropic:claude", "mock"])
    assert order == ["anthropic:claude", "openai:gpt-4o", "mock"]


def test_fallback_tries_healthier_provider_first():
    reg = ProviderHealthRegistry()
    for _ in range(3):
        reg.record_failure("openai:gpt-4o")  # primary marked unhealthy
    primary = NamedStub("openai:gpt-4o")
    secondary = NamedStub("anthropic:claude")
    fb = FallbackProvider([primary, secondary], health_registry=reg)
    _text, served = fb.generate_verbose("", "hi")
    assert served.name == "anthropic:claude"  # healthier one preferred


def test_fallback_records_health_outcomes():
    reg = ProviderHealthRegistry()
    fb = FallbackProvider([FlakyProvider("openai:gpt-4o"), NamedStub("mock")], health_registry=reg)
    _text, served = fb.generate_verbose("", "hi")
    assert served.name == "mock"
    assert reg.status("openai:gpt-4o") == DEGRADED  # one failure recorded
    assert reg.status("mock") == HEALTHY


# ---------------------------------------------------------------------------
# Fake boto3 (for the Bedrock usage-parse test)
# ---------------------------------------------------------------------------
class _FakeBotoConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeBotoCoreError(Exception):
    pass


class _FakeNoCredentialsError(_FakeBotoCoreError):
    pass


class _FakeClientError(Exception):
    def __init__(self, code, message="boom"):
        self.response = {"Error": {"Code": code, "Message": message}}
        super().__init__(f"{code}: {message}")


class _FakeBody:
    def __init__(self, b):
        self._b = b

    def read(self):
        return self._b


class _FakeBedrockClient:
    def __init__(self, response_payload=None):
        self._payload = response_payload or {"content": [{"type": "text", "text": "x"}]}

    def invoke_model(self, **kwargs):
        return {"body": _FakeBody(json.dumps(self._payload).encode("utf-8"))}


@contextmanager
def _fake_boto3(client):
    names = ["boto3", "botocore", "botocore.config", "botocore.exceptions"]
    saved = {n: sys.modules.get(n) for n in names}
    boto3_mod = types.ModuleType("boto3")
    boto3_mod.client = lambda *a, **k: client
    botocore_mod = types.ModuleType("botocore")
    config_mod = types.ModuleType("botocore.config")
    config_mod.Config = _FakeBotoConfig
    exc_mod = types.ModuleType("botocore.exceptions")
    exc_mod.BotoCoreError = _FakeBotoCoreError
    exc_mod.ClientError = _FakeClientError
    exc_mod.NoCredentialsError = _FakeNoCredentialsError
    botocore_mod.config = config_mod
    botocore_mod.exceptions = exc_mod
    sys.modules.update({
        "boto3": boto3_mod, "botocore": botocore_mod,
        "botocore.config": config_mod, "botocore.exceptions": exc_mod,
    })
    try:
        yield client
    finally:
        for n, mod in saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod


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
