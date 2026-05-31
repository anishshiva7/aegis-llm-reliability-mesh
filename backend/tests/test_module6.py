"""
Tests for Module 6 — AWS Bedrock provider + production observability.

Everything runs offline and deterministically: no AWS account, no boto3 install,
no network. The Bedrock provider is exercised against a *fake* boto3/botocore
injected into ``sys.modules``, so we validate request formatting, response
parsing, and the typed-error mapping without ever leaving the process.

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module6.py -v
    ./venv/bin/python tests/test_module6.py
"""

import json
import sys
import types
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.dependencies import build_provider  # noqa: E402
from app.models.schemas import GenerationTrace, Route  # noqa: E402
from app.services.cost import (  # noqa: E402
    CostEstimate,
    estimate_cost,
    estimate_tokens,
    provider_family,
)
from app.services.metrics import MetricsCollector, RequestMetric, get_metrics  # noqa: E402
from app.services.providers import (  # noqa: E402
    FallbackProvider,
    LLMProvider,
    MockProvider,
    ProviderAPIError,
    ProviderConfigError,
    ProviderError,
    ProviderLLMClient,
)
from app.services.rag import RAGPipeline  # noqa: E402
from app.services.vector_store import ChunkRecord  # noqa: E402


# ---------------------------------------------------------------------------
# Fake boto3 / botocore (installed into sys.modules for the duration of a test)
# ---------------------------------------------------------------------------
class _FakeBotoConfig:
    """Stand-in for botocore.config.Config — just records kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeBotoCoreError(Exception):
    """Base botocore error (transport-level failures)."""


class _FakeNoCredentialsError(_FakeBotoCoreError):
    """No AWS credentials resolved."""


class _FakeClientError(Exception):
    """Mirror of botocore.exceptions.ClientError: carries a .response dict."""

    def __init__(self, code: str, message: str = "boom"):
        self.response = {"Error": {"Code": code, "Message": message}}
        super().__init__(f"{code}: {message}")


class _FakeBody:
    def __init__(self, payload_bytes: bytes):
        self._b = payload_bytes

    def read(self) -> bytes:
        return self._b


class _FakeBedrockClient:
    """A bedrock-runtime client whose invoke_model is fully controllable."""

    def __init__(self, response_payload=None, raise_exc=None, capture=None):
        self._payload = (
            response_payload
            if response_payload is not None
            else {"content": [{"type": "text", "text": "grounded bedrock answer"}]}
        )
        self._raise_exc = raise_exc
        self.capture = capture if capture is not None else {}

    def invoke_model(self, **kwargs):
        self.capture.update(kwargs)
        if self._raise_exc is not None:
            raise self._raise_exc
        return {"body": _FakeBody(json.dumps(self._payload).encode("utf-8"))}


@contextmanager
def fake_boto3(client: _FakeBedrockClient):
    """Install a fake boto3 + botocore into sys.modules; restore on exit."""
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

    sys.modules["boto3"] = boto3_mod
    sys.modules["botocore"] = botocore_mod
    sys.modules["botocore.config"] = config_mod
    sys.modules["botocore.exceptions"] = exc_mod
    try:
        yield client
    finally:
        for n, mod in saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod


@contextmanager
def no_boto3():
    """Make ``import boto3`` raise ImportError for the duration of the block."""
    saved = sys.modules.get("boto3", "__absent__")
    sys.modules["boto3"] = None  # forces ImportError on import
    try:
        yield
    finally:
        if saved == "__absent__":
            sys.modules.pop("boto3", None)
        else:
            sys.modules["boto3"] = saved


def _make_bedrock(client: _FakeBedrockClient, **overrides):
    """Construct a BedrockProvider against the fake boto3 client."""
    from app.services.providers.bedrock_provider import BedrockProvider

    kwargs = dict(region="us-east-1", model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
    kwargs.update(overrides)
    return BedrockProvider(**kwargs)


# ---------------------------------------------------------------------------
# Helpers / stubs (mirrors Module 5 test style)
# ---------------------------------------------------------------------------
def _record(text: str, source: str = "doc.txt", idx: int = 0) -> ChunkRecord:
    return ChunkRecord(text=text, source=source, chunk_index=idx)


class FakeEngine:
    def __init__(self, hits=None, total_chunks=None):
        self._hits = hits or []
        self._total = total_chunks if total_chunks is not None else len(self._hits)

    def search(self, query, top_k=None):
        return self._hits[:top_k] if top_k else self._hits

    @property
    def total_chunks(self):
        return self._total


class FailingProvider(LLMProvider):
    name = "failing"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise ProviderAPIError("simulated provider outage")


def _strong_engine():
    return FakeEngine(hits=[(0, 0.87, _record("The Eiffel Tower is in Paris."))])


# ---------------------------------------------------------------------------
# Bedrock provider — request formatting
# ---------------------------------------------------------------------------
def test_bedrock_builds_anthropic_messages_body():
    client = _FakeBedrockClient()
    with fake_boto3(client):
        provider = _make_bedrock(client, temperature=0.2, max_tokens=512)
        out = provider.generate("be careful", "What is the capital of France?")

    assert out == "grounded bedrock answer"
    # The provider should have called invoke_model with a well-formed body.
    assert client.capture["modelId"] == "anthropic.claude-3-5-sonnet-20240620-v1:0"
    body = json.loads(client.capture["body"])
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["max_tokens"] == 512
    assert body["temperature"] == 0.2
    assert body["system"] == "be careful"
    assert body["messages"] == [{"role": "user", "content": "What is the capital of France?"}]


def test_bedrock_omits_system_when_empty():
    client = _FakeBedrockClient()
    with fake_boto3(client):
        provider = _make_bedrock(client)
        provider.generate("", "hello")
    body = json.loads(client.capture["body"])
    assert "system" not in body


def test_bedrock_parses_multiple_text_blocks():
    payload = {"content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]}
    client = _FakeBedrockClient(response_payload=payload)
    with fake_boto3(client):
        provider = _make_bedrock(client)
        assert provider.generate("", "x") == "Hello world"


# ---------------------------------------------------------------------------
# Bedrock provider — error mapping
# ---------------------------------------------------------------------------
def test_bedrock_throttling_maps_to_api_error():
    client = _FakeBedrockClient(raise_exc=_FakeClientError("ThrottlingException"))
    with fake_boto3(client):
        provider = _make_bedrock(client)
        try:
            provider.generate("", "x")
            assert False, "expected ProviderAPIError"
        except ProviderAPIError:
            pass


def test_bedrock_access_denied_maps_to_config_error():
    client = _FakeBedrockClient(raise_exc=_FakeClientError("AccessDeniedException"))
    with fake_boto3(client):
        provider = _make_bedrock(client)
        try:
            provider.generate("", "x")
            assert False, "expected ProviderConfigError"
        except ProviderConfigError:
            pass


def test_bedrock_missing_credentials_maps_to_config_error():
    client = _FakeBedrockClient(raise_exc=_FakeNoCredentialsError("no creds"))
    with fake_boto3(client):
        provider = _make_bedrock(client)
        try:
            provider.generate("", "x")
            assert False, "expected ProviderConfigError"
        except ProviderConfigError:
            pass


def test_bedrock_empty_completion_is_api_error():
    client = _FakeBedrockClient(response_payload={"content": []})
    with fake_boto3(client):
        provider = _make_bedrock(client)
        try:
            provider.generate("", "x")
            assert False, "expected ProviderAPIError"
        except ProviderAPIError:
            pass


def test_bedrock_requires_region():
    client = _FakeBedrockClient()
    with fake_boto3(client):
        try:
            _make_bedrock(client, region="")
            assert False, "expected ProviderConfigError"
        except ProviderConfigError:
            pass


def test_bedrock_missing_boto3_raises_config_error():
    with no_boto3():
        from app.services.providers.bedrock_provider import BedrockProvider

        try:
            BedrockProvider(region="us-east-1")
            assert False, "expected ProviderConfigError"
        except ProviderConfigError:
            pass


# ---------------------------------------------------------------------------
# Provider factory selects bedrock
# ---------------------------------------------------------------------------
def test_factory_selects_bedrock():
    client = _FakeBedrockClient()
    with fake_boto3(client):
        from app.services.providers.bedrock_provider import BedrockProvider

        settings = replace(
            get_settings(),
            provider="bedrock",
            aws_region="us-west-2",
            bedrock_model_id="anthropic.claude-3-haiku-20240307-v1:0",
        )
        provider = build_provider(settings, "bedrock")
    assert isinstance(provider, BedrockProvider)
    assert provider.name == "bedrock:anthropic.claude-3-haiku-20240307-v1:0"
    assert provider.model == "anthropic.claude-3-haiku-20240307-v1:0"


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------
def test_provider_family_parsing():
    assert provider_family("bedrock:anthropic.claude-3-5-sonnet") == "bedrock"
    assert provider_family("openai:gpt-4o-mini") == "openai"
    assert provider_family("mock") == "mock"


def test_estimate_tokens_heuristic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1  # 4 chars / 4
    assert estimate_tokens("a" * 40) == 10


def test_estimate_cost_structure_and_positive_for_paid_family():
    est = estimate_cost("openai:gpt-4o", "x" * 4000, "y" * 4000)
    assert isinstance(est, CostEstimate)
    assert est.input_tokens_estimate == 1000
    assert est.output_tokens_estimate == 1000
    # openai pricing: 0.005 in + 0.015 out per 1K => 0.005 + 0.015 = 0.02
    assert abs(est.estimated_cost_usd - 0.02) < 1e-9


def test_estimate_cost_zero_for_mock():
    est = estimate_cost("mock", "hello there", "general kenobi")
    assert est.estimated_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------
def test_metrics_collector_aggregates():
    mc = MetricsCollector()
    mc.record(RequestMetric(provider="mock", latency_ms=10.0, overall_score=0.8))
    mc.record(
        RequestMetric(
            provider="bedrock:claude",
            latency_ms=30.0,
            overall_score=0.4,
            fallback_used=True,
            degraded=True,
            retried=True,
            estimated_cost_usd=0.002,
        )
    )
    snap = mc.snapshot()
    assert snap["total_requests"] == 2
    assert snap["requests_by_provider"] == {"mock": 1, "bedrock:claude": 1}
    assert snap["fallback_count"] == 1
    assert snap["degraded_response_count"] == 1
    assert snap["average_latency_ms"] == 20.0
    assert snap["retry_rate"] == 0.5
    assert snap["average_overall_score"] == 0.6
    assert snap["estimated_cost_usd_total"] == 0.002


def test_metrics_snapshot_empty_is_safe():
    mc = MetricsCollector()
    snap = mc.snapshot()
    assert snap["total_requests"] == 0
    assert snap["average_latency_ms"] == 0.0  # no division by zero


def test_get_metrics_is_singleton():
    assert get_metrics() is get_metrics()


# ---------------------------------------------------------------------------
# Observability threaded into the pipeline trace
# ---------------------------------------------------------------------------
def test_trace_includes_generation_metadata():
    gen = ProviderLLMClient(MockProvider())
    pipe = RAGPipeline(engine=_strong_engine(), generator=gen, metrics=MetricsCollector())
    resp = pipe.ask("Where is the Eiffel Tower?")

    g = resp.trace.generation
    assert isinstance(g, GenerationTrace)
    assert g.provider_name == "mock"
    assert g.model_name == "mock"
    assert g.fallback_used is False
    assert g.fallback_chain == ["mock"]
    assert g.estimated_input_tokens > 0
    assert g.estimated_output_tokens > 0
    assert g.estimated_cost_usd == 0.0  # mock is free


def test_trace_records_fallback_used_when_primary_fails():
    gen = ProviderLLMClient(FallbackProvider([FailingProvider(), MockProvider()]))
    pipe = RAGPipeline(engine=_strong_engine(), generator=gen)
    resp = pipe.ask("Where is the Eiffel Tower?")

    g = resp.trace.generation
    assert g is not None
    assert g.fallback_used is True
    assert g.provider_name == "mock"
    assert g.fallback_chain == ["failing", "mock"]
    assert resp.trace.generation_error is None  # fallback recovered


def test_failed_generation_still_has_generation_meta():
    gen = ProviderLLMClient(FailingProvider())
    pipe = RAGPipeline(engine=_strong_engine(), generator=gen)
    resp = pipe.ask("Where is the Eiffel Tower?")

    assert resp.trace.generation_error is not None
    g = resp.trace.generation
    assert g is not None
    assert g.provider_name == "failing"
    assert g.estimated_cost_usd == 0.0


def test_pipeline_records_metrics():
    mc = MetricsCollector()
    gen = ProviderLLMClient(MockProvider())
    pipe = RAGPipeline(engine=_strong_engine(), generator=gen, metrics=mc)
    pipe.ask("Where is the Eiffel Tower?")
    pipe.ask("Where is the Eiffel Tower?")

    snap = mc.snapshot()
    assert snap["total_requests"] == 2
    assert snap["requests_by_provider"].get("mock") == 2


def test_pipeline_without_metrics_is_noop():
    """No metrics collector injected => ask() must not blow up."""
    gen = ProviderLLMClient(MockProvider())
    pipe = RAGPipeline(engine=_strong_engine(), generator=gen)
    resp = pipe.ask("Where is the Eiffel Tower?")
    assert resp.answer


# ---------------------------------------------------------------------------
# Regression — Modules 2-5 unaffected by the metadata interface
# ---------------------------------------------------------------------------
def test_legacy_generator_without_meta_still_works():
    """A bare MockLLM (no complete_with_meta) keeps working; generation is None."""
    from app.services.generator import MockLLM

    pipe = RAGPipeline(engine=_strong_engine(), generator=MockLLM())
    resp = pipe.ask("Where is the Eiffel Tower?")
    assert resp.route is Route.RAG_ANSWER
    assert resp.trace.generation is None  # legacy path has no metadata
    assert resp.trace.generation_error is None


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
