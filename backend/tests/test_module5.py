"""
Tests for Module 5 — real LLM provider abstraction + graceful degradation.

Everything runs offline and deterministically: no SDKs, no API keys, no network.
Real providers (OpenAI/Anthropic) are exercised only through the factory's
typed-error path; generation failure and fallback are tested with stub
providers that implement the LLMProvider interface.

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module5.py -v
    ./venv/bin/python tests/test_module5.py
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.dependencies import build_generator, build_provider  # noqa: E402
from app.models.schemas import Route  # noqa: E402
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
# Helpers / stubs
# ---------------------------------------------------------------------------

def _record(text: str, source: str = "doc.txt", idx: int = 0) -> ChunkRecord:
    return ChunkRecord(text=text, source=source, chunk_index=idx)


class FakeEngine:
    """Minimal RetrievalEngine stand-in (same pattern as Modules 2-4 tests)."""

    def __init__(self, hits=None, total_chunks=None):
        self._hits = hits or []
        self._total = total_chunks if total_chunks is not None else len(self._hits)

    def search(self, query, top_k=None):
        return self._hits[:top_k] if top_k else self._hits

    @property
    def total_chunks(self):
        return self._total


class FailingProvider(LLMProvider):
    """A provider that always raises — simulates a down/erroring vendor."""

    name = "failing"

    def __init__(self, exc: ProviderError = None) -> None:
        self._exc = exc or ProviderAPIError("simulated provider outage")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise self._exc


def _strong_engine():
    return FakeEngine(hits=[(0, 0.87, _record("The Eiffel Tower is in Paris."))])


def _settings(**overrides):
    return replace(get_settings(), **overrides)


# ---------------------------------------------------------------------------
# Config defaults — offline safety
# ---------------------------------------------------------------------------

def test_default_provider_is_mock():
    """Default provider must be mock so tests/demos never call out."""
    assert get_settings().provider == "mock"


# ---------------------------------------------------------------------------
# Provider factory selection
# ---------------------------------------------------------------------------

def test_factory_builds_mock_generator():
    gen = build_generator(_settings(provider="mock", fallback_provider="mock"))
    assert isinstance(gen, ProviderLLMClient)
    assert "mock" in gen.name
    out = gen.complete("QUESTION: hi\nANSWER:")
    assert out  # deterministic mock response


def test_factory_openai_missing_key_raises_typed_error():
    """Selecting openai without a key raises a typed ProviderConfigError (no network)."""
    try:
        build_provider(_settings(provider="openai", openai_api_key=""), "openai")
        assert False, "expected ProviderConfigError"
    except ProviderConfigError:
        pass


def test_factory_anthropic_missing_key_raises_typed_error():
    try:
        build_provider(_settings(provider="anthropic", anthropic_api_key=""), "anthropic")
        assert False, "expected ProviderConfigError"
    except ProviderConfigError:
        pass


def test_factory_unknown_provider_raises():
    try:
        build_provider(_settings(), "gemini")
        assert False, "expected ProviderConfigError"
    except ProviderConfigError:
        pass


# ---------------------------------------------------------------------------
# Mock provider determinism
# ---------------------------------------------------------------------------

def test_mock_provider_deterministic_grounded_and_direct():
    mp = MockProvider()
    grounded = mp.generate("system", "CONTEXT:\n[1] foo\n\nQUESTION: where?\nANSWER:")
    direct = mp.generate("", "QUESTION: what is 2+2?\nANSWER:")
    assert "grounded" in grounded.lower()
    assert "direct" in direct.lower()
    # Determinism: same input -> same output.
    assert mp.generate("", "QUESTION: x\nANSWER:") == mp.generate("", "QUESTION: x\nANSWER:")


# ---------------------------------------------------------------------------
# Provider failure -> graceful degradation
# ---------------------------------------------------------------------------

def test_provider_failure_returns_degraded_and_preserves_trace():
    pipe = RAGPipeline(engine=_strong_engine(), generator=ProviderLLMClient(FailingProvider()))
    resp = pipe.ask("Where is the Eiffel Tower?")

    assert resp.trace is not None, "trace must survive a provider failure"
    assert resp.trace.generation_error is not None
    assert "outage" in resp.trace.generation_error
    # Evaluation reflects the failure (degraded answer is sub-threshold).
    assert resp.trace.evaluation is not None
    assert resp.trace.evaluation.should_retry is True
    # A (degraded) answer is still returned, not an exception.
    assert resp.answer


def test_provider_failure_does_not_crash_with_include_trace_false():
    pipe = RAGPipeline(engine=_strong_engine(), generator=ProviderLLMClient(FailingProvider()))
    resp = pipe.ask("Where is the Eiffel Tower?", include_trace=False)
    assert resp.trace is None
    assert resp.answer  # degraded text, no crash


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

def test_fallback_provider_recovers_on_primary_failure():
    """FallbackProvider serves from the secondary when the primary raises."""
    fp = FallbackProvider([FailingProvider(), MockProvider()])
    out = fp.generate("", "QUESTION: hi\nANSWER:")
    assert "direct" in out.lower()  # answered by the mock fallback


def test_fallback_all_fail_reraises():
    fp = FallbackProvider([FailingProvider(), FailingProvider()])
    try:
        fp.generate("", "prompt")
        assert False, "expected ProviderError when whole chain fails"
    except ProviderError:
        pass


def test_pipeline_recovers_via_fallback_no_generation_error():
    """Primary down + mock fallback => normal answer, no generation_error in trace."""
    gen = ProviderLLMClient(FallbackProvider([FailingProvider(), MockProvider()]))
    pipe = RAGPipeline(engine=_strong_engine(), generator=gen)
    resp = pipe.ask("Where is the Eiffel Tower?")

    assert resp.trace.generation_error is None
    assert resp.route is Route.RAG_ANSWER
    assert "grounded" in resp.answer.lower()


def test_pipeline_degrades_when_entire_chain_fails():
    gen = ProviderLLMClient(FallbackProvider([FailingProvider(), FailingProvider()]))
    pipe = RAGPipeline(engine=_strong_engine(), generator=gen)
    resp = pipe.ask("Where is the Eiffel Tower?")

    assert resp.trace.generation_error is not None
    assert resp.answer


# ---------------------------------------------------------------------------
# Trace integrity / regression with mock
# ---------------------------------------------------------------------------

def test_normal_mock_path_has_no_generation_error():
    """Backward-compat: the default mock path is unchanged and error-free."""
    gen = build_generator(_settings(provider="mock"))
    pipe = RAGPipeline(engine=_strong_engine(), generator=gen)
    resp = pipe.ask("Where is the Eiffel Tower?")

    assert resp.route is Route.RAG_ANSWER
    assert resp.trace.generation_error is None
    assert resp.trace.evaluation is not None
    assert resp.trace.evaluation.should_retry is False


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
