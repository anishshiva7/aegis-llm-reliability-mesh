"""
Shared application dependencies.

The RetrievalEngine owns the embedding model and the FAISS index, both of which
must persist across requests (we don't want to reload the model or lose the
index every call). So we hold a single process-wide instance and hand it to
route handlers via FastAPI's dependency-injection system.

The engine is created lazily on first access so that importing the app (e.g. in
tests that build their own engine) doesn't pay the model-load cost.
"""

from typing import Optional

from .config import Settings, get_settings
from .logging_config import get_logger
from .services.budget import get_budget_guard
from .services.generator import LLMClient
from .services.health import get_health_registry
from .services.metrics import get_metrics
from .services.metrics_store import get_metrics_store
from .services.providers import (
    FallbackProvider,
    LLMProvider,
    MockProvider,
    ProviderConfigError,
    ProviderLLMClient,
)
from .services.rag import RAGPipeline
from .services.retrieval import RetrievalEngine
from .services.retry import RetryManager
from .services.router import QueryRouter

logger = get_logger(__name__)

_engine: Optional[RetrievalEngine] = None
_pipeline: Optional[RAGPipeline] = None


def get_engine() -> RetrievalEngine:
    """Return the process-wide RetrievalEngine, building it on first use."""
    global _engine
    if _engine is None:
        logger.info("Constructing global RetrievalEngine (first request).")
        _engine = RetrievalEngine()
    return _engine


# ---------------------------------------------------------------------------
# Provider factory (Module 5)
# ---------------------------------------------------------------------------
def build_provider(settings: Settings, name: str) -> LLMProvider:
    """
    Construct a single LLMProvider by name. This is the only place that maps a
    provider name to a concrete class — the pipeline never sees this logic.

    Real-provider SDKs are imported lazily inside each class, so selecting
    ``mock`` never touches ``openai``/``anthropic``.
    """
    name = name.lower()
    if name == "mock":
        return MockProvider()
    if name == "openai":
        from .services.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.model_name,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
            max_tokens=settings.max_tokens,
        )
    if name == "anthropic":
        from .services.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.model_name,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
            max_tokens=settings.max_tokens,
        )
    if name == "bedrock":
        from .services.providers.bedrock_provider import BedrockProvider

        return BedrockProvider(
            region=settings.aws_region,
            model_id=settings.bedrock_model_id,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
            max_tokens=settings.max_tokens,
        )
    raise ProviderConfigError(
        f"Unknown provider '{name}'. Use mock|openai|anthropic|bedrock."
    )


def build_generator(settings: Optional[Settings] = None) -> LLMClient:
    """
    Build the generator the pipeline injects: the configured primary provider,
    optionally wrapped in a FallbackProvider, exposed via the LLMClient adapter.
    """
    settings = settings or get_settings()
    primary = build_provider(settings, settings.provider)

    chain: list[LLMProvider] = [primary]
    fb = settings.fallback_provider.lower()
    if fb and fb not in ("none", settings.provider.lower()):
        chain.append(build_provider(settings, fb))

    # Always wrap in the health-aware FallbackProvider — even a single-provider
    # chain — so per-provider success/failure is recorded for /providers/health
    # and the registry can reorder the chain healthiest-first (mock last) on
    # every call (Module 7). A one-element chain behaves identically to the bare
    # provider (no fallback, fallback_used stays False).
    provider: LLMProvider = FallbackProvider(
        chain, health_registry=get_health_registry()
    )

    logger.info("Generation provider ready: %s", provider.name)
    return ProviderLLMClient(provider)


def get_pipeline() -> RAGPipeline:
    """
    Return the process-wide RAGPipeline, built on first use.

    The generator is chosen by config (AEGIS_PROVIDER): ``mock`` by default,
    or a real OpenAI/Anthropic provider with an optional fallback chain. The
    pipeline itself is provider-agnostic — only this factory knows the wiring.
    """
    global _pipeline
    if _pipeline is None:
        logger.info("Constructing global RAGPipeline (first request).")
        settings = get_settings()
        _pipeline = RAGPipeline(
            engine=get_engine(),
            generator=build_generator(settings),
            router=QueryRouter(),
            # Self-healing loop governed by the cost guardrail (Module 7).
            retry_manager=RetryManager(settings, budget_guard=get_budget_guard()),
            metrics=get_metrics(),
            metrics_store=get_metrics_store(),
            budget_guard=get_budget_guard(),
        )
    return _pipeline
