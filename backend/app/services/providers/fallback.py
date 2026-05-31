"""
FallbackProvider — lightweight provider fallback chain (Module 5).

Wraps an ordered list of providers and tries each in turn. The first to return
successfully wins; a ``ProviderError`` advances to the next. If every provider
fails, the last error is re-raised so the orchestration layer can degrade.

Because it implements ``LLMProvider`` itself, fallback composes transparently —
the pipeline and adapter never know whether they hold one provider or a chain.
"""

from typing import List, Optional, Tuple

from ...logging_config import get_logger
from .base import LLMProvider, ProviderConfigError, ProviderError

logger = get_logger(__name__)


class FallbackProvider(LLMProvider):
    """Try providers in order; advance on ProviderError."""

    def __init__(
        self,
        providers: List[LLMProvider],
        health_registry=None,
    ) -> None:
        if not providers:
            raise ProviderConfigError("FallbackProvider requires at least one provider.")
        self._providers = providers
        # Optional ProviderHealthRegistry (Module 7). When set, outcomes are
        # recorded and the chain is reordered healthiest-first per call. When
        # None (the default), behaviour is identical to Modules 5/6.
        self._health = health_registry
        self.name = "fallback(" + " -> ".join(p.name for p in providers) + ")"

    @property
    def providers(self) -> List[LLMProvider]:
        """The underlying chain, primary first (read access for observability)."""
        return list(self._providers)

    @property
    def provider_names(self) -> List[str]:
        """Provider names in priority order — surfaced as the fallback chain."""
        return [p.name for p in self._providers]

    def _ordered(self) -> List[LLMProvider]:
        """Health-aware attempt order (healthiest first; mock last)."""
        if self._health is None:
            return self._providers
        names = [p.name for p in self._providers]
        preferred = self._health.order(names)
        by_name = {p.name: p for p in self._providers}
        # Preserve any duplicates/unknowns gracefully via index fallback.
        return [by_name[n] for n in preferred if n in by_name]

    def generate_verbose(
        self, system_prompt: str, user_prompt: str
    ) -> Tuple[str, LLMProvider]:
        """
        Like ``generate`` but also returns *which* provider actually served the
        request, so the caller can record fallback usage and per-provider cost.
        """
        last_error: ProviderError = ProviderError("no providers attempted")
        for i, provider in enumerate(self._ordered()):
            try:
                result = provider.generate(system_prompt, user_prompt)
                if self._health is not None:
                    self._health.record_success(provider.name)
                if i > 0:
                    logger.warning(
                        "Provider fallback engaged: served by '%s' (position %d).",
                        provider.name,
                        i,
                    )
                return result, provider
            except ProviderError as exc:
                if self._health is not None:
                    self._health.record_failure(provider.name)
                logger.warning(
                    "Provider '%s' failed (%s); trying next in chain.",
                    provider.name,
                    exc,
                )
                last_error = exc
        raise last_error

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        text, _ = self.generate_verbose(system_prompt, user_prompt)
        return text
