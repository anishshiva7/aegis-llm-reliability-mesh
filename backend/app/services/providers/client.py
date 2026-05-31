"""
ProviderLLMClient — adapter from LLMProvider to the existing LLMClient (Module 5).

The RAGPipeline depends on the Module 2 ``LLMClient.complete(prompt, system)``
interface. This adapter lets any ``LLMProvider`` (single or a fallback chain)
satisfy that interface unchanged, so introducing real providers requires *no*
pipeline edits. It maps the pipeline's (prompt, system) call onto the provider's
``generate(system_prompt, user_prompt)`` signature.

Module 6 adds an *optional* ``complete_with_meta`` method that returns a
``GenerationTrace`` alongside the text (latency, served provider, fallback usage,
and estimated tokens/cost). The plain ``complete`` contract is untouched, so the
pipeline keeps working with any legacy generator that lacks the richer method.
"""

import time
from typing import List, Optional, Tuple

from ...logging_config import get_logger
from ...models.schemas import GenerationTrace
from ..cost import cost_from_tokens, estimate_cost
from ..generator import LLMClient
from .base import LLMProvider

logger = get_logger(__name__)


class ProviderLLMClient(LLMClient):
    """Expose an LLMProvider through the pipeline's LLMClient interface."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self.name = f"provider:{provider.name}"

        # Resolve the primary provider (head of a fallback chain, or the
        # provider itself) so we can report a stable provider/model name and the
        # full chain for observability — without the pipeline knowing the shape.
        if hasattr(provider, "providers"):  # FallbackProvider
            chain: List[LLMProvider] = provider.providers
        else:
            chain = [provider]
        primary = chain[0]
        self.provider_name: str = primary.name
        self.model_name: str = getattr(primary, "model", primary.name)
        self.fallback_chain: List[str] = [p.name for p in chain]

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        # Pipeline builds the full prompt; the provider treats it as the user
        # message and `system` as the system prompt. ProviderError propagates
        # to the pipeline, which degrades gracefully.
        text, _served = self._invoke(system or "", prompt)
        return text

    def complete_with_meta(
        self, prompt: str, system: Optional[str] = None
    ) -> Tuple[str, GenerationTrace]:
        """Generate and return ``(text, GenerationTrace)`` for observability."""
        system_prompt = system or ""
        start = time.perf_counter()
        text, served = self._invoke(system_prompt, prompt)
        latency_ms = (time.perf_counter() - start) * 1000.0

        served_name = served.name
        # Prefer exact usage the provider reported on this call; otherwise fall
        # back to the chars/4 estimate. ``token_usage_source`` records which.
        usage = getattr(served, "last_usage", None)
        if usage is not None:
            in_tokens, out_tokens = usage.input_tokens, usage.output_tokens
            cost_usd = cost_from_tokens(served_name, in_tokens, out_tokens)
            usage_source = "provider"
        else:
            est = estimate_cost(served_name, system_prompt + "\n" + prompt, text)
            in_tokens, out_tokens = est.input_tokens_estimate, est.output_tokens_estimate
            cost_usd = est.estimated_cost_usd
            usage_source = "estimated"

        meta = GenerationTrace(
            provider_name=served_name,
            model_name=getattr(served, "model", served_name),
            provider_latency_ms=round(latency_ms, 2),
            fallback_used=served_name != self.provider_name,
            fallback_chain=list(self.fallback_chain),
            estimated_input_tokens=in_tokens,
            estimated_output_tokens=out_tokens,
            estimated_cost_usd=cost_usd,
            token_usage_source=usage_source,
        )
        return text, meta

    # ------------------------------------------------------------------ helpers
    def _invoke(self, system_prompt: str, prompt: str) -> Tuple[str, LLMProvider]:
        """Call the provider, returning text + the provider that served it."""
        if hasattr(self._provider, "generate_verbose"):  # FallbackProvider
            return self._provider.generate_verbose(system_prompt, prompt)
        text = self._provider.generate(system_prompt=system_prompt, user_prompt=prompt)
        return text, self._provider
