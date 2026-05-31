"""
LLMProvider — the vendor-agnostic generation interface (Module 5).

Every concrete backend (OpenAI, Anthropic, a mock, Bedrock later) implements the
single ``generate(system_prompt, user_prompt) -> str`` method. The rest of the
system depends only on this interface, so swapping or adding a vendor never
touches the routing, retrieval, retry, or evaluation layers.

Typed exceptions let the orchestration layer (RAGPipeline) catch *one* abstract
error type — ``ProviderError`` — without knowing which vendor failed or how.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenUsage:
    """
    Token usage for one generation call (Module 7).

    ``source`` records *where the numbers came from*: ``"provider"`` when the
    vendor returned exact usage, ``"estimated"`` when we fell back to the
    chars/4 heuristic. This honesty matters for cost dashboards.
    """

    input_tokens: int
    output_tokens: int
    source: str = "provider"  # "provider" | "estimated"


class ProviderError(Exception):
    """Base class for all provider failures. The pipeline catches this type."""


class ProviderConfigError(ProviderError):
    """Provider is misconfigured (missing SDK, missing API key, bad model)."""


class ProviderTimeoutError(ProviderError):
    """The provider request exceeded the configured timeout."""


class ProviderAPIError(ProviderError):
    """The provider's API returned an error (rate limit, 5xx, auth, etc.)."""


class LLMProvider(ABC):
    """Minimal interface every generation backend must implement."""

    #: Stable identifier surfaced in logs/traces (e.g. "openai:gpt-4o-mini").
    name: str = "abstract-provider"

    #: Real token usage from the most recent ``generate`` call, when the vendor
    #: reports it (Module 7). Left ``None`` => the caller estimates instead.
    #: Set inside ``generate``; read immediately after by the adapter.
    last_usage: Optional[TokenUsage] = None

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Return a completion for ``user_prompt`` guided by ``system_prompt``.

        Implementations MUST translate vendor SDK errors into the typed
        ``ProviderError`` hierarchy above so callers stay vendor-agnostic.
        """
        raise NotImplementedError
