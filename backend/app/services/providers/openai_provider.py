"""
OpenAIProvider — real generation via the OpenAI Chat Completions API (Module 5).

The ``openai`` SDK is imported lazily in ``__init__`` so the package stays an
optional dependency: the default mock path (and the whole test suite) runs
without it installed. Vendor SDK errors are translated into the typed
``ProviderError`` hierarchy so callers remain vendor-agnostic.
"""

from typing import Optional

from ...logging_config import get_logger
from .base import (
    LLMProvider,
    ProviderAPIError,
    ProviderConfigError,
    ProviderTimeoutError,
    TokenUsage,
)

logger = get_logger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(LLMProvider):
    """Generation backed by OpenAI chat models."""

    def __init__(
        self,
        api_key: str,
        model: str = "",
        temperature: float = 0.0,
        timeout: float = 30.0,
        max_tokens: int = 1024,
    ) -> None:
        if not api_key:
            raise ProviderConfigError("OpenAI provider requires an API key.")
        try:
            import openai  # lazy: optional dependency
        except ImportError as exc:  # pragma: no cover - exercised via factory test
            raise ProviderConfigError(
                "openai SDK not installed. Run: pip install openai"
            ) from exc

        self._openai = openai
        self.model = model or _DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"openai:{self.model}"
        # The SDK reads timeout per-client; instantiate once and reuse.
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout)
        logger.info("OpenAIProvider ready (model=%s).", self.model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.last_usage = None  # reset; populated from the response if present
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=messages,
            )
        except self._openai.APITimeoutError as exc:
            raise ProviderTimeoutError(f"OpenAI request timed out: {exc}") from exc
        except self._openai.APIError as exc:
            raise ProviderAPIError(f"OpenAI API error: {exc}") from exc
        except Exception as exc:  # defensive: never leak a raw vendor error
            raise ProviderAPIError(f"OpenAI generation failed: {exc}") from exc

        content: Optional[str] = resp.choices[0].message.content
        if not content:
            raise ProviderAPIError("OpenAI returned an empty completion.")

        # Exact usage from the API when present (prompt/completion tokens).
        usage = getattr(resp, "usage", None)
        if usage is not None:
            in_tok = getattr(usage, "prompt_tokens", None)
            out_tok = getattr(usage, "completion_tokens", None)
            if in_tok is not None and out_tok is not None:
                self.last_usage = TokenUsage(
                    input_tokens=int(in_tok),
                    output_tokens=int(out_tok),
                    source="provider",
                )
        return content
