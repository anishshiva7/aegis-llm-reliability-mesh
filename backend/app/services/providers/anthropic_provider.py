"""
AnthropicProvider — real generation via the Anthropic Messages API (Module 5).

The ``anthropic`` SDK is imported lazily so it remains an optional dependency.
Vendor errors are translated into the typed ``ProviderError`` hierarchy.
"""

from ...logging_config import get_logger
from .base import (
    LLMProvider,
    ProviderAPIError,
    ProviderConfigError,
    ProviderTimeoutError,
    TokenUsage,
)

logger = get_logger(__name__)

_DEFAULT_MODEL = "claude-3-5-sonnet-latest"


class AnthropicProvider(LLMProvider):
    """Generation backed by Anthropic Claude models."""

    def __init__(
        self,
        api_key: str,
        model: str = "",
        temperature: float = 0.0,
        timeout: float = 30.0,
        max_tokens: int = 1024,
    ) -> None:
        if not api_key:
            raise ProviderConfigError("Anthropic provider requires an API key.")
        try:
            import anthropic  # lazy: optional dependency
        except ImportError as exc:  # pragma: no cover - exercised via factory test
            raise ProviderConfigError(
                "anthropic SDK not installed. Run: pip install anthropic"
            ) from exc

        self._anthropic = anthropic
        self.model = model or _DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"anthropic:{self.model}"
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        logger.info("AnthropicProvider ready (model=%s).", self.model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.last_usage = None  # reset; populated from the response if present
        try:
            msg = self._client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                system=system_prompt or "",
                messages=[{"role": "user", "content": user_prompt}],
            )
        except self._anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(f"Anthropic request timed out: {exc}") from exc
        except self._anthropic.APIError as exc:
            raise ProviderAPIError(f"Anthropic API error: {exc}") from exc
        except Exception as exc:  # defensive: never leak a raw vendor error
            raise ProviderAPIError(f"Anthropic generation failed: {exc}") from exc

        # Messages API returns a list of content blocks; concatenate text blocks.
        parts = [block.text for block in msg.content if getattr(block, "type", None) == "text"]
        text = "".join(parts).strip()
        if not text:
            raise ProviderAPIError("Anthropic returned an empty completion.")

        # Exact usage from the Messages API when present.
        usage = getattr(msg, "usage", None)
        if usage is not None:
            in_tok = getattr(usage, "input_tokens", None)
            out_tok = getattr(usage, "output_tokens", None)
            if in_tok is not None and out_tok is not None:
                self.last_usage = TokenUsage(
                    input_tokens=int(in_tok),
                    output_tokens=int(out_tok),
                    source="provider",
                )
        return text
