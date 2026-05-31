"""
BedrockProvider — AWS Bedrock generation via boto3 (Module 6).

Implements the same ``LLMProvider.generate(system, user) -> str`` interface as
every other provider, so the RAG/retry/evaluation pipeline stays vendor-agnostic.
Targets Anthropic Claude on Bedrock using the Messages API request shape.

boto3 is imported lazily so it remains an optional dependency: the default mock
path and the entire test suite run without it. All AWS/boto errors are mapped
into the existing typed ``ProviderError`` hierarchy.
"""

import json
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

_DEFAULT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
_ANTHROPIC_VERSION = "bedrock-2023-05-31"


class BedrockProvider(LLMProvider):
    """Generation backed by Anthropic Claude models hosted on AWS Bedrock."""

    def __init__(
        self,
        region: str,
        model_id: str = "",
        temperature: float = 0.0,
        timeout: float = 30.0,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import boto3  # lazy: optional dependency
            from botocore.config import Config as BotoConfig
            from botocore.exceptions import (  # noqa: F401  (stored for generate())
                BotoCoreError,
                ClientError,
                NoCredentialsError,
            )
        except ImportError as exc:  # pragma: no cover - exercised via factory test
            raise ProviderConfigError(
                "boto3 not installed. Run: pip install boto3"
            ) from exc

        if not region:
            raise ProviderConfigError("Bedrock provider requires an AWS region.")

        self._ClientError = ClientError
        self._BotoCoreError = BotoCoreError
        self._NoCredentialsError = NoCredentialsError

        self.model_id = model_id or _DEFAULT_MODEL_ID
        # Surfaced in traces/cost as the model name for this provider.
        self.model = self.model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"bedrock:{self.model_id}"

        # read_timeout bounds the per-call wait; retries=0 keeps our own typed
        # error handling authoritative (Module 4/5 own retry semantics).
        boto_config = BotoConfig(
            read_timeout=timeout,
            connect_timeout=timeout,
            retries={"max_attempts": 0},
        )
        try:
            self._client = boto3.client(
                "bedrock-runtime", region_name=region, config=boto_config
            )
        except Exception as exc:  # credential resolution can fail eagerly
            raise ProviderConfigError(f"Could not create Bedrock client: {exc}") from exc

        logger.info("BedrockProvider ready (region=%s model=%s).", region, self.model_id)

    def _build_body(self, system_prompt: str, user_prompt: str) -> str:
        """Anthropic-on-Bedrock Messages API request body."""
        body = {
            "anthropic_version": _ANTHROPIC_VERSION,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt
        return json.dumps(body)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.last_usage = None  # reset; populated from the response if present
        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=self._build_body(system_prompt, user_prompt),
            )
        except self._NoCredentialsError as exc:
            raise ProviderConfigError(f"AWS credentials not found: {exc}") from exc
        except self._ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
            if code in ("ThrottlingException", "TooManyRequestsException"):
                raise ProviderAPIError(f"Bedrock throttled the request: {exc}") from exc
            if code in ("AccessDeniedException", "UnrecognizedClientException"):
                raise ProviderConfigError(f"Bedrock access denied: {exc}") from exc
            raise ProviderAPIError(f"Bedrock API error ({code}): {exc}") from exc
        except self._BotoCoreError as exc:
            # Connect/read timeouts surface as botocore errors.
            if "timeout" in str(exc).lower():
                raise ProviderTimeoutError(f"Bedrock request timed out: {exc}") from exc
            raise ProviderAPIError(f"Bedrock transport error: {exc}") from exc
        except Exception as exc:  # defensive: never leak a raw vendor error
            raise ProviderAPIError(f"Bedrock generation failed: {exc}") from exc

        return self._parse_text(response)

    def _parse_text(self, response: dict) -> str:
        """Extract assistant text from a Bedrock invoke_model response."""
        try:
            raw = response["body"].read()
            payload = json.loads(raw)
        except Exception as exc:
            raise ProviderAPIError(f"Malformed Bedrock response: {exc}") from exc

        # Anthropic Messages API: {"content": [{"type":"text","text":"..."}], ...}
        content = payload.get("content")
        if not isinstance(content, list):
            raise ProviderAPIError("Bedrock response missing 'content' list.")
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text: Optional[str] = "".join(parts).strip()
        if not text:
            raise ProviderAPIError("Bedrock returned an empty completion.")

        # Capture exact usage when Bedrock returns it; otherwise the adapter
        # falls back to the chars/4 estimate. Anthropic-on-Bedrock reports
        # {"usage": {"input_tokens": N, "output_tokens": M}}.
        usage = payload.get("usage")
        if isinstance(usage, dict) and "input_tokens" in usage and "output_tokens" in usage:
            self.last_usage = TokenUsage(
                input_tokens=int(usage["input_tokens"]),
                output_tokens=int(usage["output_tokens"]),
                source="provider",
            )
        return text
