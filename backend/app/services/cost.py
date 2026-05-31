"""
Cost estimation (Module 6).

A deliberately simple, dependency-free estimator. Token counts are *estimated*
from character length (≈ 4 chars/token) when a provider doesn't return exact
usage, and per-provider prices are coarse placeholders meant for relative
comparison and budget signals — not billing-grade accuracy.

Prices are USD per 1,000 tokens and overridable via env (AEGIS_*), so they can
be tuned without code changes.
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple

from ..logging_config import get_logger

logger = get_logger(__name__)

# ~4 characters per token is the standard rough heuristic for English text.
_CHARS_PER_TOKEN = 4

# (input_price, output_price) in USD per 1K tokens. Placeholders — tune freely.
_DEFAULT_PRICING: Dict[str, Tuple[float, float]] = {
    "mock": (0.0, 0.0),
    "openai": (0.005, 0.015),
    "anthropic": (0.003, 0.015),
    "bedrock": (0.003, 0.015),
}


def _price_for(family: str) -> Tuple[float, float]:
    """Per-1K (input, output) price, with an env override hook per family."""
    inp_default, out_default = _DEFAULT_PRICING.get(family, (0.0, 0.0))
    inp = float(os.environ.get(f"AEGIS_PRICE_{family.upper()}_INPUT", inp_default))
    out = float(os.environ.get(f"AEGIS_PRICE_{family.upper()}_OUTPUT", out_default))
    return inp, out


def provider_family(provider_name: str) -> str:
    """
    Reduce a provider name to a pricing family.

    "bedrock:anthropic.claude-3-5-..."  -> "bedrock"
    "openai:gpt-4o-mini"                 -> "openai"
    "mock"                                -> "mock"
    """
    return (provider_name or "").split(":", 1)[0].lower()


def estimate_tokens(text: str) -> int:
    """Estimate token count from characters (chars / 4). Clearly an estimate."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass
class CostEstimate:
    """Estimated usage + cost for one generation call."""

    input_tokens_estimate: int
    output_tokens_estimate: int
    estimated_cost_usd: float


def cost_from_tokens(provider_name: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for known token counts (used when the provider reports exact usage)."""
    in_price, out_price = _price_for(provider_family(provider_name))
    cost = (input_tokens / 1000.0) * in_price + (output_tokens / 1000.0) * out_price
    return round(cost, 6)


def estimate_cost(provider_name: str, input_text: str, output_text: str) -> CostEstimate:
    """Estimate tokens (from text) and cost (from per-family pricing)."""
    in_tokens = estimate_tokens(input_text)
    out_tokens = estimate_tokens(output_text)
    return CostEstimate(
        input_tokens_estimate=in_tokens,
        output_tokens_estimate=out_tokens,
        estimated_cost_usd=cost_from_tokens(provider_name, in_tokens, out_tokens),
    )
