"""
MockProvider — deterministic, offline generation (Module 5).

Wraps the existing Module 2 ``MockLLM`` so there is a single source of truth for
the deterministic response shape. This is the default provider: tests and demos
run with zero network calls and no API keys.
"""

from ...logging_config import get_logger
from ..generator import MockLLM
from .base import LLMProvider

logger = get_logger(__name__)


class MockProvider(LLMProvider):
    """Deterministic provider backed by the offline MockLLM."""

    name = "mock"

    def __init__(self) -> None:
        self._llm = MockLLM()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # MockLLM keys off the prompt body (CONTEXT: block, QUESTION: line),
        # so the user_prompt carries everything it needs.
        return self._llm.complete(user_prompt, system=system_prompt or None)
