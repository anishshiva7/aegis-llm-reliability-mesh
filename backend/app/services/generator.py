"""
Generation abstraction.

This is the seam where a real LLM (AWS Bedrock, OpenAI, a local model) will
plug in later. Everything upstream (the RAG pipeline) talks to the small
``LLMClient`` interface and never imports a vendor SDK directly. To add Bedrock
later you implement one method, ``complete()``, and swap the client in
dependencies.py — no other code changes.

For Module 2 we ship ``MockLLM``: a deterministic, offline stand-in. It returns
predictable text so tests are fast and reproducible and so the whole pipeline
is demoable without any cloud credentials.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


class LLMClient(ABC):
    """Minimal interface every concrete LLM backend must implement."""

    #: Stable identifier surfaced in traces (e.g. "bedrock:claude-3-sonnet").
    name: str = "abstract-llm"

    @abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Return a completion for ``prompt`` (optionally guided by ``system``)."""
        raise NotImplementedError


def _extract_question(prompt: str) -> str:
    """
    Pull the user question out of a structured prompt.

    Our prompts end with a 'QUESTION: ...' line (see rag.py). Parsing it keeps
    the mock's single complete() interface identical to what a real LLM sees,
    while still letting the mock echo something relevant.
    """
    for line in reversed(prompt.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("QUESTION:"):
            return stripped.split(":", 1)[1].strip()
    # Fallback: no marker found — return the whole prompt trimmed.
    return prompt.strip()


class MockLLM(LLMClient):
    """
    Deterministic offline generator.

    It detects grounded vs. direct mode by the presence of a 'CONTEXT:' block
    in the prompt (exactly how a real model would receive its context), and
    returns a fixed, assertable response. No randomness, no network.
    """

    name = "mock-llm-v0"

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        grounded = "CONTEXT:" in prompt
        question = _extract_question(prompt)
        logger.info("MockLLM.complete grounded=%s question=%r", grounded, question)

        if grounded:
            return (
                f'Based on the retrieved context, here is a grounded answer to: '
                f'"{question}". [Deterministic mock response from {self.name}. '
                f"Plug in AWS Bedrock or OpenAI to generate a real answer grounded "
                f"in the passages above.]"
            )
        return (
            f'Here is a direct answer to: "{question}". [Deterministic mock '
            f"response from {self.name}; no document retrieval was used.]"
        )
