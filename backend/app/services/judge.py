"""
Judge abstraction — the LLM-as-a-Judge evaluation interface.

JudgeClient is the interface every judge implementation must satisfy.
DeterministicJudge is the offline heuristic implementation for Module 3.

Swap in BedrockJudge or OpenAIJudge later by implementing JudgeClient and
passing the new instance to AnswerEvaluator — nothing else changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from ..models.schemas import EvaluationScores, RetrievedContext, Route
from ..logging_config import get_logger

logger = get_logger(__name__)


# Phrases that indicate an answer is admitting uncertainty rather than
# hallucinating a confident wrong answer. Honest uncertainty is a *good* signal.
_UNCERTAINTY_PHRASES = (
    "i don't know",
    "i'm not sure",
    "i cannot",
    "i can't",
    "unclear",
    "insufficient",
    "not enough information",
    "no information",
    "context does not",
    "context doesn't",
)

# Phrases present in answers that drew on retrieved passages.
_GROUNDED_PHRASES = (
    "based on the retrieved",
    "based on the provided",
    "according to",
    "the context",
    "retrieved context",
    "provided context",
)


@dataclass
class JudgeInput:
    """All signals available to a judge implementation."""

    query: str
    answer: str
    route: Route
    retrieved_chunks: List[RetrievedContext]
    top_score: Optional[float]  # best cosine similarity seen; None if no retrieval


class JudgeClient(ABC):
    """Minimal interface every judge backend must implement."""

    name: str = "abstract-judge"

    @abstractmethod
    def score(self, inp: JudgeInput) -> EvaluationScores:
        """Return dimension scores for the given answer.  All values in [0, 1]."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Heuristic helpers (private — used only by DeterministicJudge)
# ---------------------------------------------------------------------------

def _admits_uncertainty(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in _UNCERTAINTY_PHRASES)


def _uses_retrieved_context(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in _GROUNDED_PHRASES)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_relevance(
    route: Route, top_score: Optional[float], answer: str
) -> float:
    """How on-topic is the answer?  Proxied by retrieval similarity for RAG."""
    if route is Route.NEEDS_CLARIFICATION:
        # We chose not to answer — neutral, not a failure.
        return 0.50
    if route is Route.DIRECT_ANSWER:
        # A direct answer's relevance reflects whether a substantive answer was
        # produced for a well-formed query — NOT retrieval similarity. The router
        # may have run a probe (low score for e.g. "What is 2+2?"), but that probe
        # score is irrelevant to the quality of a direct, ungrounded answer.
        if _admits_uncertainty(answer):
            return 0.55
        return 0.70 if answer.strip() else 0.30
    if top_score is None:
        # No retrieval was attempted (direct path with empty index).
        return 0.60
    # Cosine scores from sentence-transformers are in [-1, 1]; clamp to [0, 1].
    return _clamp(top_score)


def _score_groundedness(
    route: Route,
    retrieved_chunks: List[RetrievedContext],
    top_score: Optional[float],
    answer: str,
) -> float:
    """Is the answer anchored to retrieved evidence?"""
    if route is Route.NEEDS_CLARIFICATION:
        return 0.50  # not applicable

    if route is Route.DIRECT_ANSWER:
        # Direct answers carry no retrieval grounding *by design* — grounding is
        # not applicable, so we score it neutrally rather than penalising it.
        # A direct factual/conversational answer ("2+2 is 4") is not "ungrounded
        # and therefore bad"; grounding simply doesn't apply.
        return 0.50

    # RAG_ANSWER path
    if not retrieved_chunks:
        # Claimed grounded but nothing was retrieved — worst case.
        return 0.10

    base = _clamp(top_score or 0.0)
    # Bonus when the answer text explicitly references the context it was given.
    bonus = 0.10 if _uses_retrieved_context(answer) else 0.0
    return _clamp(base + bonus)


def _score_completeness(answer: str, route: Route) -> float:
    """Does the answer fully address the query (length heuristic)?"""
    if route is Route.NEEDS_CLARIFICATION:
        return 0.50  # clarification is intentionally brief

    length = len(answer.strip())
    if length < 20:
        return 0.25  # suspiciously short
    if length < 80:
        return 0.60
    return 0.85


def _score_hallucination_risk(
    route: Route,
    retrieved_chunks: List[RetrievedContext],
    top_score: Optional[float],
    answer: str,
) -> float:
    """Probability the answer contains fabricated content."""
    if route is Route.NEEDS_CLARIFICATION:
        # We deferred — no claim made, no fabrication risk.
        return 0.15

    if _admits_uncertainty(answer):
        # The model flagged its own uncertainty — low fabrication risk.
        return 0.15

    if route is Route.DIRECT_ANSWER:
        # A direct answer is not automatically a fabrication risk just because no
        # retrieved context was used. The router deliberately chose the direct
        # path (a greeting, a simple factual/math question, or a query with no
        # relevant documents), so we assign a low-moderate baseline rather than
        # punishing the absence of grounding it never needed.
        return 0.30

    # RAG_ANSWER path
    if not retrieved_chunks:
        # Said it was grounded but had nothing to ground on.
        return 0.75

    # Risk falls as retrieval quality rises.
    risk = _clamp(1.0 - (top_score or 0.0))
    # If answer explicitly references context, fabrication is less likely.
    if _uses_retrieved_context(answer):
        risk = _clamp(risk - 0.15)
    return risk


def _score_confidence(
    route: Route,
    top_score: Optional[float],
    answer: str,
) -> float:
    """How confident should we be in this answer's overall quality?"""
    if route is Route.NEEDS_CLARIFICATION:
        # We explicitly said we don't have enough to answer.
        return 0.30

    if _admits_uncertainty(answer):
        return 0.40

    if route is Route.RAG_ANSWER and top_score is not None:
        # Linearly scale: top_score=0.30 → 0.65, top_score=1.0 → 1.0
        return _clamp(0.50 + top_score * 0.50)

    # DIRECT_ANSWER with no retrieval signal — a clean, route-appropriate direct
    # answer should be trusted, not treated as a coin-flip.
    return 0.65


def _overall_score(scores: EvaluationScores) -> float:
    """
    Weighted composite.  Hallucination risk is inverted (high risk = bad).

    Weights reflect the priorities of a reliability-focused system:
      groundedness and hallucination safety are most critical;
      completeness and raw relevance are secondary.
    """
    return _clamp(
        scores.relevance * 0.20
        + scores.groundedness * 0.25
        + scores.completeness * 0.15
        + (1.0 - scores.hallucination_risk) * 0.25
        + scores.confidence * 0.15
    )


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------

class DeterministicJudge(JudgeClient):
    """
    Offline heuristic judge — no LLM calls, fully deterministic.

    Uses route type, retrieval scores, answer length, and linguistic signals
    to produce calibrated scores.  Replace with BedrockJudge or OpenAIJudge
    (both must implement JudgeClient) when real LLM evaluation is needed.
    """

    name = "deterministic-judge-v1"

    def score(self, inp: JudgeInput) -> EvaluationScores:
        relevance = _score_relevance(inp.route, inp.top_score, inp.answer)
        groundedness = _score_groundedness(
            inp.route, inp.retrieved_chunks, inp.top_score, inp.answer
        )
        completeness = _score_completeness(inp.answer, inp.route)
        hallucination_risk = _score_hallucination_risk(
            inp.route, inp.retrieved_chunks, inp.top_score, inp.answer
        )
        confidence = _score_confidence(inp.route, inp.top_score, inp.answer)

        partial = EvaluationScores(
            relevance=round(relevance, 3),
            groundedness=round(groundedness, 3),
            completeness=round(completeness, 3),
            hallucination_risk=round(hallucination_risk, 3),
            confidence=round(confidence, 3),
            overall_score=0.0,  # filled below after all dimensions are known
        )
        overall = _overall_score(partial)
        logger.debug(
            "DeterministicJudge route=%s top_score=%s overall=%.3f",
            inp.route,
            inp.top_score,
            overall,
        )
        return EvaluationScores(
            **{**partial.model_dump(), "overall_score": round(overall, 3)}
        )
