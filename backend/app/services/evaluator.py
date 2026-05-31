"""
AnswerEvaluator — orchestrates the evaluation pass after generation.

Responsibilities:
  1. Package the answer + routing context into a JudgeInput.
  2. Delegate scoring to a JudgeClient (DeterministicJudge by default).
  3. Apply config-driven thresholds to produce the should_retry flag.
  4. Return a fully populated EvaluationResult ready to embed in RouteTrace.

This layer is intentionally thin: all scoring logic lives in judge.py so
the two concerns stay testable independently.
"""

from typing import List, Optional

from ..config import get_settings
from ..logging_config import get_logger
from ..models.schemas import EvaluationResult, EvaluationScores, RetrievedContext, Route
from .judge import DeterministicJudge, JudgeClient, JudgeInput

logger = get_logger(__name__)


def _build_reason(scores: EvaluationScores, should_retry: bool) -> str:
    """Produce a human-readable summary of why the evaluation landed where it did."""
    if not should_retry:
        if scores.overall_score >= 0.80:
            return (
                "Strong retrieval match and answer appears well grounded in retrieved context."
            )
        return (
            f"Answer meets quality thresholds (overall={scores.overall_score:.2f}). "
            "No retry needed."
        )

    reasons: List[str] = []
    settings = get_settings()

    if scores.overall_score < settings.retry_threshold:
        reasons.append(
            f"overall_score {scores.overall_score:.2f} is below retry threshold "
            f"{settings.retry_threshold:.2f}"
        )
    if scores.hallucination_risk > settings.hallucination_threshold:
        reasons.append(
            f"hallucination_risk {scores.hallucination_risk:.2f} exceeds threshold "
            f"{settings.hallucination_threshold:.2f}"
        )
    if scores.confidence < settings.confidence_floor:
        reasons.append(
            f"confidence {scores.confidence:.2f} is below floor "
            f"{settings.confidence_floor:.2f}"
        )

    return "Retry recommended: " + "; ".join(reasons) + "."


def _should_retry(scores: EvaluationScores, route: Route) -> bool:
    """
    NEEDS_CLARIFICATION is a routing decision, not a quality failure — never retry it.
    For all other routes, apply the three-way threshold check.
    """
    if route is Route.NEEDS_CLARIFICATION:
        return False

    settings = get_settings()
    return (
        scores.overall_score < settings.retry_threshold
        or scores.hallucination_risk > settings.hallucination_threshold
        or scores.confidence < settings.confidence_floor
    )


class AnswerEvaluator:
    """
    Wraps a JudgeClient and converts its raw scores into a full EvaluationResult.

    Usage:
        evaluator = AnswerEvaluator()          # uses DeterministicJudge
        result = evaluator.evaluate(
            query="...", answer="...", route=Route.RAG_ANSWER,
            retrieved_chunks=[...], top_score=0.87,
        )
    """

    def __init__(self, judge: Optional[JudgeClient] = None) -> None:
        self.judge = judge or DeterministicJudge()
        logger.info("AnswerEvaluator ready (judge=%s).", self.judge.name)

    def evaluate(
        self,
        query: str,
        answer: str,
        route: Route,
        retrieved_chunks: List[RetrievedContext],
        top_score: Optional[float],
    ) -> EvaluationResult:
        inp = JudgeInput(
            query=query,
            answer=answer,
            route=route,
            retrieved_chunks=retrieved_chunks,
            top_score=top_score,
        )
        scores = self.judge.score(inp)
        retry = _should_retry(scores, route)
        reason = _build_reason(scores, retry)

        logger.info(
            "Evaluation route=%s overall=%.3f should_retry=%s",
            route,
            scores.overall_score,
            retry,
        )
        return EvaluationResult(
            scores=scores,
            should_retry=retry,
            evaluation_reason=reason,
        )
