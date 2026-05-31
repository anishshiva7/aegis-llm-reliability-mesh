"""
Tests for Module 3 — LLM-as-a-Judge evaluation layer.

These tests exercise:
  - DeterministicJudge scoring for each route scenario
  - AnswerEvaluator threshold logic (should_retry)
  - Full pipeline integration (evaluation embedded in trace)
  - Backward-compat: all Module 2 traces still have an evaluation field

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module3.py -v
    ./venv/bin/python tests/test_module3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import RetrievedContext, Route  # noqa: E402
from app.services.evaluator import AnswerEvaluator  # noqa: E402
from app.services.generator import MockLLM  # noqa: E402
from app.services.judge import DeterministicJudge, JudgeInput  # noqa: E402
from app.services.rag import RAGPipeline  # noqa: E402
from app.services.vector_store import ChunkRecord  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ctx(text: str, score: float, source: str = "doc.txt", idx: int = 0) -> RetrievedContext:
    return RetrievedContext(chunk_id=0, text=text, score=score, source=source, chunk_index=idx)


def _record(text: str, source: str = "doc.txt", idx: int = 0) -> ChunkRecord:
    return ChunkRecord(text=text, source=source, chunk_index=idx)


def _judge_input(
    query: str,
    answer: str,
    route: Route,
    chunks: list,
    top_score,
) -> JudgeInput:
    return JudgeInput(
        query=query,
        answer=answer,
        route=route,
        retrieved_chunks=chunks,
        top_score=top_score,
    )


class FakeEngine:
    """Minimal RetrievalEngine stand-in (reused from test_module2 pattern)."""

    def __init__(self, hits=None, total_chunks=None):
        self._hits = hits or []
        self._total = total_chunks if total_chunks is not None else len(self._hits)

    def search(self, query, top_k=None):
        return self._hits[:top_k] if top_k else self._hits

    @property
    def total_chunks(self):
        return self._total


def _pipeline(engine):
    return RAGPipeline(engine=engine, generator=MockLLM())


# ---------------------------------------------------------------------------
# DeterministicJudge — unit tests
# ---------------------------------------------------------------------------

def test_strong_rag_answer_gets_high_scores():
    """Well-grounded RAG answer with strong retrieval → high overall, no retry."""
    grounded_answer = (
        "Based on the retrieved context, the Eiffel Tower is in Paris. "
        "[Deterministic mock response from mock-llm-v0.]"
    )
    chunks = [_ctx("The Eiffel Tower is in Paris, France.", score=0.87)]
    inp = _judge_input("Where is the Eiffel Tower?", grounded_answer, Route.RAG_ANSWER, chunks, 0.87)

    scores = DeterministicJudge().score(inp)

    assert scores.groundedness >= 0.80, f"Expected groundedness >= 0.80, got {scores.groundedness}"
    assert scores.hallucination_risk <= 0.25, f"Expected low risk, got {scores.hallucination_risk}"
    assert scores.overall_score >= 0.70, f"Expected overall >= 0.70, got {scores.overall_score}"


def test_rag_answer_with_no_chunks_gets_low_groundedness():
    """RAG route but no chunks retrieved → groundedness tanks."""
    inp = _judge_input(
        "Tell me about pricing",
        "Based on context, pricing is flexible.",
        Route.RAG_ANSWER,
        chunks=[],
        top_score=None,
    )
    scores = DeterministicJudge().score(inp)

    assert scores.groundedness <= 0.15, f"Expected low groundedness, got {scores.groundedness}"
    assert scores.hallucination_risk >= 0.70, f"Expected high risk, got {scores.hallucination_risk}"


def test_direct_answer_not_penalized_for_missing_retrieval():
    """
    A clean DIRECT_ANSWER must NOT be penalized just because a routing probe
    found weak/no retrieval (Module 8 — Part E). The probe score is irrelevant
    to a direct answer's quality: grounding is "not applicable", not "bad".
    """
    inp = _judge_input(
        "Explain quantum entanglement",
        "Here is a direct answer to: 'Explain quantum entanglement'.",
        Route.DIRECT_ANSWER,
        chunks=[],
        top_score=0.10,  # weak routing-probe signal — must NOT drag scores down
    )
    scores = DeterministicJudge().score(inp)

    # Hallucination risk is low-moderate, not automatically high.
    assert scores.hallucination_risk <= 0.50, (
        f"Direct answer should not be auto-flagged as fabrication, got "
        f"{scores.hallucination_risk}"
    )
    # Groundedness is neutral (not applicable), not punished.
    assert scores.groundedness >= 0.45, (
        f"Direct-answer groundedness should be neutral, got {scores.groundedness}"
    )
    # Relevance reflects answer presence, not the weak probe score.
    assert scores.relevance >= 0.60, (
        f"Direct-answer relevance should not track probe score, got "
        f"{scores.relevance}"
    )
    # Net effect: a clean direct answer clears the retry threshold.
    assert scores.overall_score >= 0.60, (
        f"Clean direct answer should pass, got overall={scores.overall_score}"
    )


def test_uncertainty_admission_lowers_hallucination_risk():
    """Answer that admits uncertainty → lower hallucination_risk, lower confidence."""
    inp = _judge_input(
        "What is the company revenue?",
        "I don't know the company revenue based on the provided context.",
        Route.RAG_ANSWER,
        chunks=[_ctx("Some unrelated text.", score=0.35)],
        top_score=0.35,
    )
    scores = DeterministicJudge().score(inp)

    assert scores.hallucination_risk <= 0.20, f"Honest uncertainty should lower risk, got {scores.hallucination_risk}"
    assert scores.confidence <= 0.50


def test_clarification_route_never_triggers_retry():
    """NEEDS_CLARIFICATION is a routing decision — evaluator must not flag it for retry."""
    evaluator = AnswerEvaluator()
    result = evaluator.evaluate(
        query="tell me more",
        answer="I need a bit more detail to answer confidently.",
        route=Route.NEEDS_CLARIFICATION,
        retrieved_chunks=[],
        top_score=None,
    )

    assert result.should_retry is False, "Clarification route must never trigger retry"


def test_low_confidence_triggers_should_retry():
    """Evaluate a scenario where confidence is below the floor → should_retry=True."""
    evaluator = AnswerEvaluator()
    # RAG with no chunks: groundedness=0.1, confidence will be low, overall low
    result = evaluator.evaluate(
        query="What does the doc say?",
        answer="Based on retrieved context, I cannot determine the answer.",
        route=Route.RAG_ANSWER,
        retrieved_chunks=[],  # no context → many scores tank
        top_score=None,
    )

    # With no chunks and RAG route, overall_score should be below 0.60 threshold
    assert result.should_retry is True, (
        f"Expected retry, got should_retry=False (overall={result.scores.overall_score:.3f})"
    )


def test_strong_answer_does_not_trigger_retry():
    """High-quality grounded answer → should_retry=False."""
    evaluator = AnswerEvaluator()
    grounded = "Based on the retrieved context, here is a grounded answer."
    result = evaluator.evaluate(
        query="Where is the Eiffel Tower?",
        answer=grounded,
        route=Route.RAG_ANSWER,
        retrieved_chunks=[_ctx("The Eiffel Tower is in Paris.", score=0.90)],
        top_score=0.90,
    )

    assert result.should_retry is False
    assert result.scores.overall_score >= 0.70


# ---------------------------------------------------------------------------
# Integration — full pipeline produces evaluation in trace
# ---------------------------------------------------------------------------

def test_trace_includes_evaluation_object():
    """End-to-end: RouteTrace must contain a populated evaluation field."""
    strong = [(3, 0.87, _record("The Eiffel Tower is in Paris.", "facts.txt", 3))]
    pipe = _pipeline(FakeEngine(hits=strong))
    resp = pipe.ask("Where is the Eiffel Tower?", include_trace=True)

    assert resp.trace is not None
    assert resp.trace.evaluation is not None
    eval_result = resp.trace.evaluation
    assert 0.0 <= eval_result.scores.overall_score <= 1.0
    assert isinstance(eval_result.should_retry, bool)
    assert eval_result.evaluation_reason  # non-empty string


def test_trace_evaluation_has_all_score_fields():
    """Evaluation scores object must expose every expected dimension."""
    pipe = _pipeline(FakeEngine(hits=[], total_chunks=0))
    resp = pipe.ask("What is 2+2?", include_trace=True)

    scores = resp.trace.evaluation.scores
    for field in ("relevance", "groundedness", "completeness", "hallucination_risk", "confidence", "overall_score"):
        val = getattr(scores, field)
        assert 0.0 <= val <= 1.0, f"{field} out of range: {val}"


def test_include_trace_false_still_evaluates_but_omits_from_response():
    """When include_trace=False the response trace is None, but no crash occurs."""
    pipe = _pipeline(FakeEngine(hits=[], total_chunks=0))
    resp = pipe.ask("Hello", include_trace=False)

    assert resp.trace is None   # evaluation is computed but not surfaced
    assert resp.answer           # answer still returned


# ---------------------------------------------------------------------------
# Backward compat — existing Module 2 scenarios still work
# ---------------------------------------------------------------------------

def test_module2_direct_answer_still_has_evaluation_in_trace():
    pipe = _pipeline(FakeEngine(hits=[], total_chunks=0))
    resp = pipe.ask("What is the capital of France?")
    assert resp.route is Route.DIRECT_ANSWER
    assert resp.trace.evaluation is not None


def test_module2_rag_answer_still_has_evaluation_in_trace():
    strong = [(3, 0.87, _record("Paris is the capital of France.", "facts.txt", 3))]
    pipe = _pipeline(FakeEngine(hits=strong))
    resp = pipe.ask("What is the capital of France?")
    assert resp.route is Route.RAG_ANSWER
    assert resp.trace.evaluation is not None
    assert resp.trace.evaluation.scores.groundedness > 0


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failures = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
