"""
Tests for Module 8 — DIRECT_ANSWER evaluation fix (Part E).

Before Module 8 the DeterministicJudge penalised DIRECT_ANSWER for the low
retrieval-probe score the router happened to see (e.g. "What is 2+2?" probes the
index, finds nothing relevant, and was then scored as if that low similarity
meant a bad answer). That made simple factual/math/conversational questions look
like failures and sometimes degraded them.

Module 8 decouples direct-answer quality from the routing probe:
  * relevance reflects answer presence / route appropriateness, NOT probe score
  * groundedness is neutral ("not applicable"), not punished
  * hallucination risk is low-moderate, not automatically high
  * a clean direct answer clears the retry threshold (should_retry=False)

RAG grounding behaviour is intentionally LEFT UNCHANGED — a RAG answer with no
chunks still tanks, and a document-intent query with no relevant context can
still clarify/retry rather than confidently fabricate.

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module8_evaluator_direct.py -v
    ./venv/bin/python tests/test_module8_evaluator_direct.py
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
# Helpers
# ---------------------------------------------------------------------------
def _ctx(text: str, score: float) -> RetrievedContext:
    return RetrievedContext(chunk_id=0, text=text, score=score, source="d", chunk_index=0)


def _record(text: str) -> ChunkRecord:
    return ChunkRecord(text=text, source="d", chunk_index=0)


class FakeEngine:
    """Returns scripted hits; total_chunks controls whether a probe runs."""

    def __init__(self, hits=None, total_chunks=None):
        self._hits = hits or []
        self._total = total_chunks if total_chunks is not None else len(self._hits)

    def search(self, query, top_k=None):
        return self._hits[:top_k] if top_k else self._hits

    @property
    def total_chunks(self):
        return self._total


class ScriptedEngine:
    """Always returns one hit at a fixed score, so route hinges on that score."""

    def __init__(self, total_chunks, score):
        self._total = total_chunks
        self._score = score

    def search(self, query, top_k=None):
        return [(0, self._score, _record("unrelated filler text for probing only"))]

    @property
    def total_chunks(self):
        return self._total


def _direct_judge(top_score):
    """
    Score a clean direct answer with a given probe top_score.

    Uses the actual MockLLM output so the judge-unit tests stay in sync with
    the mock's current response format without needing manual updates.
    """
    from app.services.generator import MockLLM  # noqa: PLC0415

    answer = MockLLM().complete(f"QUESTION: What is 2+2?\nANSWER:")
    return DeterministicJudge().score(
        JudgeInput(
            query="What is 2+2?",
            answer=answer,
            route=Route.DIRECT_ANSWER,
            retrieved_chunks=[],
            top_score=top_score,
        )
    )


# ---------------------------------------------------------------------------
# Unit: judge no longer penalises DIRECT for the probe score
# ---------------------------------------------------------------------------
def test_direct_answer_score_independent_of_probe_score():
    """A weak vs. absent probe score must not change a direct answer's verdict."""
    weak = _direct_judge(top_score=0.05)
    none = _direct_judge(top_score=None)

    # The weak probe must NOT drag relevance/groundedness/risk down vs no probe.
    assert weak.relevance == none.relevance
    assert weak.groundedness == none.groundedness
    assert weak.hallucination_risk == none.hallucination_risk
    assert weak.confidence == none.confidence


def test_direct_answer_groundedness_is_neutral_not_bad():
    s = _direct_judge(top_score=0.05)
    assert 0.45 <= s.groundedness <= 0.6, f"expected neutral groundedness, got {s.groundedness}"


def test_direct_answer_hallucination_not_automatically_high():
    s = _direct_judge(top_score=0.05)
    assert s.hallucination_risk <= 0.5, f"expected low/moderate risk, got {s.hallucination_risk}"


def test_clean_direct_answer_clears_thresholds():
    """relevance>=0.60, hallucination<=0.70, confidence>=0.40, overall>=0.60."""
    from app.services.generator import MockLLM  # noqa: PLC0415

    answer = MockLLM().complete("QUESTION: What is 2+2?\nANSWER:")
    ev = AnswerEvaluator().evaluate(
        query="What is 2+2?",
        answer=answer,
        route=Route.DIRECT_ANSWER,
        retrieved_chunks=[],
        top_score=0.05,
    )
    assert ev.scores.overall_score >= 0.60
    assert ev.scores.confidence >= 0.40
    assert ev.scores.hallucination_risk <= 0.70
    assert ev.should_retry is False


# ---------------------------------------------------------------------------
# Integration: pipeline behaviour for the canonical cases in the spec
# ---------------------------------------------------------------------------
def test_what_is_2_plus_2_routes_direct_and_passes():
    """'What is 2+2?' → DIRECT_ANSWER, should_retry=false, not degraded."""
    pipe = RAGPipeline(engine=FakeEngine(hits=[], total_chunks=0), generator=MockLLM())
    resp = pipe.ask("What is 2+2?")

    assert resp.route is Route.DIRECT_ANSWER
    assert resp.trace.evaluation.should_retry is False
    assert resp.trace.retry is None  # no self-healing needed


def test_direct_answer_with_weak_probe_still_passes():
    """A direct route reached via a weak probe (chunks indexed) still passes cleanly."""
    pipe = RAGPipeline(engine=ScriptedEngine(total_chunks=5, score=0.12), generator=MockLLM())
    resp = pipe.ask("What is your favourite colour")

    assert resp.route is Route.DIRECT_ANSWER
    assert resp.trace.retrieval_probe_used is True  # a probe DID run
    assert resp.trace.evaluation.should_retry is False
    # A non-empty direct answer must not be marked degraded.
    degraded = resp.trace.retry.degraded_response if resp.trace.retry else False
    assert degraded is False
    assert resp.answer.strip()


def test_rag_answer_still_relies_on_grounding():
    """RAG route with NO chunks must still tank (grounding behaviour unchanged)."""
    scores = DeterministicJudge().score(
        JudgeInput(
            query="What does the document say about pricing?",
            answer="Based on the retrieved context, pricing is flexible.",
            route=Route.RAG_ANSWER,
            retrieved_chunks=[],
            top_score=None,
        )
    )
    assert scores.groundedness <= 0.15
    assert scores.hallucination_risk >= 0.70


def test_document_intent_with_no_relevant_context_clarifies():
    """A doc-intent query with no relevant passages clarifies — it doesn't fabricate."""
    pipe = RAGPipeline(engine=ScriptedEngine(total_chunks=5, score=0.04), generator=MockLLM())
    resp = pipe.ask("What does the document say about the refund policy?")

    assert resp.route is Route.NEEDS_CLARIFICATION
    # Clarification is a deliberate, safe outcome — never a retry storm.
    assert resp.trace.evaluation.should_retry is False


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
