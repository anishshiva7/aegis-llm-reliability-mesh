"""
Tests for Module 2 (query router + basic RAG).

These tests use a FakeEngine instead of the real RetrievalEngine so they run
fast and deterministically with no model download. The FakeEngine lets each
test dial in the retrieval scores the router will see, which is exactly the
signal routing decisions depend on.

Run from the backend/ directory:

    ./venv/bin/python -m pytest tests/test_module2.py -v
    # or, without pytest:
    ./venv/bin/python tests/test_module2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import Route  # noqa: E402
from app.services.generator import MockLLM  # noqa: E402
from app.services.rag import RAGPipeline  # noqa: E402
from app.services.vector_store import ChunkRecord  # noqa: E402


class FakeEngine:
    """
    Stand-in for RetrievalEngine.

    ``hits`` is a list of (chunk_id, score, ChunkRecord) tuples returned from
    search(). ``total_chunks`` reports the index size the router checks.
    """

    def __init__(self, hits=None, total_chunks=None):
        self._hits = hits or []
        # Default total to the number of hits unless explicitly overridden
        # (lets us simulate an empty index with total_chunks=0).
        self._total = total_chunks if total_chunks is not None else len(self._hits)

    def search(self, query, top_k=None):
        return self._hits[: top_k] if top_k else self._hits

    @property
    def total_chunks(self):
        return self._total


def _record(text, source="doc.txt", idx=0):
    return ChunkRecord(text=text, source=source, chunk_index=idx)


def _pipeline(engine):
    return RAGPipeline(engine=engine, generator=MockLLM())


# --------------------------------------------------------------------------- #
# DIRECT_ANSWER
# --------------------------------------------------------------------------- #
def test_direct_answer_when_no_documents():
    # Empty index -> nothing to ground on -> direct.
    pipe = _pipeline(FakeEngine(hits=[], total_chunks=0))
    resp = pipe.ask("What is the capital of France?")
    assert resp.route is Route.DIRECT_ANSWER
    assert resp.trace.retrieval_used is False
    assert resp.trace.generation_mode == "direct"
    assert resp.answer  # non-empty


def test_direct_answer_when_retrieval_is_weak():
    # Index has content but the best match is below the RAG threshold.
    weak = [(0, 0.12, _record("totally unrelated text about gardening"))]
    pipe = _pipeline(FakeEngine(hits=weak))
    resp = pipe.ask("Explain quantum entanglement in detail")
    assert resp.route is Route.DIRECT_ANSWER
    assert resp.trace.retrieval_used is True  # probe ran, but match too weak
    assert resp.trace.top_score == 0.12


# --------------------------------------------------------------------------- #
# RAG_ANSWER
# --------------------------------------------------------------------------- #
def test_rag_answer_on_strong_match():
    strong = [
        (3, 0.87, _record("The Eiffel Tower is located in Paris, France.", "facts.txt", 3)),
        (1, 0.40, _record("Python is a programming language.", "facts.txt", 1)),
    ]
    pipe = _pipeline(FakeEngine(hits=strong))
    resp = pipe.ask("Where is the Eiffel Tower?")
    assert resp.route is Route.RAG_ANSWER
    assert resp.trace.retrieval_used is True
    assert resp.trace.generation_mode == "grounded"
    assert resp.trace.top_score == 0.87
    # The retrieved context is surfaced in the trace, top hit first.
    assert resp.trace.retrieved[0].text.startswith("The Eiffel Tower")
    assert resp.trace.retrieved[0].source == "facts.txt"


# --------------------------------------------------------------------------- #
# NEEDS_CLARIFICATION
# --------------------------------------------------------------------------- #
def test_clarification_on_empty_query_is_rejected_by_schema():
    # The Pydantic model enforces min_length=1; here we test the router path
    # directly with a too-short query instead.
    pipe = _pipeline(FakeEngine(hits=[(0, 0.9, _record("x"))]))
    resp = pipe.ask("it")  # single vague pronoun
    assert resp.route is Route.NEEDS_CLARIFICATION
    assert resp.trace.retrieval_used is False
    assert resp.trace.generation_mode == "clarification"


def test_clarification_on_vague_phrase():
    pipe = _pipeline(FakeEngine(hits=[(0, 0.9, _record("x"))]))
    resp = pipe.ask("tell me more")
    assert resp.route is Route.NEEDS_CLARIFICATION


def test_clarification_when_doc_intent_but_no_relevant_context():
    # User explicitly asks about "the document" but nothing relevant is found.
    weak = [(0, 0.04, _record("irrelevant"))]
    pipe = _pipeline(FakeEngine(hits=weak))
    resp = pipe.ask("What does the uploaded document say about pricing?")
    assert resp.route is Route.NEEDS_CLARIFICATION
    assert resp.trace.retrieval_used is True


# --------------------------------------------------------------------------- #
# Forced route
# --------------------------------------------------------------------------- #
def test_force_route_overrides_heuristics():
    # A generic query that would normally be DIRECT, forced to RAG.
    strong = [(3, 0.87, _record("Grounding passage.", "f.txt", 3))]
    pipe = _pipeline(FakeEngine(hits=strong))
    resp = pipe.ask("Hello there", force_route=Route.RAG_ANSWER)
    assert resp.route is Route.RAG_ANSWER
    assert resp.trace.retrieval_used is True
    assert "forced" in resp.trace.reason.lower()
    assert resp.trace.retrieved  # forced RAG still retrieves context


def test_force_direct_skips_retrieval():
    strong = [(3, 0.87, _record("would-be context"))]
    pipe = _pipeline(FakeEngine(hits=strong))
    resp = pipe.ask("What does the document say?", force_route=Route.DIRECT_ANSWER)
    assert resp.route is Route.DIRECT_ANSWER
    assert resp.trace.retrieval_used is False


def test_include_trace_false_omits_trace():
    pipe = _pipeline(FakeEngine(hits=[], total_chunks=0))
    resp = pipe.ask("What is 2+2?", include_trace=False)
    assert resp.trace is None
    assert resp.answer


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
