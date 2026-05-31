"""
Tests for Module 8 — retrieval-probe vs. answer-context trace semantics (Part F).

The RouteTrace now carries two precise booleans instead of the single ambiguous
``retrieval_used``:

  * retrieval_probe_used — the router searched FAISS to *decide* the route.
    May be True even for DIRECT_ANSWER (a weak-match probe still ran).
  * answer_context_used — retrieved chunks were actually injected into the
    answer prompt. True only on the grounded RAG path that had chunks.

Matrix:
  DIRECT (empty index)   -> probe False, context False, retrieved []
  DIRECT (weak probe)    -> probe True,  context False, retrieved []
  RAG                    -> probe True,  context True,  retrieved non-empty
  NEEDS_CLARIFICATION    -> probe may be True (doc-intent), context False

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module8_trace_semantics.py -v
    ./venv/bin/python tests/test_module8_trace_semantics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import Route  # noqa: E402
from app.services.generator import MockLLM  # noqa: E402
from app.services.rag import RAGPipeline  # noqa: E402
from app.services.vector_store import ChunkRecord  # noqa: E402


def _record(text: str) -> ChunkRecord:
    return ChunkRecord(text=text, source="d", chunk_index=0)


class FakeEngine:
    def __init__(self, hits=None, total_chunks=None):
        self._hits = hits or []
        self._total = total_chunks if total_chunks is not None else len(self._hits)

    def search(self, query, top_k=None):
        return self._hits[:top_k] if top_k else self._hits

    @property
    def total_chunks(self):
        return self._total


class ScriptedEngine:
    def __init__(self, total_chunks, score):
        self._total = total_chunks
        self._score = score

    def search(self, query, top_k=None):
        return [(0, self._score, _record("filler passage used only for probing"))]

    @property
    def total_chunks(self):
        return self._total


def _pipe(engine):
    return RAGPipeline(engine=engine, generator=MockLLM())


# ---------------------------------------------------------------------------
# Schema default sanity
# ---------------------------------------------------------------------------
def test_route_trace_has_new_boolean_fields():
    resp = _pipe(FakeEngine(hits=[], total_chunks=0)).ask("What is 2+2?")
    assert hasattr(resp.trace, "retrieval_probe_used")
    assert hasattr(resp.trace, "answer_context_used")
    assert isinstance(resp.trace.retrieval_probe_used, bool)
    assert isinstance(resp.trace.answer_context_used, bool)


# ---------------------------------------------------------------------------
# DIRECT — empty index: no probe ran at all
# ---------------------------------------------------------------------------
def test_direct_empty_index_no_probe_no_context():
    resp = _pipe(FakeEngine(hits=[], total_chunks=0)).ask("Tell me a joke please")
    assert resp.route is Route.DIRECT_ANSWER
    assert resp.trace.retrieval_probe_used is False
    assert resp.trace.answer_context_used is False
    assert resp.trace.retrieved == []


# ---------------------------------------------------------------------------
# DIRECT — weak probe: probe ran, but chunks were NOT used to answer
# ---------------------------------------------------------------------------
def test_direct_weak_probe_used_but_no_answer_context():
    resp = _pipe(ScriptedEngine(total_chunks=5, score=0.12)).ask(
        "What is your favourite colour"
    )
    assert resp.route is Route.DIRECT_ANSWER
    assert resp.trace.retrieval_probe_used is True
    assert resp.trace.answer_context_used is False
    assert resp.trace.retrieved == []


# ---------------------------------------------------------------------------
# RAG — both probe and answer-context are true; chunks present
# ---------------------------------------------------------------------------
def test_rag_uses_probe_and_answer_context():
    strong = [(3, 0.87, _record("Paris is the capital of France."))]
    resp = _pipe(FakeEngine(hits=strong)).ask("What is the capital of France?")
    assert resp.route is Route.RAG_ANSWER
    assert resp.trace.retrieval_probe_used is True
    assert resp.trace.answer_context_used is True
    assert len(resp.trace.retrieved) >= 1


# ---------------------------------------------------------------------------
# NEEDS_CLARIFICATION (doc-intent, no relevant context): probe ran, no context
# ---------------------------------------------------------------------------
def test_clarification_doc_intent_probe_without_context():
    resp = _pipe(ScriptedEngine(total_chunks=5, score=0.03)).ask(
        "What does the document say about pricing?"
    )
    assert resp.route is Route.NEEDS_CLARIFICATION
    assert resp.trace.retrieval_probe_used is True
    assert resp.trace.answer_context_used is False


# ---------------------------------------------------------------------------
# Backward-compat: legacy retrieval_used stays consistent with the probe flag
# ---------------------------------------------------------------------------
def test_legacy_retrieval_used_matches_probe_flag():
    resp = _pipe(ScriptedEngine(total_chunks=5, score=0.12)).ask("Say something nice")
    assert resp.trace.retrieval_used == resp.trace.retrieval_probe_used


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
