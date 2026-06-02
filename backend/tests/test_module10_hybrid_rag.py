"""
Tests for Module 10 — hybrid retrieval through the full RAGPipeline (Parts E–G).

Verifies the *additive* contract:
  * architecture/relationship queries activate HYBRID retrieval (vector + graph),
  * the hybrid prompt fuses VECTOR CONTEXT and GRAPH CONTEXT blocks,
  * graph linked-chunks become first-class grounding context,
  * plain vector RAG still works unchanged,
  * direct factual queries stay DIRECT_ANSWER / vector,
  * graph telemetry is recorded.

All offline: a FakeEngine stands in for FAISS, MockLLM for generation, and a
seeded InMemoryGraphStore for the graph.

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module10_hybrid_rag.py -v
    ./venv/bin/python tests/test_module10_hybrid_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import RetrievalMode, Route  # noqa: E402
from app.services.generator import MockLLM  # noqa: E402
from app.services.graph import (  # noqa: E402
    GraphMetrics,
    GraphRetriever,
    InMemoryGraphStore,
    KnowledgeGraphBuilder,
)
from app.services.rag import (  # noqa: E402
    RAGPipeline,
    build_hybrid_prompt,
)
from app.models.schemas import RetrievedContext  # noqa: E402
from app.services.graph.models import GraphSearchResult  # noqa: E402
from app.services.router import QueryRouter  # noqa: E402
from app.services.vector_store import ChunkRecord  # noqa: E402


def _record(text, source="doc.md", idx=0):
    return ChunkRecord(text=text, source=source, chunk_index=idx)


class FakeEngine:
    def __init__(self, hits=None, total_chunks=None):
        self._hits = hits or []
        self._total = total_chunks if total_chunks is not None else len(self._hits)

    def search(self, query, top_k=None):
        return self._hits[:top_k] if top_k else self._hits

    @property
    def total_chunks(self):
        return self._total


def _graph(ingest_text=None):
    store = InMemoryGraphStore()
    b = KnowledgeGraphBuilder(store)
    b.seed()
    if ingest_text:
        b.ingest_document(ingest_text, "doc.md")
    return store


def _hybrid_pipe(engine, store=None, metrics=None):
    store = store or _graph()
    return RAGPipeline(
        engine=engine,
        generator=MockLLM(),
        router=QueryRouter(graph_available=True),
        graph_retriever=GraphRetriever(store, max_hops=2),
        graph_metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Prompt construction (Part G)
# ---------------------------------------------------------------------------
def test_build_hybrid_prompt_has_both_blocks():
    store = _graph()
    gr = GraphRetriever(store, max_hops=2)
    res = gr.retrieve("How do routing and retries interact?")
    contexts = [RetrievedContext(chunk_id=0, text="vector passage", score=0.5,
                                 source="d", chunk_index=0)]
    prompt = build_hybrid_prompt("How do routing and retries interact?", contexts, res)
    assert "VECTOR CONTEXT:" in prompt
    assert "GRAPH CONTEXT:" in prompt
    assert "Entities:" in prompt
    assert "Relationships:" in prompt
    assert "vector passage" in prompt
    # The generator's grounding contract is still satisfied.
    assert prompt.startswith("CONTEXT:")
    assert prompt.rstrip().endswith("ANSWER:")


def test_build_hybrid_prompt_handles_empty_vector():
    res = GraphSearchResult()
    prompt = build_hybrid_prompt("q", [], res)
    assert "(no vector matches)" in prompt
    assert "(no graph matches)" in prompt


# ---------------------------------------------------------------------------
# Hybrid activation (Part F/E)
# ---------------------------------------------------------------------------
def test_architecture_query_activates_hybrid_even_empty_index():
    pipe = _hybrid_pipe(FakeEngine(hits=[], total_chunks=0))
    resp = pipe.ask("How do routing and retries interact?")
    t = resp.trace
    assert resp.route is Route.RAG_ANSWER
    assert t.retrieval_mode is RetrievalMode.HYBRID
    assert t.graph_used is True
    assert t.graph is not None
    names = {e.name for e in t.graph.matched_entities}
    assert {"QueryRouter", "RetryManager"} <= names


def test_hybrid_merges_graph_chunks_into_context():
    text = "The FallbackProvider fails over to the MockProvider when a vendor is down."
    store = _graph(ingest_text=text)
    hit = (0, 0.42, _record(text))
    pipe = _hybrid_pipe(FakeEngine(hits=[hit], total_chunks=1), store=store)
    resp = pipe.ask("How does the fallback provider connect to mock?")
    t = resp.trace
    assert t.retrieval_mode is RetrievalMode.HYBRID
    assert t.answer_context_used is True
    assert t.graph.graph_chunks  # graph surfaced a linked chunk
    assert resp.answer  # mock produced a grounded answer


def test_graph_metrics_recorded_on_hybrid():
    metrics = GraphMetrics()
    pipe = _hybrid_pipe(FakeEngine(hits=[], total_chunks=0), metrics=metrics)
    pipe.ask("How do routing and retries interact?")
    snap = metrics.snapshot()
    assert snap["graph_traversals"] >= 1
    assert snap["hybrid_queries"] >= 1


# ---------------------------------------------------------------------------
# Existing behaviour preserved
# ---------------------------------------------------------------------------
def test_plain_vector_rag_still_works():
    """A non-architecture query with a strong hit stays vector RAG."""
    hit = (0, 0.85, _record("Acme refunds within 30 days of purchase."))
    pipe = _hybrid_pipe(FakeEngine(hits=[hit], total_chunks=1))
    resp = pipe.ask("What is the refund window?")
    t = resp.trace
    assert resp.route is Route.RAG_ANSWER
    assert t.retrieval_mode is RetrievalMode.VECTOR
    assert t.graph_used is False
    assert t.graph is None


def test_direct_query_stays_direct_and_vector():
    pipe = _hybrid_pipe(FakeEngine(hits=[], total_chunks=0))
    resp = pipe.ask("What is the capital of France?")
    t = resp.trace
    assert resp.route is Route.DIRECT_ANSWER
    assert t.retrieval_mode is RetrievalMode.VECTOR
    assert t.graph_used is False


def test_pipeline_without_graph_retriever_is_vector_only():
    """A pipeline built without a graph retriever never goes hybrid."""
    hit = (0, 0.85, _record("some passage"))
    pipe = RAGPipeline(
        engine=FakeEngine(hits=[hit], total_chunks=1),
        generator=MockLLM(),
        router=QueryRouter(graph_available=False),
    )
    resp = pipe.ask("How do routing and retries interact?")
    assert resp.trace.retrieval_mode is RetrievalMode.VECTOR
    assert resp.trace.graph_used is False
    assert resp.trace.graph is None


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
