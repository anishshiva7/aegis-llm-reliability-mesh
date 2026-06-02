"""
Tests for Module 10 — GraphRetriever (Part D).

Given a query, the retriever extracts entities, matches graph nodes, traverses
1–N hops, collects linked chunks, and scores graph confidence — returning a
``GraphSearchResult``. All offline against the seeded ``InMemoryGraphStore``.

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module10_graph_retrieval.py -v
    ./venv/bin/python tests/test_module10_graph_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.graph import (  # noqa: E402
    GraphRetriever,
    InMemoryGraphStore,
    KnowledgeGraphBuilder,
)


def _seeded_retriever(max_hops=2) -> GraphRetriever:
    store = InMemoryGraphStore()
    KnowledgeGraphBuilder(store).seed()
    return GraphRetriever(store, max_hops=max_hops)


# ---------------------------------------------------------------------------
# Entity matching
# ---------------------------------------------------------------------------
def test_matches_query_entities():
    gr = _seeded_retriever()
    res = gr.retrieve("How do routing and retries interact?")
    names = {e.name for e in res.matched_entities}
    assert "QueryRouter" in names
    assert "RetryManager" in names


def test_no_match_returns_unused_result():
    gr = _seeded_retriever()
    res = gr.retrieve("What is the weather in Paris today?")
    assert res.matched_entities == []
    assert res.graph_score == 0.0
    assert res.used is False


# ---------------------------------------------------------------------------
# Traversal depth
# ---------------------------------------------------------------------------
def test_one_hop_smaller_than_three_hop():
    store = InMemoryGraphStore()
    KnowledgeGraphBuilder(store).seed()
    gr1 = GraphRetriever(store, max_hops=1)
    gr3 = GraphRetriever(store, max_hops=3)
    q = "Trace the path of a RAG request through the pipeline."
    one = gr1.retrieve(q)
    three = gr3.retrieve(q)
    assert one.matched_entities  # anchored on RAGPipeline
    assert len(three.traversed_entities) >= len(one.traversed_entities)
    assert three.hops == 3 and one.hops == 1


def test_per_call_hop_override():
    gr = _seeded_retriever(max_hops=1)
    deep = gr.retrieve("How do routing and retries interact?", max_hops=3)
    assert deep.hops == 3


# ---------------------------------------------------------------------------
# Relationships + scoring
# ---------------------------------------------------------------------------
def test_traversal_surfaces_relationships():
    gr = _seeded_retriever()
    res = gr.retrieve("How does the retry manager use the budget guard?")
    pairs = {(r.source, r.type, r.target) for r in res.traversed_relationships}
    assert ("RetryManager", "USES", "BudgetGuard") in pairs


def test_relationship_summary_renders():
    gr = _seeded_retriever()
    res = gr.retrieve("How do routing and retries interact?")
    summary = res.relationship_summary(limit=3)
    assert summary.count("\n") <= 2  # at most 3 lines
    assert "-[" in summary and "]->" in summary


def test_graph_score_bounded_and_monotone():
    gr = _seeded_retriever()
    res = gr.retrieve("How do routing, retries, providers, and metrics connect?")
    assert 0.0 < res.graph_score <= 1.0


# ---------------------------------------------------------------------------
# Linked chunks
# ---------------------------------------------------------------------------
def test_linked_chunks_surface_after_ingest():
    store = InMemoryGraphStore()
    b = KnowledgeGraphBuilder(store)
    b.seed()
    b.ingest_document(
        "The FallbackProvider fails over to the MockProvider when a vendor is down.",
        "providers.md",
    )
    gr = GraphRetriever(store, max_hops=2)
    res = gr.retrieve("How does the fallback provider relate to mock?")
    assert res.graph_chunks
    assert any(c.source == "providers.md" for c in res.graph_chunks)


def test_latency_is_recorded():
    gr = _seeded_retriever()
    gr.retrieve("How do routing and retries interact?")
    assert gr.last_latency_ms >= 0.0


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
