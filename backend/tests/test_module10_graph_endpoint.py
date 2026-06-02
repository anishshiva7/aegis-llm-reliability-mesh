"""
Tests for Module 10 — POST /graph/search endpoint.

Exercises the standalone graph-search HTTP handler directly (httpx/TestClient is
not installed in this environment, so we call the route function with hand-built
dependencies, exactly as FastAPI would after resolving Depends()).

The endpoint runs graph retrieval in isolation — no LLM, no FAISS — and returns
the public GraphSearchResponse (query, graph_used, graph: GraphTrace).

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module10_graph_endpoint.py -v
    ./venv/bin/python tests/test_module10_graph_endpoint.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import (  # noqa: E402
    GraphSearchRequest,
    GraphSearchResponse,
)
from app.routers.graph import graph_search  # noqa: E402
from app.services.graph import (  # noqa: E402
    GraphMetrics,
    GraphRetriever,
    InMemoryGraphStore,
    KnowledgeGraphBuilder,
)


def _deps(max_hops=2):
    store = InMemoryGraphStore()
    KnowledgeGraphBuilder(store).seed()
    retriever = GraphRetriever(store, max_hops=max_hops)
    return retriever, GraphMetrics()


def _call(query, **kw):
    retriever, metrics = _deps()
    req = GraphSearchRequest(query=query, **kw)
    return graph_search(req, retriever=retriever, metrics=metrics)


# ---------------------------------------------------------------------------
# Happy path: relationship query anchors entities and traverses
# ---------------------------------------------------------------------------
def test_returns_graph_search_response_shape():
    resp = _call("How do routing and retries interact?")
    assert isinstance(resp, GraphSearchResponse)
    assert resp.query == "How do routing and retries interact?"
    assert resp.graph_used is True
    assert resp.graph.graph_backend == "memory"


def test_matches_expected_anchor_entities():
    resp = _call("How do routing and retries interact?")
    names = {e.name for e in resp.graph.matched_entities}
    assert "QueryRouter" in names
    assert "RetryManager" in names


def test_traversal_surfaces_relationships_and_score():
    resp = _call("How do routing and retries interact?")
    g = resp.graph
    assert len(g.traversed_entities) > len(g.matched_entities)
    assert len(g.traversed_relationships) > 0
    assert 0.0 < g.graph_score <= 1.0
    assert g.hops == 2


# ---------------------------------------------------------------------------
# Per-call overrides flow through to the retriever
# ---------------------------------------------------------------------------
def test_max_hops_override_is_respected():
    resp = _call("How do routing and retries interact?", max_hops=1)
    assert resp.graph.hops == 1


# ---------------------------------------------------------------------------
# Miss: a query that matches no ontology entity returns graph_used=False
# ---------------------------------------------------------------------------
def test_unmatched_query_reports_unused():
    resp = _call("What is the weather in Paris today?")
    assert resp.graph_used is False
    assert resp.graph.matched_entities == []
    assert resp.graph.graph_score == 0.0


# ---------------------------------------------------------------------------
# Metrics: a search records a (non-hybrid) traversal
# ---------------------------------------------------------------------------
def test_search_records_traversal_metric():
    retriever, metrics = _deps()
    before = metrics.snapshot()["graph_traversals"]
    graph_search(
        GraphSearchRequest(query="How do routing and retries interact?"),
        retriever=retriever,
        metrics=metrics,
    )
    after = metrics.snapshot()
    assert after["graph_traversals"] == before + 1
    # Standalone search is not a hybrid query.
    assert after["hybrid_queries"] == 0


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
