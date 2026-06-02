"""
Tests for Module 10 — dashboard/API contract for the Graph Trace panel (Part H).

The frontend renders a Graph Trace from the serialized RouteTrace. This locks the
JSON shape the dashboard depends on: retrieval_mode, graph_used, and a nested
``graph`` object with backend, matched/traversed entities, relationships, linked
chunks, score, hops, and latency. Also asserts backward compatibility — existing
vector/direct traces still serialize with the new fields at sensible defaults.

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module10_dashboard_contract.py -v
    ./venv/bin/python tests/test_module10_dashboard_contract.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import RetrievalMode, Route  # noqa: E402
from app.services.generator import MockLLM  # noqa: E402
from app.services.graph import (  # noqa: E402
    GraphRetriever,
    InMemoryGraphStore,
    KnowledgeGraphBuilder,
)
from app.services.rag import RAGPipeline  # noqa: E402
from app.services.router import QueryRouter  # noqa: E402
from app.services.vector_store import ChunkRecord  # noqa: E402


class FakeEngine:
    def __init__(self, hits=None, total_chunks=None):
        self._hits = hits or []
        self._total = total_chunks if total_chunks is not None else len(self._hits)

    def search(self, query, top_k=None):
        return self._hits[:top_k] if top_k else self._hits

    @property
    def total_chunks(self):
        return self._total


def _hybrid_pipe(engine):
    store = InMemoryGraphStore()
    KnowledgeGraphBuilder(store).seed()
    return RAGPipeline(
        engine=engine,
        generator=MockLLM(),
        router=QueryRouter(graph_available=True),
        graph_retriever=GraphRetriever(store, max_hops=2),
    )


def _ask_dump(query, engine=None):
    pipe = _hybrid_pipe(engine or FakeEngine(hits=[], total_chunks=0))
    resp = pipe.ask(query)
    return resp.model_dump()


# ---------------------------------------------------------------------------
# Hybrid trace shape
# ---------------------------------------------------------------------------
def test_trace_exposes_retrieval_mode_and_graph_used():
    dump = _ask_dump("How do routing and retries interact?")
    trace = dump["trace"]
    assert trace["retrieval_mode"] == "hybrid"
    assert trace["graph_used"] is True
    assert "graph" in trace and trace["graph"] is not None


def test_graph_trace_full_shape():
    dump = _ask_dump("How do routing and retries interact?")
    g = dump["trace"]["graph"]
    expected = {
        "graph_backend",
        "matched_entities",
        "traversed_entities",
        "traversed_relationships",
        "graph_chunks",
        "graph_score",
        "hops",
        "graph_latency_ms",
    }
    assert expected <= set(g)
    assert g["graph_backend"] == "memory"
    assert isinstance(g["matched_entities"], list)
    assert g["matched_entities"], "expected matched entities for an architecture query"


def test_graph_entity_and_relationship_shape():
    dump = _ask_dump("How do routing and retries interact?")
    g = dump["trace"]["graph"]
    ent = g["matched_entities"][0]
    assert set(ent) == {"name", "category", "description"}
    rel = g["traversed_relationships"][0]
    assert set(rel) == {"source", "type", "target"}


def test_graph_score_is_float_in_unit_interval():
    g = _ask_dump("How do routing and retries interact?")["trace"]["graph"]
    assert isinstance(g["graph_score"], float)
    assert 0.0 <= g["graph_score"] <= 1.0
    assert g["hops"] == 2


# ---------------------------------------------------------------------------
# Backward compatibility — vector/direct traces still valid
# ---------------------------------------------------------------------------
def test_direct_trace_defaults_are_serializable():
    dump = _ask_dump("What is the capital of France?")
    trace = dump["trace"]
    assert trace["retrieval_mode"] == "vector"
    assert trace["graph_used"] is False
    assert trace["graph"] is None


def test_vector_rag_trace_has_no_graph():
    hit = (0, 0.85, ChunkRecord(text="Acme refunds within 30 days.", source="faq", chunk_index=0))
    dump = _ask_dump("What is the refund window?", engine=FakeEngine(hits=[hit], total_chunks=1))
    trace = dump["trace"]
    assert trace["route"] == Route.RAG_ANSWER.value
    assert trace["retrieval_mode"] == "vector"
    assert trace["graph"] is None


def test_retrieval_mode_enum_values():
    assert {m.value for m in RetrievalMode} == {"vector", "graph", "hybrid"}


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
