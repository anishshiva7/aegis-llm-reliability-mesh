"""
Tests for Module 10 — GraphStore abstraction + InMemoryGraphStore (Parts A/C).

Exercises the storage-agnostic operations every backend implements, using the
fully-offline ``InMemoryGraphStore``: node/relationship MERGE semantics, chunk
linking, deterministic multi-hop BFS (1-hop and 3-hop), chunk retrieval, stats,
and clear. The same contract is what ``Neo4jGraphStore`` upholds against a real
database, so green here means the retriever/builder logic is correct regardless
of backend.

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module10_graph_store.py -v
    ./venv/bin/python tests/test_module10_graph_store.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.graph import (  # noqa: E402
    InMemoryGraphStore,
    KnowledgeGraphBuilder,
)
from app.services.graph.models import (  # noqa: E402
    EntityCategory,
    GraphNode,
    GraphRelationship,
    LinkedChunk,
)


def _store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


def _node(name: str, category=EntityCategory.COMPONENT) -> GraphNode:
    return GraphNode(name, category, f"desc of {name}")


# ---------------------------------------------------------------------------
# Writes + idempotency
# ---------------------------------------------------------------------------
def test_add_node_and_get_node():
    s = _store()
    s.add_node(_node("A"))
    got = s.get_node("A")
    assert got is not None and got.name == "A"
    assert s.get_node("missing") is None


def test_add_node_is_merge_not_duplicate():
    s = _store()
    s.add_node(_node("A"))
    s.add_node(GraphNode("A", EntityCategory.ROUTE, "updated"))
    assert s.stats()["graph_nodes"] == 1
    assert s.get_node("A").category is EntityCategory.ROUTE


def test_add_relationship_dedupes():
    s = _store()
    s.add_node(_node("A"))
    s.add_node(_node("B"))
    s.add_relationship(GraphRelationship("A", "USES", "B"))
    s.add_relationship(GraphRelationship("A", "USES", "B"))  # duplicate
    assert s.stats()["graph_relationships"] == 1


def test_relationship_autocreates_stub_endpoints():
    """An edge whose endpoints arrive later must still be traversable."""
    s = _store()
    s.add_relationship(GraphRelationship("A", "USES", "B"))
    nodes, rels = s.traverse(["A"], max_hops=1)
    assert len(rels) == 1
    # Stub endpoints aren't full nodes until add_node is called.
    assert s.get_node("A") is None


def test_find_nodes_preserves_order_and_filters_missing():
    s = _store()
    for n in ("A", "B", "C"):
        s.add_node(_node(n))
    found = s.find_nodes(["C", "missing", "A"])
    assert [n.name for n in found] == ["C", "A"]


# ---------------------------------------------------------------------------
# Traversal (1-hop and 3-hop)
# ---------------------------------------------------------------------------
def _chain_store() -> InMemoryGraphStore:
    """A -> B -> C -> D -> E linear chain."""
    s = _store()
    for n in ("A", "B", "C", "D", "E"):
        s.add_node(_node(n))
    s.add_relationship(GraphRelationship("A", "TO", "B"))
    s.add_relationship(GraphRelationship("B", "TO", "C"))
    s.add_relationship(GraphRelationship("C", "TO", "D"))
    s.add_relationship(GraphRelationship("D", "TO", "E"))
    return s


def test_traverse_one_hop():
    s = _chain_store()
    nodes, rels = s.traverse(["A"], max_hops=1)
    names = {n.name for n in nodes}
    assert names == {"A", "B"}
    assert len(rels) == 1


def test_traverse_three_hops():
    s = _chain_store()
    nodes, _ = s.traverse(["A"], max_hops=3)
    names = {n.name for n in nodes}
    # A(0) -> B(1) -> C(2) -> D(3); E is 4 hops away and excluded.
    assert names == {"A", "B", "C", "D"}
    assert "E" not in names


def test_traverse_is_bidirectional():
    """'What connects to X' works whether X is a source or a target."""
    s = _chain_store()
    nodes, _ = s.traverse(["C"], max_hops=1)
    names = {n.name for n in nodes}
    assert names == {"B", "C", "D"}


def test_traverse_includes_seed_and_is_deterministic():
    s = _chain_store()
    first = [n.name for n in s.traverse(["A"], max_hops=2)[0]]
    second = [n.name for n in s.traverse(["A"], max_hops=2)[0]]
    assert first == second  # stable ordering
    assert first[0] == "A"  # seed first


def test_traverse_unknown_seed_returns_empty():
    s = _chain_store()
    nodes, rels = s.traverse(["nonexistent"], max_hops=2)
    assert nodes == [] and rels == []


# ---------------------------------------------------------------------------
# Linked chunks
# ---------------------------------------------------------------------------
def test_link_chunk_and_retrieve():
    s = _store()
    s.add_node(_node("A"))
    s.add_node(_node("B"))
    s.link_chunk(LinkedChunk("hello world", "doc.md", 0, ("A", "B")))
    a_chunks = s.chunks_for_entities(["A"])
    assert len(a_chunks) == 1 and a_chunks[0].text == "hello world"
    # Same chunk is reachable from either linked entity.
    assert s.chunks_for_entities(["B"])[0].source == "doc.md"


def test_chunks_dedup_and_limit():
    s = _store()
    s.add_node(_node("A"))
    for i in range(5):
        s.link_chunk(LinkedChunk(f"chunk {i}", "doc.md", i, ("A",)))
    # Duplicate link is ignored.
    s.link_chunk(LinkedChunk("chunk 0", "doc.md", 0, ("A",)))
    limited = s.chunks_for_entities(["A"], limit=3)
    assert len(limited) == 3
    assert s.stats()["linked_chunks"] == 5


# ---------------------------------------------------------------------------
# Stats + clear
# ---------------------------------------------------------------------------
def test_stats_shape():
    s = _store()
    s.add_node(_node("A"))
    s.add_node(_node("B"))
    s.add_relationship(GraphRelationship("A", "USES", "B"))
    s.link_chunk(LinkedChunk("t", "d", 0, ("A",)))
    stats = s.stats()
    assert set(stats) == {"graph_nodes", "graph_relationships", "linked_chunks"}
    assert stats == {"graph_nodes": 2, "graph_relationships": 1, "linked_chunks": 1}


def test_clear_resets_everything():
    s = _store()
    s.add_node(_node("A"))
    s.add_relationship(GraphRelationship("A", "USES", "A"))
    s.link_chunk(LinkedChunk("t", "d", 0, ("A",)))
    s.clear()
    assert s.stats() == {
        "graph_nodes": 0,
        "graph_relationships": 0,
        "linked_chunks": 0,
    }


# ---------------------------------------------------------------------------
# Seed ontology via the builder
# ---------------------------------------------------------------------------
def test_builder_seeds_architecture_graph():
    s = _store()
    KnowledgeGraphBuilder(s).seed()
    stats = s.stats()
    assert stats["graph_nodes"] >= 30
    assert stats["graph_relationships"] >= 30
    # Spot-check a couple of real Aegis edges exist after seeding.
    _, rels = s.traverse(["RetryManager"], max_hops=1)
    rel_pairs = {(r.source, r.type, r.target) for r in rels}
    assert ("RetryManager", "USES", "BudgetGuard") in rel_pairs


def test_builder_ingest_links_document_chunks():
    s = _store()
    b = KnowledgeGraphBuilder(s)
    b.seed()
    text = (
        "The QueryRouter probes FAISS and routes to RAG_ANSWER. "
        "The RetryManager uses query expansion when answers are weak."
    )
    result = b.ingest_document(text, "notes.md")
    assert result["chunks"] >= 1
    assert result["entities"] >= 2
    chunks = s.chunks_for_entities(["RetryManager"])
    assert any(c.source == "notes.md" for c in chunks)


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
