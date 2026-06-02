"""
Tests for Module 10 — Neo4j integration + graceful fallback (Part B).

These run fully offline. We do not require a live Neo4j server: instead we test
(1) the backend-selection factory falls back to the in-memory store when Neo4j
is requested but unreachable, (2) the Cypher-independent logic of the Neo4j store
(relationship-type validation, record mapping, the Neo4jUnavailable contract),
and (3) that requesting the memory backend never imports the driver.

If a live Neo4j *is* configured (AEGIS_GRAPH_BACKEND=neo4j and reachable), the
optional live test exercises a real round-trip; otherwise it is skipped.

Run from backend/ directory:
    ./venv/bin/python -m pytest tests/test_module10_neo4j.py -v
    ./venv/bin/python tests/test_module10_neo4j.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.graph import InMemoryGraphStore, build_graph_store  # noqa: E402
from app.services.graph.factory import _try_neo4j  # noqa: E402
from app.services.graph.models import EntityCategory  # noqa: E402
from app.services.graph.neo4j_store import (  # noqa: E402
    Neo4jGraphStore,
    Neo4jUnavailable,
    _record_to_node,
    _safe_rel_type,
)


class _FakeSettings:
    """Minimal settings stand-in for the factory's Neo4j path."""

    graph_backend = "neo4j"
    neo4j_uri = "bolt://127.0.0.1:1"  # nothing listens here
    neo4j_username = "neo4j"
    neo4j_password = "neo4j"
    neo4j_database = "neo4j"
    graph_max_hops = 2


# ---------------------------------------------------------------------------
# Relationship-type safety (types are interpolated into Cypher)
# ---------------------------------------------------------------------------
def test_safe_rel_type_normalizes():
    assert _safe_rel_type("routes_to") == "ROUTES_TO"
    assert _safe_rel_type("fails over to") == "FAILS_OVER_TO"


def test_safe_rel_type_rejects_injection():
    # Note: plain words like "DROP DATABASE" normalize to the harmless label
    # DROP_DATABASE (a relationship *type* is just a name). What must be rejected
    # is anything carrying Cypher metacharacters or an illegal identifier shape.
    for bad in ("USES]->(x)//", "1BAD", "a-b", "x;y", "a(b)"):
        try:
            _safe_rel_type(bad)
            raise AssertionError(f"expected rejection for {bad!r}")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Record mapping (DB row -> domain node)
# ---------------------------------------------------------------------------
def test_record_to_node_maps_fields():
    data = {
        "name": "QueryRouter",
        "category": "Component",
        "description": "routes queries",
        "aliases": ["router", "routing"],
    }
    node = _record_to_node(data)
    assert node.name == "QueryRouter"
    assert node.category is EntityCategory.COMPONENT
    assert node.aliases == ("router", "routing")


def test_record_to_node_tolerates_unknown_category():
    node = _record_to_node({"name": "X", "category": "Bogus"})
    assert node.category is EntityCategory.DOCUMENT  # safe fallback


# ---------------------------------------------------------------------------
# Unavailable contract — operations before connect must fail clearly
# ---------------------------------------------------------------------------
def test_operations_before_connect_raise():
    store = Neo4jGraphStore("bolt://127.0.0.1:1", "u", "p")
    try:
        store.stats()
        raise AssertionError("expected Neo4jUnavailable")
    except Neo4jUnavailable:
        pass


def test_connect_to_dead_server_raises_unavailable():
    store = Neo4jGraphStore("bolt://127.0.0.1:1", "u", "p")
    try:
        store.connect()
        # If a driver isn't installed OR the server is dead, we expect failure.
        raise AssertionError("expected Neo4jUnavailable connecting to dead server")
    except Neo4jUnavailable:
        pass


# ---------------------------------------------------------------------------
# Factory fallback
# ---------------------------------------------------------------------------
def test_try_neo4j_returns_none_when_unreachable():
    assert _try_neo4j(_FakeSettings()) is None


def test_factory_falls_back_to_memory(monkeypatch=None):
    """AEGIS_GRAPH_BACKEND=neo4j with no server -> in-memory store, no crash."""
    prev = os.environ.get("AEGIS_GRAPH_BACKEND")
    prev_uri = os.environ.get("NEO4J_URI")
    os.environ["AEGIS_GRAPH_BACKEND"] = "neo4j"
    os.environ["NEO4J_URI"] = "bolt://127.0.0.1:1"
    try:
        # get_settings is cached; clear it so our env takes effect.
        from app.config import get_settings

        get_settings.cache_clear()
        store = build_graph_store()
        assert isinstance(store, InMemoryGraphStore)
        assert store.backend == "memory"
    finally:
        if prev is None:
            os.environ.pop("AEGIS_GRAPH_BACKEND", None)
        else:
            os.environ["AEGIS_GRAPH_BACKEND"] = prev
        if prev_uri is None:
            os.environ.pop("NEO4J_URI", None)
        else:
            os.environ["NEO4J_URI"] = prev_uri
        from app.config import get_settings

        get_settings.cache_clear()


def test_default_backend_is_memory():
    from app.config import get_settings

    get_settings.cache_clear()
    store = build_graph_store()
    assert store.backend == "memory"


# ---------------------------------------------------------------------------
# Optional live round-trip (skipped unless a real Neo4j is reachable)
# ---------------------------------------------------------------------------
def test_live_neo4j_roundtrip_if_available():
    if os.environ.get("AEGIS_GRAPH_BACKEND", "").lower() != "neo4j":
        print("  (skipped: AEGIS_GRAPH_BACKEND != neo4j)")
        return
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    store = Neo4jGraphStore(
        settings.neo4j_uri,
        settings.neo4j_username,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        store.connect()
    except Neo4jUnavailable:
        print("  (skipped: Neo4j configured but unreachable)")
        return
    from app.services.graph import KnowledgeGraphBuilder

    KnowledgeGraphBuilder(store).seed()
    stats = store.stats()
    assert stats["graph_nodes"] >= 30
    nodes, rels = store.traverse(["RetryManager"], max_hops=2)
    assert nodes and rels
    store.close()


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
