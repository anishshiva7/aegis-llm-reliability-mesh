"""
GraphStore factory (Module 10 — Part A/B).

The single seam where the backend choice is made — exactly like the LLMProvider
factory. Callers ask for ``build_graph_store()`` and never learn whether they got
Neo4j or the in-memory store.

Selection:
  * ``AEGIS_GRAPH_BACKEND=neo4j`` (and NEO4J_* configured) -> try Neo4j; on any
    connection failure, log and **fall back** to the in-memory store so Aegis
    keeps working.
  * ``AEGIS_GRAPH_BACKEND=memory`` (default) -> in-memory store.
"""

from __future__ import annotations

from ...config import get_settings
from ...logging_config import get_logger
from .base import GraphStore
from .memory_store import InMemoryGraphStore

logger = get_logger(__name__)


def build_graph_store() -> GraphStore:
    """Construct the configured GraphStore, with graceful Neo4j fallback."""
    settings = get_settings()
    backend = (settings.graph_backend or "memory").strip().lower()

    if backend == "neo4j":
        store = _try_neo4j(settings)
        if store is not None:
            return store
        logger.warning(
            "Graph backend 'neo4j' requested but unavailable; "
            "falling back to in-memory graph store."
        )

    logger.info("Using in-memory graph store (backend=memory).")
    return InMemoryGraphStore()


def _try_neo4j(settings) -> GraphStore | None:
    """Attempt to build + connect a Neo4jGraphStore; None on any failure."""
    try:
        from .neo4j_store import Neo4jGraphStore, Neo4jUnavailable
    except Exception as exc:  # pragma: no cover - import-time guard
        logger.warning("Neo4j store import failed: %s", exc)
        return None

    try:
        store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        return store.connect()
    except Neo4jUnavailable as exc:
        logger.warning("Neo4j unavailable: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unexpected Neo4j error: %s", exc)
        return None
