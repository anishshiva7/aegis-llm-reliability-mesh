"""
Neo4jGraphStore (Module 10 — Part B).

The production GraphStore backend. It speaks Cypher to a real Neo4j instance and
is feature-compatible with ``InMemoryGraphStore`` — the retriever, builder, and
pipeline cannot tell which one they hold.

Design notes
------------
* The ``neo4j`` driver is imported *lazily* inside ``connect`` so the dependency
  stays optional: nothing in the offline test/demo path imports it.
* ``connect`` is best-effort. If the driver is missing or the server is
  unreachable, it raises ``Neo4jUnavailable`` and the factory falls back to the
  in-memory store — Aegis keeps answering, just without a persistent graph.
* All entities live under the ``:Entity`` label with a ``category`` property and
  a ``name`` uniqueness constraint, so MERGE is idempotent. Document chunks are
  ``:Chunk`` nodes joined to entities by ``[:MENTIONS]``.
* Relationship *types* come from our own controlled ontology vocabulary, so it
  is safe to interpolate them into the Cypher string (parameters cannot
  parameterise a relationship type in Cypher). We still validate the token.
"""

from __future__ import annotations

import re
import threading
from typing import Dict, Iterable, List, Optional

from ...logging_config import get_logger
from .base import GraphStore
from .models import EntityCategory, GraphNode, GraphRelationship, LinkedChunk

logger = get_logger(__name__)

# Relationship types are interpolated into Cypher (the language cannot bind them
# as parameters), so constrain them to a safe identifier shape defensively.
_REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class Neo4jUnavailable(RuntimeError):
    """Raised when the driver is missing or the server cannot be reached."""


def _safe_rel_type(rel_type: str) -> str:
    token = rel_type.strip().upper().replace(" ", "_")
    if not _REL_TYPE_RE.match(token):
        raise ValueError(f"unsafe relationship type: {rel_type!r}")
    return token


class Neo4jGraphStore(GraphStore):
    backend = "neo4j"

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._driver = None
        self._lock = threading.RLock()

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> "Neo4jGraphStore":
        """Open the driver, verify connectivity, and ensure schema.

        Raises ``Neo4jUnavailable`` on any failure so the factory can fall back.
        """
        try:
            from neo4j import GraphDatabase  # lazy: keeps the dep optional
            from neo4j.exceptions import Neo4jError, ServiceUnavailable
        except Exception as exc:  # pragma: no cover - exercised only without driver
            raise Neo4jUnavailable(
                "neo4j driver not installed (pip install neo4j)"
            ) from exc

        try:
            self._driver = GraphDatabase.driver(
                self._uri, auth=(self._username, self._password)
            )
            self._driver.verify_connectivity()
            self._ensure_schema()
        except (ServiceUnavailable, Neo4jError, OSError) as exc:
            self.close()
            raise Neo4jUnavailable(f"cannot reach Neo4j at {self._uri}: {exc}") from exc
        logger.info("Connected to Neo4j at %s (db=%s)", self._uri, self._database)
        return self

    def _ensure_schema(self) -> None:
        """Create the uniqueness constraint + indexes used by reads."""
        stmts = [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX entity_category IF NOT EXISTS "
            "FOR (e:Entity) ON (e.category)",
            "CREATE INDEX chunk_key IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.source, c.chunk_index)",
        ]
        with self._driver.session(database=self._database) as session:
            for stmt in stmts:
                session.run(stmt)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _session(self):
        if self._driver is None:
            raise Neo4jUnavailable("Neo4j driver not connected")
        return self._driver.session(database=self._database)

    # -- writes ------------------------------------------------------------
    def add_node(self, node: GraphNode) -> None:
        category = (
            node.category.value
            if isinstance(node.category, EntityCategory)
            else str(node.category)
        )
        with self._session() as session:
            session.run(
                "MERGE (e:Entity {name: $name}) "
                "SET e.category = $category, e.description = $description, "
                "e.aliases = $aliases",
                name=node.name,
                category=category,
                description=node.description,
                aliases=list(node.aliases),
            )

    def add_relationship(self, rel: GraphRelationship) -> None:
        rel_type = _safe_rel_type(rel.type)
        with self._session() as session:
            session.run(
                "MERGE (s:Entity {name: $source}) "
                "MERGE (t:Entity {name: $target}) "
                f"MERGE (s)-[r:{rel_type}]->(t)",
                source=rel.source,
                target=rel.target,
            )

    def link_chunk(self, chunk: LinkedChunk) -> None:
        with self._session() as session:
            session.run(
                "MERGE (c:Chunk {source: $source, chunk_index: $chunk_index}) "
                "SET c.text = $text "
                "WITH c "
                "UNWIND $entities AS ename "
                "MATCH (e:Entity {name: ename}) "
                "MERGE (e)-[:MENTIONS]->(c)",
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                entities=list(chunk.entities),
            )

    # -- reads -------------------------------------------------------------
    def get_node(self, name: str) -> Optional[GraphNode]:
        with self._session() as session:
            record = session.run(
                "MATCH (e:Entity {name: $name}) RETURN e", name=name
            ).single()
            return _record_to_node(record["e"]) if record else None

    def find_nodes(self, names: Iterable[str]) -> List[GraphNode]:
        names = list(names)
        with self._session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.name IN $names RETURN e", names=names
            )
            found = {r["e"]["name"]: _record_to_node(r["e"]) for r in result}
        # Preserve caller order.
        return [found[n] for n in names if n in found]

    def traverse(
        self, seeds: Iterable[str], max_hops: int = 2
    ) -> tuple[List[GraphNode], List[GraphRelationship]]:
        seeds = list(seeds)
        if not seeds:
            return [], []
        hops = max(1, int(max_hops))
        # Variable-length undirected expansion up to ``hops`` away from any seed.
        # We collect the reachable nodes and the relationships among them.
        cypher = (
            f"MATCH (s:Entity) WHERE s.name IN $seeds "
            f"MATCH path = (s)-[*0..{hops}]-(n:Entity) "
            "WITH collect(DISTINCT n) AS nodes "
            "UNWIND nodes AS n "
            "OPTIONAL MATCH (n)-[r]->(m:Entity) WHERE m IN nodes "
            "RETURN nodes, collect(DISTINCT r) AS rels"
        )
        with self._session() as session:
            record = session.run(cypher, seeds=seeds).single()
        if record is None:
            return [], []

        nodes = [_record_to_node(n) for n in record["nodes"]]
        # Seeds first, then the rest, for a deterministic, readable ordering.
        seed_set = set(seeds)
        nodes.sort(key=lambda n: (n.name not in seed_set, n.name))

        rels: List[GraphRelationship] = []
        seen: set = set()
        for r in record["rels"]:
            if r is None:
                continue
            edge = GraphRelationship(
                source=r.start_node["name"],
                type=r.type,
                target=r.end_node["name"],
            )
            key = (edge.source, edge.type, edge.target)
            if key not in seen:
                seen.add(key)
                rels.append(edge)
        rels.sort(key=lambda e: (e.source, e.type, e.target))
        return nodes, rels

    def chunks_for_entities(
        self, names: Iterable[str], limit: int = 5
    ) -> List[LinkedChunk]:
        names = list(names)
        if not names:
            return []
        with self._session() as session:
            result = session.run(
                "MATCH (e:Entity)-[:MENTIONS]->(c:Chunk) "
                "WHERE e.name IN $names "
                "WITH DISTINCT c "
                "OPTIONAL MATCH (allE:Entity)-[:MENTIONS]->(c) "
                "RETURN c.text AS text, c.source AS source, "
                "c.chunk_index AS chunk_index, "
                "collect(DISTINCT allE.name) AS entities "
                "ORDER BY source, chunk_index "
                "LIMIT $limit",
                names=names,
                limit=int(limit),
            )
            return [
                LinkedChunk(
                    text=r["text"],
                    source=r["source"],
                    chunk_index=r["chunk_index"],
                    entities=tuple(r["entities"]),
                )
                for r in result
            ]

    def stats(self) -> Dict[str, int]:
        with self._session() as session:
            record = session.run(
                "MATCH (e:Entity) "
                "WITH count(e) AS nodes "
                "OPTIONAL MATCH (:Entity)-[r]->(:Entity) "
                "WITH nodes, count(r) AS rels "
                "OPTIONAL MATCH (:Entity)-[m:MENTIONS]->(:Chunk) "
                "RETURN nodes, rels, count(m) AS linked"
            ).single()
        return {
            "graph_nodes": record["nodes"] if record else 0,
            "graph_relationships": record["rels"] if record else 0,
            "linked_chunks": record["linked"] if record else 0,
        }

    def clear(self) -> None:
        with self._session() as session:
            session.run("MATCH (n) WHERE n:Entity OR n:Chunk DETACH DELETE n")


def _record_to_node(data) -> GraphNode:
    """Map a Neo4j node record to our domain ``GraphNode``."""
    category_raw = data.get("category", EntityCategory.DOCUMENT.value)
    try:
        category = EntityCategory(category_raw)
    except ValueError:  # pragma: no cover - defensive against legacy data
        category = EntityCategory.DOCUMENT
    aliases = data.get("aliases") or []
    return GraphNode(
        name=data["name"],
        category=category,
        description=data.get("description", ""),
        aliases=tuple(aliases),
    )
