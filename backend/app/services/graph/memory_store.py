"""
InMemoryGraphStore (Module 10 — Part A).

A dependency-free, fully-deterministic GraphStore used for offline tests, the
default demo, and as the graceful fallback when Neo4j is not configured or
unreachable. Adjacency is kept in plain dicts; traversal is a deterministic BFS.

It is intentionally feature-compatible with ``Neo4jGraphStore`` so the entire
test suite runs without a database, yet exercises the same retriever/builder
code paths that run against Neo4j in production.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Dict, Iterable, List, Optional

from ...logging_config import get_logger
from .base import GraphStore
from .models import GraphNode, GraphRelationship, LinkedChunk

logger = get_logger(__name__)


class InMemoryGraphStore(GraphStore):
    backend = "memory"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._nodes: Dict[str, GraphNode] = {}
        # adjacency[name] = list of (relationship) where name is the source
        self._out: Dict[str, List[GraphRelationship]] = {}
        self._in: Dict[str, List[GraphRelationship]] = {}
        self._rels: List[GraphRelationship] = []
        self._rel_keys: set = set()
        # entity name -> linked chunks
        self._chunks: Dict[str, List[LinkedChunk]] = {}
        self._chunk_keys: set = set()

    # -- writes ------------------------------------------------------------
    def add_node(self, node: GraphNode) -> None:
        with self._lock:
            self._nodes[node.name] = node
            self._out.setdefault(node.name, [])
            self._in.setdefault(node.name, [])

    def add_relationship(self, rel: GraphRelationship) -> None:
        with self._lock:
            key = (rel.source, rel.type, rel.target)
            if key in self._rel_keys:
                return
            # Tolerate edges whose endpoints arrive later: auto-create stubs.
            for endpoint in (rel.source, rel.target):
                if endpoint not in self._nodes:
                    self._out.setdefault(endpoint, [])
                    self._in.setdefault(endpoint, [])
            self._rel_keys.add(key)
            self._rels.append(rel)
            self._out.setdefault(rel.source, []).append(rel)
            self._in.setdefault(rel.target, []).append(rel)

    def link_chunk(self, chunk: LinkedChunk) -> None:
        with self._lock:
            key = (chunk.source, chunk.chunk_index)
            for entity in chunk.entities:
                bucket = self._chunks.setdefault(entity, [])
                if (entity, key) not in self._chunk_keys:
                    bucket.append(chunk)
                    self._chunk_keys.add((entity, key))

    # -- reads -------------------------------------------------------------
    def get_node(self, name: str) -> Optional[GraphNode]:
        with self._lock:
            return self._nodes.get(name)

    def find_nodes(self, names: Iterable[str]) -> List[GraphNode]:
        with self._lock:
            return [self._nodes[n] for n in names if n in self._nodes]

    def traverse(
        self, seeds: Iterable[str], max_hops: int = 2
    ) -> tuple[List[GraphNode], List[GraphRelationship]]:
        with self._lock:
            seeds = [s for s in seeds if s in self._out or s in self._nodes]
            visited: Dict[str, int] = {s: 0 for s in seeds}
            order: List[str] = list(seeds)
            rels: List[GraphRelationship] = []
            rel_seen: set = set()
            q: deque = deque((s, 0) for s in seeds)

            while q:
                name, depth = q.popleft()
                if depth >= max_hops:
                    continue
                # Expand both directions so "what connects to X" works either way.
                for rel in self._out.get(name, []) + self._in.get(name, []):
                    rkey = (rel.source, rel.type, rel.target)
                    if rkey not in rel_seen:
                        rel_seen.add(rkey)
                        rels.append(rel)
                    nxt = rel.target if rel.source == name else rel.source
                    if nxt not in visited:
                        visited[nxt] = depth + 1
                        order.append(nxt)
                        q.append((nxt, depth + 1))

            entities = [self._nodes[n] for n in order if n in self._nodes]
            return entities, rels

    def chunks_for_entities(
        self, names: Iterable[str], limit: int = 5
    ) -> List[LinkedChunk]:
        with self._lock:
            out: List[LinkedChunk] = []
            seen: set = set()
            for name in names:
                for chunk in self._chunks.get(name, []):
                    key = (chunk.source, chunk.chunk_index)
                    if key not in seen:
                        seen.add(key)
                        out.append(chunk)
                        if len(out) >= limit:
                            return out
            return out

    def stats(self) -> Dict[str, int]:
        with self._lock:
            linked = sum(len(v) for v in self._chunks.values())
            return {
                "graph_nodes": len(self._nodes),
                "graph_relationships": len(self._rels),
                "linked_chunks": linked,
            }

    def clear(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._out.clear()
            self._in.clear()
            self._rels.clear()
            self._rel_keys.clear()
            self._chunks.clear()
            self._chunk_keys.clear()
