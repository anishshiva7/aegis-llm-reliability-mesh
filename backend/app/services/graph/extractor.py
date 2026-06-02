"""
Entity / relationship extraction (Module 10 — Part C).

``EntityExtractor`` is the seam the spec calls out: deterministic today, an
LLM-backed extractor tomorrow, with zero changes to the builder or retriever.
The deterministic implementation matches text against the seed ontology's
aliases — fast, offline, and good enough to make architecture queries work.

A future ``LLMEntityExtractor`` would implement the same ``extract`` method,
calling a provider to propose (entity, category) and (source, type, target)
triples, and the rest of the system would not change.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from .models import GraphNode, GraphRelationship
from .ontology import SEED_NODES, SEED_RELATIONSHIPS


@dataclass
class Extraction:
    """Entities (and any relationships) found in a span of text."""

    entities: List[GraphNode]
    relationships: List[GraphRelationship]


class EntityExtractor(ABC):
    """Pluggable extractor: text -> (entities, relationships)."""

    @abstractmethod
    def extract(self, text: str) -> Extraction:
        ...


def _alias_pattern(alias: str) -> re.Pattern:
    """Word-boundary, case-insensitive matcher for an alias phrase."""
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_])", re.IGNORECASE)


class DeterministicEntityExtractor(EntityExtractor):
    """Match ontology aliases in text; infer edges from co-occurrence.

    An entity is "mentioned" when any of its aliases appears as a whole token /
    phrase in the text. Relationships are the subset of the seed ontology whose
    *both* endpoints are mentioned together — so co-occurrence in a passage
    surfaces the real architectural edges between the things it discusses.
    """

    def __init__(self, nodes: List[GraphNode] | None = None,
                 relationships: List[GraphRelationship] | None = None) -> None:
        self._nodes = nodes if nodes is not None else SEED_NODES
        self._relationships = (
            relationships if relationships is not None else SEED_RELATIONSHIPS
        )
        # Precompile alias patterns once: (compiled, node).
        self._patterns = []
        for node in self._nodes:
            aliases = set(node.aliases) | {node.name.lower()}
            for alias in aliases:
                self._patterns.append((_alias_pattern(alias), node))

    def extract(self, text: str) -> Extraction:
        if not text or not text.strip():
            return Extraction([], [])

        matched: dict[str, GraphNode] = {}
        for pattern, node in self._patterns:
            if node.name in matched:
                continue
            if pattern.search(text):
                matched[node.name] = node

        names = set(matched)
        rels = [
            r for r in self._relationships
            if r.source in names and r.target in names
        ]
        # Deterministic ordering for stable tests.
        entities = sorted(matched.values(), key=lambda n: n.name)
        return Extraction(entities, rels)

    def match_query_entities(self, query: str) -> List[GraphNode]:
        """Entities a *query* refers to (same matcher, entities only)."""
        return self.extract(query).entities
