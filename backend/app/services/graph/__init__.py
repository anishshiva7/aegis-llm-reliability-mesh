"""
GraphRAG package (Module 10).

A first-class, hybrid retrieval strategy inside Aegis: a knowledge graph of the
system's own architecture (the seed ontology), enriched by ingested documents,
traversed at query time and fused with FAISS vector retrieval.

Public surface:
  * Models      — GraphNode, GraphRelationship, LinkedChunk, GraphSearchResult
  * Stores      — GraphStore (ABC), InMemoryGraphStore, Neo4jGraphStore
  * Building    — KnowledgeGraphBuilder, EntityExtractor
  * Retrieval   — GraphRetriever
  * Telemetry   — GraphMetrics
  * Factory     — build_graph_store()
"""

from __future__ import annotations

from .base import GraphStore
from .builder import KnowledgeGraphBuilder
from .extractor import DeterministicEntityExtractor, EntityExtractor, Extraction
from .factory import build_graph_store
from .graph_metrics import GraphMetrics, get_graph_metrics
from .graph_retriever import GraphRetriever
from .memory_store import InMemoryGraphStore
from .models import (
    EntityCategory,
    GraphNode,
    GraphRelationship,
    GraphSearchResult,
    LinkedChunk,
)
from .ontology import SEED_NODES, SEED_RELATIONSHIPS

__all__ = [
    "GraphStore",
    "InMemoryGraphStore",
    "KnowledgeGraphBuilder",
    "EntityExtractor",
    "DeterministicEntityExtractor",
    "Extraction",
    "GraphRetriever",
    "GraphMetrics",
    "get_graph_metrics",
    "build_graph_store",
    "EntityCategory",
    "GraphNode",
    "GraphRelationship",
    "GraphSearchResult",
    "LinkedChunk",
    "SEED_NODES",
    "SEED_RELATIONSHIPS",
]
