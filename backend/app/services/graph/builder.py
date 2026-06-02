"""
KnowledgeGraphBuilder (Module 10 — Part C).

Runs *alongside* the existing FAISS ingestion path. While ``RetrievalEngine``
does ``text -> chunks -> embeddings -> FAISS``, the builder does
``text -> chunks -> entities -> relationships -> graph`` against the same chunk
boundaries, then links each chunk to the entities it mentions.

The seed ontology (Aegis's own architecture) is loaded once at construction, so
the graph can answer relationship/architecture questions even before any
document is ingested. Ingested documents *enrich* the graph; they never replace
the backbone.

Extraction is delegated to a pluggable ``EntityExtractor`` — deterministic today,
LLM-backed tomorrow — so swapping extraction strategy touches nothing here.
"""

from __future__ import annotations

from typing import List, Optional

from ...config import get_settings
from ...logging_config import get_logger
from ..chunker import chunk_text
from .base import GraphStore
from .extractor import DeterministicEntityExtractor, EntityExtractor
from .models import LinkedChunk
from .ontology import SEED_NODES, SEED_RELATIONSHIPS

logger = get_logger(__name__)


class KnowledgeGraphBuilder:
    """Populate a ``GraphStore`` from the seed ontology and ingested documents."""

    def __init__(
        self,
        store: GraphStore,
        extractor: Optional[EntityExtractor] = None,
    ) -> None:
        self.store = store
        self.extractor = extractor or DeterministicEntityExtractor()

    # ------------------------------------------------------------------ seed
    def seed(self) -> None:
        """Load Aegis's architecture graph (idempotent — backends MERGE)."""
        for node in SEED_NODES:
            self.store.add_node(node)
        for rel in SEED_RELATIONSHIPS:
            self.store.add_relationship(rel)
        logger.info(
            "Seeded knowledge graph: %d nodes, %d relationships",
            len(SEED_NODES),
            len(SEED_RELATIONSHIPS),
        )

    # --------------------------------------------------------------- ingest
    def ingest_document(
        self,
        text: str,
        source: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> dict:
        """Extract entities/relationships from a document and link its chunks.

        Mirrors ``RetrievalEngine.ingest`` chunking so chunk indices line up
        with the vector store. Returns a small stats dict for logging/telemetry.
        """
        if not text or not text.strip():
            return {"chunks": 0, "entities": 0, "relationships": 0}

        settings = get_settings()
        size = chunk_size or settings.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        chunks = chunk_text(text, chunk_size=size, chunk_overlap=overlap)

        entity_names: set = set()
        rel_keys: set = set()
        for index, chunk in enumerate(chunks):
            extraction = self.extractor.extract(chunk)
            if not extraction.entities:
                continue
            for node in extraction.entities:
                self.store.add_node(node)
                entity_names.add(node.name)
            for rel in extraction.relationships:
                self.store.add_relationship(rel)
                rel_keys.add((rel.source, rel.type, rel.target))
            self.store.link_chunk(
                LinkedChunk(
                    text=chunk,
                    source=source,
                    chunk_index=index,
                    entities=tuple(n.name for n in extraction.entities),
                )
            )

        stats = {
            "chunks": len(chunks),
            "entities": len(entity_names),
            "relationships": len(rel_keys),
        }
        logger.info(
            "Graph ingest: source=%r -> %d chunks, %d entities, %d relationships",
            source,
            stats["chunks"],
            stats["entities"],
            stats["relationships"],
        )
        return stats
