"""
RetrievalEngine — orchestrates the full retrieval pipeline.

This is the single object the API talks to. It wires together the three
lower-level services and exposes two operations:

    ingest(text, source, ...)  ->  chunk -> embed -> store
    search(query, top_k)       ->  embed query -> nearest-neighbour lookup

Keeping the orchestration here (rather than in the HTTP layer) means the
business logic is testable without spinning up FastAPI.
"""

from typing import Optional

from ..config import get_settings
from ..logging_config import get_logger
from .chunker import chunk_text
from .embedder import Embedder
from .vector_store import ChunkRecord, VectorStore

logger = get_logger(__name__)


class RetrievalEngine:
    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        settings = get_settings()
        # Allow injecting a pre-built Embedder (useful in tests); otherwise
        # build one from config. The embedder load is the slow part.
        self.embedder = embedder or Embedder(settings.embedding_model_name)
        # The store's dimensionality must match the embedder's output dim.
        self.store = VectorStore(dim=self.embedder.dim)
        logger.info("RetrievalEngine ready.")

    # ------------------------------------------------------------------ ingest
    def ingest(
        self,
        text: str,
        source: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> int:
        """
        Ingest one document. Returns the number of chunks created.

        Pipeline: split into chunks -> embed each chunk -> append to the index
        alongside its metadata (source + position).
        """
        settings = get_settings()
        size = chunk_size or settings.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

        logger.info("Ingest start: source=%r (size=%d, overlap=%d)", source, size, overlap)

        chunks = chunk_text(text, chunk_size=size, chunk_overlap=overlap)
        if not chunks:
            logger.warning("Ingest produced no chunks for source=%r", source)
            return 0

        vectors = self.embedder.encode(chunks)
        # Build a metadata record per chunk, tracking its order in the document.
        records = [
            ChunkRecord(text=chunk, source=source, chunk_index=i)
            for i, chunk in enumerate(chunks)
        ]
        self.store.add(vectors, records)

        logger.info("Ingest done: source=%r -> %d chunks", source, len(chunks))
        return len(chunks)

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: Optional[int] = None):
        """
        Embed ``query`` and return the top_k most similar chunks.

        Returns a list of (chunk_id, score, ChunkRecord) tuples.
        """
        k = top_k or get_settings().default_top_k
        logger.info("Search start: query=%r (top_k=%d)", query, k)

        # The query is embedded with the exact same model as the documents, so
        # query and chunk vectors live in the same space and are comparable.
        query_vector = self.embedder.encode([query])
        results = self.store.search(query_vector, top_k=k)

        logger.info("Search done: query=%r -> %d hits", query, len(results))
        return results

    # ------------------------------------------------------------------- stats
    @property
    def total_chunks(self) -> int:
        return self.store.size
