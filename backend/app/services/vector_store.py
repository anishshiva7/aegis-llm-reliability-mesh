"""
FAISS vector index + in-memory metadata store.

FAISS (Facebook AI Similarity Search) is a library for storing dense vectors
and querying them by nearest-neighbour very quickly. We use ``IndexFlatIP``:

  * "Flat" = brute-force exact search (compares the query against every stored
    vector). Perfect for development and modest corpora; no training needed.
  * "IP"   = inner product as the similarity metric. Because we feed it
    *L2-normalised* vectors (see embedder.py), inner product == cosine
    similarity, which lives in [-1, 1] where 1 means "identical direction".

FAISS itself only stores vectors and returns integer row ids + scores. It has
no notion of the original text. So we keep a parallel Python list, ``_metadata``,
indexed by the same row id, holding the chunk text and its provenance. This is
the "in-memory metadata store" — it lives only for the process lifetime.
"""

import faiss
import numpy as np

from ..logging_config import get_logger

logger = get_logger(__name__)


class ChunkRecord:
    """Metadata for one stored chunk (parallel to a FAISS row)."""

    __slots__ = ("text", "source", "chunk_index")

    def __init__(self, text: str, source: str, chunk_index: int) -> None:
        self.text = text
        self.source = source
        # Position of this chunk within its originating document.
        self.chunk_index = chunk_index


class VectorStore:
    """An exact-search FAISS index paired with an in-memory metadata list."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        # IndexFlatIP: exact inner-product search over `dim`-dimensional vectors.
        self._index = faiss.IndexFlatIP(dim)
        # _metadata[i] describes the vector stored at FAISS row i. The two stay
        # in lock-step because we only ever append, never delete/reorder.
        self._metadata: list[ChunkRecord] = []
        logger.info("Initialised FAISS IndexFlatIP (dim=%d).", dim)

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self._index.ntotal

    def add(self, vectors: np.ndarray, records: list[ChunkRecord]) -> None:
        """
        Append a batch of vectors and their metadata.

        ``vectors`` must be shape (len(records), dim), float32. The i-th vector
        corresponds to the i-th record.
        """
        if len(records) == 0:
            return
        if vectors.shape[0] != len(records):
            raise ValueError("vectors and records length mismatch")
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected vectors of dim {self.dim}, got {vectors.shape[1]}"
            )

        self._index.add(vectors)
        self._metadata.extend(records)
        logger.info(
            "Added %d vectors to index (total now %d).", len(records), self.size
        )

    def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> list[tuple[int, float, ChunkRecord]]:
        """
        Return up to ``top_k`` nearest chunks to ``query_vector``.

        ``query_vector`` is shape (1, dim), float32. Returns a list of
        (chunk_id, score, record) tuples sorted by descending similarity.
        """
        if self.size == 0:
            logger.warning("Search called on an empty index.")
            return []

        # FAISS clamps internally, but cap top_k at the corpus size so we don't
        # ask for more neighbours than exist (it would pad with -1 ids).
        k = min(top_k, self.size)
        # scores: (1, k) similarities; ids: (1, k) row indices into _metadata.
        scores, ids = self._index.search(query_vector, k)

        results: list[tuple[int, float, ChunkRecord]] = []
        for chunk_id, score in zip(ids[0], scores[0]):
            if chunk_id == -1:  # FAISS uses -1 to mark "no result" slots.
                continue
            results.append((int(chunk_id), float(score), self._metadata[chunk_id]))

        logger.info("Search returned %d result(s) (requested top_k=%d).", len(results), top_k)
        return results
