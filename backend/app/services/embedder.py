"""
Embedding generation via sentence-transformers.

An "embedding" is a fixed-length vector of floats that captures the *meaning*
of a piece of text. Texts with similar meaning map to vectors that are close
together in this high-dimensional space, which is what makes semantic search
possible.

We wrap the SentenceTransformer model so that:
  * the model is loaded lazily (first use), keeping import time cheap, and
  * embeddings are L2-normalised, so a dot product between two vectors equals
    their cosine similarity — see vector_store.py for why that matters.
"""

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


class Embedder:
    """Thin wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or get_settings().embedding_model_name
        logger.info("Loading embedding model '%s' ...", self.model_name)
        # This downloads the model on first run (cached afterwards) and may
        # take a few seconds. Loading here means a cold first request, but a
        # warm model for everything after.
        self._model = SentenceTransformer(self.model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model ready (dim=%d).", self._dim
        )

    @property
    def dim(self) -> int:
        """Dimensionality of the produced vectors (e.g. 384 for MiniLM)."""
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of texts into a (len(texts), dim) float32 array.

        normalize_embeddings=True scales every vector to unit length so that
        the inner product used by the FAISS index equals cosine similarity.
        float32 is required by FAISS.
        """
        if not texts:
            return np.empty((0, self._dim), dtype="float32")

        logger.info("Embedding %d text(s) ...", len(texts))
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # FAISS only accepts contiguous float32 arrays.
        return np.ascontiguousarray(vectors, dtype="float32")
