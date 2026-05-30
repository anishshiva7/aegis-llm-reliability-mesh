"""
Text chunking.

Large documents must be split into smaller passages before embedding, because
(a) embedding models have a bounded input length and (b) retrieval is more
precise when each vector represents a focused span of text rather than a whole
document.

We use a *sliding window over words* with a configurable overlap:

    chunk_size    = words per chunk
    chunk_overlap = words shared between consecutive chunks

Overlap matters: a sentence that straddles a chunk boundary would otherwise be
split in two, and neither half might match a query well. Repeating the last N
words at the start of the next chunk preserves that cross-boundary context.
"""

from ..logging_config import get_logger

logger = get_logger(__name__)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split ``text`` into overlapping word-windows.

    Returns a list of chunk strings. An empty/whitespace-only input yields [].
    """
    if chunk_overlap >= chunk_size:
        # Caller error — without forward progress the window never advances.
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Tokenise on whitespace. Using words (not characters) keeps chunk
    # boundaries human-readable and avoids slicing through the middle of words.
    words = text.split()
    if not words:
        logger.warning("chunk_text received empty/whitespace-only input")
        return []

    chunks: list[str] = []
    # `step` is how far the window slides each iteration. With overlap, the
    # window advances by (chunk_size - chunk_overlap) words at a time.
    step = chunk_size - chunk_overlap

    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        # If this window already reached the end of the document, stop — any
        # further iteration would only produce a redundant tail chunk.
        if start + chunk_size >= len(words):
            break

    logger.info(
        "Chunked text: %d words -> %d chunks (size=%d, overlap=%d)",
        len(words),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
