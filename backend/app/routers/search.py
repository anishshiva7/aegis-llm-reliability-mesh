"""
/search endpoint.

Embeds the query, runs nearest-neighbour lookup over the FAISS index, and
returns the top-k chunks with their cosine-similarity scores and provenance.
"""

from fastapi import APIRouter, Depends

from ..dependencies import get_engine
from ..logging_config import get_logger
from ..models.schemas import SearchRequest, SearchResponse, SearchResult
from ..services.retrieval import RetrievalEngine

logger = get_logger(__name__)

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    engine: RetrievalEngine = Depends(get_engine),
) -> SearchResponse:
    """Return the most semantically similar chunks to ``request.query``."""
    logger.info("POST /search query=%r top_k=%s", request.query, request.top_k)

    hits = engine.search(query=request.query, top_k=request.top_k)

    # Map internal (id, score, record) tuples onto the public response model.
    results = [
        SearchResult(
            chunk_id=chunk_id,
            text=record.text,
            score=score,
            source=record.source,
            chunk_index=record.chunk_index,
        )
        for chunk_id, score, record in hits
    ]
    return SearchResponse(query=request.query, results=results)
