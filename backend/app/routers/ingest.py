"""
/ingest endpoints.

Two ways to ingest:
  * POST /ingest        — JSON body with raw text (IngestTextRequest)
  * POST /ingest/file   — multipart file upload (text files)

Both funnel into the same RetrievalEngine.ingest() pipeline.
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..dependencies import get_engine, get_graph_builder
from ..logging_config import get_logger
from ..models.schemas import IngestResponse, IngestTextRequest
from ..services.graph import get_graph_metrics
from ..services.retrieval import RetrievalEngine

logger = get_logger(__name__)

router = APIRouter(tags=["ingest"])


def _enrich_graph(
    text: str,
    source: str,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
) -> None:
    """Mirror the FAISS ingest into the knowledge graph (Module 10).

    Best-effort: a graph failure must never break document ingestion, so any
    error here is logged and swallowed. The vector path has already succeeded.
    """
    try:
        builder = get_graph_builder()
        builder.ingest_document(
            text=text,
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # Refresh the graph-size gauges exposed via /metrics.
        get_graph_metrics().record_store_stats(builder.store.stats())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Graph enrichment failed for source=%r: %s", source, exc)


@router.post("/ingest", response_model=IngestResponse)
def ingest_text(
    request: IngestTextRequest,
    engine: RetrievalEngine = Depends(get_engine),
) -> IngestResponse:
    """Ingest a raw text blob supplied in the JSON body."""
    # Fall back to a generic label if the caller didn't name the source.
    source = request.source or "raw_text"
    logger.info("POST /ingest source=%r (%d chars)", source, len(request.text))

    chunks_created = engine.ingest(
        text=request.text,
        source=source,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    _enrich_graph(request.text, source, request.chunk_size, request.chunk_overlap)
    return IngestResponse(
        source=source,
        chunks_created=chunks_created,
        total_chunks_in_index=engine.total_chunks,
    )


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(..., description="A UTF-8 text file."),
    # Optional multipart form overrides for chunking.
    chunk_size: Optional[int] = Form(default=None),
    chunk_overlap: Optional[int] = Form(default=None),
    engine: RetrievalEngine = Depends(get_engine),
) -> IngestResponse:
    """Ingest the contents of an uploaded text file."""
    raw = await file.read()
    try:
        # We only support text content in Module 1; decode strictly so binary
        # uploads fail loudly rather than producing garbage chunks.
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        logger.warning("Rejected non-UTF-8 upload %r: %s", file.filename, exc)
        raise HTTPException(
            status_code=400, detail="File must be UTF-8 encoded text."
        ) from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    source = file.filename or "uploaded_file"
    logger.info("POST /ingest/file source=%r (%d chars)", source, len(text))

    chunks_created = engine.ingest(
        text=text,
        source=source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    _enrich_graph(text, source, chunk_size, chunk_overlap)
    return IngestResponse(
        source=source,
        chunks_created=chunks_created,
        total_chunks_in_index=engine.total_chunks,
    )
