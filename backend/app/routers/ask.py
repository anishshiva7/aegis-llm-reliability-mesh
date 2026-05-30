"""
/ask endpoint — the unified RAG entry point (Module 2).

Accepts a query, routes it (DIRECT_ANSWER / RAG_ANSWER / NEEDS_CLARIFICATION),
generates an answer via the pluggable generator, and returns the answer plus
optional trace metadata. All orchestration lives in RAGPipeline; this handler
just adapts HTTP <-> pipeline.
"""

from fastapi import APIRouter, Depends

from ..dependencies import get_pipeline
from ..logging_config import get_logger
from ..models.schemas import AskRequest, AskResponse
from ..services.rag import RAGPipeline

logger = get_logger(__name__)

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> AskResponse:
    """Route the query, generate an answer, and return it with a trace."""
    logger.info(
        "POST /ask query=%r top_k=%s force_route=%s",
        request.query, request.top_k, request.force_route,
    )
    return pipeline.ask(
        query=request.query,
        top_k=request.top_k,
        force_route=request.force_route,
        include_trace=request.include_trace,
    )
