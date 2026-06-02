"""
/graph/search endpoint (Module 10 — GraphRAG).

Runs knowledge-graph retrieval in *isolation* — no LLM call, no FAISS vector
search. Given a natural-language query, it matches ontology entities, traverses
their neighbourhood within the hop budget, links back document chunks, and
returns the full GraphTrace. Useful for inspecting exactly what the graph would
contribute to a hybrid /ask, and for demoing the knowledge graph directly.
"""

from fastapi import APIRouter, Depends

from ..dependencies import get_graph_metrics, get_graph_retriever
from ..logging_config import get_logger
from ..models.schemas import GraphSearchRequest, GraphSearchResponse
from ..services.graph.graph_metrics import GraphMetrics
from ..services.graph.graph_retriever import GraphRetriever
from ..services.rag import build_graph_trace

logger = get_logger(__name__)

router = APIRouter(tags=["graph"])


@router.post("/graph/search", response_model=GraphSearchResponse)
def graph_search(
    request: GraphSearchRequest,
    retriever: GraphRetriever = Depends(get_graph_retriever),
    metrics: GraphMetrics = Depends(get_graph_metrics),
) -> GraphSearchResponse:
    """Traverse the knowledge graph for ``request.query`` and return the trace."""
    logger.info(
        "POST /graph/search query=%r max_hops=%s chunk_limit=%s",
        request.query,
        request.max_hops,
        request.chunk_limit,
    )

    result = retriever.retrieve(
        query=request.query,
        max_hops=request.max_hops,
        chunk_limit=request.chunk_limit,
    )

    # Record the traversal in metrics (standalone search is not a hybrid query).
    latency = retriever.last_latency_ms
    metrics.record_traversal(latency, hybrid=False)

    backend = getattr(getattr(retriever, "store", None), "backend", "memory")
    trace = build_graph_trace(result, backend=backend, latency_ms=latency)

    return GraphSearchResponse(
        query=request.query,
        graph_used=result.used,
        graph=trace,
    )
