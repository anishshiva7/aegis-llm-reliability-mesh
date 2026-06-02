"""
FastAPI application entry point for Aegis Module 1 (local retrieval engine).

Run locally with:
    cd backend
    ./venv/bin/uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive API docs.
"""

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import get_engine, get_graph_store
from .logging_config import configure_logging, get_logger
from .models.schemas import StatsResponse
from .routers import ask, graph, health as providers_health, ingest, metrics, search
from .services.retrieval import RetrievalEngine

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Aegis — Reliability Mesh",
    description=(
        "Module 1: local semantic retrieval (chunk -> embed -> FAISS -> search). "
        "Module 2: query router + basic RAG generation via POST /ask."
    ),
    version="0.2.0",
)

# CORS: the Next.js dashboard (Module 7) is a separate origin. No auth in this
# MVP, so we allow all origins; tighten via a real allow-list in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the feature routers. Each owns its own path(s).
app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(ask.router)
app.include_router(graph.router)
app.include_router(metrics.router)
app.include_router(providers_health.router)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — does not load the model."""
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats(engine: RetrievalEngine = Depends(get_engine)) -> StatsResponse:
    """Report current index size and embedding model details."""
    return StatsResponse(
        total_chunks=engine.total_chunks,
        embedding_model=engine.embedder.model_name,
        embedding_dim=engine.embedder.dim,
    )


@app.on_event("startup")
def _log_startup() -> None:
    # We intentionally do NOT build the RetrievalEngine here — the embedding
    # model (~80 MB) stays lazy and loads on the first /ingest or /search call.
    #
    # The graph store IS seeded eagerly: it is pure in-memory data structures
    # (no model download) and primes the graph-metric gauges so that
    # GET /metrics shows graph_nodes/graph_relationships > 0 immediately,
    # without requiring a prior /ask or /ingest call.
    logger.info("Aegis API started — seeding knowledge graph...")
    get_graph_store()   # builds store + seeds ontology + records_store_stats()
    logger.info("Aegis API ready.")
