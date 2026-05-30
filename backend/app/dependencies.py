"""
Shared application dependencies.

The RetrievalEngine owns the embedding model and the FAISS index, both of which
must persist across requests (we don't want to reload the model or lose the
index every call). So we hold a single process-wide instance and hand it to
route handlers via FastAPI's dependency-injection system.

The engine is created lazily on first access so that importing the app (e.g. in
tests that build their own engine) doesn't pay the model-load cost.
"""

from typing import Optional

from .logging_config import get_logger
from .services.generator import MockLLM
from .services.rag import RAGPipeline
from .services.retrieval import RetrievalEngine
from .services.router import QueryRouter

logger = get_logger(__name__)

_engine: Optional[RetrievalEngine] = None
_pipeline: Optional[RAGPipeline] = None


def get_engine() -> RetrievalEngine:
    """Return the process-wide RetrievalEngine, building it on first use."""
    global _engine
    if _engine is None:
        logger.info("Constructing global RetrievalEngine (first request).")
        _engine = RetrievalEngine()
    return _engine


def get_pipeline() -> RAGPipeline:
    """
    Return the process-wide RAGPipeline (Module 2), built on first use.

    Wires the existing RetrievalEngine to the MockLLM generator and the
    heuristic QueryRouter. To go live, swap MockLLM() for a Bedrock/OpenAI
    client here — nothing else changes.
    """
    global _pipeline
    if _pipeline is None:
        logger.info("Constructing global RAGPipeline (first request).")
        _pipeline = RAGPipeline(
            engine=get_engine(),
            generator=MockLLM(),
            router=QueryRouter(),
        )
    return _pipeline
