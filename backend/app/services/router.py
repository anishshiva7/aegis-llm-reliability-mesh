"""
Query router.

Decides how each incoming query should be handled:

  * DIRECT_ANSWER       — general/simple query; answer without retrieval.
  * RAG_ANSWER          — query is likely answerable from ingested documents.
  * NEEDS_CLARIFICATION — query is empty, too short, vague, or has no usable
                          grounding and clearly expected some.

Strategy: cheap structural heuristics first (no retrieval), then a single
retrieval *probe* whose top similarity score is the deciding signal for
DIRECT vs RAG. The probe's hits are returned in the decision so the RAG stage
can reuse them instead of querying the index a second time.

Extension seam: ``QueryRouter`` accepts an optional ``classifier`` callable.
Today it's None (pure heuristics). Later you can pass an LLM-backed classifier
(e.g. a Bedrock call returning a Route) without changing any caller.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Tuple

from ..config import get_settings
from ..logging_config import get_logger
from ..models.schemas import Route

logger = get_logger(__name__)

# A retrieval hit as returned by RetrievalEngine.search().
Hit = Tuple[int, float, object]  # (chunk_id, score, ChunkRecord)


class SupportsRetrieval(Protocol):
    """Structural type for whatever the router probes (the RetrievalEngine)."""

    def search(self, query: str, top_k: Optional[int] = None) -> List[Hit]: ...

    @property
    def total_chunks(self) -> int: ...


# Tokens that, on their own, carry no answerable content.
_VAGUE_TOKENS = {
    "it", "this", "that", "they", "them", "these", "those",
    "more", "why", "how", "what", "huh", "ok", "okay",
}
# Whole-query phrases that are inherently underspecified.
_VAGUE_PHRASES = {
    "tell me more", "explain", "explain that", "go on", "continue",
    "what about it", "and", "so", "why not", "what do you mean",
}
# Smalltalk / greetings → clearly DIRECT, no documents needed.
_GREETINGS = {
    "hi", "hello", "hey", "yo", "thanks", "thank you", "good morning",
    "good evening", "how are you", "what's up",
}
# Explicit signals the user is asking about ingested material → bias to RAG.
_DOC_INTENT_KEYWORDS = (
    "document", "documents", "doc", "docs", "file", "files", "uploaded",
    "according to", "in the text", "the paper", "the article", "ingested",
    "knowledge base", "the report",
)
# Signals the query is about *system structure* — relationships, dependencies,
# data/control flow, component interactions. These benefit from graph traversal,
# so when a knowledge graph is available we answer them with HYBRID retrieval
# (FAISS vector context + Neo4j graph context). See Module 10, Part F.
_ARCHITECTURE_KEYWORDS = (
    "relationship", "relationships", "related", "depend", "dependency",
    "dependencies", "architecture", "architectural", "workflow", "work flow",
    "trace", "interact", "interaction", "interacts", "connect", "connects",
    "connected", "pipeline", "end-to-end", "end to end", "failure path",
    "data flow", "control flow", "flow through", "component", "components",
    "how does", "how do", "what happens when", "wired", "fits together",
    "downstream", "upstream", "fail over", "failover", "fall back", "fallback",
)


@dataclass
class RouteDecision:
    """Outcome of routing: the route, why, and any probe hits to reuse."""

    route: Route
    reason: str
    retrieval_used: bool = False
    top_score: Optional[float] = None
    hits: List[Hit] = field(default_factory=list)
    # How context should be gathered for this query (Module 10).
    #   vector — FAISS only (the historical default)
    #   hybrid — FAISS + knowledge-graph traversal (architecture queries)
    retrieval_mode: str = "vector"


class QueryRouter:
    def __init__(
        self,
        classifier: Optional[Callable[[str], Optional[Route]]] = None,
        graph_available: bool = False,
    ) -> None:
        # Optional LLM-backed classifier hook. If it returns a Route, we trust
        # it; if it returns None (or isn't provided), we fall back to heuristics.
        self._classifier = classifier
        # Whether a knowledge graph is wired in. When False (the default, used by
        # unit tests that build a bare router), routing behaves exactly as before
        # Module 10 — no hybrid activation, so existing behaviour is preserved.
        self._graph_available = graph_available

    @staticmethod
    def is_architecture_query(query: str) -> bool:
        """True when the query is about system structure/relationships/flow."""
        lowered = query.lower()
        return any(kw in lowered for kw in _ARCHITECTURE_KEYWORDS)

    def route(
        self,
        query: str,
        engine: SupportsRetrieval,
        top_k: Optional[int] = None,
    ) -> RouteDecision:
        settings = get_settings()
        normalized = query.strip()
        lowered = normalized.lower()
        words = normalized.split()

        logger.info("Routing query=%r (%d words)", normalized, len(words))

        # --- 1. Structural clarification checks (no retrieval) -------------
        if not normalized:
            return RouteDecision(Route.NEEDS_CLARIFICATION, "Empty query.")

        if lowered.rstrip("?.! ") in _VAGUE_PHRASES:
            return RouteDecision(
                Route.NEEDS_CLARIFICATION,
                "Query is a generic phrase with no specific subject.",
            )

        # Greetings/smalltalk are direct and shouldn't hit the index.
        if lowered.rstrip("?.! ") in _GREETINGS:
            return RouteDecision(Route.DIRECT_ANSWER, "Greeting / smalltalk.")

        # Every token is a vague/pronoun token → nothing concrete to answer.
        if words and all(w.strip("?.!,").lower() in _VAGUE_TOKENS for w in words):
            return RouteDecision(
                Route.NEEDS_CLARIFICATION,
                "Query contains only vague/pronoun tokens; subject is unclear.",
            )

        # Too short to be specific, and not a known greeting.
        if len(words) < settings.min_query_words:
            return RouteDecision(
                Route.NEEDS_CLARIFICATION,
                f"Query is too short ({len(words)} word(s)) to answer confidently.",
            )

        # --- 1.5 Architecture / relationship intent → HYBRID (Module 10) ---
        # When a knowledge graph is available and the query is about how the
        # system is wired (relationships, dependencies, flow, component
        # interaction), prefer hybrid retrieval. The seed graph always has the
        # architecture, so these are answerable even with an empty vector index;
        # any indexed documents add vector context on top.
        if self._graph_available and self.is_architecture_query(lowered):
            hits = engine.search(normalized, top_k=top_k) if engine.total_chunks else []
            top = hits[0][1] if hits else None
            logger.info("Architecture query → HYBRID retrieval (vector hits=%d)", len(hits))
            return RouteDecision(
                Route.RAG_ANSWER,
                "Relationship/architecture query — using hybrid retrieval "
                "(FAISS vector context + knowledge-graph traversal).",
                retrieval_used=bool(hits),
                top_score=top,
                hits=hits,
                retrieval_mode="hybrid",
            )

        # --- 2. Optional LLM classifier seam ------------------------------
        if self._classifier is not None:
            decided = self._classifier(normalized)
            if decided is not None:
                logger.info("LLM classifier chose %s", decided)
                # If the classifier picks RAG, still run a probe so the RAG
                # stage has hits to ground on.
                if decided is Route.RAG_ANSWER:
                    hits = engine.search(normalized, top_k=top_k)
                    top = hits[0][1] if hits else None
                    return RouteDecision(
                        decided, "LLM classifier selected RAG_ANSWER.",
                        retrieval_used=True, top_score=top, hits=hits,
                    )
                return RouteDecision(decided, "LLM classifier decision.")

        has_doc_intent = any(kw in lowered for kw in _DOC_INTENT_KEYWORDS)

        # --- 3. Retrieval probe -------------------------------------------
        # If nothing is indexed we can't do RAG at all.
        if engine.total_chunks == 0:
            if has_doc_intent:
                return RouteDecision(
                    Route.NEEDS_CLARIFICATION,
                    "Query references documents, but no documents are ingested yet.",
                )
            return RouteDecision(
                Route.DIRECT_ANSWER,
                "No documents ingested; answering from general knowledge.",
            )

        hits = engine.search(normalized, top_k=top_k)
        top_score = hits[0][1] if hits else None
        logger.info("Probe top_score=%s (doc_intent=%s)", top_score, has_doc_intent)

        # Strong semantic match → ground the answer in retrieved chunks.
        if top_score is not None and top_score >= settings.rag_score_threshold:
            return RouteDecision(
                Route.RAG_ANSWER,
                f"Strong retrieval match (top score {top_score:.3f} >= "
                f"{settings.rag_score_threshold}).",
                retrieval_used=True, top_score=top_score, hits=hits,
            )

        # The user clearly expected document grounding but nothing is relevant.
        if has_doc_intent and (
            top_score is None or top_score < settings.clarification_score_floor
        ):
            return RouteDecision(
                Route.NEEDS_CLARIFICATION,
                "Query expects document context, but no relevant passages were "
                f"found (top score {top_score}).",
                retrieval_used=True, top_score=top_score, hits=hits,
            )

        # Weak/irrelevant match → answer from general knowledge instead.
        return RouteDecision(
            Route.DIRECT_ANSWER,
            f"Weak retrieval match (top score {top_score}); answering from "
            "general knowledge.",
            retrieval_used=True, top_score=top_score, hits=hits,
        )
