"""
RAGPipeline — orchestrates the /ask flow.

Glues the router, retrieval engine, and generator together and produces a fully
populated AskResponse (answer + trace). This is the object the HTTP layer talks
to, keeping ask.py thin.

Flow:
    query -> (force_route or router) -> branch:
        DIRECT_ANSWER       -> generator on a plain prompt (no retrieval)
        RAG_ANSWER          -> retrieve top-k -> grounded prompt -> generator
        NEEDS_CLARIFICATION -> static, helpful clarification message
    -> assemble trace (route, reason, scores, latency, mode, retrieval_used)
"""

import time
from typing import List, Optional

from ..config import get_settings
from ..logging_config import get_logger
from ..models.schemas import (
    AskResponse,
    RetrievedContext,
    Route,
    RouteTrace,
)
from .generator import LLMClient
from .router import Hit, QueryRouter, RouteDecision, SupportsRetrieval

logger = get_logger(__name__)

# System instruction sent alongside grounded prompts. Kept here so a real LLM
# receives the same guardrails the mock is built around.
_GROUNDED_SYSTEM = (
    "You are Aegis, a careful assistant. Answer the question using ONLY the "
    "provided context. If the context is insufficient, say so explicitly."
)


def _hits_to_contexts(hits: List[Hit]) -> List[RetrievedContext]:
    """Map raw (chunk_id, score, ChunkRecord) tuples to API context models."""
    contexts: List[RetrievedContext] = []
    for chunk_id, score, record in hits:
        contexts.append(
            RetrievedContext(
                chunk_id=chunk_id,
                text=record.text,
                score=score,
                source=record.source,
                chunk_index=record.chunk_index,
            )
        )
    return contexts


def build_grounded_prompt(query: str, contexts: List[RetrievedContext]) -> str:
    """
    Construct a grounded prompt: numbered context block + the question.

    The 'CONTEXT:' / 'QUESTION:' structure is the contract the generator reads
    (MockLLM keys off it; a real LLM is instructed by _GROUNDED_SYSTEM). Each
    passage is labelled with its provenance and score for traceability.
    """
    lines = ["CONTEXT:"]
    for i, ctx in enumerate(contexts, start=1):
        lines.append(f"[{i}] (source={ctx.source}#{ctx.chunk_index}, "
                     f"score={ctx.score:.3f}) {ctx.text}")
    lines.append("")
    lines.append(f"QUESTION: {query}")
    lines.append("ANSWER:")
    return "\n".join(lines)


def build_direct_prompt(query: str) -> str:
    """A minimal prompt with no retrieved context."""
    return f"QUESTION: {query}\nANSWER:"


def _clarification_message(reason: str) -> str:
    """Friendly, actionable clarification text shown to the user."""
    return (
        "I need a bit more detail to answer confidently. "
        f"({reason}) Could you rephrase with a specific subject, or add context "
        "such as which document or topic you mean?"
    )


class RAGPipeline:
    def __init__(
        self,
        engine: SupportsRetrieval,
        generator: LLMClient,
        router: Optional[QueryRouter] = None,
    ) -> None:
        self.engine = engine
        self.generator = generator
        self.router = router or QueryRouter()
        logger.info("RAGPipeline ready (generator=%s).", generator.name)

    def ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        force_route: Optional[Route] = None,
        include_trace: bool = True,
    ) -> AskResponse:
        start = time.perf_counter()

        # Decide the route — either forced by the caller or chosen by the router.
        if force_route is not None:
            decision = self._forced_decision(query, force_route, top_k)
        else:
            decision = self.router.route(query, self.engine, top_k=top_k)

        logger.info("Route=%s reason=%s", decision.route, decision.reason)

        # Dispatch to the matching handler.
        if decision.route is Route.RAG_ANSWER:
            answer, contexts, mode = self._handle_rag(query, decision, top_k)
        elif decision.route is Route.NEEDS_CLARIFICATION:
            answer = _clarification_message(decision.reason)
            contexts, mode = [], "clarification"
        else:  # DIRECT_ANSWER
            answer = self.generator.complete(build_direct_prompt(query))
            contexts, mode = [], "direct"

        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info("Answered route=%s in %.1f ms", decision.route, latency_ms)

        trace = None
        if include_trace:
            trace = RouteTrace(
                route=decision.route,
                reason=decision.reason,
                retrieval_used=decision.retrieval_used,
                generation_mode=mode,
                latency_ms=round(latency_ms, 2),
                top_score=decision.top_score,
                retrieved=contexts,
            )

        return AskResponse(
            query=query, route=decision.route, answer=answer, trace=trace
        )

    # ------------------------------------------------------------------ helpers
    def _handle_rag(self, query, decision: RouteDecision, top_k):
        """Build grounded context (reusing probe hits when present) and generate."""
        # Reuse the router's probe hits if it already retrieved; otherwise (e.g.
        # a forced RAG route) retrieve now.
        hits = decision.hits if decision.hits else self.engine.search(query, top_k=top_k)
        contexts = _hits_to_contexts(hits)
        prompt = build_grounded_prompt(query, contexts)
        answer = self.generator.complete(prompt, system=_GROUNDED_SYSTEM)
        return answer, contexts, "grounded"

    def _forced_decision(self, query, force_route: Route, top_k) -> RouteDecision:
        """Build a RouteDecision for a caller-forced route (skips heuristics)."""
        reason = f"Route forced by caller ({force_route.value})."
        if force_route is Route.RAG_ANSWER:
            # Retrieve so the forced RAG path has context to ground on.
            hits = self.engine.search(query, top_k=top_k)
            top = hits[0][1] if hits else None
            return RouteDecision(
                force_route, reason, retrieval_used=True, top_score=top, hits=hits
            )
        return RouteDecision(force_route, reason)
