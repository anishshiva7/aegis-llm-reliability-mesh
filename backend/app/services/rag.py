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
    -> evaluate (Module 3)
    -> if evaluation.should_retry: RetryManager runs alternate strategies,
       re-evaluates each, and keeps the best-scoring attempt (Module 4)
    -> assemble trace from the winning attempt (+ retry sub-trace when retries ran)

Every attempt — the initial one and each retry — flows through the single
``_run_attempt`` helper, so the RetryManager only needs an ``attempt_fn``
callback and never imports the pipeline (keeps the dependency one-way).
"""

import time
from typing import List, Optional, Tuple

from ..config import get_settings
from ..logging_config import get_logger
from ..models.schemas import (
    AskResponse,
    GenerationTrace,
    GraphEntityModel,
    GraphRelationshipModel,
    GraphTrace,
    RetrievalMode,
    RetrievedContext,
    Route,
    RouteTrace,
)
from .budget import BudgetGuard
from .evaluator import AnswerEvaluator
from .generator import LLMClient
from .graph.graph_metrics import GraphMetrics
from .graph.graph_retriever import GraphRetriever
from .graph.models import GraphSearchResult
from .metrics import MetricsCollector, RequestMetric
from .metrics_store import MetricsStore
from .providers.base import ProviderError
from .retry import AttemptParams, AttemptResult, RetryContext, RetryManager
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


def build_hybrid_prompt(
    query: str,
    contexts: List[RetrievedContext],
    graph: GraphSearchResult,
) -> str:
    """
    Construct a hybrid prompt fusing vector + graph context (Module 10, Part G).

    Layout (the generator still keys off 'CONTEXT:' for grounding):

        CONTEXT:
        VECTOR CONTEXT:
        [1] (source=..#i, score=..) passage text
        ...
        GRAPH CONTEXT:
        Entities: A (Category), B (Category), ...
        Relationships:
        (A) -[REL]-> (B)
        ...
        Linked Chunks:
        - (source=..#i) passage text
        ...

        QUESTION: <query>
        ANSWER:

    The graph block makes *structural* facts (who connects to whom) explicit, so
    the model can answer relationship/architecture questions a pure vector
    retrieval would miss.
    """
    lines = ["CONTEXT:", "VECTOR CONTEXT:"]
    if contexts:
        for i, ctx in enumerate(contexts, start=1):
            lines.append(
                f"[{i}] (source={ctx.source}#{ctx.chunk_index}, "
                f"score={ctx.score:.3f}) {ctx.text}"
            )
    else:
        lines.append("(no vector matches)")

    lines.append("")
    lines.append("GRAPH CONTEXT:")
    entities = graph.traversed_entities or graph.matched_entities
    if entities:
        ent_str = ", ".join(f"{e.name} ({e.category.value})" for e in entities)
        lines.append(f"Entities: {ent_str}")
    if graph.traversed_relationships:
        lines.append("Relationships:")
        for r in graph.traversed_relationships:
            lines.append(f"({r.source}) -[{r.type}]-> ({r.target})")
    if graph.graph_chunks:
        lines.append("Linked Chunks:")
        for ch in graph.graph_chunks:
            lines.append(f"- (source={ch.source}#{ch.chunk_index}) {ch.text}")
    if not entities and not graph.graph_chunks:
        lines.append("(no graph matches)")

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


def _generation_failure_message(error: str) -> str:
    """User-facing text when the LLM provider fails and we must degrade."""
    return (
        "The answer could not be generated because the language model provider "
        f"is currently unavailable ({error}). A degraded response is being "
        "returned; please try again shortly."
    )


class RAGPipeline:
    def __init__(
        self,
        engine: SupportsRetrieval,
        generator: LLMClient,
        router: Optional[QueryRouter] = None,
        evaluator: Optional[AnswerEvaluator] = None,
        retry_manager: Optional[RetryManager] = None,
        metrics: Optional[MetricsCollector] = None,
        metrics_store: Optional[MetricsStore] = None,
        budget_guard: Optional[BudgetGuard] = None,
        graph_retriever: Optional[GraphRetriever] = None,
        graph_metrics: Optional[GraphMetrics] = None,
    ) -> None:
        self.engine = engine
        self.generator = generator
        self.router = router or QueryRouter()
        self.evaluator = evaluator or AnswerEvaluator()
        self.retry_manager = retry_manager or RetryManager()
        # Optional observability sinks. When None, recording is a no-op so
        # directly-constructed test pipelines stay zero-dependency.
        self.metrics = metrics            # in-memory collector (Module 6)
        self.metrics_store = metrics_store  # persistent SQLite store (Module 7)
        self.budget_guard = budget_guard    # daily-cost accumulator (Module 7)
        # Knowledge-graph retrieval (Module 10). When None, the pipeline behaves
        # exactly as before — no hybrid work happens — so existing unit tests
        # that construct a bare pipeline are completely unaffected.
        self.graph_retriever = graph_retriever
        self.graph_metrics = graph_metrics
        self.settings = get_settings()
        logger.info("RAGPipeline ready (generator=%s).", generator.name)

    def ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        force_route: Optional[Route] = None,
        include_trace: bool = True,
    ) -> AskResponse:
        start = time.perf_counter()
        effective_top_k = top_k if top_k is not None else self.settings.default_top_k

        # Initial attempt — same code path every retry uses.
        initial = self._run_attempt(
            AttemptParams(
                query=query, top_k=top_k, force_route=force_route, strategy="initial"
            )
        )

        # Self-healing loop (Module 4): engage only when quality is sub-par AND
        # the caller didn't pin a route. force_route is a manual debugging/eval
        # override, so we honour it exactly and never let a strategy change it.
        best = initial
        retry_trace = None
        if force_route is None and initial.evaluation.should_retry:
            ctx = RetryContext(
                original_query=query,
                effective_top_k=effective_top_k,
                initial_route=initial.decision.route,
                initial_top_score=initial.decision.top_score,
                index_size=self.engine.total_chunks,
                settings=self.settings,
            )
            outcome = self.retry_manager.run(initial, ctx, self._run_attempt)
            best = outcome.best
            retry_trace = outcome.trace

        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "Answered route=%s in %.1f ms (retries=%d)",
            best.decision.route,
            latency_ms,
            retry_trace.retry_count if retry_trace else 0,
        )

        # Operational metrics (Module 6) — recorded once per /ask, from the
        # winning attempt. No-op when no collector was injected.
        self._record_metrics(best, retry_trace, latency_ms)

        trace = None
        if include_trace:
            # Distinguish the two retrieval semantics (Module 8):
            #   retrieval_probe_used — the router searched FAISS to *decide* the
            #     route. True for RAG, weak-match DIRECT, and doc-intent
            #     clarification (any path where decision.retrieval_used is set).
            #   answer_context_used — retrieved chunks were actually fed into the
            #     answer prompt. True only on the grounded RAG path with chunks.
            retrieval_probe_used = best.decision.retrieval_used
            answer_context_used = (
                best.mode in ("grounded", "hybrid") and len(best.contexts) > 0
            )
            retrieval_mode = RetrievalMode(
                getattr(best.decision, "retrieval_mode", "vector")
            )
            graph_trace = self._build_graph_trace(best.graph_result)
            trace = RouteTrace(
                route=best.decision.route,
                reason=best.decision.reason,
                retrieval_used=best.decision.retrieval_used,
                retrieval_probe_used=retrieval_probe_used,
                answer_context_used=answer_context_used,
                generation_mode=best.mode,
                retrieval_mode=retrieval_mode,
                graph_used=graph_trace is not None and (
                    bool(graph_trace.matched_entities) or bool(graph_trace.graph_chunks)
                ),
                latency_ms=round(latency_ms, 2),
                top_score=best.decision.top_score,
                retrieved=best.contexts,
                evaluation=best.evaluation,
                retry=retry_trace,
                generation_error=best.generation_error,
                generation=best.generation,
                graph=graph_trace,
            )

        return AskResponse(
            query=query, route=best.decision.route, answer=best.answer, trace=trace
        )

    def _record_metrics(self, best: AttemptResult, retry_trace, latency_ms: float) -> None:
        """Record this request to the in-memory collector + persistent store."""
        gen = best.generation
        provider_name = (
            gen.provider_name
            if gen
            else getattr(self.generator, "provider_name", self.generator.name)
        )
        model_name = gen.model_name if gen else getattr(self.generator, "model_name", provider_name)
        score = best.evaluation.scores.overall_score
        fallback_used = gen.fallback_used if gen else False
        retry_count = retry_trace.retry_count if retry_trace else 0
        cost = gen.estimated_cost_usd if gen else 0.0
        degraded = best.generation_error is not None or (
            retry_trace.degraded_response if retry_trace else False
        )

        # In-memory collector (Module 6) — preserved for backward compatibility.
        if self.metrics is not None:
            self.metrics.record(
                RequestMetric(
                    provider=provider_name,
                    latency_ms=latency_ms,
                    overall_score=score,
                    fallback_used=fallback_used,
                    degraded=degraded,
                    retried=retry_count > 0,
                    estimated_cost_usd=cost,
                )
            )

        # Persistent store (Module 7) — raw rows for percentiles/histograms.
        if self.metrics_store is not None:
            self.metrics_store.record(
                provider_name=provider_name,
                model_name=model_name,
                route=best.decision.route.value,
                latency_ms=latency_ms,
                overall_score=score,
                retry_count=retry_count,
                fallback_used=fallback_used,
                degraded_response=degraded,
                estimated_cost_usd=cost,
            )

        # Roll realised cost into the daily budget accumulator (Module 7).
        if self.budget_guard is not None and cost:
            self.budget_guard.add_daily_cost(cost)

    # ------------------------------------------------------------------ helpers
    def _run_attempt(self, params: AttemptParams) -> AttemptResult:
        """
        Execute one full attempt: route -> dispatch -> generate -> evaluate.

        This is the unit of work both the initial call and the RetryManager
        invoke, so retrying is just running it again with different params.
        """
        if params.force_route is not None:
            decision = self._forced_decision(
                params.query, params.force_route, params.top_k
            )
        else:
            decision = self.router.route(params.query, self.engine, top_k=params.top_k)

        logger.info(
            "Attempt strategy=%s route=%s reason=%s",
            params.strategy,
            decision.route,
            decision.reason,
        )

        # Generation may call a real provider (Module 5); a ProviderError is
        # caught here so a single failure degrades gracefully instead of
        # crashing the request. Provider selection/logic lives in the factory,
        # not here — we only handle the abstract failure type.
        generation_error: Optional[str] = None
        generation: Optional[GenerationTrace] = None
        graph_result: Optional[GraphSearchResult] = None
        try:
            if decision.route is Route.RAG_ANSWER:
                answer, contexts, mode, generation, graph_result = self._handle_rag(
                    params.query, decision, params.top_k
                )
            elif decision.route is Route.NEEDS_CLARIFICATION:
                answer = _clarification_message(decision.reason)
                contexts, mode = [], "clarification"
            else:  # DIRECT_ANSWER
                answer, generation = self._complete(build_direct_prompt(params.query))
                contexts, mode = [], "direct"
        except ProviderError as exc:
            generation_error = str(exc)
            logger.warning("Generation failed (%s); returning degraded answer.", exc)
            answer = _generation_failure_message(generation_error)
            contexts, mode = [], "degraded"
            generation = self._failure_meta()

        # Evaluate against the query that actually produced this answer (which
        # may be an expanded query on a retry). A degraded answer scores low,
        # so the failure naturally flows through the retry/circuit-breaker path.
        evaluation = self.evaluator.evaluate(
            query=params.query,
            answer=answer,
            route=decision.route,
            retrieved_chunks=contexts,
            top_score=decision.top_score,
        )

        return AttemptResult(
            params=params,
            answer=answer,
            contexts=contexts,
            mode=mode,
            decision=decision,
            evaluation=evaluation,
            generation_error=generation_error,
            generation=generation,
            graph_result=graph_result,
        )

    def _complete(
        self, prompt: str, system: Optional[str] = None
    ) -> Tuple[str, Optional[GenerationTrace]]:
        """
        Generate via the injected generator, capturing observability metadata
        when available. Generators that predate Module 6 (e.g. a bare MockLLM
        in a unit test) only expose ``complete``, so we degrade to text + None.
        """
        gen = self.generator
        if hasattr(gen, "complete_with_meta"):
            return gen.complete_with_meta(prompt, system=system)
        return gen.complete(prompt, system=system), None

    def _failure_meta(self) -> GenerationTrace:
        """A GenerationTrace describing the attempted provider after a failure."""
        name = getattr(self.generator, "provider_name", self.generator.name)
        chain = getattr(self.generator, "fallback_chain", [name])
        return GenerationTrace(
            provider_name=name,
            model_name=getattr(self.generator, "model_name", name),
            provider_latency_ms=0.0,
            fallback_used=False,
            fallback_chain=list(chain),
            estimated_input_tokens=0,
            estimated_output_tokens=0,
            estimated_cost_usd=0.0,
        )

    def _handle_rag(self, query, decision: RouteDecision, top_k):
        """Build grounded context (reusing probe hits when present) and generate.

        Returns ``(answer, contexts, mode, generation, graph_result)``. When the
        router selected hybrid retrieval *and* a graph retriever is wired, this
        fuses FAISS vector context with knowledge-graph context; otherwise it is
        the historical vector-only grounded path (graph_result is None).
        """
        # Reuse the router's probe hits if it already retrieved; otherwise (e.g.
        # a forced RAG route) retrieve now.
        hits = decision.hits if decision.hits else self.engine.search(query, top_k=top_k)
        contexts = _hits_to_contexts(hits)

        use_hybrid = (
            self.graph_retriever is not None
            and decision.retrieval_mode == "hybrid"
        )
        if not use_hybrid:
            prompt = build_grounded_prompt(query, contexts)
            answer, generation = self._complete(prompt, system=_GROUNDED_SYSTEM)
            return answer, contexts, "grounded", generation, None

        # --- Hybrid: vector + graph (Module 10) ---------------------------
        graph_result = self.graph_retriever.retrieve(query)
        if self.graph_metrics is not None:
            self.graph_metrics.record_traversal(
                self.graph_retriever.last_latency_ms, hybrid=True
            )

        prompt = build_hybrid_prompt(query, contexts, graph_result)
        answer, generation = self._complete(prompt, system=_GROUNDED_SYSTEM)

        # Surface the graph's linked chunks as first-class grounding context so
        # the evaluator credits groundedness and the trace shows what was used.
        merged = list(contexts)
        seen = {(c.source, c.chunk_index) for c in merged}
        for ch in graph_result.graph_chunks:
            key = (ch.source, ch.chunk_index)
            if key not in seen:
                seen.add(key)
                merged.append(
                    RetrievedContext(
                        chunk_id=-1,
                        text=ch.text,
                        score=graph_result.graph_score,
                        source=ch.source,
                        chunk_index=ch.chunk_index,
                    )
                )
        return answer, merged, "hybrid", generation, graph_result

    def _build_graph_trace(
        self, graph_result: Optional[GraphSearchResult]
    ) -> Optional[GraphTrace]:
        """Map an internal GraphSearchResult to the API GraphTrace (Module 10)."""
        if graph_result is None:
            return None
        backend = getattr(
            getattr(self.graph_retriever, "store", None), "backend", "memory"
        )
        latency = (
            self.graph_retriever.last_latency_ms
            if self.graph_retriever is not None
            else 0.0
        )
        return GraphTrace(
            graph_backend=backend,
            matched_entities=[
                GraphEntityModel(
                    name=e.name, category=e.category.value, description=e.description
                )
                for e in graph_result.matched_entities
            ],
            traversed_entities=[
                GraphEntityModel(
                    name=e.name, category=e.category.value, description=e.description
                )
                for e in graph_result.traversed_entities
            ],
            traversed_relationships=[
                GraphRelationshipModel(source=r.source, type=r.type, target=r.target)
                for r in graph_result.traversed_relationships
            ],
            graph_chunks=[
                RetrievedContext(
                    chunk_id=-1,
                    text=ch.text,
                    score=graph_result.graph_score,
                    source=ch.source,
                    chunk_index=ch.chunk_index,
                )
                for ch in graph_result.graph_chunks
            ],
            graph_score=graph_result.graph_score,
            hops=graph_result.hops,
            graph_latency_ms=round(latency, 2),
        )

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
