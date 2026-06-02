"""
Seed knowledge graph: the Aegis architecture itself (Module 10 — Part C).

This is the deterministic backbone that lets Aegis answer relationship- and
architecture-heavy questions ("how do routing and retries interact?") even
before any document is ingested. Nodes are Aegis's real components; edges encode
how a request actually flows through the system.

Ingested documents *enrich* this graph by linking their chunks to these
entities — they never replace it. The whole ontology is data, so a future
LLM-based extractor can append to it without code changes.
"""

from __future__ import annotations

from typing import List

from .models import EntityCategory as C
from .models import GraphNode, GraphRelationship

# ---------------------------------------------------------------------------
# Nodes — (name, category, description, aliases)
# Aliases are the surface forms we match in free text and user queries.
# ---------------------------------------------------------------------------
SEED_NODES: List[GraphNode] = [
    GraphNode("QueryRouter", C.COMPONENT,
              "Classifies each query into a route using heuristics + a FAISS probe.",
              ("query router", "router", "routing", "route a query", "routes")),
    GraphNode("RAGPipeline", C.COMPONENT,
              "Orchestrates route -> retrieve -> generate -> evaluate -> self-heal.",
              ("rag pipeline", "pipeline", "orchestrator", "ask flow", "end-to-end",
               "rag request", "request flow", "request path")),
    GraphNode("RetrievalEngine", C.COMPONENT,
              "Chunks, embeds, indexes, and searches documents.",
              ("retrieval engine", "retrieval", "retrieve")),
    GraphNode("FAISS", C.RETRIEVAL,
              "In-process vector index for semantic similarity search.",
              ("faiss", "vector store", "vector index", "vector search", "vector retrieval")),
    GraphNode("Embedder", C.RETRIEVAL,
              "sentence-transformers MiniLM embedding model.",
              ("embedder", "embedding", "embeddings", "minilm", "sentence-transformers")),
    GraphNode("GraphRetriever", C.RETRIEVAL,
              "Traverses the knowledge graph to gather entities and linked chunks.",
              ("graph retriever", "graph retrieval", "graph traversal", "knowledge graph")),

    GraphNode("DIRECT_ANSWER", C.ROUTE,
              "Answer from general knowledge without retrieval.",
              ("direct_answer", "direct answer", "direct route")),
    GraphNode("RAG_ANSWER", C.ROUTE,
              "Answer grounded in retrieved document context.",
              ("rag_answer", "rag answer", "grounded answer", "rag route")),
    GraphNode("NEEDS_CLARIFICATION", C.ROUTE,
              "Decline to answer a vague or unanswerable query.",
              ("needs_clarification", "clarification", "clarify")),

    GraphNode("Evaluator", C.EVALUATION,
              "Wraps the judge and applies retry thresholds.",
              ("evaluator", "answer evaluator")),
    GraphNode("DeterministicJudge", C.EVALUATION,
              "Offline six-dimension answer scorer (LLM-as-a-Judge, no LLM call).",
              ("judge", "deterministic judge", "lm-as-a-judge", "llm-as-a-judge", "evaluation", "scoring")),
    GraphNode("should_retry", C.EVALUATION,
              "Boolean signal raised when answer quality is below threshold.",
              ("should_retry", "retry flag", "retry decision")),
    GraphNode("hallucination_risk", C.METRIC,
              "Estimated probability the answer is fabricated.",
              ("hallucination_risk", "hallucination", "hallucination risk")),

    GraphNode("RetryManager", C.RETRY,
              "Runs alternate strategies and keeps the best-scoring attempt.",
              ("retry manager", "retrymanager", "self-healing", "self healing", "retries", "retry loop")),
    GraphNode("QueryExpansion", C.RETRY,
              "Retry strategy: rewrite the query for richer retrieval.",
              ("query expansion", "expand query", "queryexpansion")),
    GraphNode("ForceRAG", C.RETRY,
              "Retry strategy: ground a previously direct answer in the index.",
              ("force rag", "force-rag", "forced grounding", "forcerag")),
    GraphNode("BudgetGuard", C.RETRY,
              "Cost/rate guardrail that caps runaway retries.",
              ("budget guard", "budgetguard", "budget", "cost guard", "guardrail")),

    GraphNode("LLMProvider", C.PROVIDER,
              "Vendor-agnostic generation interface.",
              ("llmprovider", "llm provider", "provider interface", "provider abstraction")),
    GraphNode("FallbackProvider", C.PROVIDER,
              "Health-aware chain that fails over between providers.",
              ("fallback provider", "fallbackprovider", "fallback", "fallback chain", "failover")),
    GraphNode("OpenAIProvider", C.PROVIDER, "OpenAI generation backend.",
              ("openai", "openaiprovider", "gpt")),
    GraphNode("AnthropicProvider", C.PROVIDER, "Anthropic generation backend.",
              ("anthropic", "anthropicprovider", "claude")),
    GraphNode("BedrockProvider", C.PROVIDER, "AWS Bedrock generation backend.",
              ("bedrock", "bedrockprovider", "aws bedrock")),
    GraphNode("MockProvider", C.PROVIDER, "Deterministic offline backstop provider.",
              ("mock", "mockprovider", "mock provider")),
    GraphNode("ProviderHealthRegistry", C.PROVIDER,
              "Tracks per-provider health and reorders the chain.",
              ("provider health", "health registry", "providerhealthregistry", "health monitoring")),

    GraphNode("MetricsCollector", C.METRIC,
              "Aggregates per-request operational metrics.",
              ("metrics collector", "metricscollector", "metrics")),
    GraphNode("MetricsStore", C.METRIC,
              "Persistent SQLite store for request metrics.",
              ("metrics store", "metricsstore", "sqlite metrics", "persistent metrics")),
    GraphNode("Prometheus", C.METRIC,
              "Prometheus text exposition of Aegis metrics.",
              ("prometheus", "prometheus metrics", "exposition")),
    GraphNode("Latency", C.METRIC, "End-to-end and provider latency.",
              ("latency", "p95", "p99", "percentile")),
    GraphNode("Cost", C.METRIC, "Estimated token/USD cost per request.",
              ("cost", "token cost", "usd", "tokens")),
    GraphNode("DegradedResponse", C.EVALUATION,
              "A response still below quality thresholds after retries.",
              ("degraded response", "degraded", "degraded_response")),

    GraphNode("Dashboard", C.DASHBOARD,
              "Next.js observability dashboard.",
              ("dashboard", "observability dashboard", "ui", "next.js", "nextjs")),
]

# ---------------------------------------------------------------------------
# Relationships — how a request actually flows through Aegis.
# ---------------------------------------------------------------------------
SEED_RELATIONSHIPS: List[GraphRelationship] = [
    # Routing
    GraphRelationship("RAGPipeline", "USES", "QueryRouter"),
    GraphRelationship("QueryRouter", "PROBES", "FAISS"),
    GraphRelationship("QueryRouter", "ROUTES_TO", "DIRECT_ANSWER"),
    GraphRelationship("QueryRouter", "ROUTES_TO", "RAG_ANSWER"),
    GraphRelationship("QueryRouter", "ROUTES_TO", "NEEDS_CLARIFICATION"),

    # Retrieval
    GraphRelationship("RAGPipeline", "USES", "RetrievalEngine"),
    GraphRelationship("RetrievalEngine", "USES", "FAISS"),
    GraphRelationship("RetrievalEngine", "USES", "Embedder"),
    GraphRelationship("RAG_ANSWER", "USES", "FAISS"),
    GraphRelationship("RAG_ANSWER", "USES", "GraphRetriever"),
    GraphRelationship("GraphRetriever", "TRAVERSES", "RetrievalEngine"),

    # Evaluation
    GraphRelationship("RAGPipeline", "USES", "Evaluator"),
    GraphRelationship("Evaluator", "USES", "DeterministicJudge"),
    GraphRelationship("DeterministicJudge", "PRODUCES", "should_retry"),
    GraphRelationship("DeterministicJudge", "PRODUCES", "hallucination_risk"),
    GraphRelationship("should_retry", "TRIGGERS", "RetryManager"),

    # Self-healing
    GraphRelationship("RAGPipeline", "USES", "RetryManager"),
    GraphRelationship("RetryManager", "USES", "QueryExpansion"),
    GraphRelationship("RetryManager", "USES", "ForceRAG"),
    GraphRelationship("RetryManager", "USES", "BudgetGuard"),
    GraphRelationship("RetryManager", "PRODUCES", "DegradedResponse"),
    GraphRelationship("BudgetGuard", "LIMITS", "RetryManager"),
    GraphRelationship("ForceRAG", "USES", "FAISS"),

    # Providers
    GraphRelationship("RAGPipeline", "USES", "LLMProvider"),
    GraphRelationship("LLMProvider", "IMPLEMENTS", "FallbackProvider"),
    GraphRelationship("OpenAIProvider", "IMPLEMENTS", "LLMProvider"),
    GraphRelationship("AnthropicProvider", "IMPLEMENTS", "LLMProvider"),
    GraphRelationship("BedrockProvider", "IMPLEMENTS", "LLMProvider"),
    GraphRelationship("MockProvider", "IMPLEMENTS", "LLMProvider"),
    GraphRelationship("FallbackProvider", "FAILS_OVER_TO", "MockProvider"),
    GraphRelationship("FallbackProvider", "USES", "ProviderHealthRegistry"),

    # Observability
    GraphRelationship("RAGPipeline", "USES", "MetricsCollector"),
    GraphRelationship("MetricsCollector", "RECORDS", "Latency"),
    GraphRelationship("MetricsCollector", "RECORDS", "Cost"),
    GraphRelationship("MetricsCollector", "WRITES_TO", "MetricsStore"),
    GraphRelationship("MetricsStore", "EXPOSES", "Prometheus"),
    GraphRelationship("Dashboard", "READS", "MetricsStore"),
    GraphRelationship("Dashboard", "READS", "RAGPipeline"),
    GraphRelationship("DegradedResponse", "RECORDED_BY", "MetricsCollector"),
]
