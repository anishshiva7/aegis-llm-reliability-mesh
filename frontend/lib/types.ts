// Type definitions mirroring the Aegis backend API contract
// (backend/app/models/schemas.py). Kept in one place so every panel shares
// a single source of truth for the shapes it renders.

export type Route = "DIRECT_ANSWER" | "RAG_ANSWER" | "NEEDS_CLARIFICATION";

// ---------------------------------------------------------------------------
// Ingest (Module 8 — dashboard ingestion panel)
// ---------------------------------------------------------------------------
export interface IngestRequest {
  text: string;
  source?: string | null;
  chunk_size?: number | null;
  chunk_overlap?: number | null;
}

export interface IngestResponse {
  source: string;
  chunks_created: number;
  total_chunks_in_index: number;
}

export interface AskRequest {
  query: string;
  top_k?: number | null;
  force_route?: Route | null;
  include_trace?: boolean;
}

export interface RetrievedContext {
  chunk_id: number;
  text: string;
  score: number;
  source: string;
  chunk_index: number;
}

export interface EvaluationScores {
  relevance: number;
  groundedness: number;
  completeness: number;
  hallucination_risk: number;
  confidence: number;
  overall_score: number;
}

export interface EvaluationResult {
  scores: EvaluationScores;
  should_retry: boolean;
  evaluation_reason: string;
}

export interface AttemptTrace {
  attempt: number;
  strategy: string;
  route: Route;
  overall_score: number;
  confidence: number;
  should_retry: boolean;
}

export interface RetryTrace {
  attempts: AttemptTrace[];
  retry_count: number;
  selected_best_attempt: number;
  retry_strategies_used: string[];
  score_progression: number[];
  degraded_response: boolean;
  budget_blocked: boolean;
}

export interface GenerationTrace {
  provider_name: string;
  model_name: string;
  provider_latency_ms: number;
  fallback_used: boolean;
  fallback_chain: string[];
  estimated_input_tokens: number;
  estimated_output_tokens: number;
  estimated_cost_usd: number;
  token_usage_source: "provider" | "estimated";
}

// ---------------------------------------------------------------------------
// GraphRAG / hybrid retrieval (Module 10)
// ---------------------------------------------------------------------------
export type RetrievalMode = "vector" | "graph" | "hybrid";

export interface GraphEntity {
  name: string;
  category: string;
  description: string;
}

export interface GraphRelationship {
  source: string;
  type: string;
  target: string;
}

export interface GraphTrace {
  graph_backend: string;
  matched_entities: GraphEntity[];
  traversed_entities: GraphEntity[];
  traversed_relationships: GraphRelationship[];
  graph_chunks: RetrievedContext[];
  graph_score: number;
  hops: number;
  graph_latency_ms: number;
}

// Module 10 — standalone /graph/search endpoint (graph retrieval in isolation).
export interface GraphSearchRequest {
  query: string;
  max_hops?: number | null;
  chunk_limit?: number | null;
}

export interface GraphSearchResponse {
  query: string;
  graph_used: boolean;
  graph: GraphTrace;
}

export interface RouteTrace {
  route: Route;
  reason: string;
  retrieval_used: boolean;
  // Module 8 — precise retrieval semantics.
  // retrieval_probe_used: the router searched FAISS to *decide* the route.
  // answer_context_used: retrieved chunks were injected into the answer prompt.
  retrieval_probe_used?: boolean;
  answer_context_used?: boolean;
  generation_mode: string;
  // Module 10 — how context was gathered + the knowledge-graph trace.
  retrieval_mode?: RetrievalMode;
  graph_used?: boolean;
  latency_ms: number;
  top_score?: number | null;
  retrieved: RetrievedContext[];
  evaluation?: EvaluationResult | null;
  retry?: RetryTrace | null;
  generation_error?: string | null;
  generation?: GenerationTrace | null;
  graph?: GraphTrace | null;
}

export interface AskResponse {
  query: string;
  route: Route;
  answer: string;
  trace?: RouteTrace | null;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
export interface MetricsSnapshot {
  total_requests: number;
  requests_by_provider: Record<string, number>;
  requests_by_route: Record<string, number>;
  fallback_count: number;
  degraded_response_count: number;
  average_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  average_overall_score: number;
  score_histogram: Record<string, number>;
  latency_histogram: Record<string, number>;
  retry_rate: number;
  fallback_rate: number;
  degraded_response_rate: number;
  cost_total: number;
  estimated_cost_usd_total: number;
  // Module 10 — knowledge-graph / hybrid retrieval telemetry.
  graph?: GraphMetrics;
}

export interface GraphMetrics {
  graph_nodes: number;
  graph_relationships: number;
  linked_chunks: number;
  graph_traversals: number;
  hybrid_queries: number;
  graph_latency_ms: number;
}

// ---------------------------------------------------------------------------
// Provider health
// ---------------------------------------------------------------------------
export type HealthStatus = "healthy" | "degraded" | "unhealthy";

export interface ProviderHealth {
  provider: string;
  health_status: HealthStatus;
  consecutive_failures: number;
  total_successes: number;
  total_failures: number;
  last_success_at?: number | null;
  last_failure_at?: number | null;
}

export interface ProvidersHealthResponse {
  providers: ProviderHealth[];
  recommended_order: string[];
}
