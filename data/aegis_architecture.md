# Aegis Architecture Overview

Aegis is a self-healing LLM reliability mesh. Every request to the /ask endpoint
flows through five stages: routing, retrieval, generation, evaluation, and
self-healing. Each stage is observable and the full decision trail is returned
in the response trace.

## Query routing

The QueryRouter classifies every incoming query into one of three routes:
DIRECT_ANSWER, RAG_ANSWER, or NEEDS_CLARIFICATION. It first applies cheap
structural heuristics — empty or vague queries are routed to clarification, and
greetings are answered directly without retrieval. It then fires a single FAISS
retrieval probe. When the top cosine similarity is at or above the RAG
threshold, the query is routed to RAG_ANSWER and the probe hits are reused for
grounding. A weak match is routed to DIRECT_ANSWER, and a document-intent query
with no relevant passages is routed to NEEDS_CLARIFICATION instead of being
answered from thin context.

## Retrieval

Documents are split into overlapping word-window chunks, embedded with the
all-MiniLM-L6-v2 sentence-transformer, and indexed into a FAISS inner-product
index. At query time the most semantically similar chunks are retrieved in
milliseconds and ranked by cosine similarity.

## Generation and provider fallback

Generation runs through a vendor-agnostic LLMProvider interface. Providers are
arranged in a priority chain such as OpenAI, then Anthropic, then a mock
backstop. A rolling health registry reorders the chain healthiest-first on every
request. When the primary provider fails, the fallback provider answers and the
generation trace records fallback_used as true. The mock provider is a
guaranteed offline backstop so the system always returns an answer.

## Evaluation

The DeterministicJudge scores every answer offline on six dimensions: relevance,
groundedness, completeness, hallucination risk, confidence, and a weighted
overall score. No second LLM call is required, so evaluation is free and
deterministic. For RAG answers, groundedness tracks similarity to the retrieved
chunks; for direct answers, grounding is treated as not-applicable rather than
penalised.

## Self-healing retry

When the overall score falls below the retry threshold of 0.60, the RetryManager
runs alternate strategies in order: query expansion, force-RAG grounding, and a
wider top-k retrieval. Each attempt is re-evaluated and the best-scoring answer
wins. If no attempt clears the threshold, the response is flagged as a degraded
response. A budget guard caps retry spend so self-healing never runs away.

## Observability

Every request is recorded to a persistent SQLite metrics store and exposed
through the /metrics endpoint, including P50/P95/P99 latency, score histograms,
retry rate, fallback count, and estimated cost. A Prometheus exposition format
is available at /metrics/prometheus and provider health at /providers/health.
