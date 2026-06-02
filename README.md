# Aegis — Self-Healing LLM Reliability Mesh

Aegis is a production-minded RAG system that treats **answer quality as a
first-class, measurable signal** — and acts on it. Every request is routed,
grounded, scored, and (when quality is low) automatically retried with
alternate strategies, all behind a single `/ask` endpoint and a live
observability dashboard.

It is built to be run offline with zero API keys (a deterministic mock
provider), then pointed at OpenAI, Anthropic, or AWS Bedrock by changing a
single environment variable — no pipeline code changes.

> **Not just another RAG wrapper.** Most RAG demos stop at *retrieve → stuff →
> generate*. Aegis adds the parts that matter in production: an explicit routing
> decision, an offline evaluation pass that scores every answer, a self-healing
> retry loop that measurably improves bad answers, a health-aware multi-provider
> fallback chain, cost/budget guardrails, and end-to-end observability.

---

## Why it matters

LLM apps fail quietly. They answer confidently from irrelevant context,
hallucinate when retrieval misses, and silently degrade when a provider has an
outage. Aegis makes those failure modes **visible and self-correcting**:

- **Bad answers are detected, not shipped.** An offline judge scores every
  answer on six dimensions before it is returned.
- **Low-quality answers heal themselves.** A retry loop rewrites the query,
  forces grounding, or widens retrieval, then keeps the best-scoring attempt.
- **A provider outage doesn't take you down.** A health-aware fallback chain
  reorders providers healthiest-first and always has an offline backstop.
- **Everything is observable.** Routing reasons, scores, retries, provider
  health, latency percentiles, and estimated cost are all surfaced in the trace
  and the dashboard.

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
   POST /ask  ─────────► │                 RAGPipeline                  │
                         │                                              │
                         │  1. QueryRouter                              │
                         │       heuristics + single FAISS probe        │
                         │       → DIRECT / RAG / NEEDS_CLARIFICATION   │
                         │                                              │
                         │  2. Retrieval (RAG route only)               │
                         │       vector: chunk → embed (MiniLM) → FAISS │
                         │       hybrid: FAISS ∪ Neo4j graph traversal  │
                         │              (architecture-style queries)    │
                         │                                              │
                         │  3. Generation                               │
                         │       LLMProvider interface + FallbackChain  │
                         │       mock | openai | anthropic | bedrock    │
                         │                                              │
                         │  4. Evaluation (DeterministicJudge, offline) │
                         │       relevance · groundedness · complete-   │
                         │       ness · hallucination · confidence      │
                         │                                              │
                         │  5. Self-healing (RetryManager)              │
                         │       if score < 0.60: expand-query /        │
                         │       force-RAG / wider-k → keep best        │
                         └───────────────┬──────────────────────────────┘
                                         │
              records every request ►  SQLite metrics store + in-memory collector
                                         │
   GET /metrics  ·  /metrics/prometheus  ·  /providers/health  ·  Next.js dashboard
```

Two layers, cleanly separated:

- **`backend/`** — FastAPI service. All orchestration lives in `RAGPipeline`;
  the HTTP handlers are thin. Provider selection lives in exactly one factory,
  so no routing/retrieval/scoring code knows which LLM answered.
- **`frontend/`** — Next.js 14 (App Router, TypeScript) dashboard: a query
  console that visualises the full trace, plus a metrics dashboard.

---

## Feature overview

| Capability | What it does |
|---|---|
| **Adaptive routing** | Classifies each query into `DIRECT_ANSWER`, `RAG_ANSWER`, or `NEEDS_CLARIFICATION` using structural heuristics + a single FAISS probe. |
| **Grounded retrieval** | Word-window chunking → MiniLM embeddings → FAISS inner-product search, with provenance and scores per chunk. |
| **Hybrid GraphRAG** | Relationship/architecture queries trigger a `hybrid` mode that fuses FAISS vector search with multi-hop **Neo4j knowledge-graph** traversal. Falls back to an in-memory graph store with zero dependencies. |
| **Offline evaluation** | `DeterministicJudge` scores every answer on six dimensions with **no second LLM call** — free, fast, deterministic. |
| **Self-healing retry** | When `overall_score < 0.60`, runs query-expansion / force-RAG / wider-k strategies and keeps the best-of-N. |
| **Provider abstraction** | One `LLMProvider` interface; `mock`/`openai`/`anthropic`/`bedrock` swap via env var. SDKs imported lazily. |
| **Health-aware fallback** | A rolling health registry reorders the provider chain healthiest-first; mock is the guaranteed backstop. |
| **Cost & budget guardrails** | Per-request and per-day cost caps that stop runaway retries without crashing the request. |
| **Observability** | Persistent SQLite metrics, P50/P95/P99 latency, score histograms, Prometheus exposition, provider health endpoint. |
| **Precise trace semantics** | Distinguishes a *routing probe* (`retrieval_probe_used`) from *answer grounding* (`answer_context_used`). |

---

## Quick start

Prereqs: Python 3.9+, Node 18+, `make`.

```bash
make install        # backend venv + deps, dashboard deps
make backend        # terminal 1 → API on http://127.0.0.1:8000  (docs at /docs)
make dashboard      # terminal 2 → dashboard on http://localhost:3000
make seed           # terminal 3 → ingest sample docs (once backend is up)
```

The default provider is the **offline mock** — no API keys, no network. The
first backend run downloads the ~80MB embedding model into `backend/.hf_cache`.

Run `make help` to see every target, or `make demo` for the guided flow.

---

## Demo flow

After `make seed`, open the dashboard (or use `curl`) and try:

| Query | What you'll see |
|---|---|
| `How does Aegis route a query?` | **RAG_ANSWER** — grounded in the seeded architecture doc; `answer_context_used: true`. |
| `What is Acme Cloud's refund policy?` | **RAG_ANSWER** — grounded in the seeded FAQ, high groundedness score. |
| `What is 2 + 2?` | **DIRECT_ANSWER** — computed directly, no retrieval, `should_retry: false`. |
| `What does the document say about quantum gravity?` | **NEEDS_CLARIFICATION** — doc-intent query with no relevant passage; Aegis declines to fabricate. |
| `Tell me more` | **NEEDS_CLARIFICATION** — too vague to answer confidently. |

```bash
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is Acme Cloud'\''s refund policy?"}' | jq '{route, answer}'
```

---

## Example trace

A single `/ask` returns the answer plus a full decision trail:

```jsonc
{
  "query": "What is Acme Cloud's refund policy?",
  "route": "RAG_ANSWER",
  "answer": "Based on the retrieved context, ...",
  "trace": {
    "reason": "Strong retrieval match (top score 0.612 >= 0.30).",
    "retrieval_probe_used": true,     // router searched FAISS to decide
    "answer_context_used": true,      // chunks were injected into the prompt
    "generation_mode": "grounded",
    "top_score": 0.612,
    "latency_ms": 42.7,
    "evaluation": {
      "scores": { "relevance": 0.8, "groundedness": 0.78,
                  "completeness": 0.85, "hallucination_risk": 0.18,
                  "confidence": 0.74, "overall_score": 0.79 },
      "should_retry": false
    },
    "retry": null,                    // present only when self-healing engaged
    "generation": {
      "provider_name": "mock", "model_name": "mock",
      "fallback_used": false, "estimated_cost_usd": 0.0
    }
  }
}
```

---

## Hybrid GraphRAG retrieval (Module 10)

Vector search is great at *"find me the passage that says X."* It is bad at
*"how do routing and retries interact?"* — because the answer is not in any
single chunk; it lives in the **relationships between concepts**. Aegis adds a
knowledge graph as a first-class retrieval strategy and **fuses** it with FAISS.

### Why GraphRAG?

- **Multi-hop reasoning.** "Trace the path of a RAG request end-to-end" needs to
  walk `Router → Retrieval → Generation → Evaluation → RetryManager`. A vector
  search returns the most *similar* chunk; a graph traversal returns the *connected*
  ones.
- **Structure over similarity.** Architecture, dependency, workflow, and
  failure-path questions are about edges, not cosine distance.
- **Explainability.** The graph trace shows exactly which entities and
  relationships fed the answer — not a black-box similarity score.

### Why Neo4j?

- Native property graph with a mature, declarative traversal language (Cypher)
  and real indexes/constraints — variable-length `[*1..3]` traversals are a
  one-liner, not a recursive join.
- It is the industry-standard graph database, so this mirrors how a real
  platform team would ship GraphRAG.
- **It is optional.** The `neo4j` driver is lazy-imported and Aegis ships an
  **in-memory graph store** with the identical interface. The default backend is
  `memory`, so the entire test suite and demo run with **zero graph
  infrastructure**. Point `AEGIS_GRAPH_BACKEND=neo4j` at a live server and the
  exact same pipeline persists to Neo4j; if that server is unreachable at
  startup, Aegis logs a warning and **falls back to memory instead of crashing**.

### Vector vs Graph vs Hybrid

| Mode | How it gathers context | Best for | When Aegis uses it |
|---|---|---|---|
| **Vector** | FAISS top-k over MiniLM embeddings | "What is the refund window?" — fact lookup | Default for `RAG_ANSWER`. |
| **Graph** | Entity match → multi-hop traversal → linked chunks | Pure relationship lookups | Available via the `GraphStore` directly. |
| **Hybrid** | **Vector ∪ Graph**, merged into one grounded prompt | "How do routing and retries interact?" | Auto-selected when the router detects an architecture-style query. |

The router flags a query as architecture-style on keywords like *relationship,
depend, architecture, workflow, trace, interact, connect, pipeline, end-to-end,
failure path, component*. Direct factual questions stay `DIRECT_ANSWER`; ordinary
document lookups stay vector `RAG_ANSWER`. Hybrid never *replaces* FAISS — it
**adds** graph context to it.

### How the graph is built

During ingestion, every document runs **two** pipelines in parallel:

```
text ─┬─► chunk → embed (MiniLM) → FAISS            (vector index, unchanged)
      └─► chunk → extract entities → nodes+edges → graph store + [:MENTIONS] chunk links
```

Entities are typed into nine categories — `Component, Route, Provider, Retrieval,
Evaluation, Retry, Metric, Dashboard, Document` — by a deterministic,
alias-based `EntityExtractor`. That extractor sits behind an interface, so an
LLM-based extractor can replace it later **without touching the store, retriever,
or pipeline**. The Aegis architecture ontology itself is seeded as graph data so
the system can answer questions about its own design out of the box.

### Example Cypher

The Neo4j store creates a uniqueness constraint and indexes, then `MERGE`s nodes,
relationships, and chunk links. A traversal looks like:

```cypher
// Multi-hop neighbourhood around the query's anchor entities (max 2 hops),
// then the chunks that mention any traversed entity.
MATCH (a:Entity) WHERE a.name IN $anchors
MATCH path = (a)-[*1..2]-(n:Entity)
WITH collect(DISTINCT n) + collect(DISTINCT a) AS ents, relationships(path) AS rels
UNWIND ents AS e
OPTIONAL MATCH (e)<-[:MENTIONS]-(c:Chunk)
RETURN e, rels, c
```

### Example Graph Trace

A hybrid `/ask` adds a `graph` block (and `retrieval_mode: "hybrid"`) to the trace:

```jsonc
{
  "query": "How do routing and retries interact?",
  "route": "RAG_ANSWER",
  "trace": {
    "retrieval_mode": "hybrid",
    "graph_used": true,
    "generation_mode": "hybrid",
    "graph": {
      "graph_backend": "neo4j",          // or "memory" on the fallback path
      "matched_entities": [
        { "name": "QueryRouter", "category": "Component", "description": "..." },
        { "name": "RetryManager", "category": "Retry", "description": "..." }
      ],
      "traversed_entities": [ /* 6 nodes reached within 2 hops */ ],
      "traversed_relationships": [
        { "source": "QueryRouter", "type": "ROUTES_TO", "target": "RAGPipeline" },
        { "source": "RetryManager", "type": "RETRIES", "target": "RAGPipeline" }
        /* … 9 total … */
      ],
      "graph_chunks": [ /* chunks linked to the traversed entities */ ],
      "graph_score": 0.84,
      "hops": 2,
      "graph_latency_ms": 1.7
    }
  }
}
```

The dashboard renders this as a **Graph Trace** panel (colour-coded entity chips,
the relationship edges, linked chunks, and headline stats), and the *What
happened?* summary explains in plain English: *"Aegis selected hybrid retrieval
because this was a relationship-heavy query. It combined FAISS vector search with
neo4j graph traversal across 6 entities and 9 relationships before generating the
answer."*

### Running with Neo4j (optional)

```bash
docker run -d --name aegis-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/aegispass \
  neo4j:5

pip install neo4j                          # lazy-imported; only needed for this backend
export AEGIS_GRAPH_BACKEND=neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=aegispass
# Browse the graph at http://localhost:7474 after ingesting docs.
```

Omit all of the above and Aegis runs the in-memory graph store — same behaviour,
no infrastructure.

---

## Provider modes

Selected entirely by environment variable — no code changes:

```bash
export AEGIS_PROVIDER=mock        # default: offline, deterministic, zero-cost
export AEGIS_PROVIDER=openai      # + AEGIS_OPENAI_API_KEY
export AEGIS_PROVIDER=anthropic   # + AEGIS_ANTHROPIC_API_KEY
export AEGIS_PROVIDER=bedrock     # + AWS creds via boto3's standard chain

# Keep a backstop so a vendor outage still answers (degraded, never crashing):
export AEGIS_FALLBACK_PROVIDER=mock
```

SDKs are imported lazily, so the full test suite runs with **none** of
`openai` / `anthropic` / `boto3` installed. See
[`backend/README.md`](backend/README.md) for the complete provider, Bedrock,
and pricing configuration.

---

## Testing

```bash
make test           # runs every backend suite with the offline mock provider
make build          # production build of the dashboard (verifies it compiles)
```

The backend ships **16 standalone test suites** (routing, RAG, judge, retry,
provider fallback, Bedrock/cost/metrics, ops, frontend contract, trace
semantics, the semantic mock, and five Module 10 GraphRAG suites — graph store,
Neo4j integration + fallback, graph retrieval, hybrid RAG, and the dashboard
contract). The GraphRAG suites run entirely offline against the in-memory graph
store; the Neo4j suite exercises a live round-trip only if one is reachable and
skips otherwise. Each runs offline with no API keys and can also be executed
directly:

```bash
cd backend && HF_HOME=$PWD/.hf_cache ./venv/bin/python tests/test_module4.py
```

---

## Why this is not just another RAG app

- **It evaluates itself.** Most RAG apps return whatever the model says. Aegis
  scores every answer offline and *acts* on the score.
- **It heals itself.** A measurable retry loop turns a sub-threshold answer into
  a better one — and proves it with a score progression in the trace.
- **It survives provider failure.** A health-aware fallback chain with an
  offline backstop means a vendor outage degrades one request instead of taking
  the system down.
- **It is honest about cost and uncertainty.** Token/cost estimates, budget
  guardrails, and explicit `NEEDS_CLARIFICATION` instead of confident
  fabrication.
- **It is observable end-to-end.** Every decision — route, score, retry,
  provider, latency, cost — is in the trace and on the dashboard.

---

## Repository layout

```
aegis-llm-reliability-mesh/
├── Makefile               # one-command install / run / seed / test / build
├── data/                  # sample documents for `make seed`
├── backend/               # FastAPI service (see backend/README.md)
│   ├── app/               # routers, services, schemas, config
│   │   └── services/graph/  # GraphStore, Neo4j + in-memory backends, retriever
│   ├── scripts/           # run.sh, seed.sh, *_examples.sh
│   └── tests/             # 16 offline test suites (incl. 5 GraphRAG)
└── frontend/              # Next.js 14 observability dashboard
```
