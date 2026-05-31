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
                         │       chunk → embed (MiniLM) → FAISS top-k   │
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

The backend ships **11 standalone test suites** (routing, RAG, judge, retry,
provider fallback, Bedrock/cost/metrics, ops, frontend contract, trace
semantics, and the semantic mock). Each runs offline with no API keys and can
also be executed directly:

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
│   ├── scripts/           # run.sh, seed.sh, *_examples.sh
│   └── tests/             # 11 offline test suites
└── frontend/              # Next.js 14 observability dashboard
```
