# Aegis — Retrieval Engine + RAG Router

A small FastAPI service that ingests text, splits it into overlapping chunks,
embeds them with `sentence-transformers`, indexes the vectors in FAISS, and
serves semantic similarity search (**Module 1**), plus a query router and basic
RAG generation behind a single `/ask` endpoint (**Module 2**).

- **Module 1 — retrieval:** `/ingest`, `/ingest/file`, `/search`.
- **Module 2 — routing + RAG:** `/ask` classifies each query into
  `DIRECT_ANSWER`, `RAG_ANSWER`, or `NEEDS_CLARIFICATION`, then answers via a
  pluggable generator.
- **Module 3 — LLM-as-a-Judge:** every answer is scored (relevance,
  groundedness, completeness, hallucination risk, confidence).
- **Module 4 — adaptive retry / self-healing:** low-quality answers trigger
  alternate strategies (force-RAG, query expansion, wider breadth) with
  best-of-N selection and a circuit breaker.
- **Module 5 — real provider abstraction:** a vendor-agnostic `LLMProvider`
  interface with `mock` / `openai` / `anthropic` backends, a composable
  fallback chain, and typed error handling. `mock` is the default, so tests and
  demos need no keys or network.
- **Module 6 — AWS Bedrock + observability:** a `bedrock` provider (Anthropic
  Claude on Bedrock via boto3) behind the *same* abstraction, plus per-request
  generation tracing (provider, model, latency, estimated tokens & cost) and an
  in-memory metrics snapshot at `GET /metrics`.
- **Module 7 — ops + dashboard:** persistent SQLite metrics (P50/P95/P99
  latency, score histograms, retry/fallback/cost), a Prometheus exposition at
  `GET /metrics/prometheus`, a health-aware fallback chain with `GET
  /providers/health`, a cost/rate `BudgetGuard`, and a Next.js dashboard
  (`../frontend`).
- **Module 8 — trace precision:** distinguishes a *routing probe*
  (`retrieval_probe_used`) from *answer grounding* (`answer_context_used`), and
  stops penalising `DIRECT_ANSWER` quality for a low routing-probe score.
- **Module 9 — semantic mock:** the offline `MockLLM` now returns lightweight
  deterministic *semantic* answers (arithmetic, concept definitions, and
  Aegis-architecture answers synthesised from retrieved chunks) so demos and
  tests are meaningful without any API keys.

> **Vendor-agnostic by design:** Bedrock is integrated through the same provider
> abstraction as every other backend, so the RAG / retry / evaluation pipeline
> remains vendor-agnostic — no routing, retrieval, or scoring code knows which
> LLM answered.

## Layout

```
backend/
├── app/
│   ├── main.py              # FastAPI app + /health, /stats, router wiring
│   ├── config.py            # env-driven settings (chunk size, model, top_k)
│   ├── logging_config.py    # consistent logging across every stage
│   ├── dependencies.py      # process-wide RetrievalEngine singleton
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response contracts
│   ├── routers/
│   │   ├── ingest.py        # POST /ingest  and  POST /ingest/file
│   │   ├── search.py        # POST /search
│   │   ├── ask.py           # POST /ask  (Module 2)
│   │   ├── metrics.py       # GET /metrics, /metrics/prometheus (Module 6–7)
│   │   └── health.py        # GET /providers/health (Module 7)
│   └── services/
│       ├── chunker.py       # word-window chunking with overlap
│       ├── embedder.py      # sentence-transformers wrapper (normalised)
│       ├── vector_store.py  # FAISS IndexFlatIP + in-memory metadata
│       ├── retrieval.py     # orchestrates chunk -> embed -> store / search
│       ├── router.py        # query router (heuristics + retrieval probe)
│       ├── generator.py     # LLMClient interface + deterministic MockLLM
│       ├── judge.py         # DeterministicJudge (Module 3)
│       ├── evaluator.py     # AnswerEvaluator wrapper + retry thresholds
│       ├── retry.py         # RetryManager: alternate strategies, best-of-N
│       ├── budget.py        # cost/rate guardrail (Module 7)
│       ├── metrics*.py      # in-memory + persistent SQLite metrics
│       ├── providers/       # mock/openai/anthropic/bedrock + fallback chain
│       └── rag.py           # RAGPipeline: route -> retrieve -> ground -> answer
├── tests/                   # 11 standalone, offline suites (test_module*.py)
└── scripts/
    ├── run.sh               # launches the server (sets model cache path)
    ├── seed.sh              # ingests the sample docs in ../data
    ├── curl_examples.sh     # Module 1 smoke test
    └── ask_examples.sh      # Module 2 /ask smoke test
```

> The repo root has a `Makefile` (`make install/backend/dashboard/seed/test`)
> and a Next.js observability dashboard in `../frontend` (Module 7+). See the
> [root README](../README.md) for the full one-command quick start.

## Run it

```bash
cd backend
bash scripts/run.sh          # http://127.0.0.1:8000  (interactive docs at /docs)
```

> **Model cache note:** the first run downloads the embedding model
> (`all-MiniLM-L6-v2`, ~80MB). `run.sh` points the cache at
> `backend/.hf_cache` because the global `~/.cache/huggingface` directory on
> this machine is owned by `root` and isn't writable. If you run `uvicorn`
> directly, set `HF_HOME=$PWD/.hf_cache` yourself.

## Test it

```bash
# From the repo root — runs all 11 suites offline with the mock provider:
make test

# Or individually from backend/ (most are fast; test_retrieval loads the model):
cd backend
HF_HOME=$PWD/.hf_cache ./venv/bin/python tests/test_retrieval.py   # Module 1
./venv/bin/python tests/test_module2.py                           # routing/RAG
./venv/bin/python tests/test_module4.py                           # self-healing
./venv/bin/python tests/test_module9_mock_semantic.py             # semantic mock

# Or against a live server, in a second terminal:
bash scripts/curl_examples.sh   # Module 1
bash scripts/ask_examples.sh    # Module 2
bash scripts/seed.sh            # ingest the sample docs in ../data
```

## Endpoints

| Method | Path           | Purpose                                  |
|--------|----------------|------------------------------------------|
| GET    | `/health`      | Liveness (does not load the model).      |
| POST   | `/ingest`      | Ingest raw text (JSON body).             |
| POST   | `/ingest/file` | Ingest an uploaded UTF-8 text file.      |
| POST   | `/search`      | Top-k semantic search with scores.       |
| POST   | `/ask`         | Route + answer (DIRECT / RAG / CLARIFY).  |
| GET    | `/stats`       | Index size + embedding model info.       |
| GET    | `/metrics`     | Observability snapshot (latency pctiles, scores, cost). |
| GET    | `/metrics/prometheus` | Prometheus exposition format.     |
| GET    | `/providers/health` | Per-provider health + fallback order. |

### `/ask` request body

| Field           | Type            | Default | Meaning                                   |
|-----------------|-----------------|---------|-------------------------------------------|
| `query`         | str             | —       | The user's question (required).           |
| `top_k`         | int?            | config  | Chunks to retrieve for RAG.               |
| `force_route`   | Route?          | null    | Skip the router; force a specific route.  |
| `include_trace` | bool            | true    | Include routing/retrieval trace.          |

The response always contains `route` and `answer`; when `include_trace` is true
it also returns a `trace` with `reason`, `retrieval_used`, `generation_mode`,
`latency_ms`, `top_score`, and the `retrieved` chunks.

## Configuration (env vars, prefix `AEGIS_`)

| Variable                     | Default            | Meaning                          |
|------------------------------|--------------------|----------------------------------|
| `AEGIS_EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | sentence-transformers model.     |
| `AEGIS_CHUNK_SIZE`           | `200`              | Words per chunk.                 |
| `AEGIS_CHUNK_OVERLAP`        | `40`               | Words shared between chunks.     |
| `AEGIS_DEFAULT_TOP_K`        | `5`                | Default results per search.      |
| `AEGIS_RAG_SCORE_THRESHOLD`  | `0.30`             | Top score at/above → RAG_ANSWER. |
| `AEGIS_CLARIFICATION_SCORE_FLOOR` | `0.10`        | Below → too weak to ground on.   |
| `AEGIS_MIN_QUERY_WORDS`      | `2`                | Shorter non-greetings → clarify. |
| `AEGIS_LOG_LEVEL`            | `INFO`             | Logging level.                   |

### LLM provider settings (Modules 5–6)

| Variable                     | Default     | Meaning                                        |
|------------------------------|-------------|------------------------------------------------|
| `AEGIS_PROVIDER`             | `mock`      | Primary backend: `mock`/`openai`/`anthropic`/`bedrock`. |
| `AEGIS_FALLBACK_PROVIDER`    | `mock`      | Backend tried if the primary raises (`none` to disable). |
| `AEGIS_MODEL_NAME`           | `""`        | Model for openai/anthropic (empty → provider default). |
| `AEGIS_TEMPERATURE`          | `0.0`       | Sampling temperature for real providers.       |
| `AEGIS_REQUEST_TIMEOUT`      | `30.0`      | Per-request timeout (seconds).                 |
| `AEGIS_MAX_TOKENS`           | `1024`      | Max output tokens.                             |
| `AEGIS_OPENAI_API_KEY`       | `""`        | OpenAI key (never logged).                     |
| `AEGIS_ANTHROPIC_API_KEY`    | `""`        | Anthropic key (never logged).                  |
| `AEGIS_AWS_REGION`           | `us-east-1` | Region hosting Bedrock.                        |
| `AEGIS_BEDROCK_MODEL_ID`     | `""`        | Bedrock model id (empty → Claude 3.5 Sonnet default). |
| `AEGIS_PRICE_<FAMILY>_INPUT` | per-family  | Override input price (USD/1K tokens), e.g. `AEGIS_PRICE_BEDROCK_INPUT`. |
| `AEGIS_PRICE_<FAMILY>_OUTPUT`| per-family  | Override output price (USD/1K tokens).         |

## LLM providers

Generation runs through a single `LLMProvider.generate(system, user) -> str`
interface (see `app/services/providers/`). The factory in `dependencies.py` is
the *only* place that maps a provider name to a class; the pipeline never sees
the wiring. SDKs are imported lazily, so selecting `mock` never touches
`openai` / `anthropic` / `boto3`, and the full test suite runs with none of them
installed.

Install only what you need:

```bash
pip install -r requirements-providers.txt   # all (openai + anthropic + boto3)
pip install boto3                            # AWS Bedrock only
```

### AWS Bedrock

```bash
export AEGIS_PROVIDER=bedrock
export AEGIS_AWS_REGION=us-east-1
# Optional — defaults to anthropic.claude-3-5-sonnet-20240620-v1:0
export AEGIS_BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
# Keep mock as a fallback so a Bedrock outage still answers (degraded):
export AEGIS_FALLBACK_PROVIDER=mock
bash scripts/run.sh
```

**Credentials** are resolved by boto3's standard chain — environment variables
(`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`), a shared
profile (`~/.aws/credentials`), or an attached IAM role. Aegis never reads or
logs them directly.

**Region matters:** Bedrock model availability is region-specific, and the model
id must be enabled for your account in that region (request access in the
Bedrock console under *Model access*). A model that works in `us-east-1` may not
exist in another region.

**Minimum IAM permission** to invoke a model:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel"],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
    }
  ]
}
```

Errors are mapped to the typed `ProviderError` hierarchy: missing
credentials / access-denied → `ProviderConfigError`, throttling / 5xx →
`ProviderAPIError`, timeouts → `ProviderTimeoutError`. A failure degrades the
single request (and engages any fallback) instead of crashing.

```bash
curl -s http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"What does the document say about refunds?"}' | jq .trace.generation
# {
#   "provider_name": "bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0",
#   "model_name": "anthropic.claude-3-5-sonnet-20240620-v1:0",
#   "provider_latency_ms": 812.4,
#   "fallback_used": false,
#   "fallback_chain": ["bedrock:anthropic.claude-...", "mock"],
#   "estimated_input_tokens": 156,
#   "estimated_output_tokens": 88,
#   "estimated_cost_usd": 0.001788
# }
```

## Observability (`/metrics`)

Every `/ask` records one request metric; `GET /metrics` returns a snapshot of
process-local aggregates. The collector is in-memory and **ephemeral** — it
resets on restart (swap the backend later without touching callers).

```bash
curl -s http://127.0.0.1:8000/metrics | jq .
# {
#   "total_requests": 12,
#   "requests_by_provider": {"bedrock:anthropic.claude-...": 11, "mock": 1},
#   "fallback_count": 1,
#   "degraded_response_count": 0,
#   "average_latency_ms": 734.51,
#   "retry_rate": 0.0833,
#   "average_overall_score": 0.8123,
#   "estimated_cost_usd_total": 0.021456
# }
```

> **Token & cost are estimates.** Counts use a chars/4 heuristic and per-family
> prices are coarse placeholders (tunable via `AEGIS_PRICE_*`) — useful for
> relative comparison and budget signals, not billing-grade accuracy.

```bash
# Provider/observability tests (offline — fake boto3, no AWS, no SDK install):
./venv/bin/python tests/test_module5.py   # provider abstraction + fallback
./venv/bin/python tests/test_module6.py   # bedrock + cost + metrics + tracing
```
