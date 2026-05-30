# Aegis — Retrieval Engine + RAG Router

A small FastAPI service that ingests text, splits it into overlapping chunks,
embeds them with `sentence-transformers`, indexes the vectors in FAISS, and
serves semantic similarity search (**Module 1**), plus a query router and basic
RAG generation behind a single `/ask` endpoint (**Module 2**).

- **Module 1 — retrieval:** `/ingest`, `/ingest/file`, `/search`.
- **Module 2 — routing + RAG:** `/ask` classifies each query into
  `DIRECT_ANSWER`, `RAG_ANSWER`, or `NEEDS_CLARIFICATION`, then answers via a
  pluggable generator (a deterministic `MockLLM` for now; Bedrock/OpenAI drops
  in later). Generation is intentionally a mock — no cloud credentials needed.

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
│   │   └── search.py        # POST /search
│   ├── routers/
│   │   ├── ingest.py        # POST /ingest  and  POST /ingest/file
│   │   ├── search.py        # POST /search
│   │   └── ask.py           # POST /ask  (Module 2)
│   └── services/
│       ├── chunker.py       # word-window chunking with overlap
│       ├── embedder.py      # sentence-transformers wrapper (normalised)
│       ├── vector_store.py  # FAISS IndexFlatIP + in-memory metadata
│       ├── retrieval.py     # orchestrates chunk -> embed -> store / search
│       ├── router.py        # query router (heuristics + retrieval probe)
│       ├── generator.py     # LLMClient interface + deterministic MockLLM
│       └── rag.py           # RAGPipeline: route -> retrieve -> ground -> answer
├── tests/
│   ├── test_retrieval.py    # Module 1: chunker + end-to-end ingest/search
│   └── test_module2.py      # Module 2: routing, RAG, clarification, forced
└── scripts/
    ├── run.sh               # launches the server (sets model cache path)
    ├── curl_examples.sh     # Module 1 smoke test
    └── ask_examples.sh      # Module 2 /ask smoke test
```

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
cd backend

# Module 1 — fast chunker tests + slower end-to-end (loads the model):
HF_HOME=$PWD/.hf_cache ./venv/bin/python tests/test_retrieval.py

# Module 2 — routing/RAG tests (fast, no model needed — uses a FakeEngine):
./venv/bin/python tests/test_module2.py

# Or against a live server, in a second terminal:
bash scripts/curl_examples.sh   # Module 1
bash scripts/ask_examples.sh    # Module 2
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
