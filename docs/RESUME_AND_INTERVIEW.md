# Aegis — Resume & Interview Packaging

A ready-to-use set of talking points for putting Aegis on a resume and walking
through it in interviews.

---

## Polished project description

> **Aegis — Self-Healing LLM Reliability Mesh.** A production-minded RAG service
> (FastAPI + FAISS + sentence-transformers, with a Next.js observability
> dashboard) that treats answer quality as a measurable, actionable signal.
> Every request is adaptively routed (direct / retrieval-grounded /
> clarification), generated through a vendor-agnostic provider abstraction
> (mock / OpenAI / Anthropic / AWS Bedrock) with a health-aware fallback chain,
> then scored offline by a deterministic six-dimension judge. Answers that fall
> below a quality threshold trigger a self-healing retry loop (query expansion,
> forced grounding, wider retrieval) that keeps the best-scoring attempt. Cost
> guardrails, persistent metrics, Prometheus exposition, and full per-request
> tracing make every routing, scoring, and retry decision observable.

---

## Two strongest resume bullets

- **Built a self-healing RAG reliability mesh (FastAPI, FAISS, Next.js) that
  scores every LLM answer offline across six quality dimensions and
  automatically retries sub-threshold answers via query-expansion / forced-
  grounding / wider-retrieval strategies, keeping the best-of-N result** — turning
  silent LLM failures into measured, self-correcting outcomes.

- **Designed a vendor-agnostic provider layer (mock / OpenAI / Anthropic / AWS
  Bedrock) with a health-aware fallback chain, per-request cost/latency tracing,
  budget guardrails, and Prometheus-backed observability** — so a provider outage
  degrades a single request instead of taking the system down, all behind one
  `/ask` endpoint swappable by a single environment variable.

---

## 30-second interview explanation

> "Aegis is a RAG system built around the idea that you shouldn't trust an LLM
> answer just because it sounds confident. Every request gets routed —
> direct, retrieval-grounded, or 'I need clarification' — then the answer is
> scored offline on six dimensions like groundedness and hallucination risk. If
> the score is too low, a self-healing loop retries with different strategies
> and keeps the best one. It's provider-agnostic with a fallback chain, so an
> OpenAI outage just falls back to Anthropic or an offline mock, and everything —
> routing, scores, retries, cost — is observable in a dashboard."

---

## 90-second deep technical explanation

> "The whole flow lives in a `RAGPipeline` orchestrator behind a single `/ask`
> endpoint. First a `QueryRouter` applies cheap structural heuristics, then fires
> *one* FAISS retrieval probe — the top cosine similarity decides between a
> grounded RAG answer, a direct answer, or clarification. Crucially I separated
> the *routing probe* from *answer grounding* in the trace: a weak probe can
> still route to a direct answer without polluting the grounding score, which was
> a real bug I fixed — direct answers were being penalised for a low probe score.
>
> Generation goes through an `LLMProvider` interface. There's exactly one factory
> that maps a provider name to a class, so no pipeline code knows whether OpenAI,
> Anthropic, Bedrock, or the mock answered. Providers are wrapped in a
> `FallbackProvider` with a rolling health registry that reorders the chain
> healthiest-first and keeps the offline mock as a guaranteed backstop.
>
> Then a `DeterministicJudge` scores the answer — fully offline, no second LLM
> call — on relevance, groundedness, completeness, hallucination risk,
> confidence, and a weighted overall score. If overall drops below 0.60, a
> `RetryManager` runs alternate strategies, re-evaluates each, and keeps the
> best-of-N, with a `BudgetGuard` capping retry spend. Everything is recorded to
> SQLite and surfaced via `/metrics`, a Prometheus endpoint, and a Next.js
> dashboard. It runs fully offline with a deterministic mock provider, so the 11
> test suites need no API keys."

---

## Why this is differentiated

> "Most RAG portfolio projects are a thin wrapper: embed, retrieve, stuff the
> prompt, return whatever the model says. Aegis is differentiated because it
> closes the quality loop. It *measures* answer quality offline and *acts* on it
> with a self-healing retry that demonstrably raises scores; it treats provider
> reliability as a first-class concern with a health-aware fallback chain and an
> offline backstop; and it's observable end-to-end — every route, score, retry,
> provider, and cost estimate is in the trace and on a live dashboard. It also
> shows engineering judgment: clean seams (one provider factory, a pluggable
> classifier hook), deterministic offline tests, cost guardrails, and an honest
> `NEEDS_CLARIFICATION` path instead of confident fabrication. It reads like a
> platform/reliability mindset applied to LLMs, not a tutorial follow-along."

---

## Anticipated tough questions (and honest answers)

- **"Your judge is heuristic, not a real LLM-as-judge — isn't that weak?"**
  Yes, deliberately. It's offline, free, deterministic, and good enough to drive
  a retry decision. The seam is clean: swapping in an LLM judge is a one-class
  change. I chose determinism so the reliability behaviour itself is testable.

- **"The mock provider isn't a real model."** Correct — it's a deterministic
  semantic stub so the *system* (routing, scoring, retry, fallback, metrics) is
  fully demonstrable and testable with zero keys. Point `AEGIS_PROVIDER` at a
  real vendor and nothing else changes.

- **"Metrics are in-process and ephemeral."** True — it's a weekend-MVP posture.
  The store is behind an interface, so moving to a shared DB or pushing to a real
  Prometheus/Grafana stack is additive, not a rewrite.
