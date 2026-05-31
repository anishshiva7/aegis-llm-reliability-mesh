"use client";

import { useState } from "react";
import { ingest, ApiError } from "@/lib/api";
import type { IngestResponse } from "@/lib/types";
import { Panel, Badge, ErrorBox, Spinner } from "./ui";

// One-click sample documents. Clicking POPULATES the fields (text + source)
// but never auto-submits — the user reviews, then presses "Ingest document".
const SAMPLE_DOCS: { label: string; source: string; text: string }[] = [
  {
    label: "Pricing & Refund Policy",
    source: "pricing-policy",
    text: `Aegis Pricing & Refund Policy

Aegis offers three plans. The Free tier includes 1,000 requests per month with mock-provider answers only. The Pro tier costs $49 per month and unlocks real provider routing (OpenAI, Anthropic, Bedrock), the self-healing retry loop, and the metrics dashboard. The Enterprise tier is custom-priced and adds SSO, a private model gateway, and a per-day cost guardrail you control.

Refunds: Pro subscriptions are refundable in full within 14 days of purchase, no questions asked. After 14 days, refunds are prorated for the unused portion of the current billing month. Enterprise contracts follow the refund terms negotiated in the order form. To request a refund, email billing@aegis.example with your account ID.

Overage: if you exceed your plan's monthly request quota, additional requests are billed at $0.002 each. You can set a hard daily spend cap in Settings to prevent surprise charges.`,
  },
  {
    label: "Aegis Architecture",
    source: "aegis-architecture",
    text: `Aegis Architecture Overview

Aegis is a self-optimizing LLM reliability mesh. A single /ask request flows through five stages. First, the Query Router inspects the question and runs one retrieval probe against the FAISS vector index to decide a route: DIRECT_ANSWER for simple or conversational questions, RAG_ANSWER when relevant documents exist, or NEEDS_CLARIFICATION when the query is too vague.

Second, generation runs. RAG answers are grounded in retrieved chunks; direct answers use a plain prompt. Third, an LLM-as-a-Judge evaluator scores the answer on relevance, groundedness, completeness, hallucination risk, and confidence, producing an overall score.

Fourth, if the answer is below quality thresholds, a self-healing retry loop runs alternate strategies (such as query expansion) and keeps the best-scoring attempt, governed by a daily cost guardrail. Fifth, every request is recorded to a persistent metrics store and exposed via /metrics and a Prometheus endpoint.`,
  },
  {
    label: "Provider Fallback",
    source: "provider-fallback",
    text: `Provider Fallback Strategy

Aegis never depends on a single model provider. Providers are arranged in an ordered fallback chain — for example OpenAI first, then Anthropic, with the deterministic mock provider always kept last as a guaranteed backstop.

On each request the chain is reordered healthiest-first using a rolling health registry that tracks consecutive failures, total successes, and total failures per provider. A provider is marked healthy, degraded, or unhealthy based on its recent outcomes. When the primary provider raises an error, the FallbackProvider transparently advances to the next provider in the chain and records the failure; the answer's trace reports fallback_used = true so the dashboard can surface it.

If every real provider fails, the mock provider answers and the response is flagged as degraded rather than crashing the request. This design keeps the /ask endpoint available even during multi-provider outages.`,
  },
];

const SUGGESTED_QUERIES: Record<string, string[]> = {
  "pricing-policy": [
    "How much does the Pro plan cost?",
    "Can I get a refund after 20 days?",
  ],
  "aegis-architecture": [
    "What is Aegis and how does its reliability mesh work?",
    "How does the self-healing retry loop decide to retry?",
  ],
  "provider-fallback": [
    "Summarize the provider fallback strategy.",
    "What happens when every provider fails?",
  ],
};

export function IngestPanel({
  onSuggestQuery,
}: {
  onSuggestQuery?: (query: string) => void;
}) {
  const [text, setText] = useState("");
  const [source, setSource] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<IngestResponse | null>(null);

  function loadSample(sample: (typeof SAMPLE_DOCS)[number]) {
    // Populate the fields only — the user reviews and submits manually.
    setText(sample.text);
    setSource(sample.source);
    setResult(null);
    setError(null);
  }

  async function submit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!text.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await ingest({
        text: text.trim(),
        source: source.trim() || null,
      });
      setResult(res);
    } catch (err) {
      const msg =
        err instanceof ApiError
          ? err.message
          : "Unexpected error while contacting the backend.";
      setError(msg);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  const suggestions = result ? SUGGESTED_QUERIES[result.source] ?? [] : [];

  return (
    <Panel
      title="Ingest Document"
      subtitle="POST /ingest · add knowledge so the router can ground RAG answers"
    >
      <form onSubmit={submit} className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-600">Load a sample:</span>
          {SAMPLE_DOCS.map((s) => (
            <button
              key={s.source}
              type="button"
              onClick={() => loadSample(s)}
              className="rounded-full border border-ink-700 bg-ink-800/60 px-3 py-1 text-xs text-slate-400 transition-colors hover:border-accent/50 hover:text-slate-200"
            >
              {s.label}
            </button>
          ))}
        </div>

        <div>
          <label className="kv-label mb-1.5 block">Paste document text</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={6}
            placeholder="Paste the document text you want Aegis to learn…"
            className="w-full resize-y rounded-lg border border-ink-700 bg-ink-900/70 px-3.5 py-3 text-sm text-slate-100 placeholder:text-slate-600 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/50"
          />
        </div>

        <div className="flex flex-wrap items-end gap-4">
          <div className="min-w-[15rem] flex-1">
            <label className="kv-label mb-1.5 block">Source name</label>
            <input
              type="text"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder="e.g. pricing-policy"
              className="w-full rounded-lg border border-ink-700 bg-ink-900/70 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/50"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !text.trim()}
            className="rounded-lg bg-accent px-5 py-2 text-sm font-semibold text-white shadow-lg transition-colors hover:bg-accent-soft disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? "Ingesting…" : "Ingest document"}
          </button>
        </div>
      </form>

      {loading && (
        <div className="mt-4">
          <Spinner label="Embedding & indexing chunks…" />
        </div>
      )}

      {error && !loading && (
        <div className="mt-4">
          <ErrorBox message={error} />
        </div>
      )}

      {result && !loading && (
        <div className="mt-4 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="good">ingested</Badge>
            <span>
              Added <strong>{result.chunks_created}</strong> chunk
              {result.chunks_created === 1 ? "" : "s"} from{" "}
              <span className="font-mono">{result.source}</span>. Index now holds{" "}
              <strong>{result.total_chunks_in_index}</strong> chunk
              {result.total_chunks_in_index === 1 ? "" : "s"}.
            </span>
          </div>
          {suggestions.length > 0 && (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className="text-xs text-emerald-300/70">Try asking:</span>
              {suggestions.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => onSuggestQuery?.(q)}
                  className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-200 transition-colors hover:border-emerald-400 hover:text-emerald-100"
                >
                  {q}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
