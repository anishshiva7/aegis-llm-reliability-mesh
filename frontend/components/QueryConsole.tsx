"use client";

import { useState } from "react";
import { ask, ApiError, API_BASE_URL } from "@/lib/api";
import type { AskResponse, Route } from "@/lib/types";
import { Panel, Spinner, ErrorBox, JsonPanel } from "./ui";
import { IngestPanel } from "./IngestPanel";
import { WhatHappened } from "./WhatHappened";
import { AnswerPanel } from "./AnswerPanel";
import { RetrievalPanel } from "./RetrievalPanel";
import { GraphTracePanel } from "./GraphTracePanel";
import { EvaluationPanel } from "./EvaluationPanel";
import { RetryTimeline } from "./RetryTimeline";
import { GenerationPanel } from "./GenerationPanel";

const ROUTES: { value: "" | Route; label: string }[] = [
  { value: "", label: "Auto (router decides)" },
  { value: "RAG_ANSWER", label: "Force RAG_ANSWER" },
  { value: "DIRECT_ANSWER", label: "Force DIRECT_ANSWER" },
  { value: "NEEDS_CLARIFICATION", label: "Force NEEDS_CLARIFICATION" },
];

const SAMPLES = [
  "How do routing and retries interact?",
  "Trace the path of a RAG request end-to-end.",
  "How does the self-healing retry loop decide to retry?",
  "Summarize the provider fallback strategy.",
];

export function QueryConsole() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState<string>("");
  const [forceRoute, setForceRoute] = useState<"" | Route>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);

  async function submit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await ask({
        query: query.trim(),
        top_k: topK ? Number(topK) : null,
        force_route: forceRoute || null,
        include_trace: true,
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

  const trace = result?.trace ?? null;

  return (
    <div className="space-y-6">
      <IngestPanel onSuggestQuery={(q) => setQuery(q)} />

      <Panel
        title="Query Console"
        subtitle={`POST /ask · backend ${API_BASE_URL}`}
      >
        <form onSubmit={submit} className="space-y-4">
          <div>
            <label className="kv-label mb-1.5 block">Query</label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") submit();
              }}
              rows={3}
              placeholder="Ask Aegis a question…  (⌘/Ctrl + Enter to submit)"
              className="w-full resize-y rounded-lg border border-ink-700 bg-ink-900/70 px-3.5 py-3 text-sm text-slate-100 placeholder:text-slate-600 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/50"
            />
          </div>

          <div className="flex flex-wrap items-end gap-4">
            <div className="w-28">
              <label className="kv-label mb-1.5 block">top_k</label>
              <input
                type="number"
                min={1}
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                placeholder="auto"
                className="w-full rounded-lg border border-ink-700 bg-ink-900/70 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/50"
              />
            </div>
            <div className="min-w-[15rem] flex-1">
              <label className="kv-label mb-1.5 block">force_route</label>
              <select
                value={forceRoute}
                onChange={(e) => setForceRoute(e.target.value as "" | Route)}
                className="w-full rounded-lg border border-ink-700 bg-ink-900/70 px-3 py-2 text-sm text-slate-100 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/50"
              >
                {ROUTES.map((r) => (
                  <option key={r.value} value={r.value}>
                    {r.label}
                  </option>
                ))}
              </select>
            </div>
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="rounded-lg bg-accent px-5 py-2 text-sm font-semibold text-white shadow-lg transition-colors hover:bg-accent-soft disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? "Running…" : "Run query"}
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className="text-xs text-slate-600">Try:</span>
            {SAMPLES.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setQuery(s)}
                className="rounded-full border border-ink-700 bg-ink-800/60 px-3 py-1 text-xs text-slate-400 transition-colors hover:border-accent/50 hover:text-slate-200"
              >
                {s}
              </button>
            ))}
          </div>
        </form>
      </Panel>

      {loading && (
        <div className="panel panel-body">
          <Spinner label="Routing → retrieving → generating → evaluating → self-healing…" />
        </div>
      )}

      {error && !loading && (
        <Panel title="Request failed" subtitle="The /ask call did not succeed">
          <ErrorBox message={error} />
          <p className="mt-3 text-xs text-slate-500">
            Confirm the backend is running and reachable at{" "}
            <span className="font-mono text-slate-400">{API_BASE_URL}</span>.
            Override it with{" "}
            <span className="font-mono text-slate-400">
              NEXT_PUBLIC_AEGIS_API_BASE_URL
            </span>
            .
          </p>
        </Panel>
      )}

      {result && !loading && (
        <div className="space-y-6">
          <WhatHappened data={result} />
          <AnswerPanel data={result} />

          <div className="grid gap-6 lg:grid-cols-2">
            {trace && <RetrievalPanel trace={trace} />}
            {trace?.evaluation && (
              <EvaluationPanel evaluation={trace.evaluation} />
            )}
          </div>

          {/* Self-hides for vector/direct queries — only renders on graph/hybrid */}
          {trace && <GraphTracePanel trace={trace} />}

          <div className="grid gap-6 lg:grid-cols-2">
            {trace?.retry && <RetryTimeline retry={trace.retry} />}
            {trace?.generation && <GenerationPanel gen={trace.generation} />}
          </div>

          <JsonPanel data={result} />
        </div>
      )}
    </div>
  );
}
