"use client";

import type { RouteTrace } from "@/lib/types";
import { Panel, Badge } from "./ui";
import { fixed } from "@/lib/format";

export function RetrievalPanel({ trace }: { trace: RouteTrace }) {
  const chunks = trace.retrieved ?? [];
  // Module 8 — distinguish a routing probe from actual answer grounding.
  // Fall back to the legacy retrieval_used flag for older payloads.
  const probeUsed = trace.retrieval_probe_used ?? trace.retrieval_used;
  const contextUsed =
    trace.answer_context_used ?? (trace.retrieval_used && chunks.length > 0);

  return (
    <Panel
      title="Retrieval Trace"
      subtitle="What the vector store was used for on this request"
      right={
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Badge tone={probeUsed ? "accent" : "neutral"}>
            {probeUsed ? "retrieval probe ran" : "no retrieval probe"}
          </Badge>
          <Badge tone={contextUsed ? "good" : "neutral"}>
            {contextUsed ? "document context used" : "no document context"}
          </Badge>
          {trace.top_score != null && (
            <Badge tone="neutral">top {fixed(trace.top_score, 3)}</Badge>
          )}
        </div>
      }
    >
      {chunks.length === 0 ? (
        <p className="text-sm text-slate-500">
          {probeUsed
            ? "A routing probe searched the index, but no chunks were used to ground the answer (direct or clarification route)."
            : "No retrieval was performed for this query."}
        </p>
      ) : (
        <ol className="space-y-3">
          {chunks.map((c, i) => {
            const scorePct = Math.max(0, Math.min(1, c.score)) * 100;
            return (
              <li
                key={`${c.chunk_id}-${i}`}
                className="rounded-lg border border-ink-700/60 bg-ink-800/50 p-3"
              >
                <div className="mb-2 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <span className="grid h-5 w-5 place-items-center rounded bg-ink-700 font-mono text-[10px] text-slate-300">
                      {i + 1}
                    </span>
                    <span className="font-medium text-slate-300">
                      {c.source}
                    </span>
                    <span className="text-slate-600">·</span>
                    <span>chunk #{c.chunk_index}</span>
                    <span className="text-slate-600">·</span>
                    <span>id {c.chunk_id}</span>
                  </div>
                  <span className="font-mono text-xs text-accent-soft">
                    {fixed(c.score, 3)}
                  </span>
                </div>
                <div className="mb-2 h-1 overflow-hidden rounded-full bg-ink-700">
                  <div
                    className="h-full rounded-full bg-accent"
                    style={{ width: `${scorePct}%` }}
                  />
                </div>
                <p className="line-clamp-4 text-sm leading-relaxed text-slate-300">
                  {c.text}
                </p>
              </li>
            );
          })}
        </ol>
      )}
    </Panel>
  );
}
