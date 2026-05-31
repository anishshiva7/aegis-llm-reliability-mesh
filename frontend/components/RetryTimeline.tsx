"use client";

import type { RetryTrace } from "@/lib/types";
import { Panel, Badge, BoolBadge } from "./ui";
import { pct, scoreColor } from "@/lib/format";

export function RetryTimeline({ retry }: { retry: RetryTrace }) {
  return (
    <Panel
      title="Self-Healing Retry Timeline"
      subtitle="Best-of-N attempts driven by the evaluator"
      right={
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Badge tone="neutral">{retry.retry_count} retries</Badge>
          <BoolBadge
            value={retry.degraded_response}
            trueLabel="degraded"
            falseLabel="resolved"
          />
          {retry.budget_blocked && (
            <Badge tone="warn">budget blocked</Badge>
          )}
        </div>
      }
    >
      <ol className="relative space-y-3 before:absolute before:left-[11px] before:top-2 before:bottom-2 before:w-px before:bg-ink-700">
        {retry.attempts.map((a) => {
          const selected = a.attempt === retry.selected_best_attempt;
          return (
            <li key={a.attempt} className="relative pl-8">
              <span
                className={[
                  "absolute left-0 top-1 grid h-6 w-6 place-items-center rounded-full border text-[11px] font-semibold",
                  selected
                    ? "border-emerald-400 bg-emerald-500/20 text-emerald-300"
                    : "border-ink-600 bg-ink-800 text-slate-400",
                ].join(" ")}
              >
                {a.attempt}
              </span>
              <div
                className={[
                  "rounded-lg border p-3",
                  selected
                    ? "border-emerald-500/40 bg-emerald-500/[0.06]"
                    : "border-ink-700/60 bg-ink-800/50",
                ].join(" ")}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-xs text-slate-300">
                    {a.strategy}
                  </span>
                  <Badge tone="neutral">{a.route}</Badge>
                  {selected && <Badge tone="good">selected best</Badge>}
                  {a.should_retry && <Badge tone="warn">flagged retry</Badge>}
                </div>
                <div className="mt-2 flex items-center gap-4 text-xs text-slate-400">
                  <span>
                    score{" "}
                    <span className={`font-mono ${scoreColor(a.overall_score)}`}>
                      {pct(a.overall_score)}
                    </span>
                  </span>
                  <span>
                    confidence{" "}
                    <span className="font-mono text-slate-200">
                      {pct(a.confidence)}
                    </span>
                  </span>
                </div>
              </div>
            </li>
          );
        })}
      </ol>

      {retry.score_progression.length > 1 && (
        <p className="mt-4 text-xs text-slate-500">
          Score progression:{" "}
          <span className="font-mono text-slate-300">
            {retry.score_progression.map((s) => pct(s)).join(" → ")}
          </span>
        </p>
      )}
    </Panel>
  );
}
