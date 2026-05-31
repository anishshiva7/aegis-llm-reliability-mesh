"use client";

import type { GenerationTrace } from "@/lib/types";
import { Panel, Badge, KV } from "./ui";
import { ms, usd } from "@/lib/format";

export function GenerationPanel({ gen }: { gen: GenerationTrace }) {
  const source = gen.token_usage_source;
  return (
    <Panel
      title="Generation / Provider"
      subtitle="Which provider answered and what it cost"
      right={
        <div className="flex items-center gap-2">
          <Badge tone={source === "provider" ? "good" : "neutral"}>
            tokens: {source}
          </Badge>
          {gen.fallback_used && <Badge tone="warn">fallback used</Badge>}
        </div>
      }
    >
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <KV label="Provider">{gen.provider_name}</KV>
        <KV label="Model">{gen.model_name}</KV>
        <KV label="Provider latency">{ms(gen.provider_latency_ms)}</KV>
        <KV label="Input tokens">{gen.estimated_input_tokens}</KV>
        <KV label="Output tokens">{gen.estimated_output_tokens}</KV>
        <KV label="Cost (USD)">{usd(gen.estimated_cost_usd)}</KV>
      </div>

      <div className="mt-5 border-t border-ink-700/50 pt-4">
        <p className="kv-label mb-2">Fallback chain (priority order)</p>
        <div className="flex flex-wrap items-center gap-2">
          {gen.fallback_chain.length === 0 ? (
            <span className="text-sm text-slate-500">—</span>
          ) : (
            gen.fallback_chain.map((p, i) => (
              <span key={`${p}-${i}`} className="flex items-center gap-2">
                <span
                  className={[
                    "rounded-md border px-2 py-1 font-mono text-xs",
                    p === gen.provider_name
                      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                      : "border-ink-600 bg-ink-800 text-slate-400",
                  ].join(" ")}
                >
                  {p}
                </span>
                {i < gen.fallback_chain.length - 1 && (
                  <span className="text-slate-600">→</span>
                )}
              </span>
            ))
          )}
        </div>
      </div>
    </Panel>
  );
}
