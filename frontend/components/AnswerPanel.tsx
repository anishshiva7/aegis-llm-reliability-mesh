"use client";

import type { AskResponse } from "@/lib/types";
import { Panel, Badge, BoolBadge, KV } from "./ui";

const routeTone: Record<string, "good" | "accent" | "warn"> = {
  RAG_ANSWER: "accent",
  DIRECT_ANSWER: "good",
  NEEDS_CLARIFICATION: "warn",
};

export function AnswerPanel({ data }: { data: AskResponse }) {
  const trace = data.trace ?? null;
  const gen = trace?.generation ?? null;
  const degraded = trace?.retry?.degraded_response ?? false;
  const fallbackUsed = gen?.fallback_used ?? false;

  return (
    <Panel
      title="Answer"
      subtitle="Final response served by the reliability mesh"
      right={
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Badge tone={routeTone[data.route] ?? "neutral"}>{data.route}</Badge>
          <BoolBadge
            value={fallbackUsed}
            trueLabel="served by fallback provider"
            falseLabel="served by primary provider"
            trueTone="warn"
            falseTone="good"
          />
          <BoolBadge
            value={degraded}
            trueLabel="degraded"
            falseLabel="healthy"
          />
        </div>
      }
    >
      <p className="whitespace-pre-wrap text-[15px] leading-relaxed text-slate-100">
        {data.answer}
      </p>

      <div className="mt-5 grid grid-cols-2 gap-4 border-t border-ink-700/50 pt-4 sm:grid-cols-4">
        <KV label="Route">{data.route}</KV>
        <KV label="Provider">{gen?.provider_name ?? "—"}</KV>
        <KV label="Model">{gen?.model_name ?? "—"}</KV>
        <KV label="Generation mode">{trace?.generation_mode ?? "—"}</KV>
      </div>

      {trace?.generation_error && (
        <div className="mt-4 rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          <span className="font-semibold">Generation error:</span>{" "}
          {trace.generation_error}
        </div>
      )}

      {trace?.reason && (
        <p className="mt-4 text-xs text-slate-500">
          <span className="uppercase tracking-wide">Router reason:</span>{" "}
          {trace.reason}
        </p>
      )}
    </Panel>
  );
}
