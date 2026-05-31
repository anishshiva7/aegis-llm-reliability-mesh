"use client";

import type { AskResponse } from "@/lib/types";
import { Panel, Badge } from "./ui";
import { pct, fixed, scoreLabel } from "@/lib/format";

/**
 * Plain-English summary of a result, generated entirely CLIENT-SIDE from the
 * trace (Module 8 — Part B). No extra backend call. Translates the route,
 * retrieval, evaluation, retry, and fallback signals into a few sentences a
 * non-expert can read at a glance.
 */
function buildSummary(data: AskResponse): { sentences: string[]; tone: "good" | "warn" | "bad" } {
  const trace = data.trace ?? null;
  const sentences: string[] = [];

  if (!trace) {
    return {
      sentences: ["Aegis returned an answer without a detailed trace."],
      tone: "good",
    };
  }

  const route = data.route;
  const evalResult = trace.evaluation ?? null;
  const scores = evalResult?.scores ?? null;
  const gen = trace.generation ?? null;
  const retry = trace.retry ?? null;
  const provider = gen?.provider_name ?? null;
  const chunkCount = trace.retrieved?.length ?? 0;
  const topScore = trace.top_score ?? null;

  // 1) What route was taken and why, in plain terms.
  if (route === "RAG_ANSWER") {
    let s = `Aegis recognised this as a document question and grounded its answer in ${chunkCount} retrieved chunk${chunkCount === 1 ? "" : "s"}`;
    if (topScore != null) s += ` (best match ${fixed(topScore, 2)})`;
    sentences.push(s + ".");
  } else if (route === "DIRECT_ANSWER") {
    let s = "Aegis answered directly from the model without using any documents";
    if (topScore != null)
      s += `; a quick index probe found nothing relevant (best match ${fixed(topScore, 2)})`;
    sentences.push(s + ".");
  } else {
    sentences.push(
      "Aegis decided the question was too vague to answer confidently and asked for clarification instead.",
    );
  }

  // 2) Who served it.
  if (provider) {
    const how = gen?.fallback_used
      ? "a fallback provider (the primary one failed)"
      : "the primary provider";
    sentences.push(`The response was served by ${how}: ${provider}.`);
  }

  // 3) Quality verdict.
  if (scores) {
    const label = scoreLabel(scores.overall_score).label;
    sentences.push(
      `The judge rated overall quality ${pct(scores.overall_score)} (${label}).`,
    );
  }

  // 4) Self-healing activity.
  if (retry && retry.retry_count > 0) {
    sentences.push(
      `The first attempt scored low, so Aegis self-healed with ${retry.retry_count} retr${retry.retry_count === 1 ? "y" : "ies"} (${retry.retry_strategies_used.join(", ") || "alternate strategies"}) and kept the best result.`,
    );
  }

  // 5) Degraded / fallback warnings.
  if (retry?.degraded_response) {
    sentences.push(
      "Even after retrying, the answer stayed below quality thresholds, so it is flagged as a degraded response.",
    );
  }

  // Tone: degraded → bad; retried/fallback/clarification → warn; else good.
  let tone: "good" | "warn" | "bad" = "good";
  if (retry?.degraded_response) tone = "bad";
  else if (
    (retry && retry.retry_count > 0) ||
    gen?.fallback_used ||
    route === "NEEDS_CLARIFICATION"
  )
    tone = "warn";

  return { sentences, tone };
}

export function WhatHappened({ data }: { data: AskResponse }) {
  const { sentences, tone } = buildSummary(data);
  const toneLabel =
    tone === "good" ? "clean run" : tone === "warn" ? "heads up" : "degraded";

  return (
    <Panel
      title="What happened?"
      subtitle="Plain-English summary of this request"
      right={<Badge tone={tone}>{toneLabel}</Badge>}
    >
      <ul className="space-y-2">
        {sentences.map((s, i) => (
          <li key={i} className="flex gap-2 text-sm leading-relaxed text-slate-200">
            <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
            <span>{s}</span>
          </li>
        ))}
      </ul>
    </Panel>
  );
}
