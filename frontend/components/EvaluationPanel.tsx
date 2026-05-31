"use client";

import type { EvaluationResult } from "@/lib/types";
import { Panel, ScoreBar, Badge, BoolBadge } from "./ui";
import { pct, scoreColor, scoreLabel, riskLabel } from "@/lib/format";

export function EvaluationPanel({ evaluation }: { evaluation: EvaluationResult }) {
  const s = evaluation.scores;
  const overall = scoreLabel(s.overall_score);
  const risk = riskLabel(s.hallucination_risk);
  return (
    <Panel
      title="Evaluation"
      subtitle="LLM-as-a-Judge quality assessment"
      right={
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Badge tone="neutral">
            overall{" "}
            <span className={`font-mono ${scoreColor(s.overall_score)}`}>
              {pct(s.overall_score)}
            </span>
          </Badge>
          <Badge tone={overall.tone}>{overall.label}</Badge>
          <BoolBadge
            value={evaluation.should_retry}
            trueLabel="should retry"
            falseLabel="passed"
            trueTone="warn"
          />
        </div>
      }
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <ScoreBar label="Relevance" value={s.relevance} />
        <ScoreBar label="Groundedness" value={s.groundedness} />
        <ScoreBar label="Completeness" value={s.completeness} />
        <ScoreBar label="Confidence" value={s.confidence} />
        <ScoreBar
          label={`Hallucination risk — ${risk.label} (lower is better)`}
          value={s.hallucination_risk}
          invert
        />
        <ScoreBar label="Overall score" value={s.overall_score} />
      </div>

      <div className="mt-5 rounded-lg border border-ink-700/60 bg-ink-800/50 p-3">
        <p className="kv-label mb-1">Evaluation reason</p>
        <p className="text-sm leading-relaxed text-slate-300">
          {evaluation.evaluation_reason}
        </p>
      </div>
    </Panel>
  );
}
