// Small presentation helpers shared across panels.

export function pct(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

export function fixed(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

export function ms(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return `${value.toFixed(1)} ms`;
}

export function usd(value: number): string {
  if (!Number.isFinite(value)) return "—";
  // Sub-cent costs are common with token pricing; show enough precision.
  if (value === 0) return "$0.00";
  if (value < 0.01) return `$${value.toFixed(6)}`;
  return `$${value.toFixed(4)}`;
}

export function relTime(unixSeconds?: number | null): string {
  if (!unixSeconds) return "never";
  const deltaMs = Date.now() - unixSeconds * 1000;
  const s = Math.round(deltaMs / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

// ---------------------------------------------------------------------------
// Score color/label tuning (Module 8 — Part C)
//
// Normal scores (relevance, groundedness, completeness, confidence, overall):
//   0-49%   weak / red          (needs attention)
//   50-59%  borderline / amber
//   60-74%  acceptable / yellow-green   ← passing; NOT a failure
//   75-100% strong / green
//
// Hallucination risk is read inversely — lower is better — with its own bands:
//   0-25%   low / green
//   26-50%  moderate / yellow
//   51-75%  high / orange
//   76-100% severe / red
// ---------------------------------------------------------------------------

/** Maps a normal score (0..1) to a Tailwind text color class. */
export function scoreColor(score: number): string {
  if (score >= 0.75) return "text-emerald-400";
  if (score >= 0.6) return "text-lime-400";
  if (score >= 0.5) return "text-amber-400";
  return "text-rose-400";
}

export type ScoreTone = "good" | "warn" | "bad";

/**
 * Human label + tone for a normal score. 60%+ is "acceptable" (good), so an
 * acceptable answer that passed evaluation never reads as a failure.
 */
export function scoreLabel(score: number): { label: string; tone: ScoreTone } {
  if (score >= 0.75) return { label: "strong", tone: "good" };
  if (score >= 0.6) return { label: "acceptable", tone: "good" };
  if (score >= 0.5) return { label: "borderline", tone: "warn" };
  return { label: "needs attention", tone: "bad" };
}

/** Maps a normal score to a bar fill color. */
export function barColor(score: number): string {
  if (score >= 0.75) return "bg-emerald-500";
  if (score >= 0.6) return "bg-lime-500";
  if (score >= 0.5) return "bg-amber-500";
  return "bg-rose-500";
}

/** Text color for a hallucination-risk value (lower is better). */
export function riskColor(risk: number): string {
  if (risk <= 0.25) return "text-emerald-400";
  if (risk <= 0.5) return "text-yellow-400";
  if (risk <= 0.75) return "text-orange-400";
  return "text-rose-400";
}

/** Bar fill color for a hallucination-risk value (lower is better). */
export function riskBarColor(risk: number): string {
  if (risk <= 0.25) return "bg-emerald-500";
  if (risk <= 0.5) return "bg-yellow-500";
  if (risk <= 0.75) return "bg-orange-500";
  return "bg-rose-500";
}

/** Human label for a hallucination-risk value. */
export function riskLabel(risk: number): { label: string; tone: ScoreTone } {
  if (risk <= 0.25) return { label: "low", tone: "good" };
  if (risk <= 0.5) return { label: "moderate", tone: "warn" };
  if (risk <= 0.75) return { label: "high", tone: "bad" };
  return { label: "severe", tone: "bad" };
}
