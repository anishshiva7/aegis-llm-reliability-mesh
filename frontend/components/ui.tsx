// Reusable presentational primitives: panels, badges, score bars, key/values.
"use client";

import { useState } from "react";
import { barColor, riskBarColor, pct } from "@/lib/format";

export function Panel({
  title,
  subtitle,
  right,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-head">
        <div>
          <h2 className="panel-title">{title}</h2>
          {subtitle && (
            <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>
          )}
        </div>
        {right}
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}

type Tone = "neutral" | "good" | "warn" | "bad" | "accent";

const toneClasses: Record<Tone, string> = {
  neutral: "border-ink-600 bg-ink-700/40 text-slate-300",
  good: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  warn: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  bad: "border-rose-500/40 bg-rose-500/10 text-rose-300",
  accent: "border-accent/40 bg-accent/10 text-accent-soft",
};

export function Badge({
  tone = "neutral",
  children,
}: {
  tone?: Tone;
  children: React.ReactNode;
}) {
  return <span className={`chip ${toneClasses[tone]}`}>{children}</span>;
}

export function BoolBadge({
  value,
  trueLabel,
  falseLabel,
  // For most flags "true" is bad (degraded/fallback); allow inverting.
  trueTone = "bad",
  falseTone = "good",
}: {
  value: boolean;
  trueLabel: string;
  falseLabel: string;
  trueTone?: Tone;
  falseTone?: Tone;
}) {
  return (
    <Badge tone={value ? trueTone : falseTone}>
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          value
            ? trueTone === "good"
              ? "bg-emerald-400"
              : "bg-rose-400"
            : "bg-emerald-400"
        }`}
      />
      {value ? trueLabel : falseLabel}
    </Badge>
  );
}

export function ScoreBar({
  label,
  value,
  invert = false,
}: {
  label: string;
  value: number;
  invert?: boolean;
}) {
  const widthPct = Math.max(0, Math.min(1, value)) * 100;
  // Risk metrics (invert) use the dedicated 4-band risk palette; normal scores
  // use the tuned acceptable/strong palette (Module 8 — Part C).
  const fill = invert ? riskBarColor(value) : barColor(value);
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="font-mono text-xs text-slate-200">{pct(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-ink-700">
        <div
          className={`h-full rounded-full transition-all ${fill}`}
          style={{ width: `${widthPct}%` }}
        />
      </div>
    </div>
  );
}

export function KV({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="kv-label">{label}</span>
      <span className="kv-value break-all">{children}</span>
    </div>
  );
}

export function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-ink-700/60 bg-ink-800/60 p-4">
      <p className="kv-label">{label}</p>
      <p className="mt-1 text-2xl font-semibold tracking-tight text-slate-50">
        {value}
      </p>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  );
}

/** Collapsible raw-JSON viewer used as a debug fallback on every result. */
export function JsonPanel({ data }: { data: unknown }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="panel">
      <button
        onClick={() => setOpen((o) => !o)}
        className="panel-head w-full text-left transition-colors hover:bg-ink-800/50"
      >
        <h2 className="panel-title">Raw response (debug)</h2>
        <span className="text-xs text-slate-500">{open ? "Hide" : "Show"}</span>
      </button>
      {open && (
        <pre className="scroll-thin max-h-[28rem] overflow-auto p-5 font-mono text-xs leading-relaxed text-slate-300">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-sm text-slate-400">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-ink-600 border-t-accent" />
      {label ?? "Loading…"}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
      <span className="font-semibold">Error:</span> {message}
    </div>
  );
}

export function HealthDot({ status }: { status: string }) {
  const tone =
    status === "healthy"
      ? "bg-emerald-400"
      : status === "degraded"
        ? "bg-amber-400"
        : "bg-rose-400";
  return <span className={`inline-block h-2 w-2 rounded-full ${tone}`} />;
}
