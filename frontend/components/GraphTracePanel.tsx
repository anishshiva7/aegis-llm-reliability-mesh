"use client";

import type { RouteTrace, GraphTrace } from "@/lib/types";
import { Panel, Badge } from "./ui";
import { fixed } from "@/lib/format";

// Stable color per entity category so the same kind of node reads consistently.
const categoryTone: Record<string, string> = {
  Component: "border-accent/40 bg-accent/10 text-accent-soft",
  Route: "border-sky-500/40 bg-sky-500/10 text-sky-300",
  Provider: "border-violet-500/40 bg-violet-500/10 text-violet-300",
  Retrieval: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  Evaluation: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  Retry: "border-rose-500/40 bg-rose-500/10 text-rose-300",
  Metric: "border-cyan-500/40 bg-cyan-500/10 text-cyan-300",
  Dashboard: "border-fuchsia-500/40 bg-fuchsia-500/10 text-fuchsia-300",
  Document: "border-ink-600 bg-ink-700/40 text-slate-300",
};

function EntityChip({ name, category }: { name: string; category: string }) {
  const cls = categoryTone[category] ?? categoryTone.Document;
  return (
    <span
      className={`chip ${cls}`}
      title={category}
    >
      {name}
    </span>
  );
}

export function GraphTracePanel({ trace }: { trace: RouteTrace }) {
  const graph: GraphTrace | null | undefined = trace.graph;
  const mode = trace.retrieval_mode ?? "vector";

  // Only meaningful for graph/hybrid retrieval.
  if (!graph || mode === "vector") {
    return null;
  }

  const rels = graph.traversed_relationships ?? [];
  const matched = graph.matched_entities ?? [];
  const traversed = graph.traversed_entities ?? [];
  const chunks = graph.graph_chunks ?? [];

  return (
    <Panel
      title="Graph Trace"
      subtitle="Knowledge-graph traversal that fed the hybrid answer (Module 10)"
      right={
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Badge tone="accent">{mode} retrieval</Badge>
          <Badge tone={trace.graph_used ? "good" : "neutral"}>
            {trace.graph_used ? "graph used" : "graph idle"}
          </Badge>
          <Badge tone="neutral">{graph.graph_backend}</Badge>
        </div>
      }
    >
      <div className="space-y-5">
        {/* Headline stats */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MiniStat label="Graph score" value={fixed(graph.graph_score, 2)} />
          <MiniStat label="Hops" value={String(graph.hops)} />
          <MiniStat label="Entities" value={String(traversed.length)} />
          <MiniStat label="Relationships" value={String(rels.length)} />
        </div>

        {/* Matched (anchor) entities */}
        {matched.length > 0 && (
          <div>
            <p className="kv-label mb-2">Matched entities (query anchors)</p>
            <div className="flex flex-wrap gap-2">
              {matched.map((e) => (
                <EntityChip key={e.name} name={e.name} category={e.category} />
              ))}
            </div>
          </div>
        )}

        {/* Traversed neighbourhood */}
        {traversed.length > 0 && (
          <div>
            <p className="kv-label mb-2">
              Traversed neighbourhood ({traversed.length})
            </p>
            <div className="flex flex-wrap gap-2">
              {traversed.map((e) => (
                <EntityChip key={e.name} name={e.name} category={e.category} />
              ))}
            </div>
          </div>
        )}

        {/* Relationships */}
        {rels.length > 0 && (
          <div>
            <p className="kv-label mb-2">Relationships ({rels.length})</p>
            <ul className="scroll-thin max-h-56 space-y-1 overflow-auto font-mono text-xs">
              {rels.map((r, i) => (
                <li
                  key={`${r.source}-${r.type}-${r.target}-${i}`}
                  className="rounded border border-ink-700/60 bg-ink-800/50 px-2 py-1 text-slate-300"
                >
                  <span className="text-slate-100">{r.source}</span>
                  <span className="text-slate-500"> -[</span>
                  <span className="text-accent-soft">{r.type}</span>
                  <span className="text-slate-500">]→ </span>
                  <span className="text-slate-100">{r.target}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Linked chunks */}
        {chunks.length > 0 && (
          <div>
            <p className="kv-label mb-2">Linked chunks ({chunks.length})</p>
            <ul className="space-y-2">
              {chunks.map((c, i) => (
                <li
                  key={`${c.source}-${c.chunk_index}-${i}`}
                  className="rounded-lg border border-ink-700/60 bg-ink-800/50 p-3"
                >
                  <div className="mb-1 text-xs text-slate-400">
                    {c.source} · chunk #{c.chunk_index}
                  </div>
                  <p className="line-clamp-3 text-sm leading-relaxed text-slate-300">
                    {c.text}
                  </p>
                </li>
              ))}
            </ul>
          </div>
        )}

        <p className="text-xs text-slate-500">
          Graph retrieval ran in {fixed(graph.graph_latency_ms, 1)} ms on the{" "}
          {graph.graph_backend} backend.
        </p>
      </div>
    </Panel>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-ink-700/60 bg-ink-800/60 p-3">
      <p className="kv-label">{label}</p>
      <p className="mt-1 font-mono text-lg font-semibold text-slate-50">
        {value}
      </p>
    </div>
  );
}
