"use client";

import { useCallback, useEffect, useState } from "react";
import {
  getMetrics,
  getProvidersHealth,
  getPrometheus,
  ApiError,
  API_BASE_URL,
} from "@/lib/api";
import type { MetricsSnapshot, ProvidersHealthResponse } from "@/lib/types";
import {
  Panel,
  Stat,
  Spinner,
  ErrorBox,
  Badge,
  HealthDot,
  JsonPanel,
} from "./ui";
import { ms, pct, fixed, usd, relTime } from "@/lib/format";

function Histogram({ data }: { data: Record<string, number> }) {
  const max = Math.max(1, ...Object.values(data));
  return (
    <div className="space-y-2">
      {Object.entries(data).map(([bucket, count]) => (
        <div key={bucket} className="flex items-center gap-3">
          <span className="w-24 shrink-0 text-right font-mono text-xs text-slate-500">
            {bucket}
          </span>
          <div className="h-4 flex-1 overflow-hidden rounded bg-ink-700">
            <div
              className="h-full rounded bg-accent/80"
              style={{ width: `${(count / max) * 100}%` }}
            />
          </div>
          <span className="w-8 shrink-0 font-mono text-xs text-slate-300">
            {count}
          </span>
        </div>
      ))}
    </div>
  );
}

function CountTable({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0)
    return <p className="text-sm text-slate-500">No data yet.</p>;
  return (
    <div className="space-y-2">
      {entries.map(([name, count]) => (
        <div
          key={name}
          className="flex items-center justify-between rounded-md border border-ink-700/60 bg-ink-800/50 px-3 py-2"
        >
          <span className="font-mono text-xs text-slate-300">{name}</span>
          <span className="font-mono text-sm text-slate-100">{count}</span>
        </div>
      ))}
    </div>
  );
}

export function MetricsDashboard() {
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [health, setHealth] = useState<ProvidersHealthResponse | null>(null);
  const [prom, setProm] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [auto, setAuto] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [m, h, p] = await Promise.all([
        getMetrics(),
        getProvidersHealth(),
        getPrometheus(),
      ]);
      setMetrics(m);
      setHealth(h);
      setProm(p);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Failed to load metrics from the backend.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!auto) return;
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, [auto, load]);

  if (loading) {
    return (
      <div className="panel panel-body">
        <Spinner label="Loading metrics…" />
      </div>
    );
  }

  if (error) {
    return (
      <Panel title="Metrics unavailable" subtitle={`GET /metrics · ${API_BASE_URL}`}>
        <ErrorBox message={error} />
        <button
          onClick={load}
          className="mt-3 rounded-lg border border-ink-600 px-4 py-2 text-sm text-slate-300 hover:bg-ink-700/40"
        >
          Retry
        </button>
      </Panel>
    );
  }

  if (!metrics) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-end gap-3">
        <label className="flex items-center gap-2 text-xs text-slate-400">
          <input
            type="checkbox"
            checked={auto}
            onChange={(e) => setAuto(e.target.checked)}
            className="accent-accent"
          />
          Auto-refresh (5s)
        </label>
        <button
          onClick={load}
          className="rounded-lg border border-ink-600 px-3 py-1.5 text-xs text-slate-300 hover:bg-ink-700/40"
        >
          Refresh
        </button>
      </div>

      {/* Headline KPIs */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Total requests" value={metrics.total_requests} />
        <Stat
          label="Avg latency"
          value={ms(metrics.average_latency_ms)}
          hint={`p95 ${ms(metrics.p95_latency_ms)}`}
        />
        <Stat
          label="Avg overall score"
          value={pct(metrics.average_overall_score)}
        />
        <Stat
          label="Total cost"
          value={usd(metrics.estimated_cost_usd_total)}
        />
        <Stat
          label="Fallback rate"
          value={pct(metrics.fallback_rate)}
          hint={`${metrics.fallback_count} events`}
        />
        <Stat
          label="Degraded rate"
          value={pct(metrics.degraded_response_rate)}
          hint={`${metrics.degraded_response_count} events`}
        />
        <Stat label="Retry rate" value={pct(metrics.retry_rate)} />
        <Stat
          label="Latency p50 / p99"
          value={
            <span className="text-lg">
              {fixed(metrics.p50_latency_ms, 0)} /{" "}
              {fixed(metrics.p99_latency_ms, 0)} ms
            </span>
          }
        />
      </div>

      {/* Provider health */}
      <Panel
        title="Provider Health"
        subtitle="Rolling reliability that drives fallback ordering"
        right={
          health && (
            <span className="text-xs text-slate-500">
              order:{" "}
              <span className="font-mono text-slate-400">
                {health.recommended_order.join(" → ") || "—"}
              </span>
            </span>
          )
        }
      >
        {!health || health.providers.length === 0 ? (
          <p className="text-sm text-slate-500">
            No provider activity recorded yet. Run a query first.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="text-xs uppercase tracking-wide text-slate-500">
                  <th className="py-2 pr-4 font-medium">Provider</th>
                  <th className="py-2 pr-4 font-medium">Status</th>
                  <th className="py-2 pr-4 font-medium">Consec. fails</th>
                  <th className="py-2 pr-4 font-medium">Successes</th>
                  <th className="py-2 pr-4 font-medium">Failures</th>
                  <th className="py-2 pr-4 font-medium">Last success</th>
                  <th className="py-2 font-medium">Last failure</th>
                </tr>
              </thead>
              <tbody className="font-mono text-xs text-slate-300">
                {health.providers.map((p) => (
                  <tr key={p.provider} className="border-t border-ink-700/50">
                    <td className="py-2 pr-4 text-slate-100">{p.provider}</td>
                    <td className="py-2 pr-4">
                      <span className="inline-flex items-center gap-2">
                        <HealthDot status={p.health_status} />
                        {p.health_status}
                      </span>
                    </td>
                    <td className="py-2 pr-4">{p.consecutive_failures}</td>
                    <td className="py-2 pr-4 text-emerald-300">
                      {p.total_successes}
                    </td>
                    <td className="py-2 pr-4 text-rose-300">
                      {p.total_failures}
                    </td>
                    <td className="py-2 pr-4 text-slate-500">
                      {relTime(p.last_success_at)}
                    </td>
                    <td className="py-2 text-slate-500">
                      {relTime(p.last_failure_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {/* Distributions */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Latency histogram" subtitle="Request count per latency bucket">
          <Histogram data={metrics.latency_histogram} />
        </Panel>
        <Panel title="Score histogram" subtitle="Overall evaluation score distribution">
          <Histogram data={metrics.score_histogram} />
        </Panel>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Requests by provider">
          <CountTable data={metrics.requests_by_provider} />
        </Panel>
        <Panel title="Requests by route">
          <CountTable data={metrics.requests_by_route} />
        </Panel>
      </div>

      {/* GraphRAG telemetry (Module 10) */}
      {metrics.graph && (
        <Panel
          title="GraphRAG / Hybrid Retrieval"
          subtitle="Knowledge-graph traversal telemetry (Module 10)"
          right={
            <Badge tone={metrics.graph.hybrid_queries > 0 ? "accent" : "neutral"}>
              {metrics.graph.hybrid_queries} hybrid quer
              {metrics.graph.hybrid_queries === 1 ? "y" : "ies"}
            </Badge>
          }
        >
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <Stat label="Graph nodes" value={metrics.graph.graph_nodes} />
            <Stat
              label="Relationships"
              value={metrics.graph.graph_relationships}
            />
            <Stat label="Linked chunks" value={metrics.graph.linked_chunks} />
            <Stat label="Traversals" value={metrics.graph.graph_traversals} />
            <Stat
              label="Hybrid queries"
              value={metrics.graph.hybrid_queries}
            />
            <Stat
              label="Avg graph latency"
              value={ms(metrics.graph.graph_latency_ms)}
            />
          </div>
        </Panel>
      )}

      {/* Prometheus exposition */}
      <Panel
        title="Prometheus exposition"
        subtitle="GET /metrics/prometheus · text/plain; version=0.0.4"
        right={<Badge tone="neutral">no client lib</Badge>}
      >
        <pre className="scroll-thin max-h-80 overflow-auto rounded-lg border border-ink-700/60 bg-ink-900/70 p-4 font-mono text-xs leading-relaxed text-slate-300">
          {prom || "—"}
        </pre>
      </Panel>

      <JsonPanel data={metrics} />
    </div>
  );
}
