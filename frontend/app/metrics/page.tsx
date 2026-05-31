import { MetricsDashboard } from "@/components/MetricsDashboard";

export default function MetricsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-50">
          Operational Metrics
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Persisted, deterministic observability sourced from the SQLite metrics
          store — latency percentiles, score and latency histograms, cost,
          fallback / degraded / retry rates, provider health, and a Prometheus
          exposition endpoint.
        </p>
      </div>
      <MetricsDashboard />
    </div>
  );
}
