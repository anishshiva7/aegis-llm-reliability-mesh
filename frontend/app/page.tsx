import { QueryConsole } from "@/components/QueryConsole";

export default function HomePage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-50">
          Query Console
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Send a question through the full mesh — routing, FAISS retrieval,
          generation, LLM-as-a-Judge evaluation, self-healing retries, and
          multi-provider fallback — and inspect every stage of the trace.
        </p>
      </div>
      <QueryConsole />
    </div>
  );
}
