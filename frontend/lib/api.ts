// Thin typed fetch client for the Aegis backend. Centralizes the base URL,
// error handling, and JSON parsing so components stay declarative.

import type {
  AskRequest,
  AskResponse,
  IngestRequest,
  IngestResponse,
  MetricsSnapshot,
  ProvidersHealthResponse,
} from "./types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_AEGIS_API_BASE_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000";

/** Error type carrying HTTP status + parsed body for nicer UI messages. */
export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch (err) {
    // Network-level failure (backend down / CORS / DNS).
    throw new ApiError(
      `Could not reach the Aegis backend at ${API_BASE_URL}. Is it running?`,
      0,
      String(err),
    );
  }

  const text = await res.text();
  const body = text ? safeJson(text) : null;

  if (!res.ok) {
    const detail =
      (body && typeof body === "object" && "detail" in body
        ? String((body as Record<string, unknown>).detail)
        : null) ?? `Request failed with status ${res.status}`;
    throw new ApiError(detail, res.status, body);
  }
  return body as T;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export function ask(payload: AskRequest): Promise<AskResponse> {
  return request<AskResponse>("/ask", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function ingest(payload: IngestRequest): Promise<IngestResponse> {
  return request<IngestResponse>("/ingest", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getMetrics(): Promise<MetricsSnapshot> {
  return request<MetricsSnapshot>("/metrics");
}

export function getProvidersHealth(): Promise<ProvidersHealthResponse> {
  return request<ProvidersHealthResponse>("/providers/health");
}

export async function getPrometheus(): Promise<string> {
  const res = await fetch(`${API_BASE_URL}/metrics/prometheus`, {
    cache: "no-store",
  });
  return res.text();
}
