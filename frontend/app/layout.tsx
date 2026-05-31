import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { NavLink } from "@/components/NavLink";

export const metadata: Metadata = {
  title: "Aegis — Reliability Mesh",
  description:
    "Observability dashboard for Aegis, a self-optimizing LLM reliability mesh.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 sm:px-6 lg:px-8">
          <header className="flex items-center justify-between gap-4 py-5">
            <Link href="/" className="group flex items-center gap-3">
              <span className="grid h-9 w-9 place-items-center rounded-lg bg-gradient-to-br from-accent to-indigo-500 font-bold text-white shadow-lg">
                Æ
              </span>
              <span>
                <span className="block text-base font-semibold tracking-tight text-slate-100">
                  Aegis
                </span>
                <span className="block text-[11px] uppercase tracking-[0.2em] text-slate-500">
                  Reliability Mesh
                </span>
              </span>
            </Link>
            <nav className="flex items-center gap-1 rounded-lg border border-ink-700/60 bg-ink-850/60 p-1">
              <NavLink href="/">Query Console</NavLink>
              <NavLink href="/metrics">Metrics</NavLink>
            </nav>
          </header>

          <main className="flex-1 pb-16">{children}</main>

          <footer className="border-t border-ink-700/50 py-5 text-center text-xs text-slate-600">
            Aegis Module 7 · self-optimizing LLM reliability mesh ·
            retrieval → routing → evaluation → self-healing → multi-provider
            fallback
          </footer>
        </div>
      </body>
    </html>
  );
}
