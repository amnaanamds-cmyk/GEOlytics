"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { BarList, HeroFigure, StatTile } from "@/components/charts";
import { Alert, Card, EmptyState, StatusPill } from "@/components/ui";
import { get } from "@/lib/client";
import type { Audit, PageScore } from "@/lib/types";

const SIGNAL_LABELS: Record<string, string> = {
  statistic_density: "Concrete statistics",
  citation_density: "Citations and sources",
  quotation_density: "Direct quotations",
  authority_markers: "Attribution",
  schema_coverage: "Structured data",
  heading_structure: "Heading hierarchy",
  self_containedness: "Self-contained passages",
  extractability: "Content vs boilerplate",
  answer_directness: "Direct answers",
  freshness: "Date signals",
};

export function AuditDetail({
  auditId,
  initialAudit,
  loadError,
}: {
  auditId: string;
  initialAudit: Audit | null;
  loadError: string | null;
}) {
  const [audit, setAudit] = useState<Audit | null>(initialAudit);
  const [selected, setSelected] = useState<PageScore | null>(
    initialAudit?.pages[0] ?? null,
  );
  const [error, setError] = useState<string | null>(loadError);

  const load = useCallback(async () => {
    try {
      const result = await get<Audit>(`audits/${auditId}`);
      setAudit(result);
      setSelected((current) =>
        current
          ? result.pages.find((p) => p.url === current.url) ?? current
          : result.pages[0] ?? null,
      );
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load this audit.");
    }
  }, [auditId]);

  // Polled only while the worker is still running; state is set from the
  // interval callback, not during render.
  useEffect(() => {
    if (!audit || !["pending", "running"].includes(audit.status)) return;
    const timer = setInterval(() => void load(), 4000);
    return () => clearInterval(timer);
  }, [audit, load]);

  if (error && !audit) return <Alert tone="error">{error}</Alert>;
  if (!audit) {
    return (
      <p className="text-sm" style={{ color: "var(--text-muted)" }}>
        Loading…
      </p>
    );
  }

  const warnings = (audit.summary?.warnings as string[] | undefined) ?? [];
  const crawl = audit.summary?.crawl as string | undefined;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/dashboard" className="text-sm underline underline-offset-2">
          ← Audits
        </Link>
        <h1 className="min-w-0 flex-1 truncate text-lg font-semibold">{audit.site_url}</h1>
        <StatusPill status={audit.status} />
      </div>

      {audit.score_caveat && <Alert tone="warning">{audit.score_caveat}</Alert>}

      {["pending", "running"].includes(audit.status) && (
        <Alert tone="info">
          This audit is still running. The page refreshes itself as results arrive.
        </Alert>
      )}

      <div className="grid gap-4 sm:grid-cols-3">
        <HeroFigure value={audit.overall_score} caption="Overall GEO score" />
        <StatTile
          label="Pages scored"
          value={audit.pages.length}
          hint={crawl ? crawl.split(",")[0] : undefined}
        />
        <StatTile
          label="Weakest page"
          value={
            audit.pages.length > 0
              ? audit.pages[audit.pages.length - 1].score.toFixed(1)
              : "—"
          }
          hint={audit.pages.length > 0 ? "start here" : undefined}
        />
      </div>

      {warnings.length > 0 && (
        <Card title="Notes from this run">
          <ul className="space-y-1 text-xs" style={{ color: "var(--text-secondary)" }}>
            {warnings.map((warning) => (
              <li key={warning}>• {warning}</li>
            ))}
          </ul>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Score by page"
          description="Select a page to see its signal breakdown."
        >
          {audit.pages.length === 0 ? (
            <EmptyState title="No pages scored yet" />
          ) : (
            <>
              <BarList
                title="Score by page"
                max={100}
                data={audit.pages.map((page) => ({
                  label: shorten(page.url),
                  value: page.score,
                }))}
                tableCaption="GEO score for each crawled page"
              />
              <div className="mt-4 flex flex-wrap gap-2">
                {audit.pages.map((page) => (
                  <button
                    key={page.url}
                    type="button"
                    onClick={() => setSelected(page)}
                    className="rounded-lg px-2.5 py-1 text-xs"
                    style={{
                      border: `1px solid ${
                        selected?.url === page.url ? "var(--series-1)" : "var(--border)"
                      }`,
                      color: "var(--text-primary)",
                    }}
                    aria-pressed={selected?.url === page.url}
                  >
                    {shorten(page.url)}
                  </button>
                ))}
              </div>
            </>
          )}
        </Card>

        <Card
          title={selected ? `Signals · ${shorten(selected.url)}` : "Signals"}
          description="Each signal is measured from the page, scored 0 to 1."
        >
          {selected ? (
            <BarList
              title="Signal strength"
              max={1}
              valueFormatter={(v) => v.toFixed(2)}
              data={Object.entries(selected.signals)
                .sort((a, b) => b[1] - a[1])
                .map(([key, value]) => ({
                  label: SIGNAL_LABELS[key] ?? key,
                  value,
                }))}
              tableCaption="Signal values for the selected page"
            />
          ) : (
            <EmptyState title="Select a page" />
          )}
        </Card>
      </div>

      {selected && selected.recommendations.length > 0 && (
        <Card
          title="What to change"
          description={`Ordered by how much score is recoverable on ${shorten(selected.url)}.`}
        >
          <ol className="space-y-3">
            {selected.recommendations.map((recommendation, index) => (
              <li key={recommendation} className="flex gap-3 text-sm">
                <span
                  className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs font-semibold tnum"
                  style={{ background: "var(--gridline)", color: "var(--text-primary)" }}
                >
                  {index + 1}
                </span>
                <span>{recommendation}</span>
              </li>
            ))}
          </ol>
        </Card>
      )}
    </div>
  );
}

function shorten(url: string): string {
  try {
    const parsed = new URL(url);
    const path = parsed.pathname === "/" ? "" : parsed.pathname;
    return `${parsed.host}${path}` || url;
  } catch {
    return url;
  }
}
