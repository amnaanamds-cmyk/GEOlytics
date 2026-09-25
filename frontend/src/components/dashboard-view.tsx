"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { Meter, StatTile } from "@/components/charts";
import { Alert, Button, Card, EmptyState, Field, StatusPill } from "@/components/ui";
import { RequestError, get, post } from "@/lib/client";
import type { Audit, AuditList, Usage } from "@/lib/types";

const RUNNING = new Set(["pending", "running"]);

export function DashboardView({
  initialAudits,
  initialUsage,
  loadError,
}: {
  initialAudits: Audit[];
  initialUsage: Usage | null;
  loadError: string | null;
}) {
  const [audits, setAudits] = useState<Audit[]>(initialAudits);
  const [usage, setUsage] = useState<Usage | null>(initialUsage);
  const [error, setError] = useState<string | null>(loadError);
  const [notice, setNotice] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const load = useCallback(async () => {
    try {
      const [list, quota] = await Promise.all([
        get<AuditList>("audits?limit=25"),
        get<Usage>("org/usage"),
      ]);
      setAudits(list.items);
      setUsage(quota);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load your audits.");
    }
  }, []);

  // Audits run in a worker, so the list is polled while anything is in flight
  // and left alone once everything has settled. State is set from the interval
  // callback, never synchronously during render.
  useEffect(() => {
    if (!audits.some((a) => RUNNING.has(a.status))) return;
    const timer = setInterval(() => void load(), 4000);
    return () => clearInterval(timer);
  }, [audits, load]);

  async function startAudit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setNotice(null);

    const form = new FormData(event.currentTarget);
    try {
      const result = await post<{ message: string }>("audits", {
        url: form.get("url"),
        max_pages: Number(form.get("max_pages") ?? 25),
        chunker: "sentence",
        retriever: "hybrid",
      });
      setNotice(result.message);
      (event.target as HTMLFormElement).reset();
      await load();
    } catch (err) {
      if (err instanceof RequestError && err.status === 402) {
        setError(`${err.message} — upgrade your plan in Settings to continue.`);
      } else {
        setError(err instanceof Error ? err.message : "Could not start the audit.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  const scored = audits.filter((a) => a.overall_score !== null);
  const average =
    scored.length > 0
      ? scored.reduce((sum, a) => sum + (a.overall_score ?? 0), 0) / scored.length
      : null;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <StatTile
          label="Audits this period"
          value={usage ? usage.used.audits_run.toLocaleString() : "—"}
          hint={usage ? `of ${usage.limits.audits_run.toLocaleString()} on ${usage.plan}` : undefined}
        />
        <StatTile
          label="Pages crawled"
          value={usage ? usage.used.pages_crawled.toLocaleString() : "—"}
          hint={usage ? `of ${usage.limits.pages_crawled.toLocaleString()}` : undefined}
        />
        <StatTile
          label="Average GEO score"
          value={average === null ? "—" : average.toFixed(1)}
          hint={scored.length > 0 ? `across ${scored.length} completed audits` : "no completed audits yet"}
        />
      </div>

      {usage && (
        <Card title="Usage this period" description={`Billing period ${usage.period}`}>
          <div className="grid gap-4 sm:grid-cols-2">
            <Meter
              label="Pages crawled"
              used={usage.used.pages_crawled}
              limit={usage.limits.pages_crawled}
            />
            <Meter
              label="Audits run"
              used={usage.used.audits_run}
              limit={usage.limits.audits_run}
            />
            <Meter label="Sites" used={usage.sites.used} limit={usage.sites.limit} />
            <Meter
              label="Audits running now"
              used={usage.concurrent_audits.used}
              limit={usage.concurrent_audits.limit}
            />
          </div>
        </Card>
      )}

      <Card
        title="Run an audit"
        description="Crawls the site, scores every page, and lists what to change."
      >
        <form onSubmit={startAudit} className="flex flex-wrap items-end gap-3">
          <div className="min-w-[16rem] flex-1">
            <Field
              label="Site URL"
              name="url"
              type="url"
              required
              placeholder="https://example.com"
            />
          </div>
          <div className="w-32">
            <Field
              label="Max pages"
              name="max_pages"
              type="number"
              min={1}
              max={500}
              defaultValue={25}
            />
          </div>
          <Button type="submit" loading={submitting}>
            Start audit
          </Button>
        </form>
        {notice && (
          <div className="mt-3">
            <Alert tone="success">{notice}</Alert>
          </div>
        )}
        {error && (
          <div className="mt-3">
            <Alert tone="error">{error}</Alert>
          </div>
        )}
      </Card>

      <Card title="Recent audits">
        {audits.length === 0 ? (
          <EmptyState title="No audits yet">
            Enter a site URL above to run your first one.
          </EmptyState>
        ) : (
          <ul className="divide-y" style={{ borderColor: "var(--gridline)" }}>
            {audits.map((audit) => (
              <li key={audit.audit_id} className="flex items-center gap-4 py-3">
                <div className="min-w-0 flex-1">
                  <Link
                    href={`/audits/${audit.audit_id}`}
                    className="block truncate text-sm font-medium underline-offset-2 hover:underline"
                  >
                    {audit.site_url}
                  </Link>
                  <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                    {audit.created_at
                      ? new Date(audit.created_at).toLocaleString()
                      : "just now"}
                  </p>
                </div>
                <StatusPill status={audit.status} />
                <div className="w-16 text-right text-sm font-semibold tnum">
                  {audit.overall_score === null ? "—" : audit.overall_score.toFixed(1)}
                </div>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
