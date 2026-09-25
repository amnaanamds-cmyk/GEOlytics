"use client";

import Link from "next/link";
import { useCallback, useState } from "react";

import { DeltaPlot, formatNumber } from "@/components/charts";
import { Alert, Card } from "@/components/ui";
import { get } from "@/lib/client";
import type { Experiment } from "@/lib/types";

const METRICS = ["ndcg@10", "ndcg@5", "recall@10", "mrr@10", "precision@5"];

const DEFAULT_METRIC = "ndcg@10";

export function ExperimentDetail({
  name,
  initialData,
  initialBaseline,
  loadError,
}: {
  name: string;
  initialData: Experiment | null;
  initialBaseline: string;
  loadError: string | null;
}) {
  const [metric, setMetric] = useState(DEFAULT_METRIC);
  const [baseline, setBaseline] = useState<string>(initialBaseline);
  const [data, setData] = useState<Experiment | null>(initialData);
  const [error, setError] = useState<string | null>(loadError);

  // Refetch only when a filter actually moves off the server-rendered default,
  // so the first paint needs no client round trip.
  const refetch = useCallback(
    async (nextMetric: string, nextBaseline: string) => {
      const query = new URLSearchParams({ metric: nextMetric });
      if (nextBaseline) query.set("baseline", nextBaseline);
      try {
        setData(
          await get<Experiment>(`experiments/${encodeURIComponent(name)}?${query}`),
        );
        setError(null);
      } catch (err) {
        setData(null);
        setError(err instanceof Error ? err.message : "Could not load this experiment.");
      }
    },
    [name],
  );

  const conditions = data?.runs.map((r) => r.condition) ?? [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/experiments" className="text-sm underline underline-offset-2">
          ← Experiments
        </Link>
        <h1 className="min-w-0 flex-1 truncate text-lg font-semibold">{name}</h1>
      </div>

      {/* Filters sit in one row above the charts. */}
      <div className="flex flex-wrap items-end gap-3">
        <label className="text-sm">
          <span className="mb-1.5 block font-medium">Metric</span>
          <select
            value={metric}
            onChange={(e) => {
              setMetric(e.target.value);
              void refetch(e.target.value, baseline);
            }}
            className="rounded-lg px-3 py-2 text-sm"
            style={{
              background: "var(--surface-1)",
              color: "var(--text-primary)",
              border: "1px solid var(--border)",
            }}
          >
            {METRICS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <label className="text-sm">
          <span className="mb-1.5 block font-medium">Baseline</span>
          <select
            value={baseline}
            onChange={(e) => {
              setBaseline(e.target.value);
              void refetch(metric, e.target.value);
            }}
            className="rounded-lg px-3 py-2 text-sm"
            style={{
              background: "var(--surface-1)",
              color: "var(--text-primary)",
              border: "1px solid var(--border)",
            }}
          >
            <option value="">All pairs (harsher correction)</option>
            {conditions.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
      </div>

      {error && <Alert tone="error">{error}</Alert>}

      {data && (
        <>
          <Card
            title={`Difference in ${data.metric}`}
            description="Point estimate and 95% confidence interval for each comparison."
          >
            <DeltaPlot
              metric={data.metric}
              baseline={baseline || "every other condition"}
              data={data.comparisons.map((c) => ({
                label: `${c.name_a} vs ${c.name_b}`,
                value: c.mean_diff,
                low: c.ci_low,
                high: c.ci_high,
                significant: c.significant,
                annotation: `${c.effect_label} effect · n=${c.n} · p${
                  c.p_adjusted !== null ? " (Holm)" : ""
                }=${formatNumber(c.p_adjusted ?? c.p_value, 4)}`,
              }))}
            />
          </Card>

          <Card
            title="Conditions"
            description="Chunk count and mean length are shown because chunk size confounds retrieval quality."
          >
            <div className="overflow-x-auto">
              <table className="w-full text-sm tnum">
                <thead>
                  <tr style={{ color: "var(--text-secondary)" }}>
                    {["Condition", "Chunks", "Mean tokens", "Queries", data.metric].map(
                      (header) => (
                        <th key={header} scope="col" className="px-2 py-1.5 text-left font-medium">
                          {header}
                        </th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {data.runs.map((run) => (
                    <tr key={run.condition} style={{ borderTop: "1px solid var(--gridline)" }}>
                      <td className="px-2 py-1.5">{run.condition}</td>
                      <td className="px-2 py-1.5">{run.index_stats.n_chunks ?? "—"}</td>
                      <td className="px-2 py-1.5">
                        {run.index_stats.mean_tokens
                          ? formatNumber(run.index_stats.mean_tokens, 1)
                          : "—"}
                      </td>
                      <td className="px-2 py-1.5">{run.n_queries}</td>
                      <td className="px-2 py-1.5 font-medium">
                        {run.aggregate[data.metric] !== undefined
                          ? formatNumber(run.aggregate[data.metric], 4)
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Methods">
            <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
              {data.methods_note}
            </p>
          </Card>
        </>
      )}
    </div>
  );
}
