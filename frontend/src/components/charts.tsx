"use client";

/**
 * Chart components.
 *
 * Forms follow the data's job: magnitude is a horizontal bar on one blue
 * ordinal ramp, a ratio against a limit is a meter, a single headline value is
 * a stat tile rather than a one-bar chart, and a signed difference from a
 * baseline is a diverging dot with its confidence interval.
 *
 * Marks follow the fixed specs: bars capped at 24px with a 4px rounded data
 * end and a square baseline end, >=8px dots carrying a 2px surface ring,
 * hairline recessive gridlines, and labels placed selectively rather than on
 * every mark. Every chart ships a table view, so nothing is gated behind
 * colour or hover.
 */

import { useId, useState, type ReactNode } from "react";

// --- shared -------------------------------------------------------------

export function formatNumber(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function rampStep(fraction: number): string {
  // Four ordinal steps rather than a continuous gradient: discrete bands are
  // readable, and every step clears the contrast floor against the surface.
  if (fraction >= 0.75) return "var(--seq-4)";
  if (fraction >= 0.5) return "var(--seq-3)";
  if (fraction >= 0.25) return "var(--seq-2)";
  return "var(--seq-1)";
}

export function TableView({
  caption,
  headers,
  rows,
}: {
  caption: string;
  headers: string[];
  rows: (string | number)[][];
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="text-xs underline"
        style={{ color: "var(--text-secondary)" }}
        aria-expanded={open}
      >
        {open ? "Hide" : "Show"} data table
      </button>
      {open && (
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-xs tnum">
            <caption className="sr-only">{caption}</caption>
            <thead>
              <tr>
                {headers.map((h) => (
                  <th
                    key={h}
                    scope="col"
                    className="px-2 py-1 text-left font-medium"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i} style={{ borderTop: "1px solid var(--gridline)" }}>
                  {row.map((cell, j) => (
                    <td key={j} className="px-2 py-1">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// --- stat tile ----------------------------------------------------------

export function StatTile({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "neutral" | "good" | "warning" | "critical";
}) {
  const toneColor = {
    neutral: "var(--text-primary)",
    good: "var(--status-good-text)",
    warning: "var(--text-primary)",
    critical: "var(--status-critical)",
  }[tone];

  return (
    <div className="card p-4">
      <div className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
        {label}
      </div>
      <div
        className="mt-1 text-3xl font-semibold tnum"
        style={{ color: value === "—" ? "var(--text-muted)" : toneColor }}
      >
        {value}
      </div>
      {hint && (
        <div className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
          {hint}
        </div>
      )}
    </div>
  );
}

export function HeroFigure({
  value,
  caption,
  max = 100,
  pendingLabel = "Not scored yet",
}: {
  value: number | null;
  caption: string;
  max?: number;
  pendingLabel?: string;
}) {
  return (
    <div className="card p-6">
      <div className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
        {caption}
      </div>
      {value === null ? (
        // An em-dash at hero size reads as a redaction bar, not as "no value".
        // Absence gets words instead.
        <p className="mt-3 text-sm" style={{ color: "var(--text-muted)" }}>
          {pendingLabel}
        </p>
      ) : (
        <div className="mt-2 flex items-baseline gap-2">
          <span className="text-6xl font-semibold tnum leading-none">
            {formatNumber(value, 1)}
          </span>
          <span className="text-xl tnum" style={{ color: "var(--text-muted)" }}>
            / {max}
          </span>
        </div>
      )}
    </div>
  );
}

// --- meter --------------------------------------------------------------

export function Meter({
  label,
  used,
  limit,
  unit,
}: {
  label: string;
  used: number;
  limit: number;
  unit?: string;
}) {
  const fraction = limit > 0 ? Math.min(1, used / limit) : 0;
  // Status, not the sequential ramp: this is a state (fine / nearly out / out),
  // and it always ships with the numeric label beside it.
  const tone =
    fraction >= 1 ? "var(--status-critical)"
    : fraction >= 0.8 ? "var(--status-warning)"
    : "var(--seq-3)";

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span style={{ color: "var(--text-secondary)" }}>{label}</span>
        <span className="tnum" style={{ color: "var(--text-primary)" }}>
          {used.toLocaleString()} / {limit.toLocaleString()}
          {unit ? ` ${unit}` : ""}
        </span>
      </div>
      <div
        className="mt-1.5 h-2 w-full overflow-hidden rounded-full"
        style={{ background: "var(--gridline)" }}
        role="meter"
        aria-valuenow={used}
        aria-valuemin={0}
        aria-valuemax={limit}
        aria-label={label}
      >
        <div
          className="h-full rounded-full transition-[width]"
          style={{ width: `${fraction * 100}%`, background: tone }}
        />
      </div>
    </div>
  );
}

// --- horizontal magnitude bars -----------------------------------------

export type BarDatum = { label: string; value: number; href?: string; sub?: string };

export function BarList({
  data,
  max,
  title,
  valueFormatter = (v: number) => formatNumber(v, 1),
  tableCaption,
}: {
  data: BarDatum[];
  max?: number;
  title: string;
  valueFormatter?: (v: number) => string;
  tableCaption?: string;
}) {
  const ceiling = max ?? Math.max(1, ...data.map((d) => d.value));

  if (data.length === 0) {
    return (
      <p className="text-sm" style={{ color: "var(--text-muted)" }}>
        Nothing to show yet.
      </p>
    );
  }

  return (
    <div>
      {/* One series, so no legend box: the title already names what is plotted. */}
      <ul className="space-y-3">
        {data.map((d) => {
          const fraction = Math.max(0, Math.min(1, d.value / ceiling));
          return (
            <li key={d.label}>
              <div className="flex items-baseline justify-between gap-3">
                <span
                  className="truncate text-sm"
                  style={{ color: "var(--text-primary)" }}
                  title={d.label}
                >
                  {d.label}
                </span>
                <span
                  className="shrink-0 text-sm font-medium tnum"
                  style={{ color: "var(--text-primary)" }}
                >
                  {valueFormatter(d.value)}
                </span>
              </div>
              <div
                className="mt-1 h-3 w-full rounded-sm"
                style={{ background: "var(--gridline)" }}
              >
                {/* Square at the baseline, 4px rounded at the data end. */}
                <div
                  className="h-full"
                  style={{
                    width: `${Math.max(fraction * 100, fraction > 0 ? 1.5 : 0)}%`,
                    background: rampStep(fraction),
                    borderRadius: "0 4px 4px 0",
                  }}
                />
              </div>
              {d.sub && (
                <div className="mt-0.5 text-xs" style={{ color: "var(--text-muted)" }}>
                  {d.sub}
                </div>
              )}
            </li>
          );
        })}
      </ul>
      <TableView
        caption={tableCaption ?? title}
        headers={["Item", "Value"]}
        rows={data.map((d) => [d.label, valueFormatter(d.value)])}
      />
    </div>
  );
}

// --- diverging delta plot (experiment comparisons) ----------------------

export type DeltaDatum = {
  label: string;
  value: number;
  low: number;
  high: number;
  significant: boolean;
  annotation?: string;
};

/**
 * A dot per condition at its mean difference, with its 95% interval, on a
 * shared axis centred at zero.
 *
 * Bars would be the wrong form here: the quantity is a signed difference with
 * uncertainty, and a bar implies a magnitude from zero while hiding the
 * interval entirely. The interval is the part a reader needs, because an
 * effect whose interval crosses zero is not an effect.
 */
export function DeltaPlot({
  data,
  metric,
  baseline,
}: {
  data: DeltaDatum[];
  metric: string;
  baseline: string;
}) {
  const headerId = useId();

  if (data.length === 0) {
    return (
      <p className="text-sm" style={{ color: "var(--text-muted)" }}>
        No comparisons available.
      </p>
    );
  }

  const bound = Math.max(
    0.05,
    ...data.flatMap((d) => [Math.abs(d.low), Math.abs(d.high), Math.abs(d.value)]),
  );
  const scale = (v: number) => ((v + bound) / (2 * bound)) * 100;

  return (
    <div>
      <p id={headerId} className="text-xs" style={{ color: "var(--text-secondary)" }}>
        Difference in <span className="font-medium">{metric}</span> against{" "}
        <span className="font-medium">{baseline}</span>, with 95% confidence
        intervals. An interval crossing zero is not a detected effect.
      </p>

      <ul className="mt-4 space-y-5" aria-describedby={headerId}>
        {data.map((d) => {
          const positive = d.value >= 0;
          const colour = positive ? "var(--diverge-pos)" : "var(--diverge-neg)";
          const left = scale(Math.min(d.low, d.value));
          const right = scale(Math.max(d.high, d.value));

          return (
            <li key={d.label}>
              <div className="flex items-baseline justify-between gap-3">
                <span className="truncate text-sm" title={d.label}>
                  {d.label}
                </span>
                <span className="shrink-0 text-sm font-medium tnum">
                  {d.value >= 0 ? "+" : ""}
                  {formatNumber(d.value, 4)}
                  {/* Significance is stated in text, never by colour alone. */}
                  <span
                    className="ml-2 text-xs font-normal"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {d.significant ? "significant" : "not significant"}
                  </span>
                </span>
              </div>

              <div className="relative mt-1.5 h-5">
                {/* Zero line: the reference the whole plot is read against. */}
                <div
                  className="absolute inset-y-0 w-px"
                  style={{ left: "50%", background: "var(--baseline)" }}
                  aria-hidden
                />
                {/* Confidence interval. */}
                <div
                  className="absolute top-1/2 h-0.5 -translate-y-1/2 rounded-full"
                  style={{
                    left: `${left}%`,
                    width: `${Math.max(right - left, 0.5)}%`,
                    background: colour,
                    opacity: 0.55,
                  }}
                  aria-hidden
                />
                {/* Point estimate: >=8px with a 2px surface ring. */}
                <div
                  className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
                  style={{
                    left: `${scale(d.value)}%`,
                    background: colour,
                    boxShadow: "0 0 0 2px var(--surface-1)",
                  }}
                  title={`${formatNumber(d.value, 4)} (95% CI ${formatNumber(
                    d.low,
                    4,
                  )} to ${formatNumber(d.high, 4)})`}
                />
              </div>

              <div className="mt-0.5 text-xs tnum" style={{ color: "var(--text-muted)" }}>
                95% CI [{formatNumber(d.low, 4)}, {formatNumber(d.high, 4)}]
                {d.annotation ? ` · ${d.annotation}` : ""}
              </div>
            </li>
          );
        })}
      </ul>

      <div
        className="mt-2 flex justify-between text-xs tnum"
        style={{ color: "var(--text-muted)" }}
      >
        <span>−{formatNumber(bound, 3)}</span>
        <span>0</span>
        <span>+{formatNumber(bound, 3)}</span>
      </div>

      <TableView
        caption={`Paired comparisons on ${metric} against ${baseline}`}
        headers={["Condition", "Difference", "CI low", "CI high", "Significant"]}
        rows={data.map((d) => [
          d.label,
          formatNumber(d.value, 4),
          formatNumber(d.low, 4),
          formatNumber(d.high, 4),
          d.significant ? "yes" : "no",
        ])}
      />
    </div>
  );
}
