"use client";

import Link from "next/link";
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from "react";

export function Button({
  children,
  variant = "primary",
  loading,
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "danger";
  loading?: boolean;
}) {
  const styles = {
    primary: { background: "var(--series-1)", color: "#ffffff", border: "1px solid transparent" },
    secondary: {
      background: "var(--surface-1)",
      color: "var(--text-primary)",
      border: "1px solid var(--border)",
    },
    danger: {
      background: "transparent",
      color: "var(--status-critical)",
      border: "1px solid var(--status-critical)",
    },
  }[variant];

  return (
    <button
      {...props}
      disabled={props.disabled || loading}
      style={{ ...styles, ...props.style }}
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-opacity disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
    >
      {loading && (
        <span
          className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent"
          aria-hidden
        />
      )}
      {children}
    </button>
  );
}

export function Field({
  label,
  hint,
  error,
  ...props
}: InputHTMLAttributes<HTMLInputElement> & {
  label: string;
  hint?: string;
  error?: string;
}) {
  const id = props.id ?? props.name ?? label.replace(/\s+/g, "-").toLowerCase();
  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="block text-sm font-medium">
        {label}
      </label>
      <input
        {...props}
        id={id}
        aria-invalid={Boolean(error)}
        aria-describedby={hint || error ? `${id}-hint` : undefined}
        className="w-full rounded-lg px-3 py-2 text-sm outline-none"
        style={{
          background: "var(--surface-1)",
          color: "var(--text-primary)",
          border: `1px solid ${error ? "var(--status-critical)" : "var(--border)"}`,
        }}
      />
      {(hint || error) && (
        <p
          id={`${id}-hint`}
          className="text-xs"
          style={{ color: error ? "var(--status-critical)" : "var(--text-muted)" }}
        >
          {error ?? hint}
        </p>
      )}
    </div>
  );
}

export function Alert({
  tone = "info",
  children,
}: {
  tone?: "info" | "warning" | "error" | "success";
  children: ReactNode;
}) {
  // Tone is carried by an icon and the text, never by colour alone.
  const config = {
    info: { icon: "i", colour: "var(--series-1)" },
    warning: { icon: "!", colour: "var(--status-warning)" },
    error: { icon: "!", colour: "var(--status-critical)" },
    success: { icon: "✓", colour: "var(--status-good-text)" },
  }[tone];

  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className="flex gap-2.5 rounded-lg p-3 text-sm"
      style={{ background: "var(--surface-1)", border: `1px solid ${config.colour}` }}
    >
      <span
        aria-hidden
        className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] font-bold"
        style={{ background: config.colour, color: "#ffffff" }}
      >
        {config.icon}
      </span>
      <div style={{ color: "var(--text-primary)" }}>{children}</div>
    </div>
  );
}

const STATUS_TONE: Record<string, { colour: string; icon: string }> = {
  complete: { colour: "var(--status-good)", icon: "✓" },
  running: { colour: "var(--series-1)", icon: "•" },
  pending: { colour: "var(--text-muted)", icon: "•" },
  failed: { colour: "var(--status-critical)", icon: "!" },
};

export function StatusPill({ status }: { status: string }) {
  const tone = STATUS_TONE[status] ?? STATUS_TONE.pending;
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium"
      style={{ border: `1px solid ${tone.colour}`, color: "var(--text-primary)" }}
    >
      <span aria-hidden style={{ color: tone.colour }}>
        {tone.icon}
      </span>
      {status}
    </span>
  );
}

export function Card({
  title,
  action,
  children,
  description,
}: {
  title?: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="card p-5">
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div>
            {title && <h2 className="text-sm font-semibold">{title}</h2>}
            {description && (
              <p className="mt-0.5 text-xs" style={{ color: "var(--text-secondary)" }}>
                {description}
              </p>
            )}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div
      className="rounded-lg px-4 py-10 text-center"
      style={{ border: `1px dashed var(--border)` }}
    >
      <p className="text-sm font-medium">{title}</p>
      {children && (
        <div className="mt-1 text-xs" style={{ color: "var(--text-secondary)" }}>
          {children}
        </div>
      )}
    </div>
  );
}

export function NavLink({
  href,
  active,
  children,
}: {
  href: string;
  active: boolean;
  children: ReactNode;
}) {
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className="rounded-lg px-3 py-1.5 text-sm font-medium transition-colors"
      style={{
        background: active ? "var(--surface-1)" : "transparent",
        color: active ? "var(--text-primary)" : "var(--text-secondary)",
        border: `1px solid ${active ? "var(--border)" : "transparent"}`,
      }}
    >
      {children}
    </Link>
  );
}
