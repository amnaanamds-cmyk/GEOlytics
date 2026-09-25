import type { ReactNode } from "react";

export function AuthShell({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
}) {
  return (
    <main className="mx-auto flex min-h-dvh w-full max-w-md flex-col justify-center px-4 py-12">
      <div className="mb-8">
        <p className="text-sm font-semibold tracking-tight">GEOlytics</p>
        <h1 className="mt-4 text-2xl font-semibold tracking-tight">{title}</h1>
        <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
          {subtitle}
        </p>
      </div>
      <div className="card p-6">{children}</div>
      <p className="mt-6 text-xs" style={{ color: "var(--text-muted)" }}>
        GEOlytics evaluates factors that plausibly affect AI-search visibility by
        simulating a retrieval-augmented generative engine over your site. It does
        not observe the internal ranking of any commercial answer engine.
      </p>
    </main>
  );
}
