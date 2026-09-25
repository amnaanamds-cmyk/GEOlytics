"use client";

import Link from "next/link";

import { Alert, Card, EmptyState } from "@/components/ui";

export function ExperimentList({
  names,
  loadError,
}: {
  names: string[];
  loadError: string | null;
}) {
  if (loadError) return <Alert tone="error">{loadError}</Alert>;

  return (
    <div className="space-y-6">
      <Card
        title="Experiments"
        description="Chunking and retrieval strategies compared on the same queries, with paired significance tests."
      >
        {names.length === 0 ? (
          <EmptyState title="No experiments stored yet">
            Run{" "}
            <code className="tnum">
              geolytics experiment &lt;url&gt; --persist-as &lt;name&gt;
            </code>{" "}
            to store a grid here.
          </EmptyState>
        ) : (
          <ul className="space-y-2">
            {names.map((name) => (
              <li key={name}>
                <Link
                  href={`/experiments/${encodeURIComponent(name)}`}
                  className="block rounded-lg px-3 py-2 text-sm underline-offset-2 hover:underline"
                  style={{ border: "1px solid var(--border)" }}
                >
                  {name}
                </Link>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
