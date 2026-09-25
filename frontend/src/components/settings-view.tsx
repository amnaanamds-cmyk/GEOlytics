"use client";

import { useCallback, useState } from "react";

import { Meter } from "@/components/charts";
import { Alert, Button, Card, EmptyState, Field } from "@/components/ui";
import { RequestError, del, get, post } from "@/lib/client";
import type { ApiKey, Member, Usage } from "@/lib/types";

export type Org = {
  id: number;
  slug: string;
  name: string;
  plan: string;
  status: string;
};
export type Plan = {
  name: string;
  display_name: string;
  monthly_price_cents: number;
  max_pages_per_month: number;
  max_audits_per_month: number;
  experiments_enabled: boolean;
};

export function SettingsView({
  initialOrg,
  initialUsage,
  initialPlans,
  initialKeys,
  initialMembers,
  loadError,
}: {
  initialOrg: Org | null;
  initialUsage: Usage | null;
  initialPlans: Plan[];
  initialKeys: ApiKey[];
  initialMembers: Member[];
  loadError: string | null;
}) {
  const [org] = useState<Org | null>(initialOrg);
  const [usage, setUsage] = useState<Usage | null>(initialUsage);
  const [keys, setKeys] = useState<ApiKey[]>(initialKeys);
  const [members] = useState<Member[]>(initialMembers);
  const [plans] = useState<Plan[]>(initialPlans);
  const [freshKey, setFreshKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(loadError);
  const [busy, setBusy] = useState(false);

  /** Refresh only what a mutation here can change. */
  const load = useCallback(async () => {
    try {
      const [nextKeys, nextUsage] = await Promise.all([
        get<ApiKey[]>("org/keys"),
        get<Usage>("org/usage"),
      ]);
      setKeys(nextKeys);
      setUsage(nextUsage);
    } catch {
      // A failed refresh leaves the last good view in place; the mutation
      // itself already reported its own outcome.
    }
  }, []);

  async function createKey(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    const form = new FormData(event.currentTarget);
    try {
      const created = await post<ApiKey & { token: string }>("org/keys", {
        name: form.get("name"),
        environment: "live",
      });
      setFreshKey(created.token);
      (event.target as HTMLFormElement).reset();
      await load();
    } catch (err) {
      if (err instanceof RequestError && err.status === 402) {
        setError(`${err.message} — upgrade to add more keys.`);
      } else {
        setError(err instanceof Error ? err.message : "Could not create the key.");
      }
    } finally {
      setBusy(false);
    }
  }

  async function revokeKey(id: number) {
    setError(null);
    try {
      await del(`org/keys/${id}`);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not revoke the key.");
    }
  }

  return (
    <div className="space-y-6">
      {error && <Alert tone="error">{error}</Alert>}

      <Card title="Organisation">
        {org ? (
          <dl className="grid gap-3 text-sm sm:grid-cols-3">
            <div>
              <dt style={{ color: "var(--text-secondary)" }}>Name</dt>
              <dd className="font-medium">{org.name}</dd>
            </div>
            <div>
              <dt style={{ color: "var(--text-secondary)" }}>Plan</dt>
              <dd className="font-medium capitalize">{org.plan}</dd>
            </div>
            <div>
              <dt style={{ color: "var(--text-secondary)" }}>Status</dt>
              <dd className="font-medium capitalize">{org.status}</dd>
            </div>
          </dl>
        ) : (
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            Loading…
          </p>
        )}
      </Card>

      {usage && (
        <Card title="Usage" description={`Billing period ${usage.period}`}>
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
              label="Experiment runs"
              used={usage.used.experiment_runs}
              limit={usage.limits.experiment_runs}
            />
          </div>
        </Card>
      )}

      <Card
        title="API keys"
        description="Machine access, scoped to this organisation. A key can never do more than the person who created it."
      >
        {freshKey && (
          <div className="mb-4">
            <Alert tone="warning">
              <p className="font-medium">Copy this key now — it is not shown again.</p>
              <code className="mt-2 block break-all rounded px-2 py-1.5 text-xs tnum"
                style={{ background: "var(--gridline)" }}>
                {freshKey}
              </code>
            </Alert>
          </div>
        )}

        <form onSubmit={createKey} className="mb-4 flex flex-wrap items-end gap-3">
          <div className="min-w-[14rem] flex-1">
            <Field label="Key name" name="name" required placeholder="CI pipeline" />
          </div>
          <Button type="submit" loading={busy}>
            Create key
          </Button>
        </form>

        {keys.length === 0 ? (
          <EmptyState title="No API keys yet" />
        ) : (
          <ul className="divide-y" style={{ borderColor: "var(--gridline)" }}>
            {keys.map((key) => (
              <li key={key.id} className="flex flex-wrap items-center gap-3 py-3">
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">{key.name}</p>
                  <p className="text-xs tnum" style={{ color: "var(--text-muted)" }}>
                    {key.display_hint} · {key.scopes.length} scopes ·{" "}
                    {key.last_used_at
                      ? `last used ${new Date(key.last_used_at).toLocaleDateString()}`
                      : "never used"}
                  </p>
                </div>
                {key.revoked_at ? (
                  <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                    revoked
                  </span>
                ) : (
                  <Button variant="danger" onClick={() => revokeKey(key.id)}>
                    Revoke
                  </Button>
                )}
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card title="Members">
        {members.length === 0 ? (
          <EmptyState title="Nothing to show">
            You may not have permission to view members.
          </EmptyState>
        ) : (
          <ul className="divide-y" style={{ borderColor: "var(--gridline)" }}>
            {members.map((member) => (
              <li key={member.user_id} className="flex items-center gap-3 py-3">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">
                    {member.full_name ?? member.email}
                  </p>
                  <p className="truncate text-xs" style={{ color: "var(--text-muted)" }}>
                    {member.email}
                  </p>
                </div>
                <span className="text-xs capitalize" style={{ color: "var(--text-secondary)" }}>
                  {member.role}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card title="Plans">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {plans.map((plan) => (
            <div
              key={plan.name}
              className="rounded-lg p-4"
              style={{
                border: `1px solid ${
                  org?.plan === plan.name ? "var(--series-1)" : "var(--border)"
                }`,
              }}
            >
              <p className="text-sm font-semibold">{plan.display_name}</p>
              <p className="mt-1 text-2xl font-semibold tnum">
                {plan.monthly_price_cents === 0
                  ? plan.name === "enterprise"
                    ? "Custom"
                    : "Free"
                  : `$${(plan.monthly_price_cents / 100).toFixed(0)}`}
                {plan.monthly_price_cents > 0 && (
                  <span className="text-xs font-normal" style={{ color: "var(--text-muted)" }}>
                    {" "}
                    /mo
                  </span>
                )}
              </p>
              <ul className="mt-2 space-y-1 text-xs" style={{ color: "var(--text-secondary)" }}>
                <li>{plan.max_pages_per_month.toLocaleString()} pages/month</li>
                <li>{plan.max_audits_per_month.toLocaleString()} audits/month</li>
                <li>{plan.experiments_enabled ? "Experiments included" : "No experiments"}</li>
              </ul>
              {org?.plan === plan.name && (
                <p className="mt-2 text-xs font-medium" style={{ color: "var(--series-1)" }}>
                  Current plan
                </p>
              )}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
