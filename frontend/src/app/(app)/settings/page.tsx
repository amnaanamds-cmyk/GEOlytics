import { SettingsView } from "@/components/settings-view";
import { load, loadOptional } from "@/lib/server-api";
import type { ApiKey, Member, Usage } from "@/lib/types";
import type { Org, Plan } from "@/components/settings-view";

export const metadata = { title: "Settings · GEOlytics" };
export const dynamic = "force-dynamic";

export default async function SettingsPage() {
  // Keys and members need elevated scopes. A viewer should see the page
  // without an error banner, so those two are loaded optionally.
  const [org, usage, plans, keys, members] = await Promise.all([
    load<Org>("org"),
    load<Usage>("org/usage"),
    load<Plan[]>("org/plans"),
    loadOptional<ApiKey[]>("org/keys"),
    loadOptional<Member[]>("org/members"),
  ]);

  return (
    <SettingsView
      initialOrg={org.data}
      initialUsage={usage.data}
      initialPlans={plans.data ?? []}
      initialKeys={keys ?? []}
      initialMembers={members ?? []}
      loadError={org.error}
    />
  );
}
