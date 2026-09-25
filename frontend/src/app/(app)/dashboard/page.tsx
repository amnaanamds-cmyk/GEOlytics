import { DashboardView } from "@/components/dashboard-view";
import { load } from "@/lib/server-api";
import type { AuditList, Usage } from "@/lib/types";

export const metadata = { title: "Audits · GEOlytics" };

// Rendered per request: an audit list is per-customer and changes constantly.
export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const [audits, usage] = await Promise.all([
    load<AuditList>("audits?limit=25"),
    load<Usage>("org/usage"),
  ]);

  return (
    <DashboardView
      initialAudits={audits.data?.items ?? []}
      initialUsage={usage.data}
      loadError={audits.error ?? usage.error}
    />
  );
}
