import { AuditDetail } from "@/components/audit-detail";
import { load } from "@/lib/server-api";
import type { Audit } from "@/lib/types";

export const metadata = { title: "Audit · GEOlytics" };
export const dynamic = "force-dynamic";

// `params` is a Promise in this version of Next.js and must be awaited.
export default async function AuditPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const audit = await load<Audit>(`audits/${encodeURIComponent(id)}`);

  return <AuditDetail auditId={id} initialAudit={audit.data} loadError={audit.error} />;
}
