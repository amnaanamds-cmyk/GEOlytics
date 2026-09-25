import { ExperimentList } from "@/components/experiment-list";
import { load } from "@/lib/server-api";

export const metadata = { title: "Experiments · GEOlytics" };
export const dynamic = "force-dynamic";

export default async function ExperimentsPage() {
  const names = await load<string[]>("experiments");
  return <ExperimentList names={names.data ?? []} loadError={names.error} />;
}
