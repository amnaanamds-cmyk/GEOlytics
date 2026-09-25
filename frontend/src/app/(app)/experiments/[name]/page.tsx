import { ExperimentDetail } from "@/components/experiment-detail";
import { load } from "@/lib/server-api";
import type { Experiment } from "@/lib/types";

export const metadata = { title: "Experiment · GEOlytics" };
export const dynamic = "force-dynamic";

export default async function ExperimentPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const { name } = await params;
  const decoded = decodeURIComponent(name);
  const path = `experiments/${encodeURIComponent(decoded)}`;

  // Load once to learn the conditions, then default to comparing everything
  // against the first one. All-pairs is k(k-1)/2 comparisons -- 28 for eight
  // conditions -- which is both unreadable and a far harsher multiple-testing
  // correction than the k-1 a baseline needs.
  const discovery = await load<Experiment>(`${path}?metric=ndcg@10`);
  const baseline = discovery.data?.runs[0]?.condition ?? "";

  const initial = baseline
    ? await load<Experiment>(
        `${path}?metric=ndcg@10&baseline=${encodeURIComponent(baseline)}`,
      )
    : discovery;

  return (
    <ExperimentDetail
      name={decoded}
      initialData={initial.data}
      initialBaseline={baseline}
      loadError={initial.error}
    />
  );
}
