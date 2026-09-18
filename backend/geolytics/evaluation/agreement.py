"""Agreement between synthetic and human relevance judgments.

This is the module that defends the evaluation methodology. Synthetic
judgments are cheap and plentiful; human judgments are expensive and few. The
claim being tested is that the cheap labels agree with the expensive ones well
enough to trust the cheap ones at scale.

Protocol to follow (and to describe in the report):

1. Run every condition and pool the top-k results per query -- TREC-style depth
   pooling. The pool is the set of (query, chunk) pairs any system surfaced.
2. Sample ~100 queries and label every pooled chunk by hand, blind to the
   synthetic label.
3. Compute `judgment_agreement` over those pairs.
4. Report kappa with the interpretation, and repeat the headline comparison on
   the human-labelled subset alone. If the two subsets rank the conditions the
   same way, the synthetic set is doing its job.

Cohen's kappa rather than raw agreement: relevance pools are dominated by
irrelevant pairs, so two labellings that both say "irrelevant" almost always
can show 90% raw agreement while carrying no information. Kappa subtracts the
agreement expected by chance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from geolytics.evaluation.qrels import QuerySet


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """2x2 agreement between two binary labellings of the same pairs."""

    n_pairs: int
    both_relevant: int
    both_irrelevant: int
    only_a: int
    only_b: int
    raw_agreement: float
    cohens_kappa: float
    label_a: str = "synthetic"
    label_b: str = "human"

    @property
    def interpretation(self) -> str:
        """Landis & Koch (1977) benchmarks."""
        k = self.cohens_kappa
        if k < 0.0:
            return "worse than chance"
        if k < 0.21:
            return "slight"
        if k < 0.41:
            return "fair"
        if k < 0.61:
            return "moderate"
        if k < 0.81:
            return "substantial"
        return "almost perfect"

    def summary(self) -> str:
        return (
            f"{self.label_a} vs {self.label_b}: kappa={self.cohens_kappa:.3f} "
            f"({self.interpretation}), raw agreement={self.raw_agreement:.1%}, "
            f"n={self.n_pairs} pairs "
            f"({self.both_relevant} both relevant, {self.only_a} {self.label_a}-only, "
            f"{self.only_b} {self.label_b}-only)"
        )


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa for two binary labellings of the same items."""
    if len(a) != len(b):
        raise ValueError("labellings must cover the same items")
    n = len(a)
    if n == 0:
        return float("nan")

    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    p_a = sum(a) / n
    p_b = sum(b) / n
    expected = p_a * p_b + (1.0 - p_a) * (1.0 - p_b)
    if expected == 1.0:
        # Both labellings are constant and identical: agreement is total, but
        # kappa is undefined (0/0). Report 1.0 and flag it in the write-up.
        return 1.0
    return (observed - expected) / (1.0 - expected)


def judgment_agreement(
    labelling_a: QuerySet,
    labelling_b: QuerySet,
    pool: Mapping[str, Sequence[str]],
    label_a: str = "synthetic",
    label_b: str = "human",
) -> AgreementResult:
    """Compare two judgment sets over a pooled candidate set.

    `pool` maps query_id to the chunk ids judged for that query. Only queries
    present in both labellings and in the pool are considered -- a pair nobody
    was asked about is not a disagreement.
    """
    by_id_a = {q.query_id: q for q in labelling_a}
    by_id_b = {q.query_id: q for q in labelling_b}

    flags_a: list[bool] = []
    flags_b: list[bool] = []

    for query_id in sorted(set(by_id_a) & set(by_id_b) & set(pool)):
        relevant_a = by_id_a[query_id].relevant_ids
        relevant_b = by_id_b[query_id].relevant_ids
        for chunk_id in pool[query_id]:
            flags_a.append(chunk_id in relevant_a)
            flags_b.append(chunk_id in relevant_b)

    n = len(flags_a)
    both_relevant = sum(1 for x, y in zip(flags_a, flags_b, strict=True) if x and y)
    both_irrelevant = sum(1 for x, y in zip(flags_a, flags_b, strict=True) if not x and not y)
    only_a = sum(1 for x, y in zip(flags_a, flags_b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(flags_a, flags_b, strict=True) if y and not x)

    return AgreementResult(
        n_pairs=n,
        both_relevant=both_relevant,
        both_irrelevant=both_irrelevant,
        only_a=only_a,
        only_b=only_b,
        raw_agreement=((both_relevant + both_irrelevant) / n) if n else float("nan"),
        cohens_kappa=cohens_kappa(flags_a, flags_b),
        label_a=label_a,
        label_b=label_b,
    )


def build_pool(
    rankings: Sequence[Mapping[str, Sequence[str]]], depth: int = 10
) -> dict[str, list[str]]:
    """Depth-pool the rankings of several conditions into one candidate set per query.

    Pass `RunResult.ranked` for each condition. Pooling to a fixed depth across
    all systems is the standard way to bound human labelling effort without
    biasing the pool toward any one system.
    """
    pool: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for ranking in rankings:
        for query_id, chunk_ids in ranking.items():
            bucket = pool.setdefault(query_id, [])
            marks = seen.setdefault(query_id, set())
            for chunk_id in chunk_ids[:depth]:
                if chunk_id not in marks:
                    marks.add(chunk_id)
                    bucket.append(chunk_id)
    return pool
