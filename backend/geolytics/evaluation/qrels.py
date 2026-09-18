"""Queries and relevance judgments (qrels).

This is the module the whole evaluation chapter rests on. Retrieval precision
and recall are undefined without judgments, and no judgment set exists for an
arbitrary crawled website -- it has to be constructed.

**The portability problem.** A judgment that names a chunk id is useless the
moment the chunker changes: fixed-size chunk #7 and semantic chunk #7 are
different text, so a judgment set built under one strategy cannot score
another. Since comparing chunking strategies is the entire point of this
system, judgments are anchored to the *source document* instead -- a character
span plus the gold answer text -- and projected onto whichever chunks a
strategy produced (see `geolytics.evaluation.projection`).

`provenance` records how each judgment was obtained, so synthetic and
human-labelled judgments can be reported separately and their agreement
measured.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Provenance = Literal["synthetic", "human", "heuristic"]


@dataclass(frozen=True, slots=True)
class Relevance:
    """A judgment naming a concrete chunk. Produced by projection, not authored.

    Grades follow the TREC convention: 0 irrelevant, 1 relevant, 2 highly
    relevant. Binary judgment sets simply never use 2.
    """

    chunk_id: str
    grade: int = 1

    def __post_init__(self) -> None:
        if self.grade < 0:
            raise ValueError("grade must be non-negative")


@dataclass(frozen=True, slots=True)
class SpanRelevance:
    """A chunker-independent judgment: the passage of `doc_id` that answers the query.

    `text` is the verbatim gold passage. It is kept alongside the offsets
    because offsets drift -- the fixed-size chunker defines windows over tokens
    and re-joins them, which normalises internal whitespace -- so projection
    falls back to normalised text containment when offsets disagree.
    """

    doc_id: str
    start_char: int
    end_char: int
    grade: int = 1
    text: str | None = None

    def __post_init__(self) -> None:
        if self.grade < 0:
            raise ValueError("grade must be non-negative")
        if self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")

    @property
    def length(self) -> int:
        return self.end_char - self.start_char


@dataclass(frozen=True, slots=True)
class Query:
    """An evaluation query.

    `spans` is the authored, portable ground truth. `judgments` is the
    projection of those spans onto one chunking, and is filled in by
    `projection.project_query` immediately before scoring.
    """

    query_id: str
    text: str
    spans: tuple[SpanRelevance, ...] = ()
    judgments: tuple[Relevance, ...] = ()
    provenance: Provenance = "synthetic"
    source_doc_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def relevant_ids(self) -> frozenset[str]:
        return frozenset(j.chunk_id for j in self.judgments if j.grade > 0)

    @property
    def grades(self) -> dict[str, int]:
        return {j.chunk_id: j.grade for j in self.judgments}

    @property
    def n_relevant(self) -> int:
        return len(self.relevant_ids)

    def with_judgments(self, judgments: Iterable[Relevance]) -> Query:
        from dataclasses import replace

        return replace(self, judgments=tuple(judgments))


@dataclass(slots=True)
class QuerySet:
    """A named collection of queries -- the evaluation set for one site."""

    name: str
    queries: list[Query] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Query]:
        return iter(self.queries)

    def __len__(self) -> int:
        return len(self.queries)

    def judged(self) -> QuerySet:
        """Drop queries whose projection found no relevant chunk.

        An unjudged query scores 0 on every metric under every condition, which
        deflates all conditions equally and shrinks the differences under test.
        Filter before scoring, and report how many were dropped -- a strategy
        that loses many queries to projection failure is itself a finding.
        """
        dropped = sum(1 for q in self.queries if q.n_relevant == 0)
        return QuerySet(
            name=f"{self.name}:judged",
            queries=[q for q in self.queries if q.n_relevant > 0],
            metadata={**self.metadata, "dropped_unjudged": dropped},
        )

    def filter_provenance(self, provenance: Provenance) -> QuerySet:
        return QuerySet(
            name=f"{self.name}:{provenance}",
            queries=[q for q in self.queries if q.provenance == provenance],
            metadata=dict(self.metadata),
        )

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for query in self.queries:
                fh.write(json.dumps(asdict(query), ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str | None = None) -> QuerySet:
        path = Path(path)
        queries = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                raw = json.loads(line)
                raw["spans"] = tuple(SpanRelevance(**s) for s in raw.get("spans", []))
                raw["judgments"] = tuple(Relevance(**j) for j in raw.get("judgments", []))
                queries.append(Query(**raw))
        return cls(name=name or path.stem, queries=queries)

    @classmethod
    def from_queries(cls, name: str, queries: Iterable[Query]) -> QuerySet:
        return cls(name=name, queries=list(queries))
