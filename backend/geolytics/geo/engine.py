"""A simulated generative engine: retrieve, answer, cite.

This is what makes the visibility metrics measurable. A real generative engine
is a closed system; this one is fully observable, so every factor -- which
chunks were retrieved, which were cited, how much of the answer each
contributed -- can be logged and attributed.

What it licenses you to claim: "under a retrieval-augmented generation pipeline
with these components, content with property X was cited more often". What it
does not license: any statement about what ChatGPT, Perplexity or AI Overviews
did or why.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from geolytics.chunking.base import split_sentences
from geolytics.geo.visibility import AnswerSentence, ImpressionMetrics, citation_visibility
from geolytics.index.base import ScoredChunk
from geolytics.llm import LLMClient
from geolytics.retrieval.base import Retriever

SYSTEM_PROMPT = (
    "You answer questions using only the numbered sources provided. "
    "You cite every factual sentence."
)

ANSWER_TEMPLATE = """\
Answer the question using ONLY the sources below.

Rules:
- Cite the source for every factual sentence, as [1], [2], etc.
- A sentence may cite more than one source: [1][3].
- If the sources do not answer the question, say so plainly and cite nothing.
- Do not use any knowledge beyond these sources.

Sources:
{sources}

Question: {question}

Answer:"""

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """One simulated engine response, with the full attribution trail."""

    query: str
    text: str
    sentences: tuple[AnswerSentence, ...]
    retrieved: tuple[ScoredChunk, ...]
    n_uncited_sentences: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cited_chunk_ids(self) -> set[str]:
        return {cid for s in self.sentences for cid in s.cited_chunk_ids}

    def visibility_for(self, chunk_ids: Sequence[str] | set[str]) -> ImpressionMetrics:
        return citation_visibility(self.sentences, chunk_ids)

    def visibility_for_doc(self, doc_id: str) -> ImpressionMetrics:
        target = {c.chunk_id for c in self.retrieved if c.chunk.doc_id == doc_id}
        return citation_visibility(self.sentences, target)


class SimulatedGenerativeEngine:
    """Retriever + LLM, with citations resolved back to chunk ids."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        top_k: int = 5,
        temperature: float = 0.0,
        max_source_chars: int = 1500,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.temperature = temperature
        self.max_source_chars = max_source_chars

    def answer(self, question: str) -> GeneratedAnswer:
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        if not retrieved:
            return GeneratedAnswer(
                query=question,
                text="",
                sentences=(),
                retrieved=(),
                n_uncited_sentences=0,
                metadata={"reason": "no chunks retrieved"},
            )

        prompt = ANSWER_TEMPLATE.format(
            sources=self._format_sources(retrieved), question=question
        )
        text = self.llm.complete(prompt, system=SYSTEM_PROMPT, temperature=self.temperature)
        sentences = parse_citations(text, retrieved)

        return GeneratedAnswer(
            query=question,
            text=text,
            sentences=sentences,
            retrieved=tuple(retrieved),
            n_uncited_sentences=sum(1 for s in sentences if not s.cited_chunk_ids),
            metadata={
                "retriever": self.retriever.describe(),
                "llm": self.llm.describe(),
                "top_k": self.top_k,
            },
        )

    def _format_sources(self, retrieved: Sequence[ScoredChunk]) -> str:
        blocks = []
        for i, hit in enumerate(retrieved, start=1):
            # context_text is the parent for hierarchical chunking, the chunk
            # itself otherwise -- the generator should see the wider passage.
            body = hit.chunk.context_text[: self.max_source_chars]
            url = hit.chunk.metadata.get("url", "")
            blocks.append(f"[{i}] ({url})\n{body}")
        return "\n\n".join(blocks)


def parse_citations(
    answer_text: str, retrieved: Sequence[ScoredChunk]
) -> tuple[AnswerSentence, ...]:
    """Split the answer into sentences and resolve [n] markers to chunk ids.

    Out-of-range markers are dropped rather than clamped: a small model that
    invents "[7]" when given five sources has hallucinated an attribution, and
    silently mapping it onto a real source would manufacture a citation that
    was never made. The dropped count surfaces as an uncited sentence.
    """
    sentences = split_sentences(answer_text)
    out: list[AnswerSentence] = []

    for sentence in sentences:
        indices = [int(m) for m in _CITATION_RE.findall(sentence)]
        chunk_ids = tuple(
            dict.fromkeys(
                retrieved[i - 1].chunk_id for i in indices if 1 <= i <= len(retrieved)
            )
        )
        out.append(AnswerSentence(text=sentence, cited_chunk_ids=chunk_ids))

    return tuple(out)
