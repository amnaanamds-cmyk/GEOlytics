"""Parent-child (small-to-big) chunking."""

from __future__ import annotations

from typing import Any

from geolytics.chunking.base import Chunk, ChunkingStrategy, Document, Section, make_chunk_id


class ParentChildChunker(ChunkingStrategy):
    """Index small children, return their large parents.

    The retrieval hypothesis this tests: a small chunk embeds a single focused
    idea and therefore matches a focused query well, but a small chunk is poor
    *context* for the generator. Indexing the child and handing the generator
    the parent decouples the two.

    Implementation note: the child text is what gets embedded and what
    relevance is judged against (`Chunk.chunk_id` is the child's). The parent
    is carried on `Chunk.parent_text` and surfaced via `Chunk.context_text`,
    which is what the answer-synthesis step reads.
    """

    name = "parent_child"

    def __init__(self, parent: ChunkingStrategy, child: ChunkingStrategy) -> None:
        self.parent = parent
        self.child = child

    @property
    def params(self) -> dict[str, Any]:
        return {"parent": self.parent.describe(), "child": self.child.describe()}

    def chunk(self, document: Document) -> list[Chunk]:
        parents = self.parent.chunk(document)
        children: list[Chunk] = []
        ordinal = 0

        for parent_chunk in parents:
            sub_document = Document(
                doc_id=document.doc_id,
                url=document.url,
                title=document.title,
                text=parent_chunk.text,
                sections=(
                    Section(
                        text=parent_chunk.text,
                        heading_path=parent_chunk.heading_path,
                        start_char=parent_chunk.start_char,
                    ),
                ),
                metadata=document.metadata,
            )
            for child_chunk in self.child.chunk(sub_document):
                children.append(
                    Chunk(
                        chunk_id=make_chunk_id(
                            document.doc_id, self.name, ordinal, child_chunk.text
                        ),
                        doc_id=document.doc_id,
                        text=child_chunk.text,
                        ordinal=ordinal,
                        start_char=child_chunk.start_char,
                        end_char=child_chunk.end_char,
                        heading_path=parent_chunk.heading_path,
                        parent_id=parent_chunk.chunk_id,
                        parent_text=parent_chunk.text,
                        metadata={
                            **child_chunk.metadata,
                            "parent_ordinal": parent_chunk.ordinal,
                        },
                    )
                )
                ordinal += 1

        return children
