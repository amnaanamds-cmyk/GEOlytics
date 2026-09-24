"""Shared fixtures. Nothing here touches the network or a running service."""

from __future__ import annotations

import pytest

from geolytics.chunking.base import Document, Section
from geolytics.embedding.hashing import HashingEmbedder
from geolytics.evaluation.qrels import Query, QuerySet, SpanRelevance


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=128, seed=7)


@pytest.fixture
def simple_document() -> Document:
    text = (
        "Acme Plumbing has served Lahore since 2004. "
        "Acme dispatches a plumber within 90 minutes, 24 hours a day. "
        "An emergency callout costs 2,500 PKR. "
        "A full bathroom installation takes four to six working days. "
        "Acme sources fittings from three certified suppliers."
    )
    return Document(
        doc_id="acme",
        url="https://acme.example/",
        title="Acme Plumbing",
        text=text,
        sections=(Section(text=text, heading_path=("Acme Plumbing",), start_char=0),),
    )


@pytest.fixture
def corpus() -> list[Document]:
    """A corpus with enough distinct topics for retrieval to be non-trivial."""
    topics = [
        ("drain", "Blocked drain clearing uses a motorised auger on the main stack."),
        ("boiler", "Boiler servicing covers the heat exchanger, flue and pressure valve."),
        ("leak", "Leak detection uses acoustic sensors before any wall is opened."),
        ("tap", "Tap replacement takes under an hour for a standard basin mixer."),
        ("tank", "Water tank cleaning is recommended twice each calendar year."),
        ("pipe", "Frozen pipe thawing uses warm cloths, never an open flame."),
    ]
    documents = []
    for i, (slug, sentence) in enumerate(topics):
        text = " ".join(
            [
                sentence,
                f"The {slug} service is available across Lahore and Kasur.",
                f"A {slug} job is quoted before any work begins.",
                f"Most {slug} appointments are completed in a single visit.",
            ]
        )
        documents.append(
            Document(
                doc_id=f"doc{i}",
                url=f"https://acme.example/{slug}",
                title=slug.title(),
                text=text,
                sections=(Section(text=text, heading_path=(slug.title(),), start_char=0),),
            )
        )
    return documents


@pytest.fixture
def corpus_queries(corpus) -> QuerySet:
    """One query per document, with the gold span anchored on the source text."""
    queries = []
    for document in corpus:
        gold = document.text.split(". ")[0] + "."
        start = document.text.index(gold)
        queries.append(
            Query(
                query_id=f"q_{document.doc_id}",
                text=f"how does the {document.title.lower()} service work",
                spans=(
                    SpanRelevance(
                        doc_id=document.doc_id,
                        start_char=start,
                        end_char=start + len(gold),
                        text=gold,
                    ),
                ),
                source_doc_id=document.doc_id,
            )
        )
    return QuerySet(name="fixture", queries=queries)
