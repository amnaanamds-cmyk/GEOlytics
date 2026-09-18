"""Retrieval strategies -- experimental factor #2."""

from geolytics.retrieval.base import Retriever
from geolytics.retrieval.bm25 import BM25Retriever
from geolytics.retrieval.dense import DenseRetriever
from geolytics.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion, weighted_fusion
from geolytics.retrieval.rerank import RerankingRetriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "RerankingRetriever",
    "Retriever",
    "reciprocal_rank_fusion",
    "weighted_fusion",
]
