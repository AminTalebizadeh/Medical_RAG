"""Hybrid retrieval: dense (vector) + sparse (BM25) with reciprocal rank fusion."""
from __future__ import annotations

from typing import List, Tuple

from src.ingest.documents import Document


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).
    Score = sum(1 / (k + rank_i)) over all lists where the doc appears.
    """
    doc_to_score: dict[str, float] = {}
    doc_to_doc: dict[str, Document] = {}

    for rlist in ranked_lists:
        for rank, (doc, _) in enumerate(rlist, start=1):
            # Use content hash or (source, source_id, content slice) as key to dedupe
            key = f"{doc.source}|{doc.source_id}|{doc.content[:80]}"
            doc_to_doc[key] = doc
            doc_to_score[key] = doc_to_score.get(key, 0.0) + 1.0 / (k + rank)

    sorted_keys = sorted(doc_to_score.keys(), key=lambda x: -doc_to_score[x])
    return [(doc_to_doc[k], doc_to_score[k]) for k in sorted_keys]


class HybridRetriever:
    """
    Combines dense (vector) retrieval and BM25 (sparse) retrieval,
    then fuses results with RRF.
    """

    def __init__(
        self,
        dense_retriever,  # ChromaVectorStore with .query()
        documents: List[Document],
        fusion_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.documents = documents
        self.fusion_k = fusion_k
        self._bm25_index = None
        self._bm25_corpus = None

    def _ensure_bm25(self):
        if self._bm25_index is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("rank_bm25 is required for hybrid search. pip install rank_bm25") from e
        tokenized = [d.content.lower().split() for d in self.documents]
        self._bm25_corpus = tokenized
        self._bm25_index = BM25Okapi(tokenized)

    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """Return top_k documents from BM25 with scores."""
        self._ensure_bm25()
        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)
        # Get top_k indices
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.documents[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def retrieve(
        self,
        query: str,
        top_k_dense: int = 20,
        top_k_bm25: int = 20,
        top_k_after_fusion: int = 10,
    ) -> List[Tuple[Document, float]]:
        """
        Run hybrid retrieval: dense + BM25, then RRF merge.
        Returns list of (Document, fused_score) sorted by score descending.
        """
        dense_results = self.dense_retriever.query(query, top_k=top_k_dense)
        bm25_results = self.bm25_search(query, top_k=top_k_bm25)
        fused = _reciprocal_rank_fusion([dense_results, bm25_results], k=self.fusion_k)
        return fused[:top_k_after_fusion]
