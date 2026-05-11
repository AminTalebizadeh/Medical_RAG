"""Rerank retrieved passages with a cross-encoder for better precision."""
from __future__ import annotations

from typing import List, Tuple

from src.ingest.documents import Document


class Reranker:
    """Rerank (query, document) pairs using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for reranking. pip install sentence-transformers"
                ) from e
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        doc_score_pairs: List[Tuple[Document, float]],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank pairs (Document, previous_score) by relevance to query.
        Returns top_k (Document, rerank_score).
        """
        if not doc_score_pairs:
            return []
        model = self._get_model()
        pairs = [(query, d.content) for d, _ in doc_score_pairs]
        scores = model.predict(pairs)
        scored = [(doc_score_pairs[i][0], float(scores[i])) for i in range(len(scores))]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
