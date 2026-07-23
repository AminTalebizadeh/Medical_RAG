"""Rerank retrieved passages with a cross-encoder for better precision."""
from __future__ import annotations

import os
from typing import List, Tuple

from src.ingest.documents import Document

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class Reranker:
    """Rerank (query, document) pairs using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._disabled = False

    def _get_model(self):
        if self._disabled:
            return None
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for reranking. pip install sentence-transformers"
                ) from e
            # Cross-encoder uses torch; prefer CPU to avoid broken CUDA drivers.
            self._model = CrossEncoder(self.model_name, device="cpu")
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
        try:
            model = self._get_model()
            if model is None:
                return doc_score_pairs[:top_k]
            pairs = [(query, d.content) for d, _ in doc_score_pairs]
            scores = model.predict(pairs)
            scored = [(doc_score_pairs[i][0], float(scores[i])) for i in range(len(scores))]
            scored.sort(key=lambda x: -x[1])
            return scored[:top_k]
        except Exception as e:
            # Native torch crashes are hard to catch; soft failures fall back to fusion ranking.
            print(f"Reranker failed ({e}); using fused ranking.")
            self._disabled = True
            return doc_score_pairs[:top_k]
