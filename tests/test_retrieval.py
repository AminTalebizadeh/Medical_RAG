"""Tests for retrieval: hybrid RRF, reranker (optional)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingest.documents import Document
from src.retrieval.hybrid import HybridRetriever, _reciprocal_rank_fusion
from src.retrieval.reranker import Reranker


def test_rrf_merge():
    d1 = Document("a", source="S1", source_id="1")
    d2 = Document("b", source="S2", source_id="2")
    d3 = Document("c", source="S3", source_id="3")
    list_a = [(d1, 0.9), (d2, 0.8)]
    list_b = [(d2, 0.95), (d1, 0.7), (d3, 0.6)]
    fused = _reciprocal_rank_fusion([list_a, list_b], k=60)
    # All three docs appear; d2 and d1 in both lists get higher fused scores than d3 (only in list_b)
    assert len(fused) == 3
    source_ids = {f[0].source_id for f in fused}
    assert source_ids == {"1", "2", "3"}
    # Top two should be d1 and d2 (both appear in two lists)
    top_scores = [f[1] for f in fused]
    assert top_scores[0] >= top_scores[1] >= top_scores[2]


def test_hybrid_bm25_search():
    docs = [
        Document("influenza vaccination recommended for elderly", source="CDC", source_id="flu"),
        Document("metformin for type 2 diabetes", source="DrugBank", source_id="metformin"),
    ]
    # Dense retriever mock: returns same order for any query
    class MockDense:
        def query(self, q, top_k=20, where=None):
            return [(docs[0], 0.9), (docs[1], 0.8)]
    hybrid = HybridRetriever(MockDense(), documents=docs, fusion_k=60)
    results = hybrid.retrieve("influenza vaccine", top_k_dense=5, top_k_bm25=5, top_k_after_fusion=5)
    assert len(results) <= 5
    assert all(isinstance(r[0], Document) for r in results)


def _torch_cuda_unsafe() -> bool:
    """True when CUDA torch is installed but the driver is broken (can hard-crash)."""
    try:
        import torch
        # cudaGetDeviceCount returning an error on Windows often precedes ACCESS_VIOLATION.
        if not hasattr(torch, "cuda"):
            return False
        try:
            torch.cuda.device_count()
            return not torch.cuda.is_available() and "cuda" in torch.__version__.lower()
        except Exception:
            return True
    except Exception:
        return True


@pytest.mark.slow
def test_reranker_top_k():
    """Loads cross-encoder model; run with pytest -m 'not slow' to skip."""
    if _torch_cuda_unsafe():
        pytest.skip("Skipping reranker: broken/unsupported CUDA torch would hard-crash.")
    docs = [
        Document("content A", source="S1", source_id="1"),
        Document("content B", source="S2", source_id="2"),
    ]
    pairs = [(docs[0], 0.9), (docs[1], 0.8)]
    reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    out = reranker.rerank("query about A", pairs, top_k=1)
    assert len(out) == 1
    assert isinstance(out[0][0], Document) and isinstance(out[0][1], float)
