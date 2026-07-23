"""Embedding models for medical document retrieval."""
from __future__ import annotations

import os
from typing import List

from src.ingest.documents import Document

# Avoid OpenMP duplicate-runtime crashes common with Anaconda + PyTorch on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class OnnxEmbeddingModel:
    """
    Thin wrapper around Chroma's ONNX MiniLM so callers can use .encode()
    without loading PyTorch (which can hard-crash on some Windows CUDA setups).
    """

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        vectors: List[List[float]] = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._fn(batch))
            if show_progress_bar:
                done = min(start + batch_size, total)
                print(f"Embedded {done}/{total}", end="\r", flush=True)
        if show_progress_bar and total:
            print()
        arr = np.asarray(vectors, dtype="float32")
        return arr if convert_to_numpy else arr.tolist()


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    provider: str | None = None,
):
    """
    Return an embedding model with .encode().

    Prefers Chroma ONNX MiniLM (same family as all-MiniLM-L6-v2) to avoid
    PyTorch native crashes. Falls back to sentence-transformers if needed.
    """
    backend = (
        provider
        or os.getenv("MEDICAL_RAG_EMBEDDINGS_BACKEND")
        or "onnx"
    ).lower()
    use_onnx = backend in ("onnx", "chroma", "default")
    model_l = (model_name or "").lower()
    if use_onnx and ("minilm" in model_l or not model_name):
        try:
            return OnnxEmbeddingModel()
        except Exception as e:
            print(f"ONNX embeddings unavailable ({e}); falling back to sentence-transformers.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from e
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        device = "cpu"
    return SentenceTransformer(model_name, device=device)


def embed_texts(
    model,
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = False,
) -> List[List[float]]:
    """Embed a list of text strings; returns list of vectors."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    ).tolist()


def embed_documents(
    model,
    documents: List[Document],
    batch_size: int = 32,
    show_progress: bool = False,
) -> List[List[float]]:
    """Embed document contents; returns list of vectors in same order as documents."""
    texts = [d.content for d in documents]
    return embed_texts(model, texts, batch_size=batch_size, show_progress=show_progress)
