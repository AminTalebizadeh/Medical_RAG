"""Embedding models for medical document retrieval."""
from __future__ import annotations

from typing import List

from src.ingest.documents import Document


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return a callable that embeds text(s). Uses sentence-transformers by default."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from e
    model = SentenceTransformer(model_name)
    return model


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
