"""Chunking strategies for medical documents (guidelines, monographs, textbooks)."""
from __future__ import annotations

from typing import List

from .documents import Document


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: List[str] | None = None,
) -> List[str]:
    """
    Split text into overlapping chunks, preferring paragraph/sentence boundaries.
    Uses recursive separator splitting for medical text.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks: List[str] = []

    def _split(s: str, sep_index: int) -> List[str]:
        if sep_index >= len(separators):
            # Character-level fallback
            return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size - chunk_overlap)]
        sep = separators[sep_index]
        parts = s.split(sep)
        result: List[str] = []
        current = ""
        for i, part in enumerate(parts):
            candidate = (current + (sep if current else "") + part).strip()
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > chunk_size:
                    result.extend(_split(part, sep_index + 1))
                    current = ""
                else:
                    current = part
        if current:
            result.append(current)
        return result

    raw_chunks = _split(text, 0)

    # Apply overlap by sliding window where beneficial
    for c in raw_chunks:
        if len(c) <= chunk_size:
            chunks.append(c)
        else:
            start = 0
            while start < len(c):
                end = start + chunk_size
                chunks.append(c[start:end])
                start = end - chunk_overlap
                if start >= len(c):
                    break

    return [c.strip() for c in chunks if c.strip()]


def medical_chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: List[str] | None = None,
) -> List[Document]:
    """
    Chunk a list of documents into smaller pieces while preserving metadata
    for citation (source, source_id, doc_type).
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]
    out: List[Document] = []
    for doc in documents:
        if not doc.content or not doc.content.strip():
            continue
        texts = chunk_text(
            doc.content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        for i, text in enumerate(texts):
            out.append(
                Document(
                    content=text,
                    source=doc.source,
                    source_id=doc.source_id,
                    doc_type=doc.doc_type,
                    metadata={**doc.metadata, "chunk_index": i, "total_chunks": len(texts)},
                )
            )
    return out
