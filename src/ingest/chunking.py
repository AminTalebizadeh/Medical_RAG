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
    
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]
    if not text or not text.strip():
        return []
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be >= 0 and smaller than chunk_size ({chunk_size})"
        )

    text = text.strip()

    def _split(s: str, sep_index: int) -> List[str]:
        if sep_index >= len(separators):
            # Character-level fallback when no separator works
            step = chunk_size - chunk_overlap
            return [s[i : i + chunk_size] for i in range(0, len(s), step)]
        sep = separators[sep_index]
        parts = s.split(sep)
        result: List[str] = []
        current = ""
        for part in parts:  # FIX 1 (was: for part in enumerate(parts))
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

    raw_chunks = [c.strip() for c in _split(text, 0) if c.strip()]

   
    if chunk_overlap > 0 and len(raw_chunks) > 1:
        chunks: List[str] = [raw_chunks[0]]
        for prev, cur in zip(raw_chunks, raw_chunks[1:]):
            tail = prev[-chunk_overlap:]

            if len(prev) > chunk_overlap and " " in tail:
                tail = tail.split(" ", 1)[1]
            chunks.append(f"{tail} {cur}".strip() if tail else cur)
    else:
        chunks = raw_chunks

    return [c.strip() for c in chunks if c.strip()]


def medical_chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: List[str] | None = None,
) -> List[Document]:
    
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