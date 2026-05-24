"""Document and chunk types for Medical RAG."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A single document or chunk with metadata for retrieval and citation."""

    content: str
    source: str = ""           
    source_id: str = ""        
    doc_type: str = "general" # guidelines | drug_monograph | textbook | custom
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "source_id": self.source_id,
            "doc_type": self.doc_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        return cls(
            content=d.get("content", ""),
            source=d.get("source", ""),
            source_id=d.get("source_id", ""),
            doc_type=d.get("doc_type", "general"),
            metadata=d.get("metadata", {}),
        )
