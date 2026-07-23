"""Load medical documents from guidelines, drug monographs, and custom dirs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .documents import Document


def load_guidelines(dir_path: str | Path) -> List[Document]:
    """
    Load guideline documents from a directory.
    Expects JSON files with keys: title, source, content (and optional id).
    Or .txt files with optional companion .meta.json for source.
    """
    path = Path(dir_path)
    if not path.is_dir():
        return []

    docs: List[Document] = []
    for f in path.iterdir():
        if f.suffix.lower() == ".json":
            try:
                with open(f, encoding="utf-8") as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    for item in data:
                        docs.append(_guideline_item_to_doc(item, f.stem))
                else:
                    docs.append(_guideline_item_to_doc(data, f.stem))
            except (json.JSONDecodeError, KeyError):
                continue  # skip malformed
        elif f.suffix.lower() == ".txt":
            try:
                content = f.read_text(encoding="utf-8")
                meta_path = f.with_suffix(".meta.json")
                source = "Guidelines"
                source_id = f.stem
                if meta_path.exists():
                    with open(meta_path, encoding="utf-8") as fp:
                        meta = json.load(fp)
                    source = meta.get("source", source)
                    source_id = meta.get("id", source_id)
                docs.append(
                    Document(
                        content=content,
                        source=source,
                        source_id=source_id,
                        doc_type="guidelines",
                    )
                )
            except Exception:
                continue

    return docs


def _guideline_item_to_doc(item: dict, default_id: str) -> Document:
    content = item.get("content", item.get("text", ""))
    if isinstance(content, list):
        content = "\n\n".join(str(x) for x in content)
    return Document(
        content=str(content),
        source=item.get("source", "Guidelines"),
        source_id=item.get("id", item.get("title", default_id)),
        doc_type="guidelines",
        metadata={k: v for k, v in item.items() if k not in ("content", "text", "source", "id", "title")},
    )


def load_drug_monographs(dir_path: str | Path) -> List[Document]:
    """
    Load drug monograph documents (DrugBank-style).
    Expects JSON with drug name, description, indications, interactions, etc.
    """
    path = Path(dir_path)
    if not path.is_dir():
        return []

    docs: List[Document] = []
    for f in path.iterdir():
        if f.suffix.lower() != ".json":
            continue
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
        except json.JSONDecodeError:
            continue

        if isinstance(data, list):
            for item in data:
                docs.append(_drug_item_to_doc(item))
        else:
            docs.append(_drug_item_to_doc(data))

    return docs


def _drug_item_to_doc(item: dict) -> Document:
    name = item.get("name", item.get("drug_name", "Unknown"))
    parts = []
    if item.get("description"):
        parts.append(f"Description: {item['description']}")
    if item.get("indications"):
        parts.append(f"Indications: {item['indications']}")
    if item.get("mechanism"):
        parts.append(f"Mechanism: {item['mechanism']}")
    if item.get("contraindications"):
        parts.append(f"Contraindications: {item['contraindications']}")
    if item.get("interactions"):
        parts.append(f"Interactions: {item['interactions']}")
    if item.get("adverse_effects"):
        parts.append(f"Adverse effects: {item['adverse_effects']}")
    if item.get("dosing"):
        parts.append(f"Dosing: {item['dosing']}")
    # Include any extra text fields
    for key in ("pharmacokinetics", "warnings", "pregnancy", "lactation"):
        if item.get(key):
            parts.append(f"{key.replace('_', ' ').title()}: {item[key]}")

    content = "\n\n".join(parts) if parts else str(item)
    return Document(
        content=content,
        source="DrugBank",
        source_id=name,
        doc_type="drug_monograph",
        metadata={"drug_name": name},
    )


def load_documents_from_dir(
    dir_path: str | Path,
    doc_type: str = "custom",
    source_label: str = "Custom",
) -> List[Document]:
    """Load .txt and .md files from a directory for RAG."""
    path = Path(dir_path)
    if not path.is_dir():
        return []

    docs: List[Document] = []
    for f in path.iterdir():
        if f.suffix.lower() in (".txt", ".md"):
            try:
                content = f.read_text(encoding="utf-8")
                docs.append(
                    Document(
                        content=content,
                        source=source_label,
                        source_id=f.stem,
                        doc_type=doc_type,
                    )
                )
            except Exception:
                continue
    return docs
