"""Tests for ingestion: documents, chunking, loaders."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# Add project root
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingest.documents import Document
from src.ingest.chunking import chunk_text, medical_chunk_documents
from src.ingest.loaders import load_guidelines, load_drug_monographs, load_documents_from_dir


def test_document_to_from_dict():
    d = Document("Hello", source="CDC", source_id="x", doc_type="guidelines")
    d2 = Document.from_dict(d.to_dict())
    assert d2.content == d.content and d2.source == d.source and d2.source_id == d.source_id


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_respects_size():
    text = "A. " * 200
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c) <= 100 + 20


def test_medical_chunk_preserves_metadata():
    docs = [Document("First sentence. Second sentence.", source="WHO", source_id="id1", doc_type="guidelines")]
    out = medical_chunk_documents(docs, chunk_size=50, chunk_overlap=5)
    assert len(out) >= 1
    assert all(d.source == "WHO" and d.source_id == "id1" and d.doc_type == "guidelines" for d in out)


def test_load_guidelines_json():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        with open(path / "g.json", "w", encoding="utf-8") as f:
            json.dump([{"title": "T", "source": "CDC", "content": "Get a flu shot."}], f)
        docs = load_guidelines(path)
    assert len(docs) == 1
    assert docs[0].content == "Get a flu shot."
    assert docs[0].source == "CDC"


def test_load_drug_monographs_json():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        with open(path / "d.json", "w", encoding="utf-8") as f:
            json.dump([{"name": "DrugX", "description": "A drug.", "indications": "Pain."}], f)
        docs = load_drug_monographs(path)
    assert len(docs) == 1
    assert docs[0].source_id == "DrugX"
    assert docs[0].doc_type == "drug_monograph"
    assert "A drug" in docs[0].content and "Pain" in docs[0].content


def test_load_documents_from_dir():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        (path / "a.txt").write_text("Hello world", encoding="utf-8")
        docs = load_documents_from_dir(path, doc_type="custom", source_label="Custom")
    assert len(docs) == 1
    assert docs[0].content == "Hello world"
    assert docs[0].source == "Custom"
