"""
Ingest medical documents (guidelines, drug monographs, custom docs) into the vector store.
Run from project root: python scripts/run_ingest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, get_project_root
from src.ingest import (
    load_guidelines,
    load_drug_monographs,
    load_documents_from_dir,
    medical_chunk_documents,
)
from src.retrieval import build_vector_store


def main():
    root = get_project_root()
    config = load_config()
    chunk_cfg = config.get("chunking", {})
    vs_cfg = config.get("vector_store", {})
    emb_cfg = config.get("embeddings", {})

    documents = []

    # Guidelines
    guidelines_dir = root / config.get("data", {}).get("guidelines_dir", "data/guidelines")
    if guidelines_dir.exists():
        docs = load_guidelines(guidelines_dir)
        documents.extend(docs)
        print(f"Loaded {len(docs)} guideline documents from {guidelines_dir}")

    # Drug monographs
    drugs_dir = root / config.get("data", {}).get("drug_monographs_dir", "data/drug_monographs")
    if drugs_dir.exists():
        docs = load_drug_monographs(drugs_dir)
        documents.extend(docs)
        print(f"Loaded {len(docs)} drug monograph entries from {drugs_dir}")

    # Custom docs (optional)
    custom_dir = root / config.get("data", {}).get("custom_docs_dir", "data/custom_docs")
    if custom_dir.exists():
        docs = load_documents_from_dir(custom_dir, doc_type="custom", source_label="Custom")
        documents.extend(docs)
        print(f"Loaded {len(docs)} custom documents from {custom_dir}")

    if not documents:
        print("No documents found. Add JSON/txt files under data/guidelines and data/drug_monographs.")
        return 1

    # Chunk
    chunked = medical_chunk_documents(
        documents,
        chunk_size=chunk_cfg.get("chunk_size", 512),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 64),
        separators=chunk_cfg.get("separators"),
    )
    print(f"Chunked into {len(chunked)} segments")

    # Build and persist vector store
    persist_dir = root / vs_cfg.get("persist_directory", "data/chroma_db")
    persist_dir.mkdir(parents=True, exist_ok=True)
    build_vector_store(
        chunked,
        persist_directory=persist_dir,
        collection_name=vs_cfg.get("collection_name", "medical_evidence"),
        embedding_model_name=emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_provider=emb_cfg.get("provider", "onnx"),
    )
    print(f"Vector store saved to {persist_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
