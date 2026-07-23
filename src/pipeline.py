"""Build the Medical RAG pipeline (retriever + reranker + chain) from config."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.config import load_config, get_project_root
from src.retrieval import get_retriever, HybridRetriever, Reranker
from src.generation import MedicalRAGChain
from src.generation.chain import _get_llm


def build_pipeline(config_path: Optional[str] = None):
    """
    Load config, build vector store retriever, hybrid retriever, optional reranker,
    and Medical RAG chain. Returns (chain, config).
    """
    config = load_config(config_path)
    root = get_project_root()
    vs_cfg = config.get("vector_store", {})
    ret_cfg = config.get("retrieval", {})
    llm_cfg = config.get("llm", {})
    emb_cfg = config.get("embeddings", {})

    persist_dir = root / vs_cfg.get("persist_directory", "data/chroma_db")
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. Run: python scripts/run_ingest.py"
        )

    dense = get_retriever(
        persist_directory=persist_dir,
        collection_name=vs_cfg.get("collection_name", "medical_evidence"),
        embedding_model_name=emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_provider=emb_cfg.get("provider", "onnx"),
    )
    hybrid = HybridRetriever(
        dense_retriever=dense,
        documents=dense.documents,
        fusion_k=ret_cfg.get("rrf_k", 60),
    )
    reranker = None
    if ret_cfg.get("rerank"):
        reranker = Reranker(model_name=ret_cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    llm = None
    try:
        llm = _get_llm(
            provider=llm_cfg.get("provider", "ollama"),
            model_name=llm_cfg.get("model_name", "qwen2.5-coder:3b"),
            temperature=llm_cfg.get("temperature", 0.1),
            max_tokens=llm_cfg.get("max_tokens", 1024),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        )
    except Exception as e:
        print(f"LLM not available: {e}. Answers will show a placeholder.")

    chain = MedicalRAGChain(
        retriever=hybrid,
        documents=hybrid.documents,
        reranker=reranker,
        llm=llm,
        config=config,
    )
    return chain, config
