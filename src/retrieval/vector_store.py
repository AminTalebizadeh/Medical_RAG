"""Vector store (Chroma) and dense retriever for Medical RAG."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.ingest.documents import Document
from .embeddings import get_embedding_model, embed_documents


def build_vector_store(
    documents: List[Document],
    persist_directory: str | Path,
    collection_name: str = "medical_evidence",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> "ChromaVectorStore":
    """
    Build a Chroma vector store from a list of documents.
    Persists to disk for reuse.
    """
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
    except ImportError as e:
        raise ImportError("chromadb is required. Install with: pip install chromadb") from e

    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(persist_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Medical evidence (guidelines, drug monographs)"},
    )

    model = get_embedding_model(embedding_model_name)
    embeddings = embed_documents(model, documents, batch_size=batch_size, show_progress=True)

    ids = [f"doc_{i}" for i in range(len(documents))]
    metadatas = [
        {
            "source": d.source,
            "source_id": d.source_id,
            "doc_type": d.doc_type,
            "content": d.content[:4000],  # Chroma metadata size limit; full text in documents.json
        }
        for d in documents
    ]
    # Store full content in a separate list for retrieval (we return Document objects)
    contents = [d.content for d in documents]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Persist document list for hybrid BM25 when loading later
    docs_path = Path(persist_directory) / "documents.json"
    import json
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump([d.to_dict() for d in documents], f, ensure_ascii=False, indent=0)

    return ChromaVectorStore(
        client=client,
        collection_name=collection_name,
        embedding_model=model,
        contents=contents,
        documents=documents,
    )


class ChromaVectorStore:
    """Wrapper around Chroma that returns Document objects with full content."""

    def __init__(
        self,
        client,
        collection_name: str,
        embedding_model,
        contents: List[str],
        documents: List[Document],
    ):
        self._client = client
        self._collection_name = collection_name
        self._model = embedding_model
        self._contents = contents
        self._documents = documents
        self._id_to_doc = {f"doc_{i}": documents[i] for i in range(len(documents))}

    def query(
        self,
        query_text: str,
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        """Return top_k most similar documents with scores (distance; lower is more similar for L2)."""
        collection = self._client.get_collection(self._collection_name)
        q_embedding = self._model.encode([query_text], convert_to_numpy=True).tolist()
        results = collection.query(
            query_embeddings=q_embedding,
            n_results=min(top_k, len(self._documents)),
            where=where,
            include=["metadatas", "distances"],
        )
        out: List[tuple[Document, float]] = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for i, id_ in enumerate(results["ids"][0]):
            meta = (results["metadatas"][0] or [])[i] or {}
            dist = (results["distances"][0] or [0])[i]
            doc = self._id_to_doc.get(id_)
            if doc is None:
                doc = Document(
                    content=meta.get("content", ""),
                    source=meta.get("source", ""),
                    source_id=meta.get("source_id", ""),
                    doc_type=meta.get("doc_type", "general"),
                )
            # Convert L2 distance to a simple similarity score (higher = better); optional
            score = 1.0 / (1.0 + float(dist))
            out.append((doc, score))
        return out


def get_retriever(
    persist_directory: str | Path,
    collection_name: str = "medical_evidence",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> ChromaVectorStore:
    """Load existing Chroma store and return it as a retriever (no documents list needed for query)."""
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
    except ImportError as e:
        raise ImportError("chromadb is required. Install with: pip install chromadb") from e

    path = Path(persist_directory)
    if not path.exists():
        raise FileNotFoundError(f"Vector store not found at {path}. Run ingestion first.")

    client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
    collection = client.get_collection(collection_name)
    model = get_embedding_model(embedding_model_name)

    # Rebuild documents from persisted list if available (for hybrid BM25); else from Chroma metadatas
    all_docs = []
    docs_path = path / "documents.json"
    if docs_path.exists():
        import json
        with open(docs_path, encoding="utf-8") as f:
            doc_dicts = json.load(f)
        all_docs = [Document.from_dict(d) for d in doc_dicts]
    else:
        data = collection.get(include=["metadatas"])
        for i, id_ in enumerate(data["ids"]):
            meta = (data["metadatas"] or [None])[i] or {}
            all_docs.append(
                Document(
                    content=meta.get("content", ""),
                    source=meta.get("source", ""),
                    source_id=meta.get("source_id", ""),
                    doc_type=meta.get("doc_type", "general"),
                )
            )
            
    contents_list = [d.content for d in all_docs]

    return ChromaVectorStore(
        client=client,
        collection_name=collection_name,
        embedding_model=model,
        contents=contents_list,
        documents=all_docs,
    )
