"""Medical RAG chain: retrieve evidence, then generate answer with citations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.ingest.documents import Document
from .prompts import MEDICAL_QA_SYSTEM_PROMPT, USER_QA_TEMPLATE, format_context_and_sources


@dataclass
class RAGResponse:
    """Structured response from the Medical RAG pipeline."""

    answer: str
    sources: List[dict] = field(default_factory=list)
    context_used: str = ""
    query: str = ""


def _get_llm(provider: str, model_name: str, temperature: float = 0.1, max_tokens: int = 1024):
    """Return an LLM instance based on config (OpenAI or Ollama)."""
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ImportError:
            try:
                from langchain.llms import OpenAI
                return OpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
            except ImportError as e:
                raise ImportError(
                    "OpenAI provider requires langchain-openai or langchain. "
                    "Install: pip install langchain-openai openai"
                ) from e
    if provider == "ollama":
        try:
            from langchain_community.llms import Ollama
            return Ollama(model=model_name, temperature=temperature, num_predict=max_tokens)
        except ImportError as e:
            raise ImportError(
                "Ollama provider requires langchain-community. pip install langchain-community"
            ) from e
    raise ValueError(f"Unsupported LLM provider: {provider}")


class MedicalRAGChain:
    """
    End-to-end Medical RAG: hybrid retrieval + optional rerank + LLM generation
    with citations and evidence cards.
    """

    def __init__(
        self,
        retriever,  # HybridRetriever or any with .retrieve() or .query()
        documents: List[Document],
        reranker=None,
        llm=None,
        config: Optional[dict] = None,
    ):
        self.retriever = retriever
        self.documents = documents
        self.reranker = reranker
        self.llm = llm
        self.config = config or {}
        self._retrieval_top_k = self.config.get("retrieval", {}).get("top_k_after_fusion", 10)
        self._rerank_top_k = self.config.get("retrieval", {}).get("top_k_final", 5)
        self._use_rerank = self.config.get("retrieval", {}).get("rerank", True) and reranker is not None

    def retrieve(self, query: str) -> List[tuple[Document, float]]:
        """Run retrieval (hybrid + optional rerank)."""
        if hasattr(self.retriever, "retrieve"):
            pairs = self.retriever.retrieve(
                query,
                top_k_after_fusion=self._retrieval_top_k,
            )
        else:
            pairs = self.retriever.query(query, top_k=self._retrieval_top_k)
        if self._use_rerank and self.reranker and pairs:
            pairs = self.reranker.rerank(query, pairs, top_k=self._rerank_top_k)
        elif pairs and self._rerank_top_k < len(pairs):
            pairs = pairs[: self._rerank_top_k]
        return pairs

    def generate(self, query: str, context: str, sources: List[dict]) -> str:
        """Generate answer from context using LLM."""
        if not self.llm:
            return (
                "[No LLM configured. Set OPENAI_API_KEY and install langchain-openai, "
                "or configure Ollama.]"
            )
        user_prompt = USER_QA_TEMPLATE.format(context=context, question=query)
        try:
            if hasattr(self.llm, "invoke"):
                # Prefer message format for chat models (OpenAI, etc.)
                try:
                    from langchain_core.messages import SystemMessage, HumanMessage
                    msgs = [SystemMessage(content=MEDICAL_QA_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
                    out = self.llm.invoke(msgs)
                except ImportError:
                    out = self.llm.invoke(MEDICAL_QA_SYSTEM_PROMPT + "\n\n" + user_prompt)
                if hasattr(out, "content"):
                    return out.content
                return str(out)
            return str(self.llm(user_prompt))
        except Exception as e:
            return f"[LLM error: {e}. Check API key and model name.]"

    def run(self, query: str) -> RAGResponse:
        """Run full pipeline: retrieve -> format context -> generate -> return with sources."""
        doc_score_pairs = self.retrieve(query)
        context, sources = format_context_and_sources(doc_score_pairs)
        if not context:
            return RAGResponse(
                answer="No relevant evidence was retrieved for this query. Try rephrasing or expanding your question.",
                sources=[],
                context_used="",
                query=query,
            )
        answer = self.generate(query, context, sources)
        return RAGResponse(
            answer=answer,
            sources=sources,
            context_used=context,
            query=query,
        )
