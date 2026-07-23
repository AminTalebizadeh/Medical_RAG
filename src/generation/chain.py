"""Medical RAG chain: retrieve evidence, then generate answer with citations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.ingest.documents import Document
from .prompts import MEDICAL_QA_SYSTEM_PROMPT, USER_QA_TEMPLATE, format_context_and_sources


@dataclass
class RAGResponse:
    """Structured response from the Medical RAG pipeline."""

    answer: str
    sources: List[dict] = field(default_factory=list)
    context_used: str = ""
    query: str = ""


def _get_llm(
    provider: str,
    model_name: str,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    base_url: str = "http://localhost:11434",
):
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
        # Prefer ChatOllama (message API); fall back to community Ollama LLM.
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens,
                base_url=base_url,
            )
        except ImportError:
            pass
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens,
                base_url=base_url,
            )
        except ImportError:
            pass
        try:
            from langchain_community.llms import Ollama
            return Ollama(
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens,
                base_url=base_url,
            )
        except ImportError as e:
            raise ImportError(
                "Ollama provider requires langchain-ollama or langchain-community. "
                "Install: pip install langchain-ollama langchain-community"
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
        ret_cfg = self.config.get("retrieval", {})
        self._top_k_dense = ret_cfg.get("top_k_dense", 20)
        self._top_k_bm25 = ret_cfg.get("top_k_bm25", 20)
        self._retrieval_top_k = ret_cfg.get("top_k_after_fusion", 10)
        self._rerank_top_k = ret_cfg.get("top_k_final", 5)
        self._use_rerank = ret_cfg.get("rerank", True) and reranker is not None

    def retrieve(self, query: str) -> List[tuple[Document, float]]:
        """Run retrieval (hybrid + optional rerank)."""
        if hasattr(self.retriever, "retrieve"):
            pairs = self.retriever.retrieve(
                query,
                top_k_dense=self._top_k_dense,
                top_k_bm25=self._top_k_bm25,
                top_k_after_fusion=self._retrieval_top_k,
            )
        else:
            pairs = self.retriever.query(query, top_k=self._retrieval_top_k)

        if self._use_rerank and self.reranker and pairs:
            
            try:
                pairs = self.reranker.rerank(query, pairs, top_k=self._rerank_top_k)
            except Exception as e:
                print(f"Reranker unavailable ({e}); falling back to fused ranking.")
                pairs = pairs[: self._rerank_top_k]
        elif pairs and self._rerank_top_k < len(pairs):
            pairs = pairs[: self._rerank_top_k]
        return pairs

    def generate(self, query: str, context: str) -> str:
        """Generate answer from context using the configured LLM."""
        if not self.llm:
            return (
                "[No LLM configured. Set llm.provider to ollama (local) or openai "
                "(with OPENAI_API_KEY), and install the matching packages.]"
            )
        user_prompt = USER_QA_TEMPLATE.format(context=context, question=query)
        combined = MEDICAL_QA_SYSTEM_PROMPT + "\n\n" + user_prompt
        try:
            if hasattr(self.llm, "invoke"):
                # Chat models accept messages; completion models expect a plain string.
                try:
                    from langchain_core.messages import SystemMessage, HumanMessage
                    msgs = [
                        SystemMessage(content=MEDICAL_QA_SYSTEM_PROMPT),
                        HumanMessage(content=user_prompt),
                    ]
                    out = self.llm.invoke(msgs)
                except Exception:
                    out = self.llm.invoke(combined)
                if hasattr(out, "content"):
                    return out.content
                return str(out)
            return str(self.llm(combined))
        except Exception as e:
            return f"[LLM error: {e}. Check Ollama is running and model name is correct.]"

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
        answer = self.generate(query, context)
        return RAGResponse(
            answer=answer,
            sources=sources,
            context_used=context,
            query=query,
        )