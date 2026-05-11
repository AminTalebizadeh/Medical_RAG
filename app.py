"""
Medical RAG — Streamlit app for evidence-based QA over guidelines and drug monographs.
Run: streamlit run app.py (from project root)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.pipeline import build_pipeline


def main():
    st.set_page_config(
        page_title="Medical RAG — Evidence Retrieval",
        page_icon="📋",
        layout="centered",
    )
    st.title("📋 Medical RAG — Evidence Retrieval")
    st.caption(
        "QA over clinical guidelines, drug monographs, and medical references. "
        "Answers are grounded in retrieved evidence with citations."
    )
    st.markdown("---")

    @st.cache_resource
    def load_chain():
        try:
            chain, config = build_pipeline()
            return chain, None
        except FileNotFoundError as e:
            return None, str(e)

    chain, err = load_chain()
    if err:
        st.error(f"Pipeline not ready: {err}")
        st.info("Run ingestion first: `python scripts/run_ingest.py`")
        return

    disclaimer = (
        "**Disclaimer:** This tool is for informational and evidence-retrieval support only. "
        "It does not replace professional medical advice, diagnosis, or treatment."
    )
    st.warning(disclaimer)

    query = st.text_input(
        "Ask a medical evidence question",
        placeholder="e.g. When is influenza vaccination recommended? What are metformin contraindications?",
        key="query",
    )

    if query:
        with st.spinner("Searching evidence and generating answer..."):
            response = chain.run(query)

        st.subheader("Answer")
        st.markdown(response.answer)

        if response.sources:
            st.subheader("Evidence sources")
            for s in response.sources:
                with st.expander(f"{s['label']} — {s['source']}: {s['source_id']} (score: {s.get('relevance_score', 'N/A')})"):
                    st.caption(f"Type: {s.get('doc_type', 'N/A')}")
            st.subheader("Context used (excerpts)")
            st.text_area("Retrieved passages", value=response.context_used[:8000] or "(none)", height=200, disabled=True)


if __name__ == "__main__":
    main()
