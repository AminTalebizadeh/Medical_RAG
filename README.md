# Medical RAG — Evidence Retrieval

A **Retrieval-Augmented Generation (RAG)** system for medical evidence: QA over **clinical guidelines**, **drug monographs**, and **medical textbooks** with **citations**. Matches current "evidence retrieval" use cases in medicine (e.g. DrugBank, CDC/WHO-style content).

## Features

- **Multi-source corpus**: Guidelines (CDC, WHO), drug monographs (DrugBank-style), and custom documents
- **Hybrid retrieval**: Dense (sentence-transformers) + sparse (BM25) with **Reciprocal Rank Fusion (RRF)**
- **Reranking**: Cross-encoder reranker for better precision before generation
- **Citations**: Every answer references evidence with source labels (e.g. [Source 1], [Source 2])
- **Safety**: System prompt enforces evidence-only answers and a clear **disclaimer** (not for diagnosis/treatment)

## Project structure

```
Medical_RAG/
├── config/
│   └── settings.yaml       # Embeddings, chunking, retrieval, LLM
├── data/
│   ├── guidelines/         # CDC/WHO-style JSON or TXT
│   ├── drug_monographs/    # DrugBank-style JSON
│   ├── custom_docs/        # Optional PDF/TXT
│   └── chroma_db/          # Vector store (created by ingest)
├── src/
│   ├── config.py
│   ├── pipeline.py         # Build full RAG pipeline
│   ├── ingest/             # Loaders, chunking, Document type
│   ├── retrieval/          # Embeddings, Chroma, hybrid, reranker
│   └── generation/         # Prompts, RAG chain, citations
├── scripts/
│   ├── run_ingest.py       # Ingest docs into vector store
│   └── run_qa.py           # CLI QA
├── app.py                  # Streamlit UI
├── requirements.txt
└── README.md
```

## Setup

1. **Create environment and install dependencies**

   ```bash
   cd Medical_RAG
   pip install -r requirements.txt
   ```

2. **Optional: OpenAI for generation**

   Set `OPENAI_API_KEY` for GPT-based answers. Without it, the pipeline still runs retrieval and shows a placeholder for the LLM step.

   ```bash
   set OPENAI_API_KEY=sk-...
   ```

3. **Ingest sample data (included)**

   Sample guidelines (CDC flu, WHO hypertension) and drug monographs (amoxicillin, metformin, ibuprofen) are under `data/guidelines` and `data/drug_monographs`.

   ```bash
   python scripts/run_ingest.py
   ```

4. **Ask questions**

   - **CLI**
     ```bash
     python scripts/run_qa.py "When is influenza vaccination recommended?"
     python scripts/run_qa.py   # interactive loop
     ```
   - **Streamlit**
     ```bash
     streamlit run app.py
     ```

## Configuration

Edit `config/settings.yaml` to:

- Change **embedding model** (e.g. `BAAI/bge-small-en-v1.5` for stronger retrieval)
- Adjust **chunk size/overlap** and **top_k** for retrieval and reranking
- Switch **LLM** provider (`openai` or `ollama`) and model name
- Point **data paths** to your own guidelines/drug/custom docs

## Adding your own data

- **Guidelines**: Add JSON under `data/guidelines/` with keys e.g. `title`, `source`, `content` (or `text`), and optional `id`.
- **Drug monographs**: Add JSON under `data/drug_monographs/` with keys such as `name`, `description`, `indications`, `mechanism`, `contraindications`, `interactions`, `adverse_effects`, `dosing`.
- **Custom**: Put `.txt` or `.md` in `data/custom_docs/`.

Then run **ingest** again. To avoid duplicates, clear or replace `data/chroma_db` and re-run `run_ingest.py` if you change the corpus.

## Design choices

- **Hybrid search**: Combines semantic (dense) and lexical (BM25) retrieval for better recall on medical terms and guidelines.
- **RRF**: Keeps a single ranking without tuning fusion weights.
- **Reranker**: Improves precision of the final passages sent to the LLM.
- **Evidence-only + disclaimer**: Aligns with regulatory and clinical expectations for decision-support tools.

## Tests

From project root:

```bash
pytest tests/ -v
```

## License

Use for learning and portfolio. Not for clinical or regulatory use without appropriate validation and compliance.
