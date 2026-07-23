"""Load and validate configuration for Medical RAG."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config; merge with env overrides."""
    if config_path is None:
        base = Path(__file__).resolve().parent.parent
        config_path = base / "config" / "settings.yaml"
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Env overrides (explicit MEDICAL_RAG_LLM_PROVIDER wins; do not force OpenAI when Ollama is set)
    llm_cfg = config.setdefault("llm", {})
    if os.getenv("MEDICAL_RAG_LLM_PROVIDER"):
        llm_cfg["provider"] = os.environ["MEDICAL_RAG_LLM_PROVIDER"]
    if os.getenv("MEDICAL_RAG_EMBEDDING_MODEL"):
        config.setdefault("embeddings", {})["model_name"] = os.environ["MEDICAL_RAG_EMBEDDING_MODEL"]
    if os.getenv("MEDICAL_RAG_LLM_MODEL"):
        llm_cfg["model_name"] = os.environ["MEDICAL_RAG_LLM_MODEL"]
    if os.getenv("OLLAMA_BASE_URL"):
        llm_cfg["base_url"] = os.environ["OLLAMA_BASE_URL"]
    # Only default to OpenAI when no provider was configured and a key is present
    if not llm_cfg.get("provider") and os.getenv("OPENAI_API_KEY"):
        llm_cfg["provider"] = "openai"

    return config


def get_project_root() -> Path:
    """Return project root (parent of src)."""
    return Path(__file__).resolve().parent.parent
