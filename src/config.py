"""Load and validate configuration for Medical RAG."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


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

    # Env overrides
    if os.getenv("OPENAI_API_KEY"):
        config.setdefault("llm", {})["provider"] = config.get("llm", {}).get("provider") or "openai"
    if os.getenv("MEDICAL_RAG_EMBEDDING_MODEL"):
        config.setdefault("embeddings", {})["model_name"] = os.environ["MEDICAL_RAG_EMBEDDING_MODEL"]
    if os.getenv("MEDICAL_RAG_LLM_MODEL"):
        config.setdefault("llm", {})["model_name"] = os.environ["MEDICAL_RAG_LLM_MODEL"]

    return config


def get_project_root() -> Path:
    """Return project root (parent of src)."""
    return Path(__file__).resolve().parent.parent
