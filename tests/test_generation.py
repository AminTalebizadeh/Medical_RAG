"""Tests for generation: prompts, context formatting."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingest.documents import Document
from src.generation.prompts import format_context_and_sources, MEDICAL_QA_SYSTEM_PROMPT, USER_QA_TEMPLATE


def test_format_context_and_sources():
    docs = [
        (Document("Excerpt one.", source="CDC", source_id="flu"), 0.9),
        (Document("Excerpt two.", source="DrugBank", source_id="Amoxicillin"), 0.8),
    ]
    context, sources = format_context_and_sources(docs, max_chars=500)
    assert "[Source 1]" in context and "CDC" in context
    assert "[Source 2]" in context and "DrugBank" in context
    assert len(sources) == 2
    assert sources[0]["label"] == "[Source 1]" and sources[0]["source"] == "CDC"
    assert sources[1]["relevance_score"] == 0.8


def test_user_template_placeholders():
    assert "{context}" in USER_QA_TEMPLATE and "{question}" in USER_QA_TEMPLATE


def test_system_prompt_disclaimer():
    assert "informational" in MEDICAL_QA_SYSTEM_PROMPT.lower() or "evidence" in MEDICAL_QA_SYSTEM_PROMPT.lower()
    assert "consult" in MEDICAL_QA_SYSTEM_PROMPT.lower() or "professional" in MEDICAL_QA_SYSTEM_PROMPT.lower()
