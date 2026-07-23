"""Prompts for medical QA with evidence and safety disclaimers."""

MEDICAL_QA_SYSTEM_PROMPT = """You are a medical evidence assistant. Your role is to answer questions using ONLY the provided evidence excerpts. You must:
- Base your answer strictly on the given context. If the context does not contain enough information, say so clearly.
- Cite sources by referring to the source labels (e.g., [Source 1], [Source 2]) when you use information from them.
- Use clear, professional medical language. Do not make recommendations beyond what the evidence states.
- Do not diagnose, prescribe, or advise on individual patient care—you support evidence retrieval only.

IMPORTANT DISCLAIMER: This system is for informational and evidence-retrieval support only. It does not replace professional medical advice, diagnosis, or treatment. Users should always consult a qualified healthcare provider for medical decisions."""

USER_QA_TEMPLATE = """Use the following evidence excerpts to answer the question. If the excerpts do not contain relevant information, respond with "The provided evidence does not contain sufficient information to answer this question."

## Evidence

{context}

## Question

{question}

## Instructions

Provide a concise, evidence-based answer. Cite each claim with the source label (e.g., [Source 1]). End with a one-line disclaimer: "This is for informational support only; consult a healthcare professional for medical advice." """


def format_context_and_sources(
    doc_score_pairs: list,
    max_chars: int = 12000,
) -> tuple[str, list[dict]]:
    """
    Format retrieved (Document, score) pairs into a single context string
    with [Source N] labels, and return a list of source metadata for citations.
    """
    context_parts = []
    sources = []
    total_chars = 0
    for i, (doc, score) in enumerate(doc_score_pairs, start=1):
        if total_chars >= max_chars:
            break
        label = f"[Source {i}]"
        excerpt = doc.content
        remaining = max_chars - total_chars
        if len(excerpt) > remaining:
            # FIX: guard against a negative slice index when little budget remains
            if remaining <= 50:
                break
            excerpt = excerpt[: remaining - 50] + "..."
        context_parts.append(f"{label} ({doc.source}, {doc.source_id})\n{excerpt}")
        sources.append({
            "label": label,
            "source": doc.source,
            "source_id": doc.source_id,
            "doc_type": doc.doc_type,
            "relevance_score": round(score, 4),
        })
        total_chars += len(context_parts[-1])
    return "\n\n---\n\n".join(context_parts), sources
