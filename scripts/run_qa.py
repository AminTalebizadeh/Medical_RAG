"""
Interactive Medical RAG QA (CLI).
Run from project root: python scripts/run_qa.py "Your question here"
Or without args for interactive loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import build_pipeline


def main():
    try:
        chain, _ = build_pipeline()
    except FileNotFoundError as e:
        print(e)
        print("Run: python scripts/run_ingest.py")
        return 1

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        response = chain.run(query)
        print("\n--- Answer ---\n")
        print(response.answer)
        print("\n--- Evidence sources ---\n")
        for s in response.sources:
            print(f"  {s['label']}: {s['source']} — {s['source_id']} (score: {s.get('relevance_score', 'N/A')})")
        return 0

    print("Medical RAG — Evidence retrieval (type 'quit' to exit)\n")
    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        response = chain.run(query)
        print("\n--- Answer ---\n")
        print(response.answer)
        print("\n--- Sources ---")
        for s in response.sources:
            print(f"  {s['label']}: {s['source']} — {s['source_id']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
