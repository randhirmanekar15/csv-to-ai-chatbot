"""Turn any CSV into an AI chatbot — RAG from scratch (no framework).

Serialize rows to text, embed them, index with FAISS, and answer questions by
retrieving the most similar rows. An optional generation step asks a local model
to phrase the final answer from the retrieved rows.

Inspired by Aman Kharwal's tutorial:
https://amanxai.com/2026/03/15/turn-any-csv-into-an-ai-chatbot-with-python/
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")


def _read_top_k() -> int:
    raw = os.environ.get("TOP_K", "3")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"TOP_K must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"TOP_K must be >= 1, got {value}")
    return value


TOP_K = _read_top_k()


def serialize_rows(df: pd.DataFrame) -> list[str]:
    """Turn each row into a pipe-delimited string the embedder can understand."""
    return df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()


def build_index(documents: list[str]):
    """Embed documents and build a FAISS index. Returns (model, index)."""
    if not documents:
        raise ValueError("No documents to index — is the CSV empty?")

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return model, index


def search(model, index, documents: list[str], query: str, k: int = TOP_K) -> list[str]:
    """Return the top-k most relevant rows for a query."""
    import numpy as np

    query_vector = model.encode([query])
    _distances, indices = index.search(np.array(query_vector), k)
    return [documents[i] for i in indices[0]]


def answer(query: str, rows: list[str]) -> str:
    """Optional generation: ask a local model to phrase an answer from the rows."""
    import ollama

    context = "\n".join(rows)
    prompt = (
        "Answer the question using ONLY the rows below. If the rows do not contain "
        f"the answer, say so.\n\nRows:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    model = os.environ.get("OLLAMA_MODEL", "llama3")
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    except Exception as exc:  # noqa: BLE001  surface a controlled error
        raise RuntimeError(
            "Failed to reach Ollama for answer generation. Is the daemon running "
            f"and model '{model}' pulled?"
        ) from exc
    return response["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a CSV via local RAG")
    parser.add_argument("csv", help="Path to a CSV file")
    parser.add_argument("--generate", action="store_true", help="Phrase an answer with an LLM")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Could not read CSV '{args.csv}': {exc}") from exc

    documents = serialize_rows(df)
    model, index = build_index(documents)
    print("Ready. Ask questions (Ctrl-C to quit).")
    try:
        while True:
            query = input("\n> ")
            rows = search(model, index, documents, query)
            if args.generate:
                print(answer(query, rows))
            else:
                print("\n".join(rows))
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
