# CSV to AI Chatbot (RAG from scratch)

Turn any CSV into something you can talk to — no LangChain, no API key, no cloud. Serialize rows to text, embed them, index with FAISS, retrieve the most relevant rows for a question, and optionally let a local model phrase the answer.

## Stack

| Piece | Choice |
|-------|--------|
| Data | pandas |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Index | faiss-cpu |
| Generation (optional) | Ollama (`llama3`) |

## Setup

```bash
pip install -r requirements.txt
# optional, for --generate:
ollama pull llama3
```

## Usage

```bash
python chatbot.py data.csv              # returns the most relevant rows
python chatbot.py data.csv --generate   # phrases an answer with a local LLM
```

## Test

```bash
pip install pytest
pytest        # serialize step, no embeddings needed
```

## Limitations

- Aggregate questions ("average order value") fail — row-level retrieval doesn't compute across all rows.
- `IndexFlatL2` is exact but scans every vector; switch to IVF/HNSW past a few hundred thousand rows.
- Embeddings don't do numeric reasoning (4000 > 399 means nothing to them).

---

Inspired by Aman Kharwal's tutorial, [Turn Any CSV into an AI Chatbot with Python](https://amanxai.com/2026/03/15/turn-any-csv-into-an-ai-chatbot-with-python/). Rebuilt and extended (optional local generation, interactive chat loop, configurable model/k).

MIT licensed.
