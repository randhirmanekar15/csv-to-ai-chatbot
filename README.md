# CSV to AI Chatbot — RAG from Scratch

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/License-MIT-green) ![Runs 100% Local](https://img.shields.io/badge/runs-100%25%20local-orange)

**Turn any CSV into something you can talk to — RAG built from scratch with FAISS, no framework, fully local.**

## Overview

"Talk to your data" is the single most useful thing LLMs unlocked in 2026. Instead of writing SQL, filtering spreadsheets, or scrolling through 4,000 rows looking for one answer, you ask a question in plain English and get the row that matters. This project does exactly that — point it at a CSV and start asking.

Most people reach for LangChain or LlamaIndex on day one and never learn what those frameworks are actually doing. This project builds Retrieval-Augmented Generation **from the ground up**: serialize rows to text, embed them with a sentence-transformer, index the vectors in FAISS, and search by distance. No abstractions, no magic — just the four steps that every RAG framework wraps in a thousand lines of indirection. If you understand this repo, you understand RAG.

It also runs entirely on your machine. The embedding model — `all-MiniLM-L6-v2` — is only ~80MB and runs fine on CPU, so there are no API keys, no per-token bills, and no data leaving your laptop. Optional answer generation is handled by a local Ollama model (`llama3`), so even the "phrase it nicely" step stays offline. Free, private, and fast enough to feel instant.

## Features

- **Any CSV, zero config** — drop in a file and ask questions immediately.
- **RAG built raw** — serialize → embed → index → search, all visible in one file.
- **Tiny local embedding model** — `all-MiniLM-L6-v2`, ~80MB, CPU-friendly, no API key.
- **Two modes** — return raw matching rows, or `--generate` for an LLM-phrased answer via Ollama.
- **Interactive chat loop** — keep asking follow-up questions in one session.
- **Configurable** — swap the embedding model, the number of results, or the generation model with env vars.
- **100% offline** — nothing leaves your machine.

## How it works

RAG here is three setup steps plus a search step:

1. **Serialize** — each CSV row becomes a single string like `name | role | city`, so the embedding model has natural-language context.
2. **Embed** — every serialized row becomes a 384-dimension vector via `all-MiniLM-L6-v2`. Similar rows land near each other.
3. **Index** — the vectors go into a FAISS `IndexFlatL2`, which finds nearest neighbours by distance.

At query time your question is embedded with the same model, FAISS returns the `TOP_K` closest rows, and (optionally) those rows are handed to a local LLM to phrase a clean answer.

```
            ┌──────────┐
  data.csv  │ serialize│  "col | val | val ..."
 ──────────>│   rows   │──────────────┐
            └──────────┘               ▼
                                 ┌───────────┐
                                 │  embed     │  all-MiniLM-L6-v2 (384-dim)
                                 └───────────┘
                                       │
                                       ▼
                                 ┌───────────┐
                                 │  FAISS     │  IndexFlatL2
                                 └───────────┘
                                       ▲
   "your question" ──> embed ──> search│──> top-k rows
                                       │
                                       └──> [--generate] ──> Ollama (llama3) ──> answer
```

## Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| Data loading | `pandas` | Read and flatten CSV rows |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Tiny, fast, CPU-friendly semantic vectors |
| Vector search | `faiss-cpu` (`IndexFlatL2`) | Exact nearest-neighbour retrieval |
| Math | `numpy` | Vector array handling |
| Generation (optional) | Ollama (`llama3`) | Local LLM phrases the final answer |

## Project structure

```
csv-to-ai-chatbot/
├── chatbot.py          # serialize_rows, build_index, search, answer + chat loop
├── test_chatbot.py     # tests the serialize step
├── ARTICLE.md
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/randhirmanekar15/csv-to-ai-chatbot.git
cd csv-to-ai-chatbot
pip install -r requirements.txt
# optional, for --generate:
ollama pull llama3
```

## Usage

Return the most relevant rows for your question:

```bash
python chatbot.py data.csv
```

Let a local LLM phrase a natural-language answer:

```bash
python chatbot.py data.csv --generate
```

Either command drops you into an interactive chat loop (Ctrl-C to quit).

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model used for embeddings |
| `TOP_K` | `3` | Number of rows returned per query |
| `OLLAMA_MODEL` | `llama3` | Local model used for `--generate` answers |

## Testing

```bash
pip install pytest
pytest
```

The test covers the **serialize** step — the layer most likely to silently break retrieval if a column is dropped or formatted wrong.

## Limitations

- **Aggregate questions fail.** Row-level retrieval can't compute across all rows ("what's the average salary?" returns rows, not a calculation).
- **`IndexFlatL2` scans every vector.** Exact and perfect for thousands of rows; brute-force at hundreds of thousands (use IVF/HNSW).
- **Embeddings don't reason about numbers.** For numeric filters, pair this with pandas.

## Roadmap

- [ ] Persist the FAISS index to disk (skip re-embedding on every run)
- [ ] Streamlit UI for non-terminal users
- [ ] IVF / HNSW index option for large datasets
- [ ] Hybrid retrieval (semantic + numeric/pandas filters)
- [ ] Batch embedding for faster index builds

## Credits

📖 Full write-up: [ARTICLE.md](ARTICLE.md).

Based on Aman Kharwal's tutorial, ["Turn Any CSV into an AI Chatbot with Python"](https://amanxai.com/2026/03/15/turn-any-csv-into-an-ai-chatbot-with-python/).

**What I changed vs the source tutorial:**

- Added an **optional local generation step** — retrieved rows are passed to a local Ollama LLM that phrases an answer, instead of only dumping raw rows.
- Wrapped everything in an **interactive chat loop**.
- Made the **embedding model and `TOP_K` configurable** via environment variables.

## Author

Built by **Randhir Manekar** — [randhirmanekar.com](https://randhirmanekar.com) · [github.com/randhirmanekar15](https://github.com/randhirmanekar15)

## License

MIT — see [LICENSE](LICENSE).
