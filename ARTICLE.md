# Turn Any CSV into an AI Chatbot — Building RAG From Scratch (No LangChain)

*I built a system that turns a spreadsheet into something you can talk to. No framework, no API key, no cloud bill. Here's how it works and where it falls apart.*

## Why this, why now

"Talk to your data" is the most useful thing LLMs do right now. Not writing poems — answering questions about *your* CSV, *your* support tickets, *your* product catalog. That pattern is RAG: retrieve the relevant rows, then let a model phrase the answer.

Most people reach for LangChain on day one. I think that's a mistake if you're learning. Frameworks hide the three steps that actually matter, and when something breaks you have no idea which layer failed.

So I built it from scratch. The whole thing runs locally on a laptop. The embedding model, `all-MiniLM-L6-v2`, is ~80MB and edge-sized — it encodes thousands of rows in seconds with zero API cost. Building it raw taught me more in an afternoon than a month of gluing abstractions together.

## What it does

You point it at a CSV. It reads every row, turns each one into a chunk of text, and builds a searchable index in memory. Then you ask a question in plain English — "which customers churned in Q3?" — and it pulls back the handful of rows most relevant to your question and hands them to a local model to write the answer.

No SQL. No exact-match filters. Just semantic search over your spreadsheet.

## The stack

| Layer | Tool | Why |
|---|---|---|
| Data | pandas | Load and flatten the CSV |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | 384-dim vectors, fast, fully local |
| Index | faiss-cpu | Millisecond similarity search |
| Math | numpy | Array glue between the pieces |
| Generation (my add) | Ollama (local Llama 3) | Phrases the final answer |

## How it works

Three steps: serialize, embed, index. Then search at query time.

**Serialize.** Each row becomes one string. I join the columns with a pipe so the model sees field boundaries instead of mush.

```python
documents = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
```

**Embed and index.** Encode every document into a vector, then drop those vectors into a flat L2 index.

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)          # 384-dim per row

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
```

**Search.** A query gets embedded the same way, and FAISS returns the nearest `k` rows by distance.

```python
query_vector = model.encode([query])
D, I = index.search(query_vector, k=3)
results = [documents[i] for i in I[0]]
```

That's the entire core. Maybe 15 lines that do real work. Everything LangChain wraps is right here in the open.

## What I changed

The base tutorial stops at retrieval — it hands you the top-k rows. That's a lookup tool, not a chatbot. I pushed it further:

1. **Added a generation step.** Retrieving rows isn't an answer. I feed the top-3 rows plus the question into a local Ollama model and ask it to write a natural reply grounded only in those rows. That's the line between "search result" and "chatbot."

2. **Wrapped it in a chat loop.** A `while True` REPL first, then a thin Streamlit front end so non-technical people can drop a CSV and start asking questions. Took about 30 lines.

3. **Persisted the index.** Re-embedding 50k rows on every launch is wasteful. I save the FAISS index with `faiss.write_index` and pickle the document list, so startup drops from ~40s to instant.

4. **Handled large CSVs.** I batch `model.encode` in chunks of 512 with a progress bar so a 200k-row file doesn't blow up memory or look frozen.

## Where it breaks

I'll be honest about the limits, because they're sharp.

**Aggregate questions fail.** Ask "what's the average order value?" and row-level retrieval is useless — the answer lives across *all* rows, not the top 3. RAG retrieves; it doesn't compute. For those, you need a SQL path or a pandas tool the model can call.

**FlatL2 doesn't scale.** `IndexFlatL2` is exact, which means it compares your query to every vector. Lovely at 10k rows, painful at 10M. Past a few hundred thousand, switch to `IndexIVFFlat` or HNSW and trade a little accuracy for huge speed.

**Embeddings can't do numbers.** `all-MiniLM-L6-v2` knows "expensive" loosely relates to "$4,000," but it has no concept that 4000 > 399. Any question that hinges on numeric comparison or ranges will quietly return wrong rows.

## Takeaway

RAG isn't magic and it isn't a framework — it's three steps you can read in one screen. Serialize, embed, index, search. Build it raw once and every "AI data assistant" product on the market suddenly looks a lot less mysterious.

*This project builds on Aman Kharwal's walkthrough, ["Turn Any CSV into an AI Chatbot with Python."](https://amanxai.com/2026/03/15/turn-any-csv-into-an-ai-chatbot-with-python/) I adapted his retrieval core and extended it with local generation, persistence, and a chat UI.*

### Sources
- [Aman Kharwal — Turn Any CSV into an AI Chatbot with Python](https://amanxai.com/2026/03/15/turn-any-csv-into-an-ai-chatbot-with-python/)
- [Codersera — The Open-Source LLM Landscape in 2026](https://codersera.com/blog/open-source-llms-landscape-2026/)
- [Google Cloud — AI Agent Trends 2026](https://cloud.google.com/resources/content/ai-agent-trends-2026)
