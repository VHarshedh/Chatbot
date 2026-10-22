# RAG index maintenance

Run from the repository root with the backend virtual environment active and
Ollama serving the configured embedding model:

```powershell
python -m backend.rag.vectorstore
python -m backend.rag.notebook_parser
python -m backend.rag.chat_memory
```

Reindexing is required after this upgrade. Documents now use the versioned
`chatbox_docs_v2` cosine collection. The old `chatbox_docs` L2 collection is
preserved, but is not searched. New chunks include source, domain, and system
metadata; notebook IDs include their paths to avoid same-name collisions.

Search combines cosine candidates (distance <= 0.45) and SQLite FTS5 BM25 with
weighted reciprocal rank fusion. Metadata boosting selects at most three chunks;
this is the lightweight reranking alternative described in the improvement guide.
Exact lexical matches can qualify even when dense similarity is weak.

Memory ingestion accepts non-execution answers, successful execution, or evaluation
scores >= 80. Unknown historical code is skipped. Only entries with the quality
flag are retrieved; old unverified entries remain stored but cannot be retrieved.
Live indexing and historical ingestion use the same stable content-based IDs.

`hybrid_retriever.py` is intentionally unchanged. Its context budgeting, graph
entity linking, and sibling-cell expansion remain governed by that implementation.
