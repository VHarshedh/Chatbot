# RAG index maintenance

To add only new folders from the supplied dataset archives, run from the repository root:

```powershell
python -m backend.rag.archive_ingest dataset.zip curated_scientific_scripts_dataset.zip physics-chem-dataset.zip --report docs/archive_ingestion_report.json
```

The importer skips folders already represented by non-archive sources in the active
collection, then skips indexed source paths and duplicate content across archives.
It only adds chunks; existing entries are preserved. `--dry-run` previews additions.
Notebook outputs, diagnostic images, and dataset validation reports are excluded.
Structured references and question/answer records are split into semantic records;
EXTXYZ files contribute structure summaries rather than raw coordinate dumps.
Each added chunk retains its archive and member path. The JSON report records all skips.

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

Curated GraphRAG ingestion runs at API startup, using `dataset.zip` at the workspace
root or `DATASET_ZIP_PATH` when set. Run it manually with:

```powershell
python -m backend.rag.ingest_dataset_zip
```

Relevant assets are safely extracted into ignored `rag_dataset/`. Physics triples
and exact file cards are merged into Kùzu; new recipe/law chunks use stable IDs.
Existing workspace folders remain excluded. Previously imported archive recipes
are upgraded to 2400-character chunks; markdown follows `###` sections. Python
recipes must parse with `ast` and must not import `pwtools`. This validates syntax,
not execution or scientific accuracy.

Scientific retrieval includes physics rules, exact cards for explicitly named
files, and hybrid vector/lexical search across law, recipe, and detected domains.
Molecular structure queries use vector retrieval too. Context uses
`<physchem_invariants>` and `<structure_metadata>` blocks. Notebook prompt
resolution also attaches exact cards without sending raw coordinates.
