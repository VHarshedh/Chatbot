import hashlib
import os
from pathlib import Path
import re
import sqlite3

import chromadb
from langchain_text_splitters import Language, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from backend.config import settings
from backend.rag.embeddings import OllamaEmbeddings

EXCLUDED_DIRS = {"rag_dataset","venv", ".venv", "env", "tests", "__pycache__", ".pytest_cache", "chroma_db", "kuzu_db", ".git", "node_modules", ".ipynb_checkpoints"}


def provenance(path):
    path = Path(path).resolve()
    try:
        source = path.relative_to(Path(settings.WORKSPACE_ROOT).resolve()).as_posix()
    except ValueError:
        source = path.as_posix()
    domain = next((label for folder, label in (("DFT", "DFT"), ("Molecular Dynamics", "Molecular Dynamics"), ("Training", "DeepMD Training"), ("Analysis", "Interfacial Analysis"), ("docs", "ASE Documentation")) if folder in path.parts), "General")
    name = source.lower()
    system = next((tag for words, tag in ((["tio2", "anatase", "running.ipynb"], "TiO2"), (["iro2"], "IrO2"), (["graphene"], "Graphene"), (["water"], "Water"), (["pt.cif", "platinum"], "Pt"), (["ase"], "ASE")) if any(w in name for w in words)), "General")
    return {"source": source, "domain": domain, "system_tag": system}


def chunk_header(meta):
    return f"[Source: {meta['source']} | Domain: {meta['domain']} | System: {meta['system_tag']}]\n"


def stable_id(source, content):
    return hashlib.sha256(f"{source}\0{content}".encode()).hexdigest()


def reciprocal_rank_fusion(dense_results, bm25_results, k=60, w_dense=0.6, w_bm25=0.4, top_n=10):
    scores, docs = {}, {}
    for results, weight in ((dense_results, w_dense), (bm25_results, w_bm25)):
        for rank, item in enumerate(results, 1):
            key = item["id"]
            docs.setdefault(key, item)
            scores[key] = scores.get(key, 0) + weight / (k + rank)
    return [{**docs[key], "rrf_score": scores[key]} for key in sorted(scores, key=scores.get, reverse=True)[:top_n]]


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.embeddings = OllamaEmbeddings()
        # Versioned collection: old L2 indexes must never be interpreted as cosine.
        self.collection = self.client.get_or_create_collection(name="chatbox_docs_v2", metadata={"hnsw:space": "cosine"})

    def add_documents(self, texts, metadatas, ids):
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            self.collection.upsert(documents=batch, embeddings=self.embeddings.embed_documents(batch), metadatas=metadatas[start:start + 64], ids=ids[start:start + 64])

    def semantic_search(self, query, n_results=5, filter=None):
        if not query.strip() or n_results <= 0 or not self.collection.count():
            return []
        corpus = self.collection.get(where=filter, include=["documents", "metadatas"])
        docs = [{"id": key, "content": content, "metadata": meta or {}} for key, content, meta in zip(corpus["ids"], corpus["documents"], corpus["metadatas"])]
        requested_systems = {tag for pattern, tag in ((r"\b(tio2|anatase|titanium)\b", "TiO2"), (r"\b(iro2|iridium)\b", "IrO2")) if re.search(pattern, query, re.I)}
        if len(requested_systems) == 1:
            docs = [d for d in docs if d["metadata"].get("system_tag") not in {"TiO2", "IrO2"} - requested_systems]
        if not docs:
            return []
        dense = []
        result = self.collection.query(query_embeddings=[self.embeddings.embed_query(query)], n_results=min(12, len(docs)), where=filter)
        for key, content, meta, distance in zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]):
            if distance <= 0.45 and key in {d["id"] for d in docs}:
                dense.append({"id": key, "content": content, "metadata": meta or {}, "distance": distance})
        # FTS5 supplies BM25 without a separate dependency or a stale side index.
        with sqlite3.connect(":memory:") as conn:
            conn.execute("CREATE VIRTUAL TABLE chunks USING fts5(content)")
            conn.executemany("INSERT INTO chunks(rowid, content) VALUES (?, ?)", [(i + 1, d["content"]) for i, d in enumerate(docs)])
            tokens = list(dict.fromkeys(re.findall(r"[\w]+", query.lower())))[:64]
            match = " OR ".join('"' + t + '"' for t in tokens)
            lexical = [docs[row[0] - 1] for row in conn.execute("SELECT rowid FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT 12", (match,))] if match else []
        candidates = reciprocal_rank_fusion(dense, lexical)
        query_lower = query.lower()
        def score(item):
            meta = item["metadata"]
            boost = sum(0.005 for field in ("domain", "system_tag") if meta.get(field, "General") != "General" and meta[field].lower() in query_lower)
            return item["rrf_score"] + boost
        candidates.sort(key=score, reverse=True)
        selected, seen = [], set()
        for item in candidates:
            signature = item["content"].strip()
            if signature not in seen:
                selected.append(item)
                seen.add(signature)
            if len(selected) >= min(n_results, 3):
                break
        if len(selected) == 3:
            selected[1], selected[2] = selected[2], selected[1]
        return selected

    def notebook_siblings(self, result, max_chars=1600):
        """Condense executable siblings in cell order, preserving imports/signatures."""
        import ast
        parent = result.get("metadata", {}).get("parent_id")
        if not parent:
            return ""
        data = self.collection.get(where={"$and": [{"parent_id": parent}, {"cell_type": "code"}]}, include=["documents", "metadatas"])
        blocks, used = [], 0
        for content, meta in sorted(zip(data["documents"], data["metadatas"]), key=lambda row: row[1].get("cell_index", 0)):
            if meta.get("cell_index") == result["metadata"].get("cell_index"):
                continue
            code = "\n".join(line for line in content.splitlines() if not line.startswith("[Source:"))
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            lines = []
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
                    text = ast.unparse(node)
                    if len(text) <= 300:
                        lines.append(text)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    lines.append(ast.unparse(node).splitlines()[0] + " ...")
            block = f"# Sibling cell {meta['cell_index']} (dependency outline, not executable)\n" + "\n".join(lines)
            if lines and used + len(block) <= max_chars:
                blocks.append(block)
                used += len(block)
        return "\n".join(blocks)


def ingest_readmes():
    store = VectorStore()
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=1200, chunk_overlap=150)
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "subsection")], strip_headers=False)
    for directory, folders, files in os.walk(settings.WORKSPACE_ROOT):
        folders[:] = sorted(f for f in folders if f not in EXCLUDED_DIRS)
        for name in sorted(files):
            path = Path(directory) / name
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            if path.suffix not in {".py", ".md"}:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                print(f"Skipping {path}: {exc}")
                continue
            meta = {**provenance(path), "doc_type": "source_chunk"}
            chunks = python_splitter.split_text(content) if path.suffix == ".py" else [d.page_content for d in markdown_splitter.split_text(content)]
            texts = list(dict.fromkeys(chunk_header(meta) + c for c in chunks if c.strip()))
            if texts:
                ids = [stable_id(meta["source"], t) for t in texts]
                store.add_documents(texts, [meta] * len(texts), ids)
                old = store.collection.get(where={"$and": [{"source": meta["source"]}, {"doc_type": "source_chunk"}]})
                stale = sorted(set(old["ids"]) - set(ids))
                if stale:
                    store.collection.delete(ids=stale)


if __name__ == "__main__":
    ingest_readmes()
