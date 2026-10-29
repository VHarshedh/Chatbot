"""Incremental curated archive ingestion for graph facts and verified recipes."""
import ast
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, Language
from backend.config import settings
from backend.rag.vectorstore import VectorStore, provenance, chunk_header, stable_id
from backend.rag.knowledge_graph import KnowledgeGraph

LAWS = 'PhysChem & ASE Laws'
RECIPES = 'Verified Code Recipes'


def ingest_curated_dataset_zip(zip_path, extract_dir, vector_store, knowledge_graph):
    """Safely materialize relevant assets and add only missing source/chunk IDs.

    Preserve folders previously indexed from the workspace. Existing archive recipes
    can be upgraded to section-level chunks without reindexing unrelated documents.
    """
    root = Path(extract_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    indexed = vector_store.collection.get(include=['metadatas'])
    folders = {PurePosixPath(m['source']).parts[0] for m in indexed['metadatas']
               if '/' in m.get('source', '') and not m.get('archive')}
    result = {'graph_rules': 0, 'structure_cards': 0, 'added_chunks': 0, 'skipped_sources': 0}
    md = MarkdownHeaderTextSplitter(headers_to_split_on=[('#', 'title'), ('##', 'section'), ('###', 'subsection')], strip_headers=False)
    splitter = RecursiveCharacterTextSplitter(chunk_size=2400, chunk_overlap=200)
    code = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=2400, chunk_overlap=200)
    with zipfile.ZipFile(zip_path) as archive:
        # Validate all paths before writing any file (do not use extractall).
        for member in archive.infolist():
            name = PurePosixPath(member.filename.replace('\\', '/'))
            target = (root / str(name)).resolve()
            if name.is_absolute() or '..' in name.parts or not target.is_relative_to(root) or ':' in str(name):
                raise ValueError(f'Unsafe ZIP member: {member.filename}')
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = member.filename
            is_triples = name == 'physchem_kb/physchem_knowledge_graph_triples.json'
            is_cards = name == 'physchem_kb/structure_eda_cards.json'
            if not (is_triples or is_cards or name.endswith(('.md', '.py'))):
                continue
            text = archive.read(member).decode('utf-8-sig')
            if name.endswith('.py'):
                try:
                    tree = ast.parse(text, filename=name)
                except SyntaxError:
                    result['skipped_sources'] += 1
                    continue
                if any((isinstance(n, ast.Import) and any(a.name.split('.')[0] == 'pwtools' for a in n.names)) or
                       (isinstance(n, ast.ImportFrom) and (n.module or '').split('.')[0] == 'pwtools') for n in ast.walk(tree)):
                    result['skipped_sources'] += 1
                    continue
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_text(encoding='utf-8') != text:
                target.write_text(text, encoding='utf-8')
            if is_triples:
                rows = json.loads(text)
                knowledge_graph.load_physchem_triples(rows)
                result['graph_rules'] += len(rows)
                continue
            if is_cards:
                cards = json.loads(text)
                knowledge_graph.load_structure_cards(cards)
                result['structure_cards'] += len(cards)
                continue
            if PurePosixPath(name).parts[0] in folders:
                # Existing ASE documentation already has usable chunks: retag only.
                if name == 'docs/ase_molecular_simulations.md':
                    old = vector_store.collection.get(where={'source': name}, include=['metadatas'])
                    if old['ids']:
                        vector_store.collection.update(ids=old['ids'], metadatas=[{**m, 'domain': LAWS} for m in old['metadatas']])
                result['skipped_sources'] += 1
                continue
            # Only curated physics guides and syntactically validated recipes get these labels.
            if not (name.startswith(('physchem_kb/', 'sample_questions/')) or name.endswith('_validated.py')):
                continue
            domain = LAWS if name.endswith('.md') else RECIPES
            meta = {**provenance(Path(settings.WORKSPACE_ROOT) / name), 'domain': domain,
                    'archive': Path(zip_path).name, 'archive_member': name,
                    'doc_type': 'source_chunk', 'system_tag': 'General'}
            chunks = code.split_text(text) if name.endswith('.py') else [c for section in md.split_text(text) for c in splitter.split_text(section.page_content)]
            texts = list(dict.fromkeys(chunk_header(meta) + c for c in chunks if c.strip()))
            ids = [stable_id(name, t) for t in texts]
            old = vector_store.collection.get(where={'source': name}, include=['metadatas'])
            missing = [(t, i) for t, i in zip(texts, ids) if i not in set(old['ids'])]
            if missing:
                vector_store.add_documents([t for t, i in missing], [meta] * len(missing), [i for t, i in missing])
                result['added_chunks'] += len(missing)
            stale = [i for i, m in zip(old['ids'], old['metadatas']) if m.get('archive') and i not in ids]
            if stale:
                vector_store.collection.delete(ids=stale)
    return result


def ingest_configured_dataset():
    path = Path(os.getenv('DATASET_ZIP_PATH', str(Path(settings.WORKSPACE_ROOT) / 'dataset.zip')))
    if not path.exists():
        return {'skipped': 'Dataset archive not found'}
    return ingest_curated_dataset_zip(path, Path(settings.WORKSPACE_ROOT) / 'rag_dataset',
                                      VectorStore(), KnowledgeGraph.get_instance())


if __name__ == '__main__':
    print(ingest_configured_dataset())
