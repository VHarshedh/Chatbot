"""Add previously unseen ZIP sources to the existing RAG collection."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from backend.config import settings
from backend.rag.vectorstore import VectorStore, chunk_header, provenance, stable_id


def digest(text):
    return hashlib.sha256(text.strip().replace('\r\n', '\n').encode()).hexdigest()


def records(name, text):
    """Yield semantic units, never notebook outputs or raw trajectory coordinates."""
    suffix = PurePosixPath(name).suffix.lower()
    if suffix == '.ipynb':
        for index, cell in enumerate(json.loads(text).get('cells', [])):
            if cell.get('cell_type') not in {'code', 'markdown'}:
                continue
            body = cell.get('source', '')
            body = ''.join(body) if isinstance(body, list) else body
            if body.strip():
                yield body, {'cell_index': index, 'cell_type': cell['cell_type']}
    elif suffix in {'.json', '.jsonl', '.csv'}:
        if suffix == '.csv':
            data = list(csv.DictReader(io.StringIO(text)))
        elif suffix == '.jsonl':
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
        for index, row in enumerate(data if isinstance(data, list) else [data]):
            meta = {'record_index': index}
            if isinstance(row, dict) and 'prompt' in row and 'response' in row:
                body = f"Question: {row['prompt']}\nAnswer: {row['response']}"
                meta.update({k: row[k] for k in ('category', 'physics_topic') if isinstance(row.get(k), str)})
            else:
                body = json.dumps(row, ensure_ascii=False, indent=2)
            yield body, meta
    elif suffix == '.extxyz':
        from ase.io import read
        frames = read(io.StringIO(text), index=':', format='extxyz')
        first = frames[0]
        yield json.dumps({'structure': name, 'frames': len(frames),
                          'atoms_per_frame': len(first), 'formula': first.get_chemical_formula(),
                          'element_counts': dict(Counter(first.get_chemical_symbols())),
                          'cell_A': first.cell.tolist(), 'pbc': first.pbc.tolist(),
                          'note': 'Structure summary; full coordinates are retained in the source archive.'}, indent=2), {}
    else:
        yield text, {}


def ingest(archives, report_path, dry_run=False):
    store = VectorStore()
    original = store.collection.get(include=['documents', 'metadatas'])
    existing_sources = {m.get('source') for m in original['metadatas']}
    existing_folders = {PurePosixPath(m['source']).parts[0] for m in original['metadatas']
                        if m.get('source') and '/' in m['source'] and not m.get('archive')}
    seen_chunks = set()
    for text in original['documents']:
        body = text.split('\n', 1)[1] if text.startswith('[Source:') else text
        seen_chunks.add(digest(body))
    seen_files = {m['content_sha256'] for m in original['metadatas'] if 'content_sha256' in m}
    # Also recognize renamed copies of local sources that have already been indexed.
    for source in existing_sources:
        if source:
            path = Path(settings.WORKSPACE_ROOT) / source
            if path.is_file():
                try:
                    seen_files.add(digest(path.read_text(encoding='utf-8-sig')))
                except (OSError, UnicodeError):
                    pass
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=1200, chunk_overlap=150)
    report = {'collection': store.collection.name, 'dry_run': dry_run,
              'initial_chunks': len(original['ids']), 'sources': []}
    supported = {'.py', '.md', '.sh', '.sub', '.json', '.jsonl', '.csv', '.ipynb', '.in', '.lammps', '.cif', '.extxyz'}
    for archive in archives:
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                if member.is_dir():
                    continue
                name = PurePosixPath(member.filename).as_posix()
                entry = {'archive': Path(archive).name, 'source': name, 'chunks': 0}
                report['sources'].append(entry)
                if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
                    raise ValueError(f'Unsafe archive member: {name}')
                if PurePosixPath(name).suffix.lower() not in supported or PurePosixPath(name).name in {'validation_report.json', 'eda_cleaning_report.json'}:
                    entry['status'] = 'excluded_non_reference'
                    continue
                if PurePosixPath(name).parts[0] in existing_folders:
                    entry['status'] = 'existing_folder'
                    continue
                if name in existing_sources:
                    entry['status'] = 'existing_source'
                    continue
                text = zipped.read(member).decode('utf-8-sig')
                file_hash = digest(text)
                if file_hash in seen_files:
                    entry['status'] = 'duplicate_content'
                    continue
                meta = {**provenance(Path(settings.WORKSPACE_ROOT) / name),
                        'archive': Path(archive).name, 'archive_member': name,
                        'content_sha256': file_hash, 'doc_type': 'archive_chunk'}
                if name.startswith(('physchem_', 'physchem_kb/')):
                    meta['domain'] = 'Physics and Chemistry'
                texts, metadatas, ids = [], [], []
                for body, record_meta in records(name, text):
                    split = python_splitter if name.endswith('.py') else splitter
                    for chunk in split.split_text(body):
                        signature = digest(chunk)
                        if signature in seen_chunks:
                            continue
                        seen_chunks.add(signature)
                        metadata = {**meta, **record_meta}
                        # Notebook sibling expansion relies on the cell doc_type and parent ID.
                        if name.endswith('.ipynb'):
                            metadata.update(doc_type='notebook_cell', notebook_name=PurePosixPath(name).name,
                                            parent_id=stable_id(name, 'notebook'))
                        content = chunk_header(metadata) + chunk
                        texts.append(content)
                        metadatas.append(metadata)
                        ids.append(stable_id(name, content))
                if texts and not dry_run:
                    store.add_documents(texts, metadatas, ids)
                existing_sources.add(name)
                seen_files.add(file_hash)
                entry.update(status='added' if texts else 'duplicate_chunks', chunks=len(texts))
                print(f"{entry['status']}: {name} ({len(texts)} chunks)", flush=True)
    report['final_chunks'] = store.collection.count()
    report['added_chunks'] = sum(s['chunks'] for s in report['sources'])
    report['statuses'] = dict(Counter(s['status'] for s in report['sources']))
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+')
    parser.add_argument('--report', default='archive_ingestion_report.json')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    ingest(args.archives, args.report, args.dry_run)
