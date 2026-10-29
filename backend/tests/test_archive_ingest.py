import json
import zipfile
from unittest.mock import MagicMock

from backend.rag import archive_ingest


def test_existing_folders_duplicates_and_repeat_import(tmp_path, monkeypatch):
    documents = {'old': ('existing content', {'source': 'DFT/README.md'})}
    store = MagicMock()
    store.collection.name = 'test'
    store.collection.get.side_effect = lambda **kw: {
        'ids': list(documents), 'documents': [d[0] for d in documents.values()],
        'metadatas': [d[1] for d in documents.values()]}
    store.collection.count.side_effect = lambda: len(documents)
    def add(texts, metas, ids):
        documents.update(zip(ids, zip(texts, metas)))
    store.add_documents.side_effect = add
    monkeypatch.setattr(archive_ingest, 'VectorStore', lambda: store)
    archive = tmp_path / 'dataset.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr('DFT/new.py', 'must_skip_existing_folder = True')
        zipped.writestr('new/reference.md', 'A unique scientific reference.')
        zipped.writestr('copy/reference.md', 'A unique scientific reference.')
        zipped.writestr('new/diagnostic.png', b'not text')
    report = archive_ingest.ingest([archive], tmp_path / 'reports' / 'first.json')
    assert report['added_chunks'] == 1
    assert report['statuses']['existing_folder'] == 1
    assert report['statuses']['duplicate_content'] == 1
    assert documents['old'] == ('existing content', {'source': 'DFT/README.md'})
    assert archive_ingest.ingest([archive], tmp_path / 'second.json')['added_chunks'] == 0
    assert len(documents) == 2


def test_notebook_ignores_outputs_and_qa_keeps_pair():
    notebook = {'cells': [{'cell_type': 'code', 'source': ['print(1)'],
                           'outputs': [{'text': 'private output'}]}]}
    assert list(archive_ingest.records('a.ipynb', json.dumps(notebook))) == [
        ('print(1)', {'cell_index': 0, 'cell_type': 'code'})]
    row = {'prompt': 'Question', 'response': 'Answer', 'category': 'physics'}
    body, meta = next(archive_ingest.records('a.jsonl', json.dumps(row)))
    assert body == 'Question: Question\nAnswer: Answer'
    assert meta['category'] == 'physics'
