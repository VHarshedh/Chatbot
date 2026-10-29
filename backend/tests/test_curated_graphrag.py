import json
from unittest.mock import MagicMock
import zipfile

import kuzu
import pytest
from backend.rag.knowledge_graph import KnowledgeGraph
from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.ingest_dataset_zip import ingest_curated_dataset_zip


def test_exact_graph_rules_and_file_cards(tmp_path):
    graph = KnowledgeGraph.__new__(KnowledgeGraph)
    graph.db = kuzu.Database(str(tmp_path / 'graph'), buffer_pool_size=64 * 1024 * 1024)
    graph.conn = kuzu.Connection(graph.db)
    graph._init_schema()
    rows = [{'subject': 'Anatase_TiO2', 'predicate': 'CUTOFF', 'object': '3.20'},
            {'subject': 'Rutile_IrO2', 'predicate': 'CUTOFF', 'object': '3.00'}]
    graph.load_physchem_triples(rows)
    graph.load_physchem_triples(rows)
    rules = graph.query_physchem_rules('Anatase_TiO2_Water.extxyz boundary')
    assert any('3.20' in r for r in rules)
    assert not any('3.00' in r for r in rules)
    assert len(graph.graph_query('MATCH (r:PhysChemRule) RETURN r.id')) == 2
    graph.load_structure_cards({'/mnt/data/exact.extxyz': {'total_atoms': 426}})
    assert graph.structure_metadata('Read /mnt/data/exact.extxyz')['/mnt/data/exact.extxyz']['total_atoms'] == 426
    assert graph.structure_metadata('Read other.extxyz') == {}
    assert graph.structure_metadata('Read not_exact.extxyz') == {}
    graph.conn.close()
    graph.db.close()


def test_molecular_query_keeps_vectors_and_injects_graph():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.kg = MagicMock()
    retriever.kg.graph_query.return_value = []
    retriever.kg.query_physchem_rules.return_value = ['O-H cutoff = 1.25']
    retriever.kg.structure_metadata.return_value = {'exact.extxyz': {'total_atoms': 426}}
    retriever.vector_store = MagicMock()
    retriever.vector_store.semantic_search.return_value = [{'content': 'from ase.neighborlist import neighbor_list'}]
    retriever.chat_memory = MagicMock()
    retriever.chat_memory.search_memory.return_value = []
    context = retriever.retrieve('Anatase TiO2 water molecular adsorbed boundary')
    retriever.vector_store.semantic_search.assert_called_once()
    assert 'Verified Code Recipes' in str(retriever.vector_store.semantic_search.call_args)
    assert '<physchem_invariants>' in context
    assert '<structure_metadata>' in context
    assert 'from ase.neighborlist' in context
    assert 'lowest `s * len(m_z)`' not in context


def test_zip_traversal_rejected_before_extraction(tmp_path):
    path = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('good.md', '# valid')
        archive.writestr('../escape.py', 'bad')
    store = MagicMock()
    store.collection.get.return_value = {'metadatas': [], 'ids': []}
    with pytest.raises(ValueError, match='Unsafe ZIP'):
        ingest_curated_dataset_zip(path, tmp_path / 'extracted', store, MagicMock())
    assert not (tmp_path / 'extracted' / 'good.md').exists()
    store.add_documents.assert_not_called()
