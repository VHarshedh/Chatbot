import json
from unittest.mock import MagicMock

from backend.rag.chat_memory import ChatMemoryStore
from backend.rag.vectorstore import VectorStore, reciprocal_rank_fusion
from backend.rag.notebook_parser import extract_notebook_cells
from backend.rag.query_rewriting import expand_query


def memory_store():
    store = ChatMemoryStore.__new__(ChatMemoryStore)
    store.collection = MagicMock()
    store.embeddings = MagicMock()
    store.embeddings.embed_documents.return_value = [[0.1, 0.2]]
    return store


def test_memory_rejects_failed_and_unknown_code():
    store = memory_store()
    args = dict(session_id=1, user_message="run this", assistant_message="```python\nbad()\n```", execution_required=True)
    assert not store.index_turn(**args, exit_code=1)
    assert not store.index_turn(**args)
    store.collection.upsert.assert_not_called()
    assert store.index_turn(**args, exit_code=0)
    assert store.index_turn(**args, evaluation_score=80)
    assert store.collection.upsert.call_args.kwargs["metadatas"][0]["verified_execution"] is True


def test_memory_ids_stable_across_instances():
    first, second = memory_store(), memory_store()
    args = dict(session_id=2, user_message="What is water?", assistant_message="H2O")
    first.index_turn(**args)
    second.index_turn(**args)
    assert first.collection.upsert.call_args.kwargs["ids"] == second.collection.upsert.call_args.kwargs["ids"]


def test_hybrid_search_lexical_rescue_and_cosine_gate():
    store = VectorStore.__new__(VectorStore)
    store.collection = MagicMock()
    store.embeddings = MagicMock()
    store.collection.count.return_value = 2
    store.collection.get.return_value = {"ids": ["exact", "noise"], "documents": ["ecutwfc = 80", "unrelated text"], "metadatas": [{}, {}]}
    store.collection.query.return_value = {"ids": [["noise", "exact"]], "documents": [["unrelated text", "ecutwfc = 80"]], "metadatas": [[{}, {}]], "distances": [[0.9, 0.7]]}
    results = store.semantic_search("ecutwfc")
    assert [r["id"] for r in results] == ["exact"]


def test_rrf_rewards_agreement():
    results = reciprocal_rank_fusion([{"id": "a"}, {"id": "b"}], [{"id": "b"}, {"id": "c"}])
    assert results[0]["id"] == "b"


def test_notebook_ids_use_path_and_skip_comments(tmp_path):
    docs = []
    for folder in ("Analysis", "DFT"):
        path = tmp_path / folder / "same.ipynb"
        path.parent.mkdir()
        path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["# overview"]}, {"cell_type": "code", "source": ["import numpy as np"]}]}))
        cells = extract_notebook_cells(str(path))
        assert len(cells) == 1
        assert "[Source:" in cells[0]["content"]
        docs.append(cells[0])
    assert docs[0]["id"] != docs[1]["id"]


def test_query_expansion_preserves_identifiers():
    query = "RDF and CUS with se_e2_a"
    expanded = expand_query(query)
    assert expanded.startswith(query)
    assert "radial distribution function" in expanded
    assert "coordinatively unsaturated sites" in expanded


def test_memory_backfill_preserves_quality_and_ids(tmp_path):
    import sqlite3
    path = tmp_path / "chat.db"
    response = "```python\nprint(1)\n```"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sessions (id INTEGER, title TEXT)")
        conn.execute("CREATE TABLE messages (id INTEGER, session_id INTEGER, role TEXT, content TEXT, agent_used TEXT, tool_calls_json TEXT)")
        conn.execute("INSERT INTO sessions VALUES (1, 'Test')")
        conn.executemany("INSERT INTO messages VALUES (?, 1, ?, ?, 'Assistant', ?)", [
            (1, "user", "run code", None),
            (2, "assistant", response, json.dumps({"memory_quality": {"execution_required": True, "exit_code": 0}})),
            (3, "user", "run failed code", None),
            (4, "assistant", response, json.dumps({"memory_quality": {"execution_required": True, "exit_code": 1}})),
        ])
    store = memory_store()
    store.index_turn(1, "run code", response, execution_required=True, exit_code=0)
    live_id = store.collection.upsert.call_args.kwargs["ids"]
    assert store.ingest_all_sessions(str(path)) == 1
    assert store.collection.upsert.call_args.kwargs["ids"] == live_id


def test_agentic_api_does_not_prepend_notebook(mocker):
    from backend.main import ChatRequest, chat_endpoint
    agent = mocker.patch("backend.main.AgenticRAGAgent").return_value
    resolver = mocker.patch("backend.main.resolve_prompt_context")
    chat_endpoint(ChatRequest(message="Analyze IrO2", mode="agentic"))
    assert agent.chat.call_args.kwargs["user_message"] == "Analyze IrO2"
    resolver.assert_not_called()
