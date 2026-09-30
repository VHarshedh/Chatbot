import os
import pytest
import tempfile
import sqlite3
from unittest.mock import MagicMock, patch
from backend.rag.chat_memory import ChatMemoryStore, get_chat_memory_store

def test_clean_text():
    raw = (
        "<details class=\"thinking-details\"><summary>Thinking (10 words)</summary>\n"
        "Let me figure out the equation\n</details>\n\n"
        "Here is the final answer about anatase TiO2."
    )
    cleaned = ChatMemoryStore.clean_text(raw)
    assert "<details" not in cleaned
    assert "Thinking" not in cleaned
    assert "Here is the final answer about anatase TiO2." in cleaned

def test_format_memory_context():
    store = ChatMemoryStore()
    memories = [
        {
            "session_id": 1,
            "session_title": "Pt Pseudopotential Run",
            "user_snippet": "What pseudo to use for Pt?",
            "assistant_snippet": "Use Pt.pbe-n-kjpaw_psl.1.0.0.UPF.",
            "distance": 0.25
        }
    ]
    formatted = store.format_memory_context(memories)
    assert "--- RELEVANT PAST CHAT SESSIONS MEMORY ---" in formatted
    assert "[Session #1: Pt Pseudopotential Run]" in formatted
    assert "User: What pseudo to use for Pt?" in formatted
    assert "Assistant: Use Pt.pbe-n-kjpaw_psl.1.0.0.UPF." in formatted

def test_format_memory_context_empty():
    store = ChatMemoryStore()
    assert store.format_memory_context([]) == ""

def test_search_memory_filters_current_session(tmp_path):
    store = ChatMemoryStore(persist_dir=str(tmp_path))
    # Mock collection and embeddings to test filtering logic
    mock_coll = MagicMock()
    mock_coll.count.return_value = 2
    mock_coll.query.return_value = {
        "documents": [["Doc 1", "Doc 2"]],
        "metadatas": [[
            {"session_id": 10, "session_title": "Session 10", "user_snippet": "Q10", "assistant_snippet": "A10"},
            {"session_id": 11, "session_title": "Session 11", "user_snippet": "Q11", "assistant_snippet": "A11"},
        ]],
        "distances": [[0.3, 0.4]]
    }
    store.collection = mock_coll
    store.embeddings.embed_query = MagicMock(return_value=[0.1] * 768)

    # If current_session_id is 10, session 10 must be excluded
    results = store.search_memory("test query", current_session_id=10)
    assert len(results) == 1
    assert results[0]["session_id"] == 11

def test_search_memory_distance_threshold(tmp_path):
    store = ChatMemoryStore(persist_dir=str(tmp_path))
    mock_coll = MagicMock()
    mock_coll.count.return_value = 2
    mock_coll.query.return_value = {
        "documents": [["Doc Close", "Doc Far"]],
        "metadatas": [[
            {"session_id": 1, "session_title": "S1", "user_snippet": "Close", "assistant_snippet": "Close"},
            {"session_id": 2, "session_title": "S2", "user_snippet": "Far", "assistant_snippet": "Far"},
        ]],
        "distances": [[0.4, 1.8]]
    }
    store.collection = mock_coll
    store.embeddings.embed_query = MagicMock(return_value=[0.1] * 768)

    # Max distance = 1.0 -> should exclude Doc Far (1.8)
    results = store.search_memory("test query", current_session_id=99, max_distance=1.0)
    assert len(results) == 1
    assert results[0]["session_id"] == 1

def test_ingest_all_sessions(tmp_path):
    # Setup temporary SQLite database
    db_file = tmp_path / "test_chat.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, title TEXT)")
    cur.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id INTEGER, role TEXT, content TEXT, agent_used TEXT, timestamp TEXT)")
    
    cur.execute("INSERT INTO sessions (id, title) VALUES (1, 'First Session')")
    cur.execute("INSERT INTO messages (session_id, role, content, agent_used) VALUES (1, 'user', 'How to compute bandgap?', 'User')")
    cur.execute("INSERT INTO messages (session_id, role, content, agent_used) VALUES (1, 'assistant', 'Use QE or VASP.', 'AgenticRAG')")
    conn.commit()
    conn.close()

    store = ChatMemoryStore(persist_dir=str(tmp_path / "chroma"))
    store.embeddings.embed_documents = MagicMock(return_value=[[0.05] * 768])
    mock_coll = MagicMock()
    store.collection = mock_coll

    count = store.ingest_all_sessions(sqlite_path=str(db_file))
    assert count == 1
    assert mock_coll.upsert.called
