import pytest
import sqlite3
from backend.db.models import execute_query, insert_session, insert_message, insert_evaluation, get_evaluations

@pytest.fixture
def memory_db(tmp_path):
    db_file = tmp_path / "test.db"
    import backend.config
    backend.config.settings.SQLITE_PATH = str(db_file)
    
    # Initialize schema
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        role TEXT,
        content TEXT,
        agent_used TEXT,
        tool_calls_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions (id)
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        model_name TEXT,
        prompt TEXT,
        generated_code TEXT,
        score REAL,
        grade TEXT,
        category_scores_json TEXT,
        evaluation_report_json TEXT,
        latency_seconds REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()
    yield

def test_insert_session(memory_db):
    session_id = insert_session("Test Session")
    assert session_id > 0
    
    # verify
    res = execute_query("SELECT title FROM sessions WHERE id = ?", (session_id,))
    assert len(res) == 1
    assert res[0]["title"] == "Test Session"

def test_insert_message(memory_db):
    session_id = insert_session("Test Session")
    insert_message(session_id, "user", "Hello DB", "QAAgent", "[]")
    
    res = execute_query("SELECT * FROM messages WHERE session_id = ?", (session_id,))
    assert len(res) == 1
    assert res[0]["role"] == "user"
    assert res[0]["content"] == "Hello DB"
    assert res[0]["agent_used"] == "QAAgent"

def test_insert_and_get_evaluations(memory_db):
    eval_id = insert_evaluation(
        session_id=1,
        model_name="batiai/gemma4-e2b:q4",
        prompt="Test prompt",
        generated_code="import numpy as np",
        score=95.0,
        grade="A",
        category_scores_json='{"syntax": 15}',
        evaluation_report_json='{"passed": []}',
        latency_seconds=1.5
    )
    assert eval_id > 0

    evals = get_evaluations(limit=10)
    assert len(evals) == 1
    assert evals[0]["id"] == eval_id
    assert evals[0]["score"] == 95.0
    assert evals[0]["grade"] == "A"
