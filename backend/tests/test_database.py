import pytest
import sqlite3
from backend.db.models import execute_query, insert_session, insert_message

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
