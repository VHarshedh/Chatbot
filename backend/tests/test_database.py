import pytest
import sqlite3
from backend.db.models import (
    execute_query,
    insert_session,
    insert_message,
    insert_evaluation,
    get_evaluations,
    get_prompts,
    get_prompt_by_id,
    insert_prompt,
    update_prompt,
    delete_prompt,
    reset_prompts,
    get_chat_history
)

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

def test_prompts_crud_and_reset(memory_db):
    # 1. get_prompts automatically seeds default prompts
    prompts = get_prompts()
    assert len(prompts) == 5
    first_id = prompts[0]["id"]

    # 2. get_prompt_by_id
    p = get_prompt_by_id(first_id)
    assert p is not None
    assert p["id"] == first_id

    # 3. insert_prompt
    new_p = insert_prompt(
        prompt_id="test_new_prompt",
        title="New Test Prompt",
        domain="Surface Analysis",
        prompt="Test prompt body"
    )
    assert new_p["id"] == "test_new_prompt"
    assert new_p["title"] == "New Test Prompt"
    assert len(get_prompts()) == 6

    # 4. update_prompt
    upd = update_prompt("test_new_prompt", title="Updated Title", domain="DFT")
    assert upd["title"] == "Updated Title"
    assert upd["domain"] == "DFT"
    assert upd["prompt"] == "Test prompt body"

    # 5. delete_prompt
    assert delete_prompt("test_new_prompt") is True
    assert get_prompt_by_id("test_new_prompt") is None
    assert delete_prompt("test_new_prompt") is False

    # 6. reset_prompts
    restored = reset_prompts()
    assert len(restored) == 5
    assert any(x["id"] == "interfacial_script_gen" for x in restored)

def test_get_chat_history(memory_db):
    assert get_chat_history(None) == []
    assert get_chat_history(9999) == []

    session_id = insert_session("Multi-turn Session")
    insert_message(session_id, "user", "Question 1")
    insert_message(session_id, "assistant", "Answer 1")
    insert_message(session_id, "user", "Question 2")
    insert_message(session_id, "assistant", "Answer 2")

    history = get_chat_history(session_id)
    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "Question 1"}
    assert history[1] == {"role": "assistant", "content": "Answer 1"}
    assert history[2] == {"role": "user", "content": "Question 2"}
    assert history[3] == {"role": "assistant", "content": "Answer 2"}

    # Limit check
    limited = get_chat_history(session_id, limit=2)
    assert len(limited) == 2
    assert limited[0] == {"role": "user", "content": "Question 2"}
    assert limited[1] == {"role": "assistant", "content": "Answer 2"}

def test_get_chat_history_cleans_thinking(memory_db):
    session_id = insert_session("Thinking Session")
    raw_assistant_content = (
        '<details class="thinking-details"><summary>Thinking</summary>\n'
        'Step 1: Check TiO2 structure\n'
        'Step 2: Formulate answer\n'
        '</details>\n\n'
        'Anatase TiO2 has tetragonal symmetry with space group I4_1/amd.'
    )
    insert_message(session_id, "user", "What is the space group of Anatase?")
    insert_message(session_id, "assistant", raw_assistant_content)

    history = get_chat_history(session_id)
    assert len(history) == 2
    assert history[0]["content"] == "What is the space group of Anatase?"
    assert "<details" not in history[1]["content"]
    assert "Anatase TiO2 has tetragonal symmetry" in history[1]["content"]


