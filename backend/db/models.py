import sqlite3
from typing import List, Dict, Any
import json
from backend.config import settings

def get_db_connection():
    conn = sqlite3.connect(settings.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def execute_query(query: str, params: tuple = ()) -> List[sqlite3.Row]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    results = cursor.fetchall()
    conn.close()
    return results

def insert_session(title: str) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (title) VALUES (?)", (title,))
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id

def insert_message(session_id: int, role: str, content: str, agent_used: str = None, tool_calls_json: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content, agent_used, tool_calls_json) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, agent_used, tool_calls_json)
    )
    conn.commit()
    conn.close()

def _ensure_evaluations_table(cursor):
    cursor.execute(
        """
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
        """
    )

def insert_evaluation(
    session_id: int,
    model_name: str,
    prompt: str,
    generated_code: str,
    score: float,
    grade: str,
    category_scores_json: str,
    evaluation_report_json: str,
    latency_seconds: float
) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_evaluations_table(cursor)
    cursor.execute(
        """
        INSERT INTO evaluations (
            session_id, model_name, prompt, generated_code, score, grade,
            category_scores_json, evaluation_report_json, latency_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id, model_name, prompt, generated_code, score, grade,
            category_scores_json, evaluation_report_json, latency_seconds
        )
    )
    conn.commit()
    eval_id = cursor.lastrowid
    conn.close()
    return eval_id

def get_evaluations(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_evaluations_table(cursor)
    cursor.execute(
        "SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
