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
