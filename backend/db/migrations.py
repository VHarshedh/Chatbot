import sqlite3
from backend.config import settings

def init_db():
    conn = sqlite3.connect(settings.SQLITE_PATH)
    cursor = conn.cursor()
    
    # Sessions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        title TEXT,
        last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Messages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        role TEXT,
        content TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        agent_used TEXT,
        tool_calls_json TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions (id)
    )
    """)
    
    # Jobs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        workflow_type TEXT,
        slurm_job_id TEXT,
        status TEXT,
        submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        params_json TEXT,
        output_path TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions (id)
    )
    """)
    
    # Parameters table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS parameters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER,
        key TEXT,
        value TEXT,
        unit TEXT,
        FOREIGN KEY (job_id) REFERENCES jobs (id)
    )
    """)
    
    # Artifacts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS artifacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER,
        type TEXT,
        path TEXT,
        description TEXT,
        FOREIGN KEY (job_id) REFERENCES jobs (id)
    )
    """)

    # Evaluations table (tracks model zero-shot benchmark scores and reports)
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions (id)
    )
    """)
    
    conn.commit()
    conn.close()
    print("Database initialized.")

if __name__ == "__main__":
    init_db()
