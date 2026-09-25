import sqlite3
from typing import List, Dict, Any, Optional
import json
from backend.config import settings

INTERFACIAL_ANALYSIS_BENCHMARK_PROMPT = (
    "Write a complete, self-contained scientific Python script using ASE, NumPy, and Matplotlib to analyze "
    "an interfacial metal-oxide/water system (template: IrO2/H2O) from DeepMD .extxyz trajectories across multiple DP models. "
    "Your script must:\n"
    "1. Classify surface metal atoms (M_sur from topmost z-layer), bulk vs water oxygens (O_bulk vs O_w via stoichiometry), "
    "coordinatively unsaturated metal sites (CUS), and adsorbed vs bridge oxygens (O_adb, O_br).\n"
    "2. Compute water mass density profiles along the z-axis (g/cm^3) and pair radial distribution functions (RDFs: M_cus-O_w, O_br-H_w, O_br-O_w).\n"
    "3. Track surface hydroxylation / dissociation fractions (OH vs intact H2O) using neighbor distance cutoffs (~1.2 A), and compute the water survival probability P(t) over log-spaced time lags tau via continuous set intersections.\n"
    "4. Loop over multiple DP models, computing the cross-model mean and standard deviation (mean +/- std).\n"
    "5. Save individual plots for density, RDF, survival, and fraction, as well as a combined 2x2 multi-panel summary presentation figure (full_analysis_presentation.png).\n"
    "6. Write a summary_report.txt file with average residence times (integrated from P(t)) and dissociated/molecular fractions.\n"
    "Do not beat around the bush or provide conversational filler; get straight to the point. Output only the complete Python code block."
)

DEFAULT_BENCHMARK_PROMPTS = [
    {
        "id": "interfacial_script_gen",
        "domain": "Interfacial Script Evaluation",
        "title": "Full Interfacial Analysis Script Generator",
        "prompt": INTERFACIAL_ANALYSIS_BENCHMARK_PROMPT,
        "is_default": 1
    },
    {
        "id": "dft_cutoff",
        "domain": "DFT",
        "title": "Pt SCF Convergence Parameters",
        "prompt": "What are the recommended plane-wave kinetic energy cutoff (ecutwfc) and charge density cutoff (ecutrho) for Platinum (Pt) using a PBE PAW or ONCV pseudopotential in Quantum ESPRESSO? Please write a sample &SYSTEM namelist block.",
        "is_default": 1
    },
    {
        "id": "md_chaining",
        "domain": "Molecular Dynamics",
        "title": "DeePMD LAMMPS Interface Script",
        "prompt": "Write a minimal LAMMPS in.lammps input script for an NVT production run of an interfacial water system driven by a pair_style deepmd potential. Include velocity creation, thermostatting at 300K, and periodic restart writes.",
        "is_default": 1
    },
    {
        "id": "training_format",
        "domain": "DeepMD Training",
        "title": "Raw Data to Binary Sets",
        "prompt": "In DeePMD-kit, what is the exact layout of raw data files (coord.raw, box.raw, energy.raw, force.raw) and how are they converted into set.000 numpy binary datasets for training?",
        "is_default": 1
    },
    {
        "id": "analysis_kinetics",
        "domain": "Interfacial Analysis",
        "title": "Surface Sites & Survival Probability",
        "prompt": "Explain how coordinatively unsaturated (CUS) metal sites and adsorbed water oxygens (O_adb) are structurally classified on a metal oxide surface, and write the mathematical definition of the water survival probability P(t).",
        "is_default": 1
    }
]

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

def insert_message(
    session_id: int,
    role: str,
    content: str,
    agent_used: str = None,
    tool_calls_json: str = None,
    images_json: str = None
):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(messages)")
    columns = [col[1] for col in cursor.fetchall()]
    if "images_json" not in columns:
        try:
            cursor.execute("ALTER TABLE messages ADD COLUMN images_json TEXT")
            conn.commit()
        except Exception:
            pass

    cursor.execute(
        "INSERT INTO messages (session_id, role, content, agent_used, tool_calls_json, images_json) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, role, content, agent_used, tool_calls_json, images_json)
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

def _ensure_prompts_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prompts (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            domain TEXT NOT NULL,
            prompt TEXT NOT NULL,
            is_default INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute("SELECT COUNT(*) FROM prompts")
    count = cursor.fetchone()[0]
    if count == 0:
        for p in DEFAULT_BENCHMARK_PROMPTS:
            cursor.execute(
                """
                INSERT INTO prompts (id, title, domain, prompt, is_default)
                VALUES (?, ?, ?, ?, ?)
                """,
                (p["id"], p["title"], p["domain"], p["prompt"], p.get("is_default", 1))
            )

def get_prompts() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_prompts_table(cursor)
    cursor.execute("SELECT * FROM prompts ORDER BY created_at ASC")
    rows = cursor.fetchall()
    conn.commit()
    conn.close()
    return [dict(row) for row in rows]

def get_prompt_by_id(prompt_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_prompts_table(cursor)
    cursor.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,))
    row = cursor.fetchone()
    conn.commit()
    conn.close()
    return dict(row) if row else None

def insert_prompt(prompt_id: str, title: str, domain: str, prompt: str, is_default: int = 0) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_prompts_table(cursor)
    cursor.execute(
        """
        INSERT INTO prompts (id, title, domain, prompt, is_default)
        VALUES (?, ?, ?, ?, ?)
        """,
        (prompt_id, title, domain, prompt, is_default)
    )
    conn.commit()
    cursor.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row)

def update_prompt(
    prompt_id: str,
    title: Optional[str] = None,
    domain: Optional[str] = None,
    prompt: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_prompts_table(cursor)
    cursor.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,))
    existing = cursor.fetchone()
    if not existing:
        conn.close()
        return None
    
    new_title = title if title is not None else existing["title"]
    new_domain = domain if domain is not None else existing["domain"]
    new_prompt = prompt if prompt is not None else existing["prompt"]
    
    cursor.execute(
        """
        UPDATE prompts
        SET title = ?, domain = ?, prompt = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (new_title, new_domain, new_prompt, prompt_id)
    )
    conn.commit()
    cursor.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row)

def delete_prompt(prompt_id: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_prompts_table(cursor)
    cursor.execute("SELECT id FROM prompts WHERE id = ?", (prompt_id,))
    existing = cursor.fetchone()
    if not existing:
        conn.close()
        return False
    cursor.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    conn.commit()
    conn.close()
    return True

def reset_prompts() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_prompts_table(cursor)
    cursor.execute("DELETE FROM prompts")
    for p in DEFAULT_BENCHMARK_PROMPTS:
        cursor.execute(
            """
            INSERT INTO prompts (id, title, domain, prompt, is_default)
            VALUES (?, ?, ?, ?, ?)
            """,
            (p["id"], p["title"], p["domain"], p["prompt"], p.get("is_default", 1))
        )
    conn.commit()
    cursor.execute("SELECT * FROM prompts ORDER BY created_at ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
