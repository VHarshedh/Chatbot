import sqlite3
from typing import List, Dict, Any, Optional
import json
import re
from backend.config import settings

INTERFACIAL_ANALYSIS_BENCHMARK_PROMPT = r"""Write a complete, self-contained, executable scientific Python script using ASE (`ase.io.read`, `ase.neighborlist.neighbor_list`), NumPy, and Matplotlib (`Agg` backend) to analyze an interfacial metal-oxide/water trajectory (`MO2/H2O`, stoichiometry `s = 2`, metal symbol `m = 'Ti'` or `'Ir'`) across Deep Potential (DP) ensemble models (`dp = 3`, `modelz = list(range(1, dp + 1))`).

Your script must implement all 6 stages below in full without placeholders or mock data:

1. DATA LOADING & ENSEMBLE FALLBACK:
   - Loop over `mod_idx in range(1, dp + 1)`. Try loading `f"{base_dir}/{mod_idx}/iro2_water_concat.extxyz"` with `index=slice(0, None, 10)`.
   - If the per-model file is not found, fall back to the pre-loaded `traj` variable in globals/locals (or load `traj_file` / `'/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz'`), wrapping a single frame into `[traj]` if needed: `frames = traj if isinstance(traj, list) else [traj]`.
   - Auto-detect the metal element symbol `m` (e.g., `'Ir'` or `'Ti'`) from `atoms0.get_chemical_symbols()` (any element other than `'O'` and `'H'`).

2. INTERFACIAL SITE CLASSIFICATION (Frame 0):
   - Wrap positions and enable PBC: `atoms0.wrap(); atoms0.set_pbc(True)`.
   - Surface Metal (`m_sur_index`): metal atoms with $z \ge \max(m_z) - 1.5$ Å.
   - Bulk vs. Water Oxygen (`o_bul_index` vs. `o_w_index`): sort all oxygen $z$-coordinates; the lowest $s \times N_{\text{metal}}$ oxygens ($s = 2$) are bulk lattice oxygens (`o_bul_index`), and the remaining oxygens are water oxygens (`o_w_index`).
   - CUS Metal (`m_cus_index`) & Adsorbed Water Oxygen (`o_adb_index`): pairs $(i \in m_{\text{sur}}, j \in O_w)$ with `atoms0.get_distance(i, j, mic=True) <= 2.5` Å.
   - Surface Bridge Oxygen (`o_sur_index`): the top `len(m_cus_index)` oxygens within the bulk oxide z-sorted list (`o_z[s * len(m_z) - len(m_cus_index) : s * len(m_z)]`).

3. WATER MASS DENSITY PROFILE & PAIR RDFs:
   - Z-Mass Density Profile (`bin_width = 0.10` Å): histogram $z$-coordinates of `O_w` and `H_w` into `bins`, convert counts to physical mass density ($\text{g/cm}^3$) using water molar mass $18.01528\text{ g/mol}$, Avogadro's number $6.023 \times 10^{23}$, and slab bin volume `volume = cell[0][0] * cell[1][1] * bin_width`:
     `mass_dens_ow = ((freq_ow / 6.023e23) * 18.01528) / (volume * 1e-24)`
     `mass_dens_hw = ((freq_hw / (2.0 * 6.023e23)) * 18.01528) / (volume * 1e-24)`
   - Pair Radial Distribution Functions ($g(r)$, `dr = 0.05` Å, $r \in [0, 6.5]$ Å): compute pair distance histograms normalized by spherical shell volume $4\pi r^2 dr$ for the 3 interfacial pairs:
     (a) $M_{\text{cus}}\text{--}O_w$ (`g_mow`), (b) $O_{\text{br}}\text{--}H_w$ (`g_osh`), and (c) $O_{\text{br}}\text{--}O_w$ (`g_osow`).

4. SURFACE HYDROXYLATION FRACTIONS & CONTINUOUS SURVIVAL PROBABILITY P(t):
   - For each frame, use `i_idx, _, _ = neighbor_list('ijd', frame, {('O', 'H'): 1.2}, self_interaction=False)` and `o_h_counts = np.bincount(i_idx, minlength=len(frame))`.
   - For each `idx in o_adb_idx`: classify `o_h_counts[idx] == 1` as Dissociated (`OH`/`OD`, `oh_frame`) and `o_h_counts[idx] == 2` as Molecular (`H2O`/`D2O`, `h2o_frame`). Record instantaneous percentages `frac_oh` and `frac_h2o`.
   - Convert frame lists to sets `oh_sets = [set(f) for f in indx_oh]` and `h2o_sets = [set(f) for f in indx_h2o]`.
   - Over time lags `tau_range = np.linspace(0, num_f - 1, min(101, max(2, num_f)), dtype=int)`, compute continuous survival probability $P(\tau)$ via set intersection across all intermediate frames:
     `len(set.intersection(*oh_sets[j:j+tau+1])) / len(oh_sets[j])` and `len(set.intersection(*h2o_sets[j:j+tau+1])) / len(h2o_sets[j])`.
   - Compute mean residence time $\tau = \frac{\int t P(t) dt}{\int P(t) dt}$ using `trapz_fn = getattr(np, 'trapezoid', np.trapz)`.

5. ENSEMBLE AVERAGING & INTERMEDIATE `.npz` STORAGE:
   - Compute cross-model `np.mean(..., axis=0)` and `np.std(..., axis=0)` across the `dp` models.
   - Save `simulation_results.npz` (keys: `m_symbol`, `ow_y`, `hw_y`, `bins_z`, `rd`, `g_mow`, `g_osh`, `g_osow`) and `survival_fraction_results.npz` (keys: `m_symbol`, `times_surv`, `surv_oh`, `surv_h2o`, `frac_oh`, `frac_h2o`, `dt_ns`).

6. PUBLICATION FIGURES & SUMMARY REPORT:
   - Generate and save a 2x2 presentation figure `full_analysis_presentation.png` (`figsize=(14, 11)`, `dpi=300`) with shaded $\pm 1\sigma$ uncertainty bands (`ax.fill_between(...)`) for:
     (A) Water Mass Density Profile, (B) Pair RDFs, (C) Survival Probability $P(t)$, and (D) Dissociation Fraction (%). Call `plt.close()` after saving.
   - Write `summary_report.txt` containing the ensemble mean $\pm$ std for dissociated fraction (`OD`/`OH` %), molecular fraction (`D2O`/`H2O` %), and residence times $\tau_{\text{OD}}$ and $\tau_{\text{D2O}}$ in ns.

Output ONLY the complete Python code block enclosed in ```python ... ```."""

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
        "prompt": "Specify the recommended plane-wave kinetic energy cutoff (ecutwfc in Ry) and charge-density cutoff (ecutrho in Ry) for fcc Platinum (Pt, nat=4, ntyp=1) using the ONCV norm-conserving pseudopotential (Pt_ONCV_PBE-1.2.upf, ecutrho = 4 * ecutwfc) versus a PAW pseudopotential (ecutrho = 8 to 12 * ecutwfc) in Quantum ESPRESSO. Provide a complete, syntactically valid pw.in input file including &CONTROL, &SYSTEM (ibrav=0, Marzari-Vanderbilt cold smearing degauss=0.02 Ry, nbnd=44), &ELECTRONS (conv_thr=1.0d-8), ATOMIC_SPECIES, CELL_PARAMETERS (angstrom, a=3.943 Å), ATOMIC_POSITIONS (angstrom), and K_POINTS automatic.",
        "is_default": 1
    },
    {
        "id": "md_chaining",
        "domain": "Molecular Dynamics",
        "title": "DeePMD LAMMPS Interface Script",
        "prompt": "Write a complete LAMMPS input script (in.lammps) and a companion SLURM dependency-chaining bash script (submit_from.sh) for an NVT production run of liquid water at 300 K driven by an ensemble of 3 DeepMD frozen graphs (graph1.pb, graph2.pb, graph3.pb). Include: (1) plugin load libdeepmd_lmp.so, metal units, 3D periodic boundaries, atomic style, (2) conditional read_data vs read_restart lammps.restart, (3) masses for H (2.014) and O (15.999), (4) pair_style deepmd with out_file md.out out_freq 100 and pair_coeff * *, (5) fix 1 all nvt temp 300 300 0.1 with timestep 0.0005 ps, (6) timer timeout and write_restart lammps.restart, and (7) a bash loop chaining batches via sbatch --parsable --dependency=afterok:$PREV_JOBID.",
        "is_default": 1
    },
    {
        "id": "training_format",
        "domain": "DeepMD Training",
        "title": "Raw Data to Binary Sets",
        "prompt": "Explain the exact frame-by-frame dimensional layout and physical units of DeePMD-kit raw files (coord.raw [N_frames x 3*N_atoms in Å], box.raw [N_frames x 9 in Å], energy.raw [N_frames x 1 in eV], force.raw [N_frames x 3*N_atoms in eV/Å], virial.raw [N_frames x 9 in eV], and 0-indexed type.raw [1 x N_atoms] + type_map.raw). Then write a complete, self-contained Python script that splits raw files into disjoint train/Test sets (without data leakage), converts chunks of 2000 frames into set.000/*.npy float32 binary arrays, and outputs a valid DeePMD-kit se_e2_a input.json configuration.",
        "is_default": 1
    },
    {
        "id": "analysis_kinetics",
        "domain": "Interfacial Analysis",
        "title": "Surface Sites & Survival Probability",
        "prompt": "Formulate the structural classification criteria and kinetic equations for metal-oxide/water interfaces (IrO2/H2O and Anatase TiO2/H2O): (1) mathematical definitions and z-coordinate / radial cutoffs for surface metal atoms (M_sur, z >= z_max - 1.5 Å), bulk vs water oxygens (s * N_metal), coordinatively unsaturated sites (CUS, r_M-Ow <= 2.5 Å), bridge oxygens (O_br), and dissociated OH vs molecular H2O (r_O-H <= 1.2 Å); (2) the continuous time-correlation survival probability P(t) via multi-frame set intersections over [t_0, t_0 + tau]; and (3) mean residence time tau via trapezoidal integration, alongside a concise Python implementation using ase.neighborlist and numpy.",
        "is_default": 1
    }
]

def get_db_connection():
    conn = sqlite3.connect(settings.SQLITE_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
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
    cursor.execute("PRAGMA table_info(sessions)")
    session_columns = [col[1] for col in cursor.fetchall()]
    if "last_message_at" not in session_columns:
        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN last_message_at TIMESTAMP")
            conn.commit()
        except Exception:
            pass
    cursor.execute(
        "UPDATE sessions SET last_message_at = CURRENT_TIMESTAMP WHERE id = ?",
        (session_id,)
    )
    conn.commit()
    conn.close()

def get_chat_history(
    session_id: Optional[int],
    limit: int = 20,
    before_message_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves previous conversational messages for a session formatted for LLM context.
    Returns list of dicts: [{"role": "user"|"assistant", "content": "..."}] in chronological order.
    Cleans out collapsible UI artifacts like thinking traces from assistant responses to avoid prompt bloat.
    """
    if not session_id:
        return []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        if not cursor.fetchone():
            conn.close()
            return []

        cursor.execute("PRAGMA table_info(messages)")
        columns = [col[1] for col in cursor.fetchall()]
        has_images_col = "images_json" in columns

        if before_message_id is not None:
            select_cols = "id, role, content, images_json" if has_images_col else "id, role, content"
            cursor.execute(
                f"""
                SELECT {select_cols}
                FROM messages
                WHERE session_id = ? AND id < ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, before_message_id, limit)
            )
        else:
            select_cols = "id, role, content, images_json" if has_images_col else "id, role, content"
            cursor.execute(
                f"""
                SELECT {select_cols}
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit)
            )
        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in reversed(rows):
            role = row["role"]
            content = row["content"] or ""

            # Clean thinking block from assistant messages if present so it doesn't inflate context
            if role == "assistant" and content:
                cleaned = re.sub(r'<details class="thinking-details">.*?</details>', '', content, flags=re.DOTALL).strip()
                if cleaned:
                    content = cleaned

            msg = {"role": role, "content": content}

            if role == "user" and has_images_col:
                imgs_raw = row["images_json"]
                if imgs_raw:
                    try:
                        imgs = json.loads(imgs_raw)
                        if imgs and isinstance(imgs, list):
                            msg["images"] = imgs
                    except Exception:
                        pass
            history.append(msg)
        return history
    except Exception as e:
        print(f"Notice: Failed to fetch chat history for session {session_id}: {e}")
        return []

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
