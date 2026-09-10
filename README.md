# Scientific Computing AI Chatbox & Workflow Automation

An intelligent, multi-agent AI assistant and workflow automation engine tailored for computational materials science and chemistry. This platform bridges **Density Functional Theory (DFT)**, **Deep Potential Molecular Dynamics (DeePMD/LAMMPS)**, **Machine Learning Potential Training**, and **Interfacial Trajectory Analysis** with an Agentic **Hybrid RAG** architecture, **sandboxed code execution**, and **HPC SLURM orchestration**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture & Key Capabilities](#architecture--key-capabilities)
  - [1. Multi-Agent System](#1-multi-agent-system)
  - [2. Live Token Streaming & Code Generation Studio](#2-live-token-streaming--code-generation-studio)
  - [3. Scientific LaTeX Mathematical Formula Rendering (KaTeX)](#3-scientific-latex-mathematical-formula-rendering-katex)
  - [4. Zero-Shot Interfacial Script Evaluation (100-Point Rubric)](#4-zero-shot-interfacial-script-evaluation-100-point-rubric)
  - [5. Hybrid RAG (ChromaDB + Kùzu Knowledge Graph)](#5-hybrid-rag-chromadb--kùzu-knowledge-graph)
  - [6. Secure Sandboxed Execution](#6-secure-sandboxed-execution)
  - [7. HPC & SLURM Orchestration](#7-hpc--slurm-orchestration)
  - [8. Domain Modules](#8-domain-modules)
- [Repository Structure](#repository-structure)
- [Prerequisites & System Requirements](#prerequisites--system-requirements)
- [Step-by-Step Installation & Setup](#step-by-step-installation--setup)
- [Execution Guide: What to Run](#execution-guide-what-to-run)
  - [Phase 1: Environment & Services Setup](#phase-1-environment--services-setup)
  - [Phase 2: Database & RAG Initialization](#phase-2-database--rag-initialization)
  - [Phase 3: Launching the Backend API](#phase-3-launching-the-backend-api)
  - [Phase 4: Running the AI Agents](#phase-4-running-the-ai-agents)
  - [Phase 5: Running Domain Jupyter Notebooks](#phase-5-running-domain-jupyter-notebooks)
- [Configuration Reference](#configuration-reference)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Troubleshooting & Common Issues](#troubleshooting--common-issues)

---

## Overview

Computational workflows in materials science (e.g., Quantum ESPRESSO, DeePMD-kit, LAMMPS, ASE) often require tedious parameter sweeps, custom script generation, restart-linked batch submissions on HPC clusters, and complex trajectory post-processing.

This repository provides a unified software stack:
1. **Domain Pipelines**: Pre-configured notebooks and scripts for DFT convergence sweeps, DeePMD model training, chained long-run molecular dynamics, and water/interface trajectory analysis.
2. **AI Backend (`backend/`)**: A FastAPI service powering specialized LLM agents (powered locally via Ollama) capable of domain question-answering, writing scientific analysis code, running it in an isolated Docker sandbox with automated error correction, and managing HPC jobs over SSH.

---

## Architecture & Key Capabilities

```
                  +----------------------------------------------+
                  |               User / Client                  |
                  +----------------------+-----------------------+
                                         |
                                         v
                         +-------------------------------+
                         |      FastAPI Backend API      |
                         |        (backend/main.py)      |
                         +---------------+---------------+
                                         |
                +------------------------+------------------------+
                |                                                 |
                v                                                 v
    +-----------------------+                         +-----------------------+
    |       QA Agent        |                         |    Analysis Agent     |
    |  Domain Q&A with RAG  |                         | Code Gen + Execution  |
    +-----------+-----------+                         +-----------+-----------+
                |                                                 |
                +--------------------+  +-------------------------+
                                     |  |
                                     v  v
                         +---------------------------+
                         |     Hybrid Retriever      |
                         +-------------+-------------+
                                       |
                     +-----------------+-----------------+
                     |                                   |
                     v                                   v
          +--------------------+               +--------------------+
          |  ChromaDB Vectors  |               | Kùzu Knowledge     |
          |  (Dense Embeddings)|               | Graph (Cypher)     |
          +--------------------+               +--------------------+
                                                         |
                                 +-----------------------+-----------------------+
                                 |                                               |
                                 v                                               v
                     +-----------------------+                       +-----------------------+
                     | Docker Sandbox        |                       | HPC / SLURM Tools     |
                     | (Isolated Execution)  |                       | (SSH Job Submission)  |
                     +-----------------------+                       +-----------------------+
```

### 1. Multi-Agent System
- **QA Agent (`backend/agents/qa_agent.py`)**: Answers scientific computing and simulation queries using grounded context from the hybrid retrieval system.
- **Analysis Agent (`backend/agents/analysis_agent.py`)**: Generates scientific Python analysis scripts (using ASE, NumPy, Pandas, Matplotlib), executes them within a secure Docker sandbox, inspects standard error upon failure, queries web search (DuckDuckGo) for fixes if needed, and iteratively self-corrects up to 3 attempts.
- **Single-Shot Evaluation Agent (`backend/agents/baseline_agent.py`)**: Direct evaluation baseline for `batiai/gemma4-e2b:q4` with zero external tools and zero RAG, equipped with real-time SSE token streaming.

### 2. Live Token Streaming & Code Generation Studio
- **Server-Sent Events (SSE) Streaming**: Tokens are streamed line-by-line via `POST /api/chat/stream` and `POST /api/evaluate_zero_shot/stream` with `Cache-Control: no-cache` and unbuffered transmission. Eliminates request timeouts and perceived delays.
- **Reasoning Stream Drawer**: Automatically streams the model's internal thinking process into an interactive collapsible accordion (`🧠 Thinking Process (X words)`) with live word count tracking.
- **Live Code Generation Frame**:
  - Virtual code-fence recovery renders generating Python scripts inside an active IDE container in real-time as tokens arrive.
  - Features language identification pills (`🐍 PYTHON`), a pulsing status badge (`● Generating python code...`), and an active typewriter cursor positioned at the tip of the code.
  - Seamless one-click copy button (`Copy`) upon completion.

### 3. Scientific LaTeX Mathematical Formula Rendering (KaTeX)
- **Mathematical Formula Typography**: Full mathematical typesetting powered by KaTeX embedded in the frontend, rendering equations in computational materials science with high clarity.
- **Supported Syntaxes**:
  - **Inline Math**: `$ ... $` and `\( ... \)` for physical variables, subscripts, partial charges, and Greek symbols (e.g., $N_{\text{pairs}}$, $g(r)$, $4\pi r^2 dr$, $\text{H}_2\text{O}$, $\delta^+$, $\delta^-$, $\sqrt{\sum \Delta x^2}$).
  - **Display Math**: `$$ ... $$` and `\[ ... \]` for centered block equations, fractions, and summations (e.g., $$g(r) \propto \frac{\text{Number of pairs in shell}}{\text{Total number of pairs}}$$).
- **Anti-Corruption Tokenizer**: Mathematical formulas are isolated using unique placeholder tokens prior to markdown processing, preventing asterisks, underscores, and paragraph tags from corrupting complex formulas.
- **Responsive Styling**: Display formulas feature centered dark cards with horizontal scroll capability for lengthy formulas and automatic DOM hydration.

### 4. Zero-Shot Interfacial Script Evaluation (100-Point Rubric)
- **Automated Rubric Evaluator (`backend/evaluation/script_evaluator.py`)**: Measures zero-shot script accuracy against a 100-point rubric with automated AST syntax validation:
  1. **Syntax & AST Validity (15 pts)**: Parses code using Python's `ast` module; checks for `NameError`, indentation errors, or unclosed brackets.
  2. **Library Imports (15 pts)**: Verifies presence of `ase` (or `ase.io`), `numpy`, `matplotlib.pyplot`, and `scipy`.
  3. **Interfacial Structural Classification (25 pts)**: Validates identification of surface metal atoms ($M_{\text{sur}}$), bulk vs water oxygens ($O_{\text{bulk}}, O_w$), coordinatively unsaturated sites (CUS), and bridge oxygens ($O_{\text{br}}$).
  4. **Analytical Physics & Kinetics (25 pts)**: Scores water mass density profile along the z-axis, pair radial distribution functions (RDF), residence/survival probability $P(t)$, and hydroxyl fraction calculation.
  5. **Multi-Model DP Averaging (10 pts)**: Verifies loops across DeePMD potential ensemble directories and standard deviation / error band calculation.
  6. **Artifact Generation & Reporting (10 pts)**: Checks for individual metric plots, combined 2×2 summary figure (`plt.subplots(2, 2)`), and quantitative summary report printout.
- **Letter Grades**: **A** (≥90%), **B** (≥80%), **C** (≥70%), **D** (≥60%), **F** (<60%).
- **Interactive Evaluation Card**: Displays radar category scores, passed/failed checklist badges, and latency benchmarks directly in the UI.

### 5. Hybrid RAG (ChromaDB + Kùzu Knowledge Graph)
- **Vector Search (`backend/rag/vectorstore.py`)**: Chunks and embeds all workspace markdown manuals and Python scripts using Ollama (`nomic-embed-text`) stored in a persistent ChromaDB collection.
- **Knowledge Graph (`backend/rag/knowledge_graph.py`)**: An embedded Kùzu graph database holding structured entities:
  - `Material` (e.g. `Pt`, CIF paths)
  - `Pseudopotential` (e.g. `Pt_ONCV_PBE-1.2.upf`, type, exchange-correlation)
  - `System` (e.g. `Water`, type mappings, simulation temperatures)
  - Relationships: `(:Material)-[:has_pseudo]->(:Pseudopotential)` and `(:Material)-[:part_of]->(:System)`.
- **Hybrid Retriever (`backend/rag/hybrid_retriever.py`)**: Merges dense semantic vector similarity search with structured Cypher graph queries to produce accurate context.

### 6. Secure Sandboxed Execution
- **Docker Sandbox (`backend/sandbox/`)**: Executes generated Python or Bash scripts inside an isolated container (`chatbox-sandbox:latest`):
  - `network_mode="none"` (disabled internet access during code execution).
  - `mem_limit="2g"` and `pids_limit=50` (prevents memory exhaustion and fork bombs).
  - `read_only=True` root filesystem with temporary read-write workspace (`/tmp/workspace`).
  - Read-only data mounting for trajectory files (`/mnt/data/...`).
  - Strict 120-second timeout.

### 7. HPC & SLURM Orchestration
- **SLURM Tools (`backend/tools/slurm_tools.py`)**: Handles SSH communication with HPC clusters to execute `sbatch`, track job states via `squeue` and `sacct`, inspect `slurm-*.out` log files, and cancel jobs with `scancel`. Includes a local fallback mock for testing when `HPC_HOST="localhost"`.

### 8. Domain Modules
- **DFT Convergence (`DFT/` & `backend/tools/dft_tools.py`)**: Generates Quantum ESPRESSO (`pw.x`) input files (`pw.in`) from CIF structures, sets up cutoff/k-point sweep directories, and plots total energy convergence.
- **DeePMD Training (`Training/` & `backend/tools/training_tools.py`)**: Converts `.raw` format files to `set.*` binary arrays, creates train/test splits, generates `input.json` model configurations, and prepares SLURM GPU training scripts across multiple random seeds.
- **Molecular Dynamics (`Molecular Dynamics/` & `backend/tools/md_tools.py`)**: Sets up restart-linked batch MD runs in LAMMPS with DeePMD potentials and generates `submit_from.sh` to chain batch dependencies via SLURM `afterok`.
- **Trajectory Analysis (`Analysis/` & `backend/tools/analysis_tools.py`)**: Post-processes `.extxyz` trajectories of metal-oxide/water interfaces, classifies surface/bulk species (CUS, bridge oxygen, adsorbed water), computes z-density profiles, pair RDFs, survival probabilities, and surface hydroxylation fractions.

---

## Repository Structure

```
ChatBot/
├── Analysis/                                    # Interfacial trajectory analysis
│   ├── README_Water_Interface_Analysis.md      # Detailed documentation of analysis pipeline
│   ├── Water_Density_RDF_Survival_Probability_...ipynb  # Main analysis notebook
│   └── iro2_water_concat.extxyz                # Sample interface trajectory (extxyz format)
│
├── DFT/                                         # Density Functional Theory workflows
│   ├── README_EnergyCutoffAndKpoints.md        # Documentation for QE convergence sweeps
│   ├── EnergyCutoffandKpointsScript.ipynb      # Main QE sweep & convergence notebook
│   ├── Pt.cif                                  # Platinum crystal structure file
│   └── energycutoff/                           # Generated sweep directories
│
├── Molecular Dynamics/                          # Production MD pipelines
│   ├── README_MD_Simulation_Script.md          # Documentation for chunked MD workflows
│   ├── MD_Simulation_Script.ipynb              # Batch-chaining MD generator notebook
│   ├── Starting_Configurations/                # Starting .lmp LAMMPS structure files
│   ├── Frozen_Graphs/                          # Frozen DeePMD potential graphs (.pb)
│   └── Long_Run/                               # Generated batch runs and submission scripts
│
├── Training/                                    # DeePMD-kit potential training
│   ├── README_Training_Script.md               # Documentation for raw dataset splitting & training
│   ├── Training_Script.ipynb                   # Dataset preparation & job generator notebook
│   ├── Raw_Files/                              # Raw DeepMD dataset directories (*_Water)
│   └── DP_Models/                              # Generated seeded training directories
│
├── backend/                                     # AI Engine, APIs, RAG, and Tools
│   ├── agents/                                 # Multi-Agent logic
│   │   ├── analysis_agent.py                   # Self-correcting scientific Python agent
│   │   └── qa_agent.py                         # Hybrid RAG scientific Q&A assistant
│   ├── db/                                     # Relational storage & state tracking
│   │   ├── migrations.py                       # SQLite schema initialization (sessions, jobs, artifacts)
│   │   └── models.py                           # SQLite helper queries
│   ├── rag/                                    # Retrieval Augmented Generation
│   │   ├── embeddings.py                       # Ollama embedding wrapper
│   │   ├── vectorstore.py                      # ChromaDB vector store & workspace ingestion
│   │   ├── knowledge_graph.py                  # Kùzu graph DB schema and domain facts
│   │   └── hybrid_retriever.py                 # Fused vector + graph retrieval
│   ├── sandbox/                                # Secure execution environment
│   │   ├── Dockerfile                          # Debian + QE + LAMMPS + Python container
│   │   └── executor.py                         # Docker execution manager with resource limits
│   ├── tools/                                  # Tool suite for agents and automation
│   │   ├── analysis_tools.py                   # Trajectory loading, RDF, density, survival analysis
│   │   ├── dft_tools.py                        # CIF parsing, QE input generation, convergence sweeps
│   │   ├── md_tools.py                         # LAMMPS input generator & SLURM batch chaining
│   │   ├── training_tools.py                   # DeePMD raw data parsing & input.json generation
│   │   ├── slurm_tools.py                      # SSH-based SLURM job submission & monitoring
│   │   ├── sandbox_tools.py                    # High-level sandbox execution helper
│   │   └── web_tools.py                        # DuckDuckGo search & URL scraper
│   ├── tests/                                  # Test suite (>= 95% coverage requirement)
│   │   ├── integration/                        # Live tests requiring Docker/Ollama
│   │   │   └── test_live.py
│   │   ├── test_agents.py                      # Agent unit tests
│   │   ├── test_dft_tools.py                   # DFT tool unit tests
│   │   ├── test_md_tools.py                    # MD tool unit tests
│   │   ├── test_training_tools.py              # Training tool unit tests
│   │   ├── test_analysis_tools.py              # Analysis tool unit tests
│   │   ├── test_rag.py                         # VectorStore & KnowledgeGraph tests
│   │   ├── test_sandbox_executor.py            # Sandbox executor tests
│   │   ├── test_sandbox_tools.py               # Sandbox tool wrapper tests
│   │   ├── test_slurm_tools.py                 # SLURM tool tests
│   │   ├── test_database.py                    # SQLite migrations and models tests
│   │   └── test_web_tools.py                   # Web search tool tests
│   ├── config.py                               # Central Pydantic settings
│   ├── main.py                                 # FastAPI application entry point
│   ├── pytest.ini                              # Pytest configuration and coverage flags
│   └── requirements.txt                        # Core Python dependencies
│
├── frontend/                                    # Interactive Web Chat Interface
│   ├── index.html                              # Single-page UI with benchmark cards
│   ├── style.css                               # Glassmorphic dark-mode design system
│   └── app.js                                  # Client-side chat and session controller
│
└── README.md                                    # Root documentation (this file)
```

---

## Prerequisites & System Requirements

1. **Operating System**: Windows (PowerShell/CMD), Linux, or macOS.
2. **Python**: Version 3.10, 3.11, or 3.12 (Python 3.12 recommended).
3. **Ollama**: Required for local LLM inference and embeddings.
   - Install from [ollama.com](https://ollama.com).
   - Models required:
     - LLM: `batiai/gemma4-e2b:q4` (or any configured model)
     - Embeddings: `nomic-embed-text`
4. **Docker Desktop / Docker Engine** *(Optional but required for sandboxed code execution)*:
   - Must be running if you want the `AnalysisAgent` or `sandbox_tools` to execute generated Python scripts inside the container.
5. **Jupyter Notebook / JupyterLab**:
   - Required to interactively run the notebooks in `DFT/`, `Training/`, `Molecular Dynamics/`, and `Analysis/`.

---

## Step-by-Step Installation & Setup

### 1. Activate Python Environment
If you already have a virtual environment set up in the parent directory:
```powershell
# Windows PowerShell
..\venv\Scripts\Activate

# Linux / macOS
source ../venv/bin/activate
```

Or create and activate a new virtual environment:
```bash
python -m venv venv
# Activate on Windows:
.\venv\Scripts\Activate
# Activate on Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
Install the core dependencies:
```bash
pip install -r backend/requirements.txt
```

To enable all web tools, plotting, and testing tools, ensure these packages are installed:
```bash
pip install matplotlib beautifulsoup4 duckduckgo-search docker pytest pytest-mock pytest-cov scipy
```

### 3. Setup and Pull Ollama Models
Ensure Ollama is running (`ollama serve`), then pull the required models:
```bash
ollama pull batiai/gemma4-e2b:q4
ollama pull nomic-embed-text
```

### 4. Build the Sandbox Docker Image *(Optional)*
If you want to execute scripts in the isolated sandbox:
```bash
docker build -t chatbox-sandbox:latest backend/sandbox/
```

---

## Execution Guide: What to Run

> [!IMPORTANT]
> **Working Directory**: All commands in this guide must be run from the root **`ChatBot/`** directory (`c:\Users\harsh\Chatbox_for_analysis_AI\ChatBot`), not from inside `backend/`.
> 
> If your terminal prompt is currently in `ChatBot\backend`, run:
> ```powershell
> cd ..
> ```
> This ensures Python can locate the `backend` package.

Follow this sequential checklist to initialize and run the various components of the codebase.

### Phase 1: Environment & Services Setup

1. **Verify Ollama is Running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```
2. **Verify Docker is Running** (if using sandbox):
   ```bash
   docker info
   ```

---

### Phase 2: Database & RAG Initialization

Before launching agents or the API for the first time, initialize the persistent storage layers:

1. **Initialize SQLite Database**:
   Creates `chatbox.db` with tables for sessions, messages, jobs, parameters, and artifacts:
   ```bash
   python -m backend.db.migrations
   ```
   *Expected output:* `Database initialized.`

2. **Initialize Kùzu Knowledge Graph**:
   Creates node and relationship tables under `kuzu_db/` and seeds initial materials/systems facts:
   ```bash
   python -m backend.rag.knowledge_graph
   ```
   *Expected output:* `Domain data initialized.`

3. **Ingest Documentation into ChromaDB Vector Store**:
   Scans the repository's `.md` and `.py` files, chunks them, and generates embeddings in `chroma_db/`:
   ```bash
   python -m backend.rag.vectorstore
   ```
   *Expected output:* `Ingesting <N> chunks... Ingestion complete.`

4. **Ingest Jupyter Notebooks (.ipynb) into Vector Store & Knowledge Graph**:
   Extracts clean code and markdown cells from all notebooks (filtering out noisy output data and base64 plots), embeds them into ChromaDB, and registers `Notebook` relationships in the Kùzu Knowledge Graph:
   ```bash
   python -m backend.rag.notebook_parser
   ```
   *Expected output:* `Ingesting notebook cells... Notebook vector ingestion complete. Notebook Knowledge Graph registration complete.`

---

### Phase 3: Launching the Backend API & Web Frontend

Start the FastAPI REST server:

```bash
# Using uvicorn directly
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Or run the main module
python -m backend.main
```

- **Interactive Web Chat Frontend**: Open **`http://localhost:8000/`** in your browser to access the scientific computing interface.
  - **Real-Time Token Streaming**: Watch Python code and simulation scripts type out line-by-line in real-time inside the live code studio.
  - **KaTeX LaTeX Math Rendering**: Scientific formulas ($g(r)$, $N_{\text{pairs}}$, $\sqrt{\sum \Delta x^2}$, $\text{H}_2\text{O}$, fractions, summations) automatically render into typography.
  - **One-Click Domain Benchmark Prompts**: Instant templates for Quantum ESPRESSO DFT cutoffs, LAMMPS DeePMD NVT, raw dataset formatting, and surface kinetics.
  - **100-Pt Flagship Benchmark Button**: Streams full interfacial analysis code generation live, extracts the script, validates AST syntax, scores against the rubric, and displays the evaluation card with Grade and sub-scores.
  - **Live Elapsed Timer & Latency Pill**: Live generation timer (`⏳ Streaming... 14.2s`) and locked latency badge (`⚡ 23.4s`).
- **Interactive Swagger Docs**: Open `http://localhost:8000/docs` to test REST and SSE streaming endpoints (`POST /api/chat/stream`, `POST /api/evaluate_zero_shot/stream`).

---

### Phase 4: Single-Shot Evaluation & Agent Testing

#### 1. Automated Zero-Shot Accuracy Benchmark CLI
Runs the Interfacial Water Analysis generation on `batiai/gemma4-e2b:q4`, extracts the Python script, executes AST syntax analysis, scores it across all 6 rubric categories, stores the evaluation in SQLite, and prints a formatted terminal report:
```bash
python run_benchmarks.py
```
*Expected terminal output:*
```
================================================================================
Scientific Computing AI — Zero-Shot Interfacial Script Benchmark
Model: batiai/gemma4-e2b:q4 | Pure Single-Shot (No Tools, No RAG)
================================================================================
Executing Interfacial Water Analysis benchmark prompt...
Response received in 34.2s. Extracted Python script (124 lines).

Evaluation Score: 85.0/100 (Grade B)
Category Breakdown:
  - Syntax & AST Validity:               15.0 / 15.0 pts
  - Library Imports:                      15.0 / 15.0 pts
  - Interfacial Structural Classification: 20.0 / 25.0 pts
  - Analytical Physics & Kinetics:        20.0 / 25.0 pts
  - Multi-Model DP Averaging:              5.0 / 10.0 pts
  - Artifact Generation & Reporting:      10.0 / 10.0 pts
Stored evaluation #1 in SQLite database.
================================================================================
```

#### 2. Single-Shot Baseline Agent CLI
Evaluates `batiai/gemma4-e2b:q4` directly from the command line:
```bash
python -m backend.agents.baseline_agent
```

#### 3. Run the Hybrid RAG Scientific Q&A Agent
Queries the Hybrid Retriever (ChromaDB + Kùzu Knowledge Graph) and synthesizes domain answers:
```bash
python -m backend.agents.qa_agent
```
*Example question answered:* `"What pseudopotential should I use for Pt?"`

#### 4. Run the Self-Correcting Analysis Agent
Generates scientific Python analysis code and executes it in the sandbox:
```bash
python -m backend.agents.analysis_agent
```
*Example workflow:* Asks the agent to compute a water density profile, loads context, writes code, executes in the sandbox, inspects errors, and outputs the resulting plot artifacts.

---

### Phase 5: Running Domain Jupyter Notebooks

For interactive computational science workflows, start Jupyter from the root `ChatBot/` directory:
```bash
jupyter notebook
```

1. **DFT Convergence Sweeps**:
   - Path: `DFT/EnergyCutoffandKpointsScript.ipynb`
   - Input: `DFT/Pt.cif`
   - Function: Prompts for cutoff (`ecutwfc`) or k-point grid, generates `pw.in` and SLURM `qe.sub` scripts, and plots convergence curves.
2. **DeePMD Model Training**:
   - Path: `Training/Training_Script.ipynb`
   - Input: `Training/Raw_Files/*_Water`
   - Function: Converts raw training sets to binary format, creates test sets, and produces seeded `input.json` configs + SLURM GPU jobs.
3. **Chunked Molecular Dynamics**:
   - Path: `Molecular Dynamics/MD_Simulation_Script.ipynb`
   - Input: Starting `.lmp` files and frozen DeePMD `.pb` graphs.
   - Function: Generates chained batch directories and a master `submit_from.sh` using SLURM job dependencies (`afterok`).
4. **Interfacial Trajectory Analysis**:
   - Path: `Analysis/Water_Density_RDF_Survival_Probability_Hydroxyl_Fraction_with_IrO2_Template_Seperate_Plot_Capability.ipynb`
   - Input: `Analysis/iro2_water_concat.extxyz`
   - Function: Calculates water mass density profiles along z, pair radial distribution functions (RDF), residence/survival probabilities, and surface hydroxylation fractions.

---

## Configuration Reference

All settings can be customized via environment variables or in [backend/config.py](file:///c:/Users/harsh/Chatbox_for_analysis_AI/ChatBot/backend/config.py):

| Variable | Default Value | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint for the local Ollama daemon |
| `MODEL_NAME` | `batiai/gemma4-e2b:q4` | LLM model used for agents and code generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model used for ChromaDB semantic search |
| `CHROMA_PERSIST_DIR` | `chroma_db` | Storage path for persistent vector embeddings |
| `KUZU_DB_DIR` | `kuzu_db` | Storage path for Kùzu embedded graph database |
| `SQLITE_PATH` | `chatbox.db` | File path for SQLite database |
| `HPC_HOST` | `localhost` | Hostname or IP of the remote HPC cluster (`localhost` uses mock) |
| `HPC_USER` | `user` | SSH username on the HPC cluster |
| `HPC_KEY_PATH` | `~/.ssh/id_rsa` | Private SSH key path for cluster authentication |
| `WORKSPACE_ROOT` | `ChatBot/` (computed) | Root directory used for document indexing |

To override settings, set environment variables prior to running:
```powershell
$env:MODEL_NAME="llama3.1:8b"
$env:HPC_HOST="hpc.university.edu"
$env:HPC_USER="username"
```

---

## Testing & Quality Assurance

The codebase includes comprehensive unit and integration tests with strict coverage thresholds configured in `pytest.ini`:

```bash
# Run all unit tests with coverage reporting (33/33 tests passing, 95.79% coverage)
pytest backend/tests

# Run core evaluation and agent test files
pytest backend/tests/test_script_evaluator.py backend/tests/test_api.py backend/tests/test_baseline_agent.py backend/tests/test_database.py -v

# Run live integration tests (requires Docker and Ollama to be running)
pytest -m "" backend/tests/integration/test_live.py
```

*Note:* `pytest.ini` enforces `--cov=backend --cov-report=term-missing --cov-fail-under=95`. Current coverage stands at **95.79%**, exceeding the strict 95% threshold across all agent, evaluation, database, and API modules.

---

## Troubleshooting & Common Issues

1. **`Error connecting to LLM`**:
   - Verify that Ollama is running: execute `ollama list` in your terminal.
   - Verify that the models are pulled: `ollama pull batiai/gemma4-e2b:q4` and `ollama pull nomic-embed-text`.
2. **`Docker is not running on the host`**:
   - Start Docker Desktop.
   - If running in an environment where Docker is unavailable, the `SandboxExecutor` gracefully returns a descriptive error message without crashing.
3. **`Kuzu Database Lock Error`**:
   - Kùzu allows only one read-write process per database directory. Avoid running multiple ingestion or initialization scripts at the exact same time against the same `kuzu_db` folder.
4. **HPC SSH Connection Errors**:
   - When `HPC_HOST` is set to `"localhost"`, `slurm_tools.py` runs commands locally or returns mock job IDs (`99999`). When targeting a real HPC cluster, ensure your SSH key is added to `ssh-agent` and verify you can connect manually via `ssh -i <key> <user>@<host>`.
5. **Large Trajectory File Notice**:
   - `Analysis/iro2_water_concat.extxyz` is ~4.2 GB. When loading the trajectory in the analysis notebook or agent tools, use the `stride` parameter (e.g., `stride=10`) to avoid high memory consumption.
