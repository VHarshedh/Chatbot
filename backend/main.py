import os
import json
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from backend.config import settings
from backend.agents.baseline_agent import SingleShotAgent
from backend.agents.agentic_agent import AgenticRAGAgent
from backend.evaluation.script_evaluator import ScriptEvaluator
from backend.db.models import execute_query, insert_session, insert_evaluation, get_evaluations
from backend.db.migrations import init_db

app = FastAPI(title="Scientific Computing Chatbox API", version="1.0.0")

# Setup frontend directory (located at root ChatBot/frontend)
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

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
    "Output only the complete Python code block."
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None
    system_prompt: Optional[str] = None
    mode: Optional[str] = "single_shot"

class SessionCreateRequest(BaseModel):
    title: Optional[str] = "New Session"

class EvaluationRequest(BaseModel):
    session_id: Optional[int] = None
    prompt: Optional[str] = None

@app.get("/")
def read_root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"status": "ok", "message": "Scientific Computing Chatbox API is running"}

@app.get("/api/model_info")
def get_model_info():
    """Returns active model and baseline evaluation status"""
    return {
        "model_name": settings.MODEL_NAME,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "mode": "single_shot",
        "tools_enabled": False,
        "rag_enabled": False,
        "description": "Pure Single-Shot Prompt Evaluation Mode (Zero Tools / Zero RAG)"
    }

@app.get("/api/benchmark_prompts")
def get_benchmark_prompts():
    """Curated domain prompts to benchmark Gemma4 single-shot accuracy"""
    return [
        {
            "id": "interfacial_script_gen",
            "domain": "Interfacial Script Evaluation",
            "title": "Full Interfacial Analysis Script Generator",
            "prompt": INTERFACIAL_ANALYSIS_BENCHMARK_PROMPT
        },
        {
            "id": "dft_cutoff",
            "domain": "DFT",
            "title": "Pt SCF Convergence Parameters",
            "prompt": "What are the recommended plane-wave kinetic energy cutoff (ecutwfc) and charge density cutoff (ecutrho) for Platinum (Pt) using a PBE PAW or ONCV pseudopotential in Quantum ESPRESSO? Please write a sample &SYSTEM namelist block."
        },
        {
            "id": "md_chaining",
            "domain": "Molecular Dynamics",
            "title": "DeePMD LAMMPS Interface Script",
            "prompt": "Write a minimal LAMMPS in.lammps input script for an NVT production run of an interfacial water system driven by a pair_style deepmd potential. Include velocity creation, thermostatting at 300K, and periodic restart writes."
        },
        {
            "id": "training_format",
            "domain": "DeepMD Training",
            "title": "Raw Data to Binary Sets",
            "prompt": "In DeePMD-kit, what is the exact layout of raw data files (coord.raw, box.raw, energy.raw, force.raw) and how are they converted into set.000 numpy binary datasets for training?"
        },
        {
            "id": "analysis_kinetics",
            "domain": "Interfacial Analysis",
            "title": "Surface Sites & Survival Probability",
            "prompt": "Explain how coordinatively unsaturated (CUS) metal sites and adsorbed water oxygens (O_adb) are structurally classified on a metal oxide surface, and write the mathematical definition of the water survival probability P(t)."
        }
    ]

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """Executes single-shot or agentic RAG generation and logs the conversation"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if request.mode in ["agentic", "agentic_rag"]:
        agent = AgenticRAGAgent()
    else:
        agent = SingleShotAgent()

    result = agent.chat(
        user_message=request.message,
        session_id=request.session_id,
        system_prompt=request.system_prompt
    )
    return result

@app.post("/api/chat/stream")
def chat_stream_endpoint(request: ChatRequest):
    """Executes single-shot or agentic RAG generation with live SSE token streaming"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if request.mode in ["agentic", "agentic_rag"]:
        agent = AgenticRAGAgent()
    else:
        agent = SingleShotAgent()

    return StreamingResponse(
        agent.chat_stream(
            user_message=request.message,
            session_id=request.session_id,
            system_prompt=request.system_prompt
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/evaluate_zero_shot/stream")
def evaluate_zero_shot_stream_endpoint(request: EvaluationRequest):
    """
    Executes Interfacial Water Analysis zero-shot benchmark on Gemma4 with live token streaming,
    extracts code, runs accuracy evaluator, stores in SQLite, and emits the final report card.
    """
    prompt = request.prompt.strip() if (request.prompt and request.prompt.strip()) else INTERFACIAL_ANALYSIS_BENCHMARK_PROMPT
    agent = SingleShotAgent()
    system_prompt = (
        "You are an expert computational materials scientist and Python developer. "
        "Write complete, working, syntactically correct Python code blocks using ASE, NumPy, and Matplotlib."
    )

    def stream_generator():
        stream = agent.chat_stream(
            user_message=prompt,
            session_id=request.session_id,
            system_prompt=system_prompt
        )
        full_response_text = ""
        session_id = request.session_id
        latency_seconds = 0.0

        for sse_event in stream:
            # Yield token event directly to client
            yield sse_event

            # Inspect event if it's the 'done' event
            if "data: " in sse_event:
                try:
                    payload = json.loads(sse_event.replace("data: ", "").strip())
                    if payload.get("type") == "session":
                        session_id = payload.get("session_id")
                    elif payload.get("type") == "done":
                        full_response_text = payload.get("full_response", "")
                        latency_seconds = payload.get("latency_seconds", 0.0)
                        session_id = payload.get("session_id", session_id)
                except Exception:
                    pass

        # Evaluate the completed script
        if full_response_text:
            evaluator = ScriptEvaluator()
            evaluation = evaluator.evaluate(full_response_text)

            eval_id = insert_evaluation(
                session_id=session_id,
                model_name=settings.MODEL_NAME,
                prompt=prompt,
                generated_code=evaluation["extracted_code"],
                score=evaluation["score"],
                grade=evaluation["grade"],
                category_scores_json=json.dumps(evaluation["category_scores"]),
                evaluation_report_json=json.dumps({
                    "passed_checks": evaluation["passed_checks"],
                    "failed_checks": evaluation["failed_checks"],
                    "summary": evaluation["summary"]
                }),
                latency_seconds=latency_seconds
            )

            eval_event = {
                "type": "eval_complete",
                "evaluation_id": eval_id,
                "session_id": session_id,
                "model": settings.MODEL_NAME,
                "prompt": prompt,
                "generated_code": evaluation["extracted_code"],
                "raw_response": full_response_text,
                "score": evaluation["score"],
                "grade": evaluation["grade"],
                "category_scores": evaluation["category_scores"],
                "passed_checks": evaluation["passed_checks"],
                "failed_checks": evaluation["failed_checks"],
                "summary": evaluation["summary"],
                "latency_seconds": latency_seconds
            }
            yield f"data: {json.dumps(eval_event)}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/evaluate_zero_shot")
def evaluate_zero_shot_endpoint(request: EvaluationRequest):
    """
    Executes the Interfacial Water Analysis zero-shot prompt on Gemma4,
    extracts the Python script, runs the accuracy rubric evaluator,
    and stores the benchmark metrics in SQLite.
    """
    prompt = request.prompt.strip() if (request.prompt and request.prompt.strip()) else INTERFACIAL_ANALYSIS_BENCHMARK_PROMPT
    
    agent = SingleShotAgent()
    system_prompt = (
        "You are an expert computational materials scientist and Python developer. "
        "Write complete, working, syntactically correct Python code blocks using ASE, NumPy, and Matplotlib."
    )
    
    chat_res = agent.chat(
        user_message=prompt,
        session_id=request.session_id,
        system_prompt=system_prompt
    )

    if chat_res.get("status") != "success":
        raise HTTPException(
            status_code=500,
            detail=chat_res.get("error", "Model generation failed. Check if Ollama is running.")
        )

    raw_response = chat_res.get("response", "")
    evaluator = ScriptEvaluator()
    evaluation = evaluator.evaluate(raw_response)

    eval_id = insert_evaluation(
        session_id=chat_res["session_id"],
        model_name=settings.MODEL_NAME,
        prompt=prompt,
        generated_code=evaluation["extracted_code"],
        score=evaluation["score"],
        grade=evaluation["grade"],
        category_scores_json=json.dumps(evaluation["category_scores"]),
        evaluation_report_json=json.dumps({
            "passed_checks": evaluation["passed_checks"],
            "failed_checks": evaluation["failed_checks"],
            "summary": evaluation["summary"]
        }),
        latency_seconds=chat_res.get("latency_seconds", 0.0)
    )

    return {
        "status": "success",
        "evaluation_id": eval_id,
        "session_id": chat_res["session_id"],
        "model": settings.MODEL_NAME,
        "prompt": prompt,
        "generated_code": evaluation["extracted_code"],
        "raw_response": raw_response,
        "score": evaluation["score"],
        "grade": evaluation["grade"],
        "category_scores": evaluation["category_scores"],
        "passed_checks": evaluation["passed_checks"],
        "failed_checks": evaluation["failed_checks"],
        "summary": evaluation["summary"],
        "latency_seconds": chat_res.get("latency_seconds", 0.0)
    }

@app.get("/api/evaluations")
def list_evaluations():
    """Retrieves previous zero-shot script evaluations"""
    return get_evaluations(limit=25)

@app.get("/api/sessions")
def list_sessions():
    """Retrieves previous chat sessions with their execution mode"""
    rows = execute_query(
        """
        SELECT s.id, s.title, s.created_at, s.last_message_at,
               (SELECT m.agent_used FROM messages m WHERE m.session_id = s.id AND m.role = 'assistant' ORDER BY m.id DESC LIMIT 1) as last_agent_used
        FROM sessions s ORDER BY s.last_message_at DESC LIMIT 50
        """
    )
    result = []
    for row in rows:
        try:
            agent_used = row["last_agent_used"]
        except (KeyError, IndexError):
            agent_used = None
        result.append({
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "last_message_at": row["last_message_at"],
            "mode": "agentic" if agent_used in ["AgenticRAG", "agentic"] else "zero_shot",
            "agent_used": agent_used or "SingleShotAgent"
        })
    return result

@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: int):
    """Retrieves all messages for a specific session"""
    rows = execute_query(
        "SELECT id, role, content, timestamp, agent_used, tool_calls_json FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    return [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["timestamp"],
            "agent_used": row["agent_used"],
            "tool_calls_json": row.get("tool_calls_json") if hasattr(row, "get") else None
        }
        for row in rows
    ]

@app.post("/api/sessions")
def create_session(request: SessionCreateRequest):
    """Creates a new empty chat session"""
    session_id = insert_session(title=request.title)
    return {"session_id": session_id, "title": request.title}

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: int):
    """Deletes a session and its associated messages"""
    execute_query("DELETE FROM messages WHERE session_id = ?", (session_id,))
    execute_query("DELETE FROM sessions WHERE id = ?", (session_id,))
    return {"status": "deleted", "session_id": session_id}

@app.on_event("startup")
async def startup_event():
    init_db()
    print(f"Starting Scientific Computing API with active model: {settings.MODEL_NAME}")
    print("Zero-Shot Script Generation & Accuracy Evaluation Engine active")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
