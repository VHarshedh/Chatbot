import os
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from backend.config import settings
from backend.agents.baseline_agent import SingleShotAgent
from backend.db.models import execute_query, insert_session

app = FastAPI(title="Scientific Computing Chatbox API", version="1.0.0")

# Setup frontend directory (located at root ChatBot/frontend)
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None
    system_prompt: Optional[str] = None
    mode: Optional[str] = "single_shot"

class SessionCreateRequest(BaseModel):
    title: Optional[str] = "New Session"

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
    """Executes single-shot generation and logs the conversation"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    agent = SingleShotAgent()
    result = agent.chat(
        user_message=request.message,
        session_id=request.session_id,
        system_prompt=request.system_prompt
    )
    return result

@app.get("/api/sessions")
def list_sessions():
    """Retrieves previous chat sessions"""
    rows = execute_query(
        "SELECT id, title, created_at, last_message_at FROM sessions ORDER BY last_message_at DESC LIMIT 50"
    )
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "last_message_at": row["last_message_at"]
        }
        for row in rows
    ]

@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: int):
    """Retrieves all messages for a specific session"""
    rows = execute_query(
        "SELECT id, role, content, timestamp, agent_used FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    return [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["timestamp"],
            "agent_used": row["agent_used"]
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
    print(f"Starting Scientific Computing API with active model: {settings.MODEL_NAME}")
    print("Running in pure Single-Shot Prompt Evaluation Mode (No tools / No RAG)")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
