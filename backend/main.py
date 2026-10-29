import os
import json
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from backend.config import settings
import uuid
from backend.agents.baseline_agent import SingleShotAgent
from backend.agents.cancellation import register_generation, cancel_generation, managed_stream
from backend.agents.execution_intent import requests_sandbox_execution
from backend.agents.agentic_agent import AgenticRAGAgent
from backend.evaluation.script_evaluator import ScriptEvaluator
from backend.evaluation.gemini_evaluator import GeminiEvaluator
from backend.db.models import (
    execute_query,
    insert_session,
    insert_evaluation,
    get_evaluations,
    INTERFACIAL_ANALYSIS_BENCHMARK_PROMPT,
    get_prompts,
    get_prompt_by_id,
    insert_prompt,
    update_prompt,
    delete_prompt,
    reset_prompts
)
from backend.db.migrations import init_db
from backend.notebooks.attach_to_notebook import (
    attach_script_to_notebook,
    get_latest_evaluation_code,
    extract_notebook_context,
    format_notebook_context_prompt,
    resolve_prompt_context
)

def _index_evaluated_memory(session_id, prompt, response, evaluation):
    if session_id is None or evaluation.get("audit_status") == "error" or (evaluation.get("score") or 0) < 80:
        return
    try:
        from backend.rag.chat_memory import get_chat_memory_store
        get_chat_memory_store().index_turn(
            session_id=session_id, user_message=prompt, assistant_message=response,
            agent_used="SingleShotAgent", execution_required=True,
            evaluation_score=evaluation["score"],
        )
    except Exception as exc:
        print(f"Notice: Could not index evaluated memory: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        from backend.rag.ingest_dataset_zip import ingest_configured_dataset
        await asyncio.to_thread(ingest_configured_dataset)
    except Exception as exc:
        print(f"Notice: Curated dataset sync failed: {exc}")
    try:
        from backend.rag.chat_memory import get_chat_memory_store
        get_chat_memory_store().ingest_all_sessions()
    except Exception as e:
        print(f"Notice: Background memory sync on startup skipped: {e}")

    settings.NUM_GPU = int(os.getenv("NUM_GPU", str(settings.NUM_GPU)))
    gpu_mode = "CPU-only (NUM_GPU=0)" if settings.NUM_GPU == 0 else (
        "GPU — all layers (AMD Radeon via ROCm/HIP)" if settings.NUM_GPU == -1
        else f"GPU — {settings.NUM_GPU} layers"
    )
    from backend.model_runtime import initialize_finetuned_model
    app.state.finetuned_model = await asyncio.to_thread(initialize_finetuned_model)
    print(f"Starting Scientific Computing API with active model: {settings.MODEL_NAME}")
    print(f"Inference device: {gpu_mode}")
    print(f"  To force CPU-only: set NUM_GPU=0 before starting (e.g. NUM_GPU=0 uvicorn ...)")
    print("Zero-Shot Script Generation & Accuracy Evaluation Engine active")
    try:
        yield
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass

async def startup_event():
    """Backward-compatible helper for tests invoking startup_event directly."""
    init_db()
    try:
        from backend.rag.ingest_dataset_zip import ingest_configured_dataset
        await asyncio.to_thread(ingest_configured_dataset)
    except Exception as exc:
        print(f"Notice: Curated dataset sync failed: {exc}")
    try:
        from backend.rag.chat_memory import get_chat_memory_store
        get_chat_memory_store().ingest_all_sessions()
    except Exception as e:
        print(f"Notice: Background memory sync on startup skipped: {e}")

app = FastAPI(title="Scientific Computing Chatbox API", version="1.0.0", lifespan=lifespan)

# Setup frontend directory (located at root ChatBot/frontend)
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

PUBLIC_DIR = Path(__file__).parent.parent / "public"
PUBLIC_DIR.mkdir(exist_ok=True)
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None
    system_prompt: Optional[str] = None
    mode: Optional[str] = "single_shot"
    include_notebook_context: Optional[bool] = None
    notebook_name: Optional[str] = "running.ipynb"
    images: Optional[List[str]] = None
    run_sandbox: Optional[bool] = None
    request_id: Optional[uuid.UUID] = None

class SessionCreateRequest(BaseModel):
    title: Optional[str] = "New Session"

class EvaluationRequest(BaseModel):
    session_id: Optional[int] = None
    prompt: Optional[str] = None
    notebook_name: Optional[str] = "running.ipynb"
    include_notebook_context: Optional[bool] = True

class GeminiEvaluationRequest(BaseModel):
    code: str
    prompt: Optional[str] = None
    execution_output: Optional[str] = None
    session_id: Optional[int] = None

class NotebookAttachRequest(BaseModel):
    code: Optional[str] = None
    notebook_name: Optional[str] = "running.ipynb"
    replace_placeholder: Optional[bool] = True

class PromptCreateRequest(BaseModel):
    id: Optional[str] = None
    title: str
    domain: str
    prompt: str

class PromptUpdateRequest(BaseModel):
    title: Optional[str] = None
    domain: Optional[str] = None
    prompt: Optional[str] = None

@app.get("/")
def read_root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"status": "ok", "message": "Scientific Computing Chatbox API is running"}

@app.get("/api/model_info")
def get_model_info():
    """Returns active model, GPU mode, and baseline evaluation status"""
    settings.NUM_GPU = int(os.getenv("NUM_GPU", str(settings.NUM_GPU)))
    gpu_mode = "CPU-only" if settings.NUM_GPU == 0 else (
        f"GPU (all layers)" if settings.NUM_GPU == -1 else f"GPU ({settings.NUM_GPU} layers)"
    )
    return {
        "model_name": settings.MODEL_NAME,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "mode": "single_shot",
        "tools_enabled": False,
        "rag_enabled": False,
        "gpu_mode": gpu_mode,
        "num_gpu": settings.NUM_GPU,
        "description": "Pure Single-Shot Prompt Evaluation Mode (Zero Tools / Zero RAG)"
    }

@app.get("/api/benchmark_prompts")
def get_benchmark_prompts():
    """Curated domain prompts to benchmark Gemma4 single-shot accuracy"""
    return get_prompts()

@app.post("/api/benchmark_prompts")
def create_benchmark_prompt(request: PromptCreateRequest):
    """Create a new benchmark or custom evaluation prompt"""
    title = request.title.strip()
    domain = request.domain.strip()
    prompt_text = request.prompt.strip()
    if not title or not domain or not prompt_text:
        raise HTTPException(status_code=400, detail="Title, domain, and prompt text are required.")
    
    prompt_id = request.id.strip() if request.id and request.id.strip() else f"prompt_{uuid.uuid4().hex[:8]}"
    existing = get_prompt_by_id(prompt_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Prompt with id '{prompt_id}' already exists.")
    
    created = insert_prompt(prompt_id=prompt_id, title=title, domain=domain, prompt=prompt_text, is_default=0)
    return created

@app.get("/api/benchmark_prompts/{prompt_id}")
def get_benchmark_prompt_by_id(prompt_id: str):
    """Retrieve a single prompt by its identifier"""
    prompt = get_prompt_by_id(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt

@app.put("/api/benchmark_prompts/{prompt_id}")
def update_benchmark_prompt(prompt_id: str, request: PromptUpdateRequest):
    """Update an existing prompt's title, domain, or instructions"""
    existing = get_prompt_by_id(prompt_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    updated = update_prompt(
        prompt_id=prompt_id,
        title=request.title.strip() if request.title is not None else None,
        domain=request.domain.strip() if request.domain is not None else None,
        prompt=request.prompt.strip() if request.prompt is not None else None
    )
    return updated

@app.delete("/api/benchmark_prompts/{prompt_id}")
def delete_benchmark_prompt(prompt_id: str):
    """Delete a prompt by its identifier"""
    success = delete_prompt(prompt_id)
    if not success:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"status": "success", "message": f"Prompt '{prompt_id}' deleted successfully.", "id": prompt_id}

@app.post("/api/benchmark_prompts/reset")
def reset_benchmark_prompts_endpoint():
    """Reset benchmark prompts back to factory defaults"""
    restored = reset_prompts()
    return {"status": "success", "message": "Benchmark prompts reset to factory defaults.", "prompts": restored}

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """Executes single-shot or agentic RAG generation and logs the conversation"""
    has_text = bool(request.message and request.message.strip())
    has_images = bool(request.images and len(request.images) > 0)
    if not has_text and not has_images:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if request.mode in ["agentic", "agentic_rag"]:
        agent = AgenticRAGAgent()
    else:
        agent = SingleShotAgent()

    raw_text = request.message.strip() if has_text else "Please analyze this attached scientific image / structure for materials science insights."
    user_msg = raw_text
    # Notebook references enter the same bounded retriever as other sources.
    if request.include_notebook_context is True:
        user_msg += "\nRequested notebook reference: " + (request.notebook_name or "running.ipynb")

    should_run_sandbox = request.run_sandbox if request.run_sandbox is not None else requests_sandbox_execution(raw_text)
    result = agent.chat(
        user_message=user_msg,
        session_id=request.session_id,
        system_prompt=request.system_prompt,
        images=request.images,
        **({"run_sandbox": should_run_sandbox}
           if request.mode not in ["agentic", "agentic_rag"] else {})
    )
    return result

@app.post("/api/chat/stream")
def chat_stream_endpoint(request: ChatRequest):
    """Executes single-shot or agentic RAG generation with live SSE token streaming"""
    has_text = bool(request.message and request.message.strip())
    has_images = bool(request.images and len(request.images) > 0)
    if not has_text and not has_images:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if request.mode in ["agentic", "agentic_rag"]:
        agent = AgenticRAGAgent()
    else:
        agent = SingleShotAgent()

    raw_text = request.message.strip() if has_text else "Please analyze this attached scientific image / structure for materials science insights."
    user_msg = raw_text
    # Notebook references enter the same bounded retriever as other sources.
    if request.include_notebook_context is True:
        user_msg += "\nRequested notebook reference: " + (request.notebook_name or "running.ipynb")

    should_run_sandbox = request.run_sandbox if request.run_sandbox is not None else requests_sandbox_execution(raw_text)
    request_id = str(request.request_id or uuid.uuid4())
    run = register_generation(request_id)
    stream = agent.chat_stream(
            user_message=user_msg,
            session_id=request.session_id,
            system_prompt=request.system_prompt,
            images=request.images,
            **({"run_sandbox": should_run_sandbox}
               if request.mode not in ["agentic", "agentic_rag"] else {})
        )
    return StreamingResponse(
        managed_stream(stream, request_id, run, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/chat/{request_id}/stop")
def stop_chat(request_id: uuid.UUID):
    cancel_generation(str(request_id))
    return {"status": "stopping", "request_id": str(request_id)}

@app.post("/api/evaluate/gemini")
def evaluate_gemini_endpoint(request: GeminiEvaluationRequest):
    """
    Evaluates generated Python code using the configured Gemini model with validated scoring.
    Strict scoring: incorrect output or failing code is ruled around 20%.
    """
    evaluator = GeminiEvaluator()
    result = evaluator.evaluate(
        code_or_response=request.code,
        prompt=request.prompt,
        execution_output=request.execution_output,
        session_id=request.session_id
    )
    if result.get("audit_status") == "error":
        raise HTTPException(status_code=502, detail=result["summary"])
    return result

@app.post("/api/evaluate_zero_shot/stream")
def evaluate_zero_shot_stream_endpoint(request: EvaluationRequest):
    """
    Executes Interfacial Water Analysis zero-shot benchmark on Gemma4 with live token streaming,
    grounded on the running.ipynb execution context.
    Extracts code, runs accuracy evaluator, stores in SQLite, and emits the final report card.
    """
    nb_name = request.notebook_name or "running.ipynb"
    nb_path = Path(__file__).parent / "notebooks" / nb_name
    nb_context = None
    nb_prompt_prefix = ""

    if request.include_notebook_context is not False:
        try:
            nb_context = extract_notebook_context(str(nb_path))
            nb_prompt_prefix = format_notebook_context_prompt(nb_context) + "\n\n--- BENCHMARK TASK ---\n"
        except Exception as e:
            print(f"Notice: Could not load notebook context from {nb_path}: {e}")

    base_prompt = request.prompt.strip() if (request.prompt and request.prompt.strip()) else INTERFACIAL_ANALYSIS_BENCHMARK_PROMPT
    full_prompt = f"{nb_prompt_prefix}{base_prompt}" if nb_prompt_prefix else base_prompt

    agent = SingleShotAgent()
    system_prompt = """You are an expert computational materials scientist and Python developer generating a complete, peer-review-grade analysis script for `running.ipynb`.

<thinking_protocol>
Think efficiently in 4 brief steps (maximum 120 words):
1. Map all required variables (`traj`, `traj_file`, `system_name`, or multi-model paths).
2. Confirm exact physical cutoffs: surface metal `z >= max(m_z) - 1.5` Å, bulk oxygen stoichiometry `s * N_metal`, CUS/adsorbed water distance `<= 2.5` Å, O-H covalent bond `<= 1.2` Å.
3. Verify formulas for z-mass density (g/cm³), pair RDFs g(r), continuous survival probability P(t), and trapezoidal residence time tau.
4. Ensure all required `.npz`, `.png`, and `.txt` artifacts are written to disk.
</thinking_protocol>

<execution_contract>
1. Use ONLY standard pre-installed scientific libraries: `ase`, `numpy`, `scipy`, `matplotlib`, `os`, `time`. Do NOT import `pwtools`.
2. Handle both pre-loaded notebook memory (`traj`) and multi-model directory loops (`range(1, dp + 1)`) gracefully so the script executes without error in both Colab (`running.ipynb`) and standalone Docker containers.
3. Output ONLY a single, complete, syntactically valid ```python ... ``` code block with zero prose before or after.
</execution_contract>"""

    def stream_generator():
        stream = agent.chat_stream(
            user_message=full_prompt,
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
            evaluation = evaluator.evaluate(
                full_response_text,
                prompt=base_prompt,
                notebook_context=nb_context,
                session_id=session_id
            )
            if evaluation.get("audit_status") == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': evaluation['summary']})}\n\n"
                return
            _index_evaluated_memory(session_id, full_prompt, full_response_text, evaluation)

            eval_id = evaluation.get("evaluation_id")
            if not eval_id and evaluation.get("audit_status") != "completed":
                eval_id = insert_evaluation(
                    session_id=session_id,
                    model_name=evaluation.get("model", settings.MODEL_NAME),
                    prompt=base_prompt,
                    generated_code=evaluation.get("extracted_code", ""),
                    score=evaluation.get("score", 0),
                    grade=evaluation.get("grade", "F"),
                    category_scores_json=json.dumps(evaluation.get("category_scores", {})),
                    evaluation_report_json=json.dumps({
                        "passed_checks": evaluation.get("passed_checks", evaluation.get("strengths", [])),
                        "failed_checks": evaluation.get("failed_checks", evaluation.get("critical_issues", [])),
                        "summary": evaluation.get("summary", ""),
                        "verdict": evaluation.get("verdict", ""),
                        "critical_issues": evaluation.get("critical_issues", []),
                        "strengths": evaluation.get("strengths", []),
                        "recommendations": evaluation.get("recommendations", [])
                    }),
                    latency_seconds=latency_seconds
                )

            notebook_attach_result = None
            if evaluation.get("extracted_code"):
                from backend.notebooks.attach_to_notebook import is_anatase_water_molecular_prompt
                if not is_anatase_water_molecular_prompt(base_prompt):
                    try:
                        notebook_attach_result = attach_script_to_notebook(
                            code=evaluation.get("extracted_code", ""),
                            notebook_path=str(nb_path),
                            replace_placeholder=True
                        )
                    except Exception as e:
                        notebook_attach_result = {"status": "error", "error": str(e)}

            eval_event = {
                "type": "eval_complete",
                "evaluation_id": eval_id,
                "session_id": session_id,
                "model": evaluation.get("model", settings.MODEL_NAME),
                "prompt": base_prompt,
                "generated_code": evaluation.get("extracted_code", ""),
                "raw_response": full_response_text,
                "score": evaluation.get("score", 0),
                "grade": evaluation.get("grade", "F"),
                "verdict": evaluation.get("verdict", ""),
                "category_scores": evaluation.get("category_scores", {}),
                "critical_issues": evaluation.get("critical_issues", []),
                "strengths": evaluation.get("strengths", []),
                "recommendations": evaluation.get("recommendations", []),
                "passed_checks": evaluation.get("passed_checks", evaluation.get("strengths", [])),
                "failed_checks": evaluation.get("failed_checks", evaluation.get("critical_issues", [])),
                "summary": evaluation.get("summary", ""),
                "evaluation_type": evaluation.get("evaluation_type", "gemini_strict"),
                "domain": evaluation.get("domain"),
                "latency_seconds": latency_seconds,
                "notebook_attached": notebook_attach_result,
                "notebook_context": nb_context
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
    Executes the Interfacial Water Analysis zero-shot prompt on Gemma4 grounded on running.ipynb,
    extracts the Python script, runs the accuracy rubric evaluator,
    and stores the benchmark metrics in SQLite.
    """
    nb_name = request.notebook_name or "running.ipynb"
    nb_path = Path(__file__).parent / "notebooks" / nb_name
    nb_context = None
    nb_prompt_prefix = ""

    if request.include_notebook_context is not False:
        try:
            nb_context = extract_notebook_context(str(nb_path))
            nb_prompt_prefix = format_notebook_context_prompt(nb_context) + "\n\n--- BENCHMARK TASK ---\n"
        except Exception as e:
            print(f"Notice: Could not load notebook context from {nb_path}: {e}")

    base_prompt = request.prompt.strip() if (request.prompt and request.prompt.strip()) else INTERFACIAL_ANALYSIS_BENCHMARK_PROMPT
    full_prompt = f"{nb_prompt_prefix}{base_prompt}" if nb_prompt_prefix else base_prompt

    agent = SingleShotAgent()
    system_prompt = (
        "SYSTEM DIRECTIVE: You are an expert computational materials scientist and Python developer "
        "powered by Gemma 4 with E2B sandbox capabilities.\n"
        "You are generating a script to be attached directly to the user's Jupyter notebook `running.ipynb`.\n"
        "1. Grounding & Lineage: Ground your script strictly on the pre-loaded variables, trajectory files, and environment of `running.ipynb`.\n"
        "2. ASE Syntax & Execution: Use ASE, NumPy, SciPy, and Matplotlib. In ASE, `read(path, index=':')` returns a list (`list[Atoms]`); always unpack `atoms = traj[0] if isinstance(traj, list) else traj` before accessing `.get_positions()`. For headless plots, save PNG files (`plt.savefig(..., dpi=300, bbox_inches='tight')`) and call `plt.close()`.\n"
        "3. Output Contract: Get straight to the point and do not beat around the bush. Provide no conversational fluff or preamble. "
        "Output ONLY the complete, working, syntactically correct Python code block."
    )

    chat_res = agent.chat(
        user_message=full_prompt,
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
    evaluation = evaluator.evaluate(raw_response, prompt=base_prompt, notebook_context=nb_context, session_id=chat_res["session_id"])
    if evaluation.get("audit_status") == "error":
        raise HTTPException(status_code=502, detail=evaluation["summary"])
    _index_evaluated_memory(chat_res["session_id"], full_prompt, raw_response, evaluation)

    eval_id = evaluation.get("evaluation_id")
    if not eval_id and evaluation.get("audit_status") != "completed":
        eval_id = insert_evaluation(
            session_id=chat_res["session_id"],
            model_name=evaluation.get("model", settings.MODEL_NAME),
            prompt=base_prompt,
            generated_code=evaluation.get("extracted_code", ""),
            score=evaluation.get("score", 0),
            grade=evaluation.get("grade", "F"),
            category_scores_json=json.dumps(evaluation.get("category_scores", {})),
            evaluation_report_json=json.dumps({
                "passed_checks": evaluation.get("passed_checks", evaluation.get("strengths", [])),
                "failed_checks": evaluation.get("failed_checks", evaluation.get("critical_issues", [])),
                "summary": evaluation.get("summary", ""),
                "verdict": evaluation.get("verdict", ""),
                "critical_issues": evaluation.get("critical_issues", []),
                "strengths": evaluation.get("strengths", []),
                "recommendations": evaluation.get("recommendations", [])
            }),
            latency_seconds=chat_res.get("latency_seconds", 0.0)
        )

    notebook_attach_result = None
    if evaluation.get("extracted_code"):
        from backend.notebooks.attach_to_notebook import is_anatase_water_molecular_prompt
        if not is_anatase_water_molecular_prompt(base_prompt):
            try:
                notebook_attach_result = attach_script_to_notebook(
                    code=evaluation["extracted_code"],
                    notebook_path=str(nb_path),
                    replace_placeholder=True
                )
            except Exception as e:
                notebook_attach_result = {"status": "error", "error": str(e)}

    return {
        "status": "success",
        "evaluation_id": eval_id,
        "session_id": chat_res["session_id"],
        "model": evaluation.get("model", settings.MODEL_NAME),
        "prompt": base_prompt,
        "generated_code": evaluation.get("extracted_code", ""),
        "raw_response": raw_response,
        "score": evaluation.get("score", 0),
        "grade": evaluation.get("grade", "F"),
        "verdict": evaluation.get("verdict", ""),
        "category_scores": evaluation.get("category_scores", {}),
        "critical_issues": evaluation.get("critical_issues", []),
        "strengths": evaluation.get("strengths", []),
        "recommendations": evaluation.get("recommendations", []),
        "passed_checks": evaluation.get("passed_checks", evaluation.get("strengths", [])),
        "failed_checks": evaluation.get("failed_checks", evaluation.get("critical_issues", [])),
        "summary": evaluation.get("summary", ""),
        "evaluation_type": evaluation.get("evaluation_type", "gemini_strict"),
        "latency_seconds": chat_res.get("latency_seconds", 0.0),
        "notebook_attached": notebook_attach_result,
        "notebook_context": nb_context
    }

@app.get("/api/notebooks/context")
def get_notebook_context_endpoint(notebook_name: Optional[str] = "running.ipynb"):
    """
    Retrieves the extracted execution context, environment, and pre-loaded state of running.ipynb.
    Allows frontend and callers to inspect the notebook context provided to Gemma.
    """
    nb_name = notebook_name or "running.ipynb"
    nb_path = Path(__file__).parent / "notebooks" / nb_name
    try:
        ctx = extract_notebook_context(str(nb_path))
        return {
            "status": "success",
            "notebook_name": nb_name,
            "context": ctx,
            "formatted_prompt": format_notebook_context_prompt(ctx)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract notebook context: {e}")

@app.post("/api/notebooks/attach")
def attach_notebook_endpoint(request: NotebookAttachRequest):
    """
    Attaches a Python script (or the latest generated script from evaluations)
    to the target Colab notebook (running.ipynb).
    """
    code = request.code
    if not code:
        code = get_latest_evaluation_code()
    if not code:
        raise HTTPException(status_code=400, detail="No code provided and no recent evaluation found in database.")

    nb_name = request.notebook_name or "running.ipynb"
    nb_path = Path(__file__).parent / "notebooks" / nb_name
    try:
        res = attach_script_to_notebook(
            code=code,
            notebook_path=str(nb_path),
            replace_placeholder=request.replace_placeholder if request.replace_placeholder is not None else True
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to attach script to notebook: {e}")


@app.get("/api/evaluations")
def list_evaluations():
    """Retrieves previous script evaluations with parsed reports"""
    rows = get_evaluations(limit=25)
    result = []
    for r in rows:
        item = dict(r)
        if isinstance(item.get("category_scores_json"), str):
            try:
                item["category_scores"] = json.loads(item["category_scores_json"])
            except Exception:
                item["category_scores"] = {}
        if isinstance(item.get("evaluation_report_json"), str):
            try:
                rep = json.loads(item["evaluation_report_json"])
                item["evaluation_report"] = rep
                item["passed_checks"] = rep.get("passed_checks", [])
                item["failed_checks"] = rep.get("failed_checks", [])
                item["summary"] = rep.get("summary", "")
                item["verdict"] = rep.get("verdict", "")
                item["execution_status"] = rep.get("execution_status", "Untested")
                for field in ("audit_status", "audit_version", "review_type", "execution_evidence", "raw_score", "score_cap"):
                    item[field] = rep.get(field)
                item["critical_issues"] = rep.get("critical_issues", [])
                item["strengths"] = rep.get("strengths", [])
                item["recommendations"] = rep.get("recommendations", [])
                item["evaluator_model"] = rep.get("evaluator_model", item.get("model_name"))
                item["evaluation_type"] = rep.get("evaluation_type", "gemini_strict" if "verdict" in rep else "rag_benchmark")
                item["domain"] = rep.get("domain")
            except Exception:
                item["evaluation_report"] = {}
        result.append(item)
    return result

@app.get("/api/sessions")
def list_sessions():
    """Retrieves previous chat sessions with their execution mode"""
    rows = execute_query(
        """
        SELECT s.id, s.title, s.created_at, s.last_message_at,
               (SELECT m.agent_used FROM messages m WHERE m.session_id = s.id AND m.agent_used IS NOT NULL ORDER BY m.id DESC LIMIT 1) as last_agent_used
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
        "SELECT id, role, content, timestamp, agent_used, tool_calls_json, images_json FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    return [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["timestamp"],
            "agent_used": row["agent_used"],
            "tool_calls_json": row.get("tool_calls_json") if hasattr(row, "get") else None,
            "images_json": row.get("images_json") if hasattr(row, "get") else None
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

@app.post("/api/memory/sync")
def sync_chat_memory():
    """Batch indexes all past sessions and turns from SQLite into ChromaDB memory store"""
    try:
        from backend.rag.chat_memory import get_chat_memory_store
        count = get_chat_memory_store().ingest_all_sessions()
        return {"status": "success", "turns_ingested": count, "message": f"Successfully synced {count} conversational turns into cross-session memory."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory sync failed: {e}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
