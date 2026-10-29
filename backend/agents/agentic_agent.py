import os
import json
import re
import time
import requests
from typing import Dict, Any, Generator, Optional, List
from backend.config import settings
from backend.rag.hybrid_retriever import HybridRetriever
from backend.tools.sandbox_tools import run_python_in_sandbox
from backend.tools.web_tools import search_web, read_url_content
from backend.db.models import insert_session, insert_message, get_chat_history

AGENTIC_CODEGEN_SYSTEM_PROMPT = (
    "SYSTEM DIRECTIVE: You are an expert computational materials science Python developer and simulation engineer "
    "operating inside an isolated E2B / Docker execution sandbox (chatbox-sandbox:latest).\n"
    "\n"
    "MANDATORY EXECUTION DIRECTIVES:\n"
    "1. Sandbox Capabilities: The environment has Python 3.12, ASE 3.29.0, NumPy, SciPy, Matplotlib, and Pandas installed. "
    "Containers run headless with a 2 GB RAM cap and no network access.\n"
    "2. Strict Data Provenance & ASE Syntax:\n"
    "   - CRITICAL ASE SYNTAX: `ase.io.read(path, index=':')` returns a Python list of Atoms (`list[ase.Atoms]`), NOT a single Atoms object.\n"
    "   - Calling `.get_positions()`, `.get_cell()`, or `.positions` directly on `traj` raises AttributeError: 'list' object has no attribute 'get_positions'!\n"
    "   - Always unpack or index the Atoms instance: `traj = read(path, index=':'); atoms = traj[0] if isinstance(traj, list) else traj; positions = atoms.get_positions()`.\n"
    "   - Or read directly as Atoms: `atoms = read(path, index=0)`.\n"
    "3. Coordinate Filtering & Boundary Layer Analysis:\n"
    "   - NEVER compare continuous floating-point atomic coordinates with exact equality (`==`). `pos[:, 2] == z_min` is an anti-pattern that collapses entire surface planes down to 1 or 2 isolated points.\n"
    "   - Slab Surface Boundaries along z: Use a layer thickness tolerance window `tol` (0.5 to 1.0 Å). Identify surface layers via `pos[:, 2] <= (z_min + tol)` or `pos[:, 2] >= (z_max - tol)`.\n"
    "   - 3D Unit Cell Box Boundaries: Check fractional / scaled coordinates `scaled = atoms.get_scaled_positions(wrap=True)` within a skin depth `delta` (e.g. 0.02 to 0.05) across cell faces: `np.any((scaled < delta) | (scaled > 1.0 - delta), axis=1)`.\n"
    "   - Always print the count, chemical symbols, and coordinate arrays of the identified boundary atoms for verification.\n"
    "4. Interfacial Analysis Standards: For neighbor searches, cutoff determination, and surface site classification, "
    "you MUST use `ase.neighborlist` (specifically `from ase.neighborlist import neighbor_list, natural_cutoffs` or `NeighborList`).\n"
    "5. DeepMD Model Averaging: When evaluating multi-model DeepMD potentials, loop over the models in `Training/DP_Models/DPMD_Water/1/`, `2/`, and `3/`, "
    "computing cross-model statistics (mean and standard deviation).\n"
    "6. Headless Visualization: Always save presentation figures to the current working directory as PNG files "
    "(`plt.savefig(..., dpi=300, bbox_inches='tight')`) and call `plt.close()`. NEVER call `plt.show()`.\n"
    "7. Output Contract: Output ONLY the complete, syntactically correct Python code block enclosed in ```python ... ```. "
    "Get straight to the point and do not beat around the bush. Do not include conversational filler, explanations, or pleasantries."
)

AGENTIC_FIX_SYSTEM_PROMPT = (
    "SYSTEM DIRECTIVE: You are an autonomous scientific code debugger for the E2B / Docker execution sandbox.\n"
    "A previously synthesized scientific Python script failed execution.\n"
    "1. Diagnostic Analysis: Isolate the exact failure in the stderr traceback.\n"
    "   - If 'AttributeError: 'list' object has no attribute 'get_positions'', the code treated a list from `read(path, index=':')` as an Atoms instance. Fix by unpacking: `atoms = traj[0] if isinstance(traj, list) else traj` before `.get_positions()`.\n"
    "   - If boundary analysis returned only 1-2 atoms, replace exact float comparison `== z_min` with a physical layer tolerance `z <= z_min + tol` (e.g. tol = 0.8 Å) or fractional coordinates.\n"
    "2. Precision Correction: Fix the script utilizing the diagnostic error output and DuckDuckGo web search context. "
    "Resolve missing attributes, incorrect parameter signatures, or shape mismatches while preserving working data logic.\n"
    "3. Output Contract: Output ONLY the complete, corrected Python script in a ```python ... ``` block. "
    "Provide zero conversational preamble, apologies, or explanations."
)

AGENTIC_SYNTHESIS_SYSTEM_PROMPT = (
    "SYSTEM DIRECTIVE: You are SciCompute AI, an autonomous scientific materials and chemistry assistant.\n"
    "You operate in a fully integrated environment with:\n"
    "1. Grounded workspace Jupyter notebooks (`ChatBot/backend/notebooks/running.ipynb`) and trained DeepMD models in `Training/DP_Models/DPMD_Water/`.\n"
    "2. An isolated Docker / E2B Execution Sandbox where scripts and simulations are compiled, run, and verified on CPU/GPU.\n"
    "3. DuckDuckGo Web Search tools for real-time documentation and syntax verification.\n"
    "\n"
    "STRICT EXECUTION PROTOCOLS:\n"
    "- NEVER claim you lack an execution environment, cannot run code, or cannot access files.\n"
    "- Ground all scientific conclusions on the retrieved Hybrid RAG context and verified container outputs.\n"
    "- Get straight to the point and do not beat around the bush. Eliminate pleasantries and conversational filler.\n"
    "- Format responses with clear markdown headings, concise analysis, and KaTeX LaTeX math formulas ($...$ for inline, $$...$$ for display blocks)."
)

class AgenticRAGAgent:
    """
    Autonomous Scientific AI Agent with full capabilities:
    1. Hybrid RAG: ChromaDB Vector Store + Kùzu Knowledge Graph (ASE Documentation included)
    2. Secure Sandbox: Isolated Docker execution with automated self-correction
    3. Web Search & URL Tools: DuckDuckGo documentation and syntax lookup
    """

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model or settings.MODEL_NAME
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.retriever = HybridRetriever()

    def search_web(self, query: str, max_results: int = 3) -> list:
        """Searches the web via DuckDuckGo."""
        return search_web(query, max_results=max_results)

    def read_url_content(self, url: str) -> str:
        """Fetches and extracts text from a specified URL."""
        return read_url_content(url)

    def _is_execution_request(self, text: str) -> bool:
        """Determines if the user query asks to write/run code or execute a simulation analysis"""
        lower = text.lower()
        explicit_code = bool(re.search(r"\b(python|script|code|run|execute|plot|simulate|compute|calculate)\b", lower))
        pure_question = bool(re.match(r"^\s*(what\s+is|explain|define|how\s+does)\b", lower)) and not explicit_code
        if pure_question:
            return False
        patterns = [
            r"\b(run|execute|compute|calculate|simulate|plot|write)\b",
            r"\b(python script|code block|simulation script)\b",
            r"\b(docker|container|sandbox)\b"
        ]
        return any(re.search(p, lower) for p in patterns)

    @staticmethod
    def _clean_images(images: Optional[List[str]]) -> Optional[List[str]]:
        if not images:
            return None
        cleaned = []
        for img in images:
            if not isinstance(img, str):
                continue
            img = img.strip()
            if "," in img and ";base64" in img:
                cleaned.append(img.split(",", 1)[1])
            elif img:
                cleaned.append(img)
        return cleaned if cleaned else None

    def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        timeout: int = 3600,
        images: Optional[List[str]] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for hist_msg in history:
                cleaned_hist = {"role": hist_msg["role"], "content": hist_msg["content"]}
                if hist_msg.get("images"):
                    cleaned_imgs = self._clean_images(hist_msg["images"])
                    if cleaned_imgs:
                        cleaned_hist["images"] = cleaned_imgs
                messages.append(cleaned_hist)

        user_msg = {"role": "user", "content": prompt}
        cleaned_imgs = self._clean_images(images)
        if cleaned_imgs:
            user_msg["images"] = cleaned_imgs
        messages.append(user_msg)

        payload = {
            "model": self.model,
            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_ctx": settings.NUM_CTX,       # 8192 tokens
                "num_predict": 4096,               # Cap generation at 4096 tokens to prevent runaway loops
                "temperature": 0.15,               # Low temperature for deterministic syntax & math
                "top_p": 0.90,                     # Nucleus sampling cutoff
                "top_k": 40,                       # Standard Gemma 4 top-k filter
                "repeat_penalty": 1.05,            # Mild penalty to prevent comment/code loop repetition
                "num_gpu": settings.NUM_GPU
            }
        }
        return requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=stream,
            timeout=timeout
        )

    def chat_stream(
        self,
        user_message: str,
        session_id: Optional[int] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None
    ) -> Generator[str, None, None]:
        """
        Executes the agentic pipeline with streaming SSE events:
        - Step 1: Hybrid RAG retrieval (ChromaDB + Kùzu Graph)
        - Step 2: Tool execution (Sandbox / Web search if requested)
        - Step 3: Streamed response synthesis
        """
        start_time = time.time()
        # Fetch previous conversation history for this session before inserting current message
        history = get_chat_history(session_id=session_id) if session_id else []

        if not session_id:
            title = user_message[:40] + ("..." if len(user_message) > 40 else "")
            session_id = insert_session(title=title)

        insert_message(
            session_id=session_id,
            role="user",
            content=user_message,
            agent_used="AgenticRAG",
            images_json=json.dumps(images) if images else None
        )

        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'mode': 'agentic'})}\n\n"

        trace_steps = []

        # Step 1: Hybrid RAG
        rag_start = {"type": "agent_step", "step": "rag_start", "title": "Hybrid RAG Retrieval", "status": "running", "details": "Searching ChromaDB vector store and querying Kùzu Knowledge Graph..."}
        trace_steps.append(rag_start)
        yield f"data: {json.dumps(rag_start)}\n\n"
        
        context = ""
        try:
            retrieval_query = user_message
            if history:
                last_user_msg = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
                anaphoric_words = ["it", "its", "this", "these", "that", "those", "same", "previous", "earlier", "above"]
                if any(w in user_message.lower().split() for w in anaphoric_words) and last_user_msg:
                    retrieval_query = f"{last_user_msg}\n{user_message}"
            from backend.rag.query_rewriting import expand_query
            retrieval_query = expand_query(retrieval_query)
            context = str(self.retriever.retrieve(retrieval_query, current_session_id=session_id))
            rag_snippet = context[:400] + ("..." if len(context) > 400 else "")
            rag_step = {"type": "agent_step", "step": "rag_complete", "title": "Hybrid RAG Retrieval", "status": "completed", "details": rag_snippet}
            trace_steps.append(rag_step)
            yield f"data: {json.dumps(rag_step)}\n\n"
        except Exception as e:
            rag_err = {"type": "agent_step", "step": "rag_error", "title": "Hybrid RAG Retrieval", "status": "warning", "details": f"Retrieval notice: {e}"}
            trace_steps.append(rag_err)
            yield f"data: {json.dumps(rag_err)}\n\n"

        # Step 2: Check for execution request (Sandbox + Web Search)
        execution_exit_code = None
        execution_requested = self._is_execution_request(user_message)
        sandbox_output = ""
        artifacts = []
        code_extracted = ""

        thinking_parts = []
        content_parts = []
        eval_count = 0

        if execution_requested:
            from backend.notebooks.attach_to_notebook import (
                is_anatase_water_molecular_prompt,
                is_anatase_tio2_prompt,
                extract_notebook_context,
                format_notebook_context_prompt,
                attach_script_to_notebook
            )
            is_mol = is_anatase_water_molecular_prompt(user_message)

            cg_start = {"type": "agent_step", "step": "code_gen_start", "title": "Script Synthesis", "status": "running", "details": "Synthesizing Python analysis script..."}
            trace_steps.append(cg_start)
            yield f"data: {json.dumps(cg_start)}\n\n"

            nb_context_section = ""
            data_mount_path = None
            data_mount_inst = ""
            from pathlib import Path
            root_dir = Path(__file__).resolve().parent.parent.parent
            cand1 = root_dir / "public" / "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
            cand2 = root_dir / "ChatBot" / "public" / "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
            anatase_extxyz = cand1 if cand1.exists() else (cand2 if cand2.exists() else None)

            if is_mol and anatase_extxyz:
                data_mount_path = str(anatase_extxyz)
                data_mount_inst = (
                    "7. Trajectory dataset `Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz` is mounted inside the Docker container at `/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz`.\n"
                    "   CRITICAL ASE SYNTAX: `read(path, index=':')` returns a LIST. Unpack first: `traj = read(path, index=':'); atoms = traj[0] if isinstance(traj, list) else traj`.\n"
                    "   Then extract positions: `positions = atoms.get_positions()` and symbols: `np.array(atoms.get_chemical_symbols())`.\n"
                    "   Physical System: Anatase TiO2 surface slab with molecularly adsorbed water. Surface Ti_5c (CUS) coordinate molecular water; bridging oxygens O_br hydrogen bond with water.\n"
                    "   Calculate total energy (`atoms.get_potential_energy()` or `atoms.info.get('energy')`), water density profile along z, structural metrics, and save presentation graph `anatase_tio2_water_analysis.png` (`plt.savefig(..., dpi=300, bbox_inches='tight')`, `plt.close()`).\n"
                )
            elif anatase_extxyz and any(w in user_message.lower() for w in ["anatase", "extxyz", "ti02", "tio2"]):
                data_mount_path = str(anatase_extxyz)

            if is_mol:
                grounding_requirement = (
                    "1. Load the trajectory file directly from `/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz` using ASE `read(...)`. "
                    "CRITICAL: `read(path, index=':')` returns a list of Atoms. Unpack `atoms = traj[0] if isinstance(traj, list) else traj` before accessing `.get_positions()`. Do not assume variables are pre-loaded in memory."
                )
            else:
                grounding_requirement = "1. Ground your script on the target notebook `running.ipynb` alone (pre-loaded `traj` variable in memory from `/content/MPtrj/...`)."

            history_section = ""
            if history:
                prev_msgs = [f"{h['role'].upper()}: {h['content'][:300]}" for h in history[-4:]]
                history_section = "--- PREVIOUS CHAT CONTEXT ---\n" + "\n".join(prev_msgs) + "\n\n"

            code_prompt = f"""You are an expert computational materials science Python developer.
Based on the execution context and retrieved domain context from workspace Jupyter notebooks, write a complete, robust, executable Python script to address the user request.

{history_section}{nb_context_section}--- RETRIEVED WORKSPACE NOTEBOOK CONTEXT ---
{context}

--- USER REQUEST ---
{user_message}

CRITICAL IMPLEMENTATION REQUIREMENTS:
{grounding_requirement}
2. For interfacial site identification, neighbor searches, and distance cutoffs, you MUST use `ase.neighborlist` (specifically `from ase.neighborlist import neighbor_list, natural_cutoffs` or `NeighborList`).
3. For multi-model DP averaging, reference the trained DeepMD potential models located at `Training/DP_Models/DPMD_Water/1/`, `Training/DP_Models/DPMD_Water/2/`, and `Training/DP_Models/DPMD_Water/3/`.
4. Output ONLY valid executable Python code in a ```python ... ``` block. Use ASE, NumPy, and Matplotlib.
5. Get straight to the point and do not beat around the bush. Do not include introductory pleasantries or explanations.
{data_mount_inst}"""

            try:
                code_response_parts = []
                res = self._call_llm(
                    code_prompt,
                    system_prompt=AGENTIC_CODEGEN_SYSTEM_PROMPT,
                    stream=True,
                    timeout=3600,
                    history=history
                )
                if hasattr(res, "raise_for_status"):
                    res.raise_for_status()

                if hasattr(res, "iter_lines"):
                    try:
                        for line in res.iter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line.decode("utf-8") if isinstance(line, bytes) else line)
                            except Exception:
                                continue

                            msg = chunk.get("message", {})
                            thinking_delta = msg.get("thinking", "")
                            content_delta = msg.get("content", "")

                            if thinking_delta:
                                thinking_parts.append(thinking_delta)
                                yield f"data: {json.dumps({'type': 'thinking', 'delta': thinking_delta})}\n\n"

                            if content_delta:
                                code_response_parts.append(content_delta)
                                content_parts.append(content_delta)
                                yield f"data: {json.dumps({'type': 'content', 'delta': content_delta})}\n\n"
                    except Exception:
                        pass

                # Fallback for non-streaming mocked responses in tests
                if not code_response_parts and hasattr(res, "json"):
                    try:
                        res_data = res.json()
                        msg = res_data.get("message", {})
                        if msg.get("thinking"):
                            thinking_parts.append(msg["thinking"])
                            yield f"data: {json.dumps({'type': 'thinking', 'delta': msg['thinking']})}\n\n"
                        code_content = msg.get("content", "")
                        if code_content:
                            code_response_parts.append(code_content)
                            content_parts.append(code_content)
                            yield f"data: {json.dumps({'type': 'content', 'delta': code_content})}\n\n"
                    except Exception:
                        pass

                code_response = "".join(code_response_parts).strip()
                
                # Extract code block
                code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", code_response)
                code_extracted = code_match.group(1).strip() if code_match else code_response.strip()

                cg_done = {"type": "agent_step", "step": "code_gen_complete", "title": "Script Synthesis", "status": "completed", "details": f"Generated {len(code_extracted.splitlines())} lines of scientific code."}
                trace_steps.append(cg_done)
                yield f"data: {json.dumps(cg_done)}\n\n"

                # Automatically attach script to Colab running.ipynb ONLY when not molecular extxyz
                if code_extracted and not is_mol:
                    try:
                        nb_attach = attach_script_to_notebook(code_extracted)
                        nb_step = {
                            "type": "agent_step",
                            "step": "notebook_attached",
                            "title": "Colab Notebook Updated",
                            "status": "completed",
                            "details": f"Attached script to running.ipynb (Cell #{nb_attach['target_cell_index'] + 1}). Ready to run in Colab/Kaggle!"
                        }
                        trace_steps.append(nb_step)
                        yield f"data: {json.dumps(nb_step)}\n\n"
                    except Exception:
                        pass

                # Run in Docker Sandbox
                sb_start = {"type": "agent_step", "step": "sandbox_start", "title": "Docker Sandbox Execution", "status": "running", "details": "Executing script inside isolated container (2GB RAM, no-net)..."}
                trace_steps.append(sb_start)
                yield f"data: {json.dumps(sb_start)}\n\n"
                
                sandbox_res = run_python_in_sandbox(code_extracted, data_mount_path=data_mount_path)
                execution_exit_code = sandbox_res.get("exit_code")
                if sandbox_res["exit_code"] == 0:
                    sandbox_output = sandbox_res["stdout"]
                    artifacts = sandbox_res.get("artifacts", [])
                    artifact_names = [os.path.basename(a) for a in artifacts]
                    if artifacts:
                        yield f"data: {json.dumps({'type': 'artifacts', 'artifacts': artifact_names, 'artifact_paths': artifacts})}\n\n"
                    details_str = f"Exit code 0 (Success). Output: {sandbox_output[:120]}"
                    if artifact_names:
                        details_str += f" | Generated: {', '.join(artifact_names)}"
                    sb_done = {"type": "agent_step", "step": "sandbox_complete", "title": "Docker Sandbox Execution", "status": "completed", "details": details_str}
                    trace_steps.append(sb_done)
                    yield f"data: {json.dumps(sb_done)}\n\n"
                else:
                    stderr_snippet = sandbox_res.get("stderr", "Unknown error")
                    sb_retry = {"type": "agent_step", "step": "sandbox_retry", "title": "Docker Sandbox Execution", "status": "warning", "details": f"Execution failed: {stderr_snippet[:150]}. Invoking Web Search tool..."}
                    trace_steps.append(sb_retry)
                    yield f"data: {json.dumps(sb_retry)}\n\n"
                    
                    # Web Search Tool invocation
                    web_query = f"Python {stderr_snippet.splitlines()[-1]}" if stderr_snippet else "Python ASE simulation error"
                    web_results = search_web(web_query, max_results=2)
                    web_step = {"type": "agent_step", "step": "web_complete", "title": "Web Search Tool", "status": "completed", "details": f"Searched DuckDuckGo: \"{web_query}\""}
                    trace_steps.append(web_step)
                    yield f"data: {json.dumps(web_step)}\n\n"
                    
                    # Re-run fix
                    fix_prompt = f"""The following script failed with stderr:
{stderr_snippet}

Web Search Context:
{web_results}

Original Code:
{code_extracted}

Write the corrected Python code block only."""
                    fix_res = self._call_llm(
                        fix_prompt,
                        system_prompt=AGENTIC_FIX_SYSTEM_PROMPT,
                        timeout=3600
                    )
                    fix_raw = fix_res.json().get("message", {}).get("content", "") if hasattr(fix_res, "json") else ""
                    fixed_code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", fix_raw)
                    if fixed_code_match:
                        fixed_candidate = fixed_code_match.group(1).strip()
                    else:
                        from backend.evaluation.gemini_evaluator import GeminiEvaluator
                        extracted_fallback = GeminiEvaluator.extract_python_code(fix_raw)
                        fixed_candidate = extracted_fallback.strip() if extracted_fallback else fix_raw.strip()

                    if fixed_candidate:
                        code_extracted = fixed_candidate
                        fix_delta = f"\n\n### 🔧 Self-Corrected Script\n```python\n{code_extracted}\n```\n"
                        content_parts.append(fix_delta)
                        yield f"data: {json.dumps({'type': 'content', 'delta': fix_delta})}\n\n"
                        if not is_mol:
                            try:
                                attach_script_to_notebook(code_extracted)
                            except Exception:
                                pass
                    
                    # Second execution attempt
                    retry_res = run_python_in_sandbox(code_extracted, data_mount_path=data_mount_path)
                    sandbox_output = retry_res.get("stdout", "") or retry_res.get("stderr", "")
                    artifacts = retry_res.get("artifacts", [])
                    artifact_names = [os.path.basename(a) for a in artifacts]
                    if artifacts:
                        yield f"data: {json.dumps({'type': 'artifacts', 'artifacts': artifact_names, 'artifact_paths': artifacts})}\n\n"
                    retry_exit = retry_res.get("exit_code", 1)
                    execution_exit_code = retry_exit
                    re_details = f"Self-correction finished (Exit {retry_exit})."
                    if artifact_names:
                        re_details += f" | Generated: {', '.join(artifact_names)}"
                    sb_re_done = {"type": "agent_step", "step": "sandbox_complete", "title": "Docker Sandbox Re-execution", "status": "completed", "details": re_details}
                    trace_steps.append(sb_re_done)
                    yield f"data: {json.dumps(sb_re_done)}\n\n"

            except Exception as e:
                sb_err = {"type": "agent_step", "step": "sandbox_error", "title": "Sandbox Pipeline", "status": "warning", "details": str(e)}
                trace_steps.append(sb_err)
                yield f"data: {json.dumps(sb_err)}\n\n"

        # Step 3: Stream Final Synthesized Response
        syn_start = {"type": "agent_step", "step": "synthesis_start", "title": "Agentic Synthesis", "status": "running", "details": "Synthesizing final response with RAG context and execution results..."}
        trace_steps.append(syn_start)
        yield f"data: {json.dumps(syn_start)}\n\n"

        from backend.notebooks.attach_to_notebook import is_anatase_water_molecular_prompt
        if is_anatase_water_molecular_prompt(user_message):
            conclusion_inst = "Conclude by summarizing the verified Docker container execution results and generated presentation plots, noting that execution ran strictly inside the isolated Docker sandbox environment without modifying any notebook."
        else:
            conclusion_inst = "Conclude by explicitly informing the user that their complete analysis script has been attached to `ChatBot/backend/notebooks/running.ipynb` and they can now execute the notebook in Google Colab (or on Kaggle with T4x2 GPU)."

        synthesis_prompt = f"""You are SciCompute AI, an autonomous scientific materials & chemistry assistant.
You possess direct access to:
1. Grounded target notebook `ChatBot/backend/notebooks/running.ipynb` and trained DeepMD models in `Training/DP_Models/DPMD_Water/1/`, `Training/DP_Models/DPMD_Water/2/`, and `Training/DP_Models/DPMD_Water/3/`.
2. A secure Docker Sandbox Execution Environment where code is tested and verified on CPU/GPU.
3. Web search tools (via DuckDuckGo) for real-time documentation and syntax verification.

NEVER claim you lack an execution environment, cannot run code, or cannot access files.

--- RETRIEVED HYBRID RAG CONTEXT ---
{context}

--- EXECUTION ARTIFACTS & OUTPUT ---
{f'Execution Output:\n{sandbox_output}' if sandbox_output else 'Sandboxed environment and tools are active and available.'}
{f'Generated Artifacts (Graphs/Plots): {", ".join([os.path.basename(a) for a in artifacts])}' if artifacts else ''}
{f'Generated Code:\n```python\n{code_extracted}\n```' if code_extracted else ''}

User Query:
{user_message}

Get straight to the point and do not beat around the bush. Eliminate pleasantries and conversational filler. Provide a well-structured, rigorous response with clear headings, KaTeX LaTeX math formulas ($...$ for inline, $$...$$ for display blocks), and code blocks using `ase.neighborlist` where appropriate. {conclusion_inst}"""

        if code_extracted and content_parts:
            divider = "\n\n---\n\n### Scientific Synthesis & Findings\n\n"
            content_parts.append(divider)
            yield f"data: {json.dumps({'type': 'content', 'delta': divider})}\n\n"

        try:
            stream_res = self._call_llm(
                prompt=synthesis_prompt,
                system_prompt=system_prompt or AGENTIC_SYNTHESIS_SYSTEM_PROMPT,
                stream=True,
                timeout=3600,
                images=images,
                history=history
            )
            stream_res.raise_for_status()

            if hasattr(stream_res, "iter_lines"):
                for line in stream_res.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8") if isinstance(line, bytes) else line)
                    except Exception:
                        continue

                    msg = chunk.get("message", {})
                    thinking_delta = msg.get("thinking", "")
                    content_delta = msg.get("content", "")

                    if thinking_delta:
                        thinking_parts.append(thinking_delta)
                        yield f"data: {json.dumps({'type': 'thinking', 'delta': thinking_delta})}\n\n"

                    if content_delta:
                        content_parts.append(content_delta)
                        yield f"data: {json.dumps({'type': 'content', 'delta': content_delta})}\n\n"

                    if chunk.get("eval_count"):
                        eval_count = chunk["eval_count"]

            latency = round(time.time() - start_time, 3)
            thinking = "".join(thinking_parts).strip()
            content = "".join(content_parts).strip()
            if execution_exit_code == 0 and code_extracted:
                # Persist the tested script instead of a failed pre-retry draft.
                content = re.sub(r"```(?:python)?\s*[\s\S]*?```", lambda _: "```python\n" + code_extracted + "\n```", content, count=1)

            syn_done = {"type": "agent_step", "step": "synthesis_complete", "title": "Agentic Synthesis", "status": "completed", "details": f"Synthesized response ({len(content.split())} words)."}
            trace_steps.append(syn_done)
            yield f"data: {json.dumps(syn_done)}\n\n"

            if content and thinking:
                final_text = f"<details class=\"thinking-details\"><summary><em>Thinking Process ({len(thinking.split())} words)</em></summary>\n\n{thinking}\n</details>\n\n{content}"
            elif content:
                final_text = content
            elif thinking:
                final_text = thinking
            else:
                final_text = "Analysis completed."

            insert_message(
                session_id=session_id,
                role="assistant",
                content=final_text,
                agent_used="AgenticRAG",
                tool_calls_json=json.dumps(trace_steps + [{"memory_quality": {"execution_required": execution_requested or bool(re.search(r"```", final_text)), "exit_code": execution_exit_code}}])
            )

            try:
                from backend.rag.chat_memory import get_chat_memory_store
                get_chat_memory_store().index_turn(
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=final_text,
                    agent_used="AgenticRAG",
                    execution_required=execution_requested or bool(re.search(r"```", final_text)),
                    exit_code=execution_exit_code
                )
            except Exception:
                pass

            yield f"data: {json.dumps({'type': 'done', 'latency_seconds': latency, 'eval_count': eval_count, 'session_id': session_id, 'mode': 'agentic', 'full_response': final_text, 'tool_calls': trace_steps})}\n\n"

        except Exception as e:
            err_str = f"Agentic generation error: {e}"
            yield f"data: {json.dumps({'type': 'error', 'error': err_str})}\n\n"

    def chat(
        self,
        user_message: str,
        session_id: Optional[int] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Synchronous chat execution for Agentic RAG"""
        thinking_parts = []
        content_parts = []
        current_session = session_id
        latency = 0.0
        tool_calls = []

        for sse in self.chat_stream(user_message, session_id=session_id, system_prompt=system_prompt, images=images):
            if "data: " in sse:
                try:
                    payload = json.loads(sse.replace("data: ", "").strip())
                    if payload.get("type") == "session":
                        current_session = payload.get("session_id")
                    elif payload.get("type") == "thinking":
                        thinking_parts.append(payload.get("delta", ""))
                    elif payload.get("type") == "content":
                        content_parts.append(payload.get("delta", ""))
                    elif payload.get("type") == "done":
                        latency = payload.get("latency_seconds", 0.0)
                        tool_calls = payload.get("tool_calls", [])
                except Exception:
                    pass

        content = "".join(content_parts).strip()
        thinking = "".join(thinking_parts).strip()

        if content and thinking:
            final_text = f"<details class=\"thinking-details\"><summary><em>Thinking Process ({len(thinking.split())} words)</em></summary>\n\n{thinking}\n</details>\n\n{content}"
        else:
            final_text = content or thinking or "Analysis completed."

        return {
            "status": "success",
            "response": final_text,
            "session_id": current_session,
            "model": self.model,
            "latency_seconds": latency,
            "agent_used": "AgenticRAG",
            "mode": "agentic",
            "tool_calls": tool_calls
        }

if __name__ == "__main__":
    import sys
    agent = AgenticRAGAgent()
    print("=" * 72)
    print("🤖 Autonomous Scientific AI Agent CLI (Agentic RAG)")
    print(f"Model: {agent.model} | Endpoint: {agent.base_url}")
    print("Capabilities: Hybrid RAG, Docker Sandbox, Web Search, Cross-Session Memory")
    print("Type your message below. Type 'exit', 'quit', or 'q' to end.")
    print("=" * 72)

    cli_session_id = insert_session(title="CLI Agentic Session")
    print(f"[Session #{cli_session_id} active with full multi-turn memory]\n")

    try:
        while True:
            try:
                user_query = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nExiting session. Goodbye!")
                break

            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit", "q"]:
                print("Exiting session. Goodbye!")
                break

            print("\nAgent > ", end="", flush=True)
            for sse in agent.chat_stream(user_query, session_id=cli_session_id):
                if "data: " in sse:
                    try:
                        data = json.loads(sse.replace("data: ", "").strip())
                        if data.get("type") == "agent_step":
                            print(f"\n⚡ [{data.get('title')}]: {data.get('details', '')[:90]}...", flush=True)
                            print("Agent > ", end="", flush=True)
                        elif data.get("type") == "content":
                            sys.stdout.write(data.get("delta", ""))
                            sys.stdout.flush()
                        elif data.get("type") == "done":
                            print(f"\n\n[Done — ⏱️ {data.get('latency_seconds', 0.0)}s | {len(data.get('tool_calls', []))} tool steps]\n")
                        elif data.get("type") == "error":
                            print(f"\n❌ Error: {data.get('error')}\n")
                    except Exception:
                        pass
    except KeyboardInterrupt:
        print("\n\nSession terminated by user.")
