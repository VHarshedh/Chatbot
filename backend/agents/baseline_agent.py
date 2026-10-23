import os
import re
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from backend.config import settings
from backend.db.models import insert_session, insert_message, execute_query, get_chat_history
from backend.tools.web_tools import search_web, read_url_content

def _get_data_mount_path(text: str) -> Optional[str]:
    root_dir = Path(__file__).resolve().parent.parent.parent
    cand1 = root_dir / "public" / "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
    cand2 = root_dir / "ChatBot" / "public" / "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
    cand = cand1 if cand1.exists() else (cand2 if cand2.exists() else None)
    if cand and any(w in text.lower() for w in ["anatase", "extxyz", "ti02", "tio2", "water molecular"]):
        return str(cand)
    return None

def _is_ase_query(text: str) -> bool:
    """Detects if prompt or query involves ASE or molecular dynamics simulations."""
    keywords = [
        "ase", "atoms", "neighbor_list", "extxyz", "trajectory", "anatase", "tio2",
        "molecular dynamics", "velocityverlet", "langevin", "maxwellboltzmann",
        "bfgs", "fire", "fixatoms", "fixbondlengths", "coordination", "rdf",
        "radial distribution", "dft", "quantum espresso", "deepmd", "lammps",
        "chemical_symbols", "scaled_positions", "periodic boundary"
    ]
    lower = text.lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in keywords)

def _is_web_search_query(text: str) -> bool:
    """Detects if prompt explicitly asks for web search or online lookup."""
    patterns = [
        r"\b(search (?:the )?web|web search|look up online|search online|duckduckgo|browse online)\b",
        r"\b(find online|search internet)\b"
    ]
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)

DEFAULT_SYSTEM_PROMPT = r"""You are SciCompute AI, a principal computational materials scientist and scientific Python engineer specialized in ASE (Atomic Simulation Environment), Quantum ESPRESSO (DFT), DeePMD-kit, and LAMMPS.

<execution_environment>
- You operate with an automated, isolated Docker execution sandbox (chatbox-sandbox:latest) with Python 3.12, ASE, Quantum ESPRESSO, LAMMPS, NumPy, SciPy, Matplotlib, and Pandas.
- ExtXYZ trajectories and simulation files are mounted under `/mnt/data/` (e.g. `/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz`).
- When asked to run, execute, or analyze Python code or simulations in Docker, ALWAYS output the complete Python script enclosed in a ```python ... ``` block. The backend automatically extracts your code block, executes it inside Docker, and displays the console output and plot artifacts to the user.
- NEVER claim that you cannot run code, lack an execution environment, cannot access files, or cannot run in Docker.
</execution_environment>

<tool_and_knowledge_access>
- ASE Documentation: You have comprehensive access to all ASE documentation modules, including `ase.io`, `ase.build`, `ase.neighborlist`, `ase.geometry`, `ase.md`, `ase.optimize`, and `ase.constraints`.
- Web Tools: You have access to real-time DuckDuckGo web search and URL inspection tools (`search_web`, `read_url_content`) to look up online scientific references and inspect external documentation.
</tool_and_knowledge_access>

<thinking_protocol>
Keep internal reasoning concise and structured (under 150 words):
1. Identify the physical system, chemical species (e.g., separating crystal lattice atoms from adsorbate/water molecules), units (Å, eV, Ry, ps, ns, g/cm³), and required outputs.
2. Verify ASE & NumPy data types (`read(path, index=0)` -> `Atoms`; `read(path, index=':')` -> `list[Atoms]`).
3. Verify periodic boundary conditions (`atoms.set_pbc(True)`, `neighbor_list` minimum image convention) and Matplotlib axes dimensions (2D vs `projection='3d'`).
</thinking_protocol>

<scientific_coding_standards>
1. Direct Output & Zero Fabricated Logs:
   - Begin immediately with the technical answer or executable ```python ... ``` block.
   - NEVER fabricate, guess, or type simulated terminal/console output after a code block. All code execution output is generated exclusively by the Docker sandbox container.
2. Canonical ASE & NumPy Idioms:
   - Single-Frame Loading: `atoms = read(path, index=0); atoms.set_pbc(True)`
   - Multi-Frame Loading: `traj = read(path, index=':'); frames = traj if isinstance(traj, list) else [traj]`
   - Neighbor Search & Species Separation:
     Use `from ase.neighborlist import neighbor_list`. Separate water oxygens (`O_w`) from metal-oxide lattice oxygens (`O_slab`) via O-H covalent bond cutoffs (`{('O', 'H'): 1.25}`).
   - Numerical Integration Compatibility (NumPy 1.x & 2.x):
     `trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))`
3. Headless Matplotlib Contract:
   - For 2D plots (`fig, ax = plt.subplots()`), pass only 2 coordinate arrays `ax.scatter(x, y, s=...)` and never call `ax.set_zlabel()`.
   - For 3D plots, explicitly initialize `ax = fig.add_subplot(111, projection='3d')`.
   - Always save figures via `plt.savefig(filename, dpi=300, bbox_inches='tight')` and call `plt.close()`. Never call `plt.show()`.
4. Plain Typography in Python Scripts (NO LaTeX Formatting in Code):
   - In Python scripts, plot labels, titles, legends, print statements, and comments, NEVER use LaTeX commands or math delimiters (e.g., DO NOT write `\text{}`, `\AA`, `$\text{TiO}_2$`, `$...$`, or `$$...$$`).
   - Use standard Unicode characters and plain text strings (e.g., `'TiO2'`, `'H2O'`, `'X Position (Å)'`, `'Z Position (Å)'`, `'Energy (eV)'`).
   - Format with LaTeX `$ ... $` only in markdown conversational responses outside of code blocks.
</scientific_coding_standards>"""

class SingleShotAgent:
    """
    Baseline scientific agent equipped with multi-turn conversation memory,
    Docker sandbox execution, grounded ASE documentation retrieval via Hybrid RAG,
    and real-time DuckDuckGo web search & URL inspection tools.
    """
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or settings.MODEL_NAME
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self._retriever = None

    @property
    def retriever(self):
        if self._retriever is None:
            try:
                from backend.rag.hybrid_retriever import HybridRetriever
                self._retriever = HybridRetriever()
            except Exception:
                self._retriever = None
        return self._retriever

    def retrieve_ase_docs(self, query: str, max_chunks: int = 3) -> str:
        """Retrieves authoritative ASE documentation and simulation references from Hybrid RAG."""
        try:
            if self.retriever:
                return str(self.retriever.retrieve(query, max_chunks=max_chunks))
        except Exception as e:
            return f"Error retrieving ASE documentation: {e}"
        return ""

    def search_web(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Searches the web via DuckDuckGo for real-time documentation and syntax verification."""
        try:
            return search_web(query, max_results=max_results)
        except Exception as e:
            return [{"error": f"Search failed: {e}"}]

    def read_url_content(self, url: str) -> str:
        """Fetches and extracts clean text content from a specified URL."""
        try:
            return read_url_content(url)
        except Exception as e:
            return f"Error reading URL: {e}"

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

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        run_sandbox: bool = False,
        use_rag: bool = True,
        use_web: bool = True
    ) -> Dict[str, Any]:
        """
        Directly queries the Ollama /api/chat endpoint with conversation history.
        Streams chunks with num_ctx=4096 and timeout=(10, 600) to prevent read timeouts on CPU reasoning.
        """
        sys_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        start_time = time.time()

        extra_context = []
        if use_rag and _is_ase_query(prompt):
            try:
                ase_docs = self.retrieve_ase_docs(prompt)
                if ase_docs and not ase_docs.startswith("Error"):
                    extra_context.append(f"<retrieved_ase_documentation>\n{ase_docs}\n</retrieved_ase_documentation>")
            except Exception:
                pass

        if use_web and _is_web_search_query(prompt):
            try:
                clean_q = re.sub(r"(?i)\b(search (?:the )?web (?:for)?|look up online (?:for)?|web search)\b", "", prompt).strip() or prompt
                web_hits = self.search_web(clean_q, max_results=3)
                if web_hits and not web_hits[0].get("error"):
                    formatted_web = "\n".join(f"- [{h.get('title', 'Ref')}]({h.get('href', '')}): {h.get('body', '')}" for h in web_hits if h.get("title"))
                    if formatted_web:
                        extra_context.append(f"<retrieved_web_search_results>\n{formatted_web}\n</retrieved_web_search_results>")
            except Exception:
                pass

        if use_web:
            urls = re.findall(r"https?://[^\s)\]]+", prompt)
            for u in urls[:2]:
                try:
                    u_text = self.read_url_content(u)
                    if u_text and not u_text.startswith("Error"):
                        extra_context.append(f"<retrieved_url_content url=\"{u}\">\n{u_text[:2000]}\n</retrieved_url_content>")
                except Exception:
                    pass

        if extra_context:
            sys_prompt = sys_prompt + "\n\n" + "\n\n".join(extra_context)

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
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
            "messages": messages,
            "stream": True,
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

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=(15, 3600)
            )
            response.raise_for_status()

            thinking_parts = []
            content_parts = []
            eval_count = 0
            total_duration = 0

            # Support both live streaming iter_lines and mocked unit-test responses
            lines = list(response.iter_lines()) if hasattr(response, "iter_lines") else []
            if not lines:
                try:
                    data = response.json()
                    if "message" in data:
                        content_parts.append(data["message"].get("content", ""))
                        thinking_parts.append(data["message"].get("thinking", ""))
                    elif "response" in data:
                        content_parts.append(data["response"])
                    eval_count = data.get("eval_count", 0)
                    total_duration = data.get("total_duration", 0)
                except Exception:
                    pass
            else:
                for line in lines:
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8") if isinstance(line, bytes) else line)
                        msg = chunk.get("message", {})
                        if msg.get("thinking"):
                            thinking_parts.append(msg["thinking"])
                        if msg.get("content"):
                            content_parts.append(msg["content"])
                        elif chunk.get("response"):
                            content_parts.append(chunk["response"])
                        if chunk.get("eval_count"):
                            eval_count = chunk["eval_count"]
                        if chunk.get("total_duration"):
                            total_duration = chunk["total_duration"]
                    except Exception:
                        continue

            thinking = "".join(thinking_parts).strip()
            content = "".join(content_parts).strip()
            latency = round(time.time() - start_time, 3)

            execution_exit_code = None
            code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", content)
            code_to_run = None
            if code_match:
                candidate = code_match.group(1).strip()
                if candidate and len(candidate.splitlines()) >= 3:
                    code_to_run = candidate
            elif not code_to_run:
                user_code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", prompt)
                if user_code_match:
                    code_to_run = user_code_match.group(1).strip()
                elif any(prompt.strip().startswith(prefix) for prefix in ("import ", "from ", "def ", "#")):
                    code_lines = [line for line in prompt.splitlines() if not re.search(r"\b(can you run|please run|run this in docker)\b", line, re.I)]
                    candidate = "\n".join(code_lines).strip()
                    if len(candidate.splitlines()) >= 3:
                        code_to_run = candidate

            if run_sandbox and code_to_run:
                try:
                    from backend.tools.sandbox_tools import run_python_in_sandbox
                    from backend.notebooks.attach_to_notebook import is_anatase_water_molecular_prompt
                    
                    data_mount = _get_data_mount_path(prompt)
                    sb_res = run_python_in_sandbox(code_to_run, data_mount_path=data_mount)
                    
                    exit_code = sb_res.get("exit_code", 1)
                    execution_exit_code = exit_code
                    stdout = sb_res.get("stdout", "").strip()
                    stderr = sb_res.get("stderr", "").strip()
                    artifacts = sb_res.get("artifacts", [])
                    artifact_names = [os.path.basename(a) for a in artifacts]
                    
                    status_badge = "Success (Exit Code 0)" if exit_code == 0 else f"Failed (Exit Code {exit_code})"
                    out_display = stdout if stdout else (stderr if stderr else "No console output.")
                    
                    sb_section = (
                        f"\n\n---\n\n"
                        f"### 🐳 Secure Docker Sandbox Execution\n"
                        f"- **Container Status**: {status_badge}\n"
                        f"- **Environment**: Isolated `chatbox-sandbox:latest` (CPU, `--network none`, `--memory 2g`, `--pids-limit 50`, `--read-only`)\n"
                    )
                    if artifact_names:
                        sb_section += f"- **Generated Artifacts**: {', '.join(artifact_names)}\n"
                    sb_section += f"\n**Container Output**:\n```text\n{out_display}\n```"
                    
                    if exit_code != 0 and use_web:
                        try:
                            err_lines = [l.strip() for l in stderr.splitlines() if l.strip()]
                            err_query = err_lines[-1] if err_lines else "Python ASE simulation error"
                            web_diag = self.search_web(f"Python {err_query}", max_results=2)
                            if web_diag and not web_diag[0].get("error"):
                                diag_items = "\n".join(f"- [{h.get('title', 'Ref')}]({h.get('href', '')}): {h.get('body', '')}" for h in web_diag if h.get("title"))
                                if diag_items:
                                    sb_section += f"\n\n**🌐 Web Diagnostic Insights**:\n{diag_items}"
                        except Exception:
                            pass
                    
                    content += sb_section

                    # Attach to Colab running.ipynb ONLY if not molecular extxyz
                    is_mol = is_anatase_water_molecular_prompt(prompt)
                    if not is_mol:
                        try:
                            from backend.notebooks.attach_to_notebook import attach_script_to_notebook
                            nb_res = attach_script_to_notebook(code_to_run)
                            content += (
                                f"\n\n"
                                f"📓 **Colab Notebook Ready**: This script has been automatically attached to cell #{nb_res['target_cell_index'] + 1} "
                                f"in `ChatBot/backend/notebooks/running.ipynb`. You can now open and run the notebook in Google Colab (or on Kaggle with T4x2 GPU)!"
                            )
                        except Exception:
                            pass
                except Exception as e:
                    content += f"\n\n---\n\n*Sandbox Notice: {e}*"

            if content and thinking:
                final_text = f"<details class=\"thinking-details\"><summary><em>Thinking Process ({len(thinking.split())} words)</em></summary>\n\n{thinking}\n</details>\n\n{content}"
            elif content:
                final_text = content
            elif thinking:
                final_text = thinking
            else:
                final_text = "No response generated."

            return {
                "status": "success",
                "memory_quality": {"execution_required": bool(code_match), "exit_code": execution_exit_code},
                "response": final_text,
                "thinking": thinking,
                "content": content,
                "model": self.model,
                "latency_seconds": latency,
                "mode": "single_shot",
                "total_duration_ns": total_duration,
                "eval_count": eval_count
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "error": f"Could not connect to Ollama at {self.base_url}. Make sure 'ollama serve' is running.",
                "model": self.model,
                "latency_seconds": round(time.time() - start_time, 3),
                "mode": "single_shot"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model": self.model,
                "latency_seconds": round(time.time() - start_time, 3),
                "mode": "single_shot"
            }

    def chat(
        self,
        user_message: str,
        session_id: Optional[int] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None,
        run_sandbox: bool = False,
        use_rag: bool = True,
        use_web: bool = True
    ) -> Dict[str, Any]:
        """
        Processes a chat turn, tracks session history in SQLite, and returns the response.
        """
        # Fetch previous conversation history for this session before inserting current message
        history = get_chat_history(session_id=session_id) if session_id else []

        # Create session if none provided
        if not session_id:
            title = user_message[:40] + ("..." if len(user_message) > 40 else "")
            session_id = insert_session(title=title)

        # Log user message with images
        insert_message(
            session_id=session_id,
            role="user",
            content=user_message,
            agent_used="SingleShotAgent",
            images_json=json.dumps(images) if images else None
        )

        # Generate response grounded on chat history
        result = self.generate_response(user_message, system_prompt=system_prompt, images=images, history=history, run_sandbox=run_sandbox, use_rag=use_rag, use_web=use_web)

        # Log assistant message
        if result["status"] == "success":
            insert_message(
                session_id=session_id,
                role="assistant",
                content=result["response"],
                agent_used="SingleShotAgent",
                tool_calls_json=json.dumps({"memory_quality": result.get("memory_quality", {})})
            )

            try:
                from backend.rag.chat_memory import get_chat_memory_store
                get_chat_memory_store().index_turn(
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=result["response"],
                    agent_used="SingleShotAgent",
                    **result.get("memory_quality", {})
                )
            except Exception:
                pass

        result["session_id"] = session_id
        return result


    def chat_stream(
        self,
        user_message: str,
        session_id: Optional[int] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None,
        run_sandbox: bool = False,
        use_rag: bool = True,
        use_web: bool = True
    ):
        """
        Streams a chat response token-by-token via Server-Sent Events (SSE) format: data: {...}\n\n
        Persists user and assistant messages in SQLite upon stream completion.
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
            agent_used="SingleShotAgent",
            images_json=json.dumps(images) if images else None
        )

        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'model': self.model})}\n\n"

        sys_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

        extra_context = []
        if use_rag and _is_ase_query(user_message):
            try:
                ase_docs = self.retrieve_ase_docs(user_message)
                if ase_docs and not ase_docs.startswith("Error"):
                    extra_context.append(f"<retrieved_ase_documentation>\n{ase_docs}\n</retrieved_ase_documentation>")
            except Exception:
                pass

        if use_web and _is_web_search_query(user_message):
            try:
                clean_q = re.sub(r"(?i)\b(search (?:the )?web (?:for)?|look up online (?:for)?|web search)\b", "", user_message).strip() or user_message
                web_hits = self.search_web(clean_q, max_results=3)
                if web_hits and not web_hits[0].get("error"):
                    formatted_web = "\n".join(f"- [{h.get('title', 'Ref')}]({h.get('href', '')}): {h.get('body', '')}" for h in web_hits if h.get("title"))
                    if formatted_web:
                        extra_context.append(f"<retrieved_web_search_results>\n{formatted_web}\n</retrieved_web_search_results>")
            except Exception:
                pass

        if use_web:
            urls = re.findall(r"https?://[^\s)\]]+", user_message)
            for u in urls[:2]:
                try:
                    u_text = self.read_url_content(u)
                    if u_text and not u_text.startswith("Error"):
                        extra_context.append(f"<retrieved_url_content url=\"{u}\">\n{u_text[:2000]}\n</retrieved_url_content>")
                except Exception:
                    pass

        if extra_context:
            sys_prompt = sys_prompt + "\n\n" + "\n\n".join(extra_context)

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        if history:
            for hist_msg in history:
                cleaned_hist = {"role": hist_msg["role"], "content": hist_msg["content"]}
                if hist_msg.get("images"):
                    cleaned_imgs = self._clean_images(hist_msg["images"])
                    if cleaned_imgs:
                        cleaned_hist["images"] = cleaned_imgs
                messages.append(cleaned_hist)

        user_msg = {"role": "user", "content": user_message}
        cleaned_imgs = self._clean_images(images)
        if cleaned_imgs:
            user_msg["images"] = cleaned_imgs
        messages.append(user_msg)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
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

        execution_exit_code = None
        thinking_parts = []
        content_parts = []
        eval_count = 0
        total_duration = 0

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=(15, 3600)
            )
            response.raise_for_status()

            # Directly stream tokens as they arrive from Ollama without blocking
            if hasattr(response, "iter_lines"):
                for line in response.iter_lines():
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
                    if chunk.get("total_duration"):
                        total_duration = chunk["total_duration"]

            # Fallback for non-streaming mocked responses in tests
            if not thinking_parts and not content_parts:
                try:
                    data = response.json()
                    msg = data.get("message", {})
                    if msg.get("thinking"):
                        thinking_parts.append(msg["thinking"])
                        yield f"data: {json.dumps({'type': 'thinking', 'delta': msg['thinking']})}\n\n"
                    if msg.get("content"):
                        content_parts.append(msg["content"])
                        yield f"data: {json.dumps({'type': 'content', 'delta': msg['content']})}\n\n"
                    eval_count = data.get("eval_count", 0)
                    total_duration = data.get("total_duration", 0)
                except Exception:
                    pass

            latency = round(time.time() - start_time, 3)
            thinking = "".join(thinking_parts).strip()
            content = "".join(content_parts).strip()

            # Execute script in isolated Docker sandbox if Python code block is present and sandbox execution is requested
            code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", content)
            code_to_run = None
            if code_match:
                candidate = code_match.group(1).strip()
                if candidate and len(candidate.splitlines()) >= 3:
                    code_to_run = candidate
            elif not code_to_run:
                user_code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", user_message)
                if user_code_match:
                    code_to_run = user_code_match.group(1).strip()
                elif any(user_message.strip().startswith(prefix) for prefix in ("import ", "from ", "def ", "#")):
                    code_lines = [line for line in user_message.splitlines() if not re.search(r"\b(can you run|please run|run this in docker)\b", line, re.I)]
                    candidate = "\n".join(code_lines).strip()
                    if len(candidate.splitlines()) >= 3:
                        code_to_run = candidate

            notebook_info = None
            if run_sandbox and code_to_run:
                try:
                    from backend.tools.sandbox_tools import run_python_in_sandbox
                    from backend.notebooks.attach_to_notebook import is_anatase_water_molecular_prompt
                    
                    data_mount = _get_data_mount_path(user_message)
                    sb_res = run_python_in_sandbox(code_to_run, data_mount_path=data_mount)
                    
                    exit_code = sb_res.get("exit_code", 1)
                    execution_exit_code = exit_code
                    stdout = sb_res.get("stdout", "").strip()
                    stderr = sb_res.get("stderr", "").strip()
                    artifacts = sb_res.get("artifacts", [])
                    artifact_names = [os.path.basename(a) for a in artifacts]
                    
                    if artifacts:
                        yield f"data: {json.dumps({'type': 'artifacts', 'artifacts': artifact_names, 'artifact_paths': artifacts})}\n\n"
                    
                    status_badge = "Success (Exit Code 0)" if exit_code == 0 else f"Failed (Exit Code {exit_code})"
                    out_display = stdout if stdout else (stderr if stderr else "No console output.")
                    
                    sb_section = (
                        f"\n\n---\n\n"
                        f"### 🐳 Secure Docker Sandbox Execution\n"
                        f"- **Container Status**: {status_badge}\n"
                        f"- **Environment**: Isolated `chatbox-sandbox:latest` (CPU, `--network none`, `--memory 2g`, `--pids-limit 50`, `--read-only`)\n"
                    )
                    if artifact_names:
                        sb_section += f"- **Generated Artifacts**: {', '.join(artifact_names)}\n"
                    sb_section += f"\n**Container Output**:\n```text\n{out_display}\n```"
                    
                    if exit_code != 0 and use_web:
                        try:
                            err_lines = [l.strip() for l in stderr.splitlines() if l.strip()]
                            err_query = err_lines[-1] if err_lines else "Python ASE simulation error"
                            web_diag = self.search_web(f"Python {err_query}", max_results=2)
                            if web_diag and not web_diag[0].get("error"):
                                diag_items = "\n".join(f"- [{h.get('title', 'Ref')}]({h.get('href', '')}): {h.get('body', '')}" for h in web_diag if h.get("title"))
                                if diag_items:
                                    sb_section += f"\n\n**🌐 Web Diagnostic Insights**:\n{diag_items}"
                        except Exception:
                            pass
                    
                    content += sb_section
                    yield f"data: {json.dumps({'type': 'content', 'delta': sb_section})}\n\n"
                    
                    # Attach to Colab running.ipynb ONLY if not molecular extxyz
                    is_mol = is_anatase_water_molecular_prompt(user_message)
                    if not is_mol:
                        try:
                            from backend.notebooks.attach_to_notebook import attach_script_to_notebook
                            notebook_info = attach_script_to_notebook(code_to_run)
                            notice = (
                                f"\n\n"
                                f"📓 **Colab Notebook Ready**: This script has been automatically attached to cell #{notebook_info['target_cell_index'] + 1} "
                                f"in `ChatBot/backend/notebooks/running.ipynb`. You can now open and run the notebook in Google Colab (or on Kaggle with T4x2 GPU)!"
                            )
                            content += notice
                            yield f"data: {json.dumps({'type': 'content', 'delta': notice})}\n\n"
                        except Exception as e:
                            notebook_info = {"status": "error", "error": str(e)}
                except Exception as e:
                    notice_err = f"\n\n---\n\n*Sandbox Notice: {e}*"
                    content += notice_err
                    yield f"data: {json.dumps({'type': 'content', 'delta': notice_err})}\n\n"

            if content and thinking:
                final_text = f"<details class=\"thinking-details\"><summary><em>Thinking Process ({len(thinking.split())} words)</em></summary>\n\n{thinking}\n</details>\n\n{content}"
            elif content:
                final_text = content
            elif thinking:
                final_text = thinking
            else:
                final_text = "No response generated."

            insert_message(
                session_id=session_id,
                role="assistant",
                content=final_text,
                agent_used="SingleShotAgent",
                tool_calls_json=json.dumps({"memory_quality": {"execution_required": bool(run_sandbox and code_match), "exit_code": execution_exit_code}})
            )

            try:
                from backend.rag.chat_memory import get_chat_memory_store
                get_chat_memory_store().index_turn(
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=final_text,
                    agent_used="SingleShotAgent",
                    execution_required=bool(run_sandbox and code_match), exit_code=execution_exit_code
                )
            except Exception:
                pass

            yield f"data: {json.dumps({'type': 'done', 'latency_seconds': latency, 'eval_count': eval_count, 'session_id': session_id, 'full_response': final_text})}\n\n"

        except requests.exceptions.ConnectionError:
            err_msg = f"Could not connect to Ollama at {self.base_url}. Make sure 'ollama serve' is running."
            yield f"data: {json.dumps({'type': 'error', 'error': err_msg})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

if __name__ == "__main__":
    import sys
    agent = SingleShotAgent()
    print("=" * 72)
    print("🔬 Scientific Baseline Agent CLI")
    print(f"Model: {agent.model} | Endpoint: {agent.base_url}")
    print("Capabilities: Multi-Turn Memory, Docker Sandbox Execution, ASE Documentation RAG, Web Search Tools")
    print("Type your message below. Type 'exit', 'quit', or 'q' to end.")
    print("=" * 72)

    cli_session_id = insert_session(title="CLI Baseline Session")
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
                        if data.get("type") == "content":
                            sys.stdout.write(data.get("delta", ""))
                            sys.stdout.flush()
                        elif data.get("type") == "done":
                            print(f"\n\n[Done — ⏱️ {data.get('latency_seconds', 0.0)}s]\n")
                        elif data.get("type") == "error":
                            print(f"\n❌ Error: {data.get('error')}\n")
                    except Exception:
                        pass
    except KeyboardInterrupt:
        print("\n\nSession terminated by user.")
