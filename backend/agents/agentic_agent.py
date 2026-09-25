import json
import re
import time
import requests
from typing import Dict, Any, Generator, Optional, List
from backend.config import settings
from backend.rag.hybrid_retriever import HybridRetriever
from backend.tools.sandbox_tools import run_python_in_sandbox
from backend.tools.web_tools import search_web
from backend.db.models import insert_session, insert_message

class AgenticRAGAgent:
    """
    Autonomous Scientific AI Agent with full capabilities:
    1. Hybrid RAG: ChromaDB Vector Store + Kùzu Knowledge Graph
    2. Secure Sandbox: Isolated Docker execution with automated self-correction
    3. Web Search: DuckDuckGo documentation and syntax lookup
    """

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model or settings.MODEL_NAME
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.retriever = HybridRetriever()

    def _is_execution_request(self, text: str) -> bool:
        """Determines if the user query asks to write and run code or perform simulation analysis"""
        lower = text.lower()
        patterns = [
            r"\b(run|execute|compute|calculate|simulate|plot)\b",
            r"\b(python script|code block|simulation script)\b",
            r"\b(density profile|radial distribution|rdf|survival probability)\b",
            r"\b(extxyz|trajectory|deepmd model|lammps)\b"
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
        images: Optional[List[str]] = None
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_msg = {"role": "user", "content": prompt}
        cleaned_imgs = self._clean_images(images)
        if cleaned_imgs:
            user_msg["images"] = cleaned_imgs
        messages.append(user_msg)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_ctx": settings.NUM_CTX,
                "num_predict": -1,
                "temperature": 0.2,
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
            context = str(self.retriever.retrieve(user_message))
            rag_snippet = context[:400] + ("..." if len(context) > 400 else "")
            rag_step = {"type": "agent_step", "step": "rag_complete", "title": "Hybrid RAG Retrieval", "status": "completed", "details": rag_snippet}
            trace_steps.append(rag_step)
            yield f"data: {json.dumps(rag_step)}\n\n"
        except Exception as e:
            rag_err = {"type": "agent_step", "step": "rag_error", "title": "Hybrid RAG Retrieval", "status": "warning", "details": f"Retrieval notice: {e}"}
            trace_steps.append(rag_err)
            yield f"data: {json.dumps(rag_err)}\n\n"

        # Step 2: Check for execution request (Sandbox + Web Search)
        execution_requested = self._is_execution_request(user_message)
        sandbox_output = ""
        artifacts = []
        code_extracted = ""

        thinking_parts = []
        content_parts = []
        eval_count = 0

        if execution_requested:
            cg_start = {"type": "agent_step", "step": "code_gen_start", "title": "Script Synthesis", "status": "running", "details": "Synthesizing Python analysis script using running.ipynb context & RAG..."}
            trace_steps.append(cg_start)
            yield f"data: {json.dumps(cg_start)}\n\n"

            nb_context_section = ""
            try:
                from backend.notebooks.attach_to_notebook import extract_notebook_context, format_notebook_context_prompt
                nb_ctx = extract_notebook_context()
                nb_context_section = f"{format_notebook_context_prompt(nb_ctx)}\n\n"
            except Exception:
                pass

            code_prompt = f"""You are an expert computational materials science Python developer.
Based on the following execution context from the target notebook `running.ipynb` and retrieved domain context from workspace Jupyter notebooks, write a complete, robust, executable Python script to address the user request.

{nb_context_section}--- RETRIEVED WORKSPACE NOTEBOOK CONTEXT ---
{context}

--- USER REQUEST ---
{user_message}

CRITICAL IMPLEMENTATION REQUIREMENTS:
1. Ground your script on the target notebook `running.ipynb` (pre-loaded `traj` variable in memory from `/content/MPtrj/...`).
2. For interfacial site identification, neighbor searches, and distance cutoffs, you MUST use `ase.neighborlist` (specifically `from ase.neighborlist import neighbor_list, natural_cutoffs` or `NeighborList`).
3. For multi-model DP averaging, reference the trained DeepMD potential models located at `Training/DP_Models/DPMD_Water/1/`, `Training/DP_Models/DPMD_Water/2/`, and `Training/DP_Models/DPMD_Water/3/`.
4. Variable `traj` is already loaded in `running.ipynb`. You can safely access `traj` directly or load from `traj_file`.
5. Get straight to the point and do not beat around the bush. Do not include introductory pleasantries or explanations.
6. Output ONLY valid executable Python code in a ```python ... ``` block. Use ASE, NumPy, and Matplotlib."""

            try:
                code_response_parts = []
                res = self._call_llm(code_prompt, stream=True, timeout=3600)
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

                # Automatically attach script to Colab running.ipynb
                if code_extracted:
                    try:
                        from backend.notebooks.attach_to_notebook import attach_script_to_notebook
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
                
                sandbox_res = run_python_in_sandbox(code_extracted)
                if sandbox_res["exit_code"] == 0:
                    sandbox_output = sandbox_res["stdout"]
                    artifacts = sandbox_res.get("artifacts", [])
                    sb_done = {"type": "agent_step", "step": "sandbox_complete", "title": "Docker Sandbox Execution", "status": "completed", "details": f"Exit code 0 (Success). Output: {sandbox_output[:200]}"}
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
                    fix_res = self._call_llm(fix_prompt, timeout=3600)
                    fixed_code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", fix_res.json().get("message", {}).get("content", ""))
                    if fixed_code_match:
                        code_extracted = fixed_code_match.group(1).strip()
                        try:
                            from backend.notebooks.attach_to_notebook import attach_script_to_notebook
                            attach_script_to_notebook(code_extracted)
                        except Exception:
                            pass
                    
                    # Second execution attempt
                    retry_res = run_python_in_sandbox(code_extracted)
                    sandbox_output = retry_res.get("stdout", "") or retry_res.get("stderr", "")
                    artifacts = retry_res.get("artifacts", [])
                    retry_exit = retry_res.get("exit_code", 1)
                    sb_re_done = {"type": "agent_step", "step": "sandbox_complete", "title": "Docker Sandbox Re-execution", "status": "completed", "details": f"Self-correction finished (Exit {retry_exit})."}
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

        synthesis_prompt = f"""You are SciCompute AI, an autonomous scientific materials & chemistry assistant.
You possess direct access to:
1. Grounded workspace Jupyter notebooks (.ipynb) and trained DeepMD models in `Training/DP_Models/DPMD_Water/1/`, `Training/DP_Models/DPMD_Water/2/`, and `Training/DP_Models/DPMD_Water/3/`.
2. A secure Docker Sandbox Execution Environment where code is tested and verified on CPU/GPU.
3. Web search tools (via DuckDuckGo) for real-time documentation and syntax verification.

NEVER claim you lack an execution environment, cannot run code, or cannot access files.

--- RETRIEVED HYBRID RAG CONTEXT ---
{context}

--- EXECUTION ARTIFACTS & OUTPUT ---
{f'Execution Output:\n{sandbox_output}' if sandbox_output else 'Sandboxed environment and tools are active and available.'}
{f'Generated Code:\n```python\n{code_extracted}\n```' if code_extracted else ''}

User Query:
{user_message}

Get straight to the point and do not beat around the bush. Eliminate pleasantries and conversational filler. Provide a well-structured, rigorous response with clear headings, KaTeX LaTeX math formulas ($...$ for inline, $$...$$ for display blocks), and code blocks using `ase.neighborlist` where appropriate. Conclude by explicitly informing the user that their complete analysis script has been attached to `ChatBot/backend/notebooks/running.ipynb` and they can now execute the notebook in Google Colab (or on Kaggle with T4x2 GPU)."""

        if code_extracted and content_parts:
            divider = "\n\n---\n\n### Scientific Synthesis & Findings\n\n"
            content_parts.append(divider)
            yield f"data: {json.dumps({'type': 'content', 'delta': divider})}\n\n"

        try:
            default_sys = (
                "You are SciCompute AI, an autonomous scientific materials & chemistry assistant with direct access to "
                "retrieved workspace Jupyter notebooks (.ipynb), trained DeepMD models in Training/DP_Models/, "
                "a Docker execution sandbox, and web search tools. Get straight to the point and do not beat around the bush. "
                "Never state that you cannot execute code or lack environment access."
            )
            stream_res = self._call_llm(
                prompt=synthesis_prompt,
                system_prompt=system_prompt or default_sys,
                stream=True,
                timeout=3600,
                images=images
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
                tool_calls_json=json.dumps(trace_steps)
            )

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
