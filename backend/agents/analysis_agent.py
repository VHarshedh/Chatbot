from typing import Optional
import requests
from backend.agents.cancellation import cancellable_post
from backend.config import settings
from backend.rag.hybrid_retriever import HybridRetriever
from backend.tools.sandbox_tools import run_python_in_sandbox
from backend.tools.web_tools import search_web, read_url_content
from backend.tools.registry import structure_context
from backend.tools.validation_tools import local_error_context

ANALYSIS_SYSTEM_PROMPT = (
    "SYSTEM DIRECTIVE: You are an expert computational materials scientist and autonomous Python developer "
    "operating inside an isolated E2B / Docker execution sandbox (chatbox-sandbox:latest).\n"
    "Your objective is to generate self-contained, robust, executable scientific Python scripts that run flawlessly on the first execution.\n"
    "\n"
    "EXECUTION & SANDBOX ENVIRONMENT RULES:\n"
    "1. Available packages: ASE (Atomic Simulation Environment 3.29.0), NumPy, SciPy, Matplotlib, Pandas.\n"
    "2. Data Grounding & ASE Syntax: Never assume variables are pre-loaded in memory. Always read trajectories directly using ASE `read(data_path, index=':')`.\n"
    "   CRITICAL: `read(data_path, index=':')` returns a Python list of Atoms objects (`list[ase.Atoms]`). "
    "A Python `list` does NOT have `.get_positions()` or `.cell` attributes! "
    "Always unpack or index: `traj = read(data_path, index=':'); atoms = traj[0] if isinstance(traj, list) else traj; positions = atoms.get_positions()`. "
    "Or read single frame: `atoms = read(data_path, index=0)`. Verify file existence and handle frames cleanly.\n"
    "3. Coordinate Filtering & Boundary Layer Analysis:\n"
    "   - NEVER use exact floating-point equality (`==`) on continuous coordinates (e.g. `positions[:, 2] == z_min` will collapse an entire atomic plane to 1 or 2 isolated points!).\n"
    "   - Slab Surface Boundaries along z: Always use a physical layer thickness tolerance window `tol` (0.5 to 1.0 Å). Identify surface layers via `positions[:, 2] <= (z_min + tol)` or `positions[:, 2] >= (z_max - tol)`.\n"
    "   - 3D Unit Cell Box Boundaries: Check fractional / scaled coordinates `scaled = atoms.get_scaled_positions(wrap=True)` within a skin depth `delta` (e.g. 0.02 to 0.05) across cell faces: `np.any((scaled < delta) | (scaled > 1.0 - delta), axis=1)`.\n"
    "   - Always print the count, chemical symbols, and coordinate arrays of the identified boundary atoms for verification.\n"
    "4. Interfacial Analysis: For neighbor lists, coordination, and cutoffs, use `ase.neighborlist` (e.g. `neighbor_list`, `natural_cutoffs`, or `NeighborList`).\n"
    "5. Headless Visualization: Save all plots to the current working directory as PNG files (`plt.savefig(..., dpi=300, bbox_inches='tight')`). "
    "Always call `plt.close()` after saving. NEVER call `plt.show()` (hangs headless containers).\n"
    "6. Output Contract: Output ONLY the complete, valid Python code block enclosed in ```python ... ```. "
    "Get straight to the point and do not beat around the bush. Eliminate conversational filler, explanations, or preambles."
)

ANALYSIS_FIX_SYSTEM_PROMPT = (
    "SYSTEM DIRECTIVE: You are an autonomous scientific code debugger and self-correction engine for the E2B / Docker execution sandbox.\n"
    "A previously generated script failed execution in the sandbox.\n"
    "1. Traceback Isolation: Identify the exact line and root cause in the stderr traceback.\n"
    "   - If 'AttributeError: 'list' object has no attribute 'get_positions'', unpack the ASE trajectory list: `atoms = traj[0] if isinstance(traj, list) else traj`.\n"
    "   - If boundary analysis returned only 1-2 atoms, replace exact float comparison `== z_min` with a physical layer tolerance `z <= z_min + tol` (e.g. tol = 0.8 Å) or fractional coordinates.\n"
    "2. Solution Synthesis: Apply targeted corrections using error diagnostics and web search context without regressing existing imports or functional logic.\n"
    "3. Output Contract: Output ONLY the complete, corrected, executable Python code in a ```python ... ``` block. "
    "Do not provide explanations or conversational apologies."
)

class AnalysisAgent:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.model = settings.MODEL_NAME
        self.base_url = settings.OLLAMA_BASE_URL

    def search_web(self, query: str, max_results: int = 3) -> list:
        """Searches the web via DuckDuckGo."""
        return search_web(query, max_results=max_results)

    def read_url_content(self, url: str) -> str:
        """Fetches and extracts text from a specified URL."""
        return read_url_content(url)
        
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None, stream: bool = False) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "keep_alive": settings.OLLAMA_KEEP_ALIVE,
                "messages": messages,
                "stream": stream,
                "options": {
                    "num_ctx": settings.NUM_CTX,
                    "temperature": 0.1,
                    "num_gpu": settings.NUM_GPU,
                }
            }
            response = cancellable_post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120, stream=stream
            )
            response.raise_for_status()
            if stream:
                from backend.agents.execution_repair import stream_repair_response
                return stream_repair_response(response)
            data = response.json()
            if "message" in data:
                return data["message"].get("content", "")
            return data.get("response", "")
        except Exception as e:
            # Fallback to /api/generate for backward compatibility or unit test mocks
            try:
                gen_payload = {
                    "model": self.model,
                    "keep_alive": settings.OLLAMA_KEEP_ALIVE,
                    "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                    "stream": stream
                }
                fallback_res = cancellable_post(
                    f"{self.base_url}/api/generate",
                    json=gen_payload,
                    timeout=120
                )
                fallback_res.raise_for_status()
                fdata = fallback_res.json()
                return fdata.get("response") or fdata.get("message", {}).get("content", "")
            except Exception:
                return f"Error connecting to LLM: {e}"

    def run_analysis(self, user_request: str, data_path: str = None) -> dict:
        print(f"Analysis Agent processing: {user_request}")
        
        # 1. Retrieve Context (Agentic RAG)
        context = self.retriever.retrieve(user_request) + "\n" + structure_context(user_request)
        
        # 2. Write Script
        traj_mount = data_path if data_path else '/mnt/data/traj.extxyz'
        prompt = f"""Write a self-contained scientific Python analysis script based on the following requirements:

Trajectory Mount Path: '{traj_mount}'
Save plots to current working directory as PNG files.

Retrieved Domain Context:
{context}

User Analysis Request:
{user_request}

Output the executable Python script in a ```python ... ``` block:"""

        print("Generating code...")
        code = self._call_llm(prompt, system_prompt=ANALYSIS_SYSTEM_PROMPT)
        
        # Clean markdown code blocks if the LLM output them anyway
        code = code.replace("```python", "").replace("```", "").strip()

        from backend.agents.execution_repair import execution_repairs
        execution_events = []
        for event in execution_repairs(code, user_request,
                lambda script: run_python_in_sandbox(script, data_mount_path=data_path, user_prompt=user_request),
                lambda prompt: self._call_llm(prompt, system_prompt=ANALYSIS_FIX_SYSTEM_PROMPT, stream=True)):
            if event["type"] != "execution_result":
                execution_events.append(event)
            if event["type"] == "execution_result":
                result = event["result"]
                return {**result, "status": "success" if result.get("exit_code") == 0 else "failed",
                        "code_used": event["code"], "attempts": event["attempts"], "repairs": event["repairs"], "execution_events": execution_events}

if __name__ == "__main__": # pragma: no cover
    import sys
    agent = AnalysisAgent()
    print("=" * 72)
    print("🧪 Self-Correcting Scientific Analysis Agent CLI")
    print(f"Model: {agent.model} | Endpoint: {agent.base_url}")
    print("Capabilities: RAG Context, Docker Sandbox Execution, Automated Self-Correction")
    print("Type your analysis request below. Type 'exit', 'quit', or 'q' to end.")
    print("=" * 72)
    print()

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

            print(f"\nProcessing request: '{user_query}'...")
            res = agent.run_analysis(user_query)

            if res.get("status") == "success":
                print("\n✅ Execution Succeeded!")
                print("\n--- Code Generated & Verified ---")
                print(res.get("code_used", "").strip())
                if res.get("stdout"):
                    print("\n--- Container Stdout ---")
                    print(res.get("stdout").strip())
                if res.get("artifacts"):
                    print(f"\n--- Generated Artifacts: {', '.join(res.get('artifacts'))} ---")
            else:
                print("\n❌ Execution Failed after retries:")
                print(res.get("stderr", "Unknown error").strip())
                print("\n--- Last Attempt Code ---")
                print(res.get("code_used", "").strip())
            print()
    except KeyboardInterrupt:
        print("\n\nSession terminated by user.")
