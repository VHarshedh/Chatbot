from typing import Optional
import requests
from backend.config import settings
from backend.rag.hybrid_retriever import HybridRetriever
from backend.tools.sandbox_tools import run_python_in_sandbox
from backend.tools.web_tools import search_web

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
        
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_ctx": settings.NUM_CTX,
                    "temperature": 0.1,
                    "num_gpu": settings.NUM_GPU,
                }
            }
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            if "message" in data:
                return data["message"].get("content", "")
            return data.get("response", "")
        except Exception as e:
            # Fallback to /api/generate for backward compatibility or unit test mocks
            try:
                gen_payload = {
                    "model": self.model,
                    "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                    "stream": False
                }
                fallback_res = requests.post(
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
        context = self.retriever.retrieve(user_request)
        
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

        # 3. Execution & Self-Correction Loop
        max_retries = 3
        for attempt in range(max_retries):
            print(f"Executing script (Attempt {attempt+1}/{max_retries})...")
            result = run_python_in_sandbox(code, data_mount_path=data_path)
            
            if result["exit_code"] == 0:
                print("Execution successful!")
                return {
                    "status": "success",
                    "stdout": result["stdout"],
                    "artifacts": result["artifacts"],
                    "code_used": code
                }
            
            print(f"Execution failed with code {result['exit_code']}. Stderr: {result['stderr']}")
            
            # Optional: Use web search if the error is tricky
            if "AttributeError" in result['stderr'] or "ImportError" in result['stderr']:
                err_lines = [line.strip() for line in result["stderr"].splitlines() if line.strip()]
                search_query = err_lines[-1] if err_lines else "Python ASE AttributeError"
                web_context = search_web(f"Python {search_query} solution", max_results=2)
            else:
                web_context = "No additional web context retrieved."

            # Fix code
            fix_prompt = f"""The following Python script failed execution in the sandbox. Fix the script based on the error output and web search context.

Original Code:
{code}

Error Output (stderr):
{result['stderr']}

Web Search Diagnostic Context:
{web_context}

Output the complete, corrected Python script:"""
            print("Generating fix...")
            code = self._call_llm(fix_prompt, system_prompt=ANALYSIS_FIX_SYSTEM_PROMPT)
            code = code.replace("```python", "").replace("```", "").strip()
            
        return {
            "status": "failed",
            "stderr": result["stderr"],
            "code_used": code
        }

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
