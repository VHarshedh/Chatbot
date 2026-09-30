import requests
from backend.config import settings
from backend.rag.hybrid_retriever import HybridRetriever
from backend.tools.sandbox_tools import run_python_in_sandbox
from backend.tools.web_tools import search_web

class AnalysisAgent:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.model = settings.MODEL_NAME
        self.base_url = settings.OLLAMA_BASE_URL
        
    def _call_llm(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error connecting to LLM: {e}"

    def run_analysis(self, user_request: str, data_path: str = None) -> dict:
        print(f"Analysis Agent processing: {user_request}")
        
        # 1. Retrieve Context (Agentic RAG)
        context = self.retriever.retrieve(user_request)
        
        # 2. Write Script
        prompt = f"""You are a scientific Python developer. 
Get straight to the point and do not beat around the bush.
Write a Python script using ASE, NumPy, and Pandas to fulfill the user's analysis request.
The script will run in a sandbox.
If you need a trajectory, assume it's mounted at '{data_path if data_path else '/mnt/data/traj.extxyz'}'.
Save any plots to the current working directory as PNG files.
Only output the Python code, nothing else (no markdown blocks, no conversational preamble or pleasantries).

Context:
{context}

User Request: {user_request}"""

        print("Generating code...")
        code = self._call_llm(prompt)
        
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
                search_query = result['stderr'].split('\n')[-2] # roughly the error line
                web_context = search_web(f"Python {search_query} solution", max_results=2)
            else:
                web_context = "No additional web context retrieved."

            # Fix code
            fix_prompt = f"""The following Python script failed with an error. 
Fix the script based on the error output and web search context.
Only output the fixed Python code.

Original Code:
{code}

Error Output:
{result['stderr']}

Web Search Context:
{web_context}
"""
            print("Generating fix...")
            code = self._call_llm(fix_prompt)
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
