import time
import requests
from typing import Dict, Any, Optional
from backend.config import settings
from backend.db.models import insert_session, insert_message, execute_query

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert computational materials scientist and chemist. "
    "Provide clear, accurate, and scientifically rigorous answers. "
    "When asked for simulation scripts (such as Quantum ESPRESSO, LAMMPS, or DeePMD), "
    "provide clean, syntactically correct input files and explain the underlying physics."
)

class SingleShotAgent:
    """
    A direct, pure single-shot evaluation agent.
    Runs strictly WITHOUT tool calling, RAG retrieval, or web search
    to benchmark raw model accuracy and reasoning.
    """
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or settings.MODEL_NAME
        self.base_url = base_url or settings.OLLAMA_BASE_URL

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Directly queries the Ollama /api/generate endpoint with no tool or RAG intervention.
        """
        sys_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        start_time = time.time()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": sys_prompt,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            data = response.json()
            latency = round(time.time() - start_time, 3)

            return {
                "status": "success",
                "response": data.get("response", ""),
                "model": self.model,
                "latency_seconds": latency,
                "mode": "single_shot",
                "total_duration_ns": data.get("total_duration", 0),
                "eval_count": data.get("eval_count", 0)
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

    def chat(self, user_message: str, session_id: Optional[int] = None, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes a chat turn, tracks session history in SQLite, and returns the response.
        """
        # Create session if none provided
        if not session_id:
            title = user_message[:40] + ("..." if len(user_message) > 40 else "")
            session_id = insert_session(title=title)

        # Log user message
        insert_message(
            session_id=session_id,
            role="user",
            content=user_message,
            agent_used="SingleShotAgent"
        )

        # Generate direct single-shot response
        result = self.generate_response(user_message, system_prompt=system_prompt)

        # Log assistant message
        if result["status"] == "success":
            insert_message(
                session_id=session_id,
                role="assistant",
                content=result["response"],
                agent_used="SingleShotAgent"
            )

        result["session_id"] = session_id
        return result

if __name__ == "__main__":
    agent = SingleShotAgent()
    print(f"Testing SingleShotAgent with model {agent.model}...")
    res = agent.generate_response("What is the difference between plane-wave cutoff and density cutoff in Quantum ESPRESSO?")
    print(f"Status: {res['status']} | Latency: {res.get('latency_seconds')}s")
    print(res.get("response", res.get("error")))
