import re
import time
import json
import requests
from typing import Dict, Any, Optional, List
from backend.config import settings
from backend.db.models import insert_session, insert_message, execute_query

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert computational materials scientist and chemist. "
    "Get straight to the point and do not beat around the bush. "
    "Provide clear, concise, and scientifically rigorous answers with zero filler, conversational fluff, or preamble. "
    "When asked for simulation scripts (such as ASE, Quantum ESPRESSO, LAMMPS, or DeePMD), "
    "provide clean, syntactically correct, and complete code blocks immediately."
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
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Directly queries the Ollama /api/chat endpoint with no tool or RAG intervention.
        Streams chunks with num_ctx=4096 and timeout=(10, 600) to prevent read timeouts on CPU reasoning.
        """
        sys_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        start_time = time.time()

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
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
                "num_ctx": settings.NUM_CTX,
                "num_predict": -1,
                "temperature": 0.2,
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

            # Automatically attach script to Colab running.ipynb if Python code block is present
            code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", content)
            if code_match:
                extracted = code_match.group(1).strip()
                if extracted and len(extracted.splitlines()) >= 3:
                    try:
                        from backend.notebooks.attach_to_notebook import attach_script_to_notebook
                        nb_res = attach_script_to_notebook(extracted)
                        content += (
                            f"\n\n---\n\n"
                            f"📓 **Colab Notebook Ready**: This script has been automatically attached to cell #{nb_res['target_cell_index'] + 1} "
                            f"in `ChatBot/backend/notebooks/running.ipynb`. You can now open and run the notebook in Google Colab (or on Kaggle with T4x2 GPU)!"
                        )
                    except Exception:
                        pass

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
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Processes a chat turn, tracks session history in SQLite, and returns the response.
        """
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

        # Generate direct single-shot response
        result = self.generate_response(user_message, system_prompt=system_prompt, images=images)

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

    def chat_stream(
        self,
        user_message: str,
        session_id: Optional[int] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None
    ):
        """
        Streams a chat response token-by-token via Server-Sent Events (SSE) format: data: {...}\n\n
        Persists user and assistant messages in SQLite upon stream completion.
        """
        start_time = time.time()
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
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
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
                "num_ctx": settings.NUM_CTX,
                "num_predict": -1,
                "temperature": 0.2,
                "num_gpu": settings.NUM_GPU
            }
        }

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

            # Automatically attach script to Colab running.ipynb if Python code block is present
            code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", content)
            notebook_info = None
            if code_match:
                extracted = code_match.group(1).strip()
                if extracted and len(extracted.splitlines()) >= 3:
                    try:
                        from backend.notebooks.attach_to_notebook import attach_script_to_notebook
                        notebook_info = attach_script_to_notebook(extracted)
                        notice = (
                            f"\n\n---\n\n"
                            f"📓 **Colab Notebook Ready**: This script has been automatically attached to cell #{notebook_info['target_cell_index'] + 1} "
                            f"in `ChatBot/backend/notebooks/running.ipynb`. You can now open and run the notebook in Google Colab (or on Kaggle with T4x2 GPU)!"
                        )
                        content += notice
                        yield f"data: {json.dumps({'type': 'content', 'delta': notice})}\n\n"
                    except Exception as e:
                        notebook_info = {"status": "error", "error": str(e)}

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
                agent_used="SingleShotAgent"
            )

            yield f"data: {json.dumps({'type': 'done', 'latency_seconds': latency, 'eval_count': eval_count, 'session_id': session_id, 'full_response': final_text})}\n\n"

        except requests.exceptions.ConnectionError:
            err_msg = f"Could not connect to Ollama at {self.base_url}. Make sure 'ollama serve' is running."
            yield f"data: {json.dumps({'type': 'error', 'error': err_msg})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

if __name__ == "__main__":
    agent = SingleShotAgent()
    print(f"Testing SingleShotAgent with model {agent.model}...")
    res = agent.generate_response("What is the difference between plane-wave cutoff and density cutoff in Quantum ESPRESSO?")
    print(f"Status: {res['status']} | Latency: {res.get('latency_seconds')}s")
    print(res.get("response", res.get("error")))
