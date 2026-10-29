"""Initialize the registered finetuned model without rebuilding cached weights."""

import requests

from backend.config import settings
from backend.merge_finetuned_models import (
    TARGET_OLLAMA_MODEL,
    register_unified_model_in_ollama,
)


def initialize_finetuned_model():
    result = register_unified_model_in_ollama(
        target_model_name=TARGET_OLLAMA_MODEL,
        base_ollama_model=settings.MODEL_NAME,
    )
    if result.get("status") != "success":
        raise RuntimeError(f"Finetuned model initialization failed: {result}")

    # An empty chat loads the existing registered weights without generating text.
    # Match inference options so the first request can reuse the loaded runner.
    response = requests.post(
        f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        json={
            "model": result["active_model"],
            "messages": [],
            "stream": False,
            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
            "options": {"num_ctx": settings.NUM_CTX, "num_gpu": settings.NUM_GPU},
        },
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("error") or not data.get("done"):
        raise RuntimeError(f"Finetuned model preload failed: {data}")
    settings.MODEL_NAME = result["active_model"]
    return result
