import requests
from backend.config import settings

class OllamaEmbeddings:
    def __init__(self, model: str = settings.EMBEDDING_MODEL, base_url: str = settings.OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        total = len(texts)
        if total == 0:
            return []

        # Batch size of 25 texts per request gives ~15x-20x speedup in Ollama
        batch_size = 25

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            current_done = min(i + len(batch), total)

            if total > 1:
                percent = int((current_done / total) * 100)
                bar_len = 25
                filled = int(bar_len * current_done / total)
                bar = "=" * filled + "-" * (bar_len - filled)
                print(f"\rEmbedding chunks: [{bar}] {current_done}/{total} ({percent}%)", end="", flush=True)

            try:
                # 1. High-speed batch embedding using /api/embed
                response = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": batch},
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    if "embeddings" in data:
                        embeddings.extend(data["embeddings"])
                        continue
                    elif "embedding" in data:
                        # Handles single item / mock return
                        embeddings.append(data["embedding"])
                        continue
                elif response.status_code == 404:
                    err_msg = response.text
                    if "not found" in err_msg.lower() and "model" in err_msg.lower():
                        raise RuntimeError(
                            f"Model '{self.model}' was not found in Ollama.\n"
                            f"Please pull it first by running: ollama pull {self.model}"
                        )
            except requests.exceptions.ConnectionError:
                raise RuntimeError(
                    f"Could not connect to Ollama at {self.base_url}.\n"
                    f"Please make sure Ollama is running ('ollama serve')."
                )
            except Exception:
                pass

            # Fallback: embed sequentially if batching fails
            for text in batch:
                try:
                    r = requests.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                        timeout=30
                    )
                    r.raise_for_status()
                    data = r.json()
                    embeddings.append(data.get("embedding", []))
                except Exception as e:
                    raise RuntimeError(f"Embedding failed: {e}")

        if total > 1:
            print()  # Newline after progress bar completes
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
