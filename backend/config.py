import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "batiai/gemma4-e2b:q4")
    # Keep the registered model resident across requests and backend reloads.
    OLLAMA_KEEP_ALIVE: int = -1
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    KUZU_DB_DIR: str = os.getenv("KUZU_DB_DIR", "kuzu_db")
    SQLITE_PATH: str = os.getenv("SQLITE_PATH", "chatbox.db")
    HPC_HOST: str = os.getenv("HPC_HOST", "localhost")
    HPC_USER: str = os.getenv("HPC_USER", "user")
    HPC_KEY_PATH: str = os.getenv("HPC_KEY_PATH", "~/.ssh/id_rsa")
    WORKSPACE_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # GPU control: -1 = use all available GPU layers (default, uses AMD Radeon via ROCm/HIP),
    # 0 = CPU-only mode. Override with NUM_GPU=0 environment variable to force CPU.
    NUM_GPU: int = int(os.getenv("NUM_GPU", "-1"))

    # Context window size: 8192 tokens optimal for AMD Radeon iGPU / mobile unified memory.
    # Override with NUM_CTX environment variable if needed.
    NUM_CTX: int = int(os.getenv("NUM_CTX", "8192"))

    # Gemini Evaluation Settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

settings = Settings()
