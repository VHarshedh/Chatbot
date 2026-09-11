import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "batiai/gemma4-e2b:q4"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    CHROMA_PERSIST_DIR: str = "chroma_db"
    KUZU_DB_DIR: str = "kuzu_db"
    SQLITE_PATH: str = "chatbox.db"
    HPC_HOST: str = "localhost"
    HPC_USER: str = "user"
    HPC_KEY_PATH: str = "~/.ssh/id_rsa"
    WORKSPACE_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # GPU control: -1 = use all available GPU layers (default, uses AMD Radeon via ROCm/HIP),
    # 0 = CPU-only mode. Override with NUM_GPU=0 environment variable to force CPU.
    NUM_GPU: int = -1

settings = Settings()
