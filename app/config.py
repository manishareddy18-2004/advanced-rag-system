import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "advanced_rag_data"
UPLOAD_DIR = DATA_DIR / "uploads"
SQLITE_PATH = DATA_DIR / "rag_history.db"
QDRANT_PATH = DATA_DIR / "qdrant"

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen2.5:1.5b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

COLLECTION_NAME = "rag_documents"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 4

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)