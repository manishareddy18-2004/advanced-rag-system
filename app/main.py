from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File

from app.config import LLM_MODEL, EMBEDDING_MODEL
from app.database import (
    initialize_database,
    get_documents,
    get_history
)
from app.ollama_client import check_ollama
from app.vector_store import ensure_collection
from app.rag_service import index_text, index_pdf, ask_question
from app.knowledge_graph import graph_summary
from app.schemas import TextIndexRequest, AskRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="Advanced RAG System",
    description=(
        "A local document question-answering system "
        "using Ollama, Qdrant, SQLite and FastAPI."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def home():
    return {
        "project": "Advanced RAG System",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    try:
        data = check_ollama()

        installed_models = {
            item.get("name")
            for item in data.get("models", [])
        }

        llm_available = any(
            name == LLM_MODEL
            or name.startswith(LLM_MODEL + ":")
            for name in installed_models
            if name
        )

        embedding_available = any(
            name == EMBEDDING_MODEL
            or name.startswith(EMBEDDING_MODEL + ":")
            for name in installed_models
            if name
        )

        return {
            "ollama": "connected",
            "llm_model": LLM_MODEL,
            "llm_available": llm_available,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_available": embedding_available
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama health check failed: {exc}"
        )


@app.post("/index/text")
def index_document(request: TextIndexRequest):
    try:
        return index_text(
            title=request.title,
            text=request.text
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


@app.post("/index/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()

        return index_pdf(
            filename=file.filename or "document.pdf",
            content=content
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    finally:
        await file.close()


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return ask_question(
            question=request.question,
            top_k=request.top_k
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


@app.get("/documents")
def documents():
    return {"documents": get_documents()}


@app.get("/history")
def history():
    return {"history": get_history()}


@app.get("/knowledge-graph")
def knowledge_graph():
    return graph_summary()


@app.get("/vector-store/health")
def vector_store_health():
    try:
        ensure_collection()
        return {"qdrant": "ready"}

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store check failed: {exc}"
        )