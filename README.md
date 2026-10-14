# Advanced RAG System

A modular Retrieval-Augmented Generation application that answers
questions using information retrieved from indexed documents.

## Technology Stack

- Python
- FastAPI
- Ollama
- Qdrant
- SQLite
- LangChain Text Splitters
- NetworkX
- scikit-learn

## Models

- LLM: qwen2.5:1.5b
- Embeddings: nomic-embed-text

## Architecture

1. Documents are submitted through the API.
2. Text is extracted and split into chunks.
3. Ollama generates embeddings.
4. Qdrant stores document vectors.
5. Relevant chunks are retrieved for each question.
6. Ollama generates an answer using the retrieved context.
7. SQLite records document metadata and query history.
8. A lightweight TF-IDF evaluation measures lexical relevance.

## Prerequisites

- Python 3.10 or later
- Ollama
- qwen2.5:1.5b
- nomic-embed-text

## Installation

Create a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Pull the models if necessary:

    ollama pull qwen2.5:1.5b
    ollama pull nomic-embed-text

## Run

    uvicorn app.main:app --host 127.0.0.1 --port 8000

Open:

    http://127.0.0.1:8000/docs

## API Endpoints

- GET /
- GET /health
- POST /index/text
- POST /index/pdf
- POST /ask
- GET /documents
- GET /history
- GET /knowledge-graph
- GET /vector-store/health

## Run Tests

    python -m unittest discover -s tests -v

## Notes

- Ollama must be running locally.
- Index documents before asking questions.
- Qdrant uses local persistent storage.
- The knowledge graph uses simple capitalized-phrase extraction.
- Retrieval evaluation uses TF-IDF cosine similarity and is not
  a comprehensive measure of answer correctness.
- Scanned PDFs may require OCR, which is not included.