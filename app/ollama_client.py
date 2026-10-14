import requests

from app.config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL
)


def check_ollama():
    response = requests.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=5
    )
    response.raise_for_status()
    return response.json()


def generate_embedding(text):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=120
    )

    response.raise_for_status()
    data = response.json()

    embeddings = data.get("embeddings", [])

    if not embeddings:
        raise RuntimeError(
            "Ollama returned no embedding for the supplied text."
        )

    return embeddings[0]


def generate_answer(question, context):
    prompt = f"""
You are a helpful document question-answering assistant.

Answer the question using the supplied context.
If the context does not contain the answer, say that the
available documents do not provide enough information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    answer = response.json().get("response", "").strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")

    return answer