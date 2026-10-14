from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP, UPLOAD_DIR
from app.database import save_document, save_query
from app.ollama_client import generate_answer
from app.vector_store import store_chunks, search_chunks
from app.knowledge_graph import add_text_to_graph
from app.evaluation import evaluate_retrieval


splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


def index_text(title, text, source="text"):
    text = text.strip()

    if not text:
        raise ValueError("Document text cannot be empty.")

    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("No text chunks could be created.")

    count = store_chunks(
        chunks=chunks,
        source=source,
        title=title
    )

    save_document(title, source, count)
    add_text_to_graph(text, source)

    return {
        "title": title,
        "source": source,
        "chunks_indexed": count
    }


def index_pdf(filename, content):
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    if not content.startswith(b"%PDF-"):
        raise ValueError("The uploaded file does not appear to be a PDF.")

    safe_name = Path(filename).name
    destination = UPLOAD_DIR / safe_name

    destination.write_bytes(content)

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(destination))

        text_parts = [
            page.extract_text() or ""
            for page in reader.pages
        ]

        text = "\n".join(text_parts).strip()

        if not text:
            raise ValueError(
                "No extractable text was found in this PDF."
            )

        return index_text(
            title=safe_name,
            text=text,
            source=safe_name
        )

    finally:
        destination.unlink(missing_ok=True)


def ask_question(question, top_k=4):
    retrieved = search_chunks(question, top_k=top_k)

    if not retrieved:
        raise ValueError(
            "No relevant document chunks were retrieved. "
            "Index a document first."
        )

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in retrieved
    )

    answer = generate_answer(
        question=question,
        context=context
    )

    save_query(question, answer)

    return {
        "question": question,
        "answer": answer,
        "sources": sorted({
            item["source"]
            for item in retrieved
        }),
        "evaluation": evaluate_retrieval(
            question,
            retrieved
        )
    }