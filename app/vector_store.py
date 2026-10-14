import uuid

from qdrant_client import QdrantClient, models

from app.config import (
    QDRANT_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)
from app.ollama_client import generate_embedding


client = QdrantClient(path=str(QDRANT_PATH))


def ensure_collection():
    if client.collection_exists(COLLECTION_NAME):
        return

    # Obtain the embedding dimension from the installed model.
    sample_embedding = generate_embedding(
        "Initialize the RAG vector collection."
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=len(sample_embedding),
            distance=models.Distance.COSINE
        )
    )

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD
    )


def store_chunks(chunks, source, title):
    ensure_collection()

    points = []

    for index, chunk in enumerate(chunks):
        vector = generate_embedding(chunk)

        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "source": source,
                    "title": title,
                    "chunk_index": index,
                    "embedding_model": EMBEDDING_MODEL
                }
            )
        )

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )

    return len(points)


def search_chunks(question, top_k=4):
    ensure_collection()

    query_vector = generate_embedding(question)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return [
        {
            "text": point.payload.get("text", ""),
            "source": point.payload.get("source", ""),
            "title": point.payload.get("title", ""),
            "score": point.score
        }
        for point in results
    ]