from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_retrieval(question, retrieved_documents):
    documents = [
        item.get("text", "")
        for item in retrieved_documents
        if item.get("text", "").strip()
    ]

    if not documents:
        return {
            "question": question,
            "documents_retrieved": 0,
            "relevance_scores": []
        }

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    try:
        matrix = vectorizer.fit_transform(
            [question] + documents
        )

        scores = cosine_similarity(
            matrix[0:1],
            matrix[1:]
        )[0]

    except ValueError:
        scores = [0.0] * len(documents)

    return {
        "question": question,
        "documents_retrieved": len(documents),
        "relevance_scores": [
            round(float(score), 4)
            for score in scores
        ]
    }