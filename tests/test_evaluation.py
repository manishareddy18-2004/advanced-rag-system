import unittest

from app.evaluation import evaluate_retrieval


class RetrievalEvaluationTests(unittest.TestCase):

    def test_retrieval_returns_scores(self):
        result = evaluate_retrieval(
            "What is machine learning?",
            [
                {
                    "text": (
                        "Machine learning is a field "
                        "of artificial intelligence."
                    )
                }
            ]
        )

        self.assertEqual(
            result["documents_retrieved"],
            1
        )

        self.assertEqual(
            len(result["relevance_scores"]),
            1
        )

    def test_empty_retrieval(self):
        result = evaluate_retrieval(
            "What is machine learning?",
            []
        )

        self.assertEqual(
            result["documents_retrieved"],
            0
        )

        self.assertEqual(
            result["relevance_scores"],
            []
        )


if __name__ == "__main__":
    unittest.main()