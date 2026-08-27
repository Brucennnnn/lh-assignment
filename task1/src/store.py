"""In-memory vector store: a normalised matrix and a dot product.

~50 chunks. A vector database would add a dependency and a service for no
measurable gain at this size.
"""
import numpy as np


class VectorStore:
    def __init__(self, chunks: list[dict], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        self.chunks = chunks
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, k: int) -> list[dict]:
        # Both sides are L2-normalised, so the dot product is cosine similarity.
        scores = self.vectors @ query_vector
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 4)} for i in top]
