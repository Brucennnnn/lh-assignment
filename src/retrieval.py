"""Semantic retrieval over the vector store, metadata preserved end to end."""
import numpy as np

from . import config
from .store import VectorStore


def retrieve(store: VectorStore, query_vector: np.ndarray, k: int | None = None) -> list[dict]:
    """Return the top-k chunks with their cosine score, source and metadata."""
    return store.search(query_vector, k or config.TOP_K)


def confidence(results: list[dict]) -> float:
    """Evidence strength = similarity of the best-matching chunk.

    Answers "do we hold a document that covers this question", which is a
    question about the corpus - never about whether the question was in scope.
    Heuristic and retrieval-based, not a calibrated probability.
    """
    return results[0]["score"] if results else 0.0


def has_evidence(results: list[dict]) -> bool:
    return confidence(results) >= config.RETRIEVAL_THRESHOLD
