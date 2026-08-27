"""Semantic retrieval over the vector store, metadata preserved end to end."""
import numpy as np

from . import config
from .store import VectorStore


def retrieve(store: VectorStore, query_vector: np.ndarray, k: int | None = None) -> list[dict]:
    """The raw top-k with scores and metadata. Unfiltered, so the log keeps the
    near-misses - those are what you tune the threshold against."""
    return store.search(query_vector, k or config.TOP_K)


def qualifying(results: list[dict]) -> list[dict]:
    """The chunks strong enough to answer from.

    One rule, one number, applied per chunk: a chunk that is too weak to justify
    answering is also too weak to inform the answer. TOP_K is therefore a
    maximum, not a quota - a question with one good chunk gets one, and the
    model is never handed noise it cannot tell apart from evidence.
    """
    return [r for r in results if r["score"] >= config.RETRIEVAL_THRESHOLD]


def confidence(results: list[dict]) -> float:
    """Evidence strength = similarity of the best-matching chunk, filtered or not.

    Answers "do we hold a document that covers this question", never "was the
    question in scope". Heuristic, not a calibrated probability.
    """
    return results[0]["score"] if results else 0.0
