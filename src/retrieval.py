"""Semantic retrieval over the vector store, metadata preserved end to end."""
from . import config, llm
from .store import VectorStore


def retrieve(store: VectorStore, query: str, k: int | None = None) -> list[dict]:
    """Return the top-k chunks with their cosine score, source and metadata."""
    query_vector = llm.embed([query])[0]
    return store.search(query_vector, k or config.TOP_K)


def confidence(results: list[dict]) -> float:
    """Retrieval confidence = similarity of the best-matching chunk.

    Heuristic and retrieval-based, not a calibrated probability. It answers
    "did we find anything that looks like the question" - which is the failure
    mode we care about - not "is the generated answer correct".
    """
    return results[0]["score"] if results else 0.0


def is_confident(results: list[dict]) -> bool:
    return confidence(results) >= config.RETRIEVAL_THRESHOLD
