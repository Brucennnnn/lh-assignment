import numpy as np

from src import config, retrieval


def test_top_result_is_most_similar(store, query_vector):
    results = retrieval.retrieve(store, query_vector(0.0), k=2)
    assert results[0]["source"] == "leave_policy.md"
    assert results[0]["score"] == 1.0
    assert results[1]["score"] == 0.0


def test_metadata_and_scores_preserved(store, query_vector):
    top = retrieval.retrieve(store, query_vector(0.0), k=1)[0]
    assert top["document_id"] == "doc_001"
    assert top["metadata"] == {"type": "policy", "department": "HR"}
    assert "score" in top and "content" in top


def test_respects_top_k(store, query_vector):
    assert len(retrieval.retrieve(store, query_vector(0.0), k=1)) == 1


def test_confidence_is_top_score(store, query_vector):
    results = retrieval.retrieve(store, query_vector(np.pi / 3), k=2)  # cos 60 deg
    assert retrieval.confidence(results) == 0.5


def test_qualifying_keeps_chunks_above_threshold(store, query_vector):
    results = retrieval.retrieve(store, query_vector(0.0), k=2)
    assert [r["score"] for r in results] == [1.0, 0.0]      # raw top-k, unfiltered
    assert [r["score"] for r in retrieval.qualifying(results)] == [1.0]


def test_qualifying_empty_below_threshold(store, query_vector):
    angle = float(np.arccos(config.RETRIEVAL_THRESHOLD - 0.05))
    assert retrieval.qualifying(retrieval.retrieve(store, query_vector(angle), k=2)) == []


def test_weak_chunks_never_reach_the_model(store, query_vector):
    """The 0.32 / 0.01 / 0.01 case: one chunk qualifies, the noise is dropped."""
    results = [{"score": 0.32}, {"score": 0.01}, {"score": 0.01}, {"score": 0.01}]
    assert retrieval.qualifying(results) == [{"score": 0.32}]


def test_threshold_is_inclusive(store):
    assert retrieval.qualifying([{"score": config.RETRIEVAL_THRESHOLD}]) != []


def test_confidence_of_empty_results_is_zero():
    assert retrieval.confidence([]) == 0.0
