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


def test_evidence_above_threshold(store, query_vector):
    assert retrieval.has_evidence(retrieval.retrieve(store, query_vector(0.0), k=2))


def test_no_evidence_below_threshold(store, query_vector):
    angle = float(np.arccos(config.RETRIEVAL_THRESHOLD - 0.05))
    assert not retrieval.has_evidence(retrieval.retrieve(store, query_vector(angle), k=2))


def test_confidence_of_empty_results_is_zero():
    assert retrieval.confidence([]) == 0.0
