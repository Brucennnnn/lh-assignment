import numpy as np

from src import config, retrieval


def test_top_result_is_most_similar(store, fake_embed):
    fake_embed(0.0)  # query points exactly at chunk A
    results = retrieval.retrieve(store, "how much annual leave", k=2)
    assert results[0]["source"] == "leave_policy.md"
    assert results[0]["score"] == 1.0
    assert results[1]["score"] == 0.0


def test_metadata_and_scores_preserved(store, fake_embed):
    fake_embed(0.0)
    top = retrieval.retrieve(store, "leave", k=1)[0]
    assert top["document_id"] == "doc_001"
    assert top["metadata"] == {"type": "policy", "department": "HR"}
    assert "score" in top and "content" in top


def test_respects_top_k(store, fake_embed):
    fake_embed(0.0)
    assert len(retrieval.retrieve(store, "leave", k=1)) == 1


def test_confidence_is_top_score(store, fake_embed):
    fake_embed(np.pi / 3)  # cos(60 deg) = 0.5
    assert retrieval.confidence(retrieval.retrieve(store, "vague", k=2)) == 0.5


def test_above_threshold_is_confident(store, fake_embed):
    fake_embed(0.0)
    assert retrieval.is_confident(retrieval.retrieve(store, "leave", k=2))


def test_below_threshold_is_not_confident(store, fake_embed):
    # Angle chosen so cosine lands just under the configured threshold.
    fake_embed(float(np.arccos(config.RETRIEVAL_THRESHOLD - 0.05)))
    assert not retrieval.is_confident(retrieval.retrieve(store, "unrelated", k=2))


def test_confidence_of_empty_results_is_zero():
    assert retrieval.confidence([]) == 0.0
