import pytest

from src import config, scope

OUT_OF_SCOPE = [
    "What's the weather tomorrow?",
    "Who won yesterday's football match?",
    "Write me a poem.",
    "Tell me a joke.",
    "What is the current stock price of Apple?",
    "Write me a poem about the expense policy.",
]

IN_SCOPE = [
    "What is the annual leave entitlement?",
    "How do I request reimbursement?",
    "What is the travel expense approval process?",
    "What should a new employee do during onboarding?",
]


@pytest.mark.parametrize("query", OUT_OF_SCOPE)
def test_precheck_rejects(query):
    assert scope.precheck(query) is not None


@pytest.mark.parametrize("query", IN_SCOPE)
def test_precheck_allows(query):
    assert scope.precheck(query) is None


def test_low_similarity_is_out_of_scope():
    assert scope.classify_by_score(config.SCOPE_THRESHOLD - 0.01) == "no_related_content"


def test_related_similarity_passes():
    assert scope.classify_by_score(config.SCOPE_THRESHOLD + 0.01) is None
