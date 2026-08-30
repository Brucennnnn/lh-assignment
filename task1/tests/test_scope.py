"""Scope is decided without consulting the corpus, so these tests are stable
regardless of what happens to be indexed."""
import numpy as np
import pytest

from src import scope

WRONG_FORM = [
    "What's the weather tomorrow?",
    "Who won yesterday's football match?",
    "Write me a poem.",
    "Tell me a joke.",
    "What is the current stock price of Apple?",
    "Write me a poem about the expense policy.",
    "พรุ่งนี้อากาศเป็นยังไง",
    "เล่าเรื่องตลกให้ฟังหน่อย",
    "ราคาหุ้นวันนี้เท่าไหร่",
]

COMPANY_QUESTIONS = [
    "What is the annual leave entitlement?",
    "How do I request reimbursement?",
    "What is the travel expense approval process?",
    "What should a new employee do during onboarding?",
    # In scope even though no document covers them - that is the whole point.
    "What is the maximum mortgage a branch manager can approve?",
    "Where do I park at the head office?",
    "สิทธิ์ลาพักร้อนมีกี่วัน",
]


@pytest.mark.parametrize("query", WRONG_FORM)
def test_precheck_rejects_wrong_request_form(query):
    assert scope.precheck(query) is not None


@pytest.mark.parametrize("query", COMPANY_QUESTIONS)
def test_precheck_allows_company_questions(query):
    assert scope.precheck(query) is None


def test_uncovered_question_is_still_in_scope(in_domain, query_vector):
    """A question nobody has documented is in scope. Coverage is decided later,
    by retrieval - never here."""
    assert scope.classify_domain(query_vector(0.0)) is None


def test_off_topic_question_is_out_of_scope(off_domain, query_vector):
    assert scope.classify_domain(query_vector(0.0)) == "outside_covered_domains"


def test_scope_module_never_reads_the_corpus():
    """Structural guard: if scope ever imports the store or the index, the
    separation this design depends on has been broken."""
    source = (scope.__file__ and open(scope.__file__, encoding="utf-8").read()) or ""
    for forbidden in ("from .store", "from .ingestion", "VectorStore", "load_index"):
        assert forbidden not in source


def test_domain_lists_are_policy_not_derived_from_data():
    """The covered domains are a written remit, not a summary of data/."""
    assert len(scope.COVERED_DOMAINS) >= 5
    assert len(scope.OFF_DOMAIN_EXAMPLES) >= 5
