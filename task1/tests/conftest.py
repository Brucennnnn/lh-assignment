import numpy as np
import pytest

from task1.src.store import VectorStore

# Two orthogonal unit vectors, so cosine scores in tests are exact and readable.
CHUNK_A = {"chunk_id": "doc_001_c00", "content": "Employees get 12 days of annual leave.",
           "document_id": "doc_001", "source": "leave_policy.md",
           "title": "Employee Leave Policy",
           "metadata": {"type": "policy", "department": "HR"}}
CHUNK_B = {"chunk_id": "doc_002_c00", "content": "Submit expense claims within 30 days.",
           "document_id": "doc_002", "source": "expense_policy.md",
           "title": "Expense Reimbursement Policy",
           "metadata": {"type": "policy", "department": "Finance"}}

AXIS_A = np.array([1.0, 0.0], dtype=np.float32)
AXIS_B = np.array([0.0, 1.0], dtype=np.float32)


@pytest.fixture
def store():
    return VectorStore([CHUNK_A, CHUNK_B], np.array([AXIS_A, AXIS_B]))


@pytest.fixture
def query_vector():
    """A unit vector at a chosen angle from chunk A, rotated away from chunk B,
    so chunk A always ranks first and scores exactly cos(angle)."""
    def _make(angle_to_chunk_a: float = 0.0):
        return np.array([np.cos(angle_to_chunk_a), -np.sin(angle_to_chunk_a)], dtype=np.float32)
    return _make


@pytest.fixture
def stub_embed(monkeypatch):
    """Every embedding call returns AXIS_A, so a query is in-domain by default."""
    monkeypatch.setattr("src.llm.embed",
                        lambda texts: np.array([AXIS_A] * len(texts), dtype=np.float32))


@pytest.fixture
def in_domain(monkeypatch):
    """Pin scope's centroids so classify_domain passes, with no embedding call."""
    monkeypatch.setattr("src.scope._centroids", (np.array([AXIS_A]), np.array([AXIS_B])))


@pytest.fixture
def off_domain(monkeypatch):
    """Pin scope's centroids so classify_domain rejects."""
    monkeypatch.setattr("src.scope._centroids", (np.array([AXIS_B]), np.array([AXIS_A])))
