import numpy as np
import pytest

from src.store import VectorStore

# Two orthogonal unit vectors, so cosine scores in tests are exact and readable.
CHUNK_A = {"chunk_id": "doc_001_c00", "content": "Employees get 12 days of annual leave.",
           "document_id": "doc_001", "source": "leave_policy.md",
           "title": "Employee Leave Policy",
           "metadata": {"type": "policy", "department": "HR"}}
CHUNK_B = {"chunk_id": "doc_002_c00", "content": "Submit expense claims within 30 days.",
           "document_id": "doc_002", "source": "expense_policy.md",
           "title": "Expense Reimbursement Policy",
           "metadata": {"type": "policy", "department": "Finance"}}


@pytest.fixture
def store():
    return VectorStore([CHUNK_A, CHUNK_B], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


@pytest.fixture
def fake_embed(monkeypatch):
    """Point the query embedding wherever a test needs it: fake_embed(angle).

    The angle is measured from chunk A and rotated away from chunk B, so chunk A
    always ranks first and its score is exactly cos(angle).
    """
    def _set(angle_to_chunk_a: float):
        vec = np.array([[np.cos(angle_to_chunk_a), -np.sin(angle_to_chunk_a)]], dtype=np.float32)
        monkeypatch.setattr("src.llm.embed", lambda texts: vec)
    return _set
