from task1.src import ingestion

RAW = """---
id: doc_042
source: sample_policy.md
type: policy
department: HR
title: Sample Policy
---
# Heading One

First paragraph about leave.

Second paragraph about carry over.
"""


def test_front_matter_parsed():
    doc = ingestion._parse(RAW)
    assert doc["id"] == "doc_042"
    assert doc["department"] == "HR"
    assert doc["content"].startswith("# Heading One")


def test_chunks_carry_metadata():
    chunks = ingestion.chunk_document(ingestion._parse(RAW))
    assert chunks
    for chunk in chunks:
        assert chunk["document_id"] == "doc_042"
        assert chunk["source"] == "sample_policy.md"
        assert chunk["title"] == "Sample Policy"
        assert chunk["metadata"] == {"type": "policy", "department": "HR"}


def test_heading_is_prefixed_to_chunk():
    chunks = ingestion.chunk_document(ingestion._parse(RAW))
    assert chunks[0]["content"].startswith("Heading One")


def test_real_knowledge_base_loads():
    docs = ingestion.load_documents()
    assert 5 <= len(docs) <= 10
    assert {d["type"] for d in docs} >= {"policy", "chat"}
    assert len({d["id"] for d in docs}) == len(docs)
