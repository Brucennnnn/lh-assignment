"""Thin wrapper over the OpenAI API, plus a deterministic offline stub.

Only two operations are needed: embed a list of texts, and complete a chat
prompt. Keeping them behind module-level functions means tests and the offline
evaluation can swap in `use_stub()` without touching the pipeline.
"""
import hashlib
import re

import numpy as np

from . import config

_client = None
_stubbed = False


def _openai():
    global _client
    if _client is None:
        from openai import OpenAI  # imported lazily so the stub path needs no key
        _client = OpenAI()
    return _client


def embed(texts: list[str]) -> np.ndarray:
    """Return an (n, dim) float32 matrix of L2-normalised embeddings."""
    if _stubbed:
        return _stub_embed(texts)
    resp = _openai().embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def complete(system: str, user: str) -> str:
    if _stubbed:
        return _stub_complete(user)
    resp = _openai().chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()


# --- offline stub ---------------------------------------------------------
# ponytail: hashed bag-of-words, not real semantics. Good enough to separate
# "obviously related" from "obviously unrelated" so the safety paths can be
# demonstrated without an API key. Swap for the real provider for quality work.
_DIM = 512
_df = np.zeros(_DIM, dtype=np.float32)   # document frequency per hash bucket
_n_docs = 0


def use_stub(enabled: bool = True) -> None:
    global _stubbed
    _stubbed = enabled


def _stub_embed(texts: list[str]) -> np.ndarray:
    """Hashed bag-of-words with IDF weighting.

    The first call carries the whole corpus (ingestion embeds every chunk before
    any query is asked), so document frequency is learned there and reused to
    down-weight filler words like "what" or "policy" in later queries.
    """
    global _n_docs
    counts = np.zeros((len(texts), _DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in set(re.findall(r"[a-z]{3,}", text.lower())):
            counts[i, int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % _DIM] += 1.0

    if _n_docs == 0 and len(texts) > 1:
        _df[:] = (counts > 0).sum(axis=0)
        _n_docs = len(texts)

    idf = np.log((_n_docs + 1) / (_df + 1)) + 1.0 if _n_docs else 1.0
    out = counts * idf
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def _stub_complete(user: str) -> str:
    """Echo back the first retrieved chunk so grounding/attribution stay visible."""
    match = re.search(r"\[(\d+)\][^\n]*\n(.+?)(?=\n\[\d+\]|\Z)", user, re.S)
    if not match:
        return "INSUFFICIENT_CONTEXT"
    return "[stub answer] " + " ".join(match.group(2).split())[:400] + " [1]"
