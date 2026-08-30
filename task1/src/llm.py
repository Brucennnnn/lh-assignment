"""Thin wrapper over the OpenAI API.

Only two operations are needed: embed a list of texts, and complete a chat
prompt. Keeping them behind module-level functions means the tests can
monkeypatch a single seam without touching the pipeline.
"""
import numpy as np

from . import config

_client = None


def _openai():
    global _client
    if _client is None:
        from openai import OpenAI  # imported lazily so importing this module needs no key
        _client = OpenAI()
    return _client


def embed(texts: list[str]) -> np.ndarray:
    """Return an (n, dim) float32 matrix of L2-normalised embeddings."""
    resp = _openai().embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def translate(text: str) -> str:
    """Query -> English. The corpus and the threshold are both English, so a Thai
    question embeds at ~0.2 against it and never clears the evidence gate."""
    return complete("Translate the question to English. Output only the translation.", text)


def complete(system: str, user: str) -> str:
    resp = _openai().chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()
