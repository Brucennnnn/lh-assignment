"""Single place for tunable knobs. Everything reads from env with sane defaults."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = [ROOT / "data" / "documents", ROOT / "data" / "chats"]
INDEX_PATH = Path(os.getenv("INDEX_PATH", ROOT / "data" / "index.pkl"))
LOG_PATH = Path(os.getenv("LOG_PATH", ROOT / "logs" / "rag.jsonl"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Cosine similarity bands on the top-scoring chunk.
#   score < SCOPE_THRESHOLD          -> nothing in the KB looks related -> out of scope
#   SCOPE_THRESHOLD <= score < RETRIEVAL_THRESHOLD -> related but weak -> fallback
#   score >= RETRIEVAL_THRESHOLD     -> answer it
# Tuned by hand against text-embedding-3-small; see README "Thresholds".
SCOPE_THRESHOLD = float(os.getenv("SCOPE_THRESHOLD", "0.20"))
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.32"))

# The offline stub in llm.py scores on lexical overlap, so its similarity scale
# differs from the embedding model's. --offline swaps in these bands instead.
STUB_SCOPE_THRESHOLD = float(os.getenv("STUB_SCOPE_THRESHOLD", "0.30"))
STUB_RETRIEVAL_THRESHOLD = float(os.getenv("STUB_RETRIEVAL_THRESHOLD", "0.42"))


def apply_stub_thresholds() -> None:
    global SCOPE_THRESHOLD, RETRIEVAL_THRESHOLD
    SCOPE_THRESHOLD, RETRIEVAL_THRESHOLD = STUB_SCOPE_THRESHOLD, STUB_RETRIEVAL_THRESHOLD
