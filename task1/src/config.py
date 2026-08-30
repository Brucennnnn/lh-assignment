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

# Evidence threshold on the top-scoring chunk: below this we hold no document
# that covers the question, so the answer is a fallback rather than a guess.
# This is the ONLY score-based decision in the pipeline. Scope is decided
# without it - see src/scope.py.
# Tuned by hand against text-embedding-3-small; see README "Thresholds".
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.32"))
