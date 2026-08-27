"""Raw files -> metadata + chunks -> embeddings -> on-disk vector store."""
import pickle
import re

from . import llm
from . import config
from .store import VectorStore


def load_documents() -> list[dict]:
    """Read every file under the data dirs. Front matter carries the metadata."""
    docs = []
    for directory in config.DATA_DIRS:
        for path in sorted(directory.glob("*")):
            if path.is_file() and not path.name.startswith("."):
                docs.append(_parse(path.read_text(encoding="utf-8")))
    return docs


def _parse(raw: str) -> dict:
    """Split `---` front matter (id/source/type/department/title) from the body."""
    parts = raw.split("---", 2)
    if len(parts) != 3:
        raise ValueError("document is missing --- front matter ---")
    meta = dict(
        line.split(":", 1) for line in parts[1].strip().splitlines() if ":" in line
    )
    doc = {k.strip(): v.strip() for k, v in meta.items()}
    doc["content"] = parts[2].strip()
    return doc


def chunk_document(doc: dict) -> list[dict]:
    """Pack paragraphs up to CHUNK_SIZE chars, prefixing the current heading.

    Headings are repeated into every chunk so a chunk like a bare bullet list
    still carries the words "Annual Leave Entitlement" for retrieval.
    """
    chunks, buf, heading = [], "", ""
    for para in re.split(r"\n\s*\n", doc["content"]):
        para = para.strip()
        if not para:
            continue
        if para.startswith("#"):
            heading = para.lstrip("# ").strip()
            continue
        if buf and len(buf) + len(para) > config.CHUNK_SIZE:
            chunks.append(buf)
            buf = ""
        buf = f"{buf}\n\n{para}" if buf else (f"{heading}\n{para}" if heading else para)
    if buf:
        chunks.append(buf)

    return [
        {
            "chunk_id": f"{doc['id']}_c{i:02d}",
            "content": text,
            "document_id": doc["id"],
            "source": doc["source"],
            "title": doc["title"],
            "metadata": {"type": doc["type"], "department": doc["department"]},
        }
        for i, text in enumerate(chunks)
    ]


def build_index(persist: bool = True) -> VectorStore:
    chunks = [c for doc in load_documents() for c in chunk_document(doc)]
    store = VectorStore(chunks, llm.embed([c["content"] for c in chunks]))
    if persist:
        config.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        # ponytail: pickle of a local cache we generate ourselves. Never load an
        # index from an untrusted source.
        config.INDEX_PATH.write_bytes(pickle.dumps(store))
    return store


def load_index(rebuild: bool = False) -> VectorStore:
    if rebuild or not config.INDEX_PATH.exists():
        return build_index()
    return pickle.loads(config.INDEX_PATH.read_bytes())


if __name__ == "__main__":
    store = build_index()
    print(f"indexed {len(store.chunks)} chunks -> {config.INDEX_PATH}")
