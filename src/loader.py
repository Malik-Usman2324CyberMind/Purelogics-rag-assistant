"""Loads text documents from the data directory."""
from pathlib import Path


def load_documents(data_dir: Path) -> dict:
    """Loads all .txt files as {filename: content}."""
    documents = {}
    for path in sorted(Path(data_dir).glob("*.txt")):
        documents[path.name] = path.read_text(encoding="utf-8")
    if not documents:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")
    return documents
