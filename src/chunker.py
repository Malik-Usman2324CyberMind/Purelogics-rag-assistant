"""Splits documents into self-contained chunks."""


def chunk_documents(documents: dict) -> tuple:
    """Splits each document into line-based chunks."""
    chunks, sources = [], []
    for filename, text in documents.items():
        for line in text.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("Source:") and not line.startswith("Topic:"):
                chunks.append(line)
                sources.append(filename)
    return chunks, sources
