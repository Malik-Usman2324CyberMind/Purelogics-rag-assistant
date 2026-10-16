"""Handles embedding and semantic retrieval."""
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL


class Retriever:
    def __init__(self, chunks: list, sources: list):
        self.chunks = chunks
        self.sources = sources
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.embeddings = self._embed(chunks)

    def _embed(self, texts: list) -> np.ndarray:
        emb = self.model.encode(texts)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        q_emb = self.model.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        scores = np.dot(self.embeddings, q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"chunk": self.chunks[i], "source": self.sources[i], "score": float(scores[i])}
            for i in top_idx
        ]
