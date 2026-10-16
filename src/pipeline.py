"""Main RAG pipeline that wires everything together."""
from src.loader import load_documents
from src.chunker import chunk_documents
from src.retriever import Retriever
from src.generator import Generator
from src.config import DATA_DIR, TOP_K


class RAGPipeline:
    def __init__(self, api_key: str):
        documents = load_documents(DATA_DIR)
        chunks, sources = chunk_documents(documents)
        self.retriever = Retriever(chunks, sources)
        self.generator = Generator(api_key)
        self.num_chunks = len(chunks)

    def ask(self, query: str) -> dict:
        retrieved = self.retriever.retrieve(query, top_k=TOP_K)
        result = self.generator.generate(query, retrieved)
        result["retrieved"] = retrieved
        return result
