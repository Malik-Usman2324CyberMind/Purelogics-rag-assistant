"""Central configuration for the RAG pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
QUESTIONS_CSV = ROOT_DIR / "data" / "questions.csv"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.35
TEMPERATURE = 0.1
