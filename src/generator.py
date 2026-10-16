"""Generates answers using the Groq LLM."""
from groq import Groq
from src.config import LLM_MODEL, SIMILARITY_THRESHOLD, TEMPERATURE


class Generator:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def generate(self, query: str, retrieved: list) -> dict:
        best_score = retrieved[0]["score"] if retrieved else 0.0

        if best_score < SIMILARITY_THRESHOLD:
            return {
                "answer": "I don't know based on the available information.",
                "sources": [],
                "refused": True,
            }

        context = "\n".join(r["chunk"] for r in retrieved if r["score"] >= SIMILARITY_THRESHOLD)
        sources = list({r["source"] for r in retrieved if r["score"] >= SIMILARITY_THRESHOLD})

        prompt = f"""Answer using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
Answer:"""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )

        return {
            "answer": response.choices[0].message.content.strip(),
            "sources": sources,
            "refused": False,
        }
