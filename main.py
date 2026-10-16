"""Entry point: run interactive RAG or batch evaluation."""
import os
import sys
from dotenv import load_dotenv

from src.pipeline import RAGPipeline
from src.evaluate import evaluate, save_report

load_dotenv()


def get_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY not found. Add it to your .env file.")
        sys.exit(1)
    return key


def run_interactive(pipeline: RAGPipeline):
    print("\nPureLogics RAG Chatbot (type 'exit' to quit)\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        result = pipeline.ask(query)
        print(f"\nBot: {result['answer']}")
        if result["sources"]:
            print(f"     Sources: {', '.join(result['sources'])}")
        print()


def main():
    api_key = get_api_key()
    print("Loading pipeline...")
    pipeline = RAGPipeline(api_key)
    print(f"Pipeline ready with {pipeline.num_chunks} chunks.\n")

    mode = sys.argv[1] if len(sys.argv) > 1 else "interactive"

    if mode == "eval":
        print("Running evaluation against questions.csv...\n")
        df = evaluate(pipeline)
        save_report(df)
    else:
        run_interactive(pipeline)


if __name__ == "__main__":
    main()
