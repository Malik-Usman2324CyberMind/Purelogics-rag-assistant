"""Evaluates the RAG pipeline against questions.csv."""
import pandas as pd
from src.config import QUESTIONS_CSV
from src.pipeline import RAGPipeline


def evaluate(pipeline: RAGPipeline, csv_path=QUESTIONS_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    results = []

    for i, row in df.iterrows():
        question = row["question"]
        expected = str(row["expected_source"]).strip()

        response = pipeline.ask(question)
        retrieved = response["retrieved"]
        top_source = retrieved[0]["source"] if retrieved else "NONE"
        top_score = retrieved[0]["score"] if retrieved else 0.0

        if expected.upper() == "NONE":
            correct = response["refused"]
            status = "OK refused" if correct else "FAIL hallucinated"
        else:
            correct = top_source == expected
            status = "OK" if correct else f"FAIL got {top_source}"

        results.append({
            "question": question,
            "expected_source": expected,
            "got_source": top_source,
            "score": round(top_score, 3),
            "answer": response["answer"][:100],
            "correct": correct,
            "status": status,
        })
        print(f"[{i+1}/{len(df)}] {status} | {question}")

    result_df = pd.DataFrame(results)
    total = len(result_df)
    passed = result_df["correct"].sum()
    print("\n" + "=" * 60)
    print(f"FINAL SCORE: {passed}/{total} ({100 * passed / total:.1f}%)")

    if "category" in df.columns:
        print("\nBreakdown by category:")
        merged = result_df.merge(df[["question", "category"]], on="question")
        summary = merged.groupby("category")["correct"].agg(["sum", "count"])
        for cat, r in summary.iterrows():
            print(f"  {cat}: {int(r['sum'])}/{int(r['count'])}")

    return result_df


def save_report(result_df: pd.DataFrame, path="data/evaluation_report.csv"):
    result_df.to_csv(path, index=False)
    print(f"\nReport saved: {path}")
