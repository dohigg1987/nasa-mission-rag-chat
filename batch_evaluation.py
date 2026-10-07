#!/usr/bin/env python3
"""
Batch evaluation of the NASA RAG chat system.

Loads test questions (evaluation_dataset.txt or a test_questions.json file),
runs each one through the full pipeline (retrieval from ChromaDB, answer
generation with OpenAI, RAGAS scoring) and prints a summary per question and
an aggregate (mean, min and max) for each metric. Results are also written to
evaluation_results.json and evaluation_results.csv.

Usage:
    export OPENAI_API_KEY=...
    python batch_evaluation.py --dataset evaluation_dataset.txt \
        --chroma-dir ./chroma_db_openai --collection-name nasa_space_missions_text --k 5
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

import llm_client
import rag_client
import ragas_evaluator


def load_text_dataset(path: Path) -> List[Dict[str, str]]:
    """Parse the block format used by evaluation_dataset.txt."""
    questions, current = [], {}
    field_names = {"q": "question", "category": "category", "mission": "mission", "expected": "expected"}

    def flush():
        if current.get("question"):
            questions.append(dict(current))
        current.clear()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            continue
        if not line:
            flush()
            continue
        if ":" in line:
            label, value = line.split(":", 1)
            key = field_names.get(label.strip().lower())
            if key:
                current[key] = value.strip()
                continue
        if "expected" in current:  # continuation of a long expected answer
            current["expected"] += " " + line
    flush()
    return questions


def load_dataset(path: str) -> List[Dict[str, str]]:
    """Load test questions from a .txt (block format) or .json file."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() == ".json":
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        items = data.get("questions", data) if isinstance(data, dict) else data
        questions = [{
            "question": item.get("question", ""),
            "category": item.get("category", ""),
            "mission": item.get("mission", "all"),
            "expected": item.get("expected", item.get("expected_answer", "")),
        } for item in items]
    else:
        questions = load_text_dataset(dataset_path)

    questions = [q for q in questions if q.get("question", "").strip()]
    if not questions:
        raise ValueError(f"No questions found in {dataset_path}")
    return questions


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch RAGAS evaluation of the NASA RAG system")
    parser.add_argument("--dataset", default="evaluation_dataset.txt", help="Path to .txt or .json test set")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="ChromaDB persist directory")
    parser.add_argument("--collection-name", default="nasa_space_missions_text", help="Collection name")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve per question")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI chat model for answers")
    parser.add_argument("--no-mission-filter", action="store_true",
                        help="Search all missions instead of the mission listed for each question")
    parser.add_argument("--output", default="evaluation_results", help="Output file prefix")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set the OPENAI_API_KEY environment variable first.")
        return 1
    os.environ.setdefault("CHROMA_OPENAI_API_KEY", api_key)

    questions = load_dataset(args.dataset)
    print(f"Loaded {len(questions)} questions from {args.dataset}")

    collection, ok, error = rag_client.initialize_rag_system(args.chroma_dir, args.collection_name)
    if not ok:
        print(f"Could not open ChromaDB collection: {error}")
        return 1

    rows = []
    for number, item in enumerate(questions, start=1):
        question = item["question"]
        mission = None if args.no_mission_filter else item.get("mission", "all")
        print("\n" + "=" * 80)
        print(f"[{number}/{len(questions)}] ({item.get('category', '')}) {question}")

        try:
            results = rag_client.retrieve_documents(collection, question, args.k, mission)
            documents = results["documents"][0] if results and results.get("documents") else []
            metadatas = results["metadatas"][0] if results and results.get("metadatas") else []
            context = rag_client.format_context(documents, metadatas)
            answer = llm_client.generate_response(api_key, question, context, [], args.model)
            scores = ragas_evaluator.evaluate_response_quality(
                question, answer, documents, reference=item.get("expected") or None)
        except Exception as e:
            answer, documents, scores = "", [], {"error": f"{type(e).__name__}: {e}"}

        print(f"Answer: {answer[:600]}{'...' if len(answer) > 600 else ''}")
        print(f"Retrieved chunks: {len(documents)}")
        numeric = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
        if numeric:
            print("Scores: " + ", ".join(f"{k}={v:.3f}" for k, v in numeric.items()))
        if "error" in scores:
            print(f"Evaluation error: {scores['error']}")
        if scores.get("warnings"):
            print(f"Warnings: {scores['warnings']}")

        rows.append({
            "question": question,
            "category": item.get("category", ""),
            "mission": item.get("mission", ""),
            "expected": item.get("expected", ""),
            "answer": answer,
            "retrieved_chunks": len(documents),
            **numeric,
            "error": scores.get("error", ""),
        })

    # Aggregate per metric
    metric_names = sorted({k for row in rows for k, v in row.items()
                           if isinstance(v, float) and k != "retrieved_chunks"})
    summary = {}
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print(f"{'metric':<28}{'n':>4}{'mean':>9}{'min':>9}{'max':>9}")
    for name in metric_names:
        values = [row[name] for row in rows if isinstance(row.get(name), float)]
        summary[name] = {
            "n": len(values),
            "mean": round(statistics.mean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
        print(f"{name:<28}{len(values):>4}{summary[name]['mean']:>9.3f}"
              f"{summary[name]['min']:>9.3f}{summary[name]['max']:>9.3f}")

    Path(f"{args.output}.json").write_text(
        json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(f"{args.output}.csv", index=False)
    print(f"\nSaved {args.output}.json and {args.output}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
