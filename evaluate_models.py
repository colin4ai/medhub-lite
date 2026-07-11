"""Compare generation models with retrieval held constant.

This makes paid API calls. Example:
  python evaluate_models.py --models gpt-4o-mini gpt-4o
"""
import argparse
import json

from evaluator import QAEvaluator
from qa_system import MedicalQASystem
from vector_store import VectorStore


def evaluate_models(models, test_set):
    store = VectorStore()
    reports = {}
    for model in models:
        report = QAEvaluator(MedicalQASystem(store, llm_model=model)).evaluate_from_test_set(test_set)
        details = report.pop("details")
        reports[model] = {
            **report,
            "total_tokens": sum(d["token_usage"].get("total_tokens", 0) for d in details),
            "average_latency_ms": round(
                sum(d["latency_ms"].get("total", 0) for d in details) / len(details), 1
            ),
        }
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--test-set", default="test_cases.json")
    args = parser.parse_args()
    print(json.dumps(evaluate_models(args.models, args.test_set), indent=2))
