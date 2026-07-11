"""Run labeled retrieval evaluation without generating answers."""
import json

from evaluation_setup import assert_labeled_chunks_present, isolated_evaluation_system
from evaluator import QAEvaluator


def evaluate(cases_path: str = "retrieval_cases.json", top_k: int = 3) -> dict:
    with open(cases_path) as f:
        cases = json.load(f)
    if not cases:
        raise ValueError("Retrieval evaluation set must not be empty")
    with isolated_evaluation_system() as qa:
        assert_labeled_chunks_present(qa, cases)
        return QAEvaluator(qa).evaluate_retrieval_quality(cases, top_k=top_k)


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
