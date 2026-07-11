"""Run the labeled answer-and-citation evaluation on an isolated corpus."""
import json

from evaluation_setup import isolated_evaluation_system
from evaluator import QAEvaluator


def evaluate(cases_path: str = "test_cases.json") -> dict:
    with isolated_evaluation_system() as qa:
        return QAEvaluator(qa).evaluate_from_test_set(cases_path)


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
