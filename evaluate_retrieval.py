"""Run labeled retrieval evaluation without generating answers."""
import json

from evaluator import QAEvaluator
from qa_system import MedicalQASystem


def evaluate(cases_path: str = "retrieval_cases.json", top_k: int = 3) -> dict:
    with open(cases_path) as f:
        cases = json.load(f)
    qa = MedicalQASystem()
    return QAEvaluator(qa).evaluate_retrieval_quality(cases, top_k=top_k)


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
