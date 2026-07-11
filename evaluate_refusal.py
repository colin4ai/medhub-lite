"""Evaluate answerability: correct refusals and protection against over-refusal."""
import json

from evaluation_setup import isolated_evaluation_system


def evaluate(cases_path: str = "refusal_cases.json") -> dict:
    with open(cases_path) as f:
        cases = json.load(f)
    if not cases:
        raise ValueError("Answerability evaluation set must not be empty")

    with isolated_evaluation_system() as qa:
        details = []
        for case in cases:
            response = qa.ask_question(case["question"])
            predicted = bool(response.get("answerable"))
            expected = bool(case["expected_answerable"])
            details.append({
                "question": case["question"],
                "expected_answerable": expected,
                "predicted_answerable": predicted,
                "correct": predicted == expected,
                "answer": response["answer"],
            })

    negatives = [d for d in details if not d["expected_answerable"]]
    positives = [d for d in details if d["expected_answerable"]]
    return {
        "total": len(details),
        "accuracy": sum(d["correct"] for d in details) / len(details),
        "correct_refusal_rate": sum(d["correct"] for d in negatives) / len(negatives) if negatives else None,
        "answerable_recall": sum(d["correct"] for d in positives) / len(positives) if positives else None,
        "details": details,
    }


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result, indent=2))
