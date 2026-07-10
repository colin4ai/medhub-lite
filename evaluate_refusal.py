"""Refusal evaluation: does the system decline to answer questions
whose answers are not in the corpus, instead of hallucinating?

Requires sample docs to be ingested first (run via API or CLI ingest).
Usage: python evaluate_refusal.py
"""
import json

from qa_system import MedicalQASystem

REFUSAL_MARKERS = [
    "don't have enough information",
    "do not have enough information",
    "do not contain",
    "does not contain",
    "do not mention",
    "does not mention",
    "not mentioned",
    "not specified",
    "no information",
    "any information regarding",
    "not available in",
    "cannot find",
]


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in REFUSAL_MARKERS)


def evaluate(cases_path: str = "refusal_cases.json") -> dict:
    with open(cases_path) as f:
        cases = json.load(f)

    qa = MedicalQASystem()
    # Ingest sample docs so retrieval has a real (but irrelevant) corpus
    qa.add_document("sample_docs/clinical_note_2024_03_15.txt")
    qa.add_document("sample_docs/follow_up_note_2024_03_29.txt")

    results = {"total": len(cases), "correct_refusals": 0, "hallucinations": []}

    for case in cases:
        q = case["question"]
        resp = qa.ask_question(q)
        answer = resp["answer"]
        if is_refusal(answer):
            results["correct_refusals"] += 1
        else:
            results["hallucinations"].append({"question": q, "answer": answer})

    results["refusal_rate"] = f"{results['correct_refusals']}/{results['total']}"
    return results


if __name__ == "__main__":
    r = evaluate()
    print(f"\nRefusal correctness: {r['refusal_rate']} out-of-corpus questions correctly declined\n")
    if r["hallucinations"]:
        print("HALLUCINATIONS (answered despite no source data):")
        for h in r["hallucinations"]:
            print(f"  Q: {h['question']}")
            print(f"  A: {h['answer']}\n")
