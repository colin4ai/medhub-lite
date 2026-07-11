"""Routing accuracy evaluation for the multi-agent orchestrator.

Runs labeled queries through the router only (no agent execution)
and reports accuracy plus a per-route breakdown and miss list.

Usage: python evaluate_routing.py
"""
import json
from collections import defaultdict

from agents import AgentOrchestrator


def evaluate(cases_path: str = "routing_cases.json") -> dict:
    with open(cases_path) as f:
        cases = json.load(f)
    if not cases:
        raise ValueError("Routing evaluation set must not be empty")

    # Router-only harness: reuse the orchestrator's LLM + parse helpers
    # without touching the vector store or agents.
    orch = AgentOrchestrator(qa_system=None, vector_store=None)

    results = {"total": len(cases), "correct": 0, "misses": []}
    per_route = defaultdict(lambda: {"total": 0, "correct": 0})

    for case in cases:
        query, expected = case["query"], case["expected_route"]
        parsed = orch.classify_route(query)
        predicted = parsed.get("route", "qa")

        per_route[expected]["total"] += 1
        if predicted == expected:
            results["correct"] += 1
            per_route[expected]["correct"] += 1
        else:
            results["misses"].append(
                {"query": query, "expected": expected, "predicted": predicted,
                 "reason": parsed.get("reason", "")}
            )

    results["accuracy"] = round(results["correct"] / results["total"] * 100, 1)
    results["per_route"] = {r: v for r, v in per_route.items()}
    return results


if __name__ == "__main__":
    r = evaluate()
    print(f"\nRouting accuracy: {r['accuracy']}%  ({r['correct']}/{r['total']})\n")
    print("Per-route:")
    for route, v in r["per_route"].items():
        print(f"  {route:10s} {v['correct']}/{v['total']}")
    if r["misses"]:
        print("\nMisses:")
        for m in r["misses"]:
            print(f"  [{m['expected']} -> {m['predicted']}] {m['query']}")
            if m["reason"]:
                print(f"      router said: {m['reason']}")
    print()
