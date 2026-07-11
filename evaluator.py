"""Evaluation utilities for retrieval, answer quality, and citations."""
import json
import re
from typing import Dict, List

from openai import OpenAI

import config


class QAEvaluator:
    """Evaluate generation and retrieval separately."""

    def __init__(self, qa_system):
        self.qa_system = qa_system

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize punctuation so equivalent forms such as light-duty match."""
        return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())

    @staticmethod
    def _valid_citations(response: Dict) -> bool:
        cited = set(response.get("cited_source_numbers", []))
        available = {s.get("source_number") for s in response.get("sources", [])}
        return bool(cited) and cited.issubset(available)

    def evaluate_from_test_set(self, test_set_path: str) -> Dict:
        with open(test_set_path) as f:
            test_cases = json.load(f)
        if not test_cases:
            raise ValueError("Evaluation set must not be empty")

        details = []
        for case in test_cases:
            response = self.qa_system.ask_question(case["question"])
            answer = self._normalize(response["answer"])
            # Each expected item is either a required string or a list of
            # acceptable phrasings for one required concept.
            expected_groups = []
            for item in case.get("expected_answer_contains", []):
                alternatives = item if isinstance(item, list) else [item]
                expected_groups.append([self._normalize(term) for term in alternatives])
            terms_found = [
                next((term for term in alternatives if term in answer), None)
                for alternatives in expected_groups
            ]
            content_correct = all(term is not None for term in terms_found)
            terms_found = [term for term in terms_found if term is not None]
            should_cite = case.get("should_cite_source", True)
            citations_valid = self._valid_citations(response)
            citation_correct = citations_valid if should_cite else not response.get("cited_source_numbers")
            details.append({
                "question": case["question"],
                "answer": response["answer"],
                "content_correct": content_correct,
                "citation_correct": citation_correct,
                "is_correct": content_correct and citation_correct,
                "terms_found": terms_found,
                "answerable": response.get("answerable", False),
                "sources_used": len(response.get("sources", [])),
                "token_usage": response.get("token_usage", {}),
                "latency_ms": response.get("latency_ms", {}),
            })

        total = len(details)
        correct = sum(item["is_correct"] for item in details)
        citations = sum(item["citation_correct"] for item in details)
        return {
            "total_questions": total,
            "correct_answers": correct,
            "with_valid_citations": citations,
            "accuracy": correct / total,
            "citation_rate": citations / total,
            "details": details,
        }

    def evaluate_retrieval_quality(self, cases: List[Dict], top_k: int = 5) -> Dict:
        """Calculate Recall@K when cases contain expected_chunk_ids."""
        if not cases:
            raise ValueError("Retrieval evaluation set must not be empty")

        details = []
        for case in cases:
            chunks = self.qa_system.vector_store.search(case["question"], top_k=top_k)
            retrieved = [chunk["chunk_id"] for chunk in chunks]
            expected = set(case.get("expected_chunk_ids", []))
            hits = expected.intersection(retrieved)
            recall = len(hits) / len(expected) if expected else None
            first_relevant_rank = next(
                (rank for rank, chunk_id in enumerate(retrieved, 1) if chunk_id in expected),
                None,
            )
            details.append({
                "question": case["question"],
                "retrieved_chunk_ids": retrieved,
                "expected_chunk_ids": sorted(expected),
                "recall_at_k": recall,
                "reciprocal_rank": 1 / first_relevant_rank if first_relevant_rank else 0.0,
                "top_similarity": chunks[0].get("similarity") if chunks else None,
            })

        scored = [item["recall_at_k"] for item in details if item["recall_at_k"] is not None]
        reciprocal_ranks = [item["reciprocal_rank"] for item in details if item["expected_chunk_ids"]]
        return {
            "top_k": top_k,
            "total_queries": len(details),
            "scored_queries": len(scored),
            "mean_recall_at_k": sum(scored) / len(scored) if scored else None,
            "mean_reciprocal_rank": (
                sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None
            ),
            "details": details,
        }

    def evaluate_answer_quality_with_llm(self, test_cases: List[Dict]) -> Dict:
        """Use a schema-constrained LLM judge; keep its score separate from gold checks."""
        if not test_cases:
            raise ValueError("Judge evaluation set must not be empty")
        client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=config.OPENAI_TIMEOUT_SECONDS,
            max_retries=config.OPENAI_MAX_RETRIES,
        )
        results = []
        for case in test_cases:
            response = self.qa_system.ask_question(case["question"])
            prompt = (
                "Evaluate whether the answer is correct, complete, and grounded in its cited context. "
                "Do not reward fluency. Return JSON with integer scores from 1 to 5 for accuracy, "
                "completeness, clarity, groundedness, and overall, plus a brief explanation.\n\n"
                f"Question: {case['question']}\nReference: {case.get('reference_answer', '')}\n"
                f"Answer: {response['answer']}\n"
                f"Validated evidence: {json.dumps(response.get('claim_evidence', []))}"
            )
            judge = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "system", "content": "You are a strict RAG evaluator."},
                          {"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            results.append({
                "question": case["question"],
                "system_answer": response["answer"],
                "scores": json.loads(judge.choices[0].message.content),
            })
        return {"total_evaluated": len(results), "details": results}

    def generate_evaluation_report(self, output_path: str = "evaluation_report.json") -> Dict:
        report = {
            "system_config": {
                "embedding_model": config.EMBEDDING_MODEL,
                "llm_model": config.LLM_MODEL,
                "chunk_size": config.CHUNK_SIZE,
                "top_k": config.TOP_K_RESULTS,
                "similarity_threshold": config.SIMILARITY_THRESHOLD,
            }
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        return report
