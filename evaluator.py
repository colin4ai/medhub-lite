"""
Evaluation framework for Medical Q&A System
Measures retrieval quality, answer accuracy, and citation quality
"""
from typing import List, Dict, Tuple
import json
from pathlib import Path

from qa_system import MedicalQASystem


class QAEvaluator:
    """Evaluate the medical Q&A system"""
    
    def __init__(self, qa_system: MedicalQASystem):
        self.qa_system = qa_system
    
    def evaluate_from_test_set(self, test_set_path: str) -> Dict:
        """
        Evaluate using a test set of question-answer pairs
        
        Test set format (JSON):
        [
            {
                "question": "What medications is the patient taking?",
                "expected_answer_contains": ["lisinopril", "metformin"],
                "should_cite_source": true
            },
            ...
        ]
        """
        with open(test_set_path, 'r') as f:
            test_cases = json.load(f)
        
        results = {
            "total_questions": len(test_cases),
            "correct_answers": 0,
            "with_citations": 0,
            "details": []
        }
        
        for test_case in test_cases:
            question = test_case["question"]
            expected_contains = test_case.get("expected_answer_contains", [])
            should_cite = test_case.get("should_cite_source", True)
            
            # Get answer from system
            response = self.qa_system.answer_question(question)
            answer = response["answer"].lower()
            
            # Check if expected terms are in answer
            terms_found = [term for term in expected_contains if term.lower() in answer]
            is_correct = len(terms_found) == len(expected_contains) if expected_contains else True
            
            # Check for citations
            has_citations = "[source" in answer.lower() or response.get("sources", [])
            
            if is_correct:
                results["correct_answers"] += 1
            
            if has_citations:
                results["with_citations"] += 1
            
            results["details"].append({
                "question": question,
                "answer": response["answer"],
                "is_correct": is_correct,
                "has_citations": has_citations,
                "terms_found": terms_found,
                "sources_used": len(response.get("sources", []))
            })
        
        # Calculate metrics
        results["accuracy"] = results["correct_answers"] / results["total_questions"]
        results["citation_rate"] = results["with_citations"] / results["total_questions"]
        
        return results
    
    def evaluate_retrieval_quality(self, questions: List[str]) -> Dict:
        """
        Evaluate retrieval quality
        Measures how well the system retrieves relevant chunks
        """
        retrieval_scores = []
        
        for question in questions:
            response = self.qa_system.answer_question(question)
            retrieval_scores.append({
                "question": question,
                "chunks_retrieved": response["retrieved_chunks"],
                "sources": response.get("sources", [])
            })
        
        avg_chunks = sum(r["chunks_retrieved"] for r in retrieval_scores) / len(retrieval_scores)
        
        return {
            "average_chunks_retrieved": avg_chunks,
            "total_questions": len(questions),
            "details": retrieval_scores
        }
    
    def evaluate_answer_quality_with_llm(self, test_cases: List[Dict]) -> Dict:
        """
        Use LLM-as-judge to evaluate answer quality
        
        Test case format:
        {
            "question": "...",
            "reference_answer": "...",  # Optional ground truth
            "context": "..."  # Optional context for evaluation
        }
        """
        from openai import OpenAI
        from config import Config
        
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        evaluation_results = []
        
        for test_case in test_cases:
            question = test_case["question"]
            reference = test_case.get("reference_answer", "")
            
            # Get system answer
            response = self.qa_system.answer_question(question)
            system_answer = response["answer"]
            
            # Use LLM to judge quality
            judge_prompt = f"""Evaluate the quality of this answer to a medical question.

Question: {question}

System Answer: {system_answer}

Reference Answer (if available): {reference if reference else "Not provided"}

Evaluate on a scale of 1-5 for:
1. Accuracy (based on medical correctness and alignment with reference)
2. Completeness (covers all relevant aspects)
3. Clarity (easy to understand)
4. Citation quality (properly cites sources)

Respond in JSON format:
{{
    "accuracy_score": <1-5>,
    "completeness_score": <1-5>,
    "clarity_score": <1-5>,
    "citation_score": <1-5>,
    "overall_score": <1-5>,
    "explanation": "<brief explanation>"
}}"""

            try:
                judge_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert medical document evaluator."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                scores = json.loads(judge_response.choices[0].message.content)
                
                evaluation_results.append({
                    "question": question,
                    "system_answer": system_answer,
                    "scores": scores,
                })
            
            except Exception as e:
                print(f"Error evaluating question '{question}': {e}")
                evaluation_results.append({
                    "question": question,
                    "system_answer": system_answer,
                    "scores": {"error": str(e)},
                })
        
        # Calculate averages
        valid_results = [r for r in evaluation_results if "error" not in r["scores"]]
        
        if valid_results:
            avg_scores = {
                "accuracy": sum(r["scores"]["accuracy_score"] for r in valid_results) / len(valid_results),
                "completeness": sum(r["scores"]["completeness_score"] for r in valid_results) / len(valid_results),
                "clarity": sum(r["scores"]["clarity_score"] for r in valid_results) / len(valid_results),
                "citation": sum(r["scores"]["citation_score"] for r in valid_results) / len(valid_results),
                "overall": sum(r["scores"]["overall_score"] for r in valid_results) / len(valid_results),
            }
        else:
            avg_scores = {}
        
        return {
            "average_scores": avg_scores,
            "total_evaluated": len(evaluation_results),
            "details": evaluation_results
        }
    
    def generate_evaluation_report(self, output_path: str = "evaluation_report.json") -> None:
        """Generate a comprehensive evaluation report"""
        # Example questions for evaluation
        sample_questions = [
            "What are the patient's current medications?",
            "What is the primary diagnosis?",
            "Are there any allergies documented?",
            "What are the patient's vital signs?",
            "What is the treatment plan?"
        ]
        
        retrieval_eval = self.evaluate_retrieval_quality(sample_questions)
        
        report = {
            "retrieval_evaluation": retrieval_eval,
            "system_config": {
                "embedding_model": Config.EMBEDDING_MODEL,
                "llm_model": Config.LLM_MODEL,
                "chunk_size": Config.CHUNK_SIZE,
                "top_k": Config.TOP_K_RESULTS,
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to {output_path}")
        return report
