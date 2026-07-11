"""
Evaluation framework for the Q&A system.
Measures accuracy, retrieval quality, and identifies failure modes.
"""
import json
from typing import List, Dict
from qa_system import MedicalQASystem
import time


class QAEvaluator:
    """Evaluate Q&A system performance"""
    
    def __init__(self, qa_system: MedicalQASystem):
        self.qa_system = qa_system
    
    def evaluate_question_set(self, questions: List[Dict]) -> Dict:
        """
        Evaluate system on a set of test questions.
        
        Args:
            questions: List of dicts with 'question' and 'expected_answer' (optional)
            
        Returns:
            Evaluation results
        """
        results = []
        total_time = 0
        
        print(f"\nEvaluating {len(questions)} questions...\n")
        
        for i, q in enumerate(questions, 1):
            print(f"Question {i}/{len(questions)}: {q['question'][:60]}...")
            
            start_time = time.time()
            result = self.qa_system.ask_question(q['question'])
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Store result
            eval_result = {
                'question': q['question'],
                'answer': result['answer'],
                'num_sources': result['num_sources'],
                'confidence': result['confidence'],
                'latency_seconds': round(elapsed, 2)
            }
            
            # If expected answer provided, check similarity
            if 'expected_answer' in q:
                eval_result['expected_answer'] = q['expected_answer']
                eval_result['manual_check_needed'] = True
            
            results.append(eval_result)
        
        # Calculate metrics
        avg_latency = total_time / len(questions)
        avg_sources = sum(r['num_sources'] for r in results) / len(results)
        
        summary = {
            'total_questions': len(questions),
            'avg_latency_seconds': round(avg_latency, 2),
            'avg_sources_per_answer': round(avg_sources, 2),
            'results': results
        }
        
        print("\n✓ Evaluation complete!")
        print(f"  Average latency: {avg_latency:.2f}s")
        print(f"  Average sources: {avg_sources:.1f}")
        
        return summary
    
    def evaluate_retrieval_quality(self, questions: List[str], top_k: int = 5) -> Dict:
        """
        Evaluate retrieval quality (how relevant are retrieved chunks).
        
        Args:
            questions: List of questions
            top_k: Number of chunks to retrieve
            
        Returns:
            Retrieval metrics
        """
        results = []
        
        print(f"\nEvaluating retrieval quality for {len(questions)} questions...\n")
        
        for question in questions:
            # Get retrieved chunks
            chunks = self.qa_system.vector_store.search(question, top_k=top_k)
            
            result = {
                'question': question,
                'num_retrieved': len(chunks),
                'avg_distance': sum(c.get('distance', 0) for c in chunks) / len(chunks) if chunks else 0,
                'doc_types': [c['metadata'].get('doc_type') for c in chunks]
            }
            
            results.append(result)
        
        avg_distance = sum(r['avg_distance'] for r in results) / len(results)
        
        summary = {
            'total_queries': len(questions),
            'avg_distance': round(avg_distance, 4),
            'retrieval_results': results
        }
        
        print("✓ Retrieval evaluation complete!")
        print(f"  Average distance: {avg_distance:.4f}")
        
        return summary
    
    def identify_failure_modes(self, results: List[Dict]) -> List[Dict]:
        """
        Identify common failure modes from evaluation results.
        
        Args:
            results: Evaluation results
            
        Returns:
            List of identified failure modes
        """
        failures = []
        
        # Check for low confidence answers
        low_confidence = [r for r in results if r.get('confidence') == 'low']
        if low_confidence:
            failures.append({
                'mode': 'low_confidence',
                'count': len(low_confidence),
                'percentage': (len(low_confidence) / len(results)) * 100,
                'examples': [r['question'] for r in low_confidence[:3]]
            })
        
        # Check for slow responses
        slow_responses = [r for r in results if r.get('latency_seconds', 0) > 5]
        if slow_responses:
            failures.append({
                'mode': 'slow_response',
                'count': len(slow_responses),
                'percentage': (len(slow_responses) / len(results)) * 100,
                'examples': [r['question'] for r in slow_responses[:3]]
            })
        
        # Check for insufficient sources
        few_sources = [r for r in results if r.get('num_sources', 0) < 2]
        if few_sources:
            failures.append({
                'mode': 'insufficient_sources',
                'count': len(few_sources),
                'percentage': (len(few_sources) / len(results)) * 100,
                'examples': [r['question'] for r in few_sources[:3]]
            })
        
        return failures
    
    def generate_report(self, eval_results: Dict, output_path: str = None) -> str:
        """
        Generate evaluation report.
        
        Args:
            eval_results: Evaluation results
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 60)
        report.append("MedHub Lite - Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary stats
        report.append("Summary Statistics:")
        report.append(f"  Total Questions: {eval_results['total_questions']}")
        report.append(f"  Average Latency: {eval_results['avg_latency_seconds']:.2f}s")
        report.append(f"  Average Sources: {eval_results['avg_sources_per_answer']:.1f}")
        report.append("")
        
        # Failure modes
        failures = self.identify_failure_modes(eval_results['results'])
        if failures:
            report.append("Identified Failure Modes:")
            for failure in failures:
                report.append(f"  - {failure['mode']}: {failure['count']} ({failure['percentage']:.1f}%)")
                if failure['examples']:
                    report.append(f"    Examples: {', '.join(failure['examples'][:2])}")
            report.append("")
        
        # Detailed results
        report.append("Detailed Results:")
        for i, result in enumerate(eval_results['results'], 1):
            report.append(f"\n{i}. {result['question']}")
            report.append(f"   Answer: {result['answer'][:150]}...")
            report.append(f"   Sources: {result['num_sources']} | Confidence: {result['confidence']} | Latency: {result['latency_seconds']}s")
        
        report_text = "\n".join(report)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\n✓ Report saved to: {output_path}")
        
        return report_text


def load_test_questions(file_path: str) -> List[Dict]:
    """Load test questions from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_evaluation_results(results: Dict, output_path: str):
    """Save evaluation results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize system
    qa = MedicalQASystem()
    evaluator = QAEvaluator(qa)
    
    # Sample test questions
    test_questions = [
        {
            "question": "What are the patient's current work restrictions?",
            "expected_answer": "Manual check needed"
        },
        {
            "question": "What medications is the patient taking?",
            "expected_answer": "Manual check needed"
        },
        {
            "question": "What is the primary diagnosis?",
            "expected_answer": "Manual check needed"
        }
    ]
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluator.evaluate_question_set(test_questions)
    
    # Generate report
    report = evaluator.generate_report(results, output_path="evaluation_report.txt")
    print(report)
