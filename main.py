"""
MedHub Lite - Medical Document Q&A System
Main CLI application
"""
import argparse
import sys
from pathlib import Path
import json

from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from qa_system import MedicalQASystem
from evaluator import QAEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="MedHub Lite - Medical Document Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index documents
  python main.py index --input ./medical_docs/
  
  # Ask a question
  python main.py ask "What medications is the patient taking?"
  
  # Generate medical timeline
  python main.py timeline
  
  # Run evaluation
  python main.py evaluate --test-set ./test_cases.json
  
  # Reset the database
  python main.py reset
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index medical documents")
    index_parser.add_argument("--input", "-i", required=True, help="Input directory or file")
    index_parser.add_argument("--reset", action="store_true", help="Reset database before indexing")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--no-sources", action="store_true", help="Don't show sources")
    ask_parser.add_argument("--top-k", type=int, default=Config.TOP_K_RESULTS, help="Number of chunks to retrieve")
    
    # Timeline command
    timeline_parser = subparsers.add_parser("timeline", help="Generate medical timeline")
    
    # Key findings command
    findings_parser = subparsers.add_parser("findings", help="Extract key medical findings")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show collection statistics")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate system performance")
    eval_parser.add_argument("--test-set", help="Path to test set JSON file")
    eval_parser.add_argument("--output", default="evaluation_report.json", help="Output file for report")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset the vector database")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease set up your .env file with the required API keys:")
        print("  OPENAI_API_KEY=your_key_here")
        print("  ANTHROPIC_API_KEY=your_key_here (if using Claude)")
        sys.exit(1)
    
    # Initialize components
    vector_store = VectorStore()
    
    try:
        if args.command == "index":
            index_documents(args, vector_store)
        
        elif args.command == "ask":
            ask_question(args, vector_store)
        
        elif args.command == "timeline":
            generate_timeline(vector_store)
        
        elif args.command == "findings":
            extract_findings(vector_store)
        
        elif args.command == "stats":
            show_stats(vector_store)
        
        elif args.command == "evaluate":
            run_evaluation(args, vector_store)
        
        elif args.command == "reset":
            reset_database(vector_store)
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def index_documents(args, vector_store: VectorStore):
    """Index medical documents"""
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)
    
    if args.reset:
        print("Resetting database...")
        vector_store.reset_collection()
    
    # Find all documents
    if input_path.is_file():
        files = [input_path]
    else:
        files = []
        for ext in Config.SUPPORTED_FORMATS:
            files.extend(input_path.glob(f"**/*{ext}"))
    
    if not files:
        print(f"No supported documents found in {input_path}")
        print(f"Supported formats: {', '.join(Config.SUPPORTED_FORMATS)}")
        return
    
    print(f"Found {len(files)} document(s) to index")
    
    # Process and index documents
    processor = DocumentProcessor()
    total_chunks = 0
    
    for file_path in files:
        print(f"\nProcessing: {file_path.name}")
        try:
            # Load and chunk document
            doc = processor.load_document(str(file_path))
            chunks = processor.chunk_document(doc)
            
            # Add to vector store
            vector_store.add_chunks(chunks)
            
            total_chunks += len(chunks)
            print(f"  → Added {len(chunks)} chunks")
        
        except Exception as e:
            print(f"  → Error processing {file_path.name}: {e}")
            continue
    
    print(f"\n✓ Successfully indexed {len(files)} documents ({total_chunks} chunks)")
    print(f"\nYou can now ask questions with: python main.py ask \"your question\"")


def ask_question(args, vector_store: VectorStore):
    """Ask a question"""
    qa_system = MedicalQASystem(vector_store)
    
    print(f"\nQuestion: {args.question}\n")
    
    response = qa_system.answer_question(
        args.question,
        n_results=args.top_k,
        include_sources=not args.no_sources
    )
    
    print("Answer:")
    print("-" * 80)
    print(response["answer"])
    print("-" * 80)
    
    if response["sources"] and not args.no_sources:
        print(f"\nSources ({len(response['sources'])} retrieved):")
        for source in response["sources"]:
            print(f"\n[Source {source['source_number']}]")
            print(f"  File: {source['filename']}")
            print(f"  Type: {source['document_type']}")
            if source['relevance_score']:
                print(f"  Relevance: {source['relevance_score']:.2%}")
            print(f"  Preview: {source['content_preview']}")


def generate_timeline(vector_store: VectorStore):
    """Generate medical timeline"""
    qa_system = MedicalQASystem(vector_store)
    
    print("\nGenerating medical timeline...\n")
    
    timeline = qa_system.generate_medical_timeline()
    
    print("Medical Timeline:")
    print("=" * 80)
    print(timeline)
    print("=" * 80)


def extract_findings(vector_store: VectorStore):
    """Extract key medical findings"""
    qa_system = MedicalQASystem(vector_store)
    
    print("\nExtracting key medical findings...\n")
    
    findings = qa_system.extract_key_findings(top_n=10)
    
    print("Key Medical Findings:")
    print("=" * 80)
    for i, finding in enumerate(findings, 1):
        print(f"\n{i}. {finding[:300]}{'...' if len(finding) > 300 else ''}")
    print("=" * 80)


def show_stats(vector_store: VectorStore):
    """Show collection statistics"""
    stats = vector_store.get_collection_stats()
    
    print("\nCollection Statistics:")
    print("=" * 80)
    print(f"Collection name: {stats['collection_name']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Storage location: {stats['persist_directory']}")
    print("=" * 80)


def run_evaluation(args, vector_store: VectorStore):
    """Run evaluation"""
    qa_system = MedicalQASystem(vector_store)
    evaluator = QAEvaluator(qa_system)
    
    if args.test_set:
        print(f"\nEvaluating with test set: {args.test_set}")
        results = evaluator.evaluate_from_test_set(args.test_set)
        
        print("\nEvaluation Results:")
        print("=" * 80)
        print(f"Total questions: {results['total_questions']}")
        print(f"Correct answers: {results['correct_answers']}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Citation rate: {results['citation_rate']:.2%}")
        print("=" * 80)
        
        # Save detailed results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")
    
    else:
        print("\nGenerating evaluation report...")
        report = evaluator.generate_evaluation_report(args.output)
        print(f"\nReport saved to {args.output}")


def reset_database(vector_store: VectorStore):
    """Reset the vector database"""
    confirm = input("Are you sure you want to reset the database? This will delete all indexed documents. (yes/no): ")
    
    if confirm.lower() == "yes":
        vector_store.reset_collection()
        print("\n✓ Database reset successfully")
    else:
        print("\nOperation cancelled")


if __name__ == "__main__":
    main()
