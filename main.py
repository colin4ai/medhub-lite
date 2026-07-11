"""Command-line interface for MedHub Lite."""
import argparse
import json
from pathlib import Path

import config
from evaluator import QAEvaluator
from qa_system import MedicalQASystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MedHub Lite medical document Q&A")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="Index a file or directory")
    index.add_argument("--input", "-i", required=True)
    index.add_argument("--reset", action="store_true")

    ask = commands.add_parser("ask", help="Ask a grounded question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=config.TOP_K_RESULTS)
    ask.add_argument("--no-sources", action="store_true")

    commands.add_parser("timeline", help="Show the extracted timeline")
    commands.add_parser("stats", help="Show vector-store statistics")
    commands.add_parser("reset", help="Delete all indexed chunks")

    evaluate = commands.add_parser("evaluate", help="Run the labeled Q&A evaluation")
    evaluate.add_argument("--test-set", required=True)
    evaluate.add_argument("--output", default="evaluation_report.json")
    return parser


def index_documents(qa: MedicalQASystem, input_value: str) -> int:
    path = Path(input_value)
    if not path.exists():
        raise FileNotFoundError(path)
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.suffix.lower() in {".pdf", ".txt"}
    )
    if not files:
        raise ValueError(f"No PDF or TXT documents found under {path}")
    for file_path in files:
        qa.add_document(str(file_path))
    return len(files)


def main() -> None:
    args = build_parser().parse_args()
    qa = MedicalQASystem()

    if args.command == "index":
        if args.reset:
            qa.clear_all_documents()
        count = index_documents(qa, args.input)
        print(f"Indexed {count} document(s)")
    elif args.command == "ask":
        result = qa.ask_question(args.question, top_k=args.top_k)
        print(result["answer"])
        if not args.no_sources:
            print(json.dumps(result.get("sources", []), indent=2))
    elif args.command == "timeline":
        print(json.dumps(qa.get_timeline(), indent=2))
    elif args.command == "stats":
        print(json.dumps(qa.get_system_stats(), indent=2))
    elif args.command == "reset":
        qa.clear_all_documents()
    elif args.command == "evaluate":
        results = QAEvaluator(qa).evaluate_from_test_set(args.test_set)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps({k: v for k, v in results.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
