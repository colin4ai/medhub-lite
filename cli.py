"""
Command-line interface for MedHub Lite.
"""
from pathlib import Path
from qa_system import MedicalQASystem


def print_header():
    """Print CLI header"""
    print("\n" + "="*60)
    print("  MedHub Lite - Medical Document Q&A System")
    print("  Type 'help' for commands, 'quit' to exit")
    print("="*60 + "\n")


def print_help():
    """Print available commands"""
    print("\nAvailable commands:")
    print("  add <file_path>     - Add a document to the system")
    print("  ask <question>      - Ask a question about the documents")
    print("  summary <doc_id>    - Get summary of a document")
    print("  timeline [doc_id]   - Get medical timeline")
    print("  stats               - Show system statistics")
    print("  clear               - Clear all documents")
    print("  help                - Show this help message")
    print("  quit                - Exit the program")
    print()


def main():
    """Main CLI loop"""
    print_header()
    
    # Initialize Q&A system
    print("Initializing system...")
    qa = MedicalQASystem()
    print("System ready!\n")
    
    while True:
        try:
            # Get user input
            user_input = input("medhub> ").strip()
            
            if not user_input:
                continue
            
            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # Execute command
            if command == "quit" or command == "exit":
                print("\nGoodbye!")
                break
            
            elif command == "help":
                print_help()
            
            elif command == "add":
                if not args:
                    print("Usage: add <file_path>")
                    continue
                
                file_path = args.strip()
                if not Path(file_path).exists():
                    print(f"Error: File not found: {file_path}")
                    continue
                
                try:
                    document = qa.add_document(file_path)
                    print("\n✓ Document added successfully!")
                    print(f"  ID: {document.metadata['doc_id']}")
                    print(f"  Type: {document.metadata.get('doc_type', 'unknown')}")
                    print(f"  Chunks: {len(document.chunks)}\n")
                except Exception as e:
                    print(f"Error adding document: {e}\n")
            
            elif command == "ask":
                if not args:
                    print("Usage: ask <question>")
                    continue
                
                question = args.strip()
                try:
                    result = qa.ask_question(question)
                    print("\n📝 Answer:")
                    print(f"{result['answer']}\n")
                    print(f"📚 Sources ({result['num_sources']}):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  {i}. {source['filename']} ({source['doc_type']})")
                    print(f"\n🎯 Confidence: {result['confidence']}\n")
                except Exception as e:
                    print(f"Error answering question: {e}\n")
            
            elif command == "summary":
                if not args:
                    print("Usage: summary <doc_id>")
                    continue
                
                doc_id = args.strip()
                try:
                    summary = qa.get_document_summary(doc_id)
                    if 'error' in summary:
                        print(f"Error: {summary['error']}\n")
                    else:
                        print(f"\n📄 Document Summary: {doc_id}")
                        print(f"  Chunks: {summary['num_chunks']}")
                        if 'medical_profile' in summary and summary['medical_profile']:
                            profile = summary['medical_profile']
                            print("\n  Medical Profile:")
                            print(f"    Diagnoses: {', '.join(profile.get('diagnoses', [])) or 'None found'}")
                            print(f"    Medications: {', '.join(profile.get('medications', [])) or 'None found'}")
                            print(f"    Symptoms: {', '.join(profile.get('key_symptoms', [])) or 'None found'}")
                        print()
                except Exception as e:
                    print(f"Error getting summary: {e}\n")
            
            elif command == "timeline":
                doc_id = args.strip() if args else None
                try:
                    events = qa.get_timeline(doc_id=doc_id)
                    print(f"\n📅 Medical Timeline ({len(events)} events):")
                    for event in events[:10]:  # Show first 10
                        print(f"  {event.get('date', 'Unknown')}: {event.get('event', '')[:80]}...")
                    if len(events) > 10:
                        print(f"  ... and {len(events)-10} more events")
                    print()
                except Exception as e:
                    print(f"Error getting timeline: {e}\n")
            
            elif command == "stats":
                try:
                    stats = qa.get_system_stats()
                    print("\n📊 System Statistics:")
                    print(f"  Total chunks: {stats['total_chunks']}")
                    print(f"  Collection: {stats['collection_name']}")
                    print()
                except Exception as e:
                    print(f"Error getting stats: {e}\n")
            
            elif command == "clear":
                confirm = input("Are you sure you want to clear all documents? (yes/no): ")
                if confirm.lower() == "yes":
                    try:
                        qa.clear_all_documents()
                        print("✓ All documents cleared\n")
                    except Exception as e:
                        print(f"Error clearing documents: {e}\n")
                else:
                    print("Cancelled\n")
            
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands\n")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
