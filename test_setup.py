"""
Simple test script to verify the system works.
Run this after setup to ensure everything is configured correctly.
"""
import sys
import os
import importlib

# This file is an interactive setup diagnostic rather than an assertion-based
# pytest suite. Keep pytest focused on the real regression tests.
__test__ = False

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    try:
        for module in ("openai", "chromadb", "fastapi", "pypdf", "tiktoken"):
            importlib.import_module(module)
        print("✓ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def test_api_key():
    """Test that OpenAI API key is configured"""
    print("\nTesting API key configuration...")
    try:
        import config
        if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "":
            print("✓ API key is configured")
            return True
        else:
            print("✗ API key not configured")
            print("Edit .env file and add your OPENAI_API_KEY")
            return False
    except Exception as e:
        print(f"✗ Error checking API key: {e}")
        return False

def test_modules():
    """Test that all custom modules can be imported"""
    print("\nTesting custom modules...")
    try:
        for module in (
            "document_processor", "embeddings", "vector_store", "medical_ner", "qa_system"
        ):
            importlib.import_module(module)
        print("✓ All custom modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Module import error: {e}")
        return False

def test_sample_data():
    """Test that sample data exists"""
    print("\nTesting sample data...")
    sample_path = "data/sample_documents/sample_medical_record.txt"
    if os.path.exists(sample_path):
        print("✓ Sample medical record found")
        return True
    else:
        print("✗ Sample medical record not found")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("MedHub Lite - System Test")
    print("="*60)
    print()
    
    results = []
    results.append(("Package Imports", test_imports()))
    results.append(("API Key Configuration", test_api_key()))
    results.append(("Custom Modules", test_modules()))
    results.append(("Sample Data", test_sample_data()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run the CLI: python cli.py")
        print("2. Or run the API: python api.py")
        print("3. Try the quick demo:")
        print("   python cli.py")
        print("   medhub> add data/sample_documents/sample_medical_record.txt")
        print("   medhub> ask What are the patient's work restrictions?")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
