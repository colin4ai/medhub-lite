"""
Configuration settings for MedHub Lite
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4"  # Use gpt-3.5-turbo for cost optimization
MAX_TOKENS = 1000
TEMPERATURE = 0.1  # Low temperature for factual medical information

# ChromaDB Configuration
CHROMA_PERSIST_DIR = "./data/chroma_db"
COLLECTION_NAME = "medical_documents"

# Document Processing Configuration
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 200  # tokens
MAX_CHUNKS_PER_DOC = 100

# Retrieval Configuration
TOP_K_RESULTS = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7

# Medical NER Configuration
USE_MEDICAL_NER = True
MEDICAL_ENTITIES = [
    "DIAGNOSIS",
    "MEDICATION",
    "PROCEDURE",
    "SYMPTOM",
    "BODY_PART",
    "DOSAGE"
]

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Evaluation Configuration
EVAL_QUESTIONS_PATH = "./data/evaluation/test_questions.json"
