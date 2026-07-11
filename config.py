"""
Configuration settings for MedHub Lite
"""
import os
import json
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "512"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "4000"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))

# ChromaDB Configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME = "medical_documents"

# Document Processing Configuration
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 200  # tokens
MAX_CHUNKS_PER_DOC = 100

# Retrieval Configuration
TOP_K_RESULTS = 5  # Number of chunks to retrieve
# Initial calibration on the answerability set: supported questions had top
# scores of 0.429-0.534; unsupported questions were mostly below 0.425. Keep
# environment-configurable and re-calibrate when the corpus/model changes.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.425"))
RETRIEVAL_CANDIDATE_MULTIPLIER = int(os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "3"))
VECTOR_SCORE_WEIGHT = float(os.getenv("VECTOR_SCORE_WEIGHT", "0.8"))

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
API_AUTH_KEY = os.getenv("API_AUTH_KEY", "")
try:
    TENANT_API_KEYS = json.loads(os.getenv("TENANT_API_KEYS_JSON", "{}"))
    if not isinstance(TENANT_API_KEYS, dict) or any(
        not isinstance(tenant, str) or not tenant
        or not isinstance(key, str) or not key
        for tenant, key in TENANT_API_KEYS.items()
    ):
        raise ValueError
except (json.JSONDecodeError, ValueError) as exc:
    raise ValueError("TENANT_API_KEYS_JSON must be a JSON object") from exc
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".txt"}
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "default")
ENABLE_ENTAILMENT_VERIFIER = os.getenv("ENABLE_ENTAILMENT_VERIFIER", "false").lower() == "true"
APP_VERSION = os.getenv("APP_VERSION", "dev")

# Evaluation Configuration
EVAL_QUESTIONS_PATH = "./data/evaluation/test_questions.json"
