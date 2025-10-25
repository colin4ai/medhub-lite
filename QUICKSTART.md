# MedHub Lite - Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
# Run the setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Get your key from: https://platform.openai.com/api-keys
nano .env  # or use your preferred editor
```

### 3. Test the System

```bash
# Run the CLI
python cli.py

# Try these commands:
medhub> add data/sample_documents/sample_medical_record.txt
medhub> ask What are the patient's work restrictions?
medhub> ask What medications is the patient taking?
medhub> summary sample_medical_record
medhub> stats
```

## Demo for Interview

### Show the Architecture

"Let me walk you through what I built this weekend..."

```bash
# Show the modular structure
ls -la

# Key files:
# - document_processor.py: Handles PDF/text ingestion and chunking
# - embeddings.py: OpenAI embedding generation
# - vector_store.py: ChromaDB for semantic search
# - medical_ner.py: Entity extraction (diagnoses, meds, restrictions)
# - qa_system.py: Main orchestration logic
# - api.py: REST API with FastAPI
# - evaluation.py: Built-in evaluation framework
```

### Show It Working

```bash
# 1. Start with a clean system
python cli.py
medhub> stats  # Show it's empty

# 2. Add a medical document
medhub> add data/sample_documents/sample_medical_record.txt
# Explain: "This ingests the document, chunks it semantically, 
# generates embeddings, and stores in vector database"

# 3. Ask a complex question
medhub> ask What are the patient's current work restrictions and how have they changed over time?
# Point out: "Notice it cites sources and provides the chunk IDs"

# 4. Show entity extraction
medhub> summary sample_medical_record
# Explain: "This extracts medical entities - diagnoses, meds, symptoms"
```

### Explain Design Decisions

**Chunking Strategy:**
"I used semantic chunking by paragraphs rather than fixed token sizes because medical information is context-dependent. Breaking mid-diagnosis loses meaning."

**Embedding Model:**
"I chose OpenAI text-embedding-3-small for speed and cost. In production, I'd benchmark against Bio_ClinicalBERT for medical-specific embeddings."

**LLM Choice:**
"Using GPT-4 for accuracy. Could downgrade to GPT-3.5-turbo (20x cheaper) and measure the accuracy delta. That's a classic cost vs. quality trade-off."

**Citation System:**
"Every answer links back to source chunks. Critical for claims professionals who need to verify information before making decisions."

### Show the API

```bash
# Run the API server
python api.py

# In another terminal, test the endpoints:
curl -X POST "http://localhost:8000/upload" \
  -F "file=@data/sample_documents/sample_medical_record.txt"

curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the primary diagnosis?"}'

# Or visit http://localhost:8000/docs for interactive API documentation
```

### Show Evaluation Framework

```bash
python evaluation.py

# Explain:
# "I built evaluation to measure accuracy, latency, and identify failure modes.
# In production, I'd run this continuously to track performance."
```

## Key Talking Points

### What I Built
- **Complete RAG system** for medical document Q&A
- **Modular Python code** (not notebooks) - production-ready
- **REST API** for integration with existing systems
- **Citation system** for verifiability
- **Evaluation framework** to measure performance
- Built in **one weekend** to learn the domain

### Technical Depth I Can Discuss
- Why semantic chunking matters for medical documents
- Embedding model trade-offs (cost vs. accuracy)
- Context window management with RoPE
- LLM-as-judge for evaluation
- Vector similarity metrics
- When to fine-tune vs. RAG vs. prompt engineering

### Business Value
- **Time savings**: 12-13 hours → minutes for medical review
- **Accuracy**: Citations enable verification
- **Scalability**: Handles hundreds of pages per claim
- **Integration**: REST API plugs into existing claims systems

### What I'd Do Next (V2)
1. Add medical-specific NER model (scispaCy or custom fine-tuned)
2. Implement timeline visualization
3. Add structured output (JSON responses)
4. Build evaluation dashboard
5. Benchmark different embedding models
6. Add A/B testing framework

### Connection to MedHub
"This is essentially a lightweight version of what EvolutionIQ's MedHub does - medical document synthesis for claims guidance. I built it to understand the problem space, and now I'm ready to work on the real thing."

## Common Questions & Answers

**Q: Why didn't you use medical-specific models?**
A: "For a weekend MVP, OpenAI models were fastest to get working. For production, I'd definitely benchmark Bio_ClinicalBERT or PubMedBERT. That's a perfect example of 'ship fast, iterate based on measurements.'"

**Q: How would you handle HIPAA compliance?**
A: "Great question. For production: 1) Use Business Associate Agreement with OpenAI or host models privately, 2) Implement PII detection/redaction, 3) Audit logging, 4) Access controls, 5) Data encryption at rest and in transit."

**Q: How does this scale?**
A: "ChromaDB works for MVP. For production scale, I'd evaluate Pinecone or Weaviate. Also would add caching, async processing, and potentially deploy embeddings separately from generation for better resource utilization."

**Q: How do you evaluate accuracy?**
A: "Multiple approaches: 1) LLM-as-judge comparing answers to ground truth, 2) Human evaluation on sample questions, 3) Retrieval metrics (precision/recall), 4) A/B testing in production. I built the evaluation framework to measure all of this."

## What This Demonstrates

✅ **Fast learner**: Built functional system in one weekend
✅ **Pragmatic**: Made smart trade-offs to ship quickly  
✅ **Product thinking**: Focused on business problem (claims processing time)
✅ **Technical depth**: Can discuss embeddings, chunking, evaluation
✅ **Production-ready**: Modular code, API, evaluation, not just a notebook
✅ **Domain understanding**: Researched EvolutionIQ and MedHub specifically

---

**You're not asking them to take a leap of faith. You're showing them you've already started the work.**
