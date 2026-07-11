# MedHub Lite - Medical Document Q&A System

> **Built to understand how EvolutionIQ's MedHub solves the 12-13 hour medical document review problem**

A production-oriented RAG prototype for medical document analysis, featuring hybrid
retrieval, claim-level evidence validation, tenant isolation, evaluation, and an AWS
deployment reference architecture.

## 🎯 The Problem

Claims professionals spend **12-13 hours** reviewing complex medical documentation for a single claim. They need to:
- Quickly understand hundreds of pages of medical records
- Extract key diagnoses, treatments, and restrictions
- Connect dots across fragmented documents
- Make defensible decisions with proper citations

## 💡 The Solution

MedHub Lite demonstrates a medical document Q&A system that:
- **Reduces review time**: From hours to minutes for key questions
- **Provides citations**: Every answer links back to source documents
- **Understands medical context**: Intelligent chunking preserves clinical meaning
- **Generates timelines**: Chronological view of medical events
- **Fails safely**: Refuses answers and extracted fields without validated evidence

## 🏗️ Architecture

```
Document → Chunking → Embedding → Vector Store → Hybrid Retrieval → Claims → Verification
   ↓           ↓          ↓            ↓            ↓         ↓        ↓
 Metadata   Token      OpenAI      Chroma       Dense +       Exact quotes +
 + tenant   bounded    embeddings  (prototype)  lexical       optional entailment
```

**Key Design Decisions:**
- **Medical-aware chunking**: Splits by clinical sections (HISTORY, ASSESSMENT, PLAN) to preserve medical context
- **OpenAI embeddings**: Fast, accurate, easy to deploy (can swap for Bio_ClinicalBERT if medical-specific needed)
- **ChromaDB**: Simple persistent store for a single-task prototype; a managed vector
  service is required before horizontal scaling
- **Structured generation**: Atomic claims and extracted fields require exact evidence
- **Security boundaries**: API-key authentication, tenant metadata filters, upload limits,
  non-root container execution, and HTTPS AWS ingress
- **Modular Python**: Production code structure, not notebooks

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medhub-lite.git
cd medhub-lite

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Index Medical Documents

```bash
# Index a single document
python main.py index --input ./sample_docs/patient_record.pdf

# Index an entire directory
python main.py index --input ./medical_docs/

# Reset database and re-index
python main.py index --input ./medical_docs/ --reset
```

### 3. Ask Questions

```bash
# Ask about medications
python main.py ask "What medications is the patient taking?"

# Ask about diagnoses
python main.py ask "What is the primary diagnosis?"

# Ask about treatment plan
python main.py ask "What is the recommended treatment plan?"
```

### 4. Generate Medical Timeline

```bash
python main.py timeline
```

### 5. View Statistics

```bash
python main.py stats
```

## 📊 Business hypotheses (not validated)

Based on EvolutionIQ's MedHub metrics and this prototype:

| Metric | Current | With System | Improvement |
|--------|---------|-------------|-------------|
| Time to answer key questions | 30-60 min | 2-3 min | **90-95% reduction** |
| Documents processed per day | 5-10 | 30-50 | **5x increase** |
| Evidence traceability | Manual | Automated claim-to-quote links | Requires expert validation |

These are product hypotheses, not measured outcomes. Real impact requires representative
documents, domain-expert review, user studies, and production telemetry.

## 🛠️ Technical Features

### Intelligent Document Processing
- **Multi-format support**: PDF and TXT
- **Medical section detection**: Automatically identifies HISTORY, EXAM, ASSESSMENT, etc.
- **Smart chunking**: Preserves clinical context, respects section boundaries
- **Metadata extraction**: Document type, dates, clinical notes classification

### Medical-Aware Retrieval
- **Semantic search**: Finds relevant content even with different terminology
- **Context window management**: Handles long medical documents efficiently
- **Metadata filtering**: Tenant-enforced retrieval plus document type, section, and dates

### Citation System
- **Source tracking**: Every answer includes source document references
- **Verifiable claims**: Users can click through to original text
- **Relevance scoring**: Shows which sources are most relevant

### Evaluation Framework
- **Retrieval metrics**: Measures how well the system finds relevant chunks
- **Answer quality**: Can use LLM-as-judge for automated evaluation
- **Test set support**: Run evaluation on question/answer pairs

## 📁 Project Structure

```
medhub-lite/
├── main.py                 # CLI application
├── config.py              # Configuration management
├── document_processor.py  # Document loading and chunking
├── vector_store.py        # Embeddings and ChromaDB interface
├── qa_system.py          # Question answering with citations
├── evaluator.py          # Evaluation framework
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

## 🔧 Configuration

Edit `config.py` or set environment variables:

```python
# LLM Provider
LLM_PROVIDER = "openai"  # or "anthropic"
LLM_MODEL = "gpt-4o-mini"  # or "claude-sonnet-4-20250514"

# Chunking
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 200  # tokens

# Retrieval
TOP_K_RESULTS = 5  # number of chunks to retrieve
```

## 🧪 Evaluation

Two eval harnesses: answer quality for the RAG pipeline, and routing accuracy for the multi-agent layer.

### Routing accuracy — multi-agent orchestrator

31 labeled queries across three routes (qa / extract / summarize), including deliberate
boundary cases (field-related questions, softly-phrased commands).

| Router prompt | Accuracy | QA | Extract | Summarize |
|---|---|---|---|---|
| Original (zero-shot) | 71.0% | 2/11 | 10/10 | 10/10 |
| Current (few-shot + decision rule, development set) | **100%** | 11/11 | 10/10 | 10/10 |

**Failure mode found:** the original prompt systematically misrouted field-related
questions ("What medications is the patient taking?") to extraction — the extract route's
description mentioned patient fields, so any question naming a field was classified as an
extraction command. **Fix:** few-shot examples per route plus an explicit decision rule
(questions about content → qa; commands to produce structured records → extract).

Because these examples informed prompt tuning, 100% is development-set accuracy rather
than an estimate of production generalization. `routing_regression_cases.json` contains
additional boundary cases found during testing. Because its misses were also used to add
deterministic safety and output-format rules, it is a regression set, not an unbiased
holdout. A new frozen set is required before estimating production generalization.

```bash
python evaluate_routing.py
```

### Answer quality — RAG pipeline

Labeled Q&A cases checking factual content and source citations (`test_cases.json`):
```json
[
  {
    "question": "What medications is the patient taking?",
    "expected_answer_contains": ["lisinopril", "metformin"],
    "should_cite_source": true
  }
]
```
```bash
python evaluate_qa.py
```

### Answerability and hallucination controls

The answerability set mixes questions supported by the corpus with deliberately
unanswerable questions. This catches both hallucinations and systems that obtain a high
refusal score by refusing everything.

```bash
python evaluate_refusal.py
```

Each offline evaluator builds a fresh temporary vector index from the declared sample
corpus and removes it after the run. It does not reuse the application database. This
prevents stale Chroma schemas, prior uploads, and accidental data leakage from changing
the score.

The runtime also filters retrieval below `SIMILARITY_THRESHOLD`, requires structured
answerability plus exact quoted source spans for every claim, and converts invalid or
unsupported answers into an explicit abstention. Structured extraction applies the same
rule to every populated field. These controls reduce risk but do not replace
expert review or claim-level entailment evaluation for consequential use cases.

### Retrieval quality

The retrieval benchmark labels relevant chunk IDs and reports Recall@K and mean
reciprocal rank independently of answer generation. Retrieval expands the semantic
candidate pool, applies the calibrated similarity floor, and reranks candidates using a
weighted combination of vector similarity and domain-term overlap.

```bash
python evaluate_retrieval.py
```

Before scoring, the retrieval evaluator verifies that every labeled chunk exists in the
fresh index. It fails explicitly if the corpus and labels have drifted rather than
silently counting labels for documents that were never indexed.

The latest small synthetic regression run is recorded in `evaluation_results.json`.
These development results are useful for catching regressions, not for estimating
production accuracy or clinical generalization.

Q&A responses also expose retrieval, generation, and total latency plus prompt and
completion token counts for cost and performance monitoring.

### Model comparison

Compare candidate generation models while holding retrieval and the test set constant:

```bash
python evaluate_models.py --models gpt-4o-mini gpt-4o
```

The report includes task accuracy, citation rate, token usage, and average latency. This
makes paid API calls.

## Production controls and AWS

The API provides `/health/live`, `/health/ready`, and authenticated `/metrics` endpoints,
JSON request logs, request IDs, bounded uploads, tenant-scoped data operations, model
token/latency telemetry, and optional semantic entailment verification.

The Terraform reference stack provisions HTTPS ALB ingress, ECS Fargate, ECR scanning,
encrypted EFS, Secrets Manager integration, CloudWatch logs, Route 53, and ACM. See
[`AWS_DEPLOYMENT.md`](AWS_DEPLOYMENT.md). The stack intentionally uses one ECS task because
embedded Chroma is not a horizontally scalable multi-writer service.

## 🎓 What I Learned Building This

1. **Medical document chunking is hard**: Can't break mid-diagnosis or mid-treatment plan
2. **Citation is critical**: Claims professionals need to verify everything
3. **Evaluation matters more than building**: How do you know if summaries are accurate?
4. **Trade-offs everywhere**: 
   - OpenAI embeddings vs Bio_ClinicalBERT (ease vs. medical specificity)
   - Chunk size (context vs. precision)
   - Top-K retrieval (recall vs. noise)

## 🚧 Future Improvements

**If this were a production system, next steps would be:**

1. **Enhanced Medical NLP**
   - Add medical entity extraction (diagnoses, medications, procedures)
   - Use medical-specific embedding models (Bio_ClinicalBERT)
   - Extract ICD-10 codes, medication dosages

2. **Better Evaluation**
   - Human-in-the-loop evaluation with real claims professionals
   - A/B testing framework
   - Failure mode analysis

3. **Production Features**
   - User authentication and multi-tenancy
   - Audit logs for compliance
   - HIPAA compliance (de-identification, encryption)
   - Real-time document ingestion pipeline

4. **Optimization**
   - Caching for common questions
   - Batch processing for bulk analysis
   - Cost optimization (smaller models for simple queries)

## 🤔 Design Decisions & Trade-offs

### Why OpenAI embeddings instead of Bio_ClinicalBERT?
- **Faster to implement**: No model hosting needed
- **Good enough for prototype**: Can always swap later
- **Trade-off**: Less medical-specific, but more general and robust
- **Decision point**: In production, would A/B test both

### Why ChromaDB instead of Pinecone/Weaviate?
- **Simpler deployment**: No external service needed
- **Fast prototyping**: Get started in minutes
- **Trade-off**: Less scalable than managed solutions
- **Decision point**: For 10K+ documents, would evaluate Pinecone

### Why GPT-4 instead of Claude?
- **Better instruction following**: Especially for structured outputs
- **Familiar API**: Easier for others to adapt
- **Trade-off**: Claude might be better for long documents
- **Decision point**: Code supports both, configurable

## 📝 Example Usage

```bash
# Index sample medical documents
python main.py index --input ./sample_docs/

# Ask medical questions
python main.py ask "What are the patient's current work restrictions?"

# Example output:
# Answer:
# --------------------------------------------------------------------------------
# The patient is currently restricted from lifting more than 10 pounds and
# should avoid prolonged standing or walking [Source 1]. Additionally, they
# are advised to take frequent breaks and avoid repetitive bending [Source 2].
# --------------------------------------------------------------------------------
#
# Sources (2 retrieved):
# [Source 1]
#   File: work_restrictions_2024.pdf
#   Type: clinical_note
#   Relevance: 94.3%
#   Preview: Physical therapy evaluation dated 3/15/2024 indicates patient...
```

## 🤝 Contributing

This is a learning project built to understand medical document Q&A systems. Feedback and suggestions welcome!

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built to understand how EvolutionIQ's MedHub transforms medical document review for insurance claims professionals. Inspired by their work reducing 12-13 hour review times to minutes while improving accuracy and claimant outcomes.

---

**Built in one weekend to learn medical document AI systems** • [View on GitHub](https://github.com/yourusername/medhub-lite)
