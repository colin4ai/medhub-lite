"""
Main Q&A system that orchestrates document processing, retrieval, and generation.
"""
from typing import List, Dict, Optional
import json
import time
import openai
from document_processor import DocumentProcessor, MedicalDocument
from vector_store import VectorStore
from medical_ner import MedicalNER
import config

UNSUPPORTED_ANSWER_MARKERS = (
    "no documented", "does not mention", "do not mention", "not mentioned",
    "not specified", "not available in the", "cannot determine",
)


class MedicalQASystem:
    """
    End-to-end medical document Q&A system.
    Handles document ingestion, retrieval, and answer generation.
    """
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.doc_processor = DocumentProcessor()
        self.vector_store = vector_store or VectorStore()
        self.medical_ner = MedicalNER()
        self.client = openai.OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=config.OPENAI_TIMEOUT_SECONDS,
            max_retries=config.OPENAI_MAX_RETRIES,
        )
        
        print("MedHub Lite Q&A System initialized")
        print(f"Vector store stats: {self.vector_store.get_stats()}")
    
    def add_document(self, file_path: str, source_name: Optional[str] = None) -> MedicalDocument:
        """
        Add a document to the system.
        
        Args:
            file_path: Path to the document (PDF or TXT)
            
        Returns:
            MedicalDocument object
        """
        print(f"\nProcessing document: {file_path}")
        
        # Load document
        if file_path.endswith('.pdf'):
            document = self.doc_processor.load_pdf(file_path)
        else:
            document = self.doc_processor.load_text(file_path)

        # Uploaded files may be staged under random temporary names. Preserve the
        # user-visible source name before chunk IDs and citation metadata are made.
        if source_name:
            from pathlib import Path
            document.metadata['filename'] = Path(source_name).name
            document.metadata['doc_id'] = Path(source_name).stem
        
        print(f"Loaded document: {document.metadata['doc_id']}")
        print(f"Document type: {document.metadata.get('doc_type', 'unknown')}")
        
        # Chunk document
        chunks = self.doc_processor.chunk_document(document)
        print(f"Created {len(chunks)} chunks")
        
        # Add to vector store
        self.vector_store.add_chunks(chunks)
        
        # Extract medical entities (optional)
        if config.USE_MEDICAL_NER:
            entities = self.medical_ner.extract_entities(document.content[:5000])  # First 5k chars
            print(f"Extracted entities: {sum(len(v) for v in entities.values())} total")
        
        return document
    
    def ask_question(self, question: str, top_k: int = config.TOP_K_RESULTS) -> Dict:
        """
        Answer a question about the medical documents.
        
        Args:
            question: The question to answer
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        if len(question) > config.MAX_QUERY_LENGTH:
            raise ValueError("Question exceeds maximum length")
        top_k = max(1, min(top_k, config.MAX_TOP_K))
        print(f"\nQuestion: {question}")
        
        # Retrieve relevant chunks
        print("Retrieving relevant documents...")
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        relevant_chunks = self.vector_store.search(question, top_k=top_k)
        retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 1)
        
        if not relevant_chunks:
            return {
                'answer': "I don't have enough information in the documents to answer this question.",
                'sources': [],
                'num_sources': 0,
                'retrieved_chunks': 0,
                'answerable': False,
                'cited_source_numbers': [],
                'top_similarity': None,
                'confidence': 'low',
                'claim_evidence': [],
                'token_usage': {},
                'latency_ms': {
                    'retrieval': retrieval_ms,
                    'generation': 0.0,
                    'total': round((time.perf_counter() - started) * 1000, 1),
                },
            }
        
        print(f"Found {len(relevant_chunks)} relevant chunks")
        
        # Build context from retrieved chunks
        context = self._build_context(relevant_chunks)
        
        # Generate answer using LLM
        print("Generating answer...")
        generation_started = time.perf_counter()
        generated = self._generate_answer(question, context)
        generation_ms = round((time.perf_counter() - generation_started) * 1000, 1)
        
        # Format sources with citations
        sources = self._format_sources(relevant_chunks)
        
        valid_source_numbers = {s['source_number'] for s in sources}
        claims = generated.get('claims')
        claim_evidence = []
        if isinstance(claims, list):
            generated_answer, cited, evidence_valid, claim_evidence = self._validate_claims(
                claims, relevant_chunks
            )
        else:
            # Backward-compatible validation for previously recorded responses.
            cited = set(generated.get('cited_source_numbers', []))
            generated_answer = generated.get('answer', '')
            evidence_valid = True
        if any(not isinstance(number, int) for number in cited):
            cited = set()
        admits_missing_evidence = any(
            marker in generated_answer.lower() for marker in UNSUPPORTED_ANSWER_MARKERS
        )
        answerable = (
            bool(generated.get('answerable')) and bool(cited)
            and not admits_missing_evidence and evidence_valid
        )
        citations_valid = cited.issubset(valid_source_numbers)
        if not answerable or not citations_valid:
            answer = "I don't have enough information in the documents to answer this question."
            answerable = False
            cited = set()
        else:
            answer = generated_answer
            if '[source' not in answer.lower():
                labels = ", ".join(str(number) for number in sorted(cited))
                answer = f"{answer} [Source{'s' if len(cited) > 1 else ''} {labels}]"

        top_similarity = max((c.get('similarity', 0.0) for c in relevant_chunks), default=0.0)
        if not answerable:
            confidence = 'low'
        elif top_similarity >= 0.85:
            confidence = 'high'
        else:
            confidence = 'medium'

        return {
            'answer': answer,
            'sources': sources,
            'num_sources': len(sources),
            'retrieved_chunks': len(relevant_chunks),
            'answerable': answerable,
            'cited_source_numbers': sorted(cited),
            'top_similarity': round(top_similarity, 4),
            'confidence': confidence,
            'claim_evidence': claim_evidence if answerable else [],
            'token_usage': generated.get('_usage', {}),
            'latency_ms': {
                'retrieval': retrieval_ms,
                'generation': generation_ms,
                'total': round((time.perf_counter() - started) * 1000, 1),
            },
        }

    @staticmethod
    def _validate_claims(claims: List[Dict], chunks: List[Dict]):
        """Require every claim to quote a real span from at least one cited source."""
        if not claims:
            return "", set(), False, []
        source_text = {
            index + 1: " ".join(chunk['content'].lower().split())
            for index, chunk in enumerate(chunks)
        }
        rendered = []
        cited = set()
        validated = []
        for claim in claims:
            text = claim.get('text') if isinstance(claim, dict) else None
            evidence = claim.get('evidence', []) if isinstance(claim, dict) else []
            if not isinstance(text, str) or not text.strip() or not isinstance(evidence, list):
                return "", set(), False, []
            valid_evidence = []
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                number, quote = item.get('source_number'), item.get('quote')
                normalized_quote = " ".join(quote.lower().split()) if isinstance(quote, str) else ""
                if (
                    isinstance(number, int) and number in source_text
                    and len(normalized_quote) >= 12
                    and normalized_quote in source_text[number]
                ):
                    valid_evidence.append({"source_number": number, "quote": quote})
                    cited.add(number)
            if not valid_evidence:
                return "", set(), False, []
            labels = ", ".join(str(number) for number in sorted({e['source_number'] for e in valid_evidence}))
            rendered.append(f"{text.strip()} [Source{'s' if ',' in labels else ''} {labels}]")
            validated.append({"claim": text.strip(), "evidence": valid_evidence})
        return " ".join(rendered), cited, True, validated
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks):
            source_info = f"[Source {i+1}: {chunk['metadata'].get('filename', 'unknown')} - {chunk['metadata'].get('doc_type', 'document')}]"
            context_parts.append(f"{source_info}\n{chunk['content']}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> Dict:
        """Generate answer using LLM with retrieved context"""
        
        system_prompt = """You are a medical document analyst helping claims professionals review medical records.

Your role:
1. Answer questions accurately based ONLY on the provided medical documents
2. Always cite which source document you're using (e.g., "According to Source 1...")
3. If the documents don't directly support an answer, set answerable=false
4. Focus on facts: diagnoses, treatments, restrictions, medications, and timelines
5. Use clear, professional language
6. Be precise about medical terminology

7. Never supplement the documents with general medical knowledge or speculation
8. Absence of a fact is not evidence that it is false. For history questions,
   answerable=true only when the documents explicitly state the history or its absence

Return JSON with exactly these fields:
{"answerable": true|false, "claims": [
  {"text": "one concise factual claim", "evidence": [
    {"source_number": 1, "quote": "an exact supporting quote copied from Source 1"}
  ]}
]}
Every material claim must have at least one exact, verbatim supporting quote. Keep quotes
short but specific. For an unanswerable question, return an empty claims list.

Remember: Claims professionals will use your answers to make important decisions, so accuracy and citations are critical."""

        user_prompt = f"""Based on the following medical documents, please answer this question:

Question: {question}

Medical Documents:
{context}

Please provide a clear, accurate answer with citations to the source documents."""

        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content)
            if not isinstance(parsed.get('answerable'), bool):
                raise ValueError("Model response did not contain answerable boolean")
            if not isinstance(parsed.get('claims'), list):
                raise ValueError("Model response did not contain a claims list")
            usage = getattr(response, 'usage', None)
            parsed['_usage'] = {
                'prompt_tokens': getattr(usage, 'prompt_tokens', 0) or 0,
                'completion_tokens': getattr(usage, 'completion_tokens', 0) or 0,
                'total_tokens': getattr(usage, 'total_tokens', 0) or 0,
            }
            return parsed
        
        except Exception as e:
            print(f"Error generating answer: {e}")
            raise
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Format source information for citations"""
        sources = []
        for i, chunk in enumerate(chunks):
            sources.append({
                'source_number': i + 1,
                'filename': chunk['metadata'].get('filename', 'unknown'),
                'doc_type': chunk['metadata'].get('doc_type', 'document'),
                'chunk_id': chunk['chunk_id'],
                'similarity': round(chunk.get('similarity', 0.0), 4),
                'lexical_score': round(chunk.get('lexical_score', 0.0), 4),
                'hybrid_score': round(chunk.get('hybrid_score', 0.0), 4),
                'content_preview': chunk['content'][:200] + "..."
            })
        return sources
    
    def get_document_summary(self, doc_id: str) -> Dict:
        """
        Get a summary of a specific document.
        
        Args:
            doc_id: The document ID
            
        Returns:
            Dictionary with document summary
        """
        # Search for chunks from this document
        results = self.vector_store.collection.get(
            where={"doc_id": doc_id}
        )
        
        if not results['documents']:
            return {'error': f'Document {doc_id} not found'}
        
        # Combine all chunks
        full_text = ' '.join(results['documents'])
        
        # Extract medical profile
        if config.USE_MEDICAL_NER:
            profile = self.medical_ner.summarize_medical_profile(full_text)
        else:
            profile = {}
        
        return {
            'doc_id': doc_id,
            'num_chunks': len(results['documents']),
            'medical_profile': profile,
            'metadata': results['metadatas'][0] if results['metadatas'] else {}
        }
    
    def get_timeline(self, doc_id: Optional[str] = None) -> List[Dict]:
        """
        Extract medical timeline from documents.
        
        Args:
            doc_id: Optional document ID to filter by
            
        Returns:
            List of timeline events
        """
        # Get relevant documents
        if doc_id:
            results = self.vector_store.collection.get(where={"doc_id": doc_id})
        else:
            # Get all documents
            results = self.vector_store.collection.get()
        
        if not results['documents']:
            return []
        
        # Combine text and extract events
        full_text = ' '.join(results['documents'][:10])  # Limit to first 10 chunks
        events = self.medical_ner.extract_timeline_events(full_text)
        
        return events
    
    def clear_all_documents(self):
        """Clear all documents from the system"""
        self.vector_store.clear_all()
        print("All documents cleared")
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        return self.vector_store.get_stats()
