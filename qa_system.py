"""
Main Q&A system that orchestrates document processing, retrieval, and generation.
"""
from typing import List, Dict, Optional
import openai
from document_processor import DocumentProcessor, MedicalDocument
from vector_store import VectorStore
from medical_ner import MedicalNER
import config


class MedicalQASystem:
    """
    End-to-end medical document Q&A system.
    Handles document ingestion, retrieval, and answer generation.
    """
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.medical_ner = MedicalNER()
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        print("MedHub Lite Q&A System initialized")
        print(f"Vector store stats: {self.vector_store.get_stats()}")
    
    def add_document(self, file_path: str) -> MedicalDocument:
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
        print(f"\nQuestion: {question}")
        
        # Retrieve relevant chunks
        print("Retrieving relevant documents...")
        relevant_chunks = self.vector_store.search(question, top_k=top_k)
        
        if not relevant_chunks:
            return {
                'answer': "I don't have enough information in the documents to answer this question.",
                'sources': [],
                'confidence': 'low'
            }
        
        print(f"Found {len(relevant_chunks)} relevant chunks")
        
        # Build context from retrieved chunks
        context = self._build_context(relevant_chunks)
        
        # Generate answer using LLM
        print("Generating answer...")
        answer = self._generate_answer(question, context)
        
        # Format sources with citations
        sources = self._format_sources(relevant_chunks)
        
        return {
            'answer': answer,
            'sources': sources,
            'num_sources': len(sources),
            'confidence': 'high' if len(relevant_chunks) >= 3 else 'medium'
        }
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks):
            source_info = f"[Source {i+1}: {chunk['metadata'].get('filename', 'unknown')} - {chunk['metadata'].get('doc_type', 'document')}]"
            context_parts.append(f"{source_info}\n{chunk['content']}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM with retrieved context"""
        
        system_prompt = """You are a medical document analyst helping claims professionals review medical records.

Your role:
1. Answer questions accurately based ONLY on the provided medical documents
2. Always cite which source document you're using (e.g., "According to Source 1...")
3. If the documents don't contain enough information, say so clearly
4. Focus on facts: diagnoses, treatments, restrictions, medications, and timelines
5. Use clear, professional language
6. Be precise about medical terminology

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
                max_tokens=config.MAX_TOKENS
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Format source information for citations"""
        sources = []
        for i, chunk in enumerate(chunks):
            sources.append({
                'source_number': i + 1,
                'filename': chunk['metadata'].get('filename', 'unknown'),
                'doc_type': chunk['metadata'].get('doc_type', 'document'),
                'chunk_id': chunk['chunk_id'],
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
