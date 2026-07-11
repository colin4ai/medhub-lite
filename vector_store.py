"""
Vector store module using ChromaDB for storing and retrieving document chunks.
"""
from typing import List, Dict, Optional
import re
import chromadb
import config
from embeddings import EmbeddingGenerator


class VectorStore:
    """Manage vector storage and retrieval using ChromaDB"""
    
    def __init__(self, collection_name: str = config.COLLECTION_NAME,
                 persist_directory: str = config.CHROMA_PERSIST_DIR):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # PersistentClient is explicit: data survives API/CLI process restarts.
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Atomic and compatible across Chroma versions; avoids broad exception
        # handling around differing NotFoundError implementations.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Medical document chunks for Q&A"},
        )
        
        self.embedding_generator = EmbeddingGenerator()
    
    def add_chunks(self, chunks: List[Dict]):
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'content', 'chunk_id', and metadata
        """
        if not chunks:
            return
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Extract data for ChromaDB
        documents = [chunk['content'] for chunk in chunks]
        ids = [chunk['chunk_id'] for chunk in chunks]
        metadatas = [{k: v for k, v in chunk.items() if k != 'content'} 
                     for chunk in chunks]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings_batch(documents)
        
        # Upsert makes re-ingestion idempotent for deterministic chunk IDs.
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"Successfully added {len(chunks)} chunks")
    
    def search(self, query: str, top_k: int = config.TOP_K_RESULTS,
               filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant chunks with their content and metadata
        """
        collection_size = self.collection.count()
        if collection_size == 0:
            return []

        # Generate a broader semantic candidate set, then rerank it using both
        # vector similarity and exact domain-term overlap.
        query_embedding = self.embedding_generator.generate_embedding(query)
        candidate_k = min(
            collection_size,
            max(top_k, top_k * config.RETRIEVAL_CANDIDATE_MULTIPLIER),
        )
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            where=filter_metadata
        )
        
        # Chroma's default collection returns L2 distance (lower is better).
        # Convert it to an intuitive 0..1 score and reject weak evidence.
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i] if 'distances' in results else None
                similarity = 1.0 / (1.0 + distance) if distance is not None else 0.0
                if similarity < config.SIMILARITY_THRESHOLD:
                    continue
                lexical_score = self._lexical_score(query, results['documents'][0][i])
                hybrid_score = (
                    config.VECTOR_SCORE_WEIGHT * similarity
                    + (1 - config.VECTOR_SCORE_WEIGHT) * lexical_score
                )
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': distance,
                    'similarity': similarity,
                    'lexical_score': lexical_score,
                    'hybrid_score': hybrid_score,
                    'chunk_id': results['ids'][0][i]
                })
        
        formatted_results.sort(key=lambda item: item['hybrid_score'], reverse=True)
        return formatted_results[:top_k]

    @staticmethod
    def _lexical_score(query: str, document: str) -> float:
        stop_words = {
            "a", "an", "and", "are", "did", "do", "does", "for", "had", "has",
            "have", "in", "is", "of", "the", "to", "was", "were", "what", "when",
            "with", "patient", "patients", "currently",
        }
        def stem(token: str) -> str:
            if token.endswith("ies") and len(token) > 4:
                return token[:-3] + "y"
            if token.endswith("ing") and len(token) > 5:
                return token[:-3]
            if token.endswith("s") and len(token) > 3:
                return token[:-1]
            return token

        def tokenize(text: str):
            return {
                stem(token) for token in re.findall(r"[a-z0-9]+", text.lower())
                if token not in stop_words and len(token) > 1
            }
        query_terms = tokenize(query)
        if not query_terms:
            return 0.0
        return len(query_terms.intersection(tokenize(document))) / len(query_terms)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: The chunk ID
            
        Returns:
            Chunk dictionary or None if not found
        """
        try:
            result = self.collection.get(ids=[chunk_id])
            if result['documents']:
                return {
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0],
                    'chunk_id': result['ids'][0]
                }
        except (ValueError, IndexError, KeyError):
            return None
        return None
    
    def delete_document(self, doc_id: str):
        """
        Delete all chunks from a specific document.
        
        Args:
            doc_id: The document ID
        """
        # Get all chunks for this document
        results = self.collection.get(
            where={"doc_id": doc_id}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} chunks from document {doc_id}")
    
    def clear_all(self):
        """Clear all data from the collection"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Medical document chunks for Q&A"}
        )
        print("Cleared all data from vector store")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection_name
        }
