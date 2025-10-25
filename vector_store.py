"""
Vector store module using ChromaDB for storing and retrieving document chunks.
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import config
from embeddings import EmbeddingGenerator


class VectorStore:
    """Manage vector storage and retrieval using ChromaDB"""
    
    def __init__(self, collection_name: str = config.COLLECTION_NAME,
                 persist_directory: str = config.CHROMA_PERSIST_DIR):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Medical document chunks for Q&A"}
            )
            print(f"Created new collection: {collection_name}")
        
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
        
        # Add to collection
        self.collection.add(
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
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'chunk_id': results['ids'][0][i]
                })
        
        return formatted_results
    
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
        except:
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
