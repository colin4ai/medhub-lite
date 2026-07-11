"""
Embedding generation module using OpenAI's embedding API.
"""
from typing import List
from functools import lru_cache
import openai
import config


class EmbeddingGenerator:
    """Generate embeddings for text using OpenAI"""
    
    def __init__(self, model: str = config.EMBEDDING_MODEL):
        self.model = model
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    
    @lru_cache(maxsize=config.EMBEDDING_CACHE_SIZE)
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error generating embeddings batch: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model"""
        # text-embedding-3-small has 1536 dimensions
        # text-embedding-3-large has 3072 dimensions
        if "small" in self.model:
            return 1536
        elif "large" in self.model:
            return 3072
        else:
            return 1536  # default
