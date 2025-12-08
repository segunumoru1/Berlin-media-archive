"""
OpenAI Embeddings Module
Handles all text embeddings using OpenAI's embedding models.
Works for both documents and audio transcripts.
"""

import os
from typing import List, Optional, Union
import numpy as np
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenAIEmbeddings:
    """
    OpenAI embeddings service for all text content.
    Handles both document chunks and audio transcripts.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        batch_size: int = 100
    ):
        """
        Initialize OpenAI embeddings.
        
        Args:
            model: OpenAI embedding model (text-embedding-3-large or text-embedding-ada-002)
            api_key: OpenAI API key (or use env variable)
            batch_size: Number of texts to embed in one batch
        """
        self.model = model
        self.batch_size = batch_size
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        
        # Get embedding dimensions
        self.dimensions = 3072 if "large" in model else 1536
        
        logger.info(f"OpenAIEmbeddings initialized with model: {model} ({self.dimensions}D)")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"Embedding {len(texts)} texts...")
            
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Embedded batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            logger.info(f"Successfully embedded {len(all_embeddings)} texts")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        return self.embed_text(query)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.dimensions


def get_openai_embeddings(
    model: str = "text-embedding-3-large",
    api_key: Optional[str] = None
) -> OpenAIEmbeddings:
    """
    Factory function to get OpenAI embeddings instance.
    
    Args:
        model: OpenAI embedding model
        api_key: Optional API key
        
    Returns:
        OpenAIEmbeddings instance
    """
    return OpenAIEmbeddings(model=model, api_key=api_key)