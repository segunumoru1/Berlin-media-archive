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
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()


# Try to import settings, fallback to os.getenv
try:
    from utils.config import settings
    USE_SETTINGS = True
except ImportError:
    USE_SETTINGS = False
    logger.warning("Could not import settings, using environment variables directly")


class OpenAIEmbeddings:
    """OpenAI embeddings with caching support"""
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        # Get API key
        if api_key:
            self.api_key = api_key
        elif USE_SETTINGS:
            self.api_key = settings.openai_api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key or self.api_key == "your_openai_key_here":
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in your .env file."
            )
        
        # Get model
        if model:
            self.model = model
        elif USE_SETTINGS:
            self.model = settings.embedding_model
        else:
            self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key)
        
        # Set dimensions based on model
        if "text-embedding-3-large" in self.model:
            self.dimensions = 3072
        elif "text-embedding-3-small" in self.model:
            self.dimensions = 1536
        elif "text-embedding-ada-002" in self.model:
            self.dimensions = 1536
        else:
            self.dimensions = 1536  # Default
        
        # Set batch size for batch processing
        self.batch_size = 100
        
        logger.info(f"OpenAI Embeddings initialized with model: {self.model}")
    
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