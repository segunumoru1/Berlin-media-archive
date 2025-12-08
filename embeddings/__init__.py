"""
Embeddings Module
OpenAI-based text embeddings for the Berlin Media Archive.
"""

from .openai_embeddings import OpenAIEmbeddings, get_openai_embeddings

__all__ = [
    "OpenAIEmbeddings",
    "get_openai_embeddings",
]