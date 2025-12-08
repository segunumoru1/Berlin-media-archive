"""
RAG Module
Core RAG functionality with attribution and citation.
"""

from .attribution_engine import (
    AttributionEngine,
    QueryResponse,
    Citation,
    query_archive
)

__all__ = [
    "AttributionEngine",
    "QueryResponse",
    "Citation",
    "query_archive",
]