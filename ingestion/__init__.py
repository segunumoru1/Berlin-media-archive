"""
Ingestion Module
Handles audio and document ingestion pipelines.
"""

from .audio_ingestion import AudioIngestionPipeline, TranscriptSegment
from .document_ingestion import DocumentIngestionPipeline, DocumentChunk

__all__ = [
    "AudioIngestionPipeline",
    "TranscriptSegment",
    "DocumentIngestionPipeline",
    "DocumentChunk",
]