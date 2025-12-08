"""
Ingestion Module
Handles audio and document ingestion pipelines.
"""

from .audio_ingestion import (
    AudioIngestionPipeline,
    TranscriptSegment,
    ingest_audio_file
)

from .document_ingestion import (
    DocumentIngestionPipeline,
    DocumentChunk,
    ingest_document_file,
    batch_ingest_documents
)

__all__ = [
    # Audio
    "AudioIngestionPipeline",
    "TranscriptSegment",
    "ingest_audio_file",
    
    # Documents
    "DocumentIngestionPipeline",
    "DocumentChunk",
    "ingest_document_file",
    "batch_ingest_documents",
]