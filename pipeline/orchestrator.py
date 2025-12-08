"""
Pipeline Orchestrator
Coordinates the entire multi-modal RAG pipeline.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger

from ingestion.audio_ingestion import AudioIngestionPipeline, TranscriptSegment
from ingestion.document_ingestion import DocumentIngestionPipeline, DocumentChunk
from vectorstore.chroma_store import UnifiedVectorStore
from rag.attribution_engine import AttributionEngine, QueryResponse
from utils.config import settings, ensure_directories


class BerlinArchivePipeline:
    """
    Complete pipeline for the Berlin Media Archive.
    Handles ingestion, indexing, and querying of multi-modal content.
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        reset_collection: bool = False
    ):
        """
        Initialize the complete pipeline.
        
        Args:
            collection_name: Optional custom collection name
            reset_collection: Whether to reset the collection
        """
        logger.info("=" * 80)
        logger.info("Initializing Berlin Media Archive Pipeline")
        logger.info("=" * 80)
        
        # Ensure directories exist
        ensure_directories()
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        self.audio_pipeline = AudioIngestionPipeline()
        logger.info("✓ Audio ingestion pipeline ready")
        
        self.document_pipeline = DocumentIngestionPipeline()
        logger.info("✓ Document ingestion pipeline ready")
        
        self.vector_store = UnifiedVectorStore(collection_name=collection_name)
        logger.info("✓ Vector store ready")
        
        if reset_collection:
            logger.warning("Resetting collection as requested...")
            self.vector_store.reset_collection()
        
        self.attribution_engine = AttributionEngine(self.vector_store)
        logger.info("✓ Attribution engine ready")
        
        logger.info("=" * 80)
        logger.info("Pipeline initialization complete!")
        logger.info("=" * 80)
    
    def ingest_audio_file(
        self,
        audio_path: str,
        save_transcription: bool = True
    ) -> Tuple[List[TranscriptSegment], Dict]:
        """
        Ingest and index an audio file.
        
        Args:
            audio_path: Path to audio file
            save_transcription: Whether to save transcription to disk
            
        Returns:
            Tuple of (segments, metadata)
        """
        logger.info(f"Processing audio file: {audio_path}")
        
        try:
            # Step 1: Transcribe audio
            output_dir = settings.output_dir if save_transcription else None
            segments, metadata = self.audio_pipeline.ingest_audio(
                audio_path,
                output_dir=output_dir
            )
            
            # Step 2: Add to vector store
            num_added = self.vector_store.add_audio_segments(segments, metadata)
            logger.info(f"✓ Added {num_added} audio segments to vector store")
            
            return segments, metadata
            
        except Exception as e:
            logger.error(f"Failed to process audio file: {e}")
            raise
    
    def ingest_document_file(
        self,
        document_path: str,
        save_chunks: bool = True
    ) -> Tuple[List[DocumentChunk], Dict]:
        """
        Ingest and index a document file.
        
        Args:
            document_path: Path to document file
            save_chunks: Whether to save chunks to disk
            
        Returns:
            Tuple of (chunks, metadata)
        """
        logger.info(f"Processing document: {document_path}")
        
        try:
            # Step 1: Process document
            output_dir = settings.output_dir if save_chunks else None
            chunks, metadata = self.document_pipeline.ingest_document(
                document_path,
                output_dir=output_dir
            )
            
            # Step 2: Add to vector store
            num_added = self.vector_store.add_document_chunks(chunks, metadata)
            logger.info(f"✓ Added {num_added} document chunks to vector store")
            
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            raise
    
    def batch_ingest(
        self,
        audio_dir: Optional[str] = None,
        documents_dir: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Batch ingest multiple files.
        
        Args:
            audio_dir: Directory containing audio files
            documents_dir: Directory containing documents
            
        Returns:
            Dictionary with ingestion statistics
        """
        stats = {
            "audio_files": 0,
            "audio_segments": 0,
            "document_files": 0,
            "document_chunks": 0,
            "errors": []
        }
        
        # Ingest audio files
        if audio_dir:
            audio_dir = Path(audio_dir)
            audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
            logger.info(f"Found {len(audio_files)} audio files")
            
            for audio_file in audio_files:
                try:
                    segments, _ = self.ingest_audio_file(str(audio_file))
                    stats["audio_files"] += 1
                    stats["audio_segments"] += len(segments)
                except Exception as e:
                    logger.error(f"Failed to process {audio_file.name}: {e}")
                    stats["errors"].append({"file": audio_file.name, "error": str(e)})
        
        # Ingest documents
        if documents_dir:
            documents_dir = Path(documents_dir)
            document_files = list(documents_dir.glob("*.pdf"))
            logger.info(f"Found {len(document_files)} document files")
            
            for doc_file in document_files:
                try:
                    chunks, _ = self.ingest_document_file(str(doc_file))
                    stats["document_files"] += 1
                    stats["document_chunks"] += len(chunks)
                except Exception as e:
                    logger.error(f"Failed to process {doc_file.name}: {e}")
                    stats["errors"].append({"file": doc_file.name, "error": str(e)})
        
        logger.info("=" * 80)
        logger.info("Batch Ingestion Complete!")
        logger.info(f"Audio Files: {stats['audio_files']}, Segments: {stats['audio_segments']}")
        logger.info(f"Documents: {stats['document_files']}, Chunks: {stats['document_chunks']}")
        logger.info(f"Errors: {len(stats['errors'])}")
        logger.info("=" * 80)
        
        return stats
    
    def query(
        self,
        question: str,
        save_response: bool = False,
        **kwargs
    ) -> QueryResponse:
        """
        Query the archive.
        
        Args:
            question: User's question
            save_response: Whether to save response to disk
            **kwargs: Additional arguments for query
            
        Returns:
            QueryResponse
        """
        logger.info(f"Query: {question}")
        
        response = self.attribution_engine.query_archive(question, **kwargs)
        
        if save_response:
            output_path = Path(settings.output_dir) / f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.attribution_engine.save_response(response, str(output_path))
        
        return response
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return self.vector_store.get_collection_stats()