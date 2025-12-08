"""
Comprehensive Tests for Berlin Media Archive Pipeline
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from ingestion.audio_ingestion import AudioIngestionPipeline, TranscriptSegment
from ingestion.document_ingestion import DocumentIngestionPipeline, DocumentChunk
from vectorstore.chroma_store import UnifiedVectorStore
from rag.attribution_engine import AttributionEngine
from pipeline.orchestrator import BerlinArchivePipeline


class TestAudioIngestion:
    """Tests for audio ingestion pipeline."""
    
    @pytest.fixture
    def audio_pipeline(self):
        """Create audio pipeline fixture."""
        return AudioIngestionPipeline(whisper_model="tiny", enable_diarization=False)
    
    def test_audio_pipeline_initialization(self, audio_pipeline):
        """Test audio pipeline initializes correctly."""
        assert audio_pipeline is not None
        assert audio_pipeline.whisper_model is not None
        assert audio_pipeline.whisper_model_name == "tiny"
    
    def test_transcript_segment_creation(self):
        """Test TranscriptSegment dataclass."""
        segment = TranscriptSegment(
            text="This is a test segment",
            start_time=10.5,
            end_time=15.2,
            speaker="SPEAKER_01"
        )
        
        assert segment.text == "This is a test segment"
        assert segment.start_time == 10.5
        assert segment.end_time == 15.2
        assert segment.speaker == "SPEAKER_01"
        assert "10" in segment.get_timestamp_str()
    
    def test_timestamp_formatting(self):
        """Test timestamp string formatting."""
        segment = TranscriptSegment(
            text="Test",
            start_time=125.5,  # 2:05
            end_time=185.2      # 3:05
        )
        
        timestamp = segment.get_timestamp_str()
        assert "02:05" in timestamp or "2:05" in timestamp
        assert "03:05" in timestamp or "3:05" in timestamp


class TestDocumentIngestion:
    """Tests for document ingestion pipeline."""
    
    @pytest.fixture
    def document_pipeline(self):
        """Create document pipeline fixture."""
        return DocumentIngestionPipeline(chunk_size=500, chunk_overlap=50)
    
    def test_document_pipeline_initialization(self, document_pipeline):
        """Test document pipeline initializes correctly."""
        assert document_pipeline is not None
        assert document_pipeline.chunk_size == 500
        assert document_pipeline.chunk_overlap == 50
        assert document_pipeline.text_splitter is not None
    
    def test_document_chunk_creation(self):
        """Test DocumentChunk dataclass."""
        chunk = DocumentChunk(
            text="This is a test chunk from page 5",
            page_number=5,
            chunk_index=2,
            source="/path/to/document.pdf",
            metadata={"document_name": "test.pdf"}
        )
        
        assert chunk.text == "This is a test chunk from page 5"
        assert chunk.page_number == 5
        assert chunk.chunk_index == 2
        assert chunk.chunk_id is not None  # Auto-generated
        assert "Page 5" in chunk.get_citation()
    
    def test_text_cleaning(self, document_pipeline):
        """Test text cleaning functionality."""
        dirty_text = "This   has    excessive    spaces\n\n\n\nand newlines"
        clean_text = document_pipeline._clean_text(dirty_text)
        
        assert "   " not in clean_text
        assert "\n\n\n" not in clean_text


class TestVectorStore:
    """Tests for vector store operations."""
    
    @pytest.fixture
    def vector_store(self):
        """Create vector store fixture with temp directory."""
        temp_dir = tempfile.mkdtemp()
        store = UnifiedVectorStore(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        yield store
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_vector_store_initialization(self, vector_store):
        """Test vector store initializes correctly."""
        assert vector_store is not None
        assert vector_store.collection is not None
        assert vector_store.embedding_provider is not None
    
    def test_add_audio_segments(self, vector_store):
        """Test adding audio segments to vector store."""
        segments = [
            TranscriptSegment(
                text="First segment about Berlin history",
                start_time=0.0,
                end_time=5.0,
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                text="Second segment about architecture",
                start_time=5.0,
                end_time=10.0,
                speaker="SPEAKER_02"
            )
        ]
        
        metadata = {
            "filename": "test_audio.mp3",
            "duration_seconds": 10.0
        }
        
        num_added = vector_store.add_audio_segments(segments, metadata)
        assert num_added == 2
        
        # Verify collection count
        stats = vector_store.get_collection_stats()
        assert stats["total_documents"] >= 2
    
    def test_add_document_chunks(self, vector_store):
        """Test adding document chunks to vector store."""
        chunks = [
            DocumentChunk(
                text="First chunk about German reunification",
                page_number=1,
                chunk_index=0,
                source="test_doc.pdf",
                metadata={"document_name": "test_doc.pdf"}
            ),
            DocumentChunk(
                text="Second chunk about political changes",
                page_number=1,
                chunk_index=1,
                source="test_doc.pdf",
                metadata={"document_name": "test_doc.pdf"}
            )
        ]
        
        metadata = {
            "filename": "test_doc.pdf",
            "num_pages": 5
        }
        
        num_added = vector_store.add_document_chunks(chunks, metadata)
        assert num_added == 2
    
    def test_search_functionality(self, vector_store):
        """Test basic search functionality."""
        # Add some test data
        chunks = [
            DocumentChunk(
                text="Berlin Wall fell in 1989",
                page_number=1,
                chunk_index=0,
                source="history.pdf",
                metadata={"document_name": "history.pdf"}
            )
        ]
        
        vector_store.add_document_chunks(chunks, {"filename": "history.pdf"})
        
        # Search
        results = vector_store.search("When did the Berlin Wall fall?", n_results=1)
        
        assert results is not None
        assert len(results["ids"][0]) > 0
    
    def test_metadata_filtering(self, vector_store):
        """Test search with metadata filters."""
        # Add audio and text data
        audio_segments = [
            TranscriptSegment(
                text="Audio content about Berlin",
                start_time=0.0,
                end_time=5.0
            )
        ]
        vector_store.add_audio_segments(
            audio_segments,
            {"filename": "audio.mp3"}
        )
        
        text_chunks = [
            DocumentChunk(
                text="Text content about Berlin",
                page_number=1,
                chunk_index=0,
                source="doc.pdf",
                metadata={"document_name": "doc.pdf"}
            )
        ]
        vector_store.add_document_chunks(
            text_chunks,
            {"filename": "doc.pdf"}
        )
        
        # Filter by type
        audio_only = vector_store.search(
            "Berlin",
            n_results=10,
            filter_metadata={"type": "audio"}
        )
        
        assert all(
            meta["type"] == "audio"
            for meta in audio_only["metadatas"][0]
        )


class TestAttributionEngine:
    """Tests for attribution engine."""
    
    @pytest.fixture
    def engine_with_data(self):
        """Create attribution engine with test data."""
        temp_dir = tempfile.mkdtemp()
        vector_store = UnifiedVectorStore(
            collection_name="test_attribution",
            persist_directory=temp_dir
        )
        
        # Add test data
        chunks = [
            DocumentChunk(
                text="The Berlin Wall fell on November 9, 1989, marking the end of the Cold War era.",
                page_number=12,
                chunk_index=0,
                source="history.pdf",
                metadata={"document_name": "history.pdf", "document_title": "Berlin History"}
            ),
            DocumentChunk(
                text="The reunification of Germany occurred on October 3, 1990.",
                page_number=15,
                chunk_index=0,
                source="history.pdf",
                metadata={"document_name": "history.pdf", "document_title": "Berlin History"}
            )
        ]
        
        vector_store.add_document_chunks(
            chunks,
            {"filename": "history.pdf", "num_pages": 50}
        )
        
        engine = AttributionEngine(vector_store)
        
        yield engine
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_engine_initialization(self, engine_with_data):
        """Test attribution engine initializes correctly."""
        assert engine_with_data is not None
        assert engine_with_data.vector_store is not None
        assert engine_with_data.client is not None
    
    def test_query_returns_response(self, engine_with_data):
        """Test querying returns a proper response."""
        response = engine_with_data.query_archive(
            "When did the Berlin Wall fall?"
        )
        
        assert response is not None
        assert response.query == "When did the Berlin Wall fall?"
        assert response.answer is not None
        assert len(response.answer) > 0
    
    def test_citations_present(self, engine_with_data):
        """Test that citations are included in response."""
        response = engine_with_data.query_archive(
            "When did the Berlin Wall fall?"
        )
        
        # Should have retrieved chunks
        assert response.metadata["num_chunks_retrieved"] > 0


class TestEndToEndPipeline:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline fixture."""
        temp_dir = tempfile.mkdtemp()
        pipeline = BerlinArchivePipeline(collection_name="test_e2e")
        
        yield pipeline
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self, pipeline):
        """Test full pipeline initializes."""
        assert pipeline is not None
        assert pipeline.audio_pipeline is not None
        assert pipeline.document_pipeline is not None
        assert pipeline.vector_store is not None
        assert pipeline.attribution_engine is not None
    
    def test_get_stats(self, pipeline):
        """Test getting pipeline statistics."""
        stats = pipeline.get_stats()
        
        assert "total_documents" in stats
        assert "audio_segments" in stats
        assert "text_chunks" in stats
        assert isinstance(stats["total_documents"], int)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )