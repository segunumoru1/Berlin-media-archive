"""
Unit Tests for Retrieval Logic
Testing search and ranking functionality.
"""

import pytest
import numpy as np
from vectorstore.chroma_store import UnifiedVectorStore
from ingestion.document_ingestion import DocumentChunk
import tempfile
import shutil


class TestRetrievalLogic:
    """Tests for search and retrieval functionality."""
    
    @pytest.fixture
    def vector_store_with_content(self):
        """Create vector store with diverse test content."""
        temp_dir = tempfile.mkdtemp()
        store = UnifiedVectorStore(
            collection_name="test_retrieval",
            persist_directory=temp_dir
        )
        
        # Add diverse content
        test_chunks = [
            DocumentChunk(
                text="The Berlin Wall was built in 1961 and fell in 1989.",
                page_number=1,
                chunk_index=0,
                source="berlin_history.pdf",
                metadata={"topic": "wall", "year": "1989"}
            ),
            DocumentChunk(
                text="Brandenburg Gate is a famous monument in Berlin, built in 1791.",
                page_number=2,
                chunk_index=0,
                source="berlin_monuments.pdf",
                metadata={"topic": "architecture", "year": "1791"}
            ),
            DocumentChunk(
                text="Checkpoint Charlie was a crossing point between East and West Berlin during the Cold War.",
                page_number=3,
                chunk_index=0,
                source="berlin_history.pdf",
                metadata={"topic": "wall", "era": "cold_war"}
            ),
            DocumentChunk(
                text="The Reichstag building houses the German Parliament and was completed in 1894.",
                page_number=4,
                chunk_index=0,
                source="german_politics.pdf",
                metadata={"topic": "government", "year": "1894"}
            )
        ]
        
        store.add_document_chunks(
            test_chunks,
            {"filename": "test_docs.pdf", "num_pages": 10}
        )
        
        yield store
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_semantic_search_relevance(self, vector_store_with_content):
        """Test that semantic search returns relevant results."""
        results = vector_store_with_content.search(
            "What happened to the Berlin Wall?",
            n_results=2
        )
        
        # First result should mention the wall
        assert len(results["documents"][0]) > 0
        first_doc = results["documents"][0][0].lower()
        assert "wall" in first_doc or "berlin" in first_doc
    
    def test_top_k_results(self, vector_store_with_content):
        """Test that correct number of results are returned."""
        for k in [1, 2, 3]:
            results = vector_store_with_content.search(
                "Berlin",
                n_results=k
            )
            assert len(results["ids"][0]) == min(k, 4)  # 4 docs in test data
    
    def test_metadata_filter_effectiveness(self, vector_store_with_content):
        """Test metadata filtering works correctly."""
        # Filter by topic
        results = vector_store_with_content.get_by_metadata(
            filter_metadata={"topic": "wall"}
        )
        
        assert len(results["ids"]) == 2  # Two docs about the wall
        
        # Verify all results match filter
        for metadata in results["metadatas"]:
            assert metadata["topic"] == "wall"
    
    def test_hybrid_search_combines_methods(self, vector_store_with_content):
        """Test that hybrid search combines semantic and keyword search."""
        # Query with specific year (should benefit from keyword search)
        results = vector_store_with_content.hybrid_search(
            "1989",
            n_results=2
        )
        
        assert len(results) > 0
        # Should find the doc about 1989
        texts = [r["document"] for r in results]
        assert any("1989" in text for text in texts)
    
    def test_distance_scores_are_valid(self, vector_store_with_content):
        """Test that distance scores are within expected range."""
        results = vector_store_with_content.search(
            "Berlin Wall",
            n_results=3
        )
        
        distances = results["distances"][0]
        
        # All distances should be non-negative
        assert all(d >= 0 for d in distances)
        
        # Distances should be sorted (most similar first)
        assert distances == sorted(distances)
    
    def test_empty_query_handling(self, vector_store_with_content):
        """Test handling of empty or very short queries."""
        results = vector_store_with_content.search(
            "",
            n_results=1
        )
        
        # Should not crash, may return results or empty
        assert results is not None
        assert "ids" in results
    
    def test_no_results_scenario(self, vector_store_with_content):
        """Test behavior when no results match filters."""
        results = vector_store_with_content.get_by_metadata(
            filter_metadata={"topic": "nonexistent_topic"}
        )
        
        assert len(results["ids"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])