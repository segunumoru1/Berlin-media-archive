"""
ChromaDB Vector Store
Unified vector store for audio and document chunks with hybrid search support.
Uses OpenAI embeddings.
"""

import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime

import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try to import settings
try:
    from utils.config import settings
    USE_SETTINGS = True
except ImportError:
    USE_SETTINGS = False

from embeddings.openai_embeddings import OpenAIEmbeddings
from ingestion.audio_ingestion import TranscriptSegment
from ingestion.document_ingestion import DocumentChunk
from utils.config import settings


class UnifiedVectorStore:
    """Unified vector store for multi-modal content using ChromaDB."""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize unified vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_model: OpenAI embedding model to use
        """
        
        if USE_SETTINGS:
            self.collection_name = collection_name or settings.collection_name
            self.persist_directory = persist_directory or settings.vectorstore_path
        else:
            self.collection_name = collection_name or "default_collection"
            self.persist_directory = persist_directory or "./data/vectorstore"
        
        logger.info(f"Initializing UnifiedVectorStore: {self.collection_name}")
        logger.info(f"Persist directory: {self.persist_directory}")
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI embeddings
        try:
            if USE_SETTINGS:
                embedding_model = embedding_model or settings.embedding_model
            else:
                embedding_model = embedding_model or "text-embedding-3-large"
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise
        
        logger.info(f"Using embedding model: {self.embeddings.model}")
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
            )
        )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Berlin Media Archive - Multi-modal RAG",
                    "embedding_model": self.embeddings.model,
                    "created_at": datetime.now().isoformat()
                }
            )
            logger.info(f"Collection ready: {self.collection_name}")
            logger.info(f"Current collection size: {self.collection.count()} documents")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def add_audio_segments(
        self,
        segments: List[TranscriptSegment],
        audio_metadata: Dict,
        batch_size: int = 100
    ) -> int:
        """
        Add audio transcript segments to the vector store.
        
        Args:
            segments: List of TranscriptSegment objects
            audio_metadata: Audio file metadata
            batch_size: Batch size for adding documents
            
        Returns:
            Number of segments added
        """
        logger.info(f"Adding {len(segments)} audio segments to vector store")
        
        try:
            for i in range(0, len(segments), batch_size):
                batch = segments[i:i + batch_size]
                
                # Prepare data for batch
                ids = []
                documents = []
                metadatas = []
                embeddings_list = []
                
                for segment in batch:
                    # Generate unique ID
                    segment_id = str(uuid.uuid4())
                    
                    # Prepare document text
                    doc_text = segment.text
                    
                    # Prepare metadata
                    metadata = {
                        "type": "audio",
                        "source": audio_metadata.get("filename", "unknown"),
                        "filepath": audio_metadata.get("filepath", ""),
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "timestamp": f"{segment.start_time:.2f}-{segment.end_time:.2f}",
                        "duration": segment.end_time - segment.start_time,
                        "speaker": segment.speaker or "UNKNOWN",
                        "confidence": segment.confidence or 1.0,
                        "audio_duration": audio_metadata.get("duration_seconds", 0),
                        "added_at": datetime.now().isoformat()
                    }
                    
                    ids.append(segment_id)
                    documents.append(doc_text)
                    metadatas.append(metadata)
                
                # Generate embeddings for batch
                logger.debug(f"Generating embeddings for batch of {len(documents)} segments")
                batch_embeddings = self.embeddings.embed_texts(documents)
                embeddings_list.extend(batch_embeddings)
                
                # Add batch to collection
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings_list
                )
                
                logger.debug(f"Added batch {i//batch_size + 1}: {len(batch)} segments")
            
            logger.info(f"Successfully added {len(segments)} audio segments")
            return len(segments)
            
        except Exception as e:
            logger.error(f"Failed to add audio segments: {e}", exc_info=True)
            raise
    
    def add_document_chunks(
        self,
        chunks: List[DocumentChunk],
        document_metadata: Dict,
        batch_size: int = 100
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects
            document_metadata: Document metadata
            batch_size: Batch size for adding documents
            
        Returns:
            Number of chunks added
        """
        logger.info(f"Adding {len(chunks)} document chunks to vector store")
        
        try:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Prepare data for batch
                ids = []
                documents = []
                metadatas = []
                embeddings_list = []
                
                for chunk in batch:
                    # Use chunk's own ID or generate new one
                    chunk_id = chunk.chunk_id or str(uuid.uuid4())
                    
                    # Prepare document text
                    doc_text = chunk.text
                    
                    # Prepare metadata
                    metadata = {
                        "type": "text",
                        "source": document_metadata.get("filename", "unknown"),
                        "filepath": document_metadata.get("filepath", ""),
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "chunk_id": chunk_id,
                        "char_count": len(chunk.text),
                        "word_count": len(chunk.text.split()),
                        "total_pages": document_metadata.get("num_pages", 0),
                        "document_title": chunk.metadata.get("document_title", ""),
                        "document_author": chunk.metadata.get("document_author", ""),
                        "added_at": datetime.now().isoformat()
                    }
                    
                    ids.append(chunk_id)
                    documents.append(doc_text)
                    metadatas.append(metadata)
                
                # Generate embeddings for batch
                logger.debug(f"Generating embeddings for batch of {len(documents)} chunks")
                batch_embeddings = self.embeddings.embed_texts(documents)
                embeddings_list.extend(batch_embeddings)
                
                # Add batch to collection
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings_list
                )
                
                logger.debug(f"Added batch {i//batch_size + 1}: {len(batch)} chunks")
            
            logger.info(f"Successfully added {len(chunks)} document chunks")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to add document chunks: {e}", exc_info=True)
            raise
    
    def search(
        self,
        query: str,
        n_results: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
        include_embeddings: bool = False
    ):
        """
        Semantic search in the vector store.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            include_embeddings: Whether to include embeddings in results
            
        Returns:
            Dictionary with search results
        """
        n_results = n_results or settings.top_k_results
        
        logger.info(f"Searching for: '{query}' (top {n_results})")
        if filter_metadata:
            logger.info(f"Filters: {filter_metadata}")
        
        try:
            # Generate query embedding using OpenAI
            query_embedding = self.embeddings.embed_text(query)
            
            # Prepare include list
            include: List[Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]] = ["documents", "metadatas", "distances"]
            if include_embeddings:
                include.append("embeddings")
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata,
                include=include
            )
            
            logger.info(f"Found {len(results['ids'][0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise
    
    def hybrid_search(
        self,
        query: str,
        n_results: int,
        filter_metadata: Optional[Dict] = None,
        semantic_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic (vector) and keyword (BM25) search.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            semantic_weight: Weight for semantic search (0-1)
            
        Returns:
            List of ranked results with metadata
        """
        n_results = n_results or settings.top_k_results
        keyword_weight = 1 - semantic_weight
        
        logger.info(f"Hybrid search for: '{query}'")
        logger.info(f"Weights - Semantic: {semantic_weight}, Keyword: {keyword_weight}")
        
        try:
            # 1. Semantic Search
            semantic_results = self.search(
                query=query,
                n_results=n_results * 2,
                filter_metadata=filter_metadata
            )
            
            # 2. Keyword Search (BM25)
            keyword_results = self._keyword_search(
                query=query,
                n_results=n_results * 2,
                filter_metadata=filter_metadata
            )
            
            # 3. Combine and re-rank results
            combined_results = self._combine_results(
                semantic_results,
                keyword_results,
                semantic_weight,
                keyword_weight,
                n_results
            )
            
            logger.info(f"Hybrid search returned {len(combined_results)} results")
            return combined_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            logger.warning("Falling back to semantic search only")
            semantic_results = self.search(query, n_results, filter_metadata)
            return self._format_search_results(semantic_results)
    
    def _keyword_search(
        self,
        query: str,
        n_results: int,
        filter_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Perform keyword-based search using BM25."""
        try:
            from rank_bm25 import BM25Okapi
            import numpy as np
            
            # Get all documents
            all_docs = self.collection.get(
                where=filter_metadata,
                include=["documents", "metadatas"]
            )
            
            if not all_docs["documents"]:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            # Tokenize
            tokenized_docs = [doc.lower().split() for doc in all_docs["documents"]]
            bm25 = BM25Okapi(tokenized_docs)
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)
            
            # Get top results
            top_indices = np.argsort(scores)[::-1][:n_results]
            
            # Handle case where metadatas might be None
            metadatas = all_docs.get("metadatas") or [{}] * len(all_docs["ids"])
            
            return {
                "ids": [[all_docs["ids"][i] for i in top_indices]],
                "documents": [[all_docs["documents"][i] for i in top_indices]],
                "metadatas": [[metadatas[i] for i in top_indices]],
                "distances": [[1.0 / (1.0 + scores[i]) for i in top_indices]]
            }
            
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def _combine_results(
        self,
        semantic_results: Any,
        keyword_results: Any,
        semantic_weight: float,
        keyword_weight: float,
        n_results: int
    ) -> List[Dict[str, Any]]:
        """Combine and re-rank results."""
        combined_scores = {}
        
        # Process semantic results
        for i, doc_id in enumerate(semantic_results["ids"][0]):
            distance = semantic_results["distances"][0][i]
            similarity = 1 / (1 + distance)
            combined_scores[doc_id] = {
                "semantic_score": similarity * semantic_weight,
                "keyword_score": 0,
                "document": semantic_results["documents"][0][i],
                "metadata": semantic_results["metadatas"][0][i]
            }
        
        # Add keyword results
        for i, doc_id in enumerate(keyword_results["ids"][0]):
            distance = keyword_results["distances"][0][i]
            similarity = 1 / (1 + distance)
            
            if doc_id in combined_scores:
                combined_scores[doc_id]["keyword_score"] = similarity * keyword_weight
            else:
                combined_scores[doc_id] = {
                    "semantic_score": 0,
                    "keyword_score": similarity * keyword_weight,
                    "document": keyword_results["documents"][0][i],
                    "metadata": keyword_results["metadatas"][0][i]
                }
        
        # Calculate final scores and sort
        results = []
        for doc_id, scores in combined_scores.items():
            final_score = scores["semantic_score"] + scores["keyword_score"]
            results.append({
                "id": doc_id,
                "document": scores["document"],
                "metadata": scores["metadata"],
                "score": final_score,
                "semantic_score": scores["semantic_score"],
                "keyword_score": scores["keyword_score"]
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:n_results]
    
    def _format_search_results(self, results: Any) -> List[Dict[str, Any]]:
        """Format search results."""
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 / (1 + results["distances"][0][i])
            })
        return formatted
    
    def get_by_metadata(
        self,
        filter_metadata: Dict,
        limit: Optional[int] = None
    ):
        """Get documents by metadata filter."""
        logger.info(f"Getting documents with filter: {filter_metadata}")
        try:
            results = self.collection.get(
                where=filter_metadata,
                limit=limit,
                include=["documents", "metadatas"]
            )
            logger.info(f"Found {len(results['ids'])} documents")
            return results
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            raise
    
    def delete_by_source(self, source_filename: str) -> int:
        """Delete all documents from a source."""
        logger.info(f"Deleting documents from: {source_filename}")
        try:
            docs = self.collection.get(where={"source": source_filename})
            if docs["ids"]:
                self.collection.delete(ids=docs["ids"])
                logger.info(f"Deleted {len(docs['ids'])} documents")
                return len(docs["ids"])
            return 0
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            total_count = self.collection.count()
            audio_docs = self.collection.get(where={"type": "audio"}, include=[])
            text_docs = self.collection.get(where={"type": "text"}, include=[])
            
            return {
                "total_documents": total_count,
                "audio_segments": len(audio_docs["ids"]),
                "text_chunks": len(text_docs["ids"]),
                "collection_name": self.collection_name,
                "embedding_model": self.embeddings.model
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise
    
    def reset_collection(self):
        """Reset the collection."""
        logger.warning(f"Resetting collection: {self.collection_name}")
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Berlin Media Archive - Multi-modal RAG",
                    "embedding_model": self.embeddings.model,
                    "created_at": datetime.now().isoformat()
                }
            )
            logger.info("Collection reset complete")
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            raise