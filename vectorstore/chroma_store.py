"""
Unified Vector Store with ChromaDB
Handles both audio and document embeddings with hybrid search support.
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import chromadb
from chromadb.config import Settings as ChromaSettings

# Try to import BM25 for hybrid search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not installed. Hybrid search will use semantic only.")


class UnifiedVectorStore:
    """
    Unified vector store for audio transcripts and documents.
    Supports hybrid search combining dense embeddings with BM25.
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize the unified vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory or os.getenv("VECTORSTORE_PATH", "./data/vectorstore")
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "berlin_archive")
        
        # Hybrid search settings
        self.enable_hybrid = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
        self.bm25_weight = float(os.getenv("BM25_WEIGHT", "0.3"))
        
        # BM25 index (built lazily)
        self._bm25_index = None
        self._bm25_documents = None
        self._bm25_ids = None
        
        # Create persist directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"UnifiedVectorStore initialized: {self.collection_name} at {self.persist_directory}")
        logger.info(f"Collection has {self.collection.count()} documents")
        logger.info(f"Hybrid search: {'enabled' if self.enable_hybrid and BM25_AVAILABLE else 'disabled'}")
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dicts for each text
            ids: Optional list of IDs (generated if not provided)
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of IDs for added documents
        """
        if not texts:
            logger.warning("No texts provided to add_texts")
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Ensure metadatas list matches texts
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Clean metadata - ChromaDB only accepts str, int, float, bool
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned[k] = v
                elif isinstance(v, list):
                    cleaned[k] = str(v)  # Convert lists to string
                elif v is not None:
                    cleaned[k] = str(v)  # Convert other types to string
            cleaned_metadatas.append(cleaned)
        
        try:
            if embeddings:
                self.collection.add(
                    documents=texts,
                    metadatas=cleaned_metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            else:
                # Let ChromaDB compute embeddings
                self.collection.add(
                    documents=texts,
                    metadatas=cleaned_metadatas,
                    ids=ids
                )
            
            # Invalidate BM25 index
            self._bm25_index = None
            
            logger.info(f"Added {len(texts)} texts to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        source_type: str = "document"
    ) -> List[str]:
        """
        Add documents with metadata to vector store.
        
        Args:
            documents: List of dicts with 'content' and optional metadata
            source_type: Type of source (audio/document)
            
        Returns:
            List of document IDs
        """
        texts = []
        metadatas = []
        ids = []
        
        for doc in documents:
            content = doc.get("content", doc.get("text", ""))
            if not content:
                continue
            
            texts.append(content)
            
            # Build metadata
            meta = {
                "source_type": source_type,
                "chunk_id": doc.get("chunk_id", str(uuid.uuid4())),
            }
            
            # Add optional metadata fields
            for field in ["page_number", "source_file", "timestamp_start", 
                         "timestamp_end", "speaker", "title", "author"]:
                if field in doc:
                    meta[field] = doc[field]
            
            metadatas.append(meta)
            ids.append(doc.get("id", str(uuid.uuid4())))
        
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents in collection."""
        if not BM25_AVAILABLE:
            return
        
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            if not all_docs or not all_docs['documents']:
                logger.warning("No documents to build BM25 index")
                return
            
            self._bm25_documents = all_docs['documents']
            self._bm25_ids = all_docs['ids']
            self._bm25_metadatas = all_docs['metadatas']
            
            # Tokenize documents
            tokenized_docs = [doc.lower().split() for doc in self._bm25_documents]
            
            # Build BM25 index
            self._bm25_index = BM25Okapi(tokenized_docs)
            
            logger.info(f"Built BM25 index with {len(self._bm25_documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self._bm25_index = None
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of results with content, metadata, and scores
        """
        try:
            # Build where clause for filters
            where = None
            if filters:
                where = {}
                for k, v in filters.items():
                    where[k] = v
            
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where
            )
            
            # Format results
            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "id": results['ids'][0][i] if results['ids'] else None,
                        "score": 1 - results['distances'][0][i] if results['distances'] else 0.0
                    }
                    formatted_results.append(result)
            
            logger.info(f"Search returned {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        bm25_weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic search with BM25.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional metadata filters
            bm25_weight: Weight for BM25 scores (0-1). Semantic weight = 1 - bm25_weight
            
        Returns:
            List of results with content, metadata, and combined scores
        """
        bm25_weight = bm25_weight if bm25_weight is not None else self.bm25_weight
        
        # If hybrid search disabled or BM25 not available, fall back to semantic
        if not self.enable_hybrid or not BM25_AVAILABLE:
            logger.debug("Using semantic-only search")
            return self.search(query, top_k, filters)
        
        try:
            # Build BM25 index if needed
            if self._bm25_index is None:
                self._build_bm25_index()
            
            # If BM25 index still not available, fall back to semantic
            if self._bm25_index is None:
                return self.search(query, top_k, filters)
            
            # Get semantic search results (get more to allow for re-ranking)
            semantic_k = min(top_k * 3, self.collection.count())
            semantic_results = self.search(query, semantic_k, filters)
            
            # Get BM25 scores
            tokenized_query = query.lower().split()
            bm25_scores = self._bm25_index.get_scores(tokenized_query)
            
            # Normalize BM25 scores
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            normalized_bm25 = {
                self._bm25_ids[i]: score / max_bm25 
                for i, score in enumerate(bm25_scores)
            }
            
            # Combine scores
            combined_results = []
            for result in semantic_results:
                doc_id = result.get('id')
                semantic_score = result.get('score', 0)
                bm25_score = normalized_bm25.get(doc_id, 0)
                
                # Combined score: weighted average
                combined_score = (1 - bm25_weight) * semantic_score + bm25_weight * bm25_score
                
                combined_results.append({
                    "content": result['content'],
                    "metadata": result['metadata'],
                    "id": doc_id,
                    "score": combined_score,
                    "semantic_score": semantic_score,
                    "bm25_score": bm25_score
                })
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top_k results
            final_results = combined_results[:top_k]
            
            logger.info(f"Hybrid search returned {len(final_results)} results for query: {query[:50]}...")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed, falling back to semantic: {e}")
            return self.search(query, top_k, filters)
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Alias for search() to match LangChain interface.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter: Optional metadata filters
            
        Returns:
            List of results
        """
        return self.search(query=query, top_k=k, filters=filter)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "persist_directory": self.persist_directory,
            "hybrid_search_enabled": self.enable_hybrid and BM25_AVAILABLE,
            "bm25_weight": self.bm25_weight
        }
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._bm25_index = None
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def clear(self):
        """Clear all documents from the collection."""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._bm25_index = None
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from the collection."""
        try:
            all_docs = self.collection.get()
            
            if not all_docs or not all_docs['documents']:
                return []
            
            documents = []
            for i, doc in enumerate(all_docs['documents']):
                documents.append({
                    "content": doc,
                    "metadata": all_docs['metadatas'][i] if all_docs['metadatas'] else {},
                    "id": all_docs['ids'][i] if all_docs['ids'] else None
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []