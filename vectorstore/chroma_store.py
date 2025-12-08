"""
Unified Vector Store with ChromaDB
Handles both audio and document embeddings with hybrid search support.
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

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
        collection_name: Optional[str] = None,
        reset_collection: bool = False
    ):
        """
        Initialize the unified vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
            reset_collection: If True, delete and recreate the collection
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
        self._bm25_metadatas = None
        
        # Create persist directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI embedding function
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            )
            logger.info(f"Using OpenAI embeddings: {os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')}")
        else:
            # Fallback to default
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            logger.warning("No OpenAI API key found, using default embeddings")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Handle collection creation with embedding function
        self._initialize_collection(reset_collection)
        
        logger.info(f"UnifiedVectorStore initialized: {self.collection_name} at {self.persist_directory}")
        logger.info(f"Collection has {self.collection.count()} documents")
        logger.info(f"Hybrid search: {'enabled' if self.enable_hybrid and BM25_AVAILABLE else 'disabled'}")
    
    def _initialize_collection(self, reset_collection: bool = False):
        """
        Initialize or reset the collection with proper embedding function.
        """
        try:
            # Check if collection exists
            existing_collections = [c.name for c in self.client.list_collections()]
            collection_exists = self.collection_name in existing_collections
            
            if reset_collection and collection_exists:
                logger.info(f"Resetting collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                collection_exists = False
            
            if collection_exists:
                # Try to get existing collection
                try:
                    # First try without embedding function to check compatibility
                    self.collection = self.client.get_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function
                    )
                except ValueError as e:
                    if "embedding function" in str(e).lower():
                        # Embedding function conflict - need to recreate
                        logger.warning(f"Embedding function conflict detected. Recreating collection...")
                        logger.warning("⚠️  Existing documents will be lost. Please re-ingest after restart.")
                        
                        self.client.delete_collection(self.collection_name)
                        self.collection = self.client.create_collection(
                            name=self.collection_name,
                            embedding_function=self.embedding_function,
                            metadata={"hnsw:space": "cosine"}
                        )
                    else:
                        raise
            else:
                # Create new collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def _build_where_clause(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict]:
        """
        Build ChromaDB where clause from filters.
        """
        if not filters:
            return None
        
        # Remove None values
        filters = {k: v for k, v in filters.items() if v is not None}
        
        if not filters:
            return None
        
        # Single filter
        if len(filters) == 1:
            key, value = list(filters.items())[0]
            return {key: {"$eq": value}}
        
        # Multiple filters - use $and operator
        conditions = [{k: {"$eq": v}} for k, v in filters.items()]
        return {"$and": conditions}
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add texts to the vector store.
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
                    cleaned[k] = str(v)
                elif v is not None:
                    cleaned[k] = str(v)
            cleaned_metadatas.append(cleaned)
        
        try:
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
        """
        texts = []
        metadatas = []
        ids = []
        
        for doc in documents:
            content = doc.get("content", doc.get("text", ""))
            if not content:
                continue
            
            texts.append(content)
            
            meta = {
                "source_type": source_type,
                "chunk_id": doc.get("chunk_id", str(uuid.uuid4())),
            }
            
            for field in ["page_number", "source_file", "source", "timestamp_start", 
                         "timestamp_end", "speaker", "title", "author", "timestamp"]:
                if field in doc and doc[field] is not None:
                    meta[field] = doc[field]
            
            metadatas.append(meta)
            ids.append(doc.get("id", str(uuid.uuid4())))
        
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents in collection."""
        if not BM25_AVAILABLE:
            return
        
        try:
            all_docs = self.collection.get()
            
            if not all_docs or not all_docs['documents']:
                logger.warning("No documents to build BM25 index")
                return
            
            self._bm25_documents = all_docs['documents']
            self._bm25_ids = all_docs['ids']
            self._bm25_metadatas = all_docs['metadatas']
            
            tokenized_docs = [doc.lower().split() for doc in self._bm25_documents]
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
        """
        try:
            doc_count = self.collection.count()
            logger.info(f"Searching in collection with {doc_count} documents")
            
            if doc_count == 0:
                logger.warning("Collection is empty")
                return []
            
            where = self._build_where_clause(filters)
            
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, doc_count),
                where=where
            )
            
            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        "content": doc,
                        "document": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "id": results['ids'][0][i] if results['ids'] else None,
                        "score": 1 - results['distances'][0][i] if results['distances'] else 0.0
                    }
                    formatted_results.append(result)
            
            logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
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
        """
        bm25_weight = bm25_weight if bm25_weight is not None else self.bm25_weight
        
        if not self.enable_hybrid or not BM25_AVAILABLE:
            return self.search(query, top_k, filters)
        
        try:
            if self._bm25_index is None:
                self._build_bm25_index()
            
            if self._bm25_index is None:
                return self.search(query, top_k, filters)
            
            doc_count = self.collection.count()
            semantic_k = min(top_k * 3, doc_count) if doc_count > 0 else top_k
            semantic_results = self.search(query, semantic_k, filters)
            
            if not semantic_results:
                return []
            
            tokenized_query = query.lower().split()
            bm25_scores = self._bm25_index.get_scores(tokenized_query)
            
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            normalized_bm25 = {
                self._bm25_ids[i]: score / max_bm25 
                for i, score in enumerate(bm25_scores)
            }
            
            combined_results = []
            for result in semantic_results:
                doc_id = result.get('id')
                semantic_score = result.get('score', 0)
                bm25_score = normalized_bm25.get(doc_id, 0)
                
                combined_score = (1 - bm25_weight) * semantic_score + bm25_weight * bm25_score
                
                combined_results.append({
                    "content": result['content'],
                    "document": result['content'],
                    "metadata": result['metadata'],
                    "id": doc_id,
                    "score": combined_score,
                    "semantic_score": semantic_score,
                    "bm25_score": bm25_score
                })
            
            combined_results.sort(key=lambda x: x['score'], reverse=True)
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self.search(query, top_k, filters)
    
    def similarity_search(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.search(query=query, top_k=k, filters=filter)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "persist_directory": self.persist_directory,
            "hybrid_search_enabled": self.enable_hybrid and BM25_AVAILABLE,
            "bm25_weight": self.bm25_weight
        }
    
    def delete_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            self._bm25_index = None
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def clear(self):
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            self._bm25_index = None
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        try:
            all_docs = self.collection.get()
            
            if not all_docs or not all_docs['documents']:
                return []
            
            return [
                {
                    "content": doc,
                    "document": doc,
                    "metadata": all_docs['metadatas'][i] if all_docs['metadatas'] else {},
                    "id": all_docs['ids'][i] if all_docs['ids'] else None
                }
                for i, doc in enumerate(all_docs['documents'])
            ]
            
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []