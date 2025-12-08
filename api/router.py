"""
API Router for Berlin Media Archive
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from loguru import logger
import os
from pathlib import Path
from datetime import datetime

# Create router WITHOUT prefix (prefix is added in main.py)
router = APIRouter()


# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    filter_type: Optional[str] = None
    filter_speaker: Optional[str] = None
    filter_source: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# Global instances
_pipeline = None
_vector_store = None


def get_vector_store():
    """Get or create vector store instance."""
    global _vector_store
    if _vector_store is None:
        from vectorstore.chroma_store import UnifiedVectorStore
        _vector_store = UnifiedVectorStore()
    return _vector_store


def get_pipeline():
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        from pipeline.orchestrator import BerlinArchivePipeline
        _pipeline = BerlinArchivePipeline()
    return _pipeline


@router.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting up API server...")
    try:
        # Initialize vector store
        vs = get_vector_store()
        logger.info(f"Vector store initialized with {vs.collection.count()} documents")
        
        # Initialize pipeline
        get_pipeline()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")


# ============= Query Endpoints =============

@router.post("/query", response_model=QueryResponse)
async def query_archive(request: QueryRequest):
    """
    Query the archive with natural language.
    """
    try:
        logger.info(f"Query: {request.query}")
        
        # Get vector store and check document count
        vector_store = get_vector_store()
        doc_count = vector_store.collection.count()
        logger.info(f"Vector store has {doc_count} documents")
        
        if doc_count == 0:
            return QueryResponse(
                query=request.query,
                answer="The archive is empty. Please ingest some documents or audio files first using the /ingest/document or /ingest/audio endpoints.",
                citations=[],
                metadata={"num_chunks_retrieved": 0, "num_citations": 0, "error": "empty_archive"}
            )
        
        # Build filters (only include non-None values)
        filters = {}
        if request.filter_type:
            filters["source_type"] = request.filter_type
        if request.filter_speaker:
            filters["speaker"] = request.filter_speaker
        if request.filter_source:
            filters["source_file"] = request.filter_source
        
        # Use None if no filters
        filter_metadata = filters if filters else None
        
        # Search for relevant chunks
        results = vector_store.hybrid_search(
            query=request.query,
            top_k=request.top_k,
            filters=filter_metadata
        )
        
        logger.info(f"Retrieved {len(results)} chunks")
        
        if not results:
            return QueryResponse(
                query=request.query,
                answer="I could not find any relevant information in the archive for your question.",
                citations=[],
                metadata={"num_chunks_retrieved": 0, "num_citations": 0}
            )
        
        # Build context from results
        context_parts = []
        for i, result in enumerate(results):
            metadata = result.get('metadata', {})
            source_info = metadata.get('source_file', 'Unknown source')
            page_info = f"Page {metadata.get('page_number', 'N/A')}" if metadata.get('page_number') else ""
            
            context_parts.append(f"[Source {i+1}: {source_info} {page_info}]\n{result['content']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer using LLM
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        system_prompt = """You are a helpful research assistant analyzing archived documents and audio transcripts.

Your task:
1. Answer the user's question based ONLY on the provided context
2. Include specific citations in your answer using [Source X] format
3. If the context doesn't contain enough information, say so
4. Be accurate and cite specific details with their sources

IMPORTANT: Only use information from the provided context. Do not make up information."""

        user_prompt = f"""Context from the archive:

{context}

---

Question: {request.query}

Please provide a comprehensive answer with citations to the sources."""

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4-turbo-preview"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        answer = response.choices[0].message.content
        
        # Build citations
        citations = []
        for i, result in enumerate(results):
            metadata = result.get('metadata', {})
            citations.append({
                "source_id": i + 1,
                "source_file": metadata.get('source_file', 'Unknown'),
                "source_type": metadata.get('source_type', 'document'),
                "page_number": metadata.get('page_number'),
                "content_preview": result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                "relevance_score": result.get('score', 0)
            })
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            metadata={
                "num_chunks_retrieved": len(results),
                "num_citations": len(citations),
                "llm_model": os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
            }
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============= Ingestion Endpoints =============

@router.post("/ingest/document")
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a document file (PDF, DOCX, TXT).
    """
    try:
        logger.info(f"Ingesting document: {file.filename}")
        
        # Validate file type
        allowed_extensions = [".pdf", ".docx", ".doc", ".txt"]
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
            )
        
        # Save file
        documents_dir = Path(os.getenv("DOCUMENTS_DIR", "./data/documents"))
        documents_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = documents_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Saved file to: {file_path}")
        
        # Process document
        from ingestion.document_ingestion import DocumentIngestionPipeline
        
        doc_pipeline = DocumentIngestionPipeline()
        output_dir = os.getenv("OUTPUT_DIR", "./output") + "/documents"
        
        chunks, metadata = doc_pipeline.ingest_document(str(file_path), output_dir)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        
        # Add to vector store
        items_added = 0
        if chunks:
            try:
                vector_store = get_vector_store()
                
                # Log before adding
                before_count = vector_store.collection.count()
                logger.info(f"Vector store document count before: {before_count}")
                
                # Prepare documents for vector store
                documents = []
                for chunk in chunks:
                    doc = {
                        "id": f"{file.filename}_{chunk.chunk_id}",
                        "content": chunk.content,
                        "page_number": chunk.page_number,
                        "source_file": chunk.source_file,
                        "title": metadata.get("title", ""),
                        "author": metadata.get("author", ""),
                        "chunk_id": chunk.chunk_id,
                        "source_type": "document"
                    }
                    documents.append(doc)
                
                # Add to vector store
                ids = vector_store.add_documents(documents, source_type="document")
                items_added = len(ids)
                
                # Log after adding
                after_count = vector_store.collection.count()
                logger.info(f"Vector store document count after: {after_count}")
                logger.info(f"Successfully added {items_added} chunks to vector store")
                
            except Exception as e:
                logger.error(f"Failed to add to vector store: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to add to vector store: {e}")
        
        return {
            "success": True,
            "message": "Document ingested successfully",
            "filename": file.filename,
            "items_added": items_added,
            "metadata": {
                "num_pages": metadata.get("num_pages", 0),
                "num_chunks": metadata.get("num_chunks", 0),
                "title": metadata.get("title", ""),
                "author": metadata.get("author", "")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/audio")
async def ingest_audio(
    file: UploadFile = File(...),
    enable_diarization: bool = Query(default=False)
):
    """
    Ingest an audio file.
    """
    try:
        logger.info(f"Ingesting audio: {file.filename}")
        
        # Save file
        audio_dir = Path(os.getenv("AUDIO_DIR", "./data/audio"))
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = audio_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Saved audio to: {file_path}")
        
        # Process audio
        from ingestion.audio_ingestion import AudioIngestionPipeline
        
        audio_pipeline = AudioIngestionPipeline(enable_diarization=enable_diarization)
        output_dir = os.getenv("OUTPUT_DIR", "./output") + "/audio"
        
        segments = audio_pipeline.ingest_audio(str(file_path), output_dir)
        
        logger.info(f"Created {len(segments)} segments from audio")
        
        # Add to vector store
        items_added = 0
        if segments:
            try:
                vector_store = get_vector_store()
                
                documents = []
                for i, segment in enumerate(segments):
                    doc = {
                        "id": f"{file.filename}_segment_{i}",
                        "content": segment.text,
                        "timestamp_start": segment.start_time,
                        "timestamp_end": segment.end_time,
                        "speaker": segment.speaker or "Unknown",
                        "source_file": file.filename,
                        "source_type": "audio"
                    }
                    documents.append(doc)
                
                ids = vector_store.add_documents(documents, source_type="audio")
                items_added = len(ids)
                
                logger.info(f"Added {items_added} audio segments to vector store")
                
            except Exception as e:
                logger.error(f"Failed to add audio to vector store: {e}", exc_info=True)
        
        return {
            "success": True,
            "message": "Audio ingested successfully",
            "filename": file.filename,
            "items_added": items_added,
            "metadata": {
                "num_segments": len(segments),
                "total_duration": segments[-1].end_time if segments else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Audio ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============= Status Endpoints =============

@router.get("/status")
async def get_status():
    """Get system status."""
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        
        return {
            "status": "operational",
            "vector_store": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/documents")
async def list_documents():
    """List all documents in the vector store."""
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        
        # Get unique source files
        all_docs = vector_store.get_all_documents()
        source_files = set()
        for doc in all_docs:
            source = doc.get('metadata', {}).get('source_file')
            if source:
                source_files.add(source)
        
        return {
            "total_chunks": stats["total_documents"],
            "collection_name": stats["collection_name"],
            "source_files": list(source_files),
            "num_source_files": len(source_files)
        }
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector store."""
    try:
        vector_store = get_vector_store()
        before_count = vector_store.collection.count()
        
        vector_store.clear()
        
        return {
            "success": True,
            "message": f"Cleared {before_count} documents from vector store"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))