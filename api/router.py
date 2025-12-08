"""
API Router
FastAPI endpoints for the Berlin Media Archive.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel, Field
from pathlib import Path
import shutil
from datetime import datetime

from loguru import logger

from pipeline.orchestrator import BerlinArchivePipeline
from utils.config import settings
from rag_evaluator.llm_rag_evaluator import LLMRAGEvaluator

# Initialize router
router = APIRouter(prefix="/api/v1", tags=["Berlin Archive"])

# Global pipeline instance
pipeline: Optional[BerlinArchivePipeline] = None


# Pydantic Models
class QueryRequest(BaseModel):
    """Request model for archive queries."""
    question: str = Field(..., description="The question to ask the archive")
    n_results: Optional[int] = Field(5, description="Number of results to retrieve")
    filter_type: Optional[str] = Field(None, description="Filter by type: 'audio' or 'text'")
    filter_speaker: Optional[str] = Field(None, description="Filter by speaker (audio only)")
    filter_source: Optional[str] = Field(None, description="Filter by source filename")
    save_response: bool = Field(False, description="Save response to disk")


class QueryResponse(BaseModel):
    """Response model for archive queries."""
    query: str
    answer: str
    citations: List[dict]
    num_citations: int
    num_chunks_retrieved: int
    timestamp: str

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    context: List[str]
    ground_truth: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: dict


class StatsResponse(BaseModel):
    """Statistics response."""
    total_documents: int
    audio_segments: int
    text_chunks: int
    collection_name: str
    embedding_model: str


class IngestResponse(BaseModel):
    """Ingestion response."""
    success: bool
    message: str
    filename: str
    items_added: int
    metadata: dict


# Startup/Shutdown Events
@router.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    logger.info("Starting up API server...")
    try:
        pipeline = BerlinArchivePipeline()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


# Health Check
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns the status of the API and its components.
    """
    try:
        stats = pipeline.get_stats() if pipeline else {}
        
        return HealthResponse(
            status="healthy" if pipeline else "unhealthy",
            version="1.0.0",
            components={
                "pipeline": "initialized" if pipeline else "not_initialized",
                "vector_store": "connected" if pipeline else "disconnected",
                "documents_in_store": stats.get("total_documents", 0)
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# Query Endpoint
@router.post("/query", response_model=QueryResponse)
async def query_archive(request: QueryRequest):
    """
    Query the Berlin Media Archive.
    
    This endpoint allows you to ask questions about the archived content.
    The system will retrieve relevant context and provide an answer with citations.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.info(f"Received query: {request.question}")
        
        # Build filter metadata
        filter_metadata = {}
        if request.filter_type:
            filter_metadata["type"] = request.filter_type
        if request.filter_speaker:
            filter_metadata["speaker"] = request.filter_speaker
        if request.filter_source:
            filter_metadata["source"] = request.filter_source
        
        # Query the archive
        response = pipeline.query(
            question=request.question,
            n_results=request.n_results,
            filter_metadata=filter_metadata if filter_metadata else None,
            save_response=request.save_response
        )
        
        return QueryResponse(
            query=response.query,
            answer=response.answer,
            citations=[c.to_dict() for c in response.citations],
            num_citations=len(response.citations),
            num_chunks_retrieved=response.metadata["num_chunks_retrieved"],
            timestamp=response.timestamp
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# Audio Upload Endpoint
@router.post("/ingest/audio", response_model=IngestResponse)
async def ingest_audio(
    file: UploadFile = File(..., description="Audio file to ingest"),
    save_transcription: bool = Form(True, description="Save transcription to disk")
):
    """
    Ingest an audio file into the archive.
    
    Supports MP3 and WAV formats. The audio will be transcribed with timestamps
    and speaker diarization (if enabled).
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp3', '.wav')):
            raise HTTPException(status_code=400, detail="Only MP3 and WAV files are supported")
        
        # Save uploaded file
        audio_path = Path(settings.audio_dir) / file.filename
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Uploaded audio file: {file.filename}")
        
        # Process audio
        segments, metadata = pipeline.ingest_audio_file(
            str(audio_path),
            save_transcription=save_transcription
        )
        
        return IngestResponse(
            success=True,
            message="Audio file ingested successfully",
            filename=file.filename,
            items_added=len(segments),
            metadata={
                "duration_seconds": metadata.get("duration_seconds", 0),
                "num_segments": len(segments),
                "has_diarization": any(s.speaker for s in segments)
            }
        )
        
    except Exception as e:
        logger.error(f"Audio ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio ingestion failed: {str(e)}")


# Document Upload Endpoint
@router.post("/ingest/document", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="PDF document to ingest"),
    save_chunks: bool = Form(True, description="Save chunks to disk")
):
    """
    Ingest a PDF document into the archive.
    
    The document will be chunked intelligently with page number tracking.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        doc_path = Path(settings.documents_dir) / file.filename
        with open(doc_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Uploaded document: {file.filename}")
        
        # Process document
        chunks, metadata = pipeline.ingest_document_file(
            str(doc_path),
            save_chunks=save_chunks
        )
        
        return IngestResponse(
            success=True,
            message="Document ingested successfully",
            filename=file.filename,
            items_added=len(chunks),
            metadata={
                "num_pages": metadata.get("num_pages", 0),
                "num_chunks": len(chunks),
                "title": metadata.get("title", ""),
                "author": metadata.get("author", "")
            }
        )
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")


# Batch Ingest Endpoint
@router.post("/ingest/batch")
async def batch_ingest(
    audio_dir: Optional[str] = Query(None, description="Directory containing audio files"),
    documents_dir: Optional[str] = Query(None, description="Directory containing PDF documents")
):
    """
    Batch ingest multiple files from directories.
    
    Processes all audio and document files from the specified directories.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        if not audio_dir and not documents_dir:
            raise HTTPException(
                status_code=400,
                detail="At least one of audio_dir or documents_dir must be provided"
            )
        
        stats = pipeline.batch_ingest(
            audio_dir=audio_dir,
            documents_dir=documents_dir
        )
        
        return JSONResponse(content={
            "success": True,
            "message": "Batch ingestion completed",
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch ingestion failed: {str(e)}")


# Statistics Endpoint
@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get archive statistics.
    
    Returns information about the number of documents, chunks, and segments in the archive.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        stats = pipeline.get_stats()
        return StatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# Search by Metadata Endpoint
@router.get("/search/metadata")
async def search_by_metadata(
    source_type: Optional[str] = Query(None, description="Filter by 'audio' or 'text'"),
    speaker: Optional[str] = Query(None, description="Filter by speaker"),
    source_file: Optional[str] = Query(None, description="Filter by source filename"),
    limit: Optional[int] = Query(10, description="Maximum number of results")
):
    """
    Search documents by metadata filters.
    
    Returns documents matching the specified metadata criteria.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        filter_metadata = {}
        if source_type:
            filter_metadata["type"] = source_type
        if speaker:
            filter_metadata["speaker"] = speaker
        if source_file:
            filter_metadata["source"] = source_file
        
        if not filter_metadata:
            raise HTTPException(status_code=400, detail="At least one filter must be provided")
        
        results = pipeline.vector_store.get_by_metadata(
            filter_metadata=filter_metadata,
            limit=limit
        )
        
        return JSONResponse(content={
            "filters": filter_metadata,
            "num_results": len(results["ids"]),
            "results": [
                {
                    "id": results["ids"][i],
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i]
                }
                for i in range(len(results["ids"]))
            ]
        })
        
    except Exception as e:
        logger.error(f"Metadata search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metadata search failed: {str(e)}")


# Delete by Source Endpoint
@router.delete("/delete/{source_filename}")
async def delete_by_source(source_filename: str):
    """
    Delete all documents from a specific source file.
    
    Use with caution - this operation cannot be undone.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        num_deleted = pipeline.vector_store.delete_by_source(source_filename)
        
        return JSONResponse(content={
            "success": True,
            "message": f"Deleted {num_deleted} documents from {source_filename}",
            "num_deleted": num_deleted
        })
        
    except Exception as e:
        logger.error(f"Delete operation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Delete operation failed: {str(e)}")


# Reset Collection Endpoint
@router.post("/admin/reset")
async def reset_collection(confirm: bool = Query(False, description="Must be true to confirm reset")):
    """
    Reset the entire collection (DANGER: Deletes all data).
    
    This is an administrative endpoint that should be used with extreme caution.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must confirm reset by setting confirm=true"
        )
    
    try:
        pipeline.vector_store.reset_collection()
        
        return JSONResponse(content={
            "success": True,
            "message": "Collection reset successfully",
            "warning": "All data has been deleted"
        })
        
    except Exception as e:
        logger.error(f"Reset operation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset operation failed: {str(e)}")

# RAGAS Evaluation Endpoint
@router.post("/evaluate")
async def evaluate_response(request: EvaluationRequest):
    """
    Evaluate a RAG system response.
    
    - **question**: Original question
    - **answer**: Generated answer
    - **context**: Retrieved context chunks
    - **ground_truth**: Optional ground truth answer
    """
    try:
        logger.info("Evaluating response")
        
        evaluator = LLMRAGEvaluator()
        result = evaluator.evaluate_response(
            question=request.question,
            answer=request.answer,
            retrieved_context=request.context,
            ground_truth=request.ground_truth
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))