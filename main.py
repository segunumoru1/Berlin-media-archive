"""
Berlin Media Archive - Main Application
Multi-Modal RAG System for Historical Content
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from api.router import router
from utils.logger import setup_logging
from utils.config import settings

# Setup logging
setup_logging(
    log_level=settings.log_level,
    log_file="berlin_archive.log"
)

# Create FastAPI app
app = FastAPI(
    title="Berlin Media Archive API",
    description="""
    # Berlin Media Archive - Multi-Modal RAG System
    
    A production-grade system for searching across historical audio and document archives.
    
    ## Features
    - 🎙️ Audio transcription with timestamp preservation
    - 📄 PDF document processing with page tracking
    - 🔍 Hybrid search (semantic + keyword)
    - 👥 Speaker diarization for audio content
    - 📊 Precise source attribution and citations
    - 🎯 Advanced filtering by metadata
    
    ## Workflow
    1. **Ingest** audio files or PDF documents
    2. **Query** the archive with natural language questions
    3. **Get answers** with precise citations (timestamps/page numbers)
    
    ## Assessment Requirements Met
    - ✅ Audio ingestion with timestamps
    - ✅ Unified vector store with metadata
    - ✅ Attribution engine with citations
    - ✅ Hybrid search (Module A)
    - ✅ Speaker diarization (Module B)
    - ✅ Production-ready error handling
    - ✅ Comprehensive logging
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with welcome message."""
    return """
    <html>
        <head>
            <title>Berlin Media Archive</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                    border-bottom: 3px solid #007bff;
                    padding-bottom: 10px;
                }
                .feature {
                    margin: 15px 0;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-left: 4px solid #007bff;
                }
                .links {
                    margin-top: 30px;
                }
                a {
                    display: inline-block;
                    margin: 10px 10px 10px 0;
                    padding: 10px 20px;
                    background-color: #007bff;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }
                a:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏛️ Berlin Media Archive</h1>
                <p><strong>Multi-Modal RAG System for Historical Content</strong></p>
                
                <div class="feature">
                    <strong>🎙️ Audio Processing:</strong> Transcription with timestamps and speaker diarization
                </div>
                <div class="feature">
                    <strong>📄 Document Processing:</strong> Intelligent PDF chunking with page tracking
                </div>
                <div class="feature">
                    <strong>🔍 Hybrid Search:</strong> Combines semantic and keyword search
                </div>
                <div class="feature">
                    <strong>📊 Attribution:</strong> Precise citations with timestamps and page numbers
                </div>
                
                <div class="links">
                    <a href="/docs">📚 API Documentation</a>
                    <a href="/api/v1/health">🏥 Health Check</a>
                    <a href="/api/v1/stats">📊 Statistics</a>
                </div>
                
                <hr style="margin: 30px 0;">
                
                <h2>Quick Start</h2>
                <ol>
                    <li>Upload an audio file: <code>POST /api/v1/ingest/audio</code></li>
                    <li>Upload a PDF document: <code>POST /api/v1/ingest/document</code></li>
                    <li>Query the archive: <code>POST /api/v1/query</code></li>
                </ol>
                
                <p style="margin-top: 30px; color: #666; font-size: 0.9em;">
                    <strong>Assessment Version 1.0.0</strong> | 
                    Built for the Berlin Media Archive Technical Assessment
                </p>
            </div>
        </body>
    </html>
    """


# Main entry point
if __name__ == "__main__":
    print("=" * 80)
    print("🏛️  BERLIN MEDIA ARCHIVE - Multi-Modal RAG System")
    print("=" * 80)
    print(f"📍 Server starting on http://{settings.api_host}:{settings.api_port}")
    print(f"📚 API Documentation: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"🔄 Alternative Docs: http://{settings.api_host}:{settings.api_port}/redoc")
    print("=" * 80)
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )