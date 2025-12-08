from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from loguru import logger
import sys
from pathlib import Path

from api.router import router as api_router

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add("logs/app.log", rotation="500 MB", retention="10 days", level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title="Berlin Media Archive API",
    description="RAG-based system for audio and document processing with attribution",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api/v1")

@app.on_startup
async def startup_event():
    """Initialize required directories on startup"""
    logger.info("Starting Berlin Media Archive API...")
    
    # Create necessary directories
    dirs = ["data/audio", "data/documents", "data/vectorstore", "output", "logs"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("All directories initialized")

@app.get("/")
async def root():
    return {
        "message": "Berlin Media Archive API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)