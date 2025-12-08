import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")
warnings.filterwarnings("ignore", message="`huggingface_hub` cache-system uses symlinks")
warnings.filterwarnings("ignore", message="Xet Storage is enabled")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")


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
    level=os.getenv("LOG_LEVEL", "INFO")
)
logger.add("logs/app.log", rotation="500 MB", retention="10 days", level=os.getenv("LOG_LEVEL", "INFO"))

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

async def startup_event():
    """Initialize required directories on startup"""
    logger.info("Starting Berlin Media Archive API...")
    
    # Create necessary directories
    dirs = ["data/audio", "data/documents", "data/vectorstore", "output", "logs"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("All directories initialized")

app.add_event_handler("startup", startup_event)

@app.get("/")
async def root():
    return {
        "message": "Berlin Media Archive API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_status = "configured" if api_key and api_key != "your_openai_key_here" else "missing"
    
    return {
        "status": "healthy",
        "openai_api_key": api_key_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)