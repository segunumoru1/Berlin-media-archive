"""
Configuration Management
Centralized settings using Pydantic Settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: str = Field(..., validation_alias="OPENAI_API_KEY")
    huggingface_token: str = Field(default="", validation_alias="HUGGINGFACE_TOKEN")
    
    # OpenAI Models
    llm_model: str = Field(default="gpt-4-turbo-preview", validation_alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, validation_alias="LLM_TEMPERATURE")
    embedding_model: str = Field(default="text-embedding-3-large", validation_alias="EMBEDDING_MODEL")
    
    # Whisper Settings
    whisper_model: str = Field(default="base", validation_alias="WHISPER_MODEL")
    enable_speaker_diarization: bool = Field(default=True, validation_alias="ENABLE_DIARIZATION")
    
    # Document Processing
    chunk_size: int = Field(default=1000, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, validation_alias="CHUNK_OVERLAP")
    
    # Vector Store
    collection_name: str = Field(default="berlin_archive", validation_alias="COLLECTION_NAME")
    vectorstore_path: str = Field(default="./data/vectorstore", validation_alias="VECTORSTORE_PATH")
    top_k_results: int = Field(default=5, validation_alias="TOP_K_RESULTS")
    
    # Hybrid Search
    enable_hybrid_search: bool = Field(default=True, validation_alias="ENABLE_HYBRID_SEARCH")
    bm25_weight: float = Field(default=0.3, validation_alias="BM25_WEIGHT")
    
    # Paths
    audio_dir: str = Field(default="./data/audio", validation_alias="AUDIO_DIR")
    documents_dir: str = Field(default="./data/documents", validation_alias="DOCUMENTS_DIR")
    output_dir: str = Field(default="./output", validation_alias="OUTPUT_DIR")
    logs_dir: str = Field(default="./logs", validation_alias="LOGS_DIR")
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    
    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }


# Initialize settings
try:
    settings = Settings()
except Exception as e:
    print(f"Warning: Could not load settings from .env: {e}")
    print("Using default settings...")
    settings = Settings(
        openai_api_key="your_openai_api_key_here",
        huggingface_token=""
    )


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        settings.audio_dir,
        settings.documents_dir,
        settings.output_dir,
        settings.vectorstore_path,
        settings.logs_dir,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)