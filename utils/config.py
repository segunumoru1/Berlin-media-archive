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
    openai_api_key: str = Field(validation_alias="OPENAI_API_KEY")
    huggingface_token: str = Field(default=None, validation_alias="HUGGINGFACE_TOKEN")
    
    # OpenAI Models
    llm_model: str = Field("gpt-4-turbo-preview", env="LLM_MODEL")
    llm_temperature: float = Field(0.3, env="LLM_TEMPERATURE")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    
    # Whisper Settings
    whisper_model: str = Field("base", env="WHISPER_MODEL")
    enable_speaker_diarization: bool = Field(True, env="ENABLE_DIARIZATION")
    
    # Document Processing
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # Vector Store
    collection_name: str = Field("berlin_archive", env="COLLECTION_NAME")
    vectorstore_path: str = Field("./data/vectorstore", env="VECTORSTORE_PATH")
    top_k_results: int = Field(5, env="TOP_K_RESULTS")
    
    # Hybrid Search
    enable_hybrid_search: bool = Field(True, env="ENABLE_HYBRID_SEARCH")
    bm25_weight: float = Field(0.3, env="BM25_WEIGHT")
    
    # Paths
    audio_dir: str = Field("./data/audio", env="AUDIO_DIR")
    documents_dir: str = Field("./data/documents", env="DOCUMENTS_DIR")
    output_dir: str = Field("./output", env="OUTPUT_DIR")
    
    # API Settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Initialize settings
settings = Settings()


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        settings.audio_dir,
        settings.documents_dir,
        settings.output_dir,
        settings.vectorstore_path,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)