"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ChromaDB Configuration
    chroma_persist_directory: Path = Path("./data/chroma")
    chroma_collection_name: str = "career_documents"

    # Chunking Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50

    # RAG Configuration
    retrieval_top_k: int = 5
    max_chat_history: int = 5

    # Application Settings
    log_level: str = "INFO"
    max_file_size_mb: int = 10

    @property
    def allowed_extensions(self) -> set[str]:
        """Allowed file extensions for upload."""
        return {".pdf", ".docx", ".doc", ".txt"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
