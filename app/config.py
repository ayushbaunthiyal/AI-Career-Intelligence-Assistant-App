"""
Application Configuration Management using Pydantic Settings.

This module implements the 12-factor app pattern for configuration:
- Configuration is stored in environment variables (or .env file)
- Type-safe settings using Pydantic for validation
- Sensible defaults for development
- Easy override for production via environment variables

The Settings class uses Pydantic BaseSettings which automatically:
- Loads from .env file
- Validates types
- Provides IDE autocomplete
- Handles missing required fields with clear errors
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings with Environment Variable Support.

    This class defines all configuration options for the application.
    Settings can be provided via:
    1. Environment variables (production)
    2. .env file (development)
    3. Default values (for optional settings)

    Pydantic automatically validates types and provides clear error messages
    if required settings (like OPENAI_API_KEY) are missing.
    """

    model_config = SettingsConfigDict(
        env_file=".env",  # Load from .env file if present
        env_file_encoding="utf-8",  # UTF-8 encoding for .env file
        case_sensitive=False,  # Allow OPENAI_API_KEY or openai_api_key
        extra="ignore",  # Ignore extra env vars not defined here
    )

    # =============================================================================
    # OPENAI CONFIGURATION
    # =============================================================================
    # These settings control which OpenAI models are used for LLM and embeddings.
    # gpt-4o-mini is chosen for cost-effectiveness while maintaining quality.
    # text-embedding-3-small provides 1536-dimensional vectors at low cost.
    openai_api_key: str  # Required - no default, must be provided
    openai_llm_model: str = "gpt-4o-mini"  # Default LLM for chat responses
    openai_embedding_model: str = "text-embedding-3-small"  # Default embedding model

    # =============================================================================
    # CHROMADB CONFIGURATION
    # =============================================================================
    # ChromaDB stores vector embeddings locally. The persist_directory ensures
    # data survives app restarts. The collection_name groups all documents
    # in a single collection for easy querying.
    chroma_persist_directory: Path = Path("./data/chroma")  # Where to store vectors
    chroma_collection_name: str = "career_documents"  # Collection name in ChromaDB

    # =============================================================================
    # CHUNKING CONFIGURATION
    # =============================================================================
    # These values control how documents are split into chunks:
    # - chunk_size: Maximum characters per chunk (512 = ~2000 chars, good for resumes)
    # - chunk_overlap: Characters to overlap between chunks (ensures context continuity)
    chunk_size: int = 512  # Characters per chunk (not tokens, for simplicity)
    chunk_overlap: int = 50  # Overlap between chunks to preserve context

    # =============================================================================
    # RAG CONFIGURATION
    # =============================================================================
    # These settings control retrieval behavior:
    # - retrieval_top_k: How many chunks to retrieve per query (5 = good balance)
    # - max_chat_history: How many conversation turns to remember (5 = ~10 messages)
    retrieval_top_k: int = 5  # Number of chunks to retrieve per query
    max_chat_history: int = 5  # Number of conversation turns to remember

    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    # General application configuration:
    # - log_level: Logging verbosity (INFO shows important events)
    # - max_file_size_mb: Maximum upload size to prevent abuse
    log_level: str = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR)
    max_file_size_mb: int = 10  # Maximum file upload size in megabytes

    @property
    def allowed_extensions(self) -> set[str]:
        """
        Get the set of allowed file extensions for uploads.

        This property provides a convenient way to check if a file type
        is supported. It's used by the document processor to validate
        uploads before processing.

        Returns:
            Set of allowed file extensions (e.g., {".pdf", ".docx"}).
        """
        return {".pdf", ".docx", ".doc", ".txt"}


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance (singleton pattern).

    Using @lru_cache ensures we only create one Settings instance per
    application run. This is important because:
    1. Settings are loaded from .env file once
    2. Pydantic validation happens once
    3. Memory efficiency (no duplicate Settings objects)

    The cache is cleared when the application restarts, ensuring fresh
    settings are loaded if the .env file changes.

    Returns:
        Cached Settings instance with all configuration loaded.
    """
    return Settings()
