"""Pytest configuration and shared fixtures."""

import os
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def mock_openai_api_key():
    """Mock OpenAI API key for all tests."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
        yield


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    mock = MagicMock()
    mock.openai_api_key = "test-api-key"
    mock.openai_llm_model = "gpt-4o-mini"
    mock.openai_embedding_model = "text-embedding-3-small"
    mock.chroma_persist_directory = "/tmp/test_chroma"
    mock.chroma_collection_name = "test_collection"
    mock.chunk_size = 512
    mock.chunk_overlap = 50
    mock.retrieval_top_k = 5
    mock.max_chat_history = 5
    mock.log_level = "INFO"
    mock.max_file_size_mb = 10
    mock.allowed_extensions = {".pdf", ".docx", ".doc", ".txt"}
    return mock


@pytest.fixture
def sample_resume_text() -> str:
    """Sample resume text for testing."""
    return """
    John Doe
    Senior Software Engineer
    
    Experience:
    - 5 years of Python development
    - Machine learning and data science
    - Cloud platforms (AWS, GCP)
    
    Skills:
    - Python, JavaScript, Go
    - TensorFlow, PyTorch
    - Docker, Kubernetes
    
    Education:
    - MS Computer Science, Stanford University
    """


@pytest.fixture
def sample_job_posting_text() -> str:
    """Sample job posting text for testing."""
    return """
    Senior ML Engineer
    Google
    
    Requirements:
    - 5+ years of software engineering experience
    - Strong Python skills
    - Experience with ML frameworks (TensorFlow, PyTorch)
    - Distributed systems knowledge
    
    Nice to have:
    - Published research papers
    - Experience with TPUs
    - Kubernetes experience
    """
