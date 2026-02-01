"""Tests for text chunking service."""

import pytest
from unittest.mock import patch, MagicMock

from app.services.chunking import TextChunker, DocumentChunk
from app.services.document_processor import ProcessedDocument


class TestTextChunker:
    """Test cases for TextChunker."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        with patch("app.services.chunking.get_settings") as mock:
            mock_settings = MagicMock()
            mock_settings.chunk_size = 100
            mock_settings.chunk_overlap = 20
            mock.return_value = mock_settings
            yield mock_settings

    @pytest.fixture
    def chunker(self, mock_settings) -> TextChunker:
        """Create a TextChunker instance with mocked settings."""
        return TextChunker(chunk_size=100, chunk_overlap=20)

    def test_chunk_short_document(self, chunker: TextChunker) -> None:
        """Test chunking a document shorter than chunk size."""
        doc = ProcessedDocument(
            content="This is a short document.",
            filename="short.txt",
            doc_type="resume",
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "This is a short document."
        assert chunks[0].metadata["doc_type"] == "resume"
        assert chunks[0].metadata["filename"] == "short.txt"

    def test_chunk_long_document(self, chunker: TextChunker) -> None:
        """Test chunking a document longer than chunk size."""
        # Create a long document
        long_content = " ".join(["word"] * 200)
        doc = ProcessedDocument(
            content=long_content,
            filename="long.txt",
            doc_type="job_posting",
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.metadata["doc_type"] == "job_posting" for c in chunks)

    def test_chunk_metadata(self, chunker: TextChunker) -> None:
        """Test that chunks have correct metadata."""
        doc = ProcessedDocument(
            content="Test content for metadata verification.",
            filename="meta_test.pdf",
            doc_type="resume",
        )

        chunks = chunker.chunk_document(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["doc_type"] == "resume"
            assert chunk.metadata["filename"] == "meta_test.pdf"
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)
            assert "word_count" in chunk.metadata

    def test_chunk_ids_unique(self, chunker: TextChunker) -> None:
        """Test that chunk IDs are unique."""
        doc = ProcessedDocument(
            content=" ".join(["word"] * 200),
            filename="test.txt",
            doc_type="resume",
        )

        chunks = chunker.chunk_document(doc)
        ids = [chunk.id for chunk in chunks]

        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"

    def test_chunk_text_method(self, chunker: TextChunker) -> None:
        """Test the chunk_text convenience method."""
        chunks = chunker.chunk_text(
            text="Simple text content",
            doc_type="job_posting",
            filename="job.txt",
        )

        assert len(chunks) == 1
        assert chunks[0].metadata["doc_type"] == "job_posting"


class TestDocumentChunk:
    """Test cases for DocumentChunk dataclass."""

    def test_token_estimate(self) -> None:
        """Test token estimation (roughly 4 chars per token)."""
        chunk = DocumentChunk(
            id="test-id",
            content="a" * 100,  # 100 characters
            metadata={},
        )

        assert chunk.token_estimate == 25  # 100 / 4
