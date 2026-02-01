"""Tests for document processor service."""

import io
import pytest

from app.services.document_processor import DocumentProcessor, ProcessedDocument


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""

    @pytest.fixture
    def processor(self) -> DocumentProcessor:
        """Create a DocumentProcessor instance."""
        return DocumentProcessor()

    def test_process_txt_file(self, processor: DocumentProcessor) -> None:
        """Test processing a plain text file."""
        content = b"This is a test resume.\nWith multiple lines.\nAnd skills like Python."
        file = io.BytesIO(content)

        result = processor.process(file, "test_resume.txt", "resume")

        assert isinstance(result, ProcessedDocument)
        assert result.filename == "test_resume.txt"
        assert result.doc_type == "resume"
        assert "Python" in result.content
        assert result.word_count > 0

    def test_process_text_directly(self, processor: DocumentProcessor) -> None:
        """Test processing raw text input."""
        text = """
        Senior Software Engineer
        
        Requirements:
        - 5+ years of experience
        - Python and JavaScript
        - Machine Learning background
        """

        result = processor.process_text(text, "Google SWE Role", "job_posting")

        assert result.filename == "Google SWE Role"
        assert result.doc_type == "job_posting"
        assert "Python" in result.content
        assert result.word_count > 0

    def test_unsupported_file_type(self, processor: DocumentProcessor) -> None:
        """Test that unsupported file types raise ValueError."""
        file = io.BytesIO(b"test content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.process(file, "test.xyz", "resume")

    def test_clean_text_removes_extra_whitespace(self, processor: DocumentProcessor) -> None:
        """Test that text cleaning normalizes whitespace."""
        text = "Hello    World\n\n\n\nNew Section"
        cleaned = processor._clean_text(text)

        assert "    " not in cleaned
        assert "\n\n\n\n" not in cleaned

    def test_word_count_calculated(self, processor: DocumentProcessor) -> None:
        """Test that word count is calculated correctly."""
        text = "One two three four five"
        result = processor.process_text(text, "test", "resume")

        assert result.word_count == 5


class TestProcessedDocument:
    """Test cases for ProcessedDocument dataclass."""

    def test_post_init_calculates_word_count(self) -> None:
        """Test that word count is calculated in __post_init__."""
        doc = ProcessedDocument(
            content="Hello world this is a test",
            filename="test.txt",
            doc_type="resume",
        )

        assert doc.word_count == 6

    def test_empty_content_word_count(self) -> None:
        """Test word count for empty content."""
        doc = ProcessedDocument(
            content="",
            filename="empty.txt",
            doc_type="resume",
        )

        # Empty string split returns [''], but we count actual words
        assert doc.word_count == 0
