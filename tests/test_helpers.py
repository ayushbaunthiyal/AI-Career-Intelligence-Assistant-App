"""Tests for utility helper functions."""

import pytest

from app.utils.helpers import (
    format_file_size,
    get_file_extension,
    generate_doc_id,
    generate_chunk_id,
    sanitize_filename,
)


class TestFormatFileSize:
    """Test cases for format_file_size function."""

    def test_bytes(self) -> None:
        """Test formatting bytes."""
        assert format_file_size(500) == "500.0 B"

    def test_kilobytes(self) -> None:
        """Test formatting kilobytes."""
        assert format_file_size(2048) == "2.0 KB"

    def test_megabytes(self) -> None:
        """Test formatting megabytes."""
        assert format_file_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self) -> None:
        """Test formatting gigabytes."""
        assert format_file_size(3 * 1024 * 1024 * 1024) == "3.0 GB"


class TestGetFileExtension:
    """Test cases for get_file_extension function."""

    def test_pdf_extension(self) -> None:
        """Test PDF extension extraction."""
        assert get_file_extension("resume.pdf") == ".pdf"

    def test_uppercase_extension(self) -> None:
        """Test uppercase extensions are lowercased."""
        assert get_file_extension("document.PDF") == ".pdf"

    def test_docx_extension(self) -> None:
        """Test DOCX extension."""
        assert get_file_extension("my_resume.docx") == ".docx"

    def test_no_extension(self) -> None:
        """Test file with no extension."""
        assert get_file_extension("filename") == ""

    def test_multiple_dots(self) -> None:
        """Test filename with multiple dots."""
        assert get_file_extension("file.name.pdf") == ".pdf"


class TestGenerateDocId:
    """Test cases for generate_doc_id function."""

    def test_generates_id(self) -> None:
        """Test that an ID is generated."""
        doc_id = generate_doc_id("test content", "resume")
        assert doc_id is not None
        assert len(doc_id) > 0

    def test_includes_doc_type(self) -> None:
        """Test that doc_type is included in ID."""
        doc_id = generate_doc_id("content", "resume")
        assert doc_id.startswith("resume_")

    def test_same_content_same_id(self) -> None:
        """Test that same content produces same ID."""
        id1 = generate_doc_id("identical content", "resume")
        id2 = generate_doc_id("identical content", "resume")
        assert id1 == id2

    def test_different_content_different_id(self) -> None:
        """Test that different content produces different ID."""
        id1 = generate_doc_id("content one", "resume")
        id2 = generate_doc_id("content two", "resume")
        assert id1 != id2


class TestGenerateChunkId:
    """Test cases for generate_chunk_id function."""

    def test_generates_uuid(self) -> None:
        """Test that a UUID-like string is generated."""
        chunk_id = generate_chunk_id()
        assert len(chunk_id) == 36  # UUID format: 8-4-4-4-12

    def test_unique_ids(self) -> None:
        """Test that generated IDs are unique."""
        ids = [generate_chunk_id() for _ in range(100)]
        assert len(ids) == len(set(ids))


class TestSanitizeFilename:
    """Test cases for sanitize_filename function."""

    def test_normal_filename(self) -> None:
        """Test normal filename passes through."""
        assert sanitize_filename("resume.pdf") == "resume.pdf"

    def test_removes_path_separators(self) -> None:
        """Test that path separators are replaced."""
        assert "/" not in sanitize_filename("path/to/file.pdf")
        assert "\\" not in sanitize_filename("path\\to\\file.pdf")

    def test_long_filename_truncated(self) -> None:
        """Test that very long filenames are truncated."""
        long_name = "a" * 300 + ".pdf"
        sanitized = sanitize_filename(long_name)
        assert len(sanitized) <= 255
        assert sanitized.endswith(".pdf")
