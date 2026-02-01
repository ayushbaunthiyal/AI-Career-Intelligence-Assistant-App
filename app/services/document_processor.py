"""Document processing service for PDF and DOCX files."""

import logging
from dataclasses import dataclass
from typing import BinaryIO

from docx import Document as DocxDocument
from pypdf import PdfReader

from app.utils.helpers import get_file_extension

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Represents a processed document with extracted text and metadata."""

    content: str
    filename: str
    doc_type: str  # "resume" or "job_posting"
    page_count: int = 1
    word_count: int = 0

    def __post_init__(self) -> None:
        """Calculate word count after initialization."""
        self.word_count = len(self.content.split())


class DocumentProcessor:
    """Handles extraction of text from various document formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

    def process(
        self,
        file: BinaryIO,
        filename: str,
        doc_type: str,
    ) -> ProcessedDocument:
        """Process an uploaded file and extract text content.

        Args:
            file: Binary file object.
            filename: Original filename.
            doc_type: Document type ("resume" or "job_posting").

        Returns:
            ProcessedDocument with extracted content and metadata.

        Raises:
            ValueError: If file type is not supported.
        """
        extension = get_file_extension(filename)

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        logger.info(f"Processing {doc_type}: {filename}")

        if extension == ".pdf":
            content, page_count = self._extract_pdf(file)
        elif extension in {".docx", ".doc"}:
            content, page_count = self._extract_docx(file)
        elif extension == ".txt":
            content, page_count = self._extract_txt(file)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

        # Clean up extracted text
        content = self._clean_text(content)

        logger.info(
            f"Extracted {len(content)} characters from {filename} "
            f"({page_count} pages, {len(content.split())} words)"
        )

        return ProcessedDocument(
            content=content,
            filename=filename,
            doc_type=doc_type,
            page_count=page_count,
        )

    def _extract_pdf(self, file: BinaryIO) -> tuple[str, int]:
        """Extract text from PDF file.

        Args:
            file: Binary file object.

        Returns:
            Tuple of (extracted text, page count).
        """
        reader = PdfReader(file)
        pages = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

        return "\n\n".join(pages), len(reader.pages)

    def _extract_docx(self, file: BinaryIO) -> tuple[str, int]:
        """Extract text from DOCX file.

        Args:
            file: Binary file object.

        Returns:
            Tuple of (extracted text, estimated page count).
        """
        doc = DocxDocument(file)
        paragraphs = []

        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)

        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    paragraphs.append(row_text)

        content = "\n\n".join(paragraphs)
        # Estimate page count based on character count (~3000 chars per page)
        estimated_pages = max(1, len(content) // 3000)

        return content, estimated_pages

    def _extract_txt(self, file: BinaryIO) -> tuple[str, int]:
        """Extract text from plain text file.

        Args:
            file: Binary file object.

        Returns:
            Tuple of (text content, page count of 1).
        """
        content = file.read()

        # Try to decode as UTF-8, fallback to latin-1
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

        return text, 1

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text with normalized whitespace.
        """
        # Replace multiple spaces/tabs with single space
        import re
        text = re.sub(r"[ \t]+", " ", text)
        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace
        return text.strip()

    def process_text(self, text: str, title: str, doc_type: str) -> ProcessedDocument:
        """Process raw text input (for job postings pasted as text).

        Args:
            text: Raw text content.
            title: Title for the document.
            doc_type: Document type ("resume" or "job_posting").

        Returns:
            ProcessedDocument with the text content.
        """
        content = self._clean_text(text)
        return ProcessedDocument(
            content=content,
            filename=title,
            doc_type=doc_type,
            page_count=1,
        )
