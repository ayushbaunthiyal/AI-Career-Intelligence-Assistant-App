"""Text chunking service for document segmentation."""

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.services.document_processor import ProcessedDocument
from app.utils.helpers import generate_chunk_id

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of document text with metadata."""

    id: str
    content: str
    metadata: dict

    @property
    def token_estimate(self) -> int:
        """Estimate token count (rough approximation: 4 chars per token)."""
        return len(self.content) // 4


class TextChunker:
    """Handles splitting documents into semantic chunks for embedding."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """Initialize the text chunker.

        Args:
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Overlap between chunks in characters.
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Use RecursiveCharacterTextSplitter for semantic-aware splitting
        # It tries to split on paragraphs, then sentences, then words
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ". ",    # Sentences
                ", ",    # Clauses
                " ",     # Words
                "",      # Characters (last resort)
            ],
        )

    def chunk_document(self, document: ProcessedDocument) -> list[DocumentChunk]:
        """Split a document into chunks with metadata.

        Args:
            document: ProcessedDocument to chunk.

        Returns:
            List of DocumentChunk objects.
        """
        # Split the document content
        texts = self.splitter.split_text(document.content)

        chunks = []
        for i, text in enumerate(texts):
            chunk = DocumentChunk(
                id=generate_chunk_id(),
                content=text,
                metadata={
                    "doc_type": document.doc_type,
                    "filename": document.filename,
                    "chunk_index": i,
                    "total_chunks": len(texts),
                    "word_count": len(text.split()),
                },
            )
            chunks.append(chunk)

        logger.info(
            f"Split '{document.filename}' into {len(chunks)} chunks "
            f"(avg {sum(len(c.content) for c in chunks) // max(len(chunks), 1)} chars/chunk)"
        )

        return chunks

    def chunk_text(
        self,
        text: str,
        doc_type: str,
        filename: str,
    ) -> list[DocumentChunk]:
        """Chunk raw text with minimal metadata.

        Args:
            text: Text content to chunk.
            doc_type: Document type for metadata.
            filename: Filename for metadata.

        Returns:
            List of DocumentChunk objects.
        """
        from app.services.document_processor import ProcessedDocument

        doc = ProcessedDocument(
            content=text,
            filename=filename,
            doc_type=doc_type,
        )
        return self.chunk_document(doc)
