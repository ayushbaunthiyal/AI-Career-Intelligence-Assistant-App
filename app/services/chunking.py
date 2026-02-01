"""
Text Chunking Service for Document Segmentation.

This module handles splitting documents into smaller chunks that can be
efficiently embedded and stored in the vector database. Chunking is critical
for RAG systems because:
1. Embeddings work best with focused text segments
2. Retrieval can target specific sections rather than entire documents
3. Context windows have limits - chunks must fit within LLM context

The chunking strategy uses RecursiveCharacterTextSplitter which tries to
preserve semantic boundaries (paragraphs > sentences > words) while respecting
size limits. Overlap between chunks ensures context isn't lost at boundaries.
"""

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.services.document_processor import ProcessedDocument
from app.utils.helpers import generate_chunk_id

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    Data class representing a single chunk of document text.

    Each chunk contains:
    - Unique ID for tracking and retrieval
    - Text content (the actual chunk text)
    - Metadata (document type, filename, position, etc.)

    The metadata is crucial for:
    - Filtering chunks by document type
    - Attributing retrieved chunks to source documents
    - Understanding chunk position within the original document
    """

    id: str
    content: str
    metadata: dict

    @property
    def token_estimate(self) -> int:
        """
        Estimate token count for this chunk.

        This is a rough approximation (4 characters per token) used for
        monitoring and debugging. Actual tokenization varies by model,
        but this gives a reasonable estimate for chunk size validation.

        Returns:
            Estimated number of tokens in the chunk.
        """
        return len(self.content) // 4


class TextChunker:
    """
    Text Chunker for Splitting Documents into Embeddable Chunks.

    This class uses LangChain's RecursiveCharacterTextSplitter which intelligently
    splits text while preserving semantic boundaries. It tries to split at:
    1. Paragraph breaks (best - preserves complete thoughts)
    2. Sentence boundaries (good - preserves complete ideas)
    3. Word boundaries (acceptable - preserves words)
    4. Character boundaries (last resort - may break words)

    The overlap between chunks ensures that context isn't lost at boundaries.
    For example, if a skill is mentioned at the end of one chunk and the beginning
    of the next, the overlap ensures it appears in both chunks for better retrieval.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Initialize the text chunker with size and overlap configuration.

        Chunk size (512 tokens) is optimized for:
        - Resume sections (Education, Experience entries)
        - Job posting bullet points and requirements
        - Embedding model efficiency (not too small, not too large)

        Overlap (50 tokens) ensures:
        - Context continuity across chunk boundaries
        - Important information isn't split across chunks
        - Better retrieval when queries match boundary content

        Args:
            chunk_size: Maximum chunk size in characters (defaults to config).
            chunk_overlap: Overlap between chunks in characters (defaults to config).
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Initialize RecursiveCharacterTextSplitter with semantic-aware separators
        # The splitter tries separators in order, using the first one that works
        # This preserves document structure as much as possible
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,  # Use character count (not token count) for simplicity
            separators=[
                "\n\n",  # Paragraphs - best semantic boundary
                "\n",    # Lines - good for structured documents
                ". ",    # Sentences - preserves complete thoughts
                ", ",    # Clauses - acceptable for lists
                " ",     # Words - preserves word integrity
                "",      # Characters - last resort, may break words
            ],
        )

    # =============================================================================
    # DOCUMENT CHUNKING METHODS
    # =============================================================================

    def chunk_document(self, document: ProcessedDocument) -> list[DocumentChunk]:
        """
        Split a processed document into chunks with rich metadata.

        This method takes a ProcessedDocument (from document_processor) and
        splits it into smaller chunks. Each chunk is assigned:
        - A unique ID for tracking
        - The chunk text content
        - Comprehensive metadata for filtering and attribution

        The metadata includes document type, filename, position (chunk_index),
        total chunks, and word count. This enables:
        - Filtering chunks by document type in retrieval
        - Showing source documents in responses
        - Understanding chunk position within the original document

        Args:
            document: ProcessedDocument containing text content and metadata.

        Returns:
            List of DocumentChunk objects ready for embedding and storage.
        """
        # Split the document content using the configured splitter
        # The splitter handles size limits and overlap automatically
        texts = self.splitter.split_text(document.content)

        # Create DocumentChunk objects with metadata for each split text
        chunks = []
        for i, text in enumerate(texts):
            chunk = DocumentChunk(
                id=generate_chunk_id(),  # Unique ID for each chunk
                content=text,
                metadata={
                    "doc_type": document.doc_type,  # "resume" or "job_posting"
                    "filename": document.filename,  # Original filename
                    "chunk_index": i,  # Position in the chunk sequence
                    "total_chunks": len(texts),  # Total chunks from this document
                    "word_count": len(text.split()),  # Word count for this chunk
                },
            )
            chunks.append(chunk)

        # Log chunking statistics for monitoring
        avg_chunk_size = sum(len(c.content) for c in chunks) // max(len(chunks), 1)
        logger.info(
            f"Split '{document.filename}' into {len(chunks)} chunks "
            f"(avg {avg_chunk_size} chars/chunk)"
        )

        return chunks

    def chunk_text(
        self,
        text: str,
        doc_type: str,
        filename: str,
    ) -> list[DocumentChunk]:
        """
        Chunk raw text directly (convenience method for pasted content).

        This is a convenience method that wraps raw text in a ProcessedDocument
        and then chunks it. It's used when processing pasted job postings that
        don't go through the file upload pipeline.

        Args:
            text: Raw text content to chunk.
            doc_type: Document type ("resume" or "job_posting").
            filename: Filename/title for the document.

        Returns:
            List of DocumentChunk objects with the same structure as chunk_document().
        """
        from app.services.document_processor import ProcessedDocument

        # Wrap raw text in ProcessedDocument structure
        doc = ProcessedDocument(
            content=text,
            filename=filename,
            doc_type=doc_type,
        )
        # Delegate to main chunking method
        return self.chunk_document(doc)
