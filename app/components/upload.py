"""
File Upload Component for Resumes and Job Postings.

This module handles document upload and processing. It provides two methods:
1. File Upload: Users can upload PDF/DOCX files
2. Paste Text: Users can paste job posting text directly (useful for copying
   from job boards)

When documents are uploaded, they go through the processing pipeline:
- Document extraction (text from PDF/DOCX)
- Chunking (splitting into embeddable segments)
- Embedding and storage in vector database

The component handles the single-resume model: uploading a new resume
automatically replaces the old one.
"""

from typing import Literal

import streamlit as st

from app.config import get_settings
from app.services.chunking import TextChunker
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService


def render_upload_section(
    vector_store: VectorStoreService,
    doc_processor: DocumentProcessor,
    chunker: TextChunker,
) -> None:
    """
    Render the document upload section with file and text input options.

    This component provides a tabbed interface for two upload methods:
    - File Upload: For PDF/DOCX resumes and job postings
    - Paste Text: For quickly adding job postings copied from job boards

    Both methods ultimately go through the same processing pipeline, but
    the text paste method is more convenient for job postings that users
    copy from websites.

    Args:
        vector_store: VectorStoreService for storing processed documents.
        doc_processor: DocumentProcessor for extracting text from files.
        chunker: TextChunker for splitting documents into chunks.
    """
    settings = get_settings()
    stats = vector_store.get_document_stats()

    st.subheader("📤 Upload Documents")

    # Create tabs for different upload methods
    # This provides a clean separation between file upload and text paste
    tab1, tab2 = st.tabs(["📁 Upload File", "📝 Paste Text"])

    with tab1:
        _render_file_upload(
            vector_store, doc_processor, chunker, stats, settings
        )

    with tab2:
        _render_text_input(
            vector_store, doc_processor, chunker, stats
        )


# =============================================================================
# FILE UPLOAD INTERFACE
# =============================================================================

def _render_file_upload(
    vector_store: VectorStoreService,
    doc_processor: DocumentProcessor,
    chunker: TextChunker,
    stats: dict,
    settings,
) -> None:
    """
    Render the file upload interface with document type selection.

    This provides a file picker and radio buttons for selecting document type.
    The file is validated for size before processing to prevent abuse and
    ensure reasonable processing times.

    Args:
        vector_store: VectorStoreService for storing processed documents.
        doc_processor: DocumentProcessor for extracting text.
        chunker: TextChunker for splitting documents.
        stats: Current document statistics.
        settings: Application settings for file size limits.
    """
    col1, col2 = st.columns([2, 1])

    with col1:
        # File uploader with supported file types
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "doc", "txt"],
            help=f"Max file size: {settings.max_file_size_mb}MB",
        )

    with col2:
        # Document type selector
        # Users must specify whether they're uploading a resume or job posting
        # This is important because resumes are handled differently (single resume model)
        doc_type = st.radio(
            "Document Type",
            options=["resume", "job_posting"],
            format_func=lambda x: "📄 Resume" if x == "resume" else "💼 Job Posting",
            horizontal=False,
        )

    if uploaded_file is not None:
        # Validate file size before processing
        # Large files can cause timeouts and consume excessive resources
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            st.error(f"File too large ({file_size_mb:.1f}MB). Maximum size: {settings.max_file_size_mb}MB")
            return

        # Process button - triggers the document processing pipeline
        if st.button("📥 Process & Index", type="primary"):
            _process_uploaded_file(
                uploaded_file,
                doc_type,
                vector_store,
                doc_processor,
                chunker,
                stats,
            )


def _render_text_input(
    vector_store: VectorStoreService,
    doc_processor: DocumentProcessor,
    chunker: TextChunker,
    stats: dict,
) -> None:
    """
    Render text paste interface for job postings.

    This interface allows users to paste job posting text directly, which is
    often more convenient than downloading and uploading files. Users just
    copy text from job boards and paste it here.

    The interface requires both a job title (for identification) and the job
    posting content. The button is disabled until both fields are filled.

    Args:
        vector_store: VectorStoreService for storing processed documents.
        doc_processor: DocumentProcessor for processing text.
        chunker: TextChunker for splitting documents.
        stats: Current document statistics (unused but kept for consistency).
    """
    st.info("💡 Paste job posting text directly - great for copying from job boards!")

    # Job title input - used as the filename in the vector store
    job_title = st.text_input(
        "Job Title",
        placeholder="e.g., Senior ML Engineer @ Google",
    )

    # Job posting content - the actual text to analyze
    job_text = st.text_area(
        "Job Posting Content",
        height=200,
        placeholder="Paste the full job description here...",
    )

    # Process button - disabled until both fields have content
    # This prevents processing empty documents
    if st.button("📥 Add Job Posting", type="primary", disabled=not (job_title and job_text)):
        _process_text_input(
            job_title,
            job_text,
            vector_store,
            doc_processor,
            chunker,
        )


# =============================================================================
# DOCUMENT PROCESSING PIPELINE
# =============================================================================

def _process_uploaded_file(
    uploaded_file,
    doc_type: Literal["resume", "job_posting"],
    vector_store: VectorStoreService,
    doc_processor: DocumentProcessor,
    chunker: TextChunker,
    stats: dict,
) -> None:
    """
    Process an uploaded file through the complete pipeline.

    This function orchestrates the document processing pipeline:
    1. Delete old resume if uploading a new one (single resume model)
    2. Extract text from the file (PDF/DOCX/TXT)
    3. Split into chunks for embedding
    4. Store chunks in vector database with embeddings
    5. Update UI to show success and refresh sidebar

    The processing happens in a spinner to show progress to the user.
    Errors are caught and displayed gracefully.

    Args:
        uploaded_file: Streamlit uploaded file object.
        doc_type: Type of document ("resume" or "job_posting").
        vector_store: VectorStoreService for storing chunks.
        doc_processor: DocumentProcessor for text extraction.
        chunker: TextChunker for splitting documents.
        stats: Current document statistics (for checking existing resume).
    """
    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            # =================================================================
            # STEP 1: HANDLE RESUME REPLACEMENT
            # =================================================================
            # If uploading a new resume, delete the old one first
            # This implements the single-resume model where only one resume
            # can be active at a time
            if doc_type == "resume" and stats["resume_filename"]:
                vector_store.delete_by_doc_type("resume")
                st.info(f"Replaced previous resume: {stats['resume_filename']}")

            # =================================================================
            # STEP 2: EXTRACT TEXT FROM FILE
            # =================================================================
            # Process the uploaded file to extract text content
            # This handles PDF, DOCX, and TXT formats automatically
            processed_doc = doc_processor.process(
                file=uploaded_file,
                filename=uploaded_file.name,
                doc_type=doc_type,
            )

            # =================================================================
            # STEP 3: SPLIT INTO CHUNKS
            # =================================================================
            # Chunk the document for embedding and storage
            # Chunks are sized appropriately for embedding models
            chunks = chunker.chunk_document(processed_doc)

            # =================================================================
            # STEP 4: STORE IN VECTOR DATABASE
            # =================================================================
            # Add chunks to vector store with embeddings
            # This enables semantic search for retrieval
            vector_store.add_chunks(chunks)

            # =================================================================
            # STEP 5: USER FEEDBACK
            # =================================================================
            # Show success message with processing statistics
            st.success(
                f"✅ Successfully indexed **{uploaded_file.name}**\n\n"
                f"- {processed_doc.word_count} words\n"
                f"- {len(chunks)} chunks created"
            )

            # Trigger rerun to update sidebar with new document
            # This ensures the UI reflects the current state
            st.rerun()

        except Exception as e:
            # Handle errors gracefully with user-friendly message
            st.error(f"Error processing file: {str(e)}")


def _process_text_input(
    title: str,
    text: str,
    vector_store: VectorStoreService,
    doc_processor: DocumentProcessor,
    chunker: TextChunker,
) -> None:
    """
    Process pasted text through the same pipeline as uploaded files.

    This method handles text that users paste directly (typically job postings
    copied from job boards). It goes through the same processing pipeline as
    file uploads: text cleaning, chunking, and storage in the vector database.

    Args:
        title: Job title/name (used as filename in vector store).
        text: Pasted job posting text content.
        vector_store: VectorStoreService for storing chunks.
        doc_processor: DocumentProcessor for text processing.
        chunker: TextChunker for splitting documents.
    """
    with st.spinner("Processing job posting..."):
        try:
            # Process pasted text as a document
            # This applies the same cleaning and normalization as file processing
            processed_doc = doc_processor.process_text(
                text=text,
                title=title,
                doc_type="job_posting",
            )

            # Chunk the document for embedding
            chunks = chunker.chunk_document(processed_doc)

            # Store in vector database
            vector_store.add_chunks(chunks)

            # Show success feedback
            st.success(
                f"✅ Successfully indexed **{title}**\n\n"
                f"- {processed_doc.word_count} words\n"
                f"- {len(chunks)} chunks created"
            )

            # Refresh UI to show new document in sidebar
            st.rerun()

        except Exception as e:
            st.error(f"Error processing text: {str(e)}")
