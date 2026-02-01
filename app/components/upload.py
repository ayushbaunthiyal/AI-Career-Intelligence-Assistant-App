"""File upload component for resumes and job postings."""

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
    """Render the document upload section.

    Args:
        vector_store: VectorStoreService instance.
        doc_processor: DocumentProcessor instance.
        chunker: TextChunker instance.
    """
    settings = get_settings()
    stats = vector_store.get_document_stats()

    st.subheader("📤 Upload Documents")

    # Create tabs for different upload methods
    tab1, tab2 = st.tabs(["📁 Upload File", "📝 Paste Text"])

    with tab1:
        _render_file_upload(
            vector_store, doc_processor, chunker, stats, settings
        )

    with tab2:
        _render_text_input(
            vector_store, doc_processor, chunker, stats
        )


def _render_file_upload(
    vector_store: VectorStoreService,
    doc_processor: DocumentProcessor,
    chunker: TextChunker,
    stats: dict,
    settings,
) -> None:
    """Render file upload interface."""
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "doc", "txt"],
            help=f"Max file size: {settings.max_file_size_mb}MB",
        )

    with col2:
        doc_type = st.radio(
            "Document Type",
            options=["resume", "job_posting"],
            format_func=lambda x: "📄 Resume" if x == "resume" else "💼 Job Posting",
            horizontal=False,
        )

    if uploaded_file is not None:
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            st.error(f"File too large ({file_size_mb:.1f}MB). Maximum size: {settings.max_file_size_mb}MB")
            return

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
    """Render text paste interface for job postings."""
    st.info("💡 Paste job posting text directly - great for copying from job boards!")

    job_title = st.text_input(
        "Job Title",
        placeholder="e.g., Senior ML Engineer @ Google",
    )

    job_text = st.text_area(
        "Job Posting Content",
        height=200,
        placeholder="Paste the full job description here...",
    )

    if st.button("📥 Add Job Posting", type="primary", disabled=not (job_title and job_text)):
        _process_text_input(
            job_title,
            job_text,
            vector_store,
            doc_processor,
            chunker,
        )


def _process_uploaded_file(
    uploaded_file,
    doc_type: Literal["resume", "job_posting"],
    vector_store: VectorStoreService,
    doc_processor: DocumentProcessor,
    chunker: TextChunker,
    stats: dict,
) -> None:
    """Process and index an uploaded file."""
    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            # If uploading a new resume, delete the old one first
            if doc_type == "resume" and stats["resume_filename"]:
                vector_store.delete_by_doc_type("resume")
                st.info(f"Replaced previous resume: {stats['resume_filename']}")

            # Process the document
            processed_doc = doc_processor.process(
                file=uploaded_file,
                filename=uploaded_file.name,
                doc_type=doc_type,
            )

            # Chunk the document
            chunks = chunker.chunk_document(processed_doc)

            # Add to vector store
            vector_store.add_chunks(chunks)

            st.success(
                f"✅ Successfully indexed **{uploaded_file.name}**\n\n"
                f"- {processed_doc.word_count} words\n"
                f"- {len(chunks)} chunks created"
            )

            # Trigger a rerun to update the sidebar
            st.rerun()

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def _process_text_input(
    title: str,
    text: str,
    vector_store: VectorStoreService,
    doc_processor: DocumentProcessor,
    chunker: TextChunker,
) -> None:
    """Process and index pasted text."""
    with st.spinner("Processing job posting..."):
        try:
            # Process as text
            processed_doc = doc_processor.process_text(
                text=text,
                title=title,
                doc_type="job_posting",
            )

            # Chunk the document
            chunks = chunker.chunk_document(processed_doc)

            # Add to vector store
            vector_store.add_chunks(chunks)

            st.success(
                f"✅ Successfully indexed **{title}**\n\n"
                f"- {processed_doc.word_count} words\n"
                f"- {len(chunks)} chunks created"
            )

            # Trigger a rerun to update the sidebar
            st.rerun()

        except Exception as e:
            st.error(f"Error processing text: {str(e)}")
