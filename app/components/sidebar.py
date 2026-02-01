"""
Sidebar Component for Document Management and Statistics.

This module renders the left sidebar that shows:
- Currently uploaded resume (with remove option)
- List of uploaded job postings (with individual remove options)
- Document statistics (total chunks, job posting count)
- Action buttons (clear all documents, clear chat history)

The sidebar provides a persistent view of the application state, helping
users understand what documents are available for analysis.
"""

import streamlit as st

from app.services.vector_store import VectorStoreService


def render_sidebar(vector_store: VectorStoreService) -> None:
    """
    Render the sidebar with document management and statistics.

    The sidebar provides:
    1. Resume status and removal option
    2. Job postings list with individual removal
    3. Quick statistics about indexed documents
    4. Global actions (clear all, clear chat)

    This gives users a clear view of their document library and provides
    easy access to management functions.

    Args:
        vector_store: VectorStoreService instance for querying document stats.
    """
    with st.sidebar:
        st.title("📁 Your Documents")
        st.divider()

        # Get current document statistics from vector store
        # This provides real-time view of what's indexed
        stats = vector_store.get_document_stats()

        # =====================================================================
        # RESUME SECTION
        # =====================================================================
        st.subheader("📄 Resume")
        if stats["resume_filename"]:
            # Show uploaded resume with success indicator
            st.success(f"✅ {stats['resume_filename']}")
            st.caption(f"{stats['resume_chunks']} chunks indexed")

            # Remove button - deletes all resume chunks from vector store
            if st.button("🗑️ Remove Resume", key="remove_resume"):
                vector_store.delete_by_doc_type("resume")
                # Clear any resume-related session state
                st.session_state.pop("resume_uploaded", None)
                st.rerun()  # Refresh UI to reflect removal
        else:
            st.info("No resume uploaded yet")

        st.divider()

        # =====================================================================
        # JOB POSTINGS SECTION
        # =====================================================================
        st.subheader("💼 Job Postings")
        if stats["job_postings"]:
            # List all job postings with numbered labels (Job #1, Job #2, etc.)
            # This numbering helps users reference specific jobs in questions
            for i, job_filename in enumerate(stats["job_postings"], 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**Job #{i}:** {job_filename}")
                with col2:
                    # Individual remove button for each job posting
                    if st.button("🗑️", key=f"remove_job_{i}"):
                        vector_store.delete_job_posting(job_filename)
                        st.rerun()  # Refresh to show updated list

            st.caption(f"Total: {stats['job_posting_chunks']} chunks indexed")
        else:
            st.info("No job postings uploaded yet")

        st.divider()

        # =====================================================================
        # STATISTICS SECTION
        # =====================================================================
        st.subheader("📊 Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", stats["total_chunks"])
        with col2:
            st.metric("Job Postings", len(stats["job_postings"]))

        st.divider()

        # =====================================================================
        # ACTION BUTTONS
        # =====================================================================
        # Clear all documents - removes everything from vector store
        # This is useful for starting fresh or testing
        if st.button("🗑️ Clear All Documents", type="secondary"):
            vector_store.clear_all()
            # Clear related session state to reset the application
            for key in ["resume_uploaded", "messages", "chat_history"]:
                st.session_state.pop(key, None)
            st.rerun()

        # Clear chat history - removes conversation but keeps documents
        # This allows users to start a new conversation with the same documents
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            # Also clear RAG service's internal chat history
            if "rag_service" in st.session_state:
                st.session_state.rag_service.clear_memory()
            st.rerun()
