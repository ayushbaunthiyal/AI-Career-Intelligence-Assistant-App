"""Sidebar component for document management."""

import streamlit as st

from app.services.vector_store import VectorStoreService


def render_sidebar(vector_store: VectorStoreService) -> None:
    """Render the sidebar with document management.

    Args:
        vector_store: VectorStoreService instance.
    """
    with st.sidebar:
        st.title("📁 Your Documents")
        st.divider()

        # Get document stats
        stats = vector_store.get_document_stats()

        # Resume section
        st.subheader("📄 Resume")
        if stats["resume_filename"]:
            st.success(f"✅ {stats['resume_filename']}")
            st.caption(f"{stats['resume_chunks']} chunks indexed")

            if st.button("🗑️ Remove Resume", key="remove_resume"):
                vector_store.delete_by_doc_type("resume")
                st.session_state.pop("resume_uploaded", None)
                st.rerun()
        else:
            st.info("No resume uploaded yet")

        st.divider()

        # Job postings section
        st.subheader("💼 Job Postings")
        if stats["job_postings"]:
            for i, job_filename in enumerate(stats["job_postings"], 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**Job #{i}:** {job_filename}")
                with col2:
                    if st.button("🗑️", key=f"remove_job_{i}"):
                        vector_store.delete_job_posting(job_filename)
                        st.rerun()

            st.caption(f"Total: {stats['job_posting_chunks']} chunks indexed")
        else:
            st.info("No job postings uploaded yet")

        st.divider()

        # Quick stats
        st.subheader("📊 Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", stats["total_chunks"])
        with col2:
            st.metric("Job Postings", len(stats["job_postings"]))

        st.divider()

        # Actions
        if st.button("🗑️ Clear All Documents", type="secondary"):
            vector_store.clear_all()
            # Clear session state
            for key in ["resume_uploaded", "messages", "chat_history"]:
                st.session_state.pop(key, None)
            st.rerun()

        # Clear chat button
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            if "rag_service" in st.session_state:
                st.session_state.rag_service.clear_memory()
            st.rerun()
