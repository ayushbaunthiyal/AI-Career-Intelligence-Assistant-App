"""Chat interface component."""

import streamlit as st

from app.services.rag_chain import RAGService
from app.services.vector_store import VectorStoreService

# Suggested queries for users
SUGGESTED_QUERIES = [
    "What skills am I missing for Job #1?",
    "How does my experience align with the job requirements?",
    "Compare my fit across all job postings",
    "What are my strongest qualifications for this role?",
    "Help me prepare for an interview",
    "What should I highlight in my cover letter?",
]


def render_chat_interface(
    rag_service: RAGService,
    vector_store: VectorStoreService,
) -> None:
    """Render the chat interface.

    Args:
        rag_service: RAGService instance.
        vector_store: VectorStoreService instance.
    """
    stats = vector_store.get_document_stats()

    # Check if documents are uploaded
    has_documents = stats["resume_filename"] or stats["job_postings"]

    if not has_documents:
        _render_welcome_message()
        return

    # Show document context
    _render_document_context(stats)

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.caption(
                            f"**{source['doc_type'].replace('_', ' ').title()}**: "
                            f"{source['filename']}"
                        )
                        st.text(source["content"])

    # Suggested queries (only show if no messages yet)
    if not st.session_state.messages:
        _render_suggested_queries()

    # Chat input
    if prompt := st.chat_input("Ask about your career fit..."):
        _handle_user_message(prompt, rag_service)


def _render_welcome_message() -> None:
    """Render welcome message when no documents are uploaded."""
    st.markdown("""
    ## 👋 Welcome to Career Intelligence Assistant!

    I help you analyze your resume against job postings to understand:
    - **Skill gaps** - What skills are you missing?
    - **Experience alignment** - How well does your background fit?
    - **Interview prep** - What should you prepare for?

    ### Get Started
    1. 📄 **Upload your resume** (PDF or DOCX)
    2. 💼 **Add job postings** you're interested in
    3. 💬 **Ask questions** about your fit!

    Use the upload section above to add your documents.
    """)


def _render_document_context(stats: dict) -> None:
    """Show current document context."""
    with st.expander("📋 Current Documents", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if stats["resume_filename"]:
                st.write(f"**Resume:** {stats['resume_filename']}")
            else:
                st.write("**Resume:** Not uploaded")

        with col2:
            if stats["job_postings"]:
                st.write(f"**Job Postings:** {len(stats['job_postings'])}")
                for i, job in enumerate(stats["job_postings"], 1):
                    st.caption(f"  #{i}: {job}")
            else:
                st.write("**Job Postings:** None")


def _render_suggested_queries() -> None:
    """Render suggested query buttons."""
    st.markdown("### 💡 Try asking:")

    # Create a grid of suggestion buttons
    cols = st.columns(2)
    for i, query in enumerate(SUGGESTED_QUERIES):
        with cols[i % 2]:
            if st.button(query, key=f"suggestion_{i}", use_container_width=True):
                # Trigger the query
                st.session_state.pending_query = query
                st.rerun()


def _handle_user_message(prompt: str, rag_service: RAGService) -> None:
    """Handle user message and generate response.

    Args:
        prompt: User's question.
        rag_service: RAGService instance.
    """
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []

        try:
            # Stream the response
            with st.spinner("Thinking..."):
                response_generator = rag_service.ask_stream(prompt)

                for chunk in response_generator:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

            # Get sources for the query (uses cached retrieval, no extra LLM call)
            sources = rag_service.get_sources_for_last_query(prompt)

            # Show sources
            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.caption(
                            f"**{source['doc_type'].replace('_', ' ').title()}**: "
                            f"{source['filename']}"
                        )
                        st.text(source["content"])

        except Exception as e:
            full_response = f"I encountered an error: {str(e)}. Please try again."
            message_placeholder.markdown(full_response)

        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        })


def handle_pending_query(rag_service: RAGService) -> None:
    """Handle any pending query from suggested buttons.

    Args:
        rag_service: RAGService instance.
    """
    if "pending_query" in st.session_state:
        query = st.session_state.pop("pending_query")
        _handle_user_message(query, rag_service)
