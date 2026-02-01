"""
Chat Interface Component for Career Intelligence Assistant.

This module handles the conversational UI where users ask questions about
their career fit. It provides:
- Chat message display with conversation history
- Streaming responses for real-time feedback
- Source document citations for transparency
- Suggested queries to help users get started
- Welcome message when no documents are uploaded

The chat interface integrates with the RAG service to generate intelligent
responses based on uploaded resume and job postings.
"""

import streamlit as st

from app.services.rag_chain import RAGService
from app.services.vector_store import VectorStoreService

# =============================================================================
# SUGGESTED QUERIES
# =============================================================================
# Pre-defined questions to help users understand what they can ask.
# These are shown as clickable buttons when the chat is empty, providing
# examples of useful queries and reducing the barrier to getting started.
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
    """
    Render the main chat interface for user interactions.

    This is the primary user interaction component. It:
    1. Checks if documents are uploaded (shows welcome if not)
    2. Displays conversation history
    3. Shows suggested queries for new users
    4. Handles user input and generates AI responses
    5. Displays source citations for transparency

    The chat interface uses Streamlit's chat_message components for a
    modern conversational UI similar to ChatGPT.

    Args:
        rag_service: RAGService instance for generating responses.
        vector_store: VectorStoreService instance for document statistics.
    """
    stats = vector_store.get_document_stats()

    # Check if user has uploaded any documents
    # If not, show welcome message instead of chat interface
    has_documents = stats["resume_filename"] or stats["job_postings"]

    if not has_documents:
        _render_welcome_message()
        return

    # Show current document context in a collapsible section
    # This helps users understand what documents the AI is analyzing
    _render_document_context(stats)

    # Initialize chat history in session state if not already present
    # Chat history persists across Streamlit reruns within the same session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all previous messages in the conversation
    # This creates the chat history view users see
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show source citations for assistant messages
            # This transparency helps users understand which documents
            # informed the AI's response
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.caption(
                            f"**{source['doc_type'].replace('_', ' ').title()}**: "
                            f"{source['filename']}"
                        )
                        st.text(source["content"])

    # Show suggested queries only when chat is empty
    # Once user starts chatting, suggestions are hidden to reduce clutter
    if not st.session_state.messages:
        _render_suggested_queries()

    # Chat input field - appears at the bottom
    # When user types and presses Enter, the prompt is processed
    if prompt := st.chat_input("Ask about your career fit..."):
        _handle_user_message(prompt, rag_service)


# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================

def _render_welcome_message() -> None:
    """
    Render welcome message when no documents are uploaded.

    This provides onboarding guidance for new users, explaining:
    - What the assistant can do
    - How to get started (upload resume and job postings)
    - What types of questions they can ask

    The welcome message is replaced by the chat interface once documents
    are uploaded.
    """
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
    """
    Display current document context in a collapsible section.

    This shows users which documents are currently indexed and available
    for analysis. It helps users understand what the AI can analyze and
    provides quick reference for job posting numbers (Job #1, Job #2, etc.).

    Args:
        stats: Dictionary containing document statistics from vector store.
    """
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
                # List job postings with numbers for easy reference
                # Users can reference "Job #1" in their questions
                for i, job in enumerate(stats["job_postings"], 1):
                    st.caption(f"  #{i}: {job}")
            else:
                st.write("**Job Postings:** None")


def _render_suggested_queries() -> None:
    """
    Render suggested query buttons to help users get started.

    These buttons provide example questions that users can click to
    immediately ask. This reduces the "blank page" problem and helps
    users understand the types of questions the assistant can answer.

    When a button is clicked, the query is stored in session state
    and processed on the next rerun.
    """
    st.markdown("### 💡 Try asking:")

    # Create a 2-column grid for the suggestion buttons
    # This provides a clean, organized layout
    cols = st.columns(2)
    for i, query in enumerate(SUGGESTED_QUERIES):
        with cols[i % 2]:
            if st.button(query, key=f"suggestion_{i}", use_container_width=True):
                # Store query in session state for processing on rerun
                # This pattern allows Streamlit to handle the button click
                # and then process the query in the next render cycle
                st.session_state.pending_query = query
                st.rerun()


# =============================================================================
# MESSAGE HANDLING
# =============================================================================

def _handle_user_message(prompt: str, rag_service: RAGService) -> None:
    """
    Process a user's question and generate an AI response.

    This function orchestrates the entire question-answering flow:
    1. Adds user message to chat history
    2. Displays user message in the chat UI
    3. Streams AI response token-by-token for real-time feedback
    4. Retrieves and displays source documents
    5. Saves assistant response to chat history

    The streaming approach provides better UX - users see the response
    forming in real-time rather than waiting for the complete answer.

    Args:
        prompt: User's question about their career fit.
        rag_service: RAGService instance for generating responses.
    """
    # Add user message to chat history immediately
    # This ensures the message appears even if response generation fails
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in the chat interface
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        # Use empty placeholder for streaming response
        # We update this placeholder as each token arrives
        message_placeholder = st.empty()
        full_response = ""
        sources = []

        try:
            # Stream the response token-by-token
            # The spinner shows "Thinking..." while waiting for first token
            with st.spinner("Thinking..."):
                response_generator = rag_service.ask_stream(prompt)

                # Display each chunk as it arrives
                # The "▌" cursor indicates more text is coming
                for chunk in response_generator:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                # Remove cursor and show final response
                message_placeholder.markdown(full_response)

            # Get source documents that informed this response
            # This uses the same retrieval that was used for the response,
            # so there's no extra LLM call - just formatting the sources
            sources = rag_service.get_sources_for_last_query(prompt)

            # Display source citations in an expandable section
            # Users can click to see which documents were used
            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.caption(
                            f"**{source['doc_type'].replace('_', ' ').title()}**: "
                            f"{source['filename']}"
                        )
                        st.text(source["content"])

        except Exception as e:
            # Handle errors gracefully with user-friendly message
            # This prevents the app from crashing on API errors or network issues
            full_response = f"I encountered an error: {str(e)}. Please try again."
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        # This includes sources so they're preserved in the conversation view
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        })


def handle_pending_query(rag_service: RAGService) -> None:
    """
    Handle pending query from suggested query buttons.

    When a user clicks a suggested query button, the query is stored in
    session state as "pending_query". This function processes it on the
    next render cycle, allowing Streamlit's button click handling to work
    correctly.

    Args:
        rag_service: RAGService instance for generating responses.
    """
    if "pending_query" in st.session_state:
        # Pop the query from session state (removes it after reading)
        # Then process it as if the user typed it
        query = st.session_state.pop("pending_query")
        _handle_user_message(query, rag_service)
