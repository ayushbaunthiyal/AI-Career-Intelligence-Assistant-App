"""Main Streamlit application for Career Intelligence Assistant."""

import logging

import streamlit as st

from app.components.chat import handle_pending_query, render_chat_interface
from app.components.sidebar import render_sidebar
from app.components.upload import render_upload_section
from app.config import get_settings
from app.services.chunking import TextChunker
from app.services.document_processor import DocumentProcessor
from app.services.rag_chain import RAGService
from app.services.vector_store import VectorStoreService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """Initialize session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.messages = []
        logger.info("Session state initialized")


def get_services():
    """Get or create service instances.

    Returns:
        Tuple of (vector_store, doc_processor, chunker, rag_service).
    """
    # Cache services in session state for persistence
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStoreService()

    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()

    if "chunker" not in st.session_state:
        st.session_state.chunker = TextChunker()

    if "rag_service" not in st.session_state:
        st.session_state.rag_service = RAGService(st.session_state.vector_store)

    return (
        st.session_state.vector_store,
        st.session_state.doc_processor,
        st.session_state.chunker,
        st.session_state.rag_service,
    )


def main() -> None:
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Career Intelligence Assistant",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .stApp {
            max-width: 1400px;
            margin: 0 auto;
        }
        .stChatMessage {
            padding: 1rem;
        }
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Validate configuration
    try:
        settings = get_settings()
        if not settings.openai_api_key or settings.openai_api_key == "your-openai-api-key-here":
            st.error(
                "⚠️ OpenAI API key not configured!\n\n"
                "Please set your `OPENAI_API_KEY` in the `.env` file or as an environment variable."
            )
            st.stop()
    except Exception as e:
        st.error(f"⚠️ Configuration error: {str(e)}")
        st.stop()

    # Get services
    try:
        vector_store, doc_processor, chunker, rag_service = get_services()
    except Exception as e:
        st.error(f"⚠️ Failed to initialize services: {str(e)}")
        logger.exception("Service initialization failed")
        st.stop()

    # Render sidebar
    render_sidebar(vector_store)

    # Main content area
    st.title("🎯 Career Intelligence Assistant")
    st.caption("Analyze your resume against job postings to discover skill gaps, alignment, and interview tips.")

    # Create two main sections
    with st.container():
        # Upload section (collapsible)
        with st.expander("📤 Upload Documents", expanded=True):
            render_upload_section(vector_store, doc_processor, chunker)

    st.divider()

    # Chat section
    with st.container():
        # Handle any pending queries from suggested buttons
        handle_pending_query(rag_service)

        # Render chat interface
        render_chat_interface(rag_service, vector_store)


if __name__ == "__main__":
    main()
