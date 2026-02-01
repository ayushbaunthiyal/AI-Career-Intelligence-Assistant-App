"""
Main Streamlit Application Entry Point for Career Intelligence Assistant.

This module is the entry point for the Streamlit web application. It:
1. Configures the Streamlit page (title, layout, styling)
2. Initializes session state for maintaining user data across interactions
3. Validates configuration (API keys, settings)
4. Initializes services (vector store, document processor, RAG service)
5. Renders the UI components (sidebar, upload section, chat interface)

The application follows a component-based architecture where UI rendering
is separated into dedicated component modules for maintainability.
"""

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

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Configure application-wide logging to help with debugging and monitoring.
# Logs include timestamp, module name, log level, and message for easy tracing.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def init_session_state() -> None:
    """
    Initialize Streamlit session state variables.

    Streamlit's session state persists data across reruns (user interactions).
    We use it to:
    - Track initialization status
    - Store chat messages for conversation history
    - Cache service instances (avoid recreating on every rerun)

    This function is idempotent - safe to call multiple times.
    """
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.messages = []  # Chat message history
        logger.info("Session state initialized")


# =============================================================================
# SERVICE INITIALIZATION
# =============================================================================

def get_services():
    """
    Get or create service instances (singleton pattern per session).

    Services are cached in session state to:
    1. Avoid reinitializing expensive objects (ChromaDB, embeddings) on every rerun
    2. Maintain conversation history in RAGService across interactions
    3. Preserve vector store state (uploaded documents) across page interactions

    This is critical for performance - recreating ChromaDB connections and
    embedding models on every user interaction would be very slow.

    Returns:
        Tuple of (vector_store, doc_processor, chunker, rag_service) instances.
    """
    # Initialize vector store if not already created
    # This connects to ChromaDB and sets up embeddings - expensive operation
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStoreService()

    # Initialize document processor (lightweight, but cached for consistency)
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()

    # Initialize text chunker (lightweight, but cached for consistency)
    if "chunker" not in st.session_state:
        st.session_state.chunker = TextChunker()

    # Initialize RAG service (requires vector_store, maintains chat history)
    if "rag_service" not in st.session_state:
        st.session_state.rag_service = RAGService(st.session_state.vector_store)

    return (
        st.session_state.vector_store,
        st.session_state.doc_processor,
        st.session_state.chunker,
        st.session_state.rag_service,
    )


# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Main application entry point - orchestrates the entire Streamlit app.

    This function:
    1. Configures the Streamlit page (title, layout, styling)
    2. Initializes session state for user data persistence
    3. Validates configuration (ensures API keys are set)
    4. Initializes all services (vector store, processors, RAG)
    5. Renders the UI (sidebar, upload section, chat interface)

    The app follows a top-down rendering approach where components are
    rendered in order, with the sidebar on the left and main content on the right.
    """
    # =====================================================================
    # STEP 1: PAGE CONFIGURATION
    # =====================================================================
    # Configure Streamlit page appearance and layout
    # Wide layout provides more horizontal space for the two-column design
    st.set_page_config(
        page_title="Career Intelligence Assistant",
        page_icon="🎯",
        layout="wide",  # Wide layout for sidebar + main content
        initial_sidebar_state="expanded",  # Sidebar open by default
    )

    # =====================================================================
    # STEP 2: CUSTOM STYLING
    # =====================================================================
    # Apply custom CSS for better visual appearance
    # This improves the look and feel beyond Streamlit's default styling
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

    # =====================================================================
    # STEP 3: INITIALIZE SESSION STATE
    # =====================================================================
    # Set up session state for maintaining user data across interactions
    init_session_state()

    # =====================================================================
    # STEP 4: VALIDATE CONFIGURATION
    # =====================================================================
    # Check that required configuration (especially API keys) is present
    # We stop the app early with a clear error message if configuration is invalid
    try:
        settings = get_settings()
        # Check if API key is set (not empty and not the placeholder value)
        if not settings.openai_api_key or settings.openai_api_key == "your-openai-api-key-here":
            st.error(
                "⚠️ OpenAI API key not configured!\n\n"
                "Please set your `OPENAI_API_KEY` in the `.env` file or as an environment variable."
            )
            st.stop()  # Stop app execution - can't proceed without API key
    except Exception as e:
        st.error(f"⚠️ Configuration error: {str(e)}")
        st.stop()

    # =====================================================================
    # STEP 5: INITIALIZE SERVICES
    # =====================================================================
    # Get or create service instances (cached in session state)
    # These services handle all business logic (document processing, RAG, etc.)
    try:
        vector_store, doc_processor, chunker, rag_service = get_services()
    except Exception as e:
        st.error(f"⚠️ Failed to initialize services: {str(e)}")
        logger.exception("Service initialization failed")
        st.stop()  # Stop app if services can't be initialized

    # =====================================================================
    # STEP 6: RENDER UI COMPONENTS
    # =====================================================================
    # Render sidebar with document management and statistics
    render_sidebar(vector_store)

    # Main content area header
    st.title("🎯 Career Intelligence Assistant")
    st.caption("Analyze your resume against job postings to discover skill gaps, alignment, and interview tips.")

    # Upload section (collapsible expander)
    # Users can upload resumes and job postings here
    with st.container():
        with st.expander("📤 Upload Documents", expanded=True):
            render_upload_section(vector_store, doc_processor, chunker)

    st.divider()

    # Chat section - main interaction area
    # Users ask questions about their career fit here
    with st.container():
        # Handle any pending queries from suggested query buttons
        # (when user clicks a suggested question, it's processed here)
        handle_pending_query(rag_service)

        # Render the main chat interface
        render_chat_interface(rag_service, vector_store)


if __name__ == "__main__":
    main()
