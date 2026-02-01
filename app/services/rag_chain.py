"""
RAG (Retrieval-Augmented Generation) Chain Service for Career Intelligence Assistant.

This module implements the core RAG pipeline that enables the AI assistant to answer
questions about resume-job fit by:
1. Retrieving relevant document chunks from the vector database
2. Formatting them as context for the LLM
3. Generating intelligent responses based on the retrieved context

The service includes smart query detection to handle comparison queries differently
from specific questions, ensuring fair analysis across all job postings.
"""

import logging
import re
from collections.abc import Generator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.prompts.templates import SYSTEM_PROMPT
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

# =============================================================================
# COMPARISON QUERY DETECTION KEYWORDS
# =============================================================================
# When users ask questions like "which job is best for me?" or "compare all jobs",
# we need to retrieve ALL job postings instead of just the top-k semantically
# similar chunks. This ensures fair comparison across all uploaded jobs.
# These keywords help us detect such comparison queries automatically.
COMPARISON_KEYWORDS = [
    "which job", "best job", "best fit", "best match", "compare", "comparison",
    "all jobs", "all job", "all three", "all 3", "every job", "each job",
    "best suited", "most suitable", "recommend", "should i apply",
    "which role", "which position", "rank", "ranking",
]


class RAGService:
    """
    RAG Service for Career Intelligence Question Answering.

    This service orchestrates the entire RAG pipeline:
    - Detects query intent (comparison vs specific question)
    - Retrieves relevant documents from vector store
    - Formats context for LLM consumption
    - Generates responses with source attribution
    - Maintains conversation history for multi-turn dialogues
    """

    def __init__(self, vector_store: VectorStoreService):
        """
        Initialize the RAG service with vector store and LLM configuration.

        We initialize the OpenAI LLM here with streaming enabled for better UX.
        Chat history is maintained in-memory for the session to enable contextual
        conversations where follow-up questions can reference previous answers.

        Args:
            vector_store: VectorStoreService instance for document retrieval.
        """
        self.settings = get_settings()
        self.vector_store = vector_store

        # Initialize the LLM with streaming enabled for real-time response display
        # Temperature 0.7 provides a balance between creativity and consistency
        self.llm = ChatOpenAI(
            model=self.settings.openai_llm_model,
            temperature=0.7,
            openai_api_key=self.settings.openai_api_key,
            streaming=True,
        )

        # Chat history storage for maintaining conversation context
        # We store both user questions and AI responses to enable multi-turn dialogues
        self.chat_history: list[HumanMessage | AIMessage] = []
        self.max_history = self.settings.max_chat_history

        logger.info(f"Initialized RAGService with model: {self.settings.openai_llm_model}")

    # =============================================================================
    # QUERY INTENT DETECTION
    # =============================================================================

    def _is_comparison_query(self, question: str) -> bool:
        """
        Detect if the user wants to compare across ALL job postings.

        This is crucial for accuracy: when users ask "which job is best for me?",
        we must retrieve ALL jobs, not just the top-k similar chunks. Otherwise,
        a job that's semantically different but actually the best fit might be
        missed if it doesn't rank high in similarity search.

        Args:
            question: User's question.

        Returns:
            True if this is a comparison query requiring all documents.
        """
        question_lower = question.lower()
        for keyword in COMPARISON_KEYWORDS:
            if keyword in question_lower:
                return True
        return False

    # =============================================================================
    # DOCUMENT RETRIEVAL STRATEGIES
    # =============================================================================

    def _get_all_relevant_docs(self):
        """
        Retrieve ALL documents from the vector store for comprehensive comparison.

        This method is used when the user asks comparison questions. Instead of
        using semantic similarity (which might miss relevant jobs), we fetch
        everything to ensure fair analysis. We deduplicate by content hash to
        avoid processing identical chunks multiple times.

        Returns:
            List of all unique document chunks from the vector store.
        """
        all_docs = self.vector_store.get_all_documents()

        # Deduplicate by content hash to avoid processing identical chunks
        # We use first 100 characters as a simple hash since identical chunks
        # will have the same start
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hash(doc.page_content[:100])  # Use first 100 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        logger.info(f"Retrieved {len(unique_docs)} unique document chunks for comparison")
        return unique_docs

    # =============================================================================
    # CONTEXT FORMATTING
    # =============================================================================

    def _format_docs(self, docs) -> str:
        """
        Format retrieved document chunks into a structured context string for the LLM.

        We prefix each chunk with its document type and filename so the LLM knows
        which document it's reading from. This enables accurate source attribution
        and helps the LLM distinguish between resume content and different job postings.

        Args:
            docs: List of retrieved document chunks.

        Returns:
            Formatted context string with document metadata prefixes.
        """
        formatted = []
        for doc in docs:
            doc_type = doc.metadata.get("doc_type", "document").upper()
            filename = doc.metadata.get("filename", "Unknown")
            formatted.append(f"[{doc_type}: {filename}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def _get_chat_history_text(self) -> str:
        """
        Format chat history for inclusion in LLM context.

        We include recent conversation history so the LLM can understand follow-up
        questions. We limit to last N turns (max_history * 2 because each turn
        has both user and assistant messages) to prevent token bloat. Long AI
        responses are truncated to 500 chars to save tokens while preserving context.

        Returns:
            Formatted chat history string for prompt inclusion.
        """
        if not self.chat_history:
            return "No previous conversation."

        history_parts = []
        # Last N turns (multiply by 2 because each turn has user + assistant message)
        for msg in self.chat_history[-self.max_history * 2:]:
            if isinstance(msg, HumanMessage):
                history_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                # Truncate long responses to save tokens while keeping context
                content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                history_parts.append(f"Assistant: {content}")

        return "\n".join(history_parts)

    # =============================================================================
    # MAIN QUERY PROCESSING METHODS
    # =============================================================================

    def ask(self, question: str) -> dict:
        """
        Process a user question and return a complete response with sources.

        This is the main entry point for non-streaming queries. The method:
        1. Detects query intent (comparison vs specific question)
        2. Retrieves relevant documents using appropriate strategy
        3. Formats context and chat history
        4. Invokes LLM to generate response
        5. Updates conversation history
        6. Extracts source documents for attribution

        Args:
            question: User's question about their career fit.

        Returns:
            Dictionary with 'answer' (LLM response) and 'sources' (list of source docs).
        """
        try:
            # =====================================================================
            # STEP 1: DETECT QUERY INTENT AND RETRIEVE DOCUMENTS
            # =====================================================================
            # Check if this is a comparison query requiring ALL documents
            is_comparison = self._is_comparison_query(question)

            if is_comparison:
                # For comparison queries, retrieve ALL documents to ensure fair analysis
                # This prevents missing relevant jobs that might not rank high in similarity
                docs = self._get_all_relevant_docs()
                logger.info("[MODE] Comparison query detected - using ALL documents")
            else:
                # For specific questions, use semantic similarity search
                # We use k=10 (instead of default 5) for better coverage
                retriever = self.vector_store.get_retriever(k=10)
                docs = retriever.invoke(question)

            # =====================================================================
            # STEP 2: FORMAT CONTEXT FOR LLM
            # =====================================================================
            # Format retrieved chunks with metadata so LLM knows source documents
            context = self._format_docs(docs)
            # Include recent conversation history for contextual understanding
            chat_history = self._get_chat_history_text()

            # =====================================================================
            # STEP 3: BUILD PROMPT AND INVOKE LLM
            # =====================================================================
            # Create the prompt template with system instructions and user question
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", """Use the following context from the candidate's resume and job postings to answer the question. If you don't have enough information to answer accurately, say so.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION: {question}

Provide a helpful, specific answer based on the context provided. Reference specific details from the documents when relevant."""),
            ])

            # Create the LangChain chain: prompt -> LLM -> string parser
            chain = prompt | self.llm | StrOutputParser()

            # Invoke the chain to get the LLM response
            response = chain.invoke({
                "context": context,
                "chat_history": chat_history,
                "question": question,
            })

            # =====================================================================
            # STEP 4: LOGGING AND HISTORY MANAGEMENT
            # =====================================================================
            # Log query and response for observability and debugging
            logger.info("=" * 60)
            logger.info(f"[QUERY] {question}")
            logger.info(f"[SOURCES] {[doc.metadata.get('filename') for doc in docs]}")
            logger.info(f"[RESPONSE] {response[:500]}{'...' if len(response) > 500 else ''}")
            logger.info("=" * 60)

            # Update chat history to maintain conversation context
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=response))

            # Trim history if it exceeds max length to prevent token bloat
            if len(self.chat_history) > self.max_history * 2:
                self.chat_history = self.chat_history[-self.max_history * 2:]

            # =====================================================================
            # STEP 5: EXTRACT SOURCES FOR ATTRIBUTION
            # =====================================================================
            # Extract source document information for display to user
            # This helps users understand which documents informed the answer
            sources = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "doc_type": doc.metadata.get("doc_type"),
                    "filename": doc.metadata.get("filename"),
                }
                for doc in docs
            ]

            return {
                "answer": response,
                "sources": sources,
            }

        except Exception as e:
            logger.error(f"Error in RAG chain: {e}")
            raise

    def ask_stream(self, question: str) -> Generator[str]:
        """
        Process a user question and stream the response token-by-token.

        This method provides the same functionality as ask() but streams the response
        as it's generated. This improves perceived performance - users see the answer
        forming in real-time rather than waiting for the complete response.

        The streaming approach is especially important for longer responses where
        users might wait several seconds otherwise.

        Args:
            question: User's question about their career fit.

        Yields:
            Response tokens as they're generated by the LLM.
        """
        try:
            # Same retrieval logic as ask() method
            is_comparison = self._is_comparison_query(question)

            if is_comparison:
                docs = self._get_all_relevant_docs()
                logger.info("[MODE] Comparison query detected - using ALL documents")
            else:
                retriever = self.vector_store.get_retriever(k=10)
                docs = retriever.invoke(question)

            # Format context for LLM
            context = self._format_docs(docs)
            chat_history = self._get_chat_history_text()

            # Build prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", """Use the following context from the candidate's resume and job postings to answer the question. If you don't have enough information to answer accurately, say so.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION: {question}

Provide a helpful, specific answer based on the context provided. Reference specific details from the documents when relevant."""),
            ])

            # Create the chain
            chain = prompt | self.llm | StrOutputParser()

            # Log query start for observability
            logger.info("=" * 60)
            logger.info(f"[QUERY] {question}")
            logger.info(f"[SOURCES] {[doc.metadata.get('filename') for doc in docs]}")

            # Stream the response token-by-token
            # We accumulate the full response to update chat history after streaming completes
            full_response = ""
            for chunk in chain.stream({
                "context": context,
                "chat_history": chat_history,
                "question": question,
            }):
                full_response += chunk
                yield chunk  # Yield each chunk immediately for real-time display

            # Log complete response
            logger.info(f"[RESPONSE] {full_response[:500]}{'...' if len(full_response) > 500 else ''}")
            logger.info("=" * 60)

            # Update chat history with complete response
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=full_response))

            # Trim history if too long
            if len(self.chat_history) > self.max_history * 2:
                self.chat_history = self.chat_history[-self.max_history * 2:]

        except Exception as e:
            logger.error(f"Error in RAG chain stream: {e}")
            raise

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def get_sources_for_last_query(self, question: str) -> list[dict]:
        """
        Get source documents for a query (useful after streaming completes).

        This method is called by the UI after streaming to display source citations.
        It uses the same retrieval logic as the main query methods to ensure
        consistency. We deduplicate by filename to show each source document once.

        Args:
            question: The question to get sources for.

        Returns:
            List of unique source dictionaries with document metadata.
        """
        # Use same retrieval strategy as main query methods
        is_comparison = self._is_comparison_query(question)

        if is_comparison:
            docs = self._get_all_relevant_docs()
        else:
            retriever = self.vector_store.get_retriever(k=10)
            docs = retriever.invoke(question)

        # Deduplicate sources by filename to show each document once
        # Multiple chunks from the same document are represented by a single source entry
        seen_files = set()
        unique_sources = []
        for doc in docs:
            filename = doc.metadata.get("filename")
            if filename not in seen_files:
                seen_files.add(filename)
                unique_sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "doc_type": doc.metadata.get("doc_type"),
                    "filename": filename,
                })

        return unique_sources

    def clear_memory(self) -> None:
        """
        Clear conversation history to start a fresh conversation.

        This is useful when users want to reset the context, for example when
        they upload a new resume or want to ask questions without previous context.
        """
        self.chat_history = []
        logger.info("Cleared conversation memory")

    def get_chat_history(self) -> list[dict]:
        """
        Get formatted chat history as a list of message dictionaries.

        This is useful for displaying conversation history in the UI or for
        debugging purposes. Each message is represented as a dict with 'role'
        and 'content' keys.

        Returns:
            List of message dictionaries with 'role' ('user' or 'assistant')
            and 'content' (message text).
        """
        history = []
        for msg in self.chat_history:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history
