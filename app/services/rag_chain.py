"""RAG chain service for conversational retrieval."""

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

# Keywords that indicate user wants to compare across ALL jobs
COMPARISON_KEYWORDS = [
    "which job", "best job", "best fit", "best match", "compare", "comparison",
    "all jobs", "all job", "all three", "all 3", "every job", "each job",
    "best suited", "most suitable", "recommend", "should i apply",
    "which role", "which position", "rank", "ranking",
]


class RAGService:
    """Handles RAG-based question answering for career intelligence."""

    def __init__(self, vector_store: VectorStoreService):
        """Initialize the RAG service.

        Args:
            vector_store: VectorStoreService instance for document retrieval.
        """
        self.settings = get_settings()
        self.vector_store = vector_store

        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=self.settings.openai_llm_model,
            temperature=0.7,
            openai_api_key=self.settings.openai_api_key,
            streaming=True,
        )

        # Chat history storage
        self.chat_history: list[HumanMessage | AIMessage] = []
        self.max_history = self.settings.max_chat_history

        logger.info(f"Initialized RAGService with model: {self.settings.openai_llm_model}")

    def _is_comparison_query(self, question: str) -> bool:
        """Detect if the user is asking to compare across all jobs.

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

    def _get_all_relevant_docs(self):
        """Get ALL documents (resume + all job postings) for comprehensive comparison.

        Returns:
            List of all documents from the vector store.
        """
        all_docs = self.vector_store.get_all_documents()

        # Group by filename to avoid duplicates and organize
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hash(doc.page_content[:100])  # Use first 100 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        logger.info(f"Retrieved {len(unique_docs)} unique document chunks for comparison")
        return unique_docs

    def _format_docs(self, docs) -> str:
        """Format retrieved documents into a context string.

        Args:
            docs: List of retrieved documents.

        Returns:
            Formatted context string.
        """
        formatted = []
        for doc in docs:
            doc_type = doc.metadata.get("doc_type", "document").upper()
            filename = doc.metadata.get("filename", "Unknown")
            formatted.append(f"[{doc_type}: {filename}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def _get_chat_history_text(self) -> str:
        """Get formatted chat history for context.

        Returns:
            Formatted chat history string.
        """
        if not self.chat_history:
            return "No previous conversation."

        history_parts = []
        for msg in self.chat_history[-self.max_history * 2:]:  # Last N turns
            if isinstance(msg, HumanMessage):
                history_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                history_parts.append(f"Assistant: {content}")

        return "\n".join(history_parts)

    def ask(self, question: str) -> dict:
        """Ask a question and get a response with sources.

        Args:
            question: User's question.

        Returns:
            Dictionary with 'answer' and 'sources' keys.
        """
        try:
            # Check if this is a comparison query requiring ALL documents
            is_comparison = self._is_comparison_query(question)

            if is_comparison:
                # Retrieve ALL documents for comprehensive comparison
                docs = self._get_all_relevant_docs()
                logger.info("[MODE] Comparison query detected - using ALL documents")
            else:
                # Use standard retrieval with higher k for better coverage
                retriever = self.vector_store.get_retriever(k=10)
                docs = retriever.invoke(question)

            # Format context
            context = self._format_docs(docs)
            chat_history = self._get_chat_history_text()

            # Create the prompt
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

            # Get the response
            response = chain.invoke({
                "context": context,
                "chat_history": chat_history,
                "question": question,
            })

            # Log query and response
            logger.info("=" * 60)
            logger.info(f"[QUERY] {question}")
            logger.info(f"[SOURCES] {[doc.metadata.get('filename') for doc in docs]}")
            logger.info(f"[RESPONSE] {response[:500]}{'...' if len(response) > 500 else ''}")
            logger.info("=" * 60)

            # Update chat history
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=response))

            # Trim history if too long
            if len(self.chat_history) > self.max_history * 2:
                self.chat_history = self.chat_history[-self.max_history * 2:]

            # Extract sources
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
        """Ask a question and stream the response.

        Args:
            question: User's question.

        Yields:
            Response tokens as they're generated.
        """
        try:
            # Check if this is a comparison query requiring ALL documents
            is_comparison = self._is_comparison_query(question)

            if is_comparison:
                # Retrieve ALL documents for comprehensive comparison
                docs = self._get_all_relevant_docs()
                logger.info("[MODE] Comparison query detected - using ALL documents")
            else:
                # Use standard retrieval with higher k for better coverage
                retriever = self.vector_store.get_retriever(k=10)
                docs = retriever.invoke(question)

            # Format context
            context = self._format_docs(docs)
            chat_history = self._get_chat_history_text()

            # Create the prompt
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

            # Log query start
            logger.info("=" * 60)
            logger.info(f"[QUERY] {question}")
            logger.info(f"[SOURCES] {[doc.metadata.get('filename') for doc in docs]}")

            # Stream the response
            full_response = ""
            for chunk in chain.stream({
                "context": context,
                "chat_history": chat_history,
                "question": question,
            }):
                full_response += chunk
                yield chunk

            # Log response
            logger.info(f"[RESPONSE] {full_response[:500]}{'...' if len(full_response) > 500 else ''}")
            logger.info("=" * 60)

            # Update chat history
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=full_response))

            # Trim history if too long
            if len(self.chat_history) > self.max_history * 2:
                self.chat_history = self.chat_history[-self.max_history * 2:]

        except Exception as e:
            logger.error(f"Error in RAG chain stream: {e}")
            raise

    def get_sources_for_last_query(self, question: str) -> list[dict]:
        """Get sources for a query (useful after streaming).

        Args:
            question: The question to get sources for.

        Returns:
            List of source dictionaries.
        """
        # Check if this is a comparison query
        is_comparison = self._is_comparison_query(question)

        if is_comparison:
            docs = self._get_all_relevant_docs()
        else:
            retriever = self.vector_store.get_retriever(k=10)
            docs = retriever.invoke(question)

        # Deduplicate sources by filename
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
        """Clear conversation memory."""
        self.chat_history = []
        logger.info("Cleared conversation memory")

    def get_chat_history(self) -> list[dict]:
        """Get formatted chat history.

        Returns:
            List of message dictionaries with 'role' and 'content'.
        """
        history = []
        for msg in self.chat_history:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history
