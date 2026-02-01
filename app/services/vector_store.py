"""
Vector Store Service for Document Storage and Retrieval.

This module manages the vector database (ChromaDB) that stores document chunks
with their embeddings. It provides methods to:
- Add document chunks with embeddings
- Retrieve similar chunks using semantic search
- Delete documents by type or filename
- Get statistics about stored documents

The service uses OpenAI embeddings to convert text chunks into vectors, which
enables semantic similarity search - finding documents that are conceptually
similar even if they don't share exact keywords.
"""

import logging
from typing import Literal

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.services.chunking import DocumentChunk

logger = logging.getLogger(__name__)

# Type alias for document types - helps with type safety and IDE autocomplete
DocType = Literal["resume", "job_posting"]


class VectorStoreService:
    """
    Vector Store Service for ChromaDB-based Document Storage.

    This service acts as the interface between our application and ChromaDB,
    a local vector database. It handles:
    - Converting text chunks to embeddings using OpenAI
    - Storing embeddings in ChromaDB with metadata
    - Retrieving similar chunks using semantic search
    - Managing document lifecycle (add, delete, query)

    ChromaDB is chosen for local development because it requires no external
    services and persists data to disk, making it perfect for MVP deployment.
    """

    def __init__(self):
        """
        Initialize the vector store with ChromaDB and OpenAI embeddings.

        We set up two key components:
        1. OpenAI Embeddings: Converts text to vectors (1536 dimensions)
        2. ChromaDB: Stores vectors locally with persistence enabled

        The LangChain Chroma wrapper provides a convenient interface for
        adding documents and performing similarity searches.
        """
        self.settings = get_settings()

        # Initialize OpenAI embeddings model
        # This converts text chunks into 1536-dimensional vectors that capture
        # semantic meaning, enabling similarity search beyond keyword matching
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )

        # Initialize ChromaDB client with persistence enabled
        # Data is saved to disk so it persists across app restarts
        # We disable telemetry to keep the solution completely local
        self.chroma_client = chromadb.Client(
            ChromaSettings(
                persist_directory=str(self.settings.chroma_persist_directory),
                anonymized_telemetry=False,
            )
        )

        # Initialize LangChain Chroma wrapper for easier integration
        # This provides a unified interface for adding texts and retrieving
        # similar documents using the embeddings function
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.settings.chroma_collection_name,
            embedding_function=self.embeddings,
        )

        logger.info(
            f"Initialized VectorStoreService with collection: "
            f"{self.settings.chroma_collection_name}"
        )

    # =============================================================================
    # DOCUMENT STORAGE OPERATIONS
    # =============================================================================

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Add document chunks to the vector store with embeddings.

        When a user uploads a resume or job posting, the document is split into
        chunks and each chunk is:
        1. Converted to an embedding vector (by OpenAI)
        2. Stored in ChromaDB with its metadata (doc_type, filename, etc.)
        3. Indexed for fast similarity search

        The metadata is crucial - it allows us to filter by document type and
        attribute responses to specific source documents.

        Args:
            chunks: List of DocumentChunk objects containing text and metadata.

        Returns:
            Number of chunks successfully added to the vector store.
        """
        if not chunks:
            return 0

        # Extract components needed for vector store
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.id for chunk in chunks]

        # Add to vector store - this automatically generates embeddings
        # and stores them with the metadata
        self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        doc_type = chunks[0].metadata.get("doc_type", "unknown")
        logger.info(f"Added {len(chunks)} chunks of type '{doc_type}' to vector store")

        return len(chunks)

    # =============================================================================
    # DOCUMENT DELETION OPERATIONS
    # =============================================================================

    def delete_by_doc_type(self, doc_type: DocType) -> int:
        """
        Delete all chunks of a specific document type from the vector store.

        This is used when:
        - User uploads a new resume (we delete the old one first)
        - User wants to clear all job postings

        We use ChromaDB's metadata filtering to find all chunks matching the
        document type, then delete them by their IDs.

        Args:
            doc_type: Type of document to delete ("resume" or "job_posting").

        Returns:
            Number of chunks deleted.
        """
        # Get the underlying ChromaDB collection
        collection = self.chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection_name
        )

        # Query all chunks with matching document type using metadata filter
        results = collection.get(
            where={"doc_type": doc_type},
            include=["metadatas"],
        )

        # Delete all matching chunks by their IDs
        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks of type '{doc_type}'")
            return len(results["ids"])

        return 0

    def delete_job_posting(self, filename: str) -> int:
        """
        Delete a specific job posting by its filename.

        This allows users to remove individual job postings without affecting
        others. We use a compound filter to match both document type AND filename.

        Args:
            filename: Filename of the job posting to delete.

        Returns:
            Number of chunks deleted (a job posting may have multiple chunks).
        """
        collection = self.chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection_name
        )

        # Use compound filter to match both document type and filename
        # This ensures we only delete the specific job posting, not all job postings
        results = collection.get(
            where={
                "$and": [
                    {"doc_type": "job_posting"},
                    {"filename": filename},
                ]
            },
            include=["metadatas"],
        )

        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"Deleted job posting '{filename}' ({len(results['ids'])} chunks)")
            return len(results["ids"])

        return 0

    # =============================================================================
    # DOCUMENT RETRIEVAL OPERATIONS
    # =============================================================================

    def get_retriever(
        self,
        doc_type_filter: DocType | None = None,
        k: int | None = None,
    ):
        """
        Get a LangChain retriever for semantic similarity search.

        The retriever uses Maximum Marginal Relevance (MMR) which balances:
        - Relevance: Chunks similar to the query
        - Diversity: Avoids redundant chunks from the same document section

        This is better than pure similarity search because it prevents getting
        multiple chunks that say the same thing, giving more comprehensive context.

        Args:
            doc_type_filter: Optional filter to search only resume or job_posting chunks.
            k: Number of chunks to retrieve (defaults to config setting).

        Returns:
            LangChain retriever object that can be invoked with a query string.
        """
        k = k or self.settings.retrieval_top_k

        # Build search parameters
        search_kwargs = {"k": k}
        # Add metadata filter if specified (e.g., only search in resume chunks)
        if doc_type_filter:
            search_kwargs["filter"] = {"doc_type": doc_type_filter}

        # Return retriever with MMR for diverse, relevant results
        return self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs=search_kwargs,
        )

    def similarity_search(
        self,
        query: str,
        doc_type_filter: DocType | None = None,
        k: int | None = None,
    ) -> list[tuple[str, dict, float]]:
        """
        Perform similarity search and return results with relevance scores.

        This method provides direct access to similarity search with scores,
        which can be useful for debugging or when you need to see how relevant
        each retrieved chunk is to the query.

        The score represents cosine similarity - lower scores mean more similar
        (ChromaDB uses distance, so 0 = identical, higher = less similar).

        Args:
            query: Search query text.
            doc_type_filter: Optional filter to search only specific document type.
            k: Number of results to return.

        Returns:
            List of tuples containing (content, metadata, similarity_score).
        """
        k = k or self.settings.retrieval_top_k

        # Build filter dictionary if document type filter is specified
        filter_dict = {"doc_type": doc_type_filter} if doc_type_filter else None

        # Perform similarity search with scores
        # The query is converted to an embedding, then compared against stored embeddings
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict,
        )

        # Format results as tuples for easier consumption
        return [
            (doc.page_content, doc.metadata, score)
            for doc, score in results
        ]

    # =============================================================================
    # STATISTICS AND QUERY OPERATIONS
    # =============================================================================

    def get_document_stats(self) -> dict:
        """
        Get statistics about documents stored in the vector store.

        This method aggregates information about all stored chunks to provide
        a summary view. It's used by the UI to display:
        - How many chunks are indexed
        - Which resume is uploaded
        - How many job postings are stored
        - Total document counts

        We iterate through all chunks and group by document type and filename
        to build the statistics.

        Returns:
            Dictionary containing:
            - total_chunks: Total number of chunks stored
            - resume_chunks: Number of chunks from resume
            - job_posting_chunks: Number of chunks from job postings
            - resume_filename: Name of uploaded resume file
            - job_postings: List of job posting filenames
        """
        collection = self.chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection_name
        )

        # Get all documents with their metadata
        all_docs = collection.get(include=["metadatas"])

        # Initialize stats dictionary
        stats = {
            "total_chunks": len(all_docs["ids"]),
            "resume_chunks": 0,
            "job_posting_chunks": 0,
            "resume_filename": None,
            "job_postings": [],
        }

        # Aggregate statistics by iterating through metadata
        seen_jobs = set()
        for metadata in all_docs["metadatas"]:
            doc_type = metadata.get("doc_type")
            filename = metadata.get("filename")

            if doc_type == "resume":
                stats["resume_chunks"] += 1
                stats["resume_filename"] = filename
            elif doc_type == "job_posting":
                stats["job_posting_chunks"] += 1
                # Track unique job posting filenames (avoid duplicates)
                if filename not in seen_jobs:
                    seen_jobs.add(filename)
                    stats["job_postings"].append(filename)

        return stats

    def get_all_documents(self):
        """
        Retrieve ALL documents from the vector store without filtering.

        This method is used for comparison queries where we need to analyze
        all uploaded documents, not just the most similar ones. It returns
        every chunk stored in the database, which is necessary for fair
        comparison across all job postings.

        Returns:
            List of LangChain Document objects with page_content and metadata.
        """
        from langchain_core.documents import Document

        collection = self.chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection_name
        )

        # Get all documents with their content and metadata
        all_docs = collection.get(include=["documents", "metadatas"])

        # Convert ChromaDB format to LangChain Document format
        documents = []
        for i, doc_content in enumerate(all_docs.get("documents", [])):
            metadata = all_docs["metadatas"][i] if all_docs.get("metadatas") else {}
            documents.append(Document(page_content=doc_content, metadata=metadata))

        logger.info(f"Retrieved all {len(documents)} documents from vector store")
        return documents

    def get_documents_by_type(self, doc_type: DocType):
        """
        Retrieve all documents of a specific type (resume or job_posting).

        This is useful when you want to analyze only one type of document,
        for example, to get all job posting chunks without the resume chunks.

        Args:
            doc_type: Type of document ("resume" or "job_posting").

        Returns:
            List of LangChain Document objects filtered by document type.
        """
        from langchain_core.documents import Document

        collection = self.chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection_name
        )

        # Query with metadata filter to get only specific document type
        results = collection.get(
            where={"doc_type": doc_type},
            include=["documents", "metadatas"],
        )

        # Convert to LangChain Document format
        documents = []
        for i, doc_content in enumerate(results.get("documents", [])):
            metadata = results["metadatas"][i] if results.get("metadatas") else {}
            documents.append(Document(page_content=doc_content, metadata=metadata))

        return documents

    def clear_all(self) -> int:
        """
        Clear all documents from the vector store.

        This removes all chunks from the database, effectively resetting the
        application. Used when users want to start fresh with new documents.

        Returns:
            Number of chunks deleted.
        """
        collection = self.chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection_name
        )

        # Get all document IDs
        all_docs = collection.get()
        count = len(all_docs["ids"])

        # Delete all chunks if any exist
        if all_docs["ids"]:
            collection.delete(ids=all_docs["ids"])

        logger.info(f"Cleared all {count} chunks from vector store")
        return count
