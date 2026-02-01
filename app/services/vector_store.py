"""Vector store service using ChromaDB and OpenAI embeddings."""

import logging
from typing import Literal

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.services.chunking import DocumentChunk

logger = logging.getLogger(__name__)

# Type alias for document types
DocType = Literal["resume", "job_posting"]


class VectorStoreService:
    """Manages document storage and retrieval using ChromaDB."""

    def __init__(self):
        """Initialize the vector store with ChromaDB and OpenAI embeddings."""
        self.settings = get_settings()

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )

        # Initialize ChromaDB client with persistence
        self.chroma_client = chromadb.Client(
            ChromaSettings(
                persist_directory=str(self.settings.chroma_persist_directory),
                anonymized_telemetry=False,
            )
        )

        # Initialize LangChain Chroma wrapper
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.settings.chroma_collection_name,
            embedding_function=self.embeddings,
        )

        logger.info(
            f"Initialized VectorStoreService with collection: "
            f"{self.settings.chroma_collection_name}"
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk objects to add.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.id for chunk in chunks]

        self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        doc_type = chunks[0].metadata.get("doc_type", "unknown")
        logger.info(f"Added {len(chunks)} chunks of type '{doc_type}' to vector store")

        return len(chunks)

    def delete_by_doc_type(self, doc_type: DocType) -> int:
        """Delete all chunks of a specific document type.

        Args:
            doc_type: Type of document to delete ("resume" or "job_posting").

        Returns:
            Number of documents deleted.
        """
        # Get the underlying collection
        collection = self.chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection_name
        )

        # Get all documents of this type
        results = collection.get(
            where={"doc_type": doc_type},
            include=["metadatas"],
        )

        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks of type '{doc_type}'")
            return len(results["ids"])

        return 0

    def delete_job_posting(self, filename: str) -> int:
        """Delete a specific job posting by filename.

        Args:
            filename: Filename of the job posting to delete.

        Returns:
            Number of chunks deleted.
        """
        collection = self.chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection_name
        )

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

    def get_retriever(
        self,
        doc_type_filter: DocType | None = None,
        k: int | None = None,
    ):
        """Get a retriever for similarity search.

        Args:
            doc_type_filter: Optional filter by document type.
            k: Number of documents to retrieve.

        Returns:
            LangChain retriever object.
        """
        k = k or self.settings.retrieval_top_k

        search_kwargs = {"k": k}
        if doc_type_filter:
            search_kwargs["filter"] = {"doc_type": doc_type_filter}

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
        """Perform similarity search and return results with scores.

        Args:
            query: Search query.
            doc_type_filter: Optional filter by document type.
            k: Number of results to return.

        Returns:
            List of (content, metadata, score) tuples.
        """
        k = k or self.settings.retrieval_top_k

        filter_dict = {"doc_type": doc_type_filter} if doc_type_filter else None

        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict,
        )

        return [
            (doc.page_content, doc.metadata, score)
            for doc, score in results
        ]

    def get_document_stats(self) -> dict:
        """Get statistics about stored documents.

        Returns:
            Dictionary with document counts and metadata.
        """
        collection = self.chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection_name
        )

        all_docs = collection.get(include=["metadatas"])

        stats = {
            "total_chunks": len(all_docs["ids"]),
            "resume_chunks": 0,
            "job_posting_chunks": 0,
            "resume_filename": None,
            "job_postings": [],
        }

        seen_jobs = set()
        for metadata in all_docs["metadatas"]:
            doc_type = metadata.get("doc_type")
            filename = metadata.get("filename")

            if doc_type == "resume":
                stats["resume_chunks"] += 1
                stats["resume_filename"] = filename
            elif doc_type == "job_posting":
                stats["job_posting_chunks"] += 1
                if filename not in seen_jobs:
                    seen_jobs.add(filename)
                    stats["job_postings"].append(filename)

        return stats

    def get_all_documents(self):
        """Retrieve ALL documents from the vector store.

        Returns:
            List of Document objects with page_content and metadata.
        """
        from langchain_core.documents import Document

        collection = self.chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection_name
        )

        all_docs = collection.get(include=["documents", "metadatas"])

        documents = []
        for i, doc_content in enumerate(all_docs.get("documents", [])):
            metadata = all_docs["metadatas"][i] if all_docs.get("metadatas") else {}
            documents.append(Document(page_content=doc_content, metadata=metadata))

        logger.info(f"Retrieved all {len(documents)} documents from vector store")
        return documents

    def get_documents_by_type(self, doc_type: DocType):
        """Retrieve all documents of a specific type.

        Args:
            doc_type: Type of document ("resume" or "job_posting").

        Returns:
            List of Document objects.
        """
        from langchain_core.documents import Document

        collection = self.chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection_name
        )

        results = collection.get(
            where={"doc_type": doc_type},
            include=["documents", "metadatas"],
        )

        documents = []
        for i, doc_content in enumerate(results.get("documents", [])):
            metadata = results["metadatas"][i] if results.get("metadatas") else {}
            documents.append(Document(page_content=doc_content, metadata=metadata))

        return documents

    def clear_all(self) -> int:
        """Clear all documents from the vector store.

        Returns:
            Number of chunks deleted.
        """
        collection = self.chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection_name
        )

        all_docs = collection.get()
        count = len(all_docs["ids"])

        if all_docs["ids"]:
            collection.delete(ids=all_docs["ids"])

        logger.info(f"Cleared all {count} chunks from vector store")
        return count
