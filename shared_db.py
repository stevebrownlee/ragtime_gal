"""
Shared Database Manager Module

This module provides the SharedDatabaseManager class for thread-safe access
to the Chroma vector database from both Flask and MCP server components.
"""

import os
import logging
import threading
from typing import Optional, Dict, List, Any
from contextlib import contextmanager

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from metadata_utils import MetadataQueryManager

logger = logging.getLogger(__name__)


class SharedDatabaseManager:
    """
    Manages thread-safe access to the Chroma vector database.
    Provides a single database instance shared between Flask and MCP server.
    """

    def __init__(self, persist_directory: str = None, embedding_model: str = None, ollama_base_url: str = None):
        """
        Initialize the shared database manager.

        Args:
            persist_directory: Directory for Chroma database persistence
            embedding_model: Ollama embedding model name
            ollama_base_url: Base URL for Ollama service
        """
        # Configuration
        self.persist_directory = persist_directory or os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'mistral')
        self.ollama_base_url = ollama_base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

        # Thread safety
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._db_instance: Optional[Chroma] = None
        self._embeddings_instance: Optional[OllamaEmbeddings] = None
        self._metadata_manager: Optional[MetadataQueryManager] = None
        self._initialization_error: Optional[Exception] = None

        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        logger.info(f"SharedDatabaseManager initialized with persist_directory: {self.persist_directory}")

    def _create_embeddings(self) -> OllamaEmbeddings:
        """Create Ollama embeddings instance."""
        if self._embeddings_instance is None:
            try:
                self._embeddings_instance = OllamaEmbeddings(
                    model=self.embedding_model,
                    base_url=self.ollama_base_url
                )
                logger.info(f"Created Ollama embeddings with model: {self.embedding_model}")
            except Exception as e:
                logger.error(f"Error creating Ollama embeddings: {e}")
                raise

        return self._embeddings_instance

    def _create_database(self) -> Chroma:
        """Create Chroma database instance."""
        if self._db_instance is None:
            try:
                embeddings = self._create_embeddings()
                self._db_instance = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings
                )
                logger.info(f"Created Chroma database instance at: {self.persist_directory}")
            except Exception as e:
                logger.error(f"Error creating Chroma database: {e}")
                self._initialization_error = e
                raise

        return self._db_instance

    def get_database(self) -> Chroma:
        """
        Get the shared Chroma database instance in a thread-safe manner.

        Returns:
            Chroma: The shared database instance

        Raises:
            Exception: If database initialization failed
        """
        with self._lock:
            if self._initialization_error:
                raise self._initialization_error

            if self._db_instance is None:
                self._create_database()

            return self._db_instance

    def get_embeddings(self) -> OllamaEmbeddings:
        """
        Get the shared Ollama embeddings instance in a thread-safe manner.

        Returns:
            OllamaEmbeddings: The shared embeddings instance
        """
        with self._lock:
            if self._embeddings_instance is None:
                self._create_embeddings()

            return self._embeddings_instance

    def get_metadata_manager(self) -> MetadataQueryManager:
        """
        Get the shared metadata query manager instance in a thread-safe manner.

        Returns:
            MetadataQueryManager: The shared metadata manager instance
        """
        with self._lock:
            if self._metadata_manager is None:
                self._metadata_manager = MetadataQueryManager(
                    persist_directory=self.persist_directory,
                    embedding_model=self.embedding_model,
                    ollama_base_url=self.ollama_base_url
                )
                logger.info("Created MetadataQueryManager instance")

            return self._metadata_manager

    @contextmanager
    def get_database_context(self):
        """
        Context manager for database operations with automatic error handling.

        Usage:
            with shared_db.get_database_context() as db:
                results = db.similarity_search("query")
        """
        try:
            db = self.get_database()
            yield db
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise

    def test_connection(self) -> bool:
        """
        Test the database connection and basic functionality.

        Returns:
            bool: True if connection is working, False otherwise
        """
        try:
            with self._lock:
                db = self.get_database()
                # Try a simple operation to test the connection
                collection_info = db.get()
                logger.debug(f"Database connection test successful. Documents: {len(collection_info.get('ids', []))}")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def get_database_stats(self) -> dict:
        """
        Get statistics about the database contents.

        Returns:
            dict: Database statistics
        """
        try:
            with self._lock:
                db = self.get_database()
                collection_data = db.get()

                ids = collection_data.get("ids", [])
                metadatas = collection_data.get("metadatas", [])

                # Count documents by source
                sources = {}
                file_types = {}

                for metadata in metadatas:
                    if metadata:
                        source = metadata.get("source", "unknown")
                        sources[source] = sources.get(source, 0) + 1

                        # Extract file extension
                        if "." in source:
                            ext = source.split(".")[-1].lower()
                            file_types[ext] = file_types.get(ext, 0) + 1

                return {
                    "total_documents": len(ids),
                    "unique_sources": len(sources),
                    "sources": sources,
                    "file_types": file_types,
                    "persist_directory": self.persist_directory,
                    "embedding_model": self.embedding_model
                }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                "error": str(e),
                "total_documents": 0,
                "unique_sources": 0
            }

    def reset_connection(self):
        """
        Reset the database connection (useful for error recovery).
        This will force recreation of the database instance on next access.
        """
        with self._lock:
            logger.info("Resetting database connection")
            self._db_instance = None
            self._embeddings_instance = None
            self._metadata_manager = None
            self._initialization_error = None

    def add_documents(self, texts: list, metadatas: list = None, ids: list = None):
        """
        Add documents to the database in a thread-safe manner.

        Args:
            texts: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
        """
        with self._lock:
            db = self.get_database()
            return db.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def similarity_search(self, query: str, k: int = 4, filter: dict = None):
        """
        Perform similarity search in a thread-safe manner.

        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter

        Returns:
            List of documents
        """
        with self._lock:
            db = self.get_database()
            return db.similarity_search(query=query, k=k, filter=filter)

    def similarity_search_with_score(self, query: str, k: int = 4, filter: dict = None):
        """
        Perform similarity search with scores in a thread-safe manner.

        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter

        Returns:
            List of (document, score) tuples
        """
        with self._lock:
            db = self.get_database()
            return db.similarity_search_with_score(query=query, k=k, filter=filter)

    def delete_documents(self, ids: list):
        """
        Delete documents by IDs in a thread-safe manner.

        Args:
            ids: List of document IDs to delete
        """
        with self._lock:
            db = self.get_database()
            return db.delete(ids=ids)

    def get_all_documents(self):
        """
        Get all documents from the database in a thread-safe manner.

        Returns:
            Dictionary with ids, documents, metadatas, etc.
        """
        with self._lock:
            db = self.get_database()
            return db.get()

    def purge_database(self):
        """
        Delete all documents from the database in a thread-safe manner.

        Returns:
            int: Number of documents deleted
        """
        with self._lock:
            db = self.get_database()
            all_data = db.get()
            all_ids = all_data.get("ids", [])

            if all_ids:
                db.delete(ids=all_ids)
                logger.info(f"Purged {len(all_ids)} documents from database")
                return len(all_ids)
            else:
                logger.info("Database is already empty")
                return 0

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # No cleanup needed for now
        pass

    # Enhanced metadata methods for Phase 2
    def get_all_books(self) -> List[Dict[str, Any]]:
        """Get all books with enhanced metadata"""
        with self._lock:
            metadata_manager = self.get_metadata_manager()
            return metadata_manager.get_all_books()

    def get_chapters_for_book(self, book_title: str) -> List[Dict[str, Any]]:
        """Get all chapters for a specific book"""
        with self._lock:
            metadata_manager = self.get_metadata_manager()
            return metadata_manager.get_chapters_for_book(book_title)

    def get_book_statistics(self, book_title: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a book"""
        with self._lock:
            metadata_manager = self.get_metadata_manager()
            return metadata_manager.get_book_statistics(book_title)

    def search_content_with_metadata(self, query: str, filters: Dict[str, Any] = None,
                                   k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search with metadata filtering"""
        with self._lock:
            metadata_manager = self.get_metadata_manager()
            return metadata_manager.search_content_with_metadata(query, filters, k)

    def get_chapter_content(self, book_title: str, chapter_number: int) -> List[Dict[str, Any]]:
        """Get all content chunks for a specific chapter"""
        with self._lock:
            metadata_manager = self.get_metadata_manager()
            return metadata_manager.get_chapter_content(book_title, chapter_number)

    def validate_metadata_consistency(self) -> Dict[str, Any]:
        """Validate and report on metadata consistency"""
        with self._lock:
            metadata_manager = self.get_metadata_manager()
            return metadata_manager.validate_metadata_consistency()
