"""
Vector Database Abstraction Layer

Provides a clean interface for vector database operations using ChromaDB and LangChain.
Integrates with settings and structured logging.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ragtime.config.settings import get_settings
from ragtime.monitoring.logging import get_logger, log_error, log_performance
from ragtime.models.documents import DocumentMetadata, DocumentChunk

logger = get_logger(__name__)


class VectorDatabase:
    """
    Vector database operations wrapper for ChromaDB.

    Handles document embedding, retrieval, and collection management
    with integrated settings and logging.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the vector database.

        Args:
            persist_directory: Directory for persistent storage.
                             Uses settings value if not provided.
        """
        settings = get_settings()
        self.persist_directory = persist_directory or str(Path.cwd() / 'chroma_db')
        self.embedding_model = settings.embedding_model
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.default_collection = settings.default_collection

        logger.info(
            "vector_db_initialized",
            persist_directory=self.persist_directory,
            embedding_model=self.embedding_model
        )

        # Initialize embeddings
        self._embeddings = None

    def _get_embeddings(self) -> OllamaEmbeddings:
        """Get or create the embeddings instance."""
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                model=self.embedding_model
            )
            logger.debug(
                "embeddings_initialized",
                model=self.embedding_model
            )
        return self._embeddings

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from file path.

        Automatically detects file type (PDF or text/markdown).

        Args:
            file_path: Path to the document file

        Returns:
            List of loaded document objects

        Raises:
            ValueError: If file type is unsupported
            Exception: If loading fails
        """
        try:
            file_path_obj = Path(file_path)

            # Detect file type
            with open(file_path, 'rb') as f:
                content_start = f.read(20)

            # Determine loader
            if content_start.startswith(b'%PDF'):
                logger.info("loading_pdf_document", file_path=file_path)
                loader = PyPDFLoader(file_path=file_path)
            elif file_path_obj.suffix.lower() in ['.md', '.txt']:
                logger.info("loading_text_document", file_path=file_path)
                loader = TextLoader(file_path=file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path_obj.suffix}")

            # Load documents
            documents = loader.load()

            logger.info(
                "document_loaded",
                file_path=file_path,
                num_documents=len(documents)
            )

            return documents

        except Exception as e:
            log_error(
                logger,
                e,
                context={"operation": "load_document", "file_path": file_path}
            )
            raise

    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to chunk
            chunk_size: Custom chunk size (uses settings if not provided)
            chunk_overlap: Custom overlap (uses settings if not provided)

        Returns:
            List of document chunks
        """
        try:
            chunk_size = chunk_size or self.chunk_size
            chunk_overlap = chunk_overlap or self.chunk_overlap

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            chunks = text_splitter.split_documents(documents)

            logger.info(
                "documents_chunked",
                num_input_docs=len(documents),
                num_chunks=len(chunks),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            return chunks

        except Exception as e:
            log_error(
                logger,
                e,
                context={"operation": "chunk_documents"}
            )
            raise

    def add_documents(
        self,
        documents: List[Document],
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents to add
            collection_name: Target collection (uses default if not provided)
            metadata: Optional metadata to attach to all documents

        Returns:
            bool: True if successful
        """
        try:
            collection_name = collection_name or self.default_collection

            # Add metadata to documents if provided
            if metadata:
                for doc in documents:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata.update(metadata)

            # Create/update collection
            db = Chroma.from_documents(
                documents=documents,
                embedding=self._get_embeddings(),
                persist_directory=self.persist_directory,
                collection_name=collection_name
            )

            logger.info(
                "documents_added_to_collection",
                collection=collection_name,
                num_documents=len(documents)
            )

            return True

        except Exception as e:
            log_error(
                logger,
                e,
                context={
                    "operation": "add_documents",
                    "collection": collection_name
                }
            )
            return False

    def get_retriever(
        self,
        collection_name: Optional[str] = None,
        k: Optional[int] = None,
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Get a retriever for similarity search.

        Args:
            collection_name: Collection to search (uses default if not provided)
            k: Number of documents to retrieve
            search_kwargs: Additional search parameters

        Returns:
            LangChain retriever object
        """
        try:
            settings = get_settings()
            collection_name = collection_name or self.default_collection
            k = k or settings.default_k

            # Build search kwargs
            if search_kwargs is None:
                search_kwargs = {}
            search_kwargs['k'] = k

            # Load existing collection
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self._get_embeddings(),
                collection_name=collection_name
            )

            retriever = db.as_retriever(search_kwargs=search_kwargs)

            logger.debug(
                "retriever_created",
                collection=collection_name,
                k=k
            )

            return retriever

        except Exception as e:
            log_error(
                logger,
                e,
                context={
                    "operation": "get_retriever",
                    "collection": collection_name
                }
            )
            raise

    def similarity_search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Query text
            collection_name: Collection to search
            k: Number of results
            filter_dict: Metadata filter

        Returns:
            List of matching documents
        """
        try:
            settings = get_settings()
            collection_name = collection_name or self.default_collection
            k = k or settings.default_k

            # Load collection
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self._get_embeddings(),
                collection_name=collection_name
            )

            # Perform search
            if filter_dict:
                results = db.similarity_search(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = db.similarity_search(query, k=k)

            logger.info(
                "similarity_search_completed",
                collection=collection_name,
                query_length=len(query),
                num_results=len(results),
                k=k
            )

            return results

        except Exception as e:
            log_error(
                logger,
                e,
                context={
                    "operation": "similarity_search",
                    "collection": collection_name,
                    "query": query[:100]
                }
            )
            raise

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.

        Args:
            collection_name: Name of collection to delete

        Returns:
            bool: True if successful
        """
        try:
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self._get_embeddings(),
                collection_name=collection_name
            )

            db.delete_collection()

            logger.info(
                "collection_deleted",
                collection=collection_name
            )

            return True

        except Exception as e:
            log_error(
                logger,
                e,
                context={
                    "operation": "delete_collection",
                    "collection": collection_name
                }
            )
            return False


# Global instance (singleton)
_vector_db: Optional[VectorDatabase] = None


def get_vector_db(persist_directory: Optional[str] = None) -> VectorDatabase:
    """
    Get or create the global vector database instance.

    Args:
        persist_directory: Optional persist directory

    Returns:
        VectorDatabase instance
    """
    global _vector_db

    if _vector_db is None:
        _vector_db = VectorDatabase(persist_directory=persist_directory)

    return _vector_db


def reset_vector_db() -> None:
    """Reset the global vector database instance."""
    global _vector_db
    _vector_db = None