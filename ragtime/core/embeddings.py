"""
Document Embedding Module

Handles document upload, processing, chunking, and embedding into the vector database.
Uses the VectorDatabase abstraction layer with integrated settings and logging.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from langchain_core.documents import Document

from ragtime.config.settings import get_settings
from ragtime.monitoring.logging import get_logger, log_error, log_performance
from ragtime.storage import get_vector_db
from ragtime.models.documents import DocumentMetadata, DocumentUploadResponse

logger = get_logger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt'}


def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed.

    Args:
        filename: Name of the file to check

    Returns:
        bool: True if file extension is allowed
    """
    if '.' not in filename:
        return False

    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def save_uploaded_file(file: FileStorage, temp_folder: Optional[str] = None) -> str:
    """
    Save uploaded file to temporary folder.

    Args:
        file: Uploaded file object
        temp_folder: Optional temp folder path (uses settings if not provided)

    Returns:
        str: Path to saved file

    Raises:
        ValueError: If filename is invalid
    """
    if not file.filename:
        raise ValueError("File has no filename")

    # Get temp folder from settings
    if temp_folder is None:
        temp_folder = str(Path.cwd() / '_temp')

    # Ensure temp folder exists
    Path(temp_folder).mkdir(parents=True, exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().timestamp()
    secure_name = secure_filename(file.filename)
    filename = f"{timestamp}_{secure_name}"
    file_path = os.path.join(temp_folder, filename)

    # Save file
    file.save(file_path)

    logger.info(
        "file_saved",
        filename=secure_name,
        file_path=file_path,
        size_bytes=os.path.getsize(file_path)
    )

    return file_path


def embed_document(
    file: FileStorage,
    collection_name: Optional[str] = None,
    metadata: Optional[DocumentMetadata] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> Tuple[bool, Optional[DocumentUploadResponse]]:
    """
    Embed a document into the vector database.

    This is the main entry point for document embedding. It:
    1. Validates the file
    2. Saves it temporarily
    3. Loads and chunks the document
    4. Embeds it into the vector database
    5. Cleans up temporary files

    Args:
        file: Uploaded file object
        collection_name: Target collection (uses default if not provided)
        metadata: Optional document metadata
        chunk_size: Custom chunk size (uses settings if not provided)
        chunk_overlap: Custom overlap (uses settings if not provided)

    Returns:
        Tuple of (success: bool, response: DocumentUploadResponse or None)
    """
    start_time = datetime.now()
    file_path = None

    try:
        # Validate file
        if not file.filename or not allowed_file(file.filename):
            logger.warning(
                "invalid_file_upload",
                filename=file.filename if file.filename else "no_filename"
            )
            return False, None

        settings = get_settings()
        collection_name = collection_name or settings.default_collection

        logger.info(
            "starting_document_embedding",
            filename=file.filename,
            collection=collection_name
        )

        # Save file temporarily
        file_path = save_uploaded_file(file)

        # Get vector database
        vector_db = get_vector_db()

        # Load document
        try:
            documents = vector_db.load_document(file_path)
        except ValueError as e:
            logger.error(
                "unsupported_file_type",
                filename=file.filename,
                error=str(e)
            )
            return False, None

        # Chunk documents
        chunks = vector_db.chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Add metadata to chunks if provided
        if metadata:
            metadata_dict = metadata.model_dump()
            for chunk in chunks:
                if chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata.update(metadata_dict)

        # Add documents to vector database
        success = vector_db.add_documents(
            documents=chunks,
            collection_name=collection_name
        )

        if not success:
            logger.error(
                "failed_to_add_documents",
                filename=file.filename,
                collection=collection_name
            )
            return False, None

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Log performance
        log_performance(
            logger,
            "document_embedding",
            processing_time,
            chunks_created=len(chunks),
            collection=collection_name
        )

        # Create response
        response = DocumentUploadResponse(
            message=f"Successfully embedded {len(chunks)} chunks",
            chunks_created=len(chunks),
            collection=collection_name,
            document_id=f"{collection_name}_{datetime.now().timestamp()}",
            metadata=metadata.model_dump() if metadata else None,
            processing_time_ms=processing_time
        )

        return True, response

    except Exception as e:
        log_error(
            logger,
            e,
            context={
                "operation": "embed_document",
                "filename": file.filename if file.filename else "unknown",
                "collection": collection_name
            }
        )
        return False, None

    finally:
        # Clean up temporary file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug("temp_file_removed", file_path=file_path)
            except Exception as cleanup_error:
                logger.warning(
                    "failed_to_remove_temp_file",
                    file_path=file_path,
                    error=str(cleanup_error)
                )


def embed_documents_batch(
    files: List[FileStorage],
    collection_name: Optional[str] = None,
    metadata: Optional[DocumentMetadata] = None
) -> Tuple[int, int, List[DocumentUploadResponse]]:
    """
    Embed multiple documents in batch.

    Args:
        files: List of uploaded files
        collection_name: Target collection
        metadata: Optional metadata for all documents

    Returns:
        Tuple of (successful_count, failed_count, responses)
    """
    successful = 0
    failed = 0
    responses = []

    logger.info(
        "starting_batch_embedding",
        num_files=len(files),
        collection=collection_name
    )

    for file in files:
        success, response = embed_document(
            file=file,
            collection_name=collection_name,
            metadata=metadata
        )

        if success and response:
            successful += 1
            responses.append(response)
        else:
            failed += 1

    logger.info(
        "batch_embedding_completed",
        total=len(files),
        successful=successful,
        failed=failed
    )

    return successful, failed, responses


# Legacy function for backward compatibility
def embed(file: FileStorage, collection_name: str = 'langchain') -> bool:
    """
    DEPRECATED: Legacy embed function for backward compatibility.

    Use embed_document() instead for better error handling and response data.

    Args:
        file: Uploaded file
        collection_name: Collection name

    Returns:
        bool: True if successful
    """
    success, _ = embed_document(file, collection_name)
    return success