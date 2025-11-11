"""
Core Business Logic Package

Contains the core RAG functionality:
- Document embedding and processing
- Query processing and retrieval
- Response generation

Usage:
    from ragtime.core import embed_document, allowed_file
    from ragtime.core.embeddings import embed_documents_batch
"""

from ragtime.core.embeddings import (
    embed_document,
    embed_documents_batch,
    allowed_file,
    save_uploaded_file,
    embed,  # Legacy function
)

__all__ = [
    "embed_document",
    "embed_documents_batch",
    "allowed_file",
    "save_uploaded_file",
    "embed",  # Legacy
]