"""
Core Business Logic Package

Contains the core RAG functionality:
- Document embedding and processing
- Query processing and retrieval
- Response generation

Usage:
    from ragtime.core import embed_document, query
    from ragtime.core.embeddings import embed_documents_batch
    from ragtime.core.query_processor import QueryProcessor
"""

from ragtime.core.embeddings import (
    embed_document,
    embed_documents_batch,
    allowed_file,
    save_uploaded_file,
    embed,  # Legacy function
)

from ragtime.core.query_processor import (
    query,
    query_with_feedback_optimization,
    get_query_enhancement_suggestions,
    get_feedback_summary,
    clear_optimization_cache,
    QueryProcessor,
)

__all__ = [
    # Embeddings
    "embed_document",
    "embed_documents_batch",
    "allowed_file",
    "save_uploaded_file",
    "embed",  # Legacy
    # Query Processing
    "query",
    "query_with_feedback_optimization",
    "get_query_enhancement_suggestions",
    "get_feedback_summary",
    "clear_optimization_cache",
    "QueryProcessor",
]