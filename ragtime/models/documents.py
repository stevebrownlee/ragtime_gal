"""
Document Models

Pydantic models for document handling, metadata, chunking, and API requests/responses.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class DocumentMetadata(BaseModel):
    """
    Metadata for embedded documents in the Ragtime Gal book writing assistant.

    **Why metadata matters for your book project**:

    While all your documents are chapters and world-building notes, metadata enables:

    1. **Organized Retrieval**: When you query "What happened in chapter 5?", the system
       can filter by `chapter_number` to return only that chapter's content, rather than
       searching through all embedded text.

    2. **Contextual Understanding**: The RAG system can provide better answers by knowing
       "this is world-building vs. narrative content" or "this is early vs. late chapter".
       Metadata enriches the context beyond just the text.

    3. **Multi-Book Support**: Even if you're writing one book now, `book_title` lets you
       manage multiple projects in the same database without confusion.

    4. **Analytics & Insights**: Track writing progress through `word_count`, identify
       which chapters get queried most (popularity), and monitor when content was added
       (`created_at`) for version tracking.

    5. **MCP Tool Filtering**: Your MCP tools (like `get_chapter_info`, `list_all_chapters`)
       rely on metadata to provide structured navigation. Without it, they'd only have
       raw text to work with.

    6. **Feedback Attribution**: When users rate responses, metadata links that feedback
       to specific chapters/books, enabling targeted model improvements.

    **Example**: Query "Who is the antagonist?" -> The system retrieves character notes
    (filtered by metadata indicating "character_analysis") rather than random chapter text
    mentioning multiple characters.
    """
    book_title: Optional[str] = None
    chapter_number: Optional[int] = None
    chapter_title: Optional[str] = None
    author: Optional[str] = None
    genre: Optional[str] = None
    word_count: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "book_title": "The Secret Harmony",
                "chapter_number": 5,
                "chapter_title": "The Discovery",
                "author": "Jane Doe",
                "genre": "Fantasy",
                "word_count": 2500,
                "tags": ["action", "character-development"]
            }
        }


class DocumentChunk(BaseModel):
    """Represents a chunk of a document with metadata"""
    content: str = Field(..., min_length=1, description="The chunk text content")
    metadata: DocumentMetadata = Field(..., description="Metadata for this chunk")
    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    chunk_index: int = Field(default=0, ge=0, description="Index of this chunk in the document")
    embedding_id: Optional[str] = Field(None, description="Vector database embedding ID")

    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Ensure content is not empty or just whitespace"""
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace')
        return v.strip()


class DocumentUploadRequest(BaseModel):
    """Request model for document upload via API"""
    collection_name: str = Field(
        ...,
        pattern=r'^[a-zA-Z0-9_-]+$',
        min_length=1,
        max_length=100,
        description="Name of the collection to store documents"
    )
    metadata: Optional[DocumentMetadata] = Field(
        None,
        description="Optional metadata to attach to the document"
    )
    chunk_size: Optional[int] = Field(
        None,
        ge=100,
        le=20000,
        description="Custom chunk size (overrides default)"
    )
    chunk_overlap: Optional[int] = Field(
        None,
        ge=0,
        le=1000,
        description="Custom chunk overlap (overrides default)"
    )


class DocumentUploadResponse(BaseModel):
    """Response model for successful document upload"""
    message: str = Field(..., description="Success message")
    chunks_created: int = Field(..., ge=0, description="Number of chunks created")
    collection: str = Field(..., description="Collection name where documents stored")
    document_id: str = Field(..., description="Unique document identifier")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Applied metadata"
    )
    processing_time_ms: Optional[float] = Field(
        None,
        ge=0,
        description="Time taken to process the document"
    )


class DocumentDeleteRequest(BaseModel):
    """Request model for deleting documents"""
    collection_name: str = Field(
        ...,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Collection to delete from"
    )
    document_id: Optional[str] = Field(
        None,
        description="Specific document ID to delete"
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata filter for batch deletion"
    )

    @field_validator('filter_metadata')
    @classmethod
    def validate_delete_criteria(cls, v, info):
        """Ensure either document_id or filter_metadata is provided"""
        document_id = info.data.get('document_id')
        if not document_id and not v:
            raise ValueError('Either document_id or filter_metadata must be provided')
        return v


class DocumentSearchRequest(BaseModel):
    """Request model for searching documents by metadata"""
    collection_name: str = Field(
        ...,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Collection to search"
    )
    filter_metadata: Dict[str, Any] = Field(
        ...,
        description="Metadata filters"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results"
    )


class DocumentSearchResponse(BaseModel):
    """Response model for document search"""
    documents: List[DocumentChunk] = Field(
        default_factory=list,
        description="Matching documents"
    )
    total_count: int = Field(..., ge=0, description="Total matching documents")
    query_time_ms: float = Field(..., ge=0, description="Query execution time")


class CollectionInfo(BaseModel):
    """Information about a vector database collection"""
    name: str = Field(..., description="Collection name")
    document_count: int = Field(..., ge=0, description="Number of documents")
    metadata_schema: Dict[str, str] = Field(
        default_factory=dict,
        description="Metadata field types"
    )
    created_at: Optional[datetime] = None
    size_bytes: Optional[int] = Field(None, ge=0, description="Collection size in bytes")


class CollectionListResponse(BaseModel):
    """Response model for listing collections"""
    collections: List[CollectionInfo] = Field(
        default_factory=list,
        description="List of collections"
    )
    total_count: int = Field(..., ge=0, description="Total number of collections")