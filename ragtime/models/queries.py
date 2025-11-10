"""
Query Models

Pydantic models for query requests, responses, and query enhancement.
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class EnhancementMode(str, Enum):
    """Query enhancement modes"""
    AUTO = "auto"  # Automatically determine best enhancement
    EXPAND = "expand"  # Expand query with related terms
    REPHRASE = "rephrase"  # Rephrase query for better retrieval
    NONE = "none"  # No enhancement


class QueryRequest(BaseModel):
    """Request model for queries"""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's question"
    )
    collection_name: str = Field(
        ...,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Collection to query"
    )
    k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of documents to retrieve"
    )
    enhancement_mode: EnhancementMode = Field(
        default=EnhancementMode.AUTO,
        description="Query enhancement mode"
    )
    use_conversation_context: bool = Field(
        default=True,
        description="Use conversation history for context"
    )
    similarity_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (overrides default)"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation tracking"
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata filters for retrieval"
    )

    @field_validator('question')
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        """Ensure question is not empty or whitespace"""
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError('Question cannot be empty or whitespace')
        return v_stripped


class RetrievedDocument(BaseModel):
    """Model for a retrieved document"""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score (0-1)"
    )
    chunk_id: Optional[str] = Field(None, description="Unique chunk identifier")
    rank: Optional[int] = Field(None, ge=1, description="Ranking position")


class QueryResponse(BaseModel):
    """Response model for queries"""
    answer: str = Field(..., description="Generated answer")
    sources: List[RetrievedDocument] = Field(
        default_factory=list,
        description="Retrieved source documents"
    )
    conversation_id: str = Field(..., description="Conversation identifier")
    query_id: str = Field(..., description="Unique query identifier")
    enhancement_applied: Optional[str] = Field(
        None,
        description="Description of query enhancement applied"
    )
    enhanced_query: Optional[str] = Field(
        None,
        description="The enhanced query text (if enhancement was applied)"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in milliseconds"
    )
    retrieval_time_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="Time spent on retrieval"
    )
    generation_time_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="Time spent on response generation"
    )
    model_used: Optional[str] = Field(
        None,
        description="LLM model used for generation"
    )
    tokens_used: Optional[int] = Field(
        None,
        ge=0,
        description="Total tokens used"
    )


class QueryEnhancementSuggestion(BaseModel):
    """Suggested query enhancement based on feedback patterns"""
    original_query: str = Field(..., description="Original query text")
    suggested_query: str = Field(..., description="Enhanced query text")
    reason: str = Field(..., description="Reason for suggestion")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in suggestion (0-1)"
    )
    expected_improvement: Optional[str] = Field(
        None,
        description="Expected improvement description"
    )


class BatchQueryRequest(BaseModel):
    """Request model for batch queries"""
    queries: List[QueryRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of queries to process"
    )
    parallel: bool = Field(
        default=True,
        description="Process queries in parallel"
    )


class BatchQueryResponse(BaseModel):
    """Response model for batch queries"""
    responses: List[QueryResponse] = Field(
        default_factory=list,
        description="List of query responses"
    )
    total_queries: int = Field(..., ge=0, description="Total queries processed")
    successful: int = Field(..., ge=0, description="Successfully processed")
    failed: int = Field(..., ge=0, description="Failed queries")
    total_time_ms: float = Field(..., ge=0.0, description="Total processing time")


class ConversationMessage(BaseModel):
    """A message in a conversation"""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="ISO timestamp")
    query_id: Optional[str] = Field(None, description="Associated query ID")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Ensure role is valid"""
        valid_roles = ['user', 'assistant', 'system']
        if v.lower() not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}")
        return v.lower()


class ConversationHistory(BaseModel):
    """Full conversation history"""
    conversation_id: str = Field(..., description="Unique conversation identifier")
    messages: List[ConversationMessage] = Field(
        default_factory=list,
        description="List of messages"
    )
    created_at: Optional[str] = Field(None, description="Conversation start time")
    updated_at: Optional[str] = Field(None, description="Last update time")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional conversation metadata"
    )


class ConversationSummary(BaseModel):
    """Summary of a conversation"""
    conversation_id: str = Field(..., description="Conversation identifier")
    summary: str = Field(..., description="Conversation summary")
    key_topics: List[str] = Field(
        default_factory=list,
        description="Key topics discussed"
    )
    message_count: int = Field(..., ge=0, description="Number of messages")
    created_at: Optional[str] = Field(None, description="Creation timestamp")