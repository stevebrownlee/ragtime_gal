"""
Feedback Models

Pydantic models for user feedback collection, analysis, and analytics.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, conint, field_validator


class FeedbackDimensions(BaseModel):
    """Detailed feedback dimensions for multi-aspect rating"""
    relevance: Optional[conint(ge=1, le=5)] = Field(
        None,
        description="How relevant was the answer? (1-5)"
    )
    completeness: Optional[conint(ge=1, le=5)] = Field(
        None,
        description="How complete was the answer? (1-5)"
    )
    length: Optional[conint(ge=1, le=5)] = Field(
        None,
        description="Was the answer length appropriate? (1-5)"
    )
    accuracy: Optional[conint(ge=1, le=5)] = Field(
        None,
        description="How accurate was the answer? (1-5)"
    )
    clarity: Optional[conint(ge=1, le=5)] = Field(
        None,
        description="How clear was the answer? (1-5)"
    )


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback"""
    query_id: str = Field(..., description="ID of the query being rated")
    rating: conint(ge=1, le=5) = Field(..., description="Overall rating (1-5)")
    dimensions: Optional[FeedbackDimensions] = Field(
        None,
        description="Optional detailed dimension ratings"
    )
    comments: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional user comments"
    )
    session_id: str = Field(..., description="Session identifier")
    conversation_id: Optional[str] = Field(
        None,
        description="Conversation identifier"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Optional tags for categorization"
    )

    @field_validator('comments')
    @classmethod
    def comments_not_just_whitespace(cls, v: Optional[str]) -> Optional[str]:
        """Ensure comments are not just whitespace if provided"""
        if v is not None:
            v_stripped = v.strip()
            return v_stripped if v_stripped else None
        return v


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    feedback_id: str = Field(..., description="Unique feedback identifier")
    message: str = Field(..., description="Confirmation message")
    stored_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of storage"
    )


class FeedbackEntry(BaseModel):
    """Complete feedback entry with all context"""
    feedback_id: str = Field(..., description="Unique identifier")
    query_id: str = Field(..., description="Associated query ID")
    query_text: Optional[str] = Field(None, description="Original query text")
    answer_text: Optional[str] = Field(None, description="Generated answer text")
    rating: int = Field(..., ge=1, le=5, description="Overall rating")
    dimensions: Optional[FeedbackDimensions] = None
    comments: Optional[str] = None
    session_id: str = Field(..., description="Session identifier")
    conversation_id: Optional[str] = None
    retrieved_docs: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Documents that were retrieved"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeedbackPattern(BaseModel):
    """Identified pattern in feedback data"""
    pattern_type: str = Field(..., description="Type of pattern (e.g., 'low_rating_query_type')")
    description: str = Field(..., description="Human-readable pattern description")
    frequency: int = Field(..., ge=0, description="How often this pattern occurs")
    average_rating: float = Field(..., ge=1.0, le=5.0, description="Average rating for this pattern")
    examples: List[str] = Field(
        default_factory=list,
        description="Example queries showing this pattern"
    )
    recommendation: Optional[str] = Field(
        None,
        description="Recommended action to address this pattern"
    )


class FeedbackAnalytics(BaseModel):
    """Analytics from feedback data"""
    total_feedback_count: int = Field(..., ge=0, description="Total feedback entries")
    average_rating: float = Field(..., ge=0.0, le=5.0, description="Average overall rating")
    rating_distribution: Dict[int, int] = Field(
        default_factory=dict,
        description="Count of each rating (1-5)"
    )
    positive_patterns: List[FeedbackPattern] = Field(
        default_factory=list,
        description="Patterns associated with positive feedback"
    )
    negative_patterns: List[FeedbackPattern] = Field(
        default_factory=list,
        description="Patterns associated with negative feedback"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations"
    )
    dimension_averages: Optional[Dict[str, float]] = Field(
        None,
        description="Average scores for each feedback dimension"
    )
    time_period_start: Optional[datetime] = Field(
        None,
        description="Start of analysis period"
    )
    time_period_end: Optional[datetime] = Field(
        None,
        description="End of analysis period"
    )


class FeedbackSummaryRequest(BaseModel):
    """Request for feedback analytics summary"""
    collection_name: Optional[str] = Field(
        None,
        description="Filter by collection"
    )
    start_date: Optional[datetime] = Field(
        None,
        description="Start date for analysis"
    )
    end_date: Optional[datetime] = Field(
        None,
        description="End date for analysis"
    )
    min_rating: Optional[conint(ge=1, le=5)] = Field(
        None,
        description="Minimum rating filter"
    )
    max_rating: Optional[conint(ge=1, le=5)] = Field(
        None,
        description="Maximum rating filter"
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Filter by tags"
    )

    @field_validator('end_date')
    @classmethod
    def end_after_start(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Ensure end_date is after start_date"""
        start_date = info.data.get('start_date')
        if v and start_date and v < start_date:
            raise ValueError('end_date must be after start_date')
        return v


class FeedbackListResponse(BaseModel):
    """Response for listing feedback entries"""
    feedback_entries: List[FeedbackEntry] = Field(
        default_factory=list,
        description="List of feedback entries"
    )
    total_count: int = Field(..., ge=0, description="Total matching entries")
    page: int = Field(default=1, ge=1, description="Current page")
    page_size: int = Field(default=50, ge=1, le=100, description="Entries per page")
    has_more: bool = Field(default=False, description="More entries available")


class FeedbackStatistics(BaseModel):
    """Statistical summary of feedback"""
    total_feedback: int = Field(..., ge=0)
    average_rating: float = Field(..., ge=0.0, le=5.0)
    median_rating: float = Field(..., ge=0.0, le=5.0)
    rating_std_dev: float = Field(..., ge=0.0)
    positive_count: int = Field(..., ge=0, description="Ratings >= 4")
    neutral_count: int = Field(..., ge=0, description="Rating == 3")
    negative_count: int = Field(..., ge=0, description="Ratings <= 2")
    response_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Percentage of queries that received feedback"
    )