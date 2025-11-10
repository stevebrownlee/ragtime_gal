"""
Ragtime Models Package

Pydantic models for type-safe data validation and serialization.

This package provides models for:
- Documents: Document metadata, chunks, upload/download operations
- Queries: Query requests, responses, conversation history
- Feedback: User feedback collection and analytics
- Training: Training data generation and model fine-tuning
- Responses: Standardized API responses and error handling

Usage:
    from ragtime.models import QueryRequest, QueryResponse
    from ragtime.models import DocumentMetadata, FeedbackRequest
    from ragtime.models import Settings
"""

# Document models
from ragtime.models.documents import (
    DocumentMetadata,
    DocumentChunk,
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentDeleteRequest,
    DocumentSearchRequest,
    DocumentSearchResponse,
    CollectionInfo,
    CollectionListResponse,
)

# Query models
from ragtime.models.queries import (
    EnhancementMode,
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    QueryEnhancementSuggestion,
    BatchQueryRequest,
    BatchQueryResponse,
    ConversationMessage,
    ConversationHistory,
    ConversationSummary,
)

# Feedback models
from ragtime.models.feedback import (
    FeedbackDimensions,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackEntry,
    FeedbackPattern,
    FeedbackAnalytics,
    FeedbackSummaryRequest,
    FeedbackListResponse,
    FeedbackStatistics,
)

# Training models
from ragtime.models.training import (
    TrainingStatus,
    TrainingDataConfig,
    FineTuningConfig,
    TrainingDataPair,
    TrainingDataset,
    TrainingJob,
    TrainingJobRequest,
    TrainingJobResponse,
    ABTestConfig,
    ABTestResult,
)

# Response models
from ragtime.models.responses import (
    ResponseStatus,
    APIResponse,
    ErrorResponse,
    ValidationErrorDetail,
    ValidationErrorResponse,
    HealthStatus,
    ComponentHealth,
    HealthCheckResponse,
    PaginationInfo,
    PaginatedResponse,
    BatchOperationResult,
    MetricsResponse,
    CacheInfo,
    SystemInfoResponse,
)

# Settings
from ragtime.config.settings import Settings, get_settings, reload_settings

__all__ = [
    # Document models
    "DocumentMetadata",
    "DocumentChunk",
    "DocumentUploadRequest",
    "DocumentUploadResponse",
    "DocumentDeleteRequest",
    "DocumentSearchRequest",
    "DocumentSearchResponse",
    "CollectionInfo",
    "CollectionListResponse",
    # Query models
    "EnhancementMode",
    "QueryRequest",
    "QueryResponse",
    "RetrievedDocument",
    "QueryEnhancementSuggestion",
    "BatchQueryRequest",
    "BatchQueryResponse",
    "ConversationMessage",
    "ConversationHistory",
    "ConversationSummary",
    # Feedback models
    "FeedbackDimensions",
    "FeedbackRequest",
    "FeedbackResponse",
    "FeedbackEntry",
    "FeedbackPattern",
    "FeedbackAnalytics",
    "FeedbackSummaryRequest",
    "FeedbackListResponse",
    "FeedbackStatistics",
    # Training models
    "TrainingStatus",
    "TrainingDataConfig",
    "FineTuningConfig",
    "TrainingDataPair",
    "TrainingDataset",
    "TrainingJob",
    "TrainingJobRequest",
    "TrainingJobResponse",
    "ABTestConfig",
    "ABTestResult",
    # Response models
    "ResponseStatus",
    "APIResponse",
    "ErrorResponse",
    "ValidationErrorDetail",
    "ValidationErrorResponse",
    "HealthStatus",
    "ComponentHealth",
    "HealthCheckResponse",
    "PaginationInfo",
    "PaginatedResponse",
    "BatchOperationResult",
    "MetricsResponse",
    "CacheInfo",
    "SystemInfoResponse",
    # Settings
    "Settings",
    "get_settings",
    "reload_settings",
]