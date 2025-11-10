"""
Response Models

Pydantic models for standardized API responses, including success, error, and health check responses.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class ResponseStatus(str, Enum):
    """Standard response statuses"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PENDING = "pending"


class APIResponse(BaseModel):
    """Standard API response wrapper"""
    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="Human-readable message")
    data: Optional[Any] = Field(None, description="Response data")
    errors: Optional[List[str]] = Field(
        None,
        description="List of errors (if any)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request identifier for tracing"
    )


class ErrorResponse(BaseModel):
    """Error response model"""
    status: ResponseStatus = Field(
        default=ResponseStatus.ERROR,
        description="Always 'error'"
    )
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    request_id: Optional[str] = None
    stack_trace: Optional[str] = Field(
        None,
        description="Stack trace (only in debug mode)"
    )


class ValidationErrorDetail(BaseModel):
    """Detail for a validation error"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Optional[Any] = Field(None, description="Invalid value provided")


class ValidationErrorResponse(BaseModel):
    """Response for validation errors"""
    status: ResponseStatus = Field(
        default=ResponseStatus.ERROR,
        description="Always 'error'"
    )
    error_code: str = Field(
        default="VALIDATION_ERROR",
        description="Always 'VALIDATION_ERROR'"
    )
    message: str = Field(
        default="Validation failed",
        description="Error message"
    )
    errors: List[ValidationErrorDetail] = Field(
        default_factory=list,
        description="List of validation errors"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthStatus(str, Enum):
    """Health check statuses"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a system component"""
    name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Component health status")
    message: Optional[str] = Field(None, description="Status message")
    latency_ms: Optional[float] = Field(
        None,
        ge=0,
        description="Component response latency"
    )
    last_check: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last health check time"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional component metadata"
    )


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint"""
    status: HealthStatus = Field(..., description="Overall system health")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Check timestamp"
    )
    uptime_seconds: Optional[float] = Field(
        None,
        ge=0,
        description="System uptime in seconds"
    )
    version: Optional[str] = Field(None, description="Application version")
    components: List[ComponentHealth] = Field(
        default_factory=list,
        description="Individual component health"
    )
    system_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="System-level metrics (CPU, memory, etc.)"
    )


class PaginationInfo(BaseModel):
    """Pagination information for list responses"""
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_items: int = Field(..., ge=0, description="Total items available")
    total_pages: int = Field(..., ge=0, description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_previous: bool = Field(..., description="Has previous page")


class PaginatedResponse(BaseModel):
    """Response for paginated list endpoints"""
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="Response status"
    )
    data: List[Any] = Field(
        default_factory=list,
        description="List of items"
    )
    pagination: PaginationInfo = Field(..., description="Pagination info")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchOperationResult(BaseModel):
    """Result of a batch operation"""
    total: int = Field(..., ge=0, description="Total items processed")
    successful: int = Field(..., ge=0, description="Successfully processed")
    failed: int = Field(..., ge=0, description="Failed to process")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Errors for failed items"
    )
    duration_ms: float = Field(..., ge=0, description="Operation duration")


class MetricsResponse(BaseModel):
    """Response for metrics endpoint"""
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Metrics timestamp"
    )
    period_start: datetime = Field(..., description="Metrics period start")
    period_end: datetime = Field(..., description="Metrics period end")
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Collected metrics"
    )


class CacheInfo(BaseModel):
    """Cache statistics"""
    enabled: bool = Field(..., description="Whether caching is enabled")
    size: int = Field(..., ge=0, description="Current cache size")
    max_size: int = Field(..., ge=0, description="Maximum cache size")
    hit_count: int = Field(..., ge=0, description="Cache hits")
    miss_count: int = Field(..., ge=0, description="Cache misses")
    hit_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cache hit rate (0-1)"
    )
    eviction_count: int = Field(..., ge=0, description="Items evicted")


class SystemInfoResponse(BaseModel):
    """System information response"""
    version: str = Field(..., description="Application version")
    python_version: str = Field(..., description="Python version")
    platform: str = Field(..., description="Operating system platform")
    uptime_seconds: float = Field(..., ge=0, description="System uptime")
    cache_info: Optional[CacheInfo] = Field(None, description="Cache statistics")
    database_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Database information"
    )
    model_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model information"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)