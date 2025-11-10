"""
Training Models

Pydantic models for training data generation, model fine-tuning, and A/B testing.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class TrainingStatus(str, Enum):
    """Training job statuses"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingDataConfig(BaseModel):
    """
    Configuration for training data generation from user feedback.

    This class controls how the system learns from user feedback to improve
    the embedding model through fine-tuning. Understanding these parameters
    helps non-ML experts optimize model training for their specific use case.
    """

    positive_threshold: int = Field(default=4, ge=1, le=5)
    """
    Rating threshold for positive training examples (default: 4/5 stars).

    **What it does**: Feedback with ratings >= this value are treated as "good"
    query-document pairs that the model should learn to retrieve.

    **Why it matters**: Sets the quality bar for training examples. A higher
    threshold (5) means only perfect responses train the model, while a lower
    threshold (3-4) provides more training data but with varying quality.

    **For your book project**: Use 4 to balance quality and quantity. Chapter
    retrieval that users rate 4+ stars indicates the embedding model correctly
    understood the semantic relationship between query and content.
    """

    negative_threshold: int = Field(default=2, ge=1, le=5)
    """
    Rating threshold for negative training examples (default: 2/5 stars).

    **What it does**: Feedback with ratings <= this value are treated as "bad"
    query-document pairs that the model should learn NOT to retrieve.

    **Why it matters**: Helps the model learn what NOT to do. If users
    consistently rate certain chapter retrievals poorly, the model learns
    to avoid those pairings.

    **For your book project**: Keep at 2 to identify truly poor matches. When
    users query about "character motivations" but get unrelated world-building
    notes rated 1-2 stars, the model learns to distinguish better.
    """

    hard_negative_similarity: float = Field(default=0.7, ge=0.0, le=1.0)
    """
    Similarity score threshold for "hard negative" examples (default: 0.7).

    **What it does**: Identifies documents that are semantically similar to the
    query but still got low ratings. These are "hard" because they LOOK relevant
    but aren't actually helpful.

    **Why it matters**: Hard negatives are the most valuable training data. They
    teach the model subtle distinctions. Easy negatives (totally unrelated
    documents) don't improve the model much.

    **For your book project**: At 0.7, this captures cases where the model
    retrieves a chapter about "betrayal" when you asked about "trust" -
    semantically related but contextually wrong. The model learns nuanced
    differences between similar concepts in your narrative.
    """

    min_positive_pairs: int = Field(default=50, ge=1)
    """
    Minimum number of positive training pairs required (default: 50).

    **What it does**: Sets a quality threshold - the model won't train until
    you have at least this many high-quality query-document pairs.

    **Why it matters**: Too few examples lead to overfitting (model memorizes
    instead of learning patterns). Too many delays training. 50 is a reasonable
    starting point for fine-tuning.

    **For your book project**: After about 50 queries rated 4+ stars, the model
    has seen enough positive patterns to start learning your narrative structure,
    character relationships, and thematic connections.
    """

    min_negative_pairs: int = Field(default=50, ge=1)
    """
    Minimum number of negative training pairs required (default: 50).

    **What it does**: Ensures the model learns from enough "bad" examples
    before training begins.

    **Why it matters**: Models need to learn what to avoid as much as what to
    seek. Balanced positive/negative examples prevent the model from becoming
    too eager (retrieving everything) or too conservative (retrieving nothing).

    **For your book project**: 50 negative examples teach the model to distinguish
    between superficially similar but contextually different content (e.g., two
    characters with similar arcs but different narrative purposes).
    """

    include_hard_negatives: bool = Field(
        default=True,
        description="Include hard negative examples in training"
    )
    max_pairs_per_query: int = Field(
        default=10,
        ge=1,
        description="Maximum training pairs to generate per query"
    )


class FineTuningConfig(BaseModel):
    """Configuration for model fine-tuning"""
    base_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Base embedding model to fine-tune"
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Training batch size"
    )
    num_epochs: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=2e-5,
        gt=0.0,
        lt=1.0,
        description="Learning rate for optimizer"
    )
    warmup_steps: int = Field(
        default=100,
        ge=0,
        description="Number of warmup steps"
    )
    output_path: Optional[str] = Field(
        None,
        description="Path to save fine-tuned model"
    )
    use_amp: bool = Field(
        default=True,
        description="Use automatic mixed precision"
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="Gradient accumulation steps"
    )
    max_grad_norm: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum gradient norm for clipping"
    )

    @field_validator('output_path', mode='before')
    @classmethod
    def set_default_output_path(cls, v):
        """Set default output path if not provided"""
        if v is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            return f"models/finetuned_{timestamp}"
        return v


class TrainingDataPair(BaseModel):
    """A single training data pair"""
    query: str = Field(..., description="Query text")
    positive_doc: str = Field(..., description="Positive (relevant) document")
    negative_doc: Optional[str] = Field(
        None,
        description="Negative (irrelevant) document"
    )
    score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Relevance score"
    )
    is_hard_negative: bool = Field(
        default=False,
        description="Whether negative is a hard negative"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class TrainingDataset(BaseModel):
    """Collection of training data pairs"""
    pairs: List[TrainingDataPair] = Field(
        default_factory=list,
        description="Training pairs"
    )
    config: TrainingDataConfig = Field(
        default_factory=TrainingDataConfig,
        description="Configuration used to generate data"
    )
    collection_name: str = Field(..., description="Source collection")
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Dataset generation timestamp"
    )
    statistics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dataset statistics"
    )


class TrainingJob(BaseModel):
    """Represents a training job"""
    job_id: str = Field(..., description="Unique job identifier")
    status: TrainingStatus = Field(
        default=TrainingStatus.PENDING,
        description="Current job status"
    )
    config: FineTuningConfig = Field(..., description="Fine-tuning configuration")
    dataset_id: Optional[str] = Field(
        None,
        description="Training dataset identifier"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Job creation time"
    )
    started_at: Optional[datetime] = Field(
        None,
        description="Job start time"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Job completion time"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Training metrics (loss, accuracy, etc.)"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if failed"
    )
    model_path: Optional[str] = Field(
        None,
        description="Path to trained model"
    )
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Training progress (0-1)"
    )


class TrainingJobRequest(BaseModel):
    """Request to start a training job"""
    dataset_id: Optional[str] = Field(
        None,
        description="Pre-generated dataset ID (or will generate)"
    )
    training_config: FineTuningConfig = Field(
        default_factory=FineTuningConfig,
        description="Fine-tuning configuration"
    )
    data_config: Optional[TrainingDataConfig] = Field(
        None,
        description="Data generation config (if dataset not provided)"
    )
    collection_name: Optional[str] = Field(
        None,
        description="Collection to generate data from"
    )


class TrainingJobResponse(BaseModel):
    """Response for training job request"""
    job_id: str = Field(..., description="Assigned job ID")
    status: TrainingStatus = Field(..., description="Initial status")
    message: str = Field(..., description="Status message")
    estimated_time_minutes: Optional[int] = Field(
        None,
        description="Estimated completion time"
    )


class ABTestConfig(BaseModel):
    """Configuration for A/B testing models"""
    test_name: str = Field(..., description="Name of the A/B test")
    model_a: str = Field(..., description="Path to model A")
    model_b: str = Field(..., description="Path to model B")
    traffic_split: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Traffic percentage for model A (0-1)"
    )
    collection_name: str = Field(..., description="Collection to test on")
    min_queries: int = Field(
        default=100,
        ge=10,
        description="Minimum queries before analysis"
    )
    duration_hours: Optional[int] = Field(
        None,
        ge=1,
        description="Test duration in hours"
    )


class ABTestResult(BaseModel):
    """Results from an A/B test"""
    test_id: str = Field(..., description="Test identifier")
    model_a_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Model A performance metrics"
    )
    model_b_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Model B performance metrics"
    )
    winner: Optional[str] = Field(
        None,
        description="Winning model (A or B)"
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Statistical confidence in result"
    )
    queries_processed: int = Field(..., ge=0, description="Total queries in test")
    recommendation: str = Field(..., description="Recommendation based on results")