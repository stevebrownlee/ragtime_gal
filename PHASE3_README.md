# Phase 3: Model Fine-tuning System

## Overview

Phase 3 implements an automated system for fine-tuning embedding models using feedback data collected through Phase 1. This phase transforms user feedback into training data and uses it to continuously improve the semantic understanding of the RAG system's embedding model.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Phase 3: Model Fine-tuning System                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Training Data Pipeline                      │  │
│  │                                                                 │  │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐      │  │
│  │  │   ConPort   │───▶│  Training    │───▶│  Training   │      │  │
│  │  │  Feedback   │    │    Data      │    │   Pairs     │      │  │
│  │  │   Storage   │    │  Generator   │    │  (Positive/ │      │  │
│  │  │             │    │              │    │  Negative)  │      │  │
│  │  └─────────────┘    └──────────────┘    └─────────────┘      │  │
│  │                                                                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Model Fine-tuning Pipeline                  │  │
│  │                                                                 │  │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐      │  │
│  │  │  Training   │───▶│    Model     │───▶│ Fine-tuned  │      │  │
│  │  │    Pairs    │    │  Fine-tuner  │    │   Model     │      │  │
│  │  │             │    │              │    │             │      │  │
│  │  └─────────────┘    └──────────────┘    └─────────────┘      │  │
│  │                            │                      │            │  │
│  │                            ▼                      ▼            │  │
│  │                    ┌──────────────┐    ┌─────────────┐       │  │
│  │                    │  Validation  │    │   Ollama    │       │  │
│  │                    │  & Metrics   │    │ Conversion  │       │  │
│  │                    └──────────────┘    └─────────────┘       │  │
│  │                                                                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              A/B Testing & Model Comparison                    │  │
│  │                                                                 │  │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐      │  │
│  │  │  Baseline   │    │     A/B      │    │  Winner     │      │  │
│  │  │   Model     │───▶│   Testing    │───▶│  Selection  │      │  │
│  │  │             │    │   Framework  │    │             │      │  │
│  │  └─────────────┘    └──────────────┘    └─────────────┘      │  │
│  │         │                    │                    │            │  │
│  │         ▼                    ▼                    ▼            │  │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐      │  │
│  │  │ Fine-tuned  │    │  Performance │    │  Production │      │  │
│  │  │   Model     │    │   Metrics    │    │ Deployment  │      │  │
│  │  └─────────────┘    └──────────────┘    └─────────────┘      │  │
│  │                                                                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Training Data Generator

**File**: [`training_data_generator.py`](training_data_generator.py)

Transforms user feedback from ConPort into training pairs suitable for fine-tuning sentence transformer models.

**Features**:
- Extracts feedback data from ConPort's UserFeedback category
- Generates positive pairs from highly-rated interactions (rating ≥ 4)
- Generates negative pairs from low-rated interactions (rating ≤ 2)
- Implements hard negative mining for improved model discrimination
- Supports synthetic data generation for data augmentation
- Configurable rating thresholds and pair limits

**Key Classes**:
- `TrainingDataGenerator`: Main class for training data generation
- `TrainingPair`: Data structure representing a query-document training pair

**Usage Example**:
```python
from training_data_generator import TrainingDataGenerator, create_training_data_generator
from conport_client import ConPortClient

# Initialize with ConPort client
conport_client = ConPortClient()
workspace_id = "/path/to/workspace"

generator = create_training_data_generator(
    conport_client=conport_client,
    workspace_id=workspace_id,
    chroma_db=chroma_db,
    embedding_model="all-MiniLM-L6-v2"
)

# Generate training data from feedback
training_pairs = generator.generate_training_data(
    min_positive_samples=50,
    min_negative_samples=50,
    include_hard_negatives=True
)

# Export to various formats
generator.export_training_data(training_pairs, "training_data.csv", format_type="csv")
generator.export_training_data(training_pairs, "training_data.json", format_type="json")
```

### 2. Model Fine-tuner

**File**: [`model_finetuner.py`](model_finetuner.py)

Implements the complete fine-tuning pipeline for sentence-transformer embedding models using training data generated from user feedback.

**Features**:
- Loads and fine-tunes sentence-transformer models
- Supports multiple loss functions (CosineSimilarity, Triplet, MultipleNegativesRanking)
- Implements validation and early stopping
- Tracks training metrics and history
- Saves trained models with metadata
- Converts models to Ollama format for deployment
- Provides model evaluation on test data

**Key Classes**:
- `ModelFineTuner`: Main fine-tuning orchestrator
- `FineTuningConfig`: Configuration dataclass for training parameters
- `TrainingMetrics`: Tracks training progress and evaluation scores

**Usage Example**:
```python
from model_finetuner import ModelFineTuner, FineTuningConfig, create_model_finetuner

# Configure fine-tuning
config = FineTuningConfig(
    base_model_name="all-MiniLM-L6-v2",
    output_model_path="./fine_tuned_models",
    batch_size=16,
    num_epochs=4,
    learning_rate=2e-5,
    loss_function="CosineSimilarityLoss"
)

# Create fine-tuner
finetuner = create_model_finetuner(config=config)

# Load base model
finetuner.load_base_model()

# Load training data
train_examples, val_examples = finetuner.load_training_data(
    "training_data.csv",
    format_type="csv",
    validation_split=0.2
)

# Fine-tune the model
result = finetuner.fine_tune_model(
    train_examples=train_examples,
    val_examples=val_examples,
    model_name_suffix="feedback_v1"
)

print(f"Model saved to: {result['model_path']}")
print(f"Training duration: {result['training_duration']:.2f} seconds")
```

### 3. A/B Testing Framework

**Status**: 🚧 **TO BE IMPLEMENTED**

The A/B testing framework will enable systematic comparison of baseline and fine-tuned models to ensure improvements before deployment.

**Planned Features**:
- Parallel model deployment for comparison
- Automated traffic splitting between models
- Statistical significance testing
- Performance metric collection and comparison
- Winner selection based on configurable criteria

## API Endpoints

The following Flask API endpoints provide programmatic access to Phase 3 functionality:

### Training Data Generation

#### `POST /training/generate-data`

Generate training data from user feedback.

**Request Body**:
```json
{
  "min_positive_samples": 50,
  "min_negative_samples": 50,
  "include_hard_negatives": true,
  "days_back": 90,
  "export_format": "csv"
}
```

**Response**:
```json
{
  "success": true,
  "training_data_path": "./training_data/training_20240108_143022.csv",
  "statistics": {
    "total_pairs": 150,
    "positive_pairs": 75,
    "negative_pairs": 75,
    "hard_negative_pairs": 25
  }
}
```

**Status**: 🚧 **TO BE IMPLEMENTED**

### Model Fine-tuning

#### `POST /training/fine-tune`

Start a model fine-tuning job.

**Request Body**:
```json
{
  "training_data_path": "./training_data/training_20240108_143022.csv",
  "base_model": "all-MiniLM-L6-v2",
  "model_name_suffix": "feedback_v1",
  "config": {
    "batch_size": 16,
    "num_epochs": 4,
    "learning_rate": 2e-5,
    "loss_function": "CosineSimilarityLoss"
  }
}
```

**Response**:
```json
{
  "success": true,
  "job_id": "ft_20240108_143022",
  "status": "running",
  "message": "Fine-tuning job started"
}
```

**Status**: 🚧 **TO BE IMPLEMENTED**

#### `GET /training/status/<job_id>`

Check the status of a fine-tuning job.

**Response**:
```json
{
  "job_id": "ft_20240108_143022",
  "status": "completed",
  "progress": 100,
  "model_path": "./fine_tuned_models/feedback_v1_20240108_143022",
  "training_duration": 1234.56,
  "metrics": {
    "final_train_loss": 0.123,
    "final_eval_score": 0.876
  }
}
```

**Status**: 🚧 **TO BE IMPLEMENTED**

### A/B Testing

#### `POST /testing/ab-test/start`

Start an A/B test comparing two models.

**Request Body**:
```json
{
  "baseline_model": "all-MiniLM-L6-v2",
  "test_model": "./fine_tuned_models/feedback_v1_20240108_143022",
  "traffic_split": 0.5,
  "duration_hours": 24,
  "metrics": ["response_quality", "relevance_score", "user_satisfaction"]
}
```

**Response**:
```json
{
  "success": true,
  "test_id": "ab_20240108_143022",
  "status": "active",
  "start_time": "2024-01-08T14:30:22Z",
  "estimated_end_time": "2024-01-09T14:30:22Z"
}
```

**Status**: 🚧 **TO BE IMPLEMENTED**

#### `GET /testing/ab-test/<test_id>/results`

Get results of an A/B test.

**Response**:
```json
{
  "test_id": "ab_20240108_143022",
  "status": "completed",
  "baseline_metrics": {
    "avg_response_quality": 3.8,
    "avg_relevance_score": 0.72,
    "avg_user_satisfaction": 3.9
  },
  "test_metrics": {
    "avg_response_quality": 4.2,
    "avg_relevance_score": 0.81,
    "avg_user_satisfaction": 4.3
  },
  "statistical_significance": true,
  "p_value": 0.023,
  "recommendation": "deploy_test_model"
}
```

**Status**: 🚧 **TO BE IMPLEMENTED**

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Fine-tuning Configuration
FINETUNING_ENABLED=true
BASE_MODEL_NAME=all-MiniLM-L6-v2
FINETUNED_MODELS_PATH=./fine_tuned_models
TRAINING_DATA_PATH=./training_data

# Training Parameters
BATCH_SIZE=16
NUM_EPOCHS=4
LEARNING_RATE=2e-5
LOSS_FUNCTION=CosineSimilarityLoss
MAX_SEQ_LENGTH=512
USE_AMP=true

# Data Generation
MIN_POSITIVE_SAMPLES=50
MIN_NEGATIVE_SAMPLES=50
TRAINING_DATA_MIN_SAMPLES=100
INCLUDE_HARD_NEGATIVES=true
POSITIVE_RATING_THRESHOLD=4
NEGATIVE_RATING_THRESHOLD=2

# A/B Testing
AB_TESTING_ENABLED=true
AB_TEST_DURATION_HOURS=24
AB_TEST_TRAFFIC_SPLIT=0.5
AB_TEST_SIGNIFICANCE_LEVEL=0.05
```

### Fine-tuning Configuration

The `FineTuningConfig` dataclass allows programmatic configuration:

```python
from model_finetuner import FineTuningConfig

config = FineTuningConfig(
    base_model_name="all-MiniLM-L6-v2",
    output_model_path="./fine_tuned_models",
    batch_size=16,
    num_epochs=4,
    learning_rate=2e-5,
    warmup_steps=100,
    evaluation_steps=500,
    save_steps=500,
    max_seq_length=512,
    loss_function="CosineSimilarityLoss",
    use_amp=True,
    early_stopping_patience=3,
    validation_split=0.2,
    random_seed=42
)
```

## Integration with ConPort

Phase 3 deeply integrates with ConPort's feedback storage system:

1. **Feedback Retrieval**: `TrainingDataGenerator` queries ConPort for user feedback data stored in the `UserFeedback` category
2. **Quality Filtering**: Only uses feedback with valid ratings and sufficient context
3. **Training Tracking**: Fine-tuning jobs and results can be logged to ConPort as custom data
4. **Model Versioning**: Model metadata stored alongside training history

### Example ConPort Integration

```python
from conport_client import ConPortClient
from training_data_generator import create_training_data_generator
from model_finetuner import create_model_finetuner

# Initialize ConPort client
conport_client = ConPortClient()
workspace_id = "/path/to/workspace"

# Generate training data from ConPort feedback
generator = create_training_data_generator(
    conport_client=conport_client,
    workspace_id=workspace_id
)

feedback_data = generator.get_feedback_data(days_back=90, min_samples=100)
print(f"Retrieved {len(feedback_data)} feedback entries from ConPort")

training_pairs = generator.generate_training_data_from_feedback(feedback_data)
print(f"Generated {len(training_pairs)} training pairs")

# Log training job to ConPort
conport_client.log_custom_data({
    "workspace_id": workspace_id,
    "category": "ModelTraining",
    "key": f"training_job_{timestamp}",
    "value": {
        "status": "started",
        "training_pairs": len(training_pairs),
        "base_model": "all-MiniLM-L6-v2"
    }
})
```

## Deployment Workflow

### 1. Generate Training Data

```bash
# Via API (when implemented)
curl -X POST http://localhost:8084/training/generate-data \
  -H "Content-Type: application/json" \
  -d '{
    "min_positive_samples": 50,
    "min_negative_samples": 50,
    "include_hard_negatives": true
  }'

# Or via Python
python -c "
from training_data_generator import create_training_data_generator
from conport_client import ConPortClient

client = ConPortClient()
generator = create_training_data_generator(
    conport_client=client,
    workspace_id='$PWD'
)
pairs = generator.generate_training_data()
generator.export_training_data(pairs, 'training_data.csv')
"
```

### 2. Fine-tune Model

```bash
# Via API (when implemented)
curl -X POST http://localhost:8084/training/fine-tune \
  -H "Content-Type: application/json" \
  -d '{
    "training_data_path": "./training_data.csv",
    "base_model": "all-MiniLM-L6-v2"
  }'

# Or via Python
python -c "
from model_finetuner import create_model_finetuner, FineTuningConfig

config = FineTuningConfig(base_model_name='all-MiniLM-L6-v2')
finetuner = create_model_finetuner(config=config)
finetuner.load_base_model()

train_examples, val_examples = finetuner.load_training_data('training_data.csv')
result = finetuner.fine_tune_model(train_examples, val_examples)
print(f'Model: {result[\"model_path\"]}')
"
```

### 3. Convert to Ollama Format

```python
from model_finetuner import create_model_finetuner

finetuner = create_model_finetuner()
result = finetuner.convert_to_ollama_format(
    model_path="./fine_tuned_models/feedback_v1_20240108",
    ollama_model_name="ragtime-feedback-v1"
)

# Follow instructions in result['instructions_path']
```

### 4. Deploy and A/B Test

```bash
# Start A/B test (when implemented)
curl -X POST http://localhost:8084/testing/ab-test/start \
  -H "Content-Type: application/json" \
  -d '{
    "baseline_model": "all-MiniLM-L6-v2",
    "test_model": "./fine_tuned_models/feedback_v1_20240108",
    "duration_hours": 24
  }'
```

## Best Practices

### Training Data Quality

1. **Minimum Sample Size**: Collect at least 100 feedback samples before fine-tuning
2. **Balanced Data**: Maintain roughly equal positive and negative examples
3. **Hard Negatives**: Include hard negative examples to improve discrimination
4. **Regular Updates**: Re-train periodically as new feedback accumulates

### Fine-tuning Strategy

1. **Start Small**: Begin with 2-4 epochs and monitor validation metrics
2. **Learning Rate**: Start with 2e-5, adjust based on convergence
3. **Validation Split**: Use 20% of data for validation to prevent overfitting
4. **Early Stopping**: Enable early stopping to avoid overtraining

### A/B Testing

1. **Baseline First**: Always establish baseline metrics before testing
2. **Sufficient Duration**: Run tests for at least 24-48 hours
3. **Statistical Significance**: Only deploy if p-value < 0.05
4. **Monitor Metrics**: Track multiple metrics (quality, relevance, satisfaction)

### Production Deployment

1. **Backup Models**: Keep previous model versions for rollback
2. **Gradual Rollout**: Use traffic splitting for safe deployment
3. **Monitoring**: Track model performance in production
4. **Feedback Loop**: Continue collecting feedback on new model

## Performance Metrics

### Training Metrics

- **Train Loss**: Loss on training data (lower is better)
- **Validation Score**: Cosine similarity accuracy on validation set
- **Correlation**: Correlation between predicted and actual similarities

### Evaluation Metrics

- **Embedding Quality**: Cosine/Manhattan/Euclidean accuracy
- **Retrieval Performance**: Precision@K, Recall@K, MRR
- **User Satisfaction**: Average rating, feedback sentiment

### A/B Test Metrics

- **Response Quality**: Average user rating
- **Relevance Score**: Document retrieval accuracy
- **User Satisfaction**: Overall satisfaction rating
- **Statistical Significance**: P-value for metric improvements

## Troubleshooting

### Common Issues

**Issue**: "sentence-transformers not available"
```bash
Solution: pip install sentence-transformers
```

**Issue**: "Insufficient training data"
```bash
Solution: Collect more user feedback before attempting fine-tuning
Minimum: 100 samples (50 positive, 50 negative)
```

**Issue**: "Model overfitting"
```bash
Solution:
- Increase validation_split to 0.3
- Reduce num_epochs
- Enable early_stopping_patience
- Add more training data
```

**Issue**: "Out of memory during training"
```bash
Solution:
- Reduce batch_size (try 8 or 4)
- Reduce max_seq_length (try 256)
- Disable use_amp if needed
- Use a smaller base model
```

## Future Enhancements

### Planned Features

1. **Multi-task Fine-tuning**: Train on multiple objectives simultaneously
2. **Active Learning**: Identify and request feedback on uncertain cases
3. **Domain Adaptation**: Specialized fine-tuning for specific domains
4. **Model Distillation**: Create smaller, faster models from fine-tuned versions
5. **Continuous Learning**: Automated retraining pipeline based on feedback accumulation

### Research Opportunities

1. **Feedback Quality Weighting**: Weight training examples by feedback quality
2. **Query Expansion**: Use fine-tuned models for query enhancement
3. **Hybrid Approaches**: Combine fine-tuned embeddings with traditional retrieval
4. **Cross-lingual Fine-tuning**: Support multiple languages in training data

## References

### Documentation
- [`README.md`](README.md) - Main system documentation
- [`PHASE1_README.md`](PHASE1_README.md) - Feedback collection system
- [`PHASE2_README.md`](PHASE2_README.md) - Retrieval optimization
- [`docs/CONPORT_INTEGRATION.md`](docs/CONPORT_INTEGRATION.md) - ConPort integration guide

### External Resources
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Ollama Model Guide](https://ollama.ai/library)
- [Fine-tuning Best Practices](https://huggingface.co/docs/transformers/training)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the example code in [`model_finetuner.py`](model_finetuner.py)
3. Examine [`training_data_generator.py`](training_data_generator.py) implementation
4. Consult the main [`README.md`](README.md) for system overview