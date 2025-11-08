# Getting Started with Phase 2 & Phase 3 Features

## Quick Start Guide

This guide will walk you through testing the new Phase 2 (Retrieval Optimization) and Phase 3 (Model Fine-tuning) features.

## Prerequisites

### 1. Install Required Dependencies

First, ensure you have the necessary Python packages:

```bash
# Install sentence-transformers for Phase 3
pip install sentence-transformers torch

# Or if using Pipenv
pipenv install sentence-transformers torch
```

### 2. Configure Environment Variables

Update your `.env` file with Phase 2 and Phase 3 settings:

```bash
# Phase 2: Retrieval Optimization
USE_FEEDBACK_OPTIMIZATION=true
QUERY_ENHANCEMENT_ENABLED=true
FEEDBACK_ANALYTICS_ENABLED=true
MIN_FEEDBACK_SAMPLES=10

# Phase 3: Model Fine-tuning
FINETUNING_ENABLED=true
BASE_MODEL_NAME=all-MiniLM-L6-v2
FINETUNED_MODELS_PATH=./fine_tuned_models
TRAINING_DATA_PATH=./training_data
BATCH_SIZE=16
NUM_EPOCHS=4
LEARNING_RATE=2e-5
```

### 3. Start the Application

```bash
# Start the server
python app.py

# Server should be running on http://localhost:8084
```

## Testing Phase 2: Retrieval Optimization

Phase 2 features work automatically when you have sufficient feedback data. Here's how to test them:

### Step 1: Generate Some Feedback Data

First, you need to submit queries and provide feedback:

```bash
# 1. Submit a query
curl -X POST http://localhost:8084/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I implement authentication?",
    "collection": "langchain"
  }'

# 2. Submit feedback (repeat this with different queries and ratings)
curl -X POST http://localhost:8084/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "rating": 5,
    "query": "How do I implement authentication?",
    "response": "To implement authentication...",
    "relevance": "very_relevant",
    "completeness": "complete",
    "comments": "Very helpful response with clear examples"
  }'
```

**Tip**: Submit at least 10-20 different queries with feedback to see optimization in action.

### Step 2: View Feedback Analytics

Once you have feedback data, view the analytics:

```bash
# Get comprehensive feedback analytics
curl http://localhost:8084/feedback/analytics?days_back=30

# Get quick summary
curl http://localhost:8084/feedback/summary?days_back=7
```

**Expected Output**:
```json
{
  "summary": {
    "total_feedback": 15,
    "average_rating": 4.2,
    "rating_distribution": {
      "5": 8,
      "4": 5,
      "3": 2
    }
  },
  "successful_patterns": {
    "common_terms": ["authentication", "implement", "security"],
    "query_structures": ["How do I...", "What is..."]
  },
  "recommendations": {
    "query_formulation": "Use specific technical terms",
    "query_structure": "Start with 'How do I' for implementation questions"
  }
}
```

### Step 3: Test Query Enhancement

With feedback data, your queries are now automatically enhanced:

```bash
# Submit a query (it will be enhanced behind the scenes)
curl -X POST http://localhost:8084/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "auth problem",
    "collection": "langchain"
  }'
```

Check the server logs to see query enhancement in action:
```
INFO:query_enhancer:Enhanced query from "auth problem" to "authentication security implementation problem fix"
INFO:query_enhancer:Applied enhancements: ['expanded_with_terms', 'reformatted_structure']
```

### Step 4: Monitor Optimization Impact

Query responses now include optimization metadata:

```json
{
  "message": "...",
  "conversation_active": true,
  "optimization_metadata": {
    "query_enhanced": true,
    "enhancements_applied": ["expanded_with_terms"],
    "documents_reranked": true,
    "confidence": 0.85
  }
}
```

## Testing Phase 3: Model Fine-tuning

Phase 3 allows you to fine-tune embedding models based on user feedback.

### Prerequisites for Phase 3

You need:
- At least 100 feedback samples (50 positive, 50 negative)
- Sufficient disk space (~500MB for models)
- Time for training (5-30 minutes depending on data size)

### Step 1: Generate Training Data

```bash
# Generate training data from your feedback
curl -X POST http://localhost:8084/training/generate-data \
  -H "Content-Type: application/json" \
  -d '{
    "min_positive_samples": 50,
    "min_negative_samples": 50,
    "include_hard_negatives": true,
    "days_back": 90,
    "export_format": "csv"
  }'
```

**Expected Output**:
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

**If you get "Insufficient feedback data"**: You need to collect more feedback first. Go back to Step 1 and submit more queries with feedback.

### Step 2: Start Model Fine-tuning

```bash
# Start fine-tuning (this runs in the background)
curl -X POST http://localhost:8084/training/fine-tune \
  -H "Content-Type: application/json" \
  -d '{
    "training_data_path": "./training_data/training_20240108_143022.csv",
    "base_model": "all-MiniLM-L6-v2",
    "model_name_suffix": "feedback_v1",
    "config": {
      "batch_size": 16,
      "num_epochs": 4,
      "learning_rate": 2e-5
    }
  }'
```

**Expected Output**:
```json
{
  "success": true,
  "job_id": "ft_20240108_143022",
  "status": "running",
  "message": "Fine-tuning job started"
}
```

### Step 3: Monitor Training Progress

```bash
# Check training status (repeat this periodically)
curl http://localhost:8084/training/status/ft_20240108_143022
```

**During Training**:
```json
{
  "job_id": "ft_20240108_143022",
  "status": "running",
  "progress": 45,
  "start_time": "2024-01-08T14:30:22Z"
}
```

**When Complete**:
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

### Step 4: Start A/B Test

Compare your fine-tuned model with the baseline:

```bash
# Start A/B test
curl -X POST http://localhost:8084/testing/ab-test/start \
  -H "Content-Type: application/json" \
  -d '{
    "baseline_model": "all-MiniLM-L6-v2",
    "test_model": "./fine_tuned_models/feedback_v1_20240108_143022",
    "traffic_split": 0.5,
    "duration_hours": 24,
    "metrics": ["response_quality", "relevance_score", "user_satisfaction"]
  }'
```

**Expected Output**:
```json
{
  "success": true,
  "test_id": "ab_20240108_143022",
  "status": "active",
  "start_time": "2024-01-08T14:30:22Z",
  "estimated_end_time": "2024-01-09T14:30:22Z"
}
```

### Step 5: Check A/B Test Results

```bash
# Get test results
curl http://localhost:8084/testing/ab-test/ab_20240108_143022/results
```

**Expected Output**:
```json
{
  "test_id": "ab_20240108_143022",
  "status": "active",
  "baseline_metrics": {
    "query_count": 45,
    "avg_rating": 3.8,
    "avg_relevance_score": 0.72
  },
  "test_metrics": {
    "query_count": 48,
    "avg_rating": 4.2,
    "avg_relevance_score": 0.81
  },
  "statistical_significance": true,
  "p_value": 0.023,
  "recommendation": "deploy_test_model"
}
```

## Quick Testing Script

Here's a complete script to quickly test the features:

```bash
#!/bin/bash

echo "=== Testing Phase 2 & Phase 3 Features ==="

# 1. Check health
echo "\n1. Checking server health..."
curl -s http://localhost:8084/health | jq

# 2. Get feedback analytics
echo "\n2. Getting feedback analytics..."
curl -s "http://localhost:8084/feedback/analytics?days_back=30" | jq

# 3. Generate training data (if enough feedback)
echo "\n3. Generating training data..."
curl -s -X POST http://localhost:8084/training/generate-data \
  -H "Content-Type: application/json" \
  -d '{
    "min_positive_samples": 10,
    "min_negative_samples": 10,
    "include_hard_negatives": true
  }' | jq

# 4. Start fine-tuning (uncomment if training data generated successfully)
# echo "\n4. Starting fine-tuning..."
# curl -s -X POST http://localhost:8084/training/fine-tune \
#   -H "Content-Type: application/json" \
#   -d '{
#     "training_data_path": "./training_data/training_YYYYMMDD_HHMMSS.csv",
#     "base_model": "all-MiniLM-L6-v2"
#   }' | jq

echo "\n=== Testing Complete ==="
```

Save this as `test_phase2_phase3.sh` and run:
```bash
chmod +x test_phase2_phase3.sh
./test_phase2_phase3.sh
```

## Python Testing Examples

### Test Phase 2 Programmatically

```python
import requests
import json

# Configuration
BASE_URL = "http://localhost:8084"

# 1. Get feedback analytics
response = requests.get(f"{BASE_URL}/feedback/analytics", params={"days_back": 30})
analytics = response.json()
print(f"Total feedback: {analytics['summary']['total_feedback']}")
print(f"Average rating: {analytics['summary']['average_rating']}")

# 2. Submit a query with optimization enabled
query_response = requests.post(
    f"{BASE_URL}/query",
    json={
        "query": "How do I fix authentication issues?",
        "collection": "langchain"
    }
)
result = query_response.json()
print(f"Query enhanced: {result.get('optimization_metadata', {}).get('query_enhanced', False)}")
```

### Test Phase 3 Programmatically

```python
import requests
import time

BASE_URL = "http://localhost:8084"

# 1. Generate training data
print("Generating training data...")
response = requests.post(
    f"{BASE_URL}/training/generate-data",
    json={
        "min_positive_samples": 50,
        "min_negative_samples": 50,
        "include_hard_negatives": True
    }
)
data = response.json()
if data.get('success'):
    training_path = data['training_data_path']
    print(f"Training data generated: {training_path}")
    print(f"Total pairs: {data['statistics']['total_pairs']}")

    # 2. Start fine-tuning
    print("\nStarting fine-tuning...")
    ft_response = requests.post(
        f"{BASE_URL}/training/fine-tune",
        json={
            "training_data_path": training_path,
            "base_model": "all-MiniLM-L6-v2",
            "model_name_suffix": "test_v1"
        }
    )
    ft_data = ft_response.json()
    job_id = ft_data.get('job_id')
    print(f"Job ID: {job_id}")

    # 3. Monitor progress
    print("\nMonitoring progress...")
    while True:
        status_response = requests.get(f"{BASE_URL}/training/status/{job_id}")
        status = status_response.json()
        print(f"Status: {status['status']}, Progress: {status.get('progress', 0)}%")

        if status['status'] in ['completed', 'failed']:
            break

        time.sleep(30)  # Check every 30 seconds

    if status['status'] == 'completed':
        print(f"\n✅ Training completed!")
        print(f"Model path: {status['model_path']}")
        print(f"Duration: {status['training_duration']}s")
else:
    print(f"❌ Error: {data.get('error')}")
```

## Monitoring Dashboard

Visit the monitoring dashboard to see Phase 2 and Phase 3 activity:

```
http://localhost:8084/monitoring
```

The dashboard shows:
- Feedback analytics and trends
- Query enhancement statistics
- Training job status
- A/B test results
- System performance metrics

## Troubleshooting

### "Insufficient feedback data"

**Problem**: Not enough feedback to generate training data or enable optimization.

**Solution**:
1. Submit more queries and feedback (need at least 50-100 samples)
2. Lower the `min_positive_samples` and `min_negative_samples` temporarily
3. Check ConPort is properly storing feedback:
   ```bash
   curl http://localhost:8084/health
   # Should show: "conport_available": true
   ```

### "sentence-transformers not available"

**Problem**: Missing dependency for Phase 3.

**Solution**:
```bash
pip install sentence-transformers torch
```

### "Training job failed"

**Problem**: Fine-tuning job encountered an error.

**Solution**:
1. Check the job status for error details:
   ```bash
   curl http://localhost:8084/training/status/YOUR_JOB_ID
   ```
2. Check server logs for detailed error messages
3. Verify training data file exists and is properly formatted
4. Ensure sufficient disk space and memory

### "Query enhancement not working"

**Problem**: Queries not being enhanced despite feedback data.

**Solution**:
1. Verify `USE_FEEDBACK_OPTIMIZATION=true` in `.env`
2. Check you have enough feedback data (minimum 10 samples)
3. Look for enhancement metadata in query responses
4. Check server logs for query_enhancer activity

## Next Steps

1. **Collect More Feedback**: The more feedback you collect, the better the optimization and fine-tuning
2. **Monitor Metrics**: Use the monitoring dashboard to track improvement trends
3. **Iterate on Models**: Fine-tune multiple versions and A/B test them
4. **Production Deployment**: Once satisfied with results, deploy the fine-tuned model

## Additional Resources

- **Phase 2 Documentation**: [`PHASE2_README.md`](PHASE2_README.md)
- **Phase 3 Documentation**: [`PHASE3_README.md`](PHASE3_README.md)
- **Main README**: [`README.md`](README.md)
- **ConPort Integration**: [`docs/CONPORT_INTEGRATION.md`](docs/CONPORT_INTEGRATION.md)