# ConPort Integration for Feedback Storage

This document explains the ConPort integration implemented for storing and analyzing user feedback in the RAG system.

## Overview

The ConPort integration provides persistent storage for user feedback data, enabling:
- Long-term feedback storage and retrieval
- Pattern analysis for system improvement
- Training data generation for model fine-tuning
- Performance monitoring and analytics

## Architecture

### Components

1. **ConPortClient** (`conport_client.py`)
   - Wrapper for ConPort MCP client functionality
   - Handles data storage and retrieval operations
   - Provides fallback to local storage when ConPort is unavailable

2. **FeedbackAnalyzer** (`feedback_analyzer.py`)
   - Analyzes stored feedback data for patterns
   - Generates insights and recommendations
   - Supports query enhancement suggestions

3. **Flask Integration** (`app.py`)
   - Feedback submission endpoint (`/feedback`)
   - Analytics endpoints (`/feedback/analytics`, `/feedback/summary`)
   - Health check with ConPort status

## Data Flow

```
User Feedback → Flask Route → ConPort Client → Storage (ConPort/Local)
                                    ↓
Analytics Endpoints ← FeedbackAnalyzer ← Stored Feedback Data
```

## Storage Structure

### Feedback Data Format

Each feedback entry is stored with the following structure:

```json
{
  "feedback_id": "uuid-string",
  "timestamp": 1640995200.0,
  "session_id": "session-uuid",
  "rating": 4,
  "query": "What is our vacation policy?",
  "response": "Our vacation policy allows...",
  "detailed_feedback": {
    "relevance": "very_relevant",
    "completeness": "complete",
    "length": "just_right",
    "comments": "Very helpful answer"
  },
  "document_ids": ["doc_123", "doc_456"],
  "metadata": {
    "template_used": "standard",
    "temperature": 0.7,
    "processing_time": 2.3
  },
  "conversation_context": {
    "history_length": 3,
    "is_follow_up": false
  }
}
```

### ConPort Storage

- **Category**: `UserFeedback`
- **Key**: `feedback_id` (UUID)
- **Value**: Complete feedback object
- **Metadata**: Rating, timestamp, session_id for quick filtering

### Local Backup Storage

When ConPort is unavailable, data is stored locally in:
```
./.conport_local/UserFeedback/{feedback_id}.json
```

## API Endpoints

### Submit Feedback
```http
POST /feedback
Content-Type: application/json

{
  "rating": 4,
  "query": "What is our vacation policy?",
  "response": "Our vacation policy allows...",
  "relevance": "very_relevant",
  "completeness": "complete",
  "length": "just_right",
  "comments": "Very helpful answer"
}
```

**Response:**
```json
{
  "message": "Feedback submitted successfully",
  "feedback_id": "fb_abc123",
  "storage_method": "conport"
}
```

### Get Analytics
```http
GET /feedback/analytics?days_back=30&min_rating=3
```

**Response:**
```json
{
  "summary": {
    "total_feedback": 150,
    "average_rating": 4.2,
    "rating_distribution": {"1": 5, "2": 10, "3": 25, "4": 60, "5": 50},
    "high_rated_percentage": 73.3
  },
  "successful_patterns": {
    "average_length": 8.5,
    "common_words": {"policy": 15, "vacation": 12},
    "question_percentage": 75.0
  },
  "recommendations": {
    "query_enhancement": ["Encourage longer, more detailed queries"],
    "similarity_threshold": 0.75,
    "query_expansion_terms": ["policy", "benefits", "time-off"]
  }
}
```

### Get Summary
```http
GET /feedback/summary?days_back=7
```

**Response:**
```json
{
  "period": "Last 7 days",
  "total_feedback": 25,
  "average_rating": 4.1,
  "rating_distribution": {"3": 3, "4": 12, "5": 10},
  "trend": "positive"
}
```

## Configuration

### Environment Variables

- `CONPORT_WORKSPACE_ID`: Override default workspace ID
- `CONPORT_SERVER_URL`: ConPort MCP server URL (when available)
- `FEEDBACK_STORAGE_PATH`: Local backup storage path

### Initialization

The ConPort client is automatically initialized when the Flask app starts:

```python
from conport_client import initialize_conport_client

# Initialize ConPort client
conport_client = initialize_conport_client(workspace_id=os.getcwd())
```

## Usage Examples

### Basic Feedback Submission

```python
import requests

feedback_data = {
    "rating": 5,
    "query": "How do I reset my password?",
    "response": "To reset your password, go to...",
    "relevance": "completely_relevant",
    "completeness": "complete",
    "length": "just_right"
}

response = requests.post(
    "http://localhost:8084/feedback",
    json=feedback_data
)

print(response.json())
```

### Retrieving Analytics

```python
import requests

# Get analytics for the last 30 days
response = requests.get(
    "http://localhost:8084/feedback/analytics?days_back=30"
)

analytics = response.json()
print(f"Average rating: {analytics['summary']['average_rating']}")
```

## Testing

Use the provided test script to verify the integration:

```bash
python test_feedback_integration.py
```

This script tests:
- Health check with ConPort status
- Feedback submission and storage
- Local storage fallback
- Analytics and summary endpoints

## Troubleshooting

### Common Issues

1. **ConPort Unavailable**
   - System automatically falls back to local storage
   - Check logs for ConPort connection errors
   - Verify MCP server configuration

2. **Storage Permissions**
   - Ensure write permissions for `.conport_local` directory
   - Check disk space availability

3. **Analytics Empty**
   - Verify feedback data exists in storage
   - Check date range parameters
   - Ensure proper data format

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check ConPort client status:

```python
from conport_client import get_conport_client

client = get_conport_client()
print(f"ConPort available: {client.is_client_available()}")
print(f"Workspace ID: {client.get_workspace_id()}")
```

## Future Enhancements

1. **Real MCP Integration**
   - Replace simulation with actual MCP tool calls
   - Implement proper ConPort server connection

2. **Advanced Analytics**
   - Trend analysis over time
   - Correlation analysis between feedback and document performance
   - Predictive insights

3. **Data Export/Import**
   - Backup and restore functionality
   - Data migration tools
   - Integration with external analytics platforms

4. **Performance Optimization**
   - Caching for frequently accessed analytics
   - Batch processing for large datasets
   - Asynchronous storage operations

## Security Considerations

- Feedback data may contain sensitive information
- Implement proper access controls for analytics endpoints
- Consider data retention policies
- Ensure secure storage of user session data

## Monitoring

The system provides several monitoring capabilities:

- Health check endpoint includes ConPort status
- Logging for all storage operations
- Error tracking for failed operations
- Performance metrics for analytics queries

Monitor these metrics to ensure system health and performance.