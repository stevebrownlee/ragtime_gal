# API Documentation: Feedback-Driven RAG System

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL and Versioning](#base-url-and-versioning)
4. [Core Endpoints](#core-endpoints)
5. [Feedback System Endpoints](#feedback-system-endpoints)
6. [Phase 3 Endpoints (Model Fine-tuning)](#phase-3-endpoints-model-fine-tuning)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [Examples](#examples)
10. [SDKs and Libraries](#sdks-and-libraries)

## Overview

The Feedback-Driven RAG System provides a comprehensive REST API for document embedding, querying, feedback collection, and model fine-tuning. This API enables integration with external applications and automated workflows.

### API Features
- Document upload and embedding
- Intelligent query processing with conversation memory
- Comprehensive feedback collection
- Model fine-tuning and A/B testing
- Performance monitoring and analytics
- Collection management

### Supported Content Types
- **Request**: `application/json`, `multipart/form-data`
- **Response**: `application/json`

## Authentication

Currently, the API uses session-based authentication for web interface integration. For production deployments, additional authentication methods may be configured.

### Session Management
```http
# Sessions are automatically managed through cookies
# No explicit authentication required for basic operations
```

### Future Authentication Methods
- API Keys (planned)
- OAuth 2.0 (planned)
- JWT Tokens (planned)

## Base URL and Versioning

### Base URL
```
http://localhost:8084/api/v1
```

### Current Version
- **API Version**: 1.0
- **System Version**: Phase 5 (Documentation and Monitoring)

## Core Endpoints

### 1. Health Check

Check system status and availability.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0",
  "conport_available": true,
  "workspace_id": "/path/to/workspace",
  "components": {
    "database": "healthy",
    "embeddings": "healthy",
    "llm": "healthy",
    "conport": "healthy"
  }
}
```

### 2. Document Embedding

Upload and embed documents into the vector database.

```http
POST /embed
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (required): Document file (PDF or Markdown)
- `collection_name` (optional): Target collection name (default: "langchain")

**Example Request:**
```bash
curl -X POST http://localhost:8084/embed \
  -F "file=@document.pdf" \
  -F "collection_name=company-policies"
```

**Response:**
```json
{
  "message": "File embedded successfully into collection \"company-policies\"",
  "collection_name": "company-policies",
  "document_id": "doc_12345",
  "chunks_created": 15,
  "processing_time": 45.2
}
```

**Error Response:**
```json
{
  "error": "No file part",
  "code": "MISSING_FILE",
  "details": "Request must include a file in the 'file' field"
}
```

### 3. Query Processing

Submit queries to the RAG system with conversation context.

```http
POST /query
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "What is our vacation policy?",
  "template": "standard",
  "temperature": 0.7,
  "collection_name": "company-policies",
  "max_tokens": 500
}
```

**Parameters:**
- `query` (required): The question or query text
- `template` (optional): Response style ("standard", "creative", "sixthwood")
- `temperature` (optional): Response creativity (0.0-1.0, default: 0.7)
- `collection_name` (optional): Target collection to search
- `max_tokens` (optional): Maximum response length

**Response:**
```json
{
  "message": "Our vacation policy allows...",
  "conversation_active": true,
  "is_follow_up": false,
  "history_length": 1,
  "document_ids": ["doc_123", "doc_456"],
  "confidence_score": 0.85,
  "processing_time": 2.3,
  "metadata": {
    "retrieved_chunks": 4,
    "template_used": "standard",
    "temperature": 0.7
  }
}
```

### 4. Conversation Management

#### Get Conversation Status
```http
GET /conversation-status
```

**Response:**
```json
{
  "conversation_active": true,
  "history_length": 5,
  "session_id": "sess_abc123",
  "last_interaction": "2024-01-15T10:25:00Z"
}
```

#### Clear Conversation History
```http
POST /clear-history
```

**Response:**
```json
{
  "message": "Conversation history cleared successfully",
  "conversation_active": false
}
```

### 5. Collection Management

#### List Collections
```http
GET /collections
```

**Response:**
```json
{
  "collections": [
    {
      "name": "company-policies",
      "document_count": 25,
      "created_date": "2024-01-10T09:00:00Z",
      "last_updated": "2024-01-15T08:30:00Z"
    },
    {
      "name": "technical-docs",
      "document_count": 150,
      "created_date": "2024-01-05T14:20:00Z",
      "last_updated": "2024-01-14T16:45:00Z"
    }
  ],
  "total_collections": 2
}
```

#### Get Collection Details
```http
GET /collections/{collection_name}
```

**Response:**
```json
{
  "name": "company-policies",
  "document_count": 25,
  "total_chunks": 450,
  "created_date": "2024-01-10T09:00:00Z",
  "last_updated": "2024-01-15T08:30:00Z",
  "documents": [
    {
      "id": "doc_123",
      "filename": "vacation-policy.pdf",
      "upload_date": "2024-01-10T09:15:00Z",
      "chunk_count": 8
    }
  ]
}
```

### 6. Database Management

#### Purge Database
```http
POST /purge
```

**Response:**
```json
{
  "message": "Database purged successfully. Removed 1250 documents.",
  "documents_removed": 1250,
  "collections_affected": 5
}
```

## Feedback System Endpoints

### 1. Submit Feedback

Submit user feedback on query responses.

```http
POST /feedback
Content-Type: application/json
```

**Request Body:**
```json
{
  "rating": 4,
  "query": "What is our vacation policy?",
  "response": "Our vacation policy allows...",
  "relevance": "very_relevant",
  "completeness": "complete",
  "length": "just_right",
  "comments": "Very helpful, but could include more examples",
  "is_follow_up": false
}
```

**Parameters:**
- `rating` (required): Integer rating 1-5
- `query` (required): Original query text
- `response` (required): System response text
- `relevance` (optional): "not_relevant", "somewhat_relevant", "very_relevant", "completely_relevant"
- `completeness` (optional): "incomplete", "somewhat_complete", "complete", "too_detailed"
- `length` (optional): "too_short", "just_right", "too_long"
- `comments` (optional): Free-text feedback
- `is_follow_up` (optional): Boolean indicating if this was a follow-up query

**Response:**
```json
{
  "message": "Feedback submitted successfully",
  "feedback_id": "fb_abc123",
  "timestamp": "2024-01-15T10:30:00Z",
  "storage_method": "conport"
}
```

**Response Fields:**
- `message`: Success confirmation message
- `feedback_id`: Unique identifier for the stored feedback
- `timestamp`: When the feedback was processed
- `storage_method`: Either "conport" (if ConPort MCP is available) or "local_backup" (if storing locally)

### 2. Get Feedback Analytics

Retrieve aggregated feedback analytics.

```http
GET /feedback/analytics
```

**Query Parameters:**
- `days_back` (optional): Number of days to analyze (default: 30)
- `collection_name` (optional): Filter by collection
- `min_rating` (optional): Minimum rating to include

**Response:**
```json
{
  "summary": {
    "total_feedback": 150,
    "average_rating": 4.2,
    "rating_distribution": {
      "1": 5,
      "2": 10,
      "3": 25,
      "4": 60,
      "5": 50
    }
  },
  "trends": {
    "daily_averages": [
      {"date": "2024-01-14", "average_rating": 4.1, "count": 12},
      {"date": "2024-01-15", "average_rating": 4.3, "count": 8}
    ]
  },
  "insights": {
    "top_issues": [
      "Response too brief",
      "Missing specific examples"
    ],
    "improvement_areas": [
      "completeness",
      "relevance"
    ]
  }
}
```

### 3. Get Query Enhancement Suggestions

Get suggestions for improving query quality.

```http
POST /feedback/enhance-query
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "time off",
  "context": "employee benefits"
}
```

**Response:**
```json
{
  "original_query": "time off",
  "enhanced_suggestions": [
    "What is the process for requesting vacation time?",
    "How many vacation days do employees receive?",
    "What types of paid time off are available?"
  ],
  "reasoning": "More specific queries typically receive better responses",
  "confidence": 0.8
}
```

## Phase 3 Endpoints (Model Fine-tuning)

### 1. Generate Training Data

Create training data from feedback for model fine-tuning.

```http
POST /training/generate-data
Content-Type: application/json
```

**Request Body:**
```json
{
  "days_back": 30,
  "min_positive_rating": 4,
  "max_negative_rating": 2,
  "max_pairs": 1000,
  "format": "sentence_transformers"
}
```

**Response:**
```json
{
  "message": "Training data generated successfully",
  "training_pairs": 450,
  "positive_pairs": 280,
  "negative_pairs": 170,
  "data_file": "training_data_20240115.json",
  "quality_score": 0.85
}
```

### 2. Fine-tune Model

Start model fine-tuning process.

```http
POST /training/fine-tune
Content-Type: application/json
```

**Request Body:**
```json
{
  "base_model": "all-MiniLM-L6-v2",
  "training_data_file": "training_data_20240115.json",
  "epochs": 3,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "model_name": "custom_model_v1"
}
```

**Response:**
```json
{
  "message": "Fine-tuning started successfully",
  "job_id": "ft_job_123",
  "estimated_duration": "45 minutes",
  "status": "running",
  "progress_url": "/training/status/ft_job_123"
}
```

### 3. Check Training Status

Monitor fine-tuning progress.

```http
GET /training/status/{job_id}
```

**Response:**
```json
{
  "job_id": "ft_job_123",
  "status": "running",
  "progress": 0.65,
  "current_epoch": 2,
  "total_epochs": 3,
  "elapsed_time": "28 minutes",
  "estimated_remaining": "17 minutes",
  "metrics": {
    "loss": 0.23,
    "accuracy": 0.87
  }
}
```

### 4. A/B Testing

#### Start A/B Test
```http
POST /testing/ab-test/start
Content-Type: application/json
```

**Request Body:**
```json
{
  "test_name": "model_comparison_v1",
  "model_a": "original_model",
  "model_b": "custom_model_v1",
  "traffic_split": 0.5,
  "duration_days": 7,
  "success_metrics": ["rating", "relevance"]
}
```

**Response:**
```json
{
  "message": "A/B test started successfully",
  "test_id": "ab_test_456",
  "start_date": "2024-01-15T10:30:00Z",
  "end_date": "2024-01-22T10:30:00Z",
  "status": "active"
}
```

#### Get A/B Test Results
```http
GET /testing/ab-test/{test_id}/results
```

**Response:**
```json
{
  "test_id": "ab_test_456",
  "status": "completed",
  "duration": "7 days",
  "results": {
    "model_a": {
      "queries": 150,
      "average_rating": 4.1,
      "average_relevance": 0.82
    },
    "model_b": {
      "queries": 145,
      "average_rating": 4.4,
      "average_relevance": 0.89
    },
    "statistical_significance": 0.95,
    "winner": "model_b",
    "improvement": "7.3% better rating, 8.5% better relevance"
  }
}
```

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": "Human-readable error message",
  "code": "ERROR_CODE",
  "details": "Additional technical details",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_abc123"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `MISSING_PARAMETER` | 400 | Required parameter not provided |
| `INVALID_FORMAT` | 400 | Invalid data format or structure |
| `FILE_TOO_LARGE` | 413 | Uploaded file exceeds size limit |
| `UNSUPPORTED_FORMAT` | 415 | File format not supported |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server-side processing error |
| `SERVICE_UNAVAILABLE` | 503 | System temporarily unavailable |

### Error Examples

**Missing Required Parameter:**
```json
{
  "error": "Missing required field: query",
  "code": "MISSING_PARAMETER",
  "details": "The 'query' field is required for query processing",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_abc123"
}
```

**Invalid Rating Value:**
```json
{
  "error": "Rating must be an integer between 1 and 5",
  "code": "INVALID_FORMAT",
  "details": "Received rating: 6, expected: 1-5",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_def456"
}
```

## Rate Limiting

### Current Limits

| Endpoint Category | Requests per Minute | Requests per Hour |
|-------------------|---------------------|-------------------|
| Query Processing | 60 | 1000 |
| Document Upload | 10 | 100 |
| Feedback Submission | 30 | 500 |
| Analytics | 20 | 200 |
| Training Operations | 5 | 20 |

### Rate Limit Headers

Responses include rate limiting information:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642248600
```

### Rate Limit Exceeded Response

```json
{
  "error": "Rate limit exceeded",
  "code": "RATE_LIMIT_EXCEEDED",
  "details": "Maximum 60 requests per minute allowed",
  "retry_after": 30,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Examples

### Complete Workflow Example

```python
import requests
import json

# Base URL
base_url = "http://localhost:8084"

# 1. Upload a document
with open("policy.pdf", "rb") as f:
    response = requests.post(
        f"{base_url}/embed",
        files={"file": f},
        data={"collection_name": "policies"}
    )
print(f"Upload: {response.json()}")

# 2. Query the document
query_data = {
    "query": "What is the vacation policy?",
    "template": "standard",
    "temperature": 0.7
}
response = requests.post(
    f"{base_url}/query",
    json=query_data
)
result = response.json()
print(f"Query: {result['message']}")

# 3. Submit feedback
feedback_data = {
    "rating": 4,
    "query": query_data["query"],
    "response": result["message"],
    "relevance": "very_relevant",
    "completeness": "complete",
    "comments": "Very helpful response"
}
response = requests.post(
    f"{base_url}/feedback",
    json=feedback_data
)
print(f"Feedback: {response.json()}")

# 4. Get analytics
response = requests.get(f"{base_url}/feedback/analytics?days_back=7")
analytics = response.json()
print(f"Average rating: {analytics['summary']['average_rating']}")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const baseURL = 'http://localhost:8084';

async function uploadDocument(filePath, collectionName) {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));
  form.append('collection_name', collectionName);

  try {
    const response = await axios.post(`${baseURL}/embed`, form, {
      headers: form.getHeaders()
    });
    return response.data;
  } catch (error) {
    console.error('Upload failed:', error.response.data);
    throw error;
  }
}

async function queryDocument(query, template = 'standard') {
  try {
    const response = await axios.post(`${baseURL}/query`, {
      query,
      template,
      temperature: 0.7
    });
    return response.data;
  } catch (error) {
    console.error('Query failed:', error.response.data);
    throw error;
  }
}

async function submitFeedback(rating, query, response, comments = '') {
  try {
    const feedbackResponse = await axios.post(`${baseURL}/feedback`, {
      rating,
      query,
      response,
      relevance: 'very_relevant',
      completeness: 'complete',
      comments
    });
    return feedbackResponse.data;
  } catch (error) {
    console.error('Feedback failed:', error.response.data);
    throw error;
  }
}

// Usage example
async function main() {
  try {
    // Upload document
    const uploadResult = await uploadDocument('./document.pdf', 'policies');
    console.log('Upload successful:', uploadResult);

    // Query document
    const queryResult = await queryDocument('What is the vacation policy?');
    console.log('Query result:', queryResult.message);

    // Submit feedback
    const feedbackResult = await submitFeedback(
      4,
      'What is the vacation policy?',
      queryResult.message,
      'Very helpful response'
    );
    console.log('Feedback submitted:', feedbackResult);

  } catch (error) {
    console.error('Error:', error.message);
  }
}

main();
```

### cURL Examples

```bash
# Upload document
curl -X POST http://localhost:8084/embed \
  -F "file=@document.pdf" \
  -F "collection_name=policies"

# Query document
curl -X POST http://localhost:8084/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the vacation policy?",
    "template": "standard",
    "temperature": 0.7
  }'

# Submit feedback
curl -X POST http://localhost:8084/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "rating": 4,
    "query": "What is the vacation policy?",
    "response": "Our vacation policy allows...",
    "relevance": "very_relevant",
    "completeness": "complete",
    "comments": "Very helpful response"
  }'

# Get analytics
curl -X GET "http://localhost:8084/feedback/analytics?days_back=7"

# Start A/B test
curl -X POST http://localhost:8084/testing/ab-test/start \
  -H "Content-Type: application/json" \
  -d '{
    "test_name": "model_comparison_v1",
    "model_a": "original_model",
    "model_b": "custom_model_v1",
    "traffic_split": 0.5,
    "duration_days": 7
  }'
```

## SDKs and Libraries

### Python SDK (Planned)

```python
from ragtime_client import RAGClient

client = RAGClient(base_url="http://localhost:8084")

# Upload document
result = client.embed_document("document.pdf", collection="policies")

# Query
response = client.query("What is the vacation policy?", template="standard")

# Submit feedback
client.submit_feedback(
    rating=4,
    query="What is the vacation policy?",
    response=response.message,
    comments="Very helpful"
)
```

### JavaScript SDK (Planned)

```javascript
import { RAGClient } from 'ragtime-client';

const client = new RAGClient({ baseURL: 'http://localhost:8084' });

// Upload document
const uploadResult = await client.embedDocument('./document.pdf', 'policies');

// Query
const queryResult = await client.query('What is the vacation policy?');

// Submit feedback
await client.submitFeedback({
  rating: 4,
  query: 'What is the vacation policy?',
  response: queryResult.message,
  comments: 'Very helpful'
});
```

---

*This API documentation is regularly updated. For the latest version and additional examples, check the system documentation or contact your administrator.*