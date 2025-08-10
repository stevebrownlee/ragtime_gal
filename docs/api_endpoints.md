# Flask API Documentation
Generated on: 2025-08-10T09:29:08.897304

## Overview
This document describes all available REST API endpoints for the Ragtime Gal web interface.

### GET /

**Description:** Main web interface for document upload and querying

**Response:** HTML page with upload form and query interface

### POST /upload

**Description:** Upload documents (PDF or Markdown) to the vector database

**Parameters:**
- `file` (file): Document file to upload
- `book_title` (string): Optional book title override

**Response:** JSON with upload status and document metadata

**Example:**
```bash
curl -X POST -F 'file=@chapter1.md' http://localhost:5000/upload
```

### POST /query

**Description:** Query the vector database for similar content

**Parameters:**
- `query` (string): Search query text
- `max_results` (integer): Maximum number of results (default: 5)

**Response:** JSON with search results and similarity scores

**Example:**
```bash
curl -X POST -H 'Content-Type: application/json' -d '{"query":"character development"}' http://localhost:5000/query
```

### POST /purge

**Description:** Clear all documents from the vector database

**Response:** JSON with purge status

**Example:**
```bash
curl -X POST http://localhost:5000/purge
```

### GET /health

**Description:** Health check endpoint for monitoring

**Response:** JSON with system health status

**Example:**
```bash
curl http://localhost:5000/health
```
