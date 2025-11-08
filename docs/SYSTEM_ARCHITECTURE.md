# System Architecture: Feedback-Driven RAG System

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Phase-by-Phase Architecture](#phase-by-phase-architecture)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [Component Interactions](#component-interactions)
6. [Database Schemas](#database-schemas)
7. [ConPort Integration](#conport-integration)
8. [Security Architecture](#security-architecture)
9. [Performance Considerations](#performance-considerations)
10. [Scalability Design](#scalability-design)

## Overview

The Feedback-Driven RAG (Retrieval-Augmented Generation) System is a comprehensive, multi-phase architecture designed to continuously improve document-based question answering through user feedback and machine learning. The system evolves through five distinct phases, each building upon the previous to create a sophisticated, self-improving AI system.

### System Goals
- **Intelligent Document Retrieval**: Semantic search across document collections
- **Conversational Memory**: Context-aware multi-turn conversations
- **Continuous Learning**: Feedback-driven system improvements
- **Model Fine-tuning**: Automated embedding model optimization
- **Production Readiness**: Comprehensive testing and monitoring

### Key Characteristics
- **Modular Design**: Each phase can operate independently
- **Backward Compatibility**: New phases don't break existing functionality
- **Scalable Architecture**: Designed for production deployment
- **Privacy-First**: Local processing with optional cloud integration
- **Extensible Framework**: Easy to add new features and capabilities

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Feedback-Driven RAG System                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   Phase 5       │  │   Phase 4       │  │   Phase 3       │                │
│  │ Documentation   │  │ Testing &       │  │ Model Fine-     │                │
│  │ & Monitoring    │  │ Validation      │  │ tuning System   │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐                                      │
│  │   Phase 2       │  │   Phase 1       │                                      │
│  │ Retrieval       │  │ Feedback        │                                      │
│  │ Optimization    │  │ Collection      │                                      │
│  └─────────────────┘  └─────────────────┘                                      │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                              Core System                                       │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   Web Interface │  │   Flask API     │  │   Conversation  │                │
│  │   (HTML/JS)     │  │   Server        │  │   Management    │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   Vector DB     │  │   LLM Engine    │  │   ConPort       │                │
│  │   (ChromaDB)    │  │   (Ollama)      │  │   Memory        │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend:**
- HTML5/CSS3/JavaScript
- Bootstrap for responsive design
- AJAX for dynamic interactions

**Backend:**
- Python 3.8+
- Flask web framework
- Gunicorn WSGI server (production)

**AI/ML Components:**
- Ollama for local LLM inference
- Sentence Transformers for embeddings
- ChromaDB for vector storage
- Scikit-learn for analytics

**Data Storage:**
- ChromaDB (vector embeddings)
- ConPort (structured memory)
- File system (documents, models)

**Infrastructure:**
- Docker containerization
- Nginx reverse proxy (production)
- Prometheus monitoring
- Grafana dashboards

## Phase-by-Phase Architecture

### Phase 1: Feedback Collection System

```
Phase 1 Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    Feedback Collection                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Web UI        │    │   API Endpoint  │                    │
│  │   Feedback      │◄──►│   /feedback     │                    │
│  │   Forms         │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                   │                             │
│                                   ▼                             │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   ConPort       │    │   Feedback      │                    │
│  │   Storage       │◄───│   Processor     │                    │
│  │                 │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Components:**
- **Feedback UI**: Star ratings, detailed feedback forms
- **Feedback API**: REST endpoint for feedback submission
- **Feedback Processor**: Validates and structures feedback data
- **ConPort Integration**: Stores feedback with conversation context

### Phase 2: Retrieval Optimization

```
Phase 2 Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                  Retrieval Optimization                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Feedback      │    │   Query         │                    │
│  │   Analyzer      │◄───│   Enhancer      │                    │
│  │                 │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Pattern       │    │   Enhanced      │                    │
│  │   Recognition   │    │   Query         │                    │
│  │                 │    │   Processing    │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                   │                             │
│                                   ▼                             │
│                         ┌─────────────────┐                    │
│                         │   Improved      │                    │
│                         │   Retrieval     │                    │
│                         │                 │                    │
│                         └─────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Components:**
- **Feedback Analyzer**: Identifies patterns in user feedback
- **Query Enhancer**: Improves query formulation based on patterns
- **Pattern Recognition**: ML algorithms for feedback analysis
- **Enhanced Query Processing**: Optimized retrieval pipeline

### Phase 3: Model Fine-tuning System

```
Phase 3 Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    Model Fine-tuning                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Training Data │    │   Model         │                    │
│  │   Generator     │◄───│   Fine-tuner    │                    │
│  │                 │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Positive/     │    │   Custom        │                    │
│  │   Negative      │    │   Embedding     │                    │
│  │   Pairs         │    │   Models        │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                   │                             │
│                                   ▼                             │
│                         ┌─────────────────┐                    │
│                         │   A/B Testing   │                    │
│                         │   Framework     │                    │
│                         │                 │                    │
│                         └─────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Components:**
- **Training Data Generator**: Creates training pairs from feedback
- **Model Fine-tuner**: Fine-tunes sentence transformer models
- **Custom Embedding Models**: Specialized models for domain
- **A/B Testing Framework**: Compares model performance

### Phase 4: Testing and Validation

```
Phase 4 Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                  Testing & Validation                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Integration   │    │   Performance   │                    │
│  │   Testing       │    │   Testing       │                    │
│  │                 │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Validation    │    │   Regression    │                    │
│  │   Testing       │    │   Testing       │                    │
│  │                 │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
│                         ┌─────────────────┐                    │
│                         │   Test          │                    │
│                         │   Orchestrator  │                    │
│                         │                 │                    │
│                         └─────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Components:**
- **Integration Testing**: End-to-end workflow validation
- **Performance Testing**: Load testing and benchmarking
- **Validation Testing**: Effectiveness measurement
- **Regression Testing**: Backward compatibility assurance

### Phase 5: Documentation and Monitoring

```
Phase 5 Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                Documentation & Monitoring                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   User          │    │   Technical     │                    │
│  │   Documentation │    │   Documentation │                    │
│  │                 │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Monitoring    │    │   Deployment    │                    │
│  │   Dashboard     │    │   Guide         │                    │
│  │                 │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
│                         ┌─────────────────┐                    │
│                         │   Production    │                    │
│                         │   Support       │                    │
│                         │                 │                    │
│                         └─────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Components:**
- **User Documentation**: Guides, tutorials, FAQ
- **Technical Documentation**: Architecture, APIs, development
- **Monitoring Dashboard**: Real-time system metrics
- **Deployment Guide**: Production setup and maintenance

## Data Flow Diagrams

### Primary Query Flow

```
User Query Flow:
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  User   │───►│   Web   │───►│  Flask  │───►│ Query   │
│ Input   │    │   UI    │    │   API   │    │Processor│
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                   │
                                                   ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│Enhanced │◄───│Conversation│◄─│ Vector  │◄───│ Query   │
│Response │    │ Manager │    │Database │    │Enhancer │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
     │                                             │
     ▼                                             ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  User   │◄───│   Web   │◄───│  Flask  │◄───│   LLM   │
│Response │    │   UI    │    │   API   │    │ Engine  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### Feedback Processing Flow

```
Feedback Flow:
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  User   │───►│Feedback │───►│Feedback │───►│ConPort  │
│Feedback │    │   UI    │    │   API   │    │Storage  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                   │
                                                   ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│Training │◄───│ Model   │◄───│ Pattern │◄───│Feedback │
│  Data   │    │Fine-tune│    │Analysis │    │Analyzer │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
     │                                             │
     ▼                                             ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│Enhanced │───►│   A/B   │───►│ Query   │───►│Improved │
│ Models  │    │Testing  │    │Enhancer │    │ System  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### Document Processing Flow

```
Document Flow:
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│Document │───►│  File   │───►│Document │───►│  Text   │
│ Upload  │    │Handler  │    │ Parser  │    │Extractor│
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                   │
                                                   ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Vector  │◄───│Embedding│◄───│  Chunk  │◄───│  Text   │
│Database │    │ Engine  │    │Processor│    │Splitter │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

## Component Interactions

### Core Component Relationships

```
Component Interaction Map:

┌─────────────────┐         ┌─────────────────┐
│   Flask API     │◄───────►│  Web Interface  │
│   Server        │         │                 │
└─────────────────┘         └─────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  Conversation   │◄───────►│  Session        │
│  Manager        │         │  Management     │
└─────────────────┘         └─────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│   Query         │◄───────►│  Template       │
│   Processor     │         │  Manager        │
└─────────────────┘         └─────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│   Vector        │◄───────►│   LLM           │
│   Database      │         │   Engine        │
└─────────────────┘         └─────────────────┘
```

### Phase Integration Points

```
Phase Integration:

Phase 1 (Feedback) ──┐
                     │
Phase 2 (Retrieval) ─┼──► ConPort Memory ◄──┐
                     │                       │
Phase 3 (Fine-tune) ─┘                      │
                                             │
Phase 4 (Testing) ──────────────────────────┘
                     │
Phase 5 (Monitoring) ┘
```

## Database Schemas

### ChromaDB Schema

**Collections Structure:**
```python
{
  "collection_name": "string",
  "metadata": {
    "created_date": "timestamp",
    "document_count": "integer",
    "last_updated": "timestamp"
  },
  "documents": [
    {
      "id": "string",
      "content": "string",
      "metadata": {
        "source": "string",
        "page": "integer",
        "chunk_index": "integer",
        "document_id": "string"
      },
      "embedding": "vector[384]"
    }
  ]
}
```

**Document Metadata:**
```python
{
  "document_id": "string",
  "filename": "string",
  "upload_date": "timestamp",
  "file_size": "integer",
  "chunk_count": "integer",
  "collection_name": "string",
  "processing_status": "string"
}
```

### ConPort Schema

**Feedback Data Structure:**
```python
{
  "category": "UserFeedback",
  "key": "feedback_id",
  "value": {
    "feedback_id": "string",
    "timestamp": "float",
    "session_id": "string",
    "rating": "integer",
    "query": "string",
    "response": "string",
    "detailed_feedback": {
      "relevance": "string",
      "completeness": "string",
      "length": "string",
      "comments": "string"
    },
    "document_ids": ["string"],
    "metadata": "object",
    "conversation_context": {
      "history_length": "integer",
      "is_follow_up": "boolean"
    }
  }
}
```

**System Patterns:**
```python
{
  "pattern_id": "integer",
  "name": "string",
  "description": "string",
  "tags": ["string"],
  "created_date": "timestamp",
  "usage_count": "integer"
}
```

**Progress Tracking:**
```python
{
  "progress_id": "integer",
  "description": "string",
  "status": "string",
  "created_date": "timestamp",
  "updated_date": "timestamp",
  "linked_item_type": "string",
  "linked_item_id": "string"
}
```

## ConPort Integration

### Memory Architecture

```
ConPort Memory Structure:

┌─────────────────────────────────────────────────────────────────┐
│                        ConPort Memory                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Product       │    │   Active        │                    │
│  │   Context       │    │   Context       │                    │
│  │                 │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Decisions     │    │   Progress      │                    │
│  │   Log           │    │   Tracking      │                    │
│  │                 │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   System        │    │   Custom        │                    │
│  │   Patterns      │    │   Data          │                    │
│  │                 │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow with ConPort

```
ConPort Integration Flow:

User Interaction ──► Flask API ──► ConPort Storage
                                        │
                                        ▼
Analytics ◄──── Feedback Analyzer ◄──── ConPort Retrieval
    │                                   │
    ▼                                   ▼
Query Enhancement ──► Model Training ──► System Improvement
```

## Security Architecture

### Security Layers

```
Security Architecture:

┌─────────────────────────────────────────────────────────────────┐
│                      Security Layers                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Transport     │    │   Application   │                    │
│  │   Security      │    │   Security      │                    │
│  │   (HTTPS/TLS)   │    │   (Auth/RBAC)   │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Data          │    │   Infrastructure│                    │
│  │   Security      │    │   Security      │                    │
│  │   (Encryption)  │    │   (Containers)  │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Security Features

**Data Protection:**
- Local data storage by default
- Optional encryption at rest
- Secure session management
- Input validation and sanitization

**Access Control:**
- Session-based authentication
- Role-based access control (planned)
- API rate limiting
- Request validation

**Privacy Features:**
- No external data sharing by default
- Anonymized feedback analytics
- Configurable data retention
- GDPR compliance ready

## Performance Considerations

### Performance Architecture

```
Performance Optimization:

┌─────────────────────────────────────────────────────────────────┐
│                   Performance Layers                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Caching       │    │   Load          │                    │
│  │   Layer         │    │   Balancing     │                    │
│  │                 │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Database      │    │   Model         │                    │
│  │   Optimization  │    │   Optimization  │                    │
│  │                 │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Optimization Strategies

**Query Performance:**
- Vector index optimization
- Conversation history caching
- Template pre-compilation
- Batch processing for embeddings

**Model Performance:**
- Model quantization options
- GPU acceleration support
- Async processing pipelines
- Memory-efficient fine-tuning

**System Performance:**
- Connection pooling
- Response compression
- Static asset caching
- Database query optimization

## Scalability Design

### Horizontal Scaling

```
Scalability Architecture:

┌─────────────────────────────────────────────────────────────────┐
│                    Horizontal Scaling                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Load          │    │   API           │                    │
│  │   Balancer      │───►│   Instances     │                    │
│  │                 │    │   (Multiple)    │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                   │                             │
│                                   ▼                             │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Shared        │    │   Distributed   │                    │
│  │   Storage       │◄───│   Processing    │                    │
│  │                 │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Scaling Strategies

**Application Scaling:**
- Stateless API design
- Container orchestration
- Auto-scaling policies
- Health check endpoints

**Data Scaling:**
- Database sharding
- Read replicas
- Distributed caching
- CDN integration

**Processing Scaling:**
- Async task queues
- Distributed model inference
- Batch processing optimization
- Resource pooling

---

*This system architecture document provides a comprehensive overview of the feedback-driven RAG system. For implementation details, refer to the Developer Guide and API Documentation.*