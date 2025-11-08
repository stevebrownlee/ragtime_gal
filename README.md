# Ragtime Gal - RAG Server with MCP Integration

A sophisticated Retrieval-Augmented Generation (RAG) system that combines Flask-based web UI with Model Context Protocol (MCP) server capabilities, featuring advanced feedback collection, query optimization, and automated model fine-tuning.

## High-Level Features

### Core RAG Capabilities
- **Document Processing & Embedding**: Upload and process PDF and Markdown files into a Chroma vector database using Ollama embeddings
- **Intelligent Query System**: Context-aware question answering with conversation history management
- **Multi-Collection Support**: Organize documents into separate collections for better organization
- **Session-Based Conversations**: Maintain conversation context across queries within a session

### Advanced Query Optimization
- **Feedback-Driven Enhancement**: Automatically improve queries based on historical user feedback patterns
- **Adaptive Similarity Thresholds**: Dynamic adjustment of retrieval thresholds based on query characteristics
- **Document Re-ranking**: Reorder retrieved documents based on historical performance data
- **Multiple Enhancement Modes**: Auto, expand, rephrase, or no enhancement options

### Model Fine-Tuning System
- **Training Data Generation**: Automatically generate training pairs from user feedback
- **Hard Negative Mining**: Identify challenging negative examples for better model discrimination
- **Automated Fine-Tuning Pipeline**: Fine-tune embedding models using sentence-transformers
- **A/B Testing Framework**: Compare model performance with statistical significance testing

### MCP Integration (Book Writing Assistant)
- **Vector-Based Content Search**: Semantic search across book content with metadata filtering
- **Chapter Management**: Tools for analyzing, organizing, and navigating book chapters
- **Character Analysis**: Track and analyze character mentions using semantic search
- **Writing Analytics**: Comprehensive statistics, readability metrics, and pattern analysis
- **Content Management**: CRUD operations for chapters and books with safety checks

### User Feedback & Analytics
- **5-Star Rating System**: Collect detailed user satisfaction ratings
- **Multi-Dimensional Feedback**: Track relevance, completeness, and response length
- **Pattern Analysis**: Identify successful query patterns and problematic areas
- **Real-Time Analytics**: Access feedback insights and recommendations through API endpoints

### Monitoring & Performance
- **Real-Time Monitoring Dashboard**: Track system performance metrics at `/monitoring`
- **Query Caching**: TTL-based caching for improved response times
- **Performance Metrics**: Monitor response times, throughput, and resource usage
- **Error Tracking**: Comprehensive error logging and statistics

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| **Document Management** |
| `GET` | `/` | Main web interface for document upload and querying |
| `POST` | `/embed` | Upload and embed PDF/Markdown files into vector database |
| `POST` | `/purge` | Delete all documents from all collections |
| `GET` | `/collections` | List all available collections with document counts |
| **Query & Conversation** |
| `POST` | `/query` | Query the vector database with conversation context |
| `POST` | `/clear-history` | Clear the conversation history for current session |
| `GET` | `/conversation-status` | Get current conversation status and history length |
| **Feedback System** |
| `POST` | `/feedback` | Submit user feedback on query responses (1-5 stars) |
| `GET` | `/feedback/analytics` | Get detailed feedback analytics and patterns |
| `GET` | `/feedback/summary` | Get quick feedback summary for specified time period |
| **Model Training** |
| `POST` | `/training/generate-data` | Generate training data from user feedback |
| `POST` | `/training/fine-tune` | Start a model fine-tuning job |
| `GET` | `/training/status/<job_id>` | Get the status and progress of a training job |
| **A/B Testing** |
| `POST` | `/testing/ab-test/start` | Start an A/B test comparing two models |
| `GET` | `/testing/ab-test/<test_id>/results` | Get results and recommendations for an A/B test |
| **System Health** |
| `GET` | `/health` | Health check endpoint with ConPort status |
| `GET` | `/monitoring` | Access the monitoring dashboard |

## Training Mechanism Overview

The system implements a sophisticated feedback-driven training pipeline:

### 1. Feedback Collection
- Users rate query responses on a 5-star scale
- Additional dimensions tracked: relevance, completeness, response length
- All feedback stored in ConPort with complete query/response/document context
- Session tracking links feedback to conversation flow

### 2. Training Data Generation

The `TrainingDataGenerator` class processes feedback into training pairs:

**Positive Pairs (Rating ≥ 4)**
- Query paired with relevant documents from the response
- Labeled with score of 1.0
- Represents successful retrievals

**Negative Pairs (Rating ≤ 2)**
- Query paired with documents that didn't help
- Labeled with score of 0.0
- Helps model learn what *not* to retrieve

**Hard Negative Mining**
- Identifies documents that are semantically similar but contextually wrong
- Uses similarity threshold (default 0.7) to find challenging examples
- Improves model's discrimination capabilities

**Export Formats**
- CSV format for sentence-transformers (query, document, label)
- JSON format for custom training pipelines
- Configurable output paths and naming

### 3. Model Fine-Tuning

The `ModelFinetuner` class handles the actual training:

**Process Flow**
1. Load base sentence-transformer model (default: `all-MiniLM-L6-v2`)
2. Parse training data into `InputExample` objects
3. Split into training/validation sets (default 80/20)
4. Configure training parameters:
   - Batch size (default: 16)
   - Number of epochs (default: 4)
   - Learning rate (default: 2e-5)
   - Loss function: CosineSimilarityLoss
5. Train with automatic mixed precision (AMP) for efficiency
6. Save fine-tuned model with timestamp and suffix

**Training Parameters**
- Configurable through environment variables or API parameters
- Supports various sentence-transformer base models
- Validation tracking for monitoring overfitting
- GPU support when available

### 4. Model Deployment & Testing

**A/B Testing Framework**
- Compare baseline vs. fine-tuned model performance
- Configurable traffic split (default 50/50)
- Track multiple metrics: response quality, relevance, user satisfaction
- Statistical significance testing with configurable thresholds
- Automatic recommendations based on performance delta

**Deployment Strategy**
1. Generate training data (minimum 50 positive + 50 negative pairs)
2. Fine-tune model (typically 2-4 hours depending on dataset size)
3. Start A/B test with 24-48 hour duration
4. Monitor results endpoint for statistical significance
5. Deploy winning model based on recommendations

### 5. Continuous Improvement Loop

The system creates a continuous improvement cycle:

```
User Query → RAG Response → User Feedback → Training Data →
Model Fine-Tuning → A/B Testing → Model Deployment → (repeat)
```

**Key Metrics Tracked**
- Positive/negative pair counts
- Hard negative mining success rate
- Training loss and validation accuracy
- A/B test statistical significance
- User satisfaction trends over time

## Quick Start

### Prerequisites
- Python 3.8+
- Ollama running locally (for embeddings)
- ConPort MCP server configured

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ragtime-gal
```

2. Install dependencies:
```bash
pipenv install
```

3. Configure environment:
```bash
cp .env.template .env
# Edit .env with your settings
```

4. Start the server:
```bash
pipenv run python app.py
```

5. Access the web interface:
```
http://localhost:8084
```

## Configuration

Key environment variables (see `.env.template`):

```bash
# Server Configuration
PORT=8084
DEBUG=False
SECRET_KEY=your-secret-key

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=mistral

# Storage Paths
TEMP_FOLDER=./_temp
CHROMA_PERSIST_DIR=./chroma_db

# Training Configuration
TRAINING_DATA_PATH=./training_data
FINETUNED_MODELS_PATH=./fine_tuned_models
BASE_MODEL_NAME=all-MiniLM-L6-v2
BATCH_SIZE=16
NUM_EPOCHS=4
LEARNING_RATE=2e-5
```

## Documentation

Detailed documentation available in the `docs/` directory:

- [API Documentation](docs/API_DOCUMENTATION.md) - Complete API reference
- [System Architecture](docs/SYSTEM_ARCHITECTURE.md) - Technical architecture details
- [ConPort Integration](docs/CONPORT_INTEGRATION.md) - MCP integration guide
- [User Guide](docs/USER_GUIDE.md) - End-user documentation

## Architecture

### Technology Stack
- **Backend**: Flask (Python)
- **Vector Database**: Chroma
- **Embeddings**: Ollama (local, mistral model)
- **MCP Protocol**: Model Context Protocol for VSCode integration
- **Training**: sentence-transformers, PyTorch
- **Monitoring**: Custom monitoring dashboard with real-time metrics

### Key Components
- **app.py**: Main Flask application with all API endpoints
- **embed.py/embed_enhanced.py**: Document processing and embedding
- **query.py**: Query execution with conversation context
- **feedback_analyzer.py**: Feedback pattern analysis
- **query_enhancer.py**: Query optimization and re-ranking
- **training_data_generator.py**: Training pair generation
- **model_finetuner.py**: Model fine-tuning pipeline
- **monitoring_dashboard.py**: Performance monitoring system
- **conport_client.py**: ConPort MCP client integration

## Contributing

This project combines RAG capabilities with MCP server functionality. When contributing:

1. Test both web UI and MCP tool functionality
2. Ensure ConPort integration remains functional
3. Validate feedback collection and analytics
4. Test training pipeline with sufficient data
5. Update documentation for new features

## Acknowledgments

Built with:
- LangChain for RAG orchestration
- ChromaDB for vector storage
- Ollama for local embeddings
- sentence-transformers for fine-tuning
- Flask for web framework
- ConPort for project memory management