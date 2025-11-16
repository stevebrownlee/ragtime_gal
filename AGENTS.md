# AGENTS.md - AI Coding Agent Guide

## Project Overview

**Ragtime Gal** is a RAG (Retrieval-Augmented Generation) system built with Flask that combines web UI with Model Context Protocol (MCP) server capabilities. It features document processing, intelligent querying, feedback-driven query optimization, and automated model fine-tuning.

## Technology Stack

- **Language**: Python 3.8+
- **Framework**: Flask (web server)
- **Vector Database**: Chroma
- **Embeddings**: Ollama (local, mistral model)
- **Training**: sentence-transformers, PyTorch
- **Package Manager**: pipenv
- **MCP Integration**: ConPort client for project memory

## Common Commands

### Setup & Installation
```bash
# Install dependencies
pipenv install

# Activate virtual environment
pipenv shell
```

### Running the Application
```bash
# Start the Flask server (default port 8084)
pipenv run python app.py

# Run with specific port
PORT=8085 pipenv run python app.py
```

### Testing
```bash
# Run tests (if test suite exists)
pipenv run pytest tests/

# Check for specific test files in tests/ directory
```

### Code Quality
```bash
# No specific linter/formatter configured
# Check for standard Python tools if needed:
# pipenv run flake8
# pipenv run black .
# pipenv run mypy .
```

## Project Structure

### Core Application Files
- **app.py**: Main Flask application with all API endpoints
- **embed.py**: Basic document processing and embedding
- **embed_enhanced.py**: Enhanced embedding with optimizations
- **query.py**: Query execution logic
- **conversation.py**: Conversation history management
- **enhanced_conversation.py**: Enhanced conversation features

### Advanced Features
- **feedback_analyzer.py**: Analyzes user feedback patterns
- **query_enhancer.py**: Query optimization and document re-ranking
- **query_classifier.py**: Classifies query types
- **training_data_generator.py**: Generates training pairs from feedback
- **model_finetuner.py**: Fine-tunes embedding models
- **conport_client.py**: ConPort MCP client integration
- **monitoring_dashboard.py**: Real-time performance monitoring

### Compatibility Layer
Files ending in `_compat.py` provide backward compatibility with older interfaces.

### Configuration
- **Environment**: `.env` file (template: `.env.template`)
- **Prompts**: `prompt_templates.json`
- **Docker**: `docker-compose.yml`

### Storage Directories
- **chroma_db/**: Chroma vector database persistence
- **_temp/**: Temporary file uploads
- **logs/**: Application logs
- **context_portal/**: ConPort data
- **.conport_local/**: Local ConPort configuration

## Code Conventions

### File Naming
- Snake_case for Python files (e.g., `query_enhancer.py`)
- Compatibility files suffixed with `_compat.py`
- Main application: `app.py`

### API Endpoints
- RESTful patterns with descriptive paths
- POST for state-changing operations
- GET for read-only operations
- Endpoint groups: document management, query/conversation, feedback, training, A/B testing, monitoring

### Key Patterns
1. **Session-based**: Use Flask sessions for conversation tracking
2. **Collection-based**: Documents organized in Chroma collections
3. **Feedback-driven**: User feedback drives query enhancement and model training
4. **Async-ready**: Training jobs should be asynchronous (check for job tracking)

### Dependencies
Before adding new libraries:
1. Check `Pipfile` for existing dependencies
2. Verify compatibility with Python 3.8+
3. Use `pipenv install <package>` to add new dependencies

## Configuration

### Environment Variables
Key variables from `.env`:
```bash
PORT=8084                           # Server port
DEBUG=False                         # Debug mode
OLLAMA_BASE_URL=http://localhost:11434  # Ollama endpoint
EMBEDDING_MODEL=mistral             # Ollama embedding model
TEMP_FOLDER=./_temp                 # Upload directory
CHROMA_PERSIST_DIR=./chroma_db      # Vector DB path
TRAINING_DATA_PATH=./training_data  # Training data output
BASE_MODEL_NAME=all-MiniLM-L6-v2   # Sentence-transformer base
BATCH_SIZE=16                       # Training batch size
NUM_EPOCHS=4                        # Training epochs
LEARNING_RATE=2e-5                  # Training learning rate
```

## API Endpoint Categories

### Document Management
- `GET /` - Web UI
- `POST /embed` - Upload & embed files
- `POST /purge` - Delete all documents
- `GET /collections` - List collections

### Query & Conversation
- `POST /query` - Query with context
- `POST /clear-history` - Clear conversation
- `GET /conversation-status` - Get conversation state

### Feedback System
- `POST /feedback` - Submit rating (1-5 stars)
- `GET /feedback/analytics` - Detailed analytics
- `GET /feedback/summary` - Quick summary

### Model Training
- `POST /training/generate-data` - Generate training pairs
- `POST /training/fine-tune` - Start fine-tuning
- `GET /training/status/<job_id>` - Check training progress

### A/B Testing
- `POST /testing/ab-test/start` - Start A/B test
- `GET /testing/ab-test/<test_id>/results` - Get results

### System Health
- `GET /health` - Health check
- `GET /monitoring` - Monitoring dashboard

## Important Workflows

### Adding Documents
1. Files uploaded to `_temp/`
2. Processed by `embed.py` or `embed_enhanced.py`
3. Embedded using Ollama
4. Stored in Chroma collections

### Query Flow
1. Query received at `/query` endpoint
2. Enhanced by `query_enhancer.py` (if enabled)
3. Executed by `query.py` with conversation context
4. Results re-ranked based on feedback patterns
5. Response returned with sources

### Training Pipeline
1. Collect feedback via `/feedback` endpoint
2. Generate training data: `/training/generate-data`
3. Fine-tune model: `/training/fine-tune`
4. A/B test: `/testing/ab-test/start`
5. Monitor results and deploy winner

## External Dependencies

### Required Services
- **Ollama**: Must be running locally for embeddings (default: localhost:11434)
- **ConPort**: MCP server for project memory (optional but recommended)

### File Upload Support
- PDF files
- Markdown files (.md)

## Testing Strategy

1. Check `tests/` directory for existing test patterns
2. API endpoints should be tested with various inputs
3. Test both compatibility and enhanced versions of modules
4. Verify ConPort integration separately

## Debugging

### Logs
- Check `logs/` directory for application logs
- Error log: `ragtime_gal_errors.log`
- Monitoring database: `monitoring.db`

### Common Issues
- Ollama not running: Check `OLLAMA_BASE_URL`
- ConPort connection: Check health endpoint
- File upload failures: Verify `TEMP_FOLDER` permissions

## Security Notes

- Session-based authentication for conversations
- File uploads sanitized and stored in temp directory
- No secrets should be committed to repository
- Use `.env` for configuration, not hardcoded values

## Documentation References

Detailed docs in `docs/`:
- `API_DOCUMENTATION.md` - Complete API reference
- `SYSTEM_ARCHITECTURE.md` - Technical architecture
- `CONPORT_INTEGRATION.md` - MCP integration guide
- `USER_GUIDE.md` - End-user documentation
