# Configuration and Deployment Guide
Generated on: 2025-08-10T09:29:08.897379

## Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Database Configuration
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=mistral

# MCP Server Configuration
MCP_SERVER_NAME=ragtime-gal-mcp
MCP_SERVER_VERSION=1.0.0
BOOK_DIRECTORY=./books

# Performance Configuration
CHUNK_SIZE=7500
CHUNK_OVERLAP=100
MAX_UPLOAD_SIZE=16777216  # 16MB

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/ragtime_gal.log
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running
- Git

### Step-by-step Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ragtime-gal
```

2. **Install dependencies:**
```bash
pip install pipenv
pipenv install
pipenv shell
```

3. **Set up environment variables:**
```bash
cp .env.template .env
# Edit .env with your configuration
```

4. **Start Ollama (if not already running):**
```bash
ollama serve
ollama pull mistral  # Download embedding model
```

5. **Run the application:**
```bash
python app.py
```

## VSCode Integration

### MCP Server Setup

1. **Install RooCode extension** in VSCode

2. **Configure MCP server** in VSCode settings:
```json
{
  "roocode.mcpServers": {
    "ragtime-gal": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/ragtime-gal"
    }
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Error
**Problem:** Cannot connect to Ollama server

**Solution:**
- Ensure Ollama is running: `ollama serve`
- Check if mistral model is available: `ollama list`
- If not available: `ollama pull mistral`

#### 2. Database Permission Error
**Problem:** Cannot create or access Chroma database

**Solution:**
- Check directory permissions
- Ensure CHROMA_DB_PATH directory is writable
- Try running with elevated permissions if necessary

#### 3. MCP Server Not Responding
**Problem:** MCP tools not available in VSCode

**Solution:**
- Check MCP server logs for errors
- Verify Python path and dependencies
- Restart VSCode and RooCode extension
- Check MCP server configuration in VSCode settings

#### 4. Large File Upload Issues
**Problem:** Cannot upload large documents

**Solution:**
- Increase MAX_UPLOAD_SIZE in .env
- Check available disk space
- Consider splitting large documents into chapters

## Performance Optimization

### Database Optimization
- Adjust CHUNK_SIZE and CHUNK_OVERLAP for your content
- Use SSD storage for better I/O performance
- Monitor memory usage with large document collections

### Query Optimization
- Use specific queries rather than broad searches
- Limit result count for better performance
- Use metadata filters to narrow search scope

### System Resources
- Allocate sufficient RAM (minimum 4GB recommended)
- Use multi-core CPU for better embedding performance
- Monitor disk space usage
