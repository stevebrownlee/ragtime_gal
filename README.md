# Ragtime Gal - Enhanced Conversational RAG System with MCP Integration

Ragtime Gal is a sophisticated document-based question answering system that combines a web interface for document management with Model Context Protocol (MCP) server capabilities for seamless VSCode integration. It's designed specifically for book writing assistance, providing powerful analysis tools and conversational AI capabilities.

## 🌟 Key Features

### Core RAG Capabilities
- **Vector-based conversation memory**: Uses embeddings to find semantically relevant previous interactions
- **Dynamic context management**: Intelligently selects and formats context based on query type
- **Conversation summarization**: Compresses older conversation turns to manage token usage
- **Sophisticated query classification**: Determines if a query is a follow-up using both regex and semantic similarity

### Book Writing Assistance
- **Chapter-aware document processing**: Automatically extracts chapter information from markdown files
- **Character analysis**: Track character mentions and relationships across chapters
- **Writing statistics**: Comprehensive text analysis including word counts, readability metrics
- **Grammar and style checking**: Integrated language tools for content quality assessment
- **Content management**: Add, update, and organize book content through multiple interfaces

### MCP Integration
- **VSCode integration**: Full MCP server with 17 specialized tools for book analysis
- **Dual interface**: Web UI for document management, MCP tools for VSCode workflow
- **Vector database backend**: Single source of truth eliminating file I/O bottlenecks
- **Real-time synchronization**: Changes made in either interface are immediately available in both

## 🏗️ Architecture

The system follows a modular architecture with integrated Flask web server and MCP server components:

### Core Components

- **TemplateManager**: Manages prompt templates with dynamic context sections
- **ContextManager**: Handles context selection and formatting for different query types
- **QueryClassifier**: Classifies queries to determine appropriate context handling
- **ConversationEmbedder**: Embeds conversation interactions for vector-based retrieval
- **ConversationSummarizer**: Generates summaries of conversation history
- **EnhancedConversation**: Extends the base Conversation class with vector-based retrieval
- **SharedDatabaseManager**: Thread-safe database access for both Flask and MCP servers
- **MCPServerManager**: Handles MCP server lifecycle and integration

### MCP Tools Available

1. **Content Search & Navigation**
   - `search_book_content`: Vector-based content search with chapter filtering
   - `get_chapter_info`: Detailed chapter information and metadata
   - `list_all_chapters`: Chapter listing with sorting and statistics
   - `get_book_structure`: Complete book structure and organization
   - `list_all_books`: Available books in the database

2. **Character & Content Analysis**
   - `analyze_character_mentions`: Character frequency and context analysis
   - `get_writing_statistics`: Comprehensive text statistics and metrics
   - `analyze_readability`: Flesch-Kincaid scores and reading level assessment
   - `check_grammar_and_style`: Grammar checking and style suggestions
   - `analyze_writing_patterns`: Writing pattern analysis and insights

3. **Content Management**
   - `add_chapter_content`: Create new chapters with proper metadata
   - `update_chapter_content`: Modify existing chapter content safely
   - `delete_chapter`: Remove chapters with safety checks
   - `delete_book`: Remove entire books with confirmation
   - `reorder_chapters`: Reorganize chapter structure

4. **System Management**
   - `get_server_status`: Check server health and status
   - `test_database_connection`: Verify database connectivity

## 🚀 Installation

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai/) running locally
- [pipenv](https://pipenv.pypa.io/) for dependency management

### Required Ollama Models

Install the required models in Ollama:

```bash
ollama pull mistral    # For embeddings and LLM
# Or use your preferred model - configure in .env
```

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ragtime-gal
   ```

2. **Install dependencies:**
   ```bash
   pipenv install
   ```

3. **Download spaCy language model:**
   ```bash
   pipenv run python -m spacy download en_core_web_sm
   ```

4. **Set up environment configuration:**
   ```bash
   cp .env.template .env
   # Edit .env with your preferred settings
   ```

5. **Initialize the database:**
   ```bash
   pipenv run python app.py
   # This will create the vector database on first run
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.template`:

```bash
# Model settings
EMBEDDING_MODEL=mistral          # Ollama embedding model
LLM_MODEL=mistral               # Ollama LLM model
LLM_TEMPERATURE=1.0             # Response creativity (0.0-2.0)

# Server settings
PORT=8084                       # Web server port
DEBUG=false                     # Debug mode
SECRET_KEY=your-secret-key-here # Flask session key

# Path settings
TEMP_FOLDER=./_temp             # Temporary file storage
CHROMA_PERSIST_DIR=./chroma_db  # Vector database location
PROMPT_TEMPLATES_PATH=./prompt_templates.json
TEMPLATE_PATH=./template.html

# MCP Server settings
BOOK_DIRECTORY=.                # Book content directory
MCP_SERVER_NAME=Ragtime Gal MCP Server

# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434

# Retrieval settings
RETRIEVAL_K=4                   # Number of documents to retrieve
PROMPT_TEMPLATE=standard        # Default prompt template
```

### MCP Configuration

For VSCode integration, add the MCP server configuration to your MCP settings:

```json
{
  "mcpServers": {
    "ragtime-gal": {
      "command": "/path/to/your/venv/bin/python",
      "args": ["/path/to/ragtime-gal/mcp_standalone.py"],
      "env": {
        "BOOK_DIRECTORY": "/path/to/chapter/files"
      }
    }
  }
}
```

## 🎯 Usage

### Web Interface

1. **Start the application:**
   ```bash
   pipenv run python app.py
   # Or use the pipenv script:
   pipenv run start
   ```

2. **Access the web interface:**
   Open `http://localhost:8084` in your browser

3. **Upload documents:**
   - Click "Choose Files" to select PDF or Markdown files
   - The system automatically processes and embeds documents
   - Chapter information is extracted from markdown headers

4. **Query your documents:**
   - Enter questions in the query field
   - Select response style (Standard, Creative, or Sixth Wood)
   - Adjust temperature for response creativity
   - The system maintains conversation context automatically

### VSCode Integration (MCP)

1. **Install MCP-compatible VSCode extension** (like RooCode)

2. **Configure the MCP server** in your VSCode settings

3. **Use MCP tools directly in VSCode:**
   - Search book content: `search_book_content`
   - Analyze characters: `analyze_character_mentions`
   - Get writing statistics: `get_writing_statistics`
   - Add new chapters: `add_chapter_content`
   - And many more...

### Command Line Usage

You can also run the MCP server standalone for testing:

```bash
pipenv run python mcp_standalone.py
```

## 📚 Document Processing

### Supported Formats

- **PDF files**: Automatically extracted and chunked
- **Markdown files**: Chapter-aware processing with metadata extraction

### Chapter Detection

For markdown files, the system automatically detects chapters using:
- H1 headers (`# Chapter Title`)
- H2 headers (`## Chapter Title`)
- Numbered chapters (`# Chapter 1: Title`)

### Metadata Extraction

Each document chunk includes:
- Chapter title and number
- Book title (from filename or content)
- Word count and character count
- Upload timestamp
- File type and source information

## 🔧 Customization

### Prompt Templates

Customize response styles by editing `prompt_templates.json`:

```json
{
  "base_templates": {
    "standard": "Your standard template...",
    "creative": "Your creative template...",
    "sixthwood": "Your custom template..."
  },
  "system_instructions": {
    "standard": "System instructions...",
    "creative": "Creative instructions...",
    "sixthwood": "Custom instructions..."
  },
  "context_formats": {
    "initial": "Initial query format...",
    "follow_up": "Follow-up format...",
    "with_previous_content": "Context with history..."
  }
}
```

### Adding New Response Styles

1. Add new templates to `prompt_templates.json`
2. Update the web interface dropdown in `template.html`
3. Restart the application

## 🧪 Testing

### Run the test suite:

```bash
# Run all tests
pipenv run pytest

# Run specific phase tests
pipenv run python test_phase1.py  # Basic functionality
pipenv run python test_phase2.py  # Enhanced metadata
pipenv run python test_phase3.py  # Core MCP tools
pipenv run python test_phase4.py  # Advanced analysis
pipenv run python test_phase5.py  # Content management
pipenv run python test_phase6.py  # Full integration
```

### Test MCP Tools

```bash
# Test MCP server directly
pipenv run python -c "
from mcp_standalone import app
# Test individual tools
"
```

## 🔍 Troubleshooting

### Common Issues

1. **Ollama connection errors:**
   - Ensure Ollama is running: `ollama serve`
   - Check `OLLAMA_BASE_URL` in `.env`
   - Verify required models are installed

2. **Database issues:**
   - Delete `./chroma_db` directory to reset
   - Check file permissions
   - Ensure sufficient disk space

3. **MCP server not connecting:**
   - Verify Python path in MCP configuration
   - Check environment variables
   - Review MCP server logs

4. **Memory issues with large books:**
   - Reduce `RETRIEVAL_K` value
   - Increase system memory
   - Process books in smaller chunks

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set in .env
DEBUG=true

# Or run with debug flag
pipenv run python app.py --debug
```

### Logs

Check application logs for detailed error information:
- Web server logs: Console output
- MCP server logs: VSCode MCP extension logs
- Database logs: ChromaDB logs in console

## 🚀 Performance Optimization

### Vector Database

- **Indexing**: Automatic indexing on metadata fields
- **Caching**: Query result caching for frequent searches
- **Connection pooling**: Thread-safe database access

### Memory Management

- **Chunking**: Configurable chunk size (default: 7500 chars)
- **Overlap**: Configurable overlap (default: 100 chars)
- **Batch processing**: Efficient bulk operations

### Query Optimization

- **Semantic search**: Vector-based similarity search
- **Metadata filtering**: Efficient chapter and book filtering
- **Result limiting**: Configurable result counts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pipenv install --dev

# Run code formatting
pipenv run black .

# Run linting
pipenv run flake8

# Run tests
pipenv run pytest
```

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for RAG capabilities
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
- Powered by [Ollama](https://ollama.ai/) for local LLM inference
- MCP integration via [Model Context Protocol](https://modelcontextprotocol.io/)
- NLP analysis with [spaCy](https://spacy.io/) and [TextStat](https://github.com/textstat/textstat)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include logs and configuration details

---

**Happy writing! 📖✨**