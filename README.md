# Enhanced Conversational RAG System

This project implements a document-based question answering system with enhanced conversational capabilities using local language models.

## Key Features

- **Vector-based conversation memory**: Uses embeddings to find semantically relevant previous interactions
- **Dynamic context management**: Intelligently selects and formats context based on query type
- **Conversation summarization**: Compresses older conversation turns to manage token usage
- **Sophisticated query classification**: Determines if a query is a follow-up using both regex and semantic similarity

## Architecture

The system follows a modular architecture with the following components:

### Core Components

- **TemplateManager**: Manages prompt templates with dynamic context sections
- **ContextManager**: Handles context selection and formatting for different query types
- **QueryClassifier**: Classifies queries to determine appropriate context handling
- **ConversationEmbedder**: Embeds conversation interactions for vector-based retrieval
- **ConversationSummarizer**: Generates summaries of conversation history
- **EnhancedConversation**: Extends the base Conversation class with vector-based retrieval

## Usage

### Running the Application

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the application:
   ```
   python app.py
   ```

3. Open a web browser and navigate to `http://localhost:8080`

### Embedding Documents

1. Upload PDF or Markdown files through the web interface
2. The system will automatically embed the documents and store them in the vector database

### Querying Documents

1. Enter your question in the query input field
2. Select a response style (Standard, Creative, or Sixth Wood)
3. Adjust the temperature if desired
4. Submit your query

The system will:
1. Retrieve relevant documents from the vector database
2. Classify your query to determine if it's a follow-up question
3. Select appropriate context from conversation history if needed
4. Generate a response using the selected template and context

## Configuration

The system can be configured through environment variables:

- `LLM_MODEL`: The language model to use (default: 'sixthwood')
- `EMBEDDING_MODEL`: The embedding model to use (default: 'mistral')
- `CHROMA_PERSIST_DIR`: Directory for the vector database (default: './chroma_db')
- `OLLAMA_BASE_URL`: Base URL for Ollama API (default: 'http://localhost:11434')
- `RETRIEVAL_K`: Number of documents to retrieve (default: 4)
- `PROMPT_TEMPLATES_PATH`: Path to prompt templates file (default: './prompt_templates.json')

## Customizing Templates

The system uses a unified template structure defined in `prompt_templates.json`:

```json
{
  "base_templates": {
    "standard": "...",
    "creative": "...",
    "sixthwood": "..."
  },
  "system_instructions": {
    "standard": "...",
    "creative": "...",
    "sixthwood": "..."
  },
  "context_formats": {
    "initial": "...",
    "follow_up": "...",
    "with_previous_content": "..."
  }
}
```

You can customize these templates to change the system's response style and context handling.

## Implementation Notes

This system implements a modern approach to conversational RAG with:

1. **Unified templating with dynamic context**: Uses a single template per style with dynamic context sections
2. **Vector-based conversation memory**: Uses embedding-based retrieval for conversation history
3. **Structured memory management**: Implements multi-tier memory with summary generation
4. **Dynamic prompt construction**: Adapts prompts based on conversation state

This approach reduces template maintenance burden, improves context relevance, and provides more flexible conversation handling.