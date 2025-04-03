# Migration Guide: Modernizing the Conversational RAG System

This guide provides step-by-step instructions for migrating from the original template-based approach to the new modular, vector-based conversational RAG system.

## Migration Overview

The migration is structured in phases to allow for incremental changes while maintaining system functionality throughout the process:

1. **Phase 1**: Refactor Template Management
2. **Phase 2**: Implement Vector-Based Conversation Memory
3. **Phase 3**: Implement Query Classification and Context Manager
4. **Phase 4**: Implement Memory Management and Summarization
5. **Phase 5**: Integration and Testing

## Phase 1: Refactor Template Management

### Step 1: Update Prompt Templates

The first step is to update the `prompt_templates.json` file to use the new unified template structure:

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
  },
  "legacy_templates": {
    // Keep original templates for backward compatibility
  }
}
```

### Step 2: Create TemplateManager Class

Create a new file `template_manager.py` that implements the `TemplateManager` class:

```python
class TemplateManager:
    def __init__(self, templates_path=None):
        # Load templates from file

    def get_legacy_template(self, template_name):
        # Get a template using the legacy format

    def get_prompt(self, style, query_type, context_params):
        # Get a prompt using the new template format

    def get_system_instruction(self, style):
        # Get the system instruction for a given style
```

### Step 3: Update Query Function

Update the `query.py` file to use the new `TemplateManager` class:

```python
from template_manager import TemplateManager

# Initialize managers (lazy loading)
_template_manager = None

def get_template_manager():
    """Get or create the template manager singleton"""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager

def query(question, template_name=None, temperature=None, conversation=None):
    # Use template manager to get prompt
    template_manager = get_template_manager()
    # ...
```

## Phase 2: Implement Vector-Based Conversation Memory

### Step 1: Create ConversationEmbedder Class

Create a new file `conversation_embedder.py` that implements the `ConversationEmbedder` class:

```python
class ConversationEmbedder:
    def __init__(self, model=None, base_url=None):
        # Initialize embeddings

    def embed_query(self, query):
        # Embed a query string

    def embed_interaction(self, query, response):
        # Embed a conversation interaction

    def cosine_similarity(self, vec1, vec2):
        # Calculate cosine similarity between two vectors

    def find_most_similar(self, query_embedding, embeddings, top_k=2):
        # Find the most similar embeddings to a query embedding
```

### Step 2: Create EnhancedConversation Class

Create a new file `enhanced_conversation.py` that implements the `EnhancedConversation` class:

```python
class EnhancedConversation(Conversation):
    def __init__(self):
        # Initialize with embedder

    def add_interaction(self, query, response, document_ids=None, metadata=None):
        # Add interaction with embedding

    def get_relevant_interactions(self, query, top_k=2):
        # Get relevant interactions using vector similarity

    def get_most_relevant_content(self, query, max_interactions=2):
        # Get relevant content using vector similarity

    def is_follow_up_question(self, query):
        # Enhance follow-up detection with semantic similarity
```

### Step 3: Update Session Management

Add functions to manage the enhanced conversation in the session:

```python
def get_enhanced_conversation_from_session(session):
    # Get enhanced conversation from session

def update_enhanced_conversation_in_session(session, conversation):
    # Update enhanced conversation in session
```

## Phase 3: Implement Query Classification and Context Manager

### Step 1: Create QueryClassifier Class

Create a new file `query_classifier.py` that implements the `QueryClassifier` class:

```python
class QueryClassifier:
    def __init__(self, embedder=None):
        # Initialize with patterns and embedder

    def classify(self, query, conversation):
        # Classify query based on patterns and semantic similarity
        # Return classification with query_type, confidence, etc.
```

### Step 2: Update ContextManager Class

Update the `context_manager.py` file to use the `QueryClassifier`:

```python
class ContextManager:
    def __init__(self, template_manager=None):
        # Initialize with template manager and query classifier

    def get_context_params(self, query, conversation, retrieved_docs, classification=None):
        # Get context parameters with enhanced retrieval

    def get_prompt(self, query, conversation, retrieved_docs, style="standard"):
        # Get prompt using query classifier and template manager

    def get_conversation_memory(self, query, conversation):
        # Get structured memory from conversation history
```

## Phase 4: Implement Memory Management and Summarization

### Step 1: Create ConversationSummarizer Class

Create a new file `conversation_summarizer.py` that implements the `ConversationSummarizer` class:

```python
class ConversationSummarizer:
    def __init__(self, model=None, base_url=None):
        # Initialize with LLM for summarization

    def summarize(self, conversation, max_tokens=500):
        # Generate a summary of the conversation history

    def _format_short_conversation(self, history):
        # Format a short conversation without summarization

    def _format_full_history(self, history):
        # Format the full conversation history

    def _format_recent_history(self, history, max_recent=3):
        # Format recent history with a brief summary of older interactions
```

### Step 2: Enhance EnhancedConversation Class

Update the `EnhancedConversation` class to use the `ConversationSummarizer`:

```python
class EnhancedConversation(EnhancedConversation):
    def __init__(self):
        # Initialize with summarizer
        self.summarizer = ConversationSummarizer()
        self.summary = ""
        self.summary_updated_at = 0

    def get_summary(self):
        # Get or generate conversation summary
```

## Phase 5: Integration and Testing

### Step 1: Update App.py

Update `app.py` to use the enhanced conversation classes:

```python
from enhanced_conversation import (
    EnhancedConversation,
    get_enhanced_conversation_from_session as get_conversation_from_session,
    update_enhanced_conversation_in_session as update_conversation_in_session
)
```

### Step 2: Test the System

Create test cases to verify the functionality of the new system:

1. Test basic query functionality
2. Test follow-up question detection
3. Test conversation memory retrieval
4. Test summarization

### Step 3: Monitor and Optimize

Monitor the system's performance and optimize as needed:

1. Adjust similarity thresholds for follow-up detection
2. Tune summarization parameters
3. Optimize embedding generation and storage

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Ensure all required packages are installed
2. **Embedding model errors**: Check that the Ollama server is running and the embedding model is available
3. **Template format errors**: Verify that the prompt_templates.json file has the correct structure

### Debugging Tips

1. Enable debug logging to see detailed information about the system's operation
2. Use the `get_conversation_memory` method to inspect the conversation memory
3. Check the classification results to understand how queries are being classified

## Conclusion

This migration guide provides a structured approach to modernizing the conversational RAG system. By following these steps, you can implement a more sophisticated and maintainable system that provides better context handling and conversation management.

The new system reduces template maintenance burden, improves context relevance, and provides more flexible conversation handling while maintaining backward compatibility with the original system.