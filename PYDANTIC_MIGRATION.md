# Pydantic AI Migration Strategy

## Table of Contents
- [Overview](#overview)
- [Migration Stages](#migration-stages)
- [Stage 1: Replace Orchestration Layer](#stage-1-replace-orchestration-layer)
- [Stage 2: Native Chroma Integration](#stage-2-native-chroma-integration)
- [Stage 3: Replace Document Utilities](#stage-3-replace-document-utilities)
- [Risk Mitigation](#risk-mitigation)
- [Testing Strategy](#testing-strategy)
- [Rollback Plan](#rollback-plan)
- [References](#references)

## Overview

### Goal
Migrate from LangChain to Pydantic AI to gain:
- **Developer Control**: Explicit Python orchestration vs. LCEL abstractions
- **Type Safety**: Pydantic-validated outputs throughout the pipeline
- **Maintainability**: Clearer code flow and easier refactoring
- **Performance**: Reduced framework overhead

### Timeline
- **Stage 1 (Simple)**: 2-6 hours - High ROI, low risk
- **Stage 2 (Moderate)**: 4-8 hours - Replace vector DB wrapper
- **Stage 3 (Moderate)**: 3-6 hours - Replace document utilities
- **Total**: 1-2 days for complete migration

### Current LangChain Dependencies
```python
# From Pipfile
langchain = "*"
langchain-text-splitters = "*"
langchain-community = "*"
langchain-chroma = "*"
langchain-ollama = "*"
langchain-core = "*"
```

---

## Migration Stages

### Decision Points

**Option A: Incremental Migration (Recommended)**
- Start with Stage 1 only
- Keep LangChain RAG utilities temporarily
- Gain 80% of benefits with minimal disruption
- Evaluate before proceeding to Stage 2/3

**Option B: Full Migration**
- Complete all three stages
- Remove all LangChain dependencies
- Full control over entire pipeline

---

## Stage 1: Replace Orchestration Layer

### Scope
Replace LangChain's LCEL chains, prompts, and chat models with Pydantic AI agents.

**Files to modify:**
- `ragtime/core/query_processor.py` - Main query processing logic
- `query.py` - Legacy query interface
- `Pipfile` - Add new dependencies

**Keep unchanged:**
- Vector DB wrapper (`ragtime/storage/vector_db.py`)
- Document embedding (`embed.py`, `embed_enhanced.py`)
- Document loaders and splitters
- Feedback optimization modules

### Dependencies to Add

```toml
# Add to Pipfile [packages]
pydantic-ai = "*"
openai = "*"  # For OpenAI-compatible client
```

Run:
```bash
pipenv install pydantic-ai openai
```

### Critical: Ollama Connection via Pydantic AI

**Important Note:** This migration guide has been updated to reflect the current Pydantic AI API (as of the implementation date). The original guide used deprecated classes that have since been renamed.

**Key API Changes:**
- `OpenAIModel` → `OpenAIChatModel` (renamed for clarity)
- Model initialization simplified: Use `provider='ollama'` string instead of creating client objects
- Temperature moved to `Agent`'s `model_settings` parameter
- Pydantic AI automatically connects to Ollama at `localhost:11434/v1` when `provider='ollama'`

**Endpoint format**: `http://localhost:11434/v1`

Test Ollama is running:
```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ilyr",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**Environment Variables:**
While Pydantic AI uses Ollama's default endpoint automatically, you should still add these to `.env` for configuration flexibility:
```bash
# Ollama base URL (used by existing LangChain code)
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI-compatible endpoint (for reference, though Pydantic AI uses default)
OLLAMA_OPENAI_BASE_URL=http://localhost:11434/v1

# Feature flag to toggle between LangChain and Pydantic AI
USE_PYDANTIC_AI=false
```

### Example: Replace Query Processor Chat Generation

**Before (LangChain LCEL):**
```python
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def _generate_response(self, query, docs, context_info, temperature, conversation):
    prompt_text = context_info["prompt"]
    system_instruction = context_info.get("system_instruction")

    llm = ChatOllama(
        model=self.settings.llm_model,
        base_url=self.settings.ollama_base_url,
        temperature=temperature,
        system=system_instruction or None,
        num_predict=self.settings.max_output_tokens,
        # ... more params
    )

    prompt = ChatPromptTemplate.from_template(prompt_text)
    context_params = self.context_manager.get_context_params(...)

    chain = (
        {
            "question": RunnablePassthrough(),
            "context": lambda _: context_params.get("context", ""),
            # ... more params
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(query)
    return response
```

**After (Pydantic AI):**
```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

class QueryResponse(BaseModel):
    """Validated response from the LLM."""
    answer: str = Field(..., description="The answer to the user's query")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Optional confidence score")

def _generate_response(self, query, docs, context_info, temperature, conversation):
    prompt_text = context_info["prompt"]
    system_instruction = context_info.get("system_instruction", "")

    # Get context parameters (unchanged)
    context_params = self.context_manager.get_context_params(
        query=query,
        conversation=conversation,
        retrieved_docs=docs,
        classification=context_info.get("classification")
    )

    # Build full prompt with context
    # Expand the prompt template with actual values
    full_prompt = prompt_text.format(
        question=query,
        context=context_params.get("context", ""),
        previous_content=context_params.get("previous_content", ""),
        conversation_history=context_params.get("conversation_history", ""),
        conversation_summary=context_params.get("conversation_summary", "")
    )

    # Create model - Pydantic AI handles Ollama connection automatically
    model = OpenAIChatModel(
        model_name=self.settings.llm_model,
        provider='ollama'  # Automatically connects to localhost:11434/v1
    )

    # Create agent with system prompt
    agent = Agent(
        model=model,
        system_prompt=system_instruction,
        result_type=str,  # Use QueryResponse for typed output
        model_settings={'temperature': temperature}
    )

    # Run agent
    result = agent.run_sync(full_prompt)

    # Extract response
    # If using QueryResponse: return result.data.answer
    return result.data  # Returns the string directly

# Alternative: Typed response with validation
def _generate_response_typed(self, query, docs, context_info, temperature, conversation):
    """Version with type-safe response validation."""
    # ... same setup as above ...

    # Create model
    model = OpenAIChatModel(
        model_name=self.settings.llm_model,
        provider='ollama'
    )

    agent = Agent(
        model=model,
        system_prompt=system_instruction,
        result_type=QueryResponse,  # Type-validated output
        model_settings={'temperature': temperature}
    )

    result = agent.run_sync(full_prompt)

    self.logger.info(
        "response_generated",
        response_length=len(result.data.answer),
        confidence=result.data.confidence
    )

    return result.data.answer
```

### Example: Simplified Query Function

**Create new file: `ragtime/core/pydantic_query.py`**

```python
"""
Pydantic AI-based query processor.
Simplified, type-safe alternative to LangChain LCEL.
"""

import structlog
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from ragtime.config.settings import get_settings
from ragtime.models.queries import QueryRequest, QueryResponse, RetrievedDocument

# Import existing managers (unchanged)
from template_manager import TemplateManager
from context_manager import ContextManager
from feedback_analyzer import create_feedback_analyzer
from query_enhancer import create_query_enhancer

# Import vector DB (Stage 1: keep LangChain version)
from ragtime.storage.vector_db import get_vector_db

logger = structlog.get_logger(__name__)


class LLMResponse(BaseModel):
    """Type-safe LLM response."""
    answer: str = Field(..., description="The generated answer")
    reasoning: str | None = Field(None, description="Optional reasoning steps")


class PydanticQueryProcessor:
    """
    Query processor using Pydantic AI.

    Provides explicit control over RAG pipeline with type-safe outputs.
    """

    def __init__(
        self,
        settings=None,
        template_manager: Optional[TemplateManager] = None,
        context_manager: Optional[ContextManager] = None
    ):
        self.settings = settings or get_settings()
        self.logger = logger.bind(component="pydantic_query_processor")

        self.template_manager = template_manager or TemplateManager()
        self.context_manager = context_manager or ContextManager(
            template_manager=self.template_manager
        )

        self.logger.info(
            "pydantic_query_processor_initialized",
            llm_model=self.settings.llm_model
        )

    def process_query(
        self,
        question: str,
        collection_name: Optional[str] = None,
        template_name: Optional[str] = None,
        temperature: Optional[float] = None,
        conversation: Optional[Any] = None,
        use_typed_output: bool = False
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Process a query with Pydantic AI.

        Args:
            question: The question to ask
            collection_name: Vector database collection
            template_name: Prompt template name
            temperature: LLM temperature
            conversation: Conversation object with history
            use_typed_output: Return validated LLMResponse instead of raw string

        Returns:
            Tuple of (response, document_ids, metadata)
        """
        try:
            # Use defaults
            template_name = template_name or self.settings.prompt_template
            temperature = temperature if temperature is not None else self.settings.llm_temperature
            collection_name = collection_name or self.settings.default_collection

            self.logger.info(
                "processing_query",
                question=question[:100],
                template=template_name,
                temperature=temperature
            )

            # Step 1: Retrieve documents (using existing LangChain wrapper for now)
            docs, document_ids = self._retrieve_documents(question, collection_name)

            # Step 2: Build context and prompt
            context_info = self.context_manager.get_prompt(
                query=question,
                conversation=conversation,
                retrieved_docs=docs,
                style=template_name
            )

            # Step 3: Generate response with Pydantic AI
            response = self._generate_with_agent(
                question=question,
                docs=docs,
                context_info=context_info,
                temperature=temperature,
                conversation=conversation,
                use_typed_output=use_typed_output
            )

            # Build metadata
            metadata = {
                "is_follow_up": context_info.get("is_follow_up", False),
                "template_used": template_name,
                "query_type": context_info.get("query_type"),
                "engine": "pydantic_ai"
            }

            return response, document_ids, metadata

        except Exception as e:
            self.logger.error("query_processing_failed", error=str(e), exc_info=True)
            return f"An error occurred: {str(e)}", [], {"error": str(e)}

    def _retrieve_documents(
        self,
        query: str,
        collection_name: str
    ) -> Tuple[List[Any], List[str]]:
        """Retrieve documents using existing vector DB."""
        try:
            vector_db = get_vector_db()

            # Use existing similarity search
            docs = vector_db.similarity_search(
                query=query,
                collection_name=collection_name,
                k=self.settings.retrieval_k
            )

            # Extract document IDs
            document_ids = []
            for doc in docs:
                if hasattr(doc, 'metadata') and 'id' in doc.metadata:
                    document_ids.append(doc.metadata['id'])

            self.logger.info(
                "documents_retrieved",
                num_docs=len(docs),
                collection=collection_name
            )

            return docs, document_ids

        except Exception as e:
            self.logger.error("document_retrieval_failed", error=str(e))
            return [], []

    def _generate_with_agent(
        self,
        question: str,
        docs: List[Any],
        context_info: Dict[str, Any],
        temperature: float,
        conversation: Optional[Any],
        use_typed_output: bool
    ) -> str:
        """Generate response using Pydantic AI agent."""

        # Get context parameters
        context_params = self.context_manager.get_context_params(
            query=question,
            conversation=conversation,
            retrieved_docs=docs,
            classification=context_info.get("classification")
        )

        # Expand prompt template
        prompt_text = context_info["prompt"]
        full_prompt = prompt_text.format(
            question=question,
            context=context_params.get("context", ""),
            previous_content=context_params.get("previous_content", ""),
            conversation_history=context_params.get("conversation_history", ""),
            conversation_summary=context_params.get("conversation_summary", "")
        )

        system_instruction = context_info.get("system_instruction", "")

        # Create model - Pydantic AI handles Ollama connection automatically
        model = OpenAIChatModel(
            model_name=self.settings.llm_model,
            provider='ollama'  # Automatically connects to localhost:11434/v1
        )

        # Create agent
        result_type = LLMResponse if use_typed_output else str
        agent = Agent(
            model=model,
            system_prompt=system_instruction,
            result_type=result_type,
            model_settings={'temperature': temperature}
        )

        # Run agent
        result = agent.run_sync(full_prompt)

        # Extract response
        if use_typed_output:
            response = result.data.answer
            if result.data.reasoning:
                self.logger.debug("llm_reasoning", reasoning=result.data.reasoning)
        else:
            response = result.data

        self.logger.info("response_generated", response_length=len(response))

        return response
```

### Configuration Updates

**Add to `ragtime/config/settings.py`:**

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Pydantic AI settings
    ollama_openai_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Ollama OpenAI-compatible endpoint"
    )

    use_pydantic_ai: bool = Field(
        default=False,
        description="Use Pydantic AI instead of LangChain for orchestration"
    )
```

### Update `.env`

```bash
# Add these lines
OLLAMA_OPENAI_BASE_URL=http://localhost:11434/v1
USE_PYDANTIC_AI=false  # Set to true when ready to switch
```

### Integration: Update API Endpoints

**In `ragtime/api/queries.py`:**

```python
from ragtime.core.query_processor import QueryProcessor
from ragtime.core.pydantic_query import PydanticQueryProcessor
from ragtime.config.settings import get_settings

@bp.route('/query', methods=['POST'])
def query_endpoint():
    settings = get_settings()
    data = request.get_json()

    # Choose processor based on config
    if settings.use_pydantic_ai:
        processor = PydanticQueryProcessor()
    else:
        processor = QueryProcessor()

    response, doc_ids, metadata = processor.process_query(
        question=data.get('question'),
        collection_name=data.get('collection'),
        template_name=data.get('template'),
        temperature=data.get('temperature')
    )

    # ... rest of endpoint logic
```

### Testing Stage 1

**Create `tests/test_pydantic_query.py`:**

```python
import pytest
from ragtime.core.pydantic_query import PydanticQueryProcessor

def test_basic_query():
    """Test basic query processing with Pydantic AI."""
    processor = PydanticQueryProcessor()

    response, doc_ids, metadata = processor.process_query(
        question="What is Ragtime Gal?",
        collection_name="langchain"
    )

    assert isinstance(response, str)
    assert len(response) > 0
    assert metadata["engine"] == "pydantic_ai"

def test_typed_output():
    """Test type-safe response validation."""
    processor = PydanticQueryProcessor()

    response, doc_ids, metadata = processor.process_query(
        question="What is Ragtime Gal?",
        use_typed_output=True
    )

    assert isinstance(response, str)
    # Response should be validated through LLMResponse model
```

Run tests:
```bash
pipenv run pytest tests/test_pydantic_query.py -v
```

---

## Stage 2: Native Chroma Integration

### Scope
Replace LangChain's Chroma wrapper with native ChromaDB client and custom Ollama embeddings adapter.

**Files to modify:**
- `ragtime/storage/vector_db.py` - Vector database abstraction
- `ragtime/core/pydantic_query.py` - Update to use native retrieval

### Dependencies

ChromaDB is already installed. No new dependencies needed.

### Critical: Embedding Model Validation

⚠️ **Important**: Ensure you're using an embedding-capable Ollama model, not a chat model.

**Valid embedding models:**
- `nomic-embed-text`
- `all-minilm`
- `mxbai-embed-large`

**Invalid (chat models):**
- `mistral` (this is a chat model, not for embeddings!)
- `ilyr` (custom chat model)

**Test your current embedding model:**
```bash
# This should work if EMBEDDING_MODEL is embedding-capable
curl http://localhost:11434/api/embeddings \
  -d '{
    "model": "mistral",
    "prompt": "test"
  }'
```

If you get an error, pull an embedding model:
```bash
ollama pull nomic-embed-text
```

**Update `.env`:**
```bash
# Change from chat model to embedding model
EMBEDDING_MODEL=nomic-embed-text  # Was: mistral
```

### Create Ollama Embedding Adapter

**Create `ragtime/embeddings/ollama_adapter.py`:**

```python
"""
Native Ollama embedding adapter for ChromaDB.
Replaces LangChain OllamaEmbeddings.
"""

import requests
from typing import List, Optional
import structlog
from chromadb.utils.embedding_functions import EmbeddingFunction

logger = structlog.get_logger(__name__)


class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function using Ollama API.

    Reference: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        batch_size: int = 32
    ):
        """
        Initialize Ollama embedding function.

        Args:
            model: Ollama embedding model name (e.g., 'nomic-embed-text')
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            batch_size: Number of texts to embed in parallel
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size

        # Validate model is available
        self._validate_model()

        logger.info(
            "ollama_embedding_function_initialized",
            model=model,
            base_url=base_url
        )

    def _validate_model(self):
        """Validate that the embedding model is available."""
        try:
            # Test with a simple embedding
            test_response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": "test"},
                timeout=10
            )
            test_response.raise_for_status()
            logger.info("embedding_model_validated", model=self.model)
        except Exception as e:
            logger.error(
                "embedding_model_validation_failed",
                model=self.model,
                error=str(e)
            )
            raise ValueError(
                f"Embedding model '{self.model}' not available. "
                f"Pull it with: ollama pull {self.model}"
            ) from e

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            input: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches to avoid overwhelming Ollama
        for i in range(0, len(input), self.batch_size):
            batch = input[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)

        logger.debug(
            "embeddings_generated",
            num_texts=len(input),
            embedding_dim=len(embeddings[0]) if embeddings else 0
        )

        return embeddings

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        embeddings = []

        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()
                embedding = data.get("embedding")

                if not embedding:
                    raise ValueError(f"No embedding returned for text: {text[:50]}")

                embeddings.append(embedding)

            except Exception as e:
                logger.error(
                    "embedding_generation_failed",
                    text_preview=text[:100],
                    error=str(e)
                )
                # Return zero vector as fallback
                # Get dimension from previous successful embedding
                dim = len(embeddings[0]) if embeddings else 768  # Default dimension
                embeddings.append([0.0] * dim)

        return embeddings


# Async version for better performance
class AsyncOllamaEmbeddingFunction(EmbeddingFunction):
    """
    Async version of Ollama embedding function.
    Use for better performance with large document sets.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        batch_size: int = 32
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size

        logger.info(
            "async_ollama_embedding_function_initialized",
            model=model
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Synchronous wrapper for async embedding generation."""
        import asyncio
        return asyncio.run(self._embed_async(input))

    async def _embed_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings asynchronously."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            tasks = []
            for text in texts:
                tasks.append(self._embed_one_async(session, text))

            embeddings = await asyncio.gather(*tasks)
            return embeddings

    async def _embed_one_async(
        self,
        session: "aiohttp.ClientSession",
        text: str
    ) -> List[float]:
        """Embed a single text asynchronously."""
        try:
            async with session.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("embedding", [])
        except Exception as e:
            logger.error("async_embedding_failed", error=str(e))
            return [0.0] * 768  # Fallback dimension
```

### Update Vector Database Class

**Replace `ragtime/storage/vector_db.py` (or create `ragtime/storage/native_vector_db.py`):**

```python
"""
Native ChromaDB vector database abstraction.
Replaces LangChain Chroma wrapper with direct ChromaDB client.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog

from ragtime.config.settings import get_settings
from ragtime.models.documents import DocumentChunk, DocumentMetadata
from ragtime.embeddings.ollama_adapter import OllamaEmbeddingFunction

logger = structlog.get_logger(__name__)


class NativeVectorDatabase:
    """
    ChromaDB vector database with native client.

    Reference: https://docs.trychroma.com/reference/py-client
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize native vector database.

        Args:
            persist_directory: Directory for persistent storage
        """
        settings = get_settings()
        self.persist_directory = persist_directory or str(Path.cwd() / 'chroma_db')
        self.embedding_model = settings.embedding_model
        self.default_collection = settings.default_collection

        # Create ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create embedding function
        self.embedding_function = OllamaEmbeddingFunction(
            model=self.embedding_model,
            base_url=settings.ollama_base_url
        )

        logger.info(
            "native_vector_db_initialized",
            persist_directory=self.persist_directory,
            embedding_model=self.embedding_model
        )

    def get_or_create_collection(self, name: Optional[str] = None):
        """
        Get or create a collection.

        Args:
            name: Collection name (uses default if not provided)

        Returns:
            ChromaDB collection
        """
        name = name or self.default_collection

        collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        logger.debug("collection_accessed", name=name)
        return collection

    def add_documents(
        self,
        texts: List[str],
        collection_name: Optional[str] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents to collection.

        Args:
            texts: List of document texts
            collection_name: Target collection
            metadatas: Optional metadata for each document
            ids: Optional IDs for documents (auto-generated if not provided)

        Returns:
            bool: Success status
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            # Generate IDs if not provided
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in texts]

            # Ensure metadatas match texts length
            if metadatas is None:
                metadatas = [{} for _ in texts]

            # Add to collection (ChromaDB handles embedding generation)
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(
                "documents_added",
                collection=collection_name,
                num_documents=len(texts)
            )

            return True

        except Exception as e:
            logger.error("add_documents_failed", error=str(e), exc_info=True)
            return False

    def similarity_search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Perform similarity search.

        Args:
            query: Query text
            collection_name: Collection to search
            k: Number of results
            filter_dict: Metadata filter

        Returns:
            List of DocumentChunk objects
        """
        try:
            settings = get_settings()
            k = k or settings.retrieval_k
            collection = self.get_or_create_collection(collection_name)

            # Query collection
            results = collection.query(
                query_texts=[query],
                n_results=k,
                where=filter_dict  # Metadata filtering
            )

            # Convert to DocumentChunk objects
            chunks = []
            for i in range(len(results['ids'][0])):
                chunk = DocumentChunk(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                    distance=results['distances'][0][i] if results.get('distances') else None
                )
                chunks.append(chunk)

            logger.info(
                "similarity_search_completed",
                collection=collection_name,
                num_results=len(chunks),
                k=k
            )

            return chunks

        except Exception as e:
            logger.error("similarity_search_failed", error=str(e), exc_info=True)
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.

        Args:
            collection_name: Name of collection to delete

        Returns:
            bool: Success status
        """
        try:
            self.client.delete_collection(name=collection_name)
            logger.info("collection_deleted", collection=collection_name)
            return True
        except Exception as e:
            logger.error("delete_collection_failed", error=str(e))
            return False

    def list_collections(self) -> List[str]:
        """
        List all collections.

        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [col.name for col in collections]


# Singleton instance
_native_vector_db: Optional[NativeVectorDatabase] = None


def get_native_vector_db(persist_directory: Optional[str] = None) -> NativeVectorDatabase:
    """Get or create the global native vector database instance."""
    global _native_vector_db

    if _native_vector_db is None:
        _native_vector_db = NativeVectorDatabase(persist_directory=persist_directory)

    return _native_vector_db
```

### Update Document Models

**Ensure `ragtime/models/documents.py` has DocumentChunk:**

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class DocumentChunk(BaseModel):
    """A chunk of a document with metadata."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    distance: Optional[float] = None  # Similarity distance
```

### Update Query Processor to Use Native VectorDB

**In `ragtime/core/pydantic_query.py`, update `_retrieve_documents`:**

```python
from ragtime.storage.native_vector_db import get_native_vector_db

def _retrieve_documents(
    self,
    query: str,
    collection_name: str
) -> Tuple[List[DocumentChunk], List[str]]:
    """Retrieve documents using native ChromaDB client."""
    try:
        vector_db = get_native_vector_db()

        # Use native similarity search
        chunks = vector_db.similarity_search(
            query=query,
            collection_name=collection_name,
            k=self.settings.retrieval_k
        )

        # Extract document IDs
        document_ids = [chunk.id for chunk in chunks]

        self.logger.info(
            "documents_retrieved",
            num_docs=len(chunks),
            collection=collection_name
        )

        return chunks, document_ids

    except Exception as e:
        self.logger.error("document_retrieval_failed", error=str(e))
        return [], []
```

### Add Async Support (Optional)

For better performance with large document sets:

```bash
pipenv install aiohttp
```

**Use `AsyncOllamaEmbeddingFunction` in production:**

```python
from ragtime.embeddings.ollama_adapter import AsyncOllamaEmbeddingFunction

# In NativeVectorDatabase.__init__
self.embedding_function = AsyncOllamaEmbeddingFunction(
    model=self.embedding_model,
    base_url=settings.ollama_base_url,
    batch_size=64  # Process 64 texts in parallel
)
```

### Testing Stage 2

```python
# tests/test_native_vector_db.py
import pytest
from ragtime.storage.native_vector_db import NativeVectorDatabase

def test_add_and_search():
    """Test adding documents and searching."""
    db = NativeVectorDatabase()

    # Add test documents
    texts = ["AI is amazing", "Machine learning rocks", "Python is great"]
    success = db.add_documents(texts, collection_name="test")
    assert success

    # Search
    results = db.similarity_search("artificial intelligence", collection_name="test", k=2)
    assert len(results) > 0
    assert results[0].content in texts

def test_metadata_filtering():
    """Test metadata filtering."""
    db = NativeVectorDatabase()

    texts = ["doc1", "doc2"]
    metadatas = [{"category": "A"}, {"category": "B"}]

    db.add_documents(texts, metadatas=metadatas, collection_name="test_filter")

    # Filter by metadata
    results = db.similarity_search(
        "doc",
        collection_name="test_filter",
        filter_dict={"category": "A"}
    )

    assert len(results) == 1
    assert results[0].metadata["category"] == "A"
```

---

## Stage 3: Replace Document Utilities

### Scope
Replace LangChain's document loaders and text splitters with custom implementations.

**Files to modify:**
- `embed.py` - Document embedding pipeline
- `ragtime/storage/vector_db.py` - Document loading methods

### Dependencies

You already have `pypdf` and `unstructured` in Pipfile. No new dependencies needed.

### Create Document Loaders

**Create `ragtime/loaders/document_loaders.py`:**

```python
"""
Custom document loaders to replace LangChain loaders.

References:
- pypdf: https://pypdf.readthedocs.io/en/stable/
- unstructured: https://unstructured-io.github.io/unstructured/
"""

import structlog
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

logger = structlog.get_logger(__name__)


class DocumentLoader:
    """Base document loader interface."""

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load document and return list of page dicts.

        Returns:
            List of dicts with keys: 'content', 'metadata'
        """
        raise NotImplementedError


class PDFLoader(DocumentLoader):
    """
    PDF document loader using pypdf.

    Reference: https://pypdf.readthedocs.io/en/stable/user/extract-text.html
    """

    def __init__(self, extract_images: bool = False):
        """
        Initialize PDF loader.

        Args:
            extract_images: Whether to extract images (not implemented)
        """
        self.extract_images = extract_images

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load PDF and extract text from each page.

        Args:
            file_path: Path to PDF file

        Returns:
            List of page dicts
        """
        try:
            reader = PdfReader(file_path)
            pages = []

            for page_num, page in enumerate(reader.pages):
                # Extract text
                text = page.extract_text()

                # Build metadata
                metadata = {
                    "source": str(file_path),
                    "page": page_num,
                    "total_pages": len(reader.pages)
                }

                # Add PDF metadata if available
                if reader.metadata:
                    metadata.update({
                        "title": reader.metadata.get("/Title", ""),
                        "author": reader.metadata.get("/Author", ""),
                        "creation_date": reader.metadata.get("/CreationDate", "")
                    })

                pages.append({
                    "content": text,
                    "metadata": metadata
                })

            logger.info(
                "pdf_loaded",
                file_path=file_path,
                num_pages=len(pages)
            )

            return pages

        except Exception as e:
            logger.error("pdf_load_failed", file_path=file_path, error=str(e))
            raise


class TextLoader(DocumentLoader):
    """
    Text/Markdown document loader.

    Supports .txt, .md files with various encodings.
    """

    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize text loader.

        Args:
            encoding: Text encoding (default: utf-8)
        """
        self.encoding = encoding

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load text file.

        Args:
            file_path: Path to text file

        Returns:
            List with single document dict
        """
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read()

            metadata = {
                "source": str(file_path),
                "file_type": Path(file_path).suffix
            }

            logger.info("text_file_loaded", file_path=file_path)

            return [{
                "content": content,
                "metadata": metadata
            }]

        except UnicodeDecodeError:
            # Retry with different encoding
            logger.warning(f"UTF-8 decode failed, trying latin-1", file_path=file_path)
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()

            return [{
                "content": content,
                "metadata": {
                    "source": str(file_path),
                    "encoding": "latin-1"
                }
            }]
        except Exception as e:
            logger.error("text_load_failed", file_path=file_path, error=str(e))
            raise


class UnstructuredLoader(DocumentLoader):
    """
    Advanced document loader using unstructured library.

    Supports complex PDFs, Word docs, HTML, etc.
    Reference: https://unstructured-io.github.io/unstructured/
    """

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load document using unstructured.

        Args:
            file_path: Path to document

        Returns:
            List of element dicts
        """
        try:
            from unstructured.partition.auto import partition

            # Partition the document
            elements = partition(filename=file_path)

            # Convert elements to our format
            docs = []
            for element in elements:
                docs.append({
                    "content": str(element),
                    "metadata": {
                        "source": str(file_path),
                        "element_type": element.category,
                        **element.metadata.to_dict()
                    }
                })

            logger.info(
                "unstructured_document_loaded",
                file_path=file_path,
                num_elements=len(docs)
            )

            return docs

        except Exception as e:
            logger.error("unstructured_load_failed", file_path=file_path, error=str(e))
            raise


def load_document(file_path: str, loader_type: str = "auto") -> List[Dict[str, Any]]:
    """
    Load a document with automatic loader selection.

    Args:
        file_path: Path to document
        loader_type: Loader type ('auto', 'pdf', 'text', 'unstructured')

    Returns:
        List of document dicts
    """
    path = Path(file_path)

    # Auto-detect loader
    if loader_type == "auto":
        if path.suffix.lower() == '.pdf':
            loader_type = "pdf"
        elif path.suffix.lower() in ['.txt', '.md']:
            loader_type = "text"
        else:
            loader_type = "unstructured"

    # Select loader
    loaders = {
        "pdf": PDFLoader(),
        "text": TextLoader(),
        "unstructured": UnstructuredLoader()
    }

    loader = loaders.get(loader_type)
    if not loader:
        raise ValueError(f"Unknown loader type: {loader_type}")

    return loader.load(file_path)
```

### Create Text Splitter

**Create `ragtime/splitters/text_splitter.py`:**

```python
"""
Custom text splitter to replace LangChain RecursiveCharacterTextSplitter.

Reference implementation based on:
https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py
"""

import structlog
from typing import List, Dict, Any

logger = structlog.get_logger(__name__)


class RecursiveTextSplitter:
    """
    Split text recursively by trying different separators.

    Tries to split by paragraph, then sentence, then word boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        Initialize text splitter.

        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: List of separators to try (in order)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators (try larger units first)
        if separators is None:
            self.separators = [
                "\n\n",  # Double newline (paragraphs)
                "\n",    # Single newline
                ". ",    # Sentence boundary
                " ",     # Word boundary
                ""       # Character boundary (fallback)
            ]
        else:
            self.separators = separators

        logger.debug(
            "text_splitter_initialized",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Try each separator recursively
        chunks = self._split_recursive(text, self.separators)

        logger.debug(
            "text_split",
            original_length=len(text),
            num_chunks=len(chunks),
            avg_chunk_size=sum(len(c) for c in chunks) / len(chunks) if chunks else 0
        )

        return chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""

        # Base case: no more separators, split by chunk size
        if not separators:
            return self._split_by_size(text)

        # Try current separator
        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by separator
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator means character-level split
            splits = list(text)

        # Merge small splits and recursively split large ones
        chunks = []
        current_chunk = ""

        for split in splits:
            # If split is too large, recurse with next separator
            if len(split) > self.chunk_size:
                # Add current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Recursively split the large piece
                sub_chunks = self._split_recursive(split, remaining_separators)
                chunks.extend(sub_chunks)
            else:
                # Check if adding this split would exceed chunk size
                test_chunk = current_chunk + separator + split if current_chunk else split

                if len(test_chunk) <= self.chunk_size:
                    current_chunk = test_chunk
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = split

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Apply overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return [c for c in chunks if c]  # Remove empty chunks

    def _split_by_size(self, text: str) -> List[str]:
        """Split text into fixed-size chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end - self.chunk_overlap > start else end

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between chunks."""
        if not chunks:
            return chunks

        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Take last N characters from previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk

            # Prepend to current chunk if not already there
            if not current_chunk.startswith(overlap_text):
                overlapped.append(overlap_text + " " + current_chunk)
            else:
                overlapped.append(current_chunk)

        return overlapped

    def split_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Split documents into chunks, preserving metadata.

        Args:
            documents: List of document dicts with 'content' and 'metadata'

        Returns:
            List of chunk dicts
        """
        all_chunks = []

        for doc in documents:
            content = doc["content"]
            metadata = doc.get("metadata", {})

            # Split text
            text_chunks = self.split_text(content)

            # Create chunk dicts
            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(text_chunks)

                all_chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata
                })

        logger.info(
            "documents_chunked",
            num_input_docs=len(documents),
            num_output_chunks=len(all_chunks)
        )

        return all_chunks
```

### Update Embedding Pipeline

**Update `embed.py` to use custom loaders/splitters:**

```python
"""
Updated embedding pipeline using custom loaders and splitters.
"""

from ragtime.loaders.document_loaders import load_document
from ragtime.splitters.text_splitter import RecursiveTextSplitter
from ragtime.storage.native_vector_db import get_native_vector_db
from ragtime.config.settings import get_settings
import structlog

logger = structlog.get_logger(__name__)


def embed_file(
    file_path: str,
    collection_name: str = 'langchain',
    loader_type: str = 'auto'
) -> bool:
    """
    Embed a file into the vector database.

    Args:
        file_path: Path to file
        collection_name: Collection to add to
        loader_type: Loader type ('auto', 'pdf', 'text', 'unstructured')

    Returns:
        bool: Success status
    """
    try:
        settings = get_settings()

        # Load document
        docs = load_document(file_path, loader_type=loader_type)
        logger.info("document_loaded", file_path=file_path, num_docs=len(docs))

        # Split into chunks
        splitter = RecursiveTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        logger.info("document_chunked", num_chunks=len(chunks))

        # Prepare for vector DB
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # Add to vector DB
        vector_db = get_native_vector_db()
        success = vector_db.add_documents(
            texts=texts,
            metadatas=metadatas,
            collection_name=collection_name
        )

        if success:
            logger.info("file_embedded_successfully", file_path=file_path)

        return success

    except Exception as e:
        logger.error("embedding_failed", file_path=file_path, error=str(e))
        return False
```

### Testing Stage 3

```python
# tests/test_loaders_splitters.py
from ragtime.loaders.document_loaders import PDFLoader, TextLoader
from ragtime.splitters.text_splitter import RecursiveTextSplitter

def test_pdf_loader():
    """Test PDF loading."""
    loader = PDFLoader()
    docs = loader.load("test.pdf")
    assert len(docs) > 0
    assert "content" in docs[0]
    assert "metadata" in docs[0]

def test_text_splitter():
    """Test text splitting."""
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)

    text = "This is a long text. " * 20
    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    # Check overlap
    for i in range(1, len(chunks)):
        # Should have some overlap
        assert chunks[i-1][-10:] in chunks[i] or chunks[i][:10] in chunks[i-1]
```

---

## Risk Mitigation

### 1. Embedding Model Compatibility

**Risk**: Using chat model instead of embedding model causes failures.

**Mitigation**:
- Add validation in `OllamaEmbeddingFunction.__init__`
- Test embedding endpoint on startup
- Fail fast with clear error message

```python
# In settings validation
def validate_embedding_model(self):
    """Validate that embedding model is available."""
    import requests
    try:
        response = requests.post(
            f"{self.ollama_base_url}/api/embeddings",
            json={"model": self.embedding_model, "prompt": "test"},
            timeout=5
        )
        response.raise_for_status()
    except Exception as e:
        raise ValueError(
            f"Embedding model '{self.embedding_model}' not available. "
            f"Install with: ollama pull nomic-embed-text"
        ) from e
```

### 2. Data Loss During Migration

**Risk**: Accidental deletion of existing vector DB data.

**Mitigation**:
- Backup `chroma_db/` before migration
- Test on separate collection first
- Use feature flags to run both systems in parallel

```bash
# Backup vector database
cp -r chroma_db chroma_db.backup.$(date +%Y%m%d)
```

### 3. Performance Degradation

**Risk**: Custom implementations are slower than LangChain.

**Mitigation**:
- Benchmark before/after
- Use async embedding function for large batches
- Monitor with existing monitoring dashboard

```python
# Add performance monitoring
import time

def embed_file_with_timing(file_path, collection_name):
    start = time.time()
    result = embed_file(file_path, collection_name)
    duration = time.time() - start

    logger.info("embedding_performance", duration=duration, file_path=file_path)
    return result
```

### 4. API Compatibility Issues

**Risk**: Ollama OpenAI-compatible endpoint differs from OpenAI.

**Mitigation**:
- Test all model parameters (temperature, max_tokens, etc.)
- Verify streaming works if needed
- Add fallback error handling

### 5. Retrieval Quality Changes

**Risk**: Different chunking/retrieval affects response quality.

**Mitigation**:
- A/B test with same queries
- Compare document relevance scores
- Keep test query set for regression testing

```python
# Create test_queries.json
test_queries = [
    {"query": "What is RAG?", "expected_docs": ["rag_intro.pdf"]},
    {"query": "How does feedback work?", "expected_docs": ["feedback_doc.pdf"]}
]
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_pydantic_migration.py
import pytest
from ragtime.core.pydantic_query import PydanticQueryProcessor
from ragtime.storage.native_vector_db import NativeVectorDatabase
from ragtime.embeddings.ollama_adapter import OllamaEmbeddingFunction

def test_ollama_embeddings():
    """Test Ollama embedding generation."""
    embedder = OllamaEmbeddingFunction(model="nomic-embed-text")
    embeddings = embedder(["test text"])
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)
    assert len(embeddings[0]) > 0

def test_native_vector_db():
    """Test native vector database operations."""
    db = NativeVectorDatabase()

    # Add documents
    texts = ["AI is great", "ML is awesome"]
    success = db.add_documents(texts, collection_name="test_migration")
    assert success

    # Search
    results = db.similarity_search("artificial intelligence", collection_name="test_migration")
    assert len(results) > 0

def test_pydantic_query_processor():
    """Test Pydantic AI query processor."""
    processor = PydanticQueryProcessor()

    response, doc_ids, metadata = processor.process_query(
        question="What is Ragtime Gal?",
        collection_name="test_migration"
    )

    assert isinstance(response, str)
    assert metadata["engine"] == "pydantic_ai"
```

### Integration Tests

```python
# tests/test_end_to_end_migration.py
import pytest
from embed import embed_file
from ragtime.core.pydantic_query import PydanticQueryProcessor

def test_end_to_end_workflow():
    """Test complete embed -> query workflow."""
    # Embed test document
    success = embed_file("test_doc.pdf", collection_name="e2e_test")
    assert success

    # Query
    processor = PydanticQueryProcessor()
    response, doc_ids, metadata = processor.process_query(
        question="What is in the test document?",
        collection_name="e2e_test"
    )

    assert len(response) > 0
    assert len(doc_ids) > 0
```

### Comparison Tests

```python
# tests/test_langchain_parity.py
import pytest
from ragtime.core.query_processor import QueryProcessor  # LangChain version
from ragtime.core.pydantic_query import PydanticQueryProcessor  # Pydantic AI version

@pytest.fixture
def test_queries():
    return [
        "What is RAG?",
        "How does the feedback system work?",
        "Explain the architecture"
    ]

def test_response_similarity(test_queries):
    """Compare responses from both processors."""
    langchain_processor = QueryProcessor()
    pydantic_processor = PydanticQueryProcessor()

    for query in test_queries:
        lc_response, _, _ = langchain_processor.process_query(
            question=query,
            collection_name="test"
        )

        pa_response, _, _ = pydantic_processor.process_query(
            question=query,
            collection_name="test"
        )

        # Both should return non-empty responses
        assert len(lc_response) > 0
        assert len(pa_response) > 0

        # Responses should be reasonably similar in length
        length_ratio = len(pa_response) / len(lc_response)
        assert 0.5 < length_ratio < 2.0  # Within 2x length
```

---

## Rollback Plan

### Feature Flag Approach

Keep both systems and use feature flag to switch:

```python
# In ragtime/config/settings.py
use_pydantic_ai: bool = Field(
    default=False,
    description="Use Pydantic AI (true) or LangChain (false)"
)
use_native_vector_db: bool = Field(
    default=False,
    description="Use native ChromaDB (true) or LangChain wrapper (false)"
)
```

### Rollback Steps

1. **Set feature flags to False**:
   ```bash
   # In .env
   USE_PYDANTIC_AI=false
   USE_NATIVE_VECTOR_DB=false
   ```

2. **Restart application**:
   ```bash
   pipenv run python app.py
   ```

3. **Verify LangChain is working**:
   ```bash
   curl -X POST http://localhost:8084/query \
     -H "Content-Type: application/json" \
     -d '{"question": "test", "collection": "langchain"}'
   ```

4. **If data corruption occurred, restore backup**:
   ```bash
   rm -rf chroma_db
   cp -r chroma_db.backup.YYYYMMDD chroma_db
   ```

### Keep Both Systems Temporarily

During migration, keep both systems available:

```python
# In ragtime/api/queries.py
@bp.route('/query', methods=['POST'])
def query_endpoint():
    settings = get_settings()

    # Allow override via query parameter
    use_pydantic = request.args.get('use_pydantic', settings.use_pydantic_ai)

    if use_pydantic:
        processor = PydanticQueryProcessor()
    else:
        processor = QueryProcessor()  # LangChain version

    # Rest of endpoint...
```

This allows testing both systems side-by-side:
```bash
# Test LangChain
curl http://localhost:8084/query?use_pydantic=false -d '...'

# Test Pydantic AI
curl http://localhost:8084/query?use_pydantic=true -d '...'
```

---

## References

### Pydantic AI Documentation
- **Main docs**: https://ai.pydantic.dev/
- **Agents**: https://ai.pydantic.dev/agents/
- **Models & Providers**: https://ai.pydantic.dev/models/overview/
- **OpenAI Model**: https://ai.pydantic.dev/models/openai/
- **Tools**: https://ai.pydantic.dev/tools/
- **Examples**: https://ai.pydantic.dev/examples/

### Ollama API Documentation
- **API Reference**: https://github.com/ollama/ollama/blob/main/docs/api.md
- **Embeddings API**: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
- **OpenAI Compatibility**: https://github.com/ollama/ollama/blob/main/docs/openai.md
- **Model Library**: https://ollama.com/library

### ChromaDB Documentation
- **Python Client**: https://docs.trychroma.com/reference/py-client
- **Embeddings**: https://docs.trychroma.com/guides#embeddings
- **Querying**: https://docs.trychroma.com/guides#querying-a-collection
- **Filtering**: https://docs.trychroma.com/guides#filtering-by-metadata

### Document Processing
- **pypdf**: https://pypdf.readthedocs.io/en/stable/
- **unstructured**: https://unstructured-io.github.io/unstructured/
- **Text Splitting Theory**: https://www.pinecone.io/learn/chunking-strategies/

### Python Libraries
- **OpenAI Python**: https://github.com/openai/openai-python
- **aiohttp** (async HTTP): https://docs.aiohttp.org/
- **Pydantic**: https://docs.pydantic.dev/

### LangChain (for comparison)
- **LangChain Expression Language (LCEL)**: https://python.langchain.com/docs/expression_language/
- **RecursiveCharacterTextSplitter**: https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter

---

## Next Steps

### Immediate (Stage 1)
1. ✅ Install dependencies: `pipenv install pydantic-ai openai`
2. ✅ Add `OLLAMA_OPENAI_BASE_URL` to `.env`
3. ✅ Create `ragtime/core/pydantic_query.py`
4. ✅ Add feature flag to settings
5. ✅ Write unit tests
6. ✅ Deploy with feature flag off
7. ✅ Enable feature flag and monitor

### Short-term (Stage 2)
1. ✅ Pull embedding model: `ollama pull nomic-embed-text`
2. ✅ Update `EMBEDDING_MODEL` in `.env`
3. ✅ Create `ragtime/embeddings/ollama_adapter.py`
4. ✅ Create `ragtime/storage/native_vector_db.py`
5. ✅ Write integration tests
6. ✅ Test on non-production collection
7. ✅ Enable native vector DB feature flag

### Long-term (Stage 3)
1. ✅ Create `ragtime/loaders/document_loaders.py`
2. ✅ Create `ragtime/splitters/text_splitter.py`
3. ✅ Update `embed.py`
4. ✅ Test document processing pipeline
5. ✅ Remove LangChain dependencies from `Pipfile`
6. ✅ Update documentation

### Maintenance
- Monitor performance metrics
- Collect user feedback
- Tune chunk sizes and overlap
- Optimize embedding batch sizes
- Consider adding more advanced features (streaming, multi-agent, etc.)

---

## Questions & Support

If you encounter issues during migration:

1. Check Ollama is running: `curl http://localhost:11434/api/tags`
2. Verify embedding model: `ollama list | grep embed`
3. Test OpenAI endpoint: `curl http://localhost:11434/v1/models`
4. Review logs: `tail -f logs/ragtime_gal_errors.log`
5. Check ChromaDB: `ls -la chroma_db/`

For questions, refer to:
- Pydantic AI GitHub: https://github.com/pydantic/pydantic-ai
- Pydantic AI Slack: https://logfire.pydantic.dev/docs/join-slack/
- This migration guide

---

**Document Version**: 1.0
**Last Updated**: 2025-11-16
**Author**: Ragtime Gal Development Team
