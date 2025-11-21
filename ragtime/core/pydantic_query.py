"""
Pydantic AI-based query processor.
Simplified, type-safe alternative to LangChain LCEL.
"""

import structlog
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import openai

from ragtime.config.settings import get_settings
from ragtime.models.queries import QueryRequest, QueryResponse, RetrievedDocument

# Import existing managers (now from ragtime package)
from ragtime.utils.templates import TemplateManager
from ragtime.core.context_manager import ContextManager
from ragtime.services.feedback_analyzer import create_feedback_analyzer
from ragtime.services.query_enhancer import create_query_enhancer

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
        context_manager: Optional[ContextManager] = None,
    ):
        self.settings = settings or get_settings()
        self.logger = logger.bind(component="pydantic_query_processor")

        self.template_manager = template_manager or TemplateManager()
        self.context_manager = context_manager or ContextManager(
            template_manager=self.template_manager
        )

        # Create OpenAI client for Ollama
        self.openai_client = openai.OpenAI(
            base_url=self.settings.ollama_openai_base_url, api_key="ollama"
        )

        # Create OpenAI provider for Ollama
        self.openai_provider = OpenAIProvider(
            api_key="ollama", base_url=self.settings.ollama_openai_base_url
        )

        self.logger.info(
            "pydantic_query_processor_initialized",
            llm_model=self.settings.llm_model,
            ollama_endpoint=self.settings.ollama_openai_base_url,
        )

    def process_query(
        self,
        question: str,
        collection_name: Optional[str] = None,
        template_name: Optional[str] = None,
        temperature: Optional[float] = None,
        conversation: Optional[Any] = None,
        use_typed_output: bool = False,
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
            temperature = (
                temperature
                if temperature is not None
                else self.settings.llm_temperature
            )
            collection_name = collection_name or self.settings.default_collection

            self.logger.info(
                "processing_query",
                question=question[:100],
                template=template_name,
                temperature=temperature,
            )

            # Step 1: Retrieve documents (using existing LangChain wrapper for now)
            docs, document_ids = self._retrieve_documents(question, collection_name)

            # Step 2: Build context and prompt
            context_info = self.context_manager.get_prompt(
                query=question,
                conversation=conversation,
                retrieved_docs=docs,
                style=template_name,
            )

            # Step 3: Generate response with Pydantic AI
            response = self._generate_with_agent(
                question=question,
                docs=docs,
                context_info=context_info,
                temperature=temperature,
                conversation=conversation,
                use_typed_output=use_typed_output,
            )

            # Build metadata
            metadata = {
                "is_follow_up": context_info.get("is_follow_up", False),
                "template_used": template_name,
                "query_type": context_info.get("query_type"),
                "engine": "pydantic_ai",
            }

            return response, document_ids, metadata

        except Exception as e:
            self.logger.error("query_processing_failed", error=str(e), exc_info=True)
            return f"An error occurred: {str(e)}", [], {"error": str(e)}

    def _retrieve_documents(
        self, query: str, collection_name: str
    ) -> Tuple[List[Any], List[str]]:
        """Retrieve documents using existing vector DB."""
        try:
            vector_db = get_vector_db()

            # Use existing similarity search
            docs = vector_db.similarity_search(
                query=query,
                collection_name=collection_name,
                k=self.settings.retrieval_k,
            )

            # Extract document IDs
            document_ids = []
            for doc in docs:
                if hasattr(doc, "metadata") and "id" in doc.metadata:
                    document_ids.append(doc.metadata["id"])

            self.logger.info(
                "documents_retrieved", num_docs=len(docs), collection=collection_name
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
        use_typed_output: bool,
    ) -> str:
        """Generate response using Pydantic AI agent."""

        # Get context parameters
        context_params = self.context_manager.get_context_params(
            query=question,
            conversation=conversation,
            retrieved_docs=docs,
            classification=context_info.get("classification"),
        )

        # Expand prompt template
        prompt_text = context_info["prompt"]
        full_prompt = prompt_text.format(
            question=question,
            context=context_params.get("context", ""),
            previous_content=context_params.get("previous_content", ""),
            conversation_history=context_params.get("conversation_history", ""),
            conversation_summary=context_params.get("conversation_summary", ""),
        )

        system_instruction = context_info.get("system_instruction", "")

        # Create model
        model = OpenAIChatModel(
            model_name=self.settings.llm_model,
            provider=self.openai_provider
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
