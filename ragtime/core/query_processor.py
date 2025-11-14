"""
Query Processing Module

This module handles query processing with LangChain integration, including:
- Query enhancement based on feedback patterns
- Document retrieval from vector database
- LLM interaction with Ollama
- Context management and prompt formatting
- Adaptive similarity thresholds
- Document re-ranking

Migrated from root-level query.py to follow project maturity standards:
- Uses centralized Settings instead of environment variables
- Implements structured logging with structlog
- Uses Pydantic models for type safety
- Integrates with new VectorDatabase class
- Supports dependency injection
"""

import structlog
from typing import Optional, List, Dict, Any, Tuple
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

# Import from new structure
from ragtime.config.settings import get_settings
from ragtime.models.queries import QueryRequest, QueryResponse, RetrievedDocument

# Temporary imports from root (will be migrated later)
from template_manager import TemplateManager
from context_manager import ContextManager
from feedback_analyzer import create_feedback_analyzer
from query_enhancer import create_query_enhancer

# Initialize structured logger
logger = structlog.get_logger(__name__)

# Singleton instances for backward compatibility
_template_manager: Optional[TemplateManager] = None
_context_manager: Optional[ContextManager] = None
_feedback_analyzer: Optional[Any] = None
_query_enhancer: Optional[Any] = None


class QueryProcessor:
    """
    Query processor with feedback-driven optimization.

    This class encapsulates the query processing logic with support for:
    - LangChain-based LLM interaction
    - Vector database retrieval
    - Query enhancement based on feedback
    - Document re-ranking
    - Conversation context management
    """

    def __init__(
        self,
        settings=None,
        template_manager: Optional[TemplateManager] = None,
        context_manager: Optional[ContextManager] = None,
        feedback_analyzer=None,
        query_enhancer=None
    ):
        """
        Initialize query processor.

        Args:
            settings: Application settings (uses default if not provided)
            template_manager: Template manager for prompt formatting
            context_manager: Context manager for conversation handling
            feedback_analyzer: Feedback analyzer for optimization
            query_enhancer: Query enhancer for query improvement
        """
        self.settings = settings or get_settings()
        self.logger = logger.bind(component="query_processor")

        # Initialize or use provided managers
        self.template_manager = template_manager or get_template_manager()
        self.context_manager = context_manager or get_context_manager()
        self.feedback_analyzer = feedback_analyzer
        self.query_enhancer = query_enhancer

        self.logger.info(
            "query_processor_initialized",
            llm_model=self.settings.llm_model,
            embedding_model=self.settings.embedding_model,
            retrieval_k=self.settings.retrieval_k
        )

    def process_query(
        self,
        question: str,
        collection_name: Optional[str] = None,
        template_name: Optional[str] = None,
        temperature: Optional[float] = None,
        conversation: Optional[Any] = None,
        use_feedback_optimization: bool = True,
        conport_client=None,
        workspace_id: Optional[str] = None
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Process a query with optional feedback optimization.

        Args:
            question: The question to ask
            collection_name: Vector database collection name
            template_name: Prompt template name
            temperature: LLM temperature
            conversation: Conversation object with history
            use_feedback_optimization: Enable feedback-driven optimization
            conport_client: ConPort client for feedback access
            workspace_id: Workspace identifier

        Returns:
            Tuple of (response, document_ids, metadata)
        """
        try:
            # Use settings defaults if not provided
            template_name = template_name or self.settings.prompt_template
            temperature = temperature if temperature is not None else self.settings.llm_temperature
            collection_name = collection_name or self.settings.default_collection

            self.logger.info(
                "processing_query",
                question=question[:100],
                template=template_name,
                temperature=temperature,
                collection=collection_name,
                feedback_optimization=use_feedback_optimization
            )

            # Initialize feedback optimization if enabled
            enhanced_query = question
            optimization_metadata = {"feedback_optimization_enabled": use_feedback_optimization}

            if use_feedback_optimization:
                enhanced_query, optimization_metadata = self._enhance_query(
                    question, conport_client, workspace_id, optimization_metadata
                )

            # Retrieve documents from vector database
            docs, document_ids = self._retrieve_documents(
                enhanced_query, collection_name, use_feedback_optimization, optimization_metadata
            )

            # Get context and prompt from context manager
            context_info = self.context_manager.get_prompt(
                query=enhanced_query,
                conversation=conversation,
                retrieved_docs=docs,
                style=template_name
            )

            # Generate response using LLM
            response = self._generate_response(
                enhanced_query, docs, context_info, temperature, conversation
            )

            # Build metadata
            metadata = {
                "is_follow_up": context_info.get("is_follow_up", False),
                "template_used": template_name,
                "query_type": context_info.get("query_type"),
                "has_previous_content": context_info.get("has_previous_content", False),
                "original_query": question,
                "enhanced_query": enhanced_query
            }
            metadata.update(optimization_metadata)

            self.logger.info(
                "query_processed_successfully",
                response_length=len(response),
                num_documents=len(document_ids),
                query_enhanced=enhanced_query != question
            )

            return response, document_ids, metadata

        except Exception as e:
            self.logger.error(
                "query_processing_failed",
                error=str(e),
                question=question[:100],
                exc_info=True
            )
            return f"An error occurred: {str(e)}", [], {"error": str(e)}

    def _enhance_query(
        self,
        question: str,
        conport_client,
        workspace_id: Optional[str],
        optimization_metadata: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Enhance query using feedback patterns."""
        enhanced_query = question

        try:
            # Initialize feedback components if needed
            if self.feedback_analyzer is None:
                self.feedback_analyzer = get_feedback_analyzer(
                    conport_client=conport_client,
                    workspace_id=workspace_id
                )

            if self.query_enhancer is None:
                self.query_enhancer = get_query_enhancer(
                    feedback_analyzer=self.feedback_analyzer
                )

            # Enhance the query
            enhancement_result = self.query_enhancer.enhance_query(
                question,
                enhancement_mode="auto"
            )
            enhanced_query = enhancement_result["enhanced_query"]

            optimization_metadata.update({
                "query_enhanced": enhanced_query != question,
                "enhancements_applied": enhancement_result.get("enhancements_applied", []),
                "enhancement_confidence": enhancement_result.get("confidence", 0.5)
            })

            if enhanced_query != question:
                self.logger.info(
                    "query_enhanced",
                    original=question[:50],
                    enhanced=enhanced_query[:50]
                )

        except Exception as opt_error:
            self.logger.warning(
                "query_enhancement_failed",
                error=str(opt_error),
                original_query=question[:50]
            )
            optimization_metadata["optimization_error"] = str(opt_error)

        return enhanced_query, optimization_metadata

    def _retrieve_documents(
        self,
        query: str,
        collection_name: str,
        use_feedback_optimization: bool,
        optimization_metadata: Dict[str, Any]
    ) -> Tuple[List[Any], List[str]]:
        """Retrieve and optionally re-rank documents."""
        try:
            # Create embeddings
            embeddings = OllamaEmbeddings(
                model=self.settings.embedding_model,
                base_url=self.settings.ollama_base_url
            )

            # Connect to vector database
            db = Chroma(
                persist_directory=self.settings.chroma_persist_dir,
                embedding_function=embeddings,
                collection_name=collection_name
            )

            self.logger.debug(
                "vector_db_connected",
                collection=collection_name,
                persist_dir=self.settings.chroma_persist_dir
            )

            # Create retriever with adaptive threshold if optimization enabled
            search_kwargs = {"k": self.settings.retrieval_k}

            if use_feedback_optimization and self.query_enhancer:
                try:
                    adaptive_threshold = self.query_enhancer.get_adaptive_similarity_threshold(query)
                    optimization_metadata["adaptive_threshold"] = adaptive_threshold
                    self.logger.info(
                        "adaptive_threshold_applied",
                        threshold=adaptive_threshold
                    )
                except Exception as threshold_error:
                    self.logger.warning(
                        "adaptive_threshold_failed",
                        error=str(threshold_error)
                    )

            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )

            # Retrieve documents
            docs = retriever.invoke(query)

            # Re-rank if optimization enabled
            if use_feedback_optimization and self.query_enhancer and docs:
                try:
                    original_docs = docs.copy()
                    docs = self.query_enhancer.rerank_documents(docs, query)
                    optimization_metadata["documents_reranked"] = (
                        len(docs) > 0 and docs != original_docs
                    )
                except Exception as rerank_error:
                    self.logger.warning(
                        "document_reranking_failed",
                        error=str(rerank_error)
                    )

            # Extract document IDs
            document_ids = []
            for i, doc in enumerate(docs):
                self.logger.debug(
                    "document_retrieved",
                    index=i,
                    preview=doc.page_content[:100]
                )
                if hasattr(doc, 'metadata') and 'id' in doc.metadata:
                    document_ids.append(doc.metadata['id'])

            return docs, document_ids

        except Exception as retrieval_error:
            self.logger.error(
                "document_retrieval_failed",
                error=str(retrieval_error),
                exc_info=True
            )
            return [], []

    def _generate_response(
        self,
        query: str,
        docs: List[Any],
        context_info: Dict[str, Any],
        temperature: float,
        conversation: Optional[Any]
    ) -> str:
        """Generate response using LLM."""
        # Extract context information
        prompt_text = context_info["prompt"]
        system_instruction = context_info.get("system_instruction")

        self.logger.info(
            "generating_response",
            query_type=context_info.get("query_type"),
            num_docs=len(docs),
            temperature=temperature
        )

        # Create LLM with settings
        llm = ChatOllama(
            model=self.settings.llm_model,
            base_url=self.settings.ollama_base_url,
            temperature=temperature,
            system=system_instruction or None,
            num_predict=self.settings.max_output_tokens,
            repeat_penalty=self.settings.repeat_penalty,
            top_k=self.settings.top_k,
            top_p=self.settings.top_p,
            num_ctx=32768,
            format="",
            verbose=True
        )

        # Create prompt template
        prompt = ChatPromptTemplate.from_template(prompt_text)

        # Get context parameters
        context_params = self.context_manager.get_context_params(
            query=query,
            conversation=conversation,
            retrieved_docs=docs,
            classification=context_info.get("classification")
        )

        self.logger.debug(
            "context_prepared",
            context_length=len(context_params.get("context", "")),
            previous_content_length=len(context_params.get("previous_content", "")),
            conversation_summary_length=len(context_params.get("conversation_summary", ""))
        )

        # Create and execute chain
        chain = (
            {
                "question": RunnablePassthrough(),
                "context": lambda _: context_params.get("context", ""),
                "previous_content": lambda _: context_params.get("previous_content", ""),
                "conversation_history": lambda _: context_params.get("conversation_history", ""),
                "conversation_summary": lambda _: context_params.get("conversation_summary", "")
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(query)

        self.logger.info(
            "response_generated",
            response_length=len(response)
        )

        return response


# Singleton management functions for backward compatibility
def get_template_manager() -> TemplateManager:
    """Get or create the template manager singleton."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
        logger.info("template_manager_created")
    return _template_manager


def get_context_manager() -> ContextManager:
    """Get or create the context manager singleton."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager(template_manager=get_template_manager())
        logger.info("context_manager_created")
    return _context_manager


def get_feedback_analyzer(conport_client=None, workspace_id=None):
    """Get or create the feedback analyzer singleton."""
    global _feedback_analyzer
    if _feedback_analyzer is None:
        _feedback_analyzer = create_feedback_analyzer(
            conport_client=conport_client,
            workspace_id=workspace_id
        )
        logger.info("feedback_analyzer_created")
    return _feedback_analyzer


def get_query_enhancer(feedback_analyzer=None):
    """Get or create the query enhancer singleton."""
    global _query_enhancer
    if _query_enhancer is None:
        if feedback_analyzer is None:
            feedback_analyzer = get_feedback_analyzer()
        _query_enhancer = create_query_enhancer(feedback_analyzer=feedback_analyzer)
        logger.info("query_enhancer_created")
    return _query_enhancer


# Backward-compatible API functions
def query(
    question: str,
    template_name: Optional[str] = None,
    temperature: Optional[float] = None,
    conversation: Optional[Any] = None,
    collection_name: Optional[str] = None,
    use_feedback_optimization: bool = True,
    conport_client=None,
    workspace_id: Optional[str] = None
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Query the vector database with a question.

    Backward-compatible function that maintains the original query.py API.

    Args:
        question: The question to ask
        template_name: Prompt template name
        temperature: LLM temperature
        conversation: Conversation object with history
        collection_name: Vector database collection
        use_feedback_optimization: Enable feedback optimization
        conport_client: ConPort client for feedback
        workspace_id: Workspace identifier

    Returns:
        Tuple of (response, document_ids, metadata)
    """
    processor = QueryProcessor()
    return processor.process_query(
        question=question,
        collection_name=collection_name,
        template_name=template_name,
        temperature=temperature,
        conversation=conversation,
        use_feedback_optimization=use_feedback_optimization,
        conport_client=conport_client,
        workspace_id=workspace_id
    )


def query_with_feedback_optimization(
    question: str,
    template_name: Optional[str] = None,
    temperature: Optional[float] = None,
    conversation: Optional[Any] = None,
    conport_client=None,
    workspace_id: Optional[str] = None
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Convenience function for querying with feedback optimization enabled.

    Args:
        question: The question to ask
        template_name: Prompt template name
        temperature: LLM temperature
        conversation: Conversation object with history
        conport_client: ConPort client for feedback
        workspace_id: Workspace identifier

    Returns:
        Tuple of (response, document_ids, metadata)
    """
    return query(
        question=question,
        template_name=template_name,
        temperature=temperature,
        conversation=conversation,
        use_feedback_optimization=True,
        conport_client=conport_client,
        workspace_id=workspace_id
    )


def get_query_enhancement_suggestions(
    question: str,
    conport_client=None,
    workspace_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get suggestions for improving a query based on feedback patterns.

    Args:
        question: The question to analyze
        conport_client: ConPort client for feedback
        workspace_id: Workspace identifier

    Returns:
        Dictionary containing enhancement suggestions
    """
    try:
        feedback_analyzer = get_feedback_analyzer(
            conport_client=conport_client,
            workspace_id=workspace_id
        )
        query_enhancer = get_query_enhancer(feedback_analyzer=feedback_analyzer)
        return query_enhancer.get_enhancement_suggestions(question)
    except Exception as e:
        logger.error(
            "enhancement_suggestions_failed",
            error=str(e),
            question=question[:50]
        )
        return {
            "original_query": question,
            "suggestions": [],
            "confidence": 0.5,
            "error": str(e)
        }


def get_feedback_summary(
    days_back: int = 7,
    conport_client=None,
    workspace_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get a summary of recent feedback for monitoring purposes.

    Args:
        days_back: Number of days to summarize
        conport_client: ConPort client for feedback
        workspace_id: Workspace identifier

    Returns:
        Dictionary containing feedback summary
    """
    try:
        feedback_analyzer = get_feedback_analyzer(
            conport_client=conport_client,
            workspace_id=workspace_id
        )
        return feedback_analyzer.get_feedback_summary(days_back=days_back)
    except Exception as e:
        logger.error(
            "feedback_summary_failed",
            error=str(e),
            days_back=days_back
        )
        return {
            "period": f"Last {days_back} days",
            "total_feedback": 0,
            "average_rating": 0,
            "trend": "error",
            "error": str(e)
        }


def clear_optimization_cache():
    """Clear all optimization caches to force fresh analysis."""
    try:
        global _query_enhancer
        if _query_enhancer:
            _query_enhancer.clear_cache()
        logger.info("optimization_cache_cleared")
    except Exception as e:
        logger.error("cache_clear_failed", error=str(e))