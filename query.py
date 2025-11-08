import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from template_manager import TemplateManager
from context_manager import ContextManager
from feedback_analyzer import create_feedback_analyzer
from query_enhancer import create_query_enhancer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables with defaults
LLM_MODEL = os.getenv('LLM_MODEL', 'sixthwood')  # Use the custom sixthwood model by default
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'mistral')  # Keep using mistral for embeddings
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', '4'))

# Initialize managers (lazy loading)
_template_manager = None
_context_manager = None
_feedback_analyzer = None
_query_enhancer = None

def get_template_manager():
    """Get or create the template manager singleton"""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager

def get_context_manager():
    """Get or create the context manager singleton"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager(template_manager=get_template_manager())
    return _context_manager

def get_feedback_analyzer(conport_client=None, workspace_id=None):
    """Get or create the feedback analyzer singleton"""
    global _feedback_analyzer
    if _feedback_analyzer is None:
        _feedback_analyzer = create_feedback_analyzer(conport_client=conport_client, workspace_id=workspace_id)
    return _feedback_analyzer

def get_query_enhancer(feedback_analyzer=None):
    """Get or create the query enhancer singleton"""
    global _query_enhancer
    if _query_enhancer is None:
        if feedback_analyzer is None:
            feedback_analyzer = get_feedback_analyzer()
        _query_enhancer = create_query_enhancer(feedback_analyzer=feedback_analyzer)
    return _query_enhancer

def query(question: str,
          template_name: Optional[str] = None,
          temperature: Optional[float] = None,
          conversation: Optional[Any] = None,
          collection_name: Optional[str] = None,
          use_feedback_optimization: bool = True,
          conport_client=None,
          workspace_id: Optional[str] = None) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Query the vector database with a question, optionally using conversation history and feedback optimization.

    This enhanced version uses the ContextManager to dynamically select and format prompts
    based on conversation context, and optionally applies feedback-driven retrieval optimization.

    Args:
        question (str): The question to ask
        template_name (str, optional): The prompt template to use. Defaults to env var or 'sixthwood'
        temperature (float, optional): Model temperature. Defaults to env var or 1.0
        conversation (Conversation, optional): Conversation object containing history
        use_feedback_optimization (bool): Whether to use feedback-driven optimization. Defaults to True
        conport_client: ConPort MCP client for feedback data access
        workspace_id (str, optional): Workspace identifier for ConPort operations

    Returns:
        Tuple[str, List[str], Dict[str, Any]]: The model's response, document IDs, and metadata
    """
    try:
        # Use arguments if provided, otherwise fall back to env vars
        if template_name is None:
            template_name = os.getenv('PROMPT_TEMPLATE', 'sixthwood')

        if temperature is None:
            temperature = float(os.getenv('LLM_TEMPERATURE', '1.0'))

        # Get context manager
        context_manager = get_context_manager()

        # Initialize feedback optimization components if enabled
        query_enhancer = None
        enhanced_query = question
        optimization_metadata = {"feedback_optimization_enabled": use_feedback_optimization}

        if use_feedback_optimization:
            try:
                feedback_analyzer = get_feedback_analyzer(conport_client=conport_client, workspace_id=workspace_id)
                query_enhancer = get_query_enhancer(feedback_analyzer=feedback_analyzer)

                # Enhance the query based on feedback patterns
                enhancement_result = query_enhancer.enhance_query(question, enhancement_mode="auto")
                enhanced_query = enhancement_result["enhanced_query"]
                optimization_metadata.update({
                    "query_enhanced": enhanced_query != question,
                    "enhancements_applied": enhancement_result.get("enhancements_applied", []),
                    "enhancement_confidence": enhancement_result.get("confidence", 0.5)
                })

                if enhanced_query != question:
                    logger.info("Query enhanced from '%s' to '%s'", question, enhanced_query)

            except Exception as opt_error:
                logger.warning("Feedback optimization failed, proceeding with original query: %s", str(opt_error))
                optimization_metadata["optimization_error"] = str(opt_error)

        logger.info("Processing query using model %s (temp=%s, style=%s): %s",
                   LLM_MODEL, temperature, template_name, enhanced_query)

        # Create embeddings and connect to the vector database
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        logger.info("Using Ollama embeddings with model: %s", EMBEDDING_MODEL)

        try:
            db = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings,
                collection_name=collection_name or 'langchain'
            )
            logger.info("Connected to ChromaDB at %s", CHROMA_PERSIST_DIR)
        except Exception as db_error:
            logger.error("Error connecting to ChromaDB: %s", str(db_error))
            return f"Error connecting to the vector database: {str(db_error)}", [], {"error": str(db_error)}

        # Create retriever with adaptive similarity threshold if feedback optimization is enabled
        try:
            search_kwargs = {"k": RETRIEVAL_K}

            # Apply adaptive similarity threshold if feedback optimization is enabled
            if use_feedback_optimization and query_enhancer:
                try:
                    adaptive_threshold = query_enhancer.get_adaptive_similarity_threshold(enhanced_query)
                    # search_kwargs["score_threshold"] = adaptive_threshold
                    optimization_metadata["adaptive_threshold"] = adaptive_threshold
                    logger.info("Using adaptive similarity threshold: %s", adaptive_threshold)
                except Exception as threshold_error:
                    logger.warning("Failed to get adaptive threshold: %s", str(threshold_error))

            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
            logger.info("Created retriever with k=%s", RETRIEVAL_K)
        except Exception as retriever_error:
            logger.error("Error creating retriever: %s", str(retriever_error))
            return f"Error setting up document retrieval: {str(retriever_error)}", [], {"error": str(retriever_error)}

        # Get relevant documents using the enhanced query
        document_ids = []
        try:
            # docs = retriever.get_relevant_documents(enhanced_query)
            docs = retriever.invoke(enhanced_query)

            # Apply document re-ranking if feedback optimization is enabled
            if use_feedback_optimization and query_enhancer and docs:
                try:
                    original_docs = docs.copy()
                    docs = query_enhancer.rerank_documents(docs, enhanced_query)
                    optimization_metadata["documents_reranked"] = len(docs) > 0 and docs != original_docs
                except Exception as rerank_error:
                    logger.warning("Document re-ranking failed: %s", str(rerank_error))

            for i, doc in enumerate(docs):
                logger.info("Retrieved document %d: %s...", i+1, doc.page_content[:100])  # Log first 100 chars
                if hasattr(doc, 'metadata') and 'id' in doc.metadata:
                    document_ids.append(doc.metadata['id'])
        except Exception as retrieval_error:
            logger.error("Error retrieving documents: %s", str(retrieval_error))
            # Continue without document retrieval
            docs = []
            logger.warning("Proceeding without document retrieval")

        # Get prompt and context information from context manager (use enhanced query for context)
        context_info = context_manager.get_prompt(
            query=enhanced_query,
            conversation=conversation,
            retrieved_docs=docs,
            style=template_name
        )

        # Extract prompt and metadata
        prompt_text = context_info["prompt"]
        system_instruction = context_info["system_instruction"]
        is_follow_up = context_info["is_follow_up"]

        logger.info("Using %s prompt with query type: %s",
                   template_name, context_info["query_type"])

        # Create LLM with the custom model and enhanced parameters for longer outputs
        max_output_tokens = int(os.getenv('MAX_OUTPUT_TOKENS', '16384'))  # Much higher default for longer outputs
        repeat_penalty = float(os.getenv('REPEAT_PENALTY', '1.1'))
        top_k = int(os.getenv('TOP_K', '40'))
        top_p = float(os.getenv('TOP_P', '0.9'))

        llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            system=system_instruction or None,  # Use system instruction if available
            num_predict=max_output_tokens,  # Configurable token limit for much longer responses
            repeat_penalty=repeat_penalty,  # Configurable repetition penalty
            top_k=top_k,  # Control diversity of token selection
            top_p=top_p,  # Nucleus sampling for better quality
            num_ctx=32768,  # Larger context window
            format="",  # Allow free-form responses
            verbose=True  # More detailed logging
        )
        logger.info("Using Ollama LLM with model: %s, temperature: %s, max_tokens: %d", LLM_MODEL, temperature, max_output_tokens)

        # Create prompt template
        prompt = ChatPromptTemplate.from_template(prompt_text)
        logger.info("Created prompt template")

        # Get context parameters from context manager to pass to the chain
        context_params = context_manager.get_context_params(
            query=enhanced_query,
            conversation=conversation,
            retrieved_docs=docs,
            classification=context_info.get("classification")
        )
        logger.info("Prepared context parameters: context=%d chars, previous_content=%d chars, conversation_summary=%d chars",
                   len(context_params.get("context", "")),
                   len(context_params.get("previous_content", "")),
                   len(context_params.get("conversation_summary", "")))

        # Set up the chain with ALL context parameters, not just the question
        # This ensures conversation history and retrieved documents are included
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
        logger.info("Created retrieval chain with full context parameters")

        # Execute the chain using the enhanced query
        logger.info("Executing chain...")
        response = chain.invoke(enhanced_query)
        logger.info("Chain execution complete")

        # Add metadata about the query for future reference, including optimization info
        metadata = {
            "is_follow_up": is_follow_up,
            "template_used": template_name,
            "query_type": context_info["query_type"],
            "has_previous_content": context_info["has_previous_content"],
            "original_query": question,
            "enhanced_query": enhanced_query
        }

        # Merge optimization metadata
        metadata.update(optimization_metadata)

        return response, document_ids, metadata

    except Exception as e:
        logger.error("Error in query function: %s", str(e))
        return f"An error occurred: {str(e)}", [], {"error": str(e)}

def query_with_feedback_optimization(question: str, template_name: Optional[str] = None,
                                   temperature: Optional[float] = None, conversation: Optional[Any] = None,
                                   conport_client=None, workspace_id: Optional[str] = None) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Convenience function for querying with feedback optimization enabled.

    Args:
        question (str): The question to ask
        template_name (str, optional): The prompt template to use
        temperature (float, optional): Model temperature
        conversation (Conversation, optional): Conversation object containing history
        conport_client: ConPort MCP client for feedback data access
        workspace_id (str, optional): Workspace identifier for ConPort operations

    Returns:
        Tuple[str, List[str], Dict[str, Any]]: The model's response, document IDs, and metadata
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

def get_query_enhancement_suggestions(question: str, conport_client=None, workspace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get suggestions for improving a query based on feedback patterns.

    Args:
        question (str): The question to analyze
        conport_client: ConPort MCP client for feedback data access
        workspace_id (str, optional): Workspace identifier for ConPort operations

    Returns:
        Dictionary containing enhancement suggestions
    """
    try:
        feedback_analyzer = get_feedback_analyzer(conport_client=conport_client, workspace_id=workspace_id)
        query_enhancer = get_query_enhancer(feedback_analyzer=feedback_analyzer)
        return query_enhancer.get_enhancement_suggestions(question)
    except Exception as e:
        logger.error(f"Error getting query enhancement suggestions: {e}")
        return {
            "original_query": question,
            "suggestions": [],
            "confidence": 0.5,
            "error": str(e)
        }

def get_feedback_summary(days_back: int = 7, conport_client=None, workspace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a summary of recent feedback for monitoring purposes.

    Args:
        days_back (int): Number of days to summarize
        conport_client: ConPort MCP client for feedback data access
        workspace_id (str, optional): Workspace identifier for ConPort operations

    Returns:
        Dictionary containing feedback summary
    """
    try:
        feedback_analyzer = get_feedback_analyzer(conport_client=conport_client, workspace_id=workspace_id)
        return feedback_analyzer.get_feedback_summary(days_back=days_back)
    except Exception as e:
        logger.error(f"Error getting feedback summary: {e}")
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
        logger.info("Optimization caches cleared")
    except Exception as e:
        logger.error(f"Error clearing optimization cache: {e}")
