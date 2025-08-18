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

def query(question: str, template_name: Optional[str] = None, temperature: Optional[float] = None,
          conversation: Optional[Any] = None) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Query the vector database with a question, optionally using conversation history.

    This enhanced version uses the ContextManager to dynamically select and format prompts
    based on conversation context.

    Args:
        question (str): The question to ask
        template_name (str, optional): The prompt template to use. Defaults to env var or 'sixthwood'
        temperature (float, optional): Model temperature. Defaults to env var or 1.0
        conversation (Conversation, optional): Conversation object containing history

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

        logger.info("Processing query using model %s (temp=%s, style=%s): %s",
                   LLM_MODEL, temperature, template_name, question)

        # Create embeddings and connect to the vector database
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        logger.info("Using Ollama embeddings with model: %s", EMBEDDING_MODEL)

        try:
            db = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings
            )
            logger.info("Connected to ChromaDB at %s", CHROMA_PERSIST_DIR)
        except Exception as db_error:
            logger.error("Error connecting to ChromaDB: %s", str(db_error))
            return f"Error connecting to the vector database: {str(db_error)}", [], {"error": str(db_error)}

        # Create retriever with appropriate settings
        try:
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVAL_K}
            )
            logger.info("Created retriever with k=%s", RETRIEVAL_K)
        except Exception as retriever_error:
            logger.error("Error creating retriever: %s", str(retriever_error))
            return f"Error setting up document retrieval: {str(retriever_error)}", [], {"error": str(retriever_error)}

        # Get relevant documents
        document_ids = []
        try:
            docs = retriever.get_relevant_documents(question)
            for i, doc in enumerate(docs):
                logger.info("Retrieved document %d: %s...", i+1, doc.page_content[:100])  # Log first 100 chars
                if hasattr(doc, 'metadata') and 'id' in doc.metadata:
                    document_ids.append(doc.metadata['id'])
        except Exception as retrieval_error:
            logger.error("Error retrieving documents: %s", str(retrieval_error))
            # Continue without document retrieval
            docs = []
            logger.warning("Proceeding without document retrieval")

        # Get prompt and context information from context manager
        context_info = context_manager.get_prompt(
            query=question,
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

        # Set up the chain using LangChain Expression Language (LCEL)
        chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Created retrieval chain")

        # Execute the chain
        logger.info("Executing chain...")
        response = chain.invoke(question)
        logger.info("Chain execution complete")

        # Add metadata about the query for future reference
        metadata = {
            "is_follow_up": is_follow_up,
            "template_used": template_name,
            "query_type": context_info["query_type"],
            "has_previous_content": context_info["has_previous_content"]
        }

        return response, document_ids, metadata

    except Exception as e:
        logger.error("Error in query function: %s", str(e))
        return f"An error occurred: {str(e)}", [], {"error": str(e)}
