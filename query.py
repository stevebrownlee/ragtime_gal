import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from prompts import get_template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables with defaults
LLM_MODEL = os.getenv('LLM_MODEL', 'sixthwood')  # Use the custom sixthwood model by default
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'mistral')  # Keep using mistral for embeddings
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', '4'))

def query(question: str, template_name: Optional[str] = None, temperature: Optional[float] = None,
          conversation: Optional[Any] = None) -> Tuple[str, List[str]]:
    """
    Query the vector database with a question, optionally using conversation history.

    This enhanced version prioritizes previously generated content for follow-up questions
    and directly includes relevant previous content in the prompt.

    Args:
        question (str): The question to ask
        template_name (str, optional): The prompt template to use. Defaults to env var or 'sixthwood'
        temperature (float, optional): Model temperature. Defaults to env var or 1.0
        conversation (Conversation, optional): Conversation object containing history

    Returns:
        Tuple[str, List[str]]: The model's response and a list of document IDs that were referenced
    """
    try:
        # Use arguments if provided, otherwise fall back to env vars
        if template_name is None:
            template_name = os.getenv('PROMPT_TEMPLATE', 'sixthwood')

        if temperature is None:
            temperature = float(os.getenv('LLM_TEMPERATURE', '1.0'))

        # Determine the appropriate template based on conversation context
        has_conversation = conversation is not None and hasattr(conversation, 'get_history') and len(conversation.get_history()) > 0

        # Analyze the query to determine if it's a follow-up question
        is_follow_up = False
        previous_content = ""

        if has_conversation:
            # Check if this is a follow-up question
            is_follow_up = conversation.is_follow_up_question(question)

            if is_follow_up:
                # Get relevant previous content
                previous_content = conversation.get_most_relevant_content(question)
                logger.info("Detected follow-up question. Including relevant previous content (%d chars)",
                           len(previous_content))

                # Select the appropriate follow-up template
                template_name = f"follow_up_{template_name}"
            else:
                # For non-follow-up questions with conversation history, use the previous content template
                template_name = f"{template_name}_with_previous_content"
                previous_content = conversation.get_most_relevant_content(question)
                if previous_content:
                    logger.info("Including potentially relevant previous content (%d chars)", len(previous_content))

        logger.info("Processing query using model %s (temp=%s, template=%s): %s",
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
            return f"Error connecting to the vector database: {str(db_error)}", []

        # Create retriever with appropriate settings
        try:
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVAL_K}
            )
            logger.info("Created retriever with k=%s", RETRIEVAL_K)
        except Exception as retriever_error:
            logger.error("Error creating retriever: %s", str(retriever_error))
            return f"Error setting up document retrieval: {str(retriever_error)}", []

        # Create LLM with the custom model
        llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            system=None,  # Use the model's built-in system prompt
            num_predict=4096,  # Request longer generations
            repeat_penalty=1.1  # Slightly penalize repetition
        )
        logger.info("Using Ollama LLM with model: %s, temperature: %s", LLM_MODEL, temperature)

        # Load the prompt template based on settings
        template_text = get_template(template_name)
        if template_text is None:
            # Fall back to standard template if the specific template doesn't exist
            logger.warning("Template %s not found, falling back to standard template", template_name)
            template_text = get_template("standard")

        logger.info("Using prompt template: %s", template_name)

        prompt = ChatPromptTemplate.from_template(template_text)
        logger.info("Created prompt template")

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

        # Set up the chain using LangChain Expression Language (LCEL)
        if is_follow_up:
            # For follow-up questions, prioritize previous content
            chain = (
                {
                    "context": lambda _: "\n".join(doc.page_content for doc in docs) if docs else "No relevant documents found.",
                    "question": RunnablePassthrough(),
                    "previous_content": lambda _: previous_content
                }
                | prompt
                | llm
                | StrOutputParser()
            )
        elif has_conversation and previous_content:
            # For non-follow-up questions with relevant previous content
            chain = (
                {
                    "context": lambda _: "\n".join(doc.page_content for doc in docs) if docs else "No relevant documents found.",
                    "question": RunnablePassthrough(),
                    "previous_content": lambda _: previous_content
                }
                | prompt
                | llm
                | StrOutputParser()
            )
        else:
            # Standard chain without conversation history
            chain = (
                {
                    "context": lambda _: "\n".join(doc.page_content for doc in docs) if docs else "No relevant documents found.",
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )
        logger.info("Created retrieval chain")

        # Execute the chain
        logger.info("Executing chain...")
        response = chain.invoke(question)
        logger.info("Chain execution complete")

        # Add metadata about the type of query for future reference
        metadata = {
            "is_follow_up": is_follow_up,
            "template_used": template_name,
            "has_previous_content": bool(previous_content)
        }

        return response, document_ids, metadata

    except Exception as e:
        logger.error("Error in query function: %s", str(e))
        return f"An error occurred: {str(e)}", [], {"error": str(e)}