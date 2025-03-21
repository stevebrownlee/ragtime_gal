import os
import logging
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

def query(question, template_name=None, temperature=None):
    """
    Query the vector database with a question.

    Args:
        question (str): The question to ask
        template_name (str, optional): The prompt template to use. Defaults to env var or 'sixthwood'
        temperature (float, optional): Model temperature. Defaults to env var or 1.0

    Returns:
        str: The model's response
    """
    try:
        # Use arguments if provided, otherwise fall back to env vars
        if template_name is None:
            template_name = os.getenv('PROMPT_TEMPLATE', 'sixthwood')

        if temperature is None:
            temperature = float(os.getenv('LLM_TEMPERATURE', '1.0'))

        logger.info("Processing query using model %s (temp=%s, template=%s): %s", LLM_MODEL, temperature, template_name, question)

        # Create embeddings and connect to the vector database
        # IMPORTANT: Use the same model for embeddings as was used during document embedding
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        logger.info("Using Ollama embeddings with model: %s", EMBEDDING_MODEL)

        db = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
        logger.info("Connected to ChromaDB at %s", CHROMA_PERSIST_DIR)

        # Create retriever
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K}
        )
        logger.info("Created retriever with k=%s", RETRIEVAL_K)

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
        logger.info("Using prompt template: %s", template_name)

        prompt = ChatPromptTemplate.from_template(template_text)
        logger.info("Created prompt template")

        docs = retriever.get_relevant_documents(question)
        for i, doc in enumerate(docs):
            logger.info("Retrieved document %d: %s...", i+1, doc.page_content[:100])  # Log first 100 chars

        # Set up the chain using LangChain Expression Language (LCEL)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Created retrieval chain")

        # Execute the chain
        logger.info("Executing chain...")
        response = chain.invoke(question)
        logger.info("Chain execution complete")

        return response

    except Exception as e:
        logger.error("Error in query function: %s", str(e))
        return f"An error occurred: {str(e)}"