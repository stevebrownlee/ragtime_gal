import os
import logging
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables with defaults
LLM_MODEL = os.getenv('LLM_MODEL', 'mistral')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'mistral')
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', '4'))

def query(question):
    try:
        logger.info(f"Processing query: {question}")

        # Create embeddings and connect to the vector database
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        logger.info(f"Using Ollama embeddings with model: {EMBEDDING_MODEL}")

        db = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
        logger.info(f"Connected to ChromaDB at {CHROMA_PERSIST_DIR}")

        # Create retriever
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K}
        )
        logger.info(f"Created retriever with k={RETRIEVAL_K}")

        # Create LLM
        llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2  # Lower temperature for more factual responses
        )
        logger.info(f"Using Ollama LLM with model: {LLM_MODEL}")

        # Create prompt template for better answers
        template = """
        Answer the question based only on the following context. If the answer is not in the context,
        say "I don't have enough information to answer this question." Don't make up information.

        Context:
        {context}

        Question: {question}

        Answer:
        """

        prompt = ChatPromptTemplate.from_template(template)
        logger.info("Created prompt template")

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
        logger.error(f"Error in query function: {str(e)}")
        return f"An error occurred: {str(e)}"