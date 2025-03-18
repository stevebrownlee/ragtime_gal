import os, datetime
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up constants
TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'mistral')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

def allowed_file(filename):
    """`True` if the file is allowed, `False` otherwise"""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in ['pdf', 'md']

def save_file(file):
    """Save the uploaded file to the temporary folder with original extension preserved"""
    ct = datetime.datetime.now()
    ts = ct.timestamp()

    # Create filename with timestamp and original filename
    filename = f"{ts}_{secure_filename(file.filename)}"
    file_path = os.path.join(TEMP_FOLDER, filename)

    file.save(file_path)
    logger.info("Saved file to %s", file_path)
    return file_path

def embed(file):
    """Embed the uploaded file into the vector database"""
    try:
        if file.filename == '' or not allowed_file(file.filename):
            logger.warning("Invalid file: %s", file.filename)
            return False

        # Extract file extension before saving
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        logger.info("Processing file with extension: %s", file_ext)

        file_path = save_file(file)
        logger.info("Processing file: %s", file_path)

        # Explicitly check file content to determine the correct loader
        with open(file_path, 'rb') as f:
            content_start = f.read(20)  # Read the first few bytes to check

        if content_start.startswith(b'%PDF'):
            logger.info("Content identified as PDF")
            loader = PyPDFLoader(file_path=file_path)
        elif any(content_start.startswith(sig) for sig in [b'#', b'-']) or b'```' in content_start or file_ext == 'md':
            logger.info("Content identified as Markdown/Text")
            loader = TextLoader(file_path=file_path)
        else:
            logger.info("Using file extension to determine type: %s", file_ext)
            if file_ext == 'pdf':
                loader = PyPDFLoader(file_path=file_path)
            elif file_ext == 'md':
                loader = TextLoader(file_path=file_path)
            else:
                logger.error("Unsupported file type and content: %s", file_ext)
                if os.path.exists(file_path):
                    os.remove(file_path)
                return False

        # Load the document
        try:
            data = loader.load()
            logger.info("Loaded %d pages/chunks from file", len(data))
        except Exception as load_error:
            logger.error("Error loading file: %s", str(load_error))
            # If loading fails with one loader, try the other
            if isinstance(loader, PyPDFLoader):
                logger.info("Falling back to TextLoader")
                loader = TextLoader(file_path=file_path)
                data = loader.load()
                logger.info("Successfully loaded with TextLoader: %d chunks", len(data))
            else:
                # Re-raise if we already tried TextLoader or for other errors
                raise

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        logger.info("Split into %d chunks", len(chunks))

        # Initialize local embeddings with Ollama
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        logger.info("Using Ollama embeddings with model: %s", EMBEDDING_MODEL)

        # Store in ChromaDB
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        logger.info("Embedded documents stored in ChromaDB at %s", CHROMA_PERSIST_DIR)

        # Clean up temp file
        os.remove(file_path)
        logger.info("Removed temporary file: %s", file_path)

        return True

    except Exception as e:
        logger.error("Error in embed function: %s", str(e))
        # Try to clean up if file exists
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info("Cleaned up temp file after error: %s", file_path)
            except:
                pass
        return False