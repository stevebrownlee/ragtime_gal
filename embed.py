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
    logger.info(f"Saved file to {file_path}")
    return file_path

def embed(file):
    try:
        if file.filename == '' or not allowed_file(file.filename):
            logger.warning(f"Invalid file: {file.filename}")
            return False

        # Extract file extension before saving
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        logger.info(f"Processing file with extension: {file_ext}")

        file_path = save_file(file)
        logger.info(f"Processing file: {file_path}")

        # Explicitly check file content to determine the correct loader
        with open(file_path, 'rb') as f:
            content_start = f.read(20)  # Read the first few bytes to check

        if content_start.startswith(b'%PDF'):
            logger.info(f"Content identified as PDF")
            loader = PyPDFLoader(file_path=file_path)
        elif any(content_start.startswith(sig) for sig in [b'#', b'-']) or b'```' in content_start or file_ext == 'md':
            logger.info(f"Content identified as Markdown/Text")
            loader = TextLoader(file_path=file_path)
        else:
            logger.info(f"Using file extension to determine type: {file_ext}")
            if file_ext == 'pdf':
                loader = PyPDFLoader(file_path=file_path)
            elif file_ext == 'md':
                loader = TextLoader(file_path=file_path)
            else:
                logger.error(f"Unsupported file type and content: {file_ext}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return False

        # Load the document
        try:
            data = loader.load()
            logger.info(f"Loaded {len(data)} pages/chunks from file")
        except Exception as load_error:
            logger.error(f"Error loading file: {str(load_error)}")
            # If loading fails with one loader, try the other
            if isinstance(loader, PyPDFLoader):
                logger.info("Falling back to TextLoader")
                loader = TextLoader(file_path=file_path)
                data = loader.load()
                logger.info(f"Successfully loaded with TextLoader: {len(data)} chunks")
            else:
                # Re-raise if we already tried TextLoader or for other errors
                raise

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        logger.info(f"Split into {len(chunks)} chunks")

        # Initialize local embeddings with Ollama
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        logger.info(f"Using Ollama embeddings with model: {EMBEDDING_MODEL}")

        # Store in ChromaDB
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        logger.info(f"Embedded documents stored in ChromaDB at {CHROMA_PERSIST_DIR}")

        # Clean up temp file
        os.remove(file_path)
        logger.info(f"Removed temporary file: {file_path}")

        return True

    except Exception as e:
        logger.error(f"Error in embed function: {str(e)}")
        # Try to clean up if file exists
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temp file after error: {file_path}")
            except:
                pass
        return False