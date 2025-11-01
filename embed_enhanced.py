"""
Enhanced Embedding Module with Multi-Model Support

This module extends the original embed.py to support multiple embedding models
including fine-tuned models for A/B testing and model switching capabilities.
"""

import os
import datetime
import logging
from typing import Optional, Dict, Any, List
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Try to import sentence-transformers for fine-tuned models
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Fine-tuned models will not be supported.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up constants
TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'mistral')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

class EmbeddingModelManager:
    """
    Manages multiple embedding models including original Ollama models
    and fine-tuned sentence-transformer models.
    """

    def __init__(self):
        self.models = {}
        self.current_model = None
        self.model_type = None  # 'ollama' or 'sentence_transformer'

    def register_ollama_model(self, model_name: str, model_id: str = None,
                            base_url: str = None) -> bool:
        """
        Register an Ollama embedding model.

        Args:
            model_name: Name to identify the model
            model_id: Ollama model identifier (defaults to model_name)
            base_url: Ollama base URL

        Returns:
            True if successful, False otherwise
        """
        try:
            model_id = model_id or model_name
            base_url = base_url or OLLAMA_BASE_URL

            embeddings = OllamaEmbeddings(
                model=model_id,
                base_url=base_url
            )

            self.models[model_name] = {
                'type': 'ollama',
                'model': embeddings,
                'config': {
                    'model_id': model_id,
                    'base_url': base_url
                }
            }

            logger.info(f"Registered Ollama model: {model_name} ({model_id})")
            return True

        except Exception as e:
            logger.error(f"Error registering Ollama model {model_name}: {e}")
            return False

    def register_sentence_transformer_model(self, model_name: str,
                                          model_path: str) -> bool:
        """
        Register a sentence-transformer model (including fine-tuned models).

        Args:
            model_name: Name to identify the model
            model_path: Path to the model or HuggingFace model name

        Returns:
            True if successful, False otherwise
        """
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.error("sentence-transformers not available")
                return False

            model = SentenceTransformer(model_path)

            self.models[model_name] = {
                'type': 'sentence_transformer',
                'model': model,
                'config': {
                    'model_path': model_path
                }
            }

            logger.info(f"Registered sentence-transformer model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Error registering sentence-transformer model {model_name}: {e}")
            return False

    def set_active_model(self, model_name: str) -> bool:
        """
        Set the active embedding model.

        Args:
            model_name: Name of the model to activate

        Returns:
            True if successful, False otherwise
        """
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not registered")
                return False

            self.current_model = model_name
            self.model_type = self.models[model_name]['type']

            logger.info(f"Set active model: {model_name} (type: {self.model_type})")
            return True

        except Exception as e:
            logger.error(f"Error setting active model {model_name}: {e}")
            return False

    def get_active_model(self):
        """Get the currently active embedding model."""
        if self.current_model and self.current_model in self.models:
            return self.models[self.current_model]['model']
        return None

    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get information about a model or all models.

        Args:
            model_name: Specific model name, or None for all models

        Returns:
            Dictionary containing model information
        """
        try:
            if model_name:
                if model_name in self.models:
                    info = self.models[model_name].copy()
                    info['is_active'] = (model_name == self.current_model)
                    return {model_name: info}
                else:
                    return {}
            else:
                # Return info for all models
                all_info = {}
                for name, model_data in self.models.items():
                    info = model_data.copy()
                    info['is_active'] = (name == self.current_model)
                    all_info[name] = info
                return all_info

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())

# Global model manager instance
model_manager = EmbeddingModelManager()

# Initialize with default Ollama model
model_manager.register_ollama_model('default', EMBEDDING_MODEL, OLLAMA_BASE_URL)
model_manager.set_active_model('default')

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

def embed(file, collection_name='langchain', model_name: str = None):
    """
    Embed the uploaded file into the vector database with specified collection name.

    Args:
        file: File to embed
        collection_name: Name of the collection
        model_name: Name of the embedding model to use (optional)

    Returns:
        True if successful, False otherwise
    """
    try:
        if file.filename == '' or not allowed_file(file.filename):
            logger.warning("Invalid file: %s", file.filename)
            return False

        # Set model if specified
        if model_name and model_name in model_manager.models:
            model_manager.set_active_model(model_name)

        # Get active embedding model
        embeddings = model_manager.get_active_model()
        if not embeddings:
            logger.error("No active embedding model available")
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

        # Handle different embedding model types
        if model_manager.model_type == 'sentence_transformer':
            # For sentence-transformer models, we need to create a custom embedding wrapper
            embeddings = SentenceTransformerEmbeddings(model_manager.get_active_model())

        logger.info("Using embedding model: %s (type: %s)",
                   model_manager.current_model, model_manager.model_type)

        # Store in ChromaDB with specified collection name
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=collection_name
        )
        logger.info("Embedded documents stored in ChromaDB collection '%s' at %s",
                   collection_name, CHROMA_PERSIST_DIR)

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

class SentenceTransformerEmbeddings:
    """
    Wrapper class to make sentence-transformer models compatible with LangChain.
    """

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            return []

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return []

def get_vector_db(collection_name: str = 'langchain', model_name: str = None):
    """
    Get vector database instance with specified embedding model.

    Args:
        collection_name: Name of the collection
        model_name: Name of the embedding model to use

    Returns:
        Chroma database instance
    """
    try:
        # Set model if specified
        if model_name and model_name in model_manager.models:
            model_manager.set_active_model(model_name)

        # Get active embedding model
        embeddings = model_manager.get_active_model()
        if not embeddings:
            logger.error("No active embedding model available")
            return None

        # Handle different embedding model types
        if model_manager.model_type == 'sentence_transformer':
            embeddings = SentenceTransformerEmbeddings(embeddings)

        return Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=collection_name
        )
    except Exception as e:
        logger.error("Error initializing vector database: %s", e)
        return None

def register_fine_tuned_model(model_name: str, model_path: str) -> bool:
    """
    Register a fine-tuned embedding model.

    Args:
        model_name: Name to identify the model
        model_path: Path to the fine-tuned model

    Returns:
        True if successful, False otherwise
    """
    return model_manager.register_sentence_transformer_model(model_name, model_path)

def switch_embedding_model(model_name: str) -> bool:
    """
    Switch to a different embedding model.

    Args:
        model_name: Name of the model to switch to

    Returns:
        True if successful, False otherwise
    """
    return model_manager.set_active_model(model_name)

def get_available_models() -> Dict[str, Any]:
    """Get information about all available embedding models."""
    return model_manager.get_model_info()

def get_current_model_info() -> Dict[str, Any]:
    """Get information about the currently active model."""
    if model_manager.current_model:
        return model_manager.get_model_info(model_manager.current_model)
    return {}

# Backward compatibility functions
def get_vector_db_original():
    """Original get_vector_db function for backward compatibility."""
    try:
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
    except Exception as e:
        logger.error("Error initializing vector database: %s", e)
        raise