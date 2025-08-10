import os, datetime, re
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import logging
from typing import Dict, List, Optional, Tuple

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

def extract_chapter_info(content: str, filename: str) -> Dict[str, any]:
    """Extract chapter information from markdown content"""
    metadata = {
        'file_type': filename.rsplit('.', 1)[1].lower() if '.' in filename else 'unknown',
        'original_filename': filename,
        'upload_timestamp': datetime.datetime.now().isoformat(),
        'word_count': len(content.split()),
        'character_count': len(content),
        'chapter_title': None,
        'chapter_number': None,
        'book_title': None,
        'has_chapters': False
    }

    # Extract book title from first H1 header
    book_title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if book_title_match:
        metadata['book_title'] = book_title_match.group(1).strip()

    # Check for chapter structure
    chapter_headers = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)
    if chapter_headers:
        metadata['has_chapters'] = True

        # Try to extract chapter number and title from various patterns
        for header in chapter_headers:
            # Pattern: "Chapter 1: Title" or "Chapter 1 - Title"
            chapter_match = re.search(r'chapter\s+(\d+)[\s\-:]+(.+)', header, re.IGNORECASE)
            if chapter_match:
                metadata['chapter_number'] = int(chapter_match.group(1))
                metadata['chapter_title'] = chapter_match.group(2).strip()
                break

            # Pattern: "1. Title" or "1 - Title"
            numbered_match = re.search(r'^(\d+)[\.\-\s]+(.+)', header)
            if numbered_match:
                metadata['chapter_number'] = int(numbered_match.group(1))
                metadata['chapter_title'] = numbered_match.group(2).strip()
                break

    return metadata

def extract_chapters_from_content(content: str, base_metadata: Dict[str, any]) -> List[Tuple[str, Dict[str, any]]]:
    """Split content into chapters and return list of (content, metadata) tuples"""
    chapters = []

    # Split by chapter headers (H1, H2, H3)
    chapter_pattern = r'^(#{1,3}\s+.+)$'
    parts = re.split(chapter_pattern, content, flags=re.MULTILINE)

    if len(parts) <= 1:
        # No chapters found, return entire content as single chapter
        metadata = base_metadata.copy()
        metadata['chapter_title'] = metadata.get('book_title', 'Untitled')
        metadata['chapter_number'] = 1
        return [(content, metadata)]

    current_content = parts[0].strip()  # Content before first chapter
    chapter_num = 1

    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            header = parts[i]
            chapter_content = parts[i + 1]

            # Combine any preceding content with this chapter
            full_content = (current_content + '\n\n' + header + '\n' + chapter_content).strip()

            # Extract chapter info from header
            chapter_metadata = base_metadata.copy()
            chapter_metadata['chapter_number'] = chapter_num

            # Extract title from header
            title_match = re.search(r'^#{1,3}\s+(.+)$', header)
            if title_match:
                title = title_match.group(1).strip()

                # Clean up common chapter patterns
                clean_title = re.sub(r'^chapter\s+\d+[\s\-:]*', '', title, flags=re.IGNORECASE)
                chapter_metadata['chapter_title'] = clean_title.strip() or title

            # Update word and character counts for this chapter
            chapter_metadata['word_count'] = len(full_content.split())
            chapter_metadata['character_count'] = len(full_content)

            chapters.append((full_content, chapter_metadata))
            current_content = ""
            chapter_num += 1

    return chapters if chapters else [(content, base_metadata)]

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
    """Embed the uploaded file into the vector database with enhanced metadata"""
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

        # Combine all document content for metadata extraction
        full_content = '\n'.join([doc.page_content for doc in data])

        # Extract base metadata from the full content
        base_metadata = extract_chapter_info(full_content, file.filename)
        logger.info("Extracted metadata: %s", base_metadata)

        # For markdown files, try to split into chapters
        if file_ext == 'md' and base_metadata.get('has_chapters', False):
            logger.info("Processing markdown file with chapters")
            chapters = extract_chapters_from_content(full_content, base_metadata)
            logger.info("Found %d chapters", len(chapters))

            # Process each chapter separately
            all_chunks = []
            for chapter_idx, (chapter_content, chapter_metadata) in enumerate(chapters):
                # Create a document for this chapter
                from langchain_core.documents import Document
                chapter_doc = Document(page_content=chapter_content, metadata=chapter_metadata)

                # Split chapter into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                chapter_chunks = text_splitter.split_documents([chapter_doc])

                # Add chunk-specific metadata
                for chunk_idx, chunk in enumerate(chapter_chunks):
                    chunk.metadata.update({
                        'chunk_index': chunk_idx,
                        'total_chunks_in_chapter': len(chapter_chunks),
                        'chapter_index': chapter_idx,
                        'total_chapters': len(chapters)
                    })

                all_chunks.extend(chapter_chunks)

            chunks = all_chunks
        else:
            # Standard processing for non-chapter content
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)

            # Add enhanced metadata to each chunk
            for chunk_idx, chunk in enumerate(chunks):
                chunk.metadata.update(base_metadata)
                chunk.metadata.update({
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                })

        logger.info("Split into %d chunks with enhanced metadata", len(chunks))

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