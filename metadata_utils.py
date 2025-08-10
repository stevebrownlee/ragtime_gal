import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up constants
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'mistral')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')


class MetadataQueryManager:
    """Manager class for querying and managing document metadata in the vector database"""

    def __init__(self, persist_directory: str = None, embedding_model: str = None, ollama_base_url: str = None):
        """Initialize the metadata query manager"""
        self.persist_directory = persist_directory or CHROMA_PERSIST_DIR
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.ollama_base_url = ollama_base_url or OLLAMA_BASE_URL

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.ollama_base_url
        )

        # Initialize database connection
        self._db = None
        logger.info("MetadataQueryManager initialized with persist_directory: %s", self.persist_directory)

    @property
    def db(self):
        """Lazy initialization of database connection"""
        if self._db is None:
            try:
                self._db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info("Connected to ChromaDB at %s", self.persist_directory)
            except Exception as e:
                logger.error("Failed to connect to ChromaDB: %s", str(e))
                raise
        return self._db

    def get_all_books(self) -> List[Dict[str, Any]]:
        """Get a list of all unique books in the database"""
        try:
            # Get all documents to extract unique book information
            all_docs = self.db.get()

            books = {}
            for i, metadata in enumerate(all_docs['metadatas']):
                book_title = metadata.get('book_title')
                if book_title and book_title not in books:
                    books[book_title] = {
                        'book_title': book_title,
                        'file_type': metadata.get('file_type', 'unknown'),
                        'original_filename': metadata.get('original_filename', 'unknown'),
                        'upload_timestamp': metadata.get('upload_timestamp'),
                        'has_chapters': metadata.get('has_chapters', False),
                        'total_chapters': 0,
                        'total_chunks': 0,
                        'total_words': 0,
                        'total_characters': 0
                    }

                # Aggregate statistics
                if book_title in books:
                    books[book_title]['total_chunks'] += 1
                    books[book_title]['total_words'] += metadata.get('word_count', 0)
                    books[book_title]['total_characters'] += metadata.get('character_count', 0)

                    # Track unique chapters
                    chapter_num = metadata.get('chapter_number')
                    if chapter_num:
                        books[book_title]['total_chapters'] = max(
                            books[book_title]['total_chapters'],
                            chapter_num
                        )

            return list(books.values())

        except Exception as e:
            logger.error("Error getting all books: %s", str(e))
            return []

    def get_chapters_for_book(self, book_title: str) -> List[Dict[str, Any]]:
        """Get all chapters for a specific book"""
        try:
            # Query for documents with the specified book title
            results = self.db.get(
                where={"book_title": book_title}
            )

            chapters = {}
            for i, metadata in enumerate(results['metadatas']):
                chapter_num = metadata.get('chapter_number')
                if chapter_num is not None:
                    if chapter_num not in chapters:
                        chapters[chapter_num] = {
                            'chapter_number': chapter_num,
                            'chapter_title': metadata.get('chapter_title', f'Chapter {chapter_num}'),
                            'book_title': book_title,
                            'word_count': 0,
                            'character_count': 0,
                            'chunk_count': 0,
                            'upload_timestamp': metadata.get('upload_timestamp')
                        }

                    # Aggregate chapter statistics
                    chapters[chapter_num]['chunk_count'] += 1
                    chapters[chapter_num]['word_count'] += metadata.get('word_count', 0)
                    chapters[chapter_num]['character_count'] += metadata.get('character_count', 0)

            # Return sorted by chapter number
            return sorted(chapters.values(), key=lambda x: x['chapter_number'])

        except Exception as e:
            logger.error("Error getting chapters for book '%s': %s", book_title, str(e))
            return []

    def search_by_metadata(self, filters: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents by metadata filters"""
        try:
            results = self.db.get(
                where=filters,
                limit=limit
            )

            documents = []
            for i in range(len(results['ids'])):
                doc = {
                    'id': results['ids'][i],
                    'content': results['documents'][i] if results['documents'] else None,
                    'metadata': results['metadatas'][i] if results['metadatas'] else {}
                }
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error("Error searching by metadata %s: %s", filters, str(e))
            return []

    def get_chapter_content(self, book_title: str, chapter_number: int) -> List[Dict[str, Any]]:
        """Get all content chunks for a specific chapter"""
        try:
            filters = {
                "book_title": book_title,
                "chapter_number": chapter_number
            }

            results = self.db.get(where=filters)

            chunks = []
            for i in range(len(results['ids'])):
                chunk = {
                    'id': results['ids'][i],
                    'content': results['documents'][i] if results['documents'] else '',
                    'metadata': results['metadatas'][i] if results['metadatas'] else {},
                    'chunk_index': results['metadatas'][i].get('chunk_index', 0) if results['metadatas'] else 0
                }
                chunks.append(chunk)

            # Sort by chunk index
            chunks.sort(key=lambda x: x['chunk_index'])
            return chunks

        except Exception as e:
            logger.error("Error getting chapter content for book '%s', chapter %d: %s",
                        book_title, chapter_number, str(e))
            return []

    def search_content_with_metadata(self, query: str, filters: Dict[str, Any] = None,
                                   k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search with metadata filtering"""
        try:
            # Perform similarity search
            if filters:
                results = self.db.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filters
                )
            else:
                results = self.db.similarity_search_with_score(
                    query=query,
                    k=k
                )

            search_results = []
            for doc, score in results:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': score
                }
                search_results.append(result)

            return search_results

        except Exception as e:
            logger.error("Error searching content with query '%s' and filters %s: %s",
                        query, filters, str(e))
            return []

    def get_book_statistics(self, book_title: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a book"""
        try:
            results = self.db.get(
                where={"book_title": book_title}
            )

            if not results['metadatas']:
                return {}

            stats = {
                'book_title': book_title,
                'total_chunks': len(results['metadatas']),
                'total_words': 0,
                'total_characters': 0,
                'chapters': set(),
                'file_type': None,
                'upload_timestamp': None,
                'has_chapters': False
            }

            for metadata in results['metadatas']:
                stats['total_words'] += metadata.get('word_count', 0)
                stats['total_characters'] += metadata.get('character_count', 0)

                if metadata.get('chapter_number'):
                    stats['chapters'].add(metadata['chapter_number'])

                if not stats['file_type']:
                    stats['file_type'] = metadata.get('file_type')

                if not stats['upload_timestamp']:
                    stats['upload_timestamp'] = metadata.get('upload_timestamp')

                if metadata.get('has_chapters'):
                    stats['has_chapters'] = True

            stats['total_chapters'] = len(stats['chapters'])
            stats['chapters'] = sorted(list(stats['chapters']))

            # Calculate averages
            if stats['total_chunks'] > 0:
                stats['avg_words_per_chunk'] = stats['total_words'] / stats['total_chunks']
                stats['avg_characters_per_chunk'] = stats['total_characters'] / stats['total_chunks']

            if stats['total_chapters'] > 0:
                stats['avg_words_per_chapter'] = stats['total_words'] / stats['total_chapters']
                stats['avg_chunks_per_chapter'] = stats['total_chunks'] / stats['total_chapters']

            return stats

        except Exception as e:
            logger.error("Error getting book statistics for '%s': %s", book_title, str(e))
            return {}

    def validate_metadata_consistency(self) -> Dict[str, Any]:
        """Validate and report on metadata consistency across all documents"""
        try:
            all_docs = self.db.get()

            validation_report = {
                'total_documents': len(all_docs['metadatas']),
                'missing_fields': {},
                'inconsistent_books': [],
                'orphaned_chapters': [],
                'field_coverage': {}
            }

            required_fields = [
                'file_type', 'original_filename', 'upload_timestamp',
                'word_count', 'character_count'
            ]

            book_info = {}

            for i, metadata in enumerate(all_docs['metadatas']):
                doc_id = all_docs['ids'][i]

                # Check for missing required fields
                for field in required_fields:
                    if field not in metadata or metadata[field] is None:
                        if field not in validation_report['missing_fields']:
                            validation_report['missing_fields'][field] = []
                        validation_report['missing_fields'][field].append(doc_id)

                # Track field coverage
                for field in metadata:
                    if field not in validation_report['field_coverage']:
                        validation_report['field_coverage'][field] = 0
                    validation_report['field_coverage'][field] += 1

                # Check book consistency
                book_title = metadata.get('book_title')
                if book_title:
                    if book_title not in book_info:
                        book_info[book_title] = {
                            'file_type': metadata.get('file_type'),
                            'has_chapters': metadata.get('has_chapters', False),
                            'chapters': set()
                        }

                    # Check for inconsistencies
                    if book_info[book_title]['file_type'] != metadata.get('file_type'):
                        if book_title not in validation_report['inconsistent_books']:
                            validation_report['inconsistent_books'].append(book_title)

                    # Track chapters
                    chapter_num = metadata.get('chapter_number')
                    if chapter_num:
                        book_info[book_title]['chapters'].add(chapter_num)

            # Check for orphaned chapters (chapters without book titles)
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get('chapter_number') and not metadata.get('book_title'):
                    validation_report['orphaned_chapters'].append(all_docs['ids'][i])

            # Calculate field coverage percentages
            total_docs = validation_report['total_documents']
            for field, count in validation_report['field_coverage'].items():
                validation_report['field_coverage'][field] = {
                    'count': count,
                    'percentage': (count / total_docs * 100) if total_docs > 0 else 0
                }

            return validation_report

        except Exception as e:
            logger.error("Error validating metadata consistency: %s", str(e))
            return {'error': str(e)}