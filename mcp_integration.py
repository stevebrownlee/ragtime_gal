"""
MCP Server Integration Module

This module provides the MCPServerManager class for managing the MCP server lifecycle
alongside the Flask web server in a unified process using threading.
"""

import asyncio
import logging
import threading
import time
import re
from typing import Optional, Dict, List, Any
from pathlib import Path
from functools import lru_cache

from mcp.server.fastmcp import FastMCP
from shared_db import SharedDatabaseManager
from metadata_utils import MetadataQueryManager

# Phase 4 dependencies for advanced analysis
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    logging.warning("textstat not available - readability analysis will be disabled")

try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False
    logging.warning("language-tool-python not available - grammar checking will be disabled")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spacy not available - advanced NLP analysis will be disabled")

logger = logging.getLogger(__name__)


class MCPServerManager:
    """
    Manages the MCP server lifecycle in a separate thread with async event loop.
    Handles startup, shutdown, and error recovery for the MCP server.
    """

    def __init__(self, shared_db: SharedDatabaseManager, book_directory: str = "."):
        """
        Initialize the MCP server manager.

        Args:
            shared_db: Shared database manager instance
            book_directory: Directory containing book files (for backward compatibility)
        """
        self.shared_db = shared_db
        self.book_directory = Path(book_directory)
        self.mcp_server: Optional[FastMCP] = None
        self.server_thread: Optional[threading.Thread] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.is_running = False
        self.shutdown_event = threading.Event()

        # Initialize metadata query manager
        self.metadata_manager = MetadataQueryManager()

        # Create the MCP server instance
        self._create_mcp_server()

    def _create_mcp_server(self):
        """Create and configure the MCP server with tools."""
        self.mcp_server = FastMCP("Ragtime Gal MCP Server")

        # Register Phase 3 core MCP tools
        self._register_phase3_tools()

        # Register Phase 4 advanced analysis tools
        self._register_phase4_tools()

        # Register Phase 5 content management tools
        self._register_phase5_tools()

        logger.info("MCP server created with Phase 3 core tools, Phase 4 advanced analysis tools, and Phase 5 content management tools registered")

    def _register_phase3_tools(self):
        """Register Phase 3 core MCP tools for vector-based content search and analysis."""

        # Phase 1 basic tools (maintained for compatibility)
        @self.mcp_server.tool()
        def get_server_status() -> dict:
            """Get the current status of the MCP server and database connection."""
            try:
                # Test database connection
                db_status = "connected" if self.shared_db.test_connection() else "disconnected"

                return {
                    "mcp_server": "running",
                    "database_status": db_status,
                    "book_directory": str(self.book_directory),
                    "tools_registered": len(self.mcp_server._tools) if hasattr(self.mcp_server, '_tools') else 0
                }
            except Exception as e:
                logger.error(f"Error getting server status: {e}")
                return {
                    "mcp_server": "error",
                    "database_status": "error",
                    "error": str(e)
                }

        @self.mcp_server.tool()
        def test_database_connection() -> dict:
            """Test the connection to the shared vector database."""
            try:
                db = self.shared_db.get_database()
                # Try to get collection info
                collection_info = db.get()
                document_count = len(collection_info.get("ids", []))

                return {
                    "status": "success",
                    "document_count": document_count,
                    "database_type": "Chroma",
                    "message": "Database connection successful"
                }
            except Exception as e:
                logger.error(f"Database connection test failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "message": "Database connection failed"
                }

        # Phase 3 Core Tools: Content Search Tools
        @self.mcp_server.tool()
        def search_book_content(query: str, book_title: str = None, chapter_number: int = None,
                              similarity_threshold: float = 0.7, max_results: int = 5) -> dict:
            """
            Perform vector-based content search across book content.

            Args:
                query: Search query text
                book_title: Optional filter by specific book title
                chapter_number: Optional filter by specific chapter number (requires book_title)
                similarity_threshold: Minimum similarity score (0.0-1.0)
                max_results: Maximum number of results to return
            """
            try:
                # Build metadata filters
                filters = {}
                if book_title:
                    filters["book_title"] = book_title
                if chapter_number is not None and book_title:
                    filters["chapter_number"] = chapter_number

                # Perform search using metadata manager
                results = self.metadata_manager.search_content_with_metadata(
                    query=query,
                    filters=filters if filters else None,
                    k=max_results
                )

                # Filter by similarity threshold
                filtered_results = [
                    result for result in results
                    if result.get('similarity_score', 0) >= similarity_threshold
                ]

                return {
                    "status": "success",
                    "query": query,
                    "filters_applied": filters,
                    "total_results": len(filtered_results),
                    "results": filtered_results
                }

            except Exception as e:
                logger.error(f"Error searching book content: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }

        # Phase 3 Core Tools: Chapter Management Tools
        @self.mcp_server.tool()
        def get_chapter_info(book_title: str, chapter_number: int) -> dict:
            """
            Get detailed information about a specific chapter.

            Args:
                book_title: Title of the book
                chapter_number: Chapter number to retrieve
            """
            try:
                # Get chapter content chunks
                chunks = self.metadata_manager.get_chapter_content(book_title, chapter_number)

                if not chunks:
                    return {
                        "status": "not_found",
                        "message": f"Chapter {chapter_number} not found in book '{book_title}'"
                    }

                # Aggregate chapter information
                total_words = sum(chunk['metadata'].get('word_count', 0) for chunk in chunks)
                total_characters = sum(chunk['metadata'].get('character_count', 0) for chunk in chunks)
                chapter_title = chunks[0]['metadata'].get('chapter_title', f'Chapter {chapter_number}')

                return {
                    "status": "success",
                    "book_title": book_title,
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "total_chunks": len(chunks),
                    "total_words": total_words,
                    "total_characters": total_characters,
                    "upload_timestamp": chunks[0]['metadata'].get('upload_timestamp'),
                    "chunks": chunks
                }

            except Exception as e:
                logger.error(f"Error getting chapter info: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }

        @self.mcp_server.tool()
        def list_all_chapters(book_title: str = None, sort_by: str = "chapter_number") -> dict:
            """
            List all chapters with metadata and statistics.

            Args:
                book_title: Optional filter by specific book title
                sort_by: Sort chapters by 'chapter_number', 'word_count', or 'upload_timestamp'
            """
            try:
                if book_title:
                    # Get chapters for specific book
                    chapters = self.metadata_manager.get_chapters_for_book(book_title)
                    books_info = [{"book_title": book_title, "chapters": chapters}]
                else:
                    # Get all books and their chapters
                    all_books = self.metadata_manager.get_all_books()
                    books_info = []
                    for book in all_books:
                        chapters = self.metadata_manager.get_chapters_for_book(book['book_title'])
                        books_info.append({
                            "book_title": book['book_title'],
                            "chapters": chapters
                        })

                # Sort chapters within each book
                for book_info in books_info:
                    if sort_by == "word_count":
                        book_info["chapters"].sort(key=lambda x: x.get('word_count', 0), reverse=True)
                    elif sort_by == "upload_timestamp":
                        book_info["chapters"].sort(key=lambda x: x.get('upload_timestamp', ''), reverse=True)
                    else:  # default to chapter_number
                        book_info["chapters"].sort(key=lambda x: x.get('chapter_number', 0))

                return {
                    "status": "success",
                    "sort_by": sort_by,
                    "total_books": len(books_info),
                    "books": books_info
                }

            except Exception as e:
                logger.error(f"Error listing chapters: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }

        # Phase 3 Core Tools: Character Analysis Tools
        @self.mcp_server.tool()
        def analyze_character_mentions(character_name: str, book_title: str = None,
                                     context_window: int = 200) -> dict:
            """
            Analyze character mentions across book content using vector search.

            Args:
                character_name: Name of the character to analyze
                book_title: Optional filter by specific book title
                context_window: Number of characters to include around each mention
            """
            try:
                # Build search query for character mentions
                search_queries = [
                    character_name,
                    f'"{character_name}"',
                    f"{character_name} said",
                    f"{character_name} was",
                    f"{character_name} did"
                ]

                all_mentions = []

                for query in search_queries:
                    # Build metadata filters
                    filters = {}
                    if book_title:
                        filters["book_title"] = book_title

                    # Search for character mentions
                    results = self.metadata_manager.search_content_with_metadata(
                        query=query,
                        filters=filters if filters else None,
                        k=10  # Get more results for character analysis
                    )

                    for result in results:
                        content = result['content']
                        # Find character mentions in content
                        content_lower = content.lower()
                        char_lower = character_name.lower()

                        start_pos = 0
                        while True:
                            pos = content_lower.find(char_lower, start_pos)
                            if pos == -1:
                                break

                            # Extract context around mention
                            context_start = max(0, pos - context_window // 2)
                            context_end = min(len(content), pos + len(character_name) + context_window // 2)
                            context = content[context_start:context_end]

                            mention = {
                                "book_title": result['metadata'].get('book_title', 'Unknown'),
                                "chapter_number": result['metadata'].get('chapter_number'),
                                "chapter_title": result['metadata'].get('chapter_title'),
                                "position_in_chunk": pos,
                                "context": context,
                                "similarity_score": result.get('similarity_score', 0),
                                "chunk_id": result['metadata'].get('chunk_index', 0)
                            }
                            all_mentions.append(mention)
                            start_pos = pos + 1

                # Remove duplicates and sort by similarity score
                unique_mentions = []
                seen_contexts = set()
                for mention in all_mentions:
                    context_key = (mention['book_title'], mention['chapter_number'], mention['context'][:50])
                    if context_key not in seen_contexts:
                        seen_contexts.add(context_key)
                        unique_mentions.append(mention)

                unique_mentions.sort(key=lambda x: x['similarity_score'], reverse=True)

                # Generate statistics
                books_mentioned = set(m['book_title'] for m in unique_mentions)
                chapters_mentioned = set((m['book_title'], m['chapter_number']) for m in unique_mentions if m['chapter_number'])

                return {
                    "status": "success",
                    "character_name": character_name,
                    "total_mentions": len(unique_mentions),
                    "books_mentioned": list(books_mentioned),
                    "chapters_mentioned": len(chapters_mentioned),
                    "mentions": unique_mentions[:20]  # Limit to top 20 mentions
                }

            except Exception as e:
                logger.error(f"Error analyzing character mentions: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }

        @self.mcp_server.tool()
        def get_book_structure(book_title: str) -> dict:
            """
            Get comprehensive book structure and navigation information.

            Args:
                book_title: Title of the book to analyze
            """
            try:
                # Get book statistics
                stats = self.metadata_manager.get_book_statistics(book_title)

                if not stats:
                    return {
                        "status": "not_found",
                        "message": f"Book '{book_title}' not found"
                    }

                # Get detailed chapter information
                chapters = self.metadata_manager.get_chapters_for_book(book_title)

                return {
                    "status": "success",
                    "book_title": book_title,
                    "statistics": stats,
                    "chapters": chapters,
                    "navigation": {
                        "total_chapters": len(chapters),
                        "chapter_numbers": [ch['chapter_number'] for ch in chapters],
                        "chapter_titles": [ch['chapter_title'] for ch in chapters]
                    }
                }

            except Exception as e:
                logger.error(f"Error getting book structure: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }

        @self.mcp_server.tool()
        def list_all_books() -> dict:
            """List all books in the database with comprehensive metadata."""
            try:
                books = self.metadata_manager.get_all_books()

                return {
                    "status": "success",
                    "total_books": len(books),
                    "books": books
                }

            except Exception as e:
                logger.error(f"Error listing books: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }

    def _register_phase4_tools(self):
        """Register Phase 4 advanced analysis tools for writing statistics, readability, and content quality."""

        # Initialize language tool for grammar checking (lazy loading)
        self._language_tool = None

        # Phase 4.1: Writing Statistics Tools
        @self.mcp_server.tool()
        def get_writing_statistics(book_title: str = None, chapter_number: int = None,
                                 include_detailed_stats: bool = True) -> dict:
            """
            Get comprehensive writing statistics from vector database content.

            Args:
                book_title: Optional filter by specific book title
                chapter_number: Optional filter by specific chapter (requires book_title)
                include_detailed_stats: Include detailed per-chapter breakdown
            """
            try:
                # Build metadata filters
                filters = {}
                if book_title:
                    filters["book_title"] = book_title
                if chapter_number is not None and book_title:
                    filters["chapter_number"] = chapter_number

                # Get all relevant content
                if book_title and chapter_number is not None:
                    # Get specific chapter
                    chunks = self.metadata_manager.get_chapter_content(book_title, chapter_number)
                    scope = f"Chapter {chapter_number} of '{book_title}'"
                elif book_title:
                    # Get all chapters for book
                    all_books = self.metadata_manager.get_all_books()
                    book_data = next((b for b in all_books if b['book_title'] == book_title), None)
                    if not book_data:
                        return {"status": "not_found", "message": f"Book '{book_title}' not found"}

                    chunks = []
                    chapters = self.metadata_manager.get_chapters_for_book(book_title)
                    for chapter in chapters:
                        chapter_chunks = self.metadata_manager.get_chapter_content(book_title, chapter['chapter_number'])
                        chunks.extend(chapter_chunks)
                    scope = f"Book '{book_title}'"
                else:
                    # Get all content
                    db = self.shared_db.get_database()
                    all_data = db.get()
                    chunks = []
                    for i, content in enumerate(all_data.get("documents", [])):
                        metadata = all_data.get("metadatas", [{}])[i] if i < len(all_data.get("metadatas", [])) else {}
                        chunks.append({"content": content, "metadata": metadata})
                    scope = "All books"

                if not chunks:
                    return {"status": "no_content", "message": f"No content found for {scope}"}

                # Calculate comprehensive statistics
                stats = self._calculate_writing_statistics(chunks, include_detailed_stats)
                stats["scope"] = scope
                stats["status"] = "success"

                return stats

            except Exception as e:
                logger.error(f"Error getting writing statistics: {e}")
                return {"status": "error", "error": str(e)}

        @self.mcp_server.tool()
        def analyze_readability(book_title: str = None, chapter_number: int = None,
                              include_all_metrics: bool = True) -> dict:
            """
            Analyze readability metrics using textstat library.

            Args:
                book_title: Optional filter by specific book title
                chapter_number: Optional filter by specific chapter (requires book_title)
                include_all_metrics: Include all available readability metrics
            """
            try:
                if not TEXTSTAT_AVAILABLE:
                    return {
                        "status": "unavailable",
                        "message": "textstat library not available - install with 'pip install textstat'"
                    }

                # Build metadata filters and get content
                filters = {}
                if book_title:
                    filters["book_title"] = book_title
                if chapter_number is not None and book_title:
                    filters["chapter_number"] = chapter_number

                # Get content chunks
                if book_title and chapter_number is not None:
                    chunks = self.metadata_manager.get_chapter_content(book_title, chapter_number)
                    scope = f"Chapter {chapter_number} of '{book_title}'"
                elif book_title:
                    chunks = []
                    chapters = self.metadata_manager.get_chapters_for_book(book_title)
                    for chapter in chapters:
                        chapter_chunks = self.metadata_manager.get_chapter_content(book_title, chapter['chapter_number'])
                        chunks.extend(chapter_chunks)
                    scope = f"Book '{book_title}'"
                else:
                    db = self.shared_db.get_database()
                    all_data = db.get()
                    chunks = []
                    for i, content in enumerate(all_data.get("documents", [])):
                        metadata = all_data.get("metadatas", [{}])[i] if i < len(all_data.get("metadatas", [])) else {}
                        chunks.append({"content": content, "metadata": metadata})
                    scope = "All books"

                if not chunks:
                    return {"status": "no_content", "message": f"No content found for {scope}"}

                # Combine all content for analysis
                full_text = " ".join(chunk["content"] for chunk in chunks)

                # Calculate readability metrics
                readability_stats = self._calculate_readability_metrics(full_text, include_all_metrics)
                readability_stats["scope"] = scope
                readability_stats["status"] = "success"
                readability_stats["total_characters"] = len(full_text)
                readability_stats["total_chunks_analyzed"] = len(chunks)

                return readability_stats

            except Exception as e:
                logger.error(f"Error analyzing readability: {e}")
                return {"status": "error", "error": str(e)}

        @self.mcp_server.tool()
        def check_grammar_and_style(book_title: str, chapter_number: int = None,
                                   max_issues: int = 20, issue_types: List[str] = None) -> dict:
            """
            Check grammar and style issues using language-tool-python.

            Args:
                book_title: Book title to analyze
                chapter_number: Optional specific chapter number
                max_issues: Maximum number of issues to return
                issue_types: Optional filter for specific issue types
            """
            try:
                if not LANGUAGE_TOOL_AVAILABLE:
                    return {
                        "status": "unavailable",
                        "message": "language-tool-python not available - install with 'pip install language-tool-python'"
                    }

                # Get content
                if chapter_number is not None:
                    chunks = self.metadata_manager.get_chapter_content(book_title, chapter_number)
                    scope = f"Chapter {chapter_number} of '{book_title}'"
                else:
                    chunks = []
                    chapters = self.metadata_manager.get_chapters_for_book(book_title)
                    for chapter in chapters:
                        chapter_chunks = self.metadata_manager.get_chapter_content(book_title, chapter['chapter_number'])
                        chunks.extend(chapter_chunks)
                    scope = f"Book '{book_title}'"

                if not chunks:
                    return {"status": "no_content", "message": f"No content found for {scope}"}

                # Initialize language tool if needed
                if self._language_tool is None:
                    self._language_tool = language_tool_python.LanguageTool('en-US')

                # Analyze grammar and style
                issues = []
                total_text_length = 0

                for chunk in chunks[:10]:  # Limit to first 10 chunks for performance
                    content = chunk["content"]
                    total_text_length += len(content)

                    # Check grammar
                    matches = self._language_tool.check(content)

                    for match in matches:
                        if len(issues) >= max_issues:
                            break

                        # Filter by issue types if specified
                        if issue_types and match.category not in issue_types:
                            continue

                        issue = {
                            "chapter_number": chunk["metadata"].get("chapter_number"),
                            "chapter_title": chunk["metadata"].get("chapter_title"),
                            "issue_type": match.category,
                            "rule_id": match.ruleId,
                            "message": match.message,
                            "context": match.context,
                            "offset": match.offset,
                            "length": match.errorLength,
                            "suggestions": match.replacements[:3] if match.replacements else []
                        }
                        issues.append(issue)

                    if len(issues) >= max_issues:
                        break

                # Categorize issues
                issue_categories = {}
                for issue in issues:
                    category = issue["issue_type"]
                    if category not in issue_categories:
                        issue_categories[category] = 0
                    issue_categories[category] += 1

                return {
                    "status": "success",
                    "scope": scope,
                    "total_issues": len(issues),
                    "issues_by_category": issue_categories,
                    "text_analyzed_length": total_text_length,
                    "chunks_analyzed": len(chunks[:10]),
                    "issues": issues
                }

            except Exception as e:
                logger.error(f"Error checking grammar and style: {e}")
                return {"status": "error", "error": str(e)}

        @self.mcp_server.tool()
        def analyze_writing_patterns(book_title: str, pattern_type: str = "sentence_structure") -> dict:
            """
            Analyze writing patterns and style consistency.

            Args:
                book_title: Book title to analyze
                pattern_type: Type of pattern analysis ('sentence_structure', 'paragraph_length', 'dialogue_ratio')
            """
            try:
                # Get all chapters for the book
                chapters = self.metadata_manager.get_chapters_for_book(book_title)
                if not chapters:
                    return {"status": "not_found", "message": f"Book '{book_title}' not found"}

                all_chunks = []
                for chapter in chapters:
                    chapter_chunks = self.metadata_manager.get_chapter_content(book_title, chapter['chapter_number'])
                    all_chunks.extend(chapter_chunks)

                if not all_chunks:
                    return {"status": "no_content", "message": f"No content found for book '{book_title}'"}

                # Analyze patterns based on type
                if pattern_type == "sentence_structure":
                    analysis = self._analyze_sentence_patterns(all_chunks)
                elif pattern_type == "paragraph_length":
                    analysis = self._analyze_paragraph_patterns(all_chunks)
                elif pattern_type == "dialogue_ratio":
                    analysis = self._analyze_dialogue_patterns(all_chunks)
                else:
                    return {"status": "invalid_pattern", "message": f"Unknown pattern type: {pattern_type}"}

                analysis["status"] = "success"
                analysis["book_title"] = book_title
                analysis["pattern_type"] = pattern_type
                analysis["chapters_analyzed"] = len(chapters)

                return analysis

            except Exception as e:
                logger.error(f"Error analyzing writing patterns: {e}")
                return {"status": "error", "error": str(e)}
    def _calculate_writing_statistics(self, chunks: List[Dict], include_detailed_stats: bool = True) -> Dict[str, Any]:
        """Calculate comprehensive writing statistics from content chunks."""
        total_words = 0
        total_characters = 0
        total_sentences = 0
        total_paragraphs = 0
        chapter_stats = {}

        for chunk in chunks:
            content = chunk["content"]
            metadata = chunk["metadata"]

            # Basic counts
            words = len(content.split())
            characters = len(content)
            sentences = len(re.split(r'[.!?]+', content)) - 1  # -1 for empty string at end
            paragraphs = len([p for p in content.split('\n\n') if p.strip()])

            total_words += words
            total_characters += characters
            total_sentences += sentences
            total_paragraphs += paragraphs

            # Chapter-level stats if detailed
            if include_detailed_stats:
                chapter_num = metadata.get('chapter_number')
                chapter_title = metadata.get('chapter_title', f'Chapter {chapter_num}')
                book_title = metadata.get('book_title', 'Unknown')

                key = f"{book_title} - {chapter_title}"
                if key not in chapter_stats:
                    chapter_stats[key] = {
                        "words": 0, "characters": 0, "sentences": 0, "paragraphs": 0,
                        "chapter_number": chapter_num, "book_title": book_title
                    }

                chapter_stats[key]["words"] += words
                chapter_stats[key]["characters"] += characters
                chapter_stats[key]["sentences"] += sentences
                chapter_stats[key]["paragraphs"] += paragraphs

        # Calculate averages
        avg_words_per_sentence = total_words / max(total_sentences, 1)
        avg_sentences_per_paragraph = total_sentences / max(total_paragraphs, 1)
        avg_words_per_paragraph = total_words / max(total_paragraphs, 1)

        # Estimate reading time (average 200 words per minute)
        reading_time_minutes = total_words / 200

        stats = {
            "total_words": total_words,
            "total_characters": total_characters,
            "total_sentences": total_sentences,
            "total_paragraphs": total_paragraphs,
            "total_chunks": len(chunks),
            "averages": {
                "words_per_sentence": round(avg_words_per_sentence, 2),
                "sentences_per_paragraph": round(avg_sentences_per_paragraph, 2),
                "words_per_paragraph": round(avg_words_per_paragraph, 2)
            },
            "estimated_reading_time_minutes": round(reading_time_minutes, 1)
        }

        if include_detailed_stats and chapter_stats:
            stats["chapter_breakdown"] = chapter_stats

        return stats

    def _calculate_readability_metrics(self, text: str, include_all_metrics: bool = True) -> Dict[str, Any]:
        """Calculate readability metrics using textstat library."""
        if not TEXTSTAT_AVAILABLE:
            return {"error": "textstat library not available"}

        metrics = {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "coleman_liau_index": textstat.coleman_liau_index(text)
        }

        if include_all_metrics:
            metrics.update({
                "gunning_fog": textstat.gunning_fog(text),
                "smog_index": textstat.smog_index(text),
                "reading_time_seconds": textstat.reading_time(text, ms_per_char=14.69),
                "lexicon_count": textstat.lexicon_count(text),
                "sentence_count": textstat.sentence_count(text),
                "avg_sentence_length": textstat.avg_sentence_length(text),
                "avg_syllables_per_word": textstat.avg_syllables_per_word(text),
                "difficult_words": textstat.difficult_words(text)
            })

        # Add interpretation
        flesch_score = metrics["flesch_reading_ease"]
        if flesch_score >= 90:
            reading_level = "Very Easy (5th grade)"
        elif flesch_score >= 80:
            reading_level = "Easy (6th grade)"
        elif flesch_score >= 70:
            reading_level = "Fairly Easy (7th grade)"
        elif flesch_score >= 60:
            reading_level = "Standard (8th-9th grade)"
        elif flesch_score >= 50:
            reading_level = "Fairly Difficult (10th-12th grade)"
        elif flesch_score >= 30:
            reading_level = "Difficult (College level)"
        else:
            reading_level = "Very Difficult (Graduate level)"

        metrics["reading_level_interpretation"] = reading_level

        return metrics

    def _analyze_sentence_patterns(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze sentence structure patterns."""
        sentence_lengths = []
        sentence_types = {"simple": 0, "compound": 0, "complex": 0}

        for chunk in chunks:
            content = chunk["content"]
            sentences = re.split(r'[.!?]+', content)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                words = len(sentence.split())
                sentence_lengths.append(words)

                # Simple heuristics for sentence types
                if ' and ' in sentence or ' or ' in sentence or ' but ' in sentence:
                    sentence_types["compound"] += 1
                elif ' because ' in sentence or ' although ' in sentence or ' while ' in sentence:
                    sentence_types["complex"] += 1
                else:
                    sentence_types["simple"] += 1

        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            min_length = min(sentence_lengths)
            max_length = max(sentence_lengths)
        else:
            avg_length = min_length = max_length = 0

        return {
            "total_sentences": len(sentence_lengths),
            "average_sentence_length": round(avg_length, 2),
            "min_sentence_length": min_length,
            "max_sentence_length": max_length,
            "sentence_types": sentence_types,
            "sentence_length_distribution": {
                "short (1-10 words)": len([l for l in sentence_lengths if l <= 10]),
                "medium (11-20 words)": len([l for l in sentence_lengths if 11 <= l <= 20]),
                "long (21+ words)": len([l for l in sentence_lengths if l > 20])
            }
        }

    def _analyze_paragraph_patterns(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze paragraph length patterns."""
        paragraph_lengths = []

        for chunk in chunks:
            content = chunk["content"]
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

            for paragraph in paragraphs:
                words = len(paragraph.split())
                paragraph_lengths.append(words)

        if paragraph_lengths:
            avg_length = sum(paragraph_lengths) / len(paragraph_lengths)
            min_length = min(paragraph_lengths)
            max_length = max(paragraph_lengths)
        else:
            avg_length = min_length = max_length = 0

        return {
            "total_paragraphs": len(paragraph_lengths),
            "average_paragraph_length": round(avg_length, 2),
            "min_paragraph_length": min_length,
            "max_paragraph_length": max_length,
            "paragraph_length_distribution": {
                "short (1-50 words)": len([l for l in paragraph_lengths if l <= 50]),
                "medium (51-150 words)": len([l for l in paragraph_lengths if 51 <= l <= 150]),
                "long (151+ words)": len([l for l in paragraph_lengths if l > 150])
            }
        }

    def _analyze_dialogue_patterns(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze dialogue vs narrative patterns."""
        total_words = 0
        dialogue_words = 0
        dialogue_blocks = 0

        for chunk in chunks:
            content = chunk["content"]
            words = content.split()
            total_words += len(words)

            # Simple heuristic: text within quotes is dialogue
            dialogue_matches = re.findall(r'"([^"]*)"', content)
            for match in dialogue_matches:
                dialogue_words += len(match.split())
                dialogue_blocks += 1

            # Also check for single quotes
            dialogue_matches = re.findall(r"'([^']*)'", content)
            for match in dialogue_matches:
                dialogue_words += len(match.split())
                dialogue_blocks += 1

        dialogue_ratio = (dialogue_words / max(total_words, 1)) * 100
        narrative_ratio = 100 - dialogue_ratio

        return {
            "total_words": total_words,
            "dialogue_words": dialogue_words,
            "narrative_words": total_words - dialogue_words,
            "dialogue_blocks": dialogue_blocks,
            "dialogue_percentage": round(dialogue_ratio, 2),
            "narrative_percentage": round(narrative_ratio, 2),
            "avg_words_per_dialogue_block": round(dialogue_words / max(dialogue_blocks, 1), 2)
        }


        logger.info("Phase 4 advanced analysis tools registered successfully")

    def _register_phase5_tools(self):
        """Register Phase 5 content management tools for adding, updating, and deleting book content."""

        # Import additional dependencies for content management
        import uuid
        from datetime import datetime

        # Phase 5.1: Content Addition Tools
        @self.mcp_server.tool()
        def add_chapter_content(book_title: str, chapter_number: int, chapter_title: str,
                              content: str, overwrite_existing: bool = False) -> dict:
            """
            Add new chapter content to the vector database.

            Args:
                book_title: Title of the book
                chapter_number: Chapter number (must be positive integer)
                chapter_title: Title of the chapter
                content: Chapter content text
                overwrite_existing: Whether to overwrite if chapter already exists
            """
            try:
                # Validate inputs
                if not book_title.strip():
                    return {"status": "error", "error": "Book title cannot be empty"}
                if not chapter_title.strip():
                    return {"status": "error", "error": "Chapter title cannot be empty"}
                if not content.strip():
                    return {"status": "error", "error": "Chapter content cannot be empty"}
                if chapter_number <= 0:
                    return {"status": "error", "error": "Chapter number must be positive"}

                # Check if chapter already exists
                existing_chunks = self.metadata_manager.get_chapter_content(book_title, chapter_number)
                if existing_chunks and not overwrite_existing:
                    return {
                        "status": "conflict",
                        "error": f"Chapter {chapter_number} already exists in '{book_title}'. Use overwrite_existing=True to replace it.",
                        "existing_chunks": len(existing_chunks)
                    }

                # If overwriting, delete existing content first
                if existing_chunks and overwrite_existing:
                    delete_result = self._delete_chapter_content(book_title, chapter_number)
                    if delete_result["status"] != "success":
                        return {
                            "status": "error",
                            "error": f"Failed to delete existing chapter: {delete_result.get('error', 'Unknown error')}"
                        }

                # Chunk the content using the same logic as embed.py
                chunks = self._chunk_content(content, book_title, chapter_number, chapter_title)

                # Add chunks to database
                db = self.shared_db.get_database()

                # Prepare data for insertion
                documents = [chunk["content"] for chunk in chunks]
                metadatas = [chunk["metadata"] for chunk in chunks]
                ids = [chunk["id"] for chunk in chunks]

                # Add to database
                self.shared_db.add_documents(
                    texts=documents,
                    metadatas=metadatas,
                    ids=ids
                )

                logger.info(f"Added chapter {chapter_number} of '{book_title}' with {len(chunks)} chunks")

                return {
                    "status": "success",
                    "message": f"Successfully added chapter {chapter_number} of '{book_title}'",
                    "book_title": book_title,
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "chunks_added": len(chunks),
                    "total_words": sum(chunk["metadata"]["word_count"] for chunk in chunks),
                    "total_characters": sum(chunk["metadata"]["character_count"] for chunk in chunks)
                }

            except Exception as e:
                logger.error(f"Error adding chapter content: {e}")
                return {"status": "error", "error": str(e)}

        # Phase 5.2: Content Update Tools
        @self.mcp_server.tool()
        def update_chapter_content(book_title: str, chapter_number: int,
                                 new_content: str = None, new_chapter_title: str = None) -> dict:
            """
            Update existing chapter content or metadata.

            Args:
                book_title: Title of the book
                chapter_number: Chapter number to update
                new_content: New content text (if provided, replaces all content)
                new_chapter_title: New chapter title (updates metadata only)
            """
            try:
                # Validate inputs
                if not book_title.strip():
                    return {"status": "error", "error": "Book title cannot be empty"}
                if chapter_number <= 0:
                    return {"status": "error", "error": "Chapter number must be positive"}
                if not new_content and not new_chapter_title:
                    return {"status": "error", "error": "Must provide either new_content or new_chapter_title"}

                # Check if chapter exists
                existing_chunks = self.metadata_manager.get_chapter_content(book_title, chapter_number)
                if not existing_chunks:
                    return {
                        "status": "not_found",
                        "error": f"Chapter {chapter_number} not found in book '{book_title}'"
                    }

                current_chapter_title = existing_chunks[0]["metadata"].get("chapter_title", f"Chapter {chapter_number}")

                if new_content:
                    # Full content replacement
                    chapter_title = new_chapter_title if new_chapter_title else current_chapter_title

                    # Delete existing content
                    delete_result = self._delete_chapter_content(book_title, chapter_number)
                    if delete_result["status"] != "success":
                        return {"status": "error", "error": f"Failed to delete existing content: {delete_result.get('error')}"}

                    # Add new content
                    chunks = self._chunk_content(new_content.strip(), book_title, chapter_number, chapter_title)

                    # Add to database
                    db = self.shared_db.get_database()
                    documents = [chunk["content"] for chunk in chunks]
                    metadatas = [chunk["metadata"] for chunk in chunks]
                    ids = [chunk["id"] for chunk in chunks]

                    self.shared_db.add_documents(texts=documents, metadatas=metadatas, ids=ids)

                    return {
                        "status": "success",
                        "message": f"Successfully updated chapter {chapter_number} content",
                        "book_title": book_title,
                        "chapter_number": chapter_number,
                        "chapter_title": chapter_title,
                        "chunks_updated": len(chunks),
                        "operation": "content_replacement"
                    }

                elif new_chapter_title:
                    # Title-only update
                    db = self.shared_db.get_database()

                    # Update metadata for all chunks in this chapter
                    updated_count = 0
                    for chunk in existing_chunks:
                        chunk_id = chunk["metadata"].get("chunk_id")
                        if chunk_id:
                            # Update the metadata
                            updated_metadata = chunk["metadata"].copy()
                            updated_metadata["chapter_title"] = new_chapter_title.strip()
                            updated_metadata["last_modified"] = datetime.now().isoformat()

                            # Update in database
                            db.update(ids=[chunk_id], metadatas=[updated_metadata])
                            updated_count += 1

                    return {
                        "status": "success",
                        "message": f"Successfully updated chapter title",
                        "book_title": book_title,
                        "chapter_number": chapter_number,
                        "old_title": current_chapter_title,
                        "new_title": new_chapter_title.strip(),
                        "chunks_updated": updated_count,
                        "operation": "title_update"
                    }

            except Exception as e:
                logger.error(f"Error updating chapter content: {e}")
                return {"status": "error", "error": str(e)}

        # Phase 5.3: Content Deletion Tools
        @self.mcp_server.tool()
        def delete_chapter(book_title: str, chapter_number: int, confirm_deletion: bool = False) -> dict:
            """
            Delete a chapter and all its content from the vector database.

            Args:
                book_title: Title of the book
                chapter_number: Chapter number to delete
                confirm_deletion: Must be True to actually perform deletion (safety check)
            """
            try:
                if not confirm_deletion:
                    return {
                        "status": "confirmation_required",
                        "message": "Deletion requires confirmation. Set confirm_deletion=True to proceed.",
                        "warning": "This operation cannot be undone!"
                    }

                # Validate inputs
                if not book_title.strip():
                    return {"status": "error", "error": "Book title cannot be empty"}
                if chapter_number <= 0:
                    return {"status": "error", "error": "Chapter number must be positive"}

                # Delete the chapter content
                result = self._delete_chapter_content(book_title, chapter_number)

                if result["status"] == "success":
                    logger.info(f"Deleted chapter {chapter_number} from '{book_title}'")

                return result

            except Exception as e:
                logger.error(f"Error deleting chapter: {e}")
                return {"status": "error", "error": str(e)}

        @self.mcp_server.tool()
        def delete_book(book_title: str, confirm_deletion: bool = False) -> dict:
            """
            Delete an entire book and all its chapters from the vector database.

            Args:
                book_title: Title of the book to delete
                confirm_deletion: Must be True to actually perform deletion (safety check)
            """
            try:
                if not confirm_deletion:
                    return {
                        "status": "confirmation_required",
                        "message": "Book deletion requires confirmation. Set confirm_deletion=True to proceed.",
                        "warning": "This operation will delete ALL chapters and cannot be undone!"
                    }

                # Validate inputs
                if not book_title.strip():
                    return {"status": "error", "error": "Book title cannot be empty"}

                # Get all chapters for the book
                chapters = self.metadata_manager.get_chapters_for_book(book_title)
                if not chapters:
                    return {
                        "status": "not_found",
                        "error": f"Book '{book_title}' not found"
                    }

                # Delete all chapters
                deleted_chapters = 0
                total_chunks_deleted = 0

                for chapter in chapters:
                    result = self._delete_chapter_content(book_title, chapter["chapter_number"])
                    if result["status"] == "success":
                        deleted_chapters += 1
                        total_chunks_deleted += result.get("chunks_deleted", 0)

                logger.info(f"Deleted book '{book_title}' with {deleted_chapters} chapters and {total_chunks_deleted} chunks")

                return {
                    "status": "success",
                    "message": f"Successfully deleted book '{book_title}'",
                    "book_title": book_title,
                    "chapters_deleted": deleted_chapters,
                    "total_chunks_deleted": total_chunks_deleted
                }

            except Exception as e:
                logger.error(f"Error deleting book: {e}")
                return {"status": "error", "error": str(e)}

        # Phase 5.4: Content Organization Tools
        @self.mcp_server.tool()
        def reorder_chapters(book_title: str, chapter_mapping: dict, confirm_reorder: bool = False) -> dict:
            """
            Reorder chapters by updating their chapter numbers.

            Args:
                book_title: Title of the book
                chapter_mapping: Dict mapping old chapter numbers to new chapter numbers
                confirm_reorder: Must be True to actually perform reordering (safety check)
            """
            try:
                if not confirm_reorder:
                    return {
                        "status": "confirmation_required",
                        "message": "Chapter reordering requires confirmation. Set confirm_reorder=True to proceed.",
                        "warning": "This operation will change chapter numbers and cannot be easily undone!"
                    }

                # Validate inputs
                if not book_title.strip():
                    return {"status": "error", "error": "Book title cannot be empty"}
                if not chapter_mapping:
                    return {"status": "error", "error": "Chapter mapping cannot be empty"}

                # Validate chapter mapping
                for old_num, new_num in chapter_mapping.items():
                    if not isinstance(old_num, int) or not isinstance(new_num, int):
                        return {"status": "error", "error": "Chapter numbers must be integers"}
                    if old_num <= 0 or new_num <= 0:
                        return {"status": "error", "error": "Chapter numbers must be positive"}

                # Check if all old chapters exist
                existing_chapters = {ch["chapter_number"] for ch in self.metadata_manager.get_chapters_for_book(book_title)}
                missing_chapters = set(chapter_mapping.keys()) - existing_chapters
                if missing_chapters:
                    return {
                        "status": "error",
                        "error": f"Chapters not found: {sorted(missing_chapters)}"
                    }

                # Check for conflicts in new numbers
                new_numbers = list(chapter_mapping.values())
                if len(new_numbers) != len(set(new_numbers)):
                    return {"status": "error", "error": "New chapter numbers must be unique"}

                # Perform the reordering
                db = self.shared_db.get_database()
                updated_chunks = 0

                for old_chapter_num, new_chapter_num in chapter_mapping.items():
                    chunks = self.metadata_manager.get_chapter_content(book_title, old_chapter_num)

                    for chunk in chunks:
                        chunk_id = chunk["metadata"].get("chunk_id")
                        if chunk_id:
                            updated_metadata = chunk["metadata"].copy()
                            updated_metadata["chapter_number"] = new_chapter_num
                            updated_metadata["last_modified"] = datetime.now().isoformat()

                            db.update(ids=[chunk_id], metadatas=[updated_metadata])
                            updated_chunks += 1

                logger.info(f"Reordered {len(chapter_mapping)} chapters in '{book_title}', updated {updated_chunks} chunks")

                return {
                    "status": "success",
                    "message": f"Successfully reordered chapters in '{book_title}'",
                    "book_title": book_title,
                    "chapters_reordered": len(chapter_mapping),
                    "chunks_updated": updated_chunks,
                    "mapping": chapter_mapping
                }

            except Exception as e:
                logger.error(f"Error reordering chapters: {e}")
                return {"status": "error", "error": str(e)}

        logger.info("Phase 5 content management tools registered successfully")

    def _chunk_content(self, content: str, book_title: str, chapter_number: int, chapter_title: str) -> List[Dict]:
        """
        Chunk content using the same logic as embed.py for consistency.

        Args:
            content: Text content to chunk
            book_title: Title of the book
            chapter_number: Chapter number
            chapter_title: Chapter title

        Returns:
            List of chunk dictionaries with content, metadata, and IDs
        """
        import uuid
        from datetime import datetime

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            # Fallback to simple chunking if langchain is not available
            return self._simple_chunk_content(content, book_title, chapter_number, chapter_title)

        # Use same chunking parameters as embed.py
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=7500,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )

        # Split the content
        chunks = text_splitter.split_text(content)

        # Create chunk objects with metadata
        chunk_objects = []
        timestamp = datetime.now().isoformat()

        for i, chunk_content in enumerate(chunks):
            chunk_id = f"{book_title}_{chapter_number}_{i}_{uuid.uuid4().hex[:8]}"

            metadata = {
                'file_type': 'md',
                'original_filename': f"{book_title}_chapter_{chapter_number}.md",
                'upload_timestamp': timestamp,
                'word_count': len(chunk_content.split()),
                'character_count': len(chunk_content),
                'chapter_title': chapter_title,
                'chapter_number': chapter_number,
                'book_title': book_title,
                'has_chapters': True,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_id': chunk_id,
                'last_modified': timestamp
            }

            chunk_objects.append({
                'id': chunk_id,
                'content': chunk_content,
                'metadata': metadata
            })

        return chunk_objects

    def _simple_chunk_content(self, content: str, book_title: str, chapter_number: int, chapter_title: str) -> List[Dict]:
        """
        Simple fallback chunking method when langchain is not available.

        Args:
            content: Text content to chunk
            book_title: Title of the book
            chapter_number: Chapter number
            chapter_title: Chapter title

        Returns:
            List of chunk dictionaries with content, metadata, and IDs
        """
        import uuid
        from datetime import datetime

        # Simple chunking by character count
        chunk_size = 7500
        chunk_overlap = 100

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size

            # If this isn't the last chunk, try to break at a sentence or paragraph
            if end < len(content):
                # Look for sentence breaks within the overlap region
                search_start = max(start, end - chunk_overlap)
                sentence_break = content.rfind('.', search_start, end)
                paragraph_break = content.rfind('\n\n', search_start, end)

                # Use the latest break point found
                break_point = max(sentence_break, paragraph_break)
                if break_point > search_start:
                    end = break_point + 1

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append(chunk_content)

            # Move start position with overlap
            start = max(start + 1, end - chunk_overlap)

        # Create chunk objects with metadata
        chunk_objects = []
        timestamp = datetime.now().isoformat()

        for i, chunk_content in enumerate(chunks):
            chunk_id = f"{book_title}_{chapter_number}_{i}_{uuid.uuid4().hex[:8]}"

            metadata = {
                'file_type': 'md',
                'original_filename': f"{book_title}_chapter_{chapter_number}.md",
                'upload_timestamp': timestamp,
                'word_count': len(chunk_content.split()),
                'character_count': len(chunk_content),
                'chapter_title': chapter_title,
                'chapter_number': chapter_number,
                'book_title': book_title,
                'has_chapters': True,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_id': chunk_id,
                'last_modified': timestamp
            }

            chunk_objects.append({
                'id': chunk_id,
                'content': chunk_content,
                'metadata': metadata
            })

        return chunk_objects

    def _delete_chapter_content(self, book_title: str, chapter_number: int) -> dict:
        """
        Helper method to delete all chunks for a specific chapter.

        Args:
            book_title: Title of the book
            chapter_number: Chapter number to delete

        Returns:
            Dict with status and details
        """
        try:
            # Get all chunks for this chapter
            chunks = self.metadata_manager.get_chapter_content(book_title, chapter_number)

            if not chunks:
                return {
                    "status": "not_found",
                    "error": f"Chapter {chapter_number} not found in book '{book_title}'"
                }

            # Extract chunk IDs
            chunk_ids = []
            for chunk in chunks:
                chunk_id = chunk["metadata"].get("chunk_id")
                if chunk_id:
                    chunk_ids.append(chunk_id)

            if not chunk_ids:
                return {
                    "status": "error",
                    "error": "No valid chunk IDs found for deletion"
                }

            # Delete from database
            db = self.shared_db.get_database()
            db.delete(ids=chunk_ids)

            return {
                "status": "success",
                "message": f"Successfully deleted chapter {chapter_number} from '{book_title}'",
                "chunks_deleted": len(chunk_ids),
                "chapter_number": chapter_number,
                "book_title": book_title
            }

        except Exception as e:
            logger.error(f"Error deleting chapter content: {e}")
            return {"status": "error", "error": str(e)}

    def start(self) -> bool:
        """
        Start the MCP server in a background thread.

        Returns:
            bool: True if server started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("MCP server is already running")
            return True

        try:
            # Create and start the server thread
            self.server_thread = threading.Thread(
                target=self._run_server_thread,
                name="MCPServerThread",
                daemon=True
            )
            self.server_thread.start()

            # Wait a moment for the server to start
            time.sleep(1)

            if self.is_running:
                logger.info("MCP server started successfully")
                return True
            else:
                logger.error("MCP server failed to start")
                return False

        except Exception as e:
            logger.error(f"Error starting MCP server: {e}")
            return False

    def _run_server_thread(self):
        """Run the MCP server in its own thread with async event loop."""
        try:
            # Create new event loop for this thread
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)

            self.is_running = True
            logger.info("MCP server thread started")

            # Run the server until shutdown is requested
            self.event_loop.run_until_complete(self._run_server())

        except Exception as e:
            logger.error(f"Error in MCP server thread: {e}")
        finally:
            self.is_running = False
            if self.event_loop:
                self.event_loop.close()
            logger.info("MCP server thread stopped")

    async def _run_server(self):
        """Run the MCP server async loop."""
        try:
            # For now, we'll run a simple keep-alive loop
            # In a full implementation, this would run the actual MCP server
            while not self.shutdown_event.is_set():
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("MCP server async loop cancelled")
        except Exception as e:
            logger.error(f"Error in MCP server async loop: {e}")

    def stop(self) -> bool:
        """
        Stop the MCP server gracefully.

        Returns:
            bool: True if server stopped successfully, False otherwise
        """
        if not self.is_running:
            logger.info("MCP server is not running")
            return True

        try:
            logger.info("Stopping MCP server...")

            # Signal shutdown
            self.shutdown_event.set()

            # Cancel any running tasks in the event loop
            if self.event_loop and not self.event_loop.is_closed():
                # Schedule the cancellation in the event loop
                asyncio.run_coroutine_threadsafe(self._cancel_tasks(), self.event_loop)

            # Wait for the server thread to finish
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=5)

                if self.server_thread.is_alive():
                    logger.warning("MCP server thread did not stop gracefully")
                    return False

            self.is_running = False
            logger.info("MCP server stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Error stopping MCP server: {e}")
            return False

    async def _cancel_tasks(self):
        """Cancel all running tasks in the event loop."""
        try:
            tasks = [task for task in asyncio.all_tasks(self.event_loop) if not task.done()]
            if tasks:
                logger.info(f"Cancelling {len(tasks)} running tasks")
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error cancelling tasks: {e}")

    def is_healthy(self) -> bool:
        """
        Check if the MCP server is healthy and responsive.

        Returns:
            bool: True if server is healthy, False otherwise
        """
        return (
            self.is_running and
            self.server_thread is not None and
            self.server_thread.is_alive() and
            self.event_loop is not None and
            not self.event_loop.is_closed()
        )

    def get_status(self) -> dict:
        """
        Get detailed status information about the MCP server.

        Returns:
            dict: Status information
        """
        return {
            "is_running": self.is_running,
            "is_healthy": self.is_healthy(),
            "thread_alive": self.server_thread.is_alive() if self.server_thread else False,
            "event_loop_closed": self.event_loop.is_closed() if self.event_loop else True,
            "book_directory": str(self.book_directory),
            "tools_count": len(self.mcp_server._tools) if hasattr(self.mcp_server, '_tools') else 0
        }