# MCP Tools Documentation
Generated on: 2025-08-10T09:29:08.896451

## Overview
This document describes all available MCP (Model Context Protocol) tools for the Ragtime Gal book writing assistant.

## Content Search Tools

### search_book_content

**Description:** Search for content within books using vector similarity

**Parameters:**
- `query` (required): Search query text
- `max_results` (optional): Maximum number of results to return
- `book_filter` (optional): Filter results by specific book
- `chapter_filter` (optional): Filter results by specific chapter
- `similarity_threshold` (optional): Minimum similarity score (0.0-1.0)

**Returns:** List of matching content chunks with metadata and similarity scores

**Example:**
```json
{
  "query": "character development",
  "max_results": 5,
  "similarity_threshold": 0.7
}
```

## Chapter Management Tools

### get_chapter_info

**Description:** Get detailed information about a specific chapter

**Parameters:**
- `book_title` (required): Title of the book
- `chapter_title` (required): Title of the chapter

**Returns:** Chapter information including word count, content summary, and metadata

### list_all_chapters

**Description:** List all chapters across all books with optional filtering and sorting

**Parameters:**
- `book_filter` (optional): Filter by specific book
- `sort_by` (optional): Sort by: chapter_number, word_count, or title

**Returns:** List of chapters with metadata

## Analysis Tools

### get_writing_statistics

**Description:** Get comprehensive writing statistics for books or chapters

**Parameters:**
- `book_title` (optional): Analyze specific book (optional)
- `chapter_title` (optional): Analyze specific chapter (optional)

**Returns:** Writing statistics including word count, reading time, and complexity metrics

### analyze_readability

**Description:** Analyze text readability using various metrics

**Parameters:**
- `book_title` (optional): Analyze specific book
- `chapter_title` (optional): Analyze specific chapter

**Returns:** Readability scores and reading level assessment

## Content Management Tools

### add_chapter_content

**Description:** Add new chapter content to the vector database

**Parameters:**
- `book_title` (required): Title of the book
- `chapter_title` (required): Title of the chapter
- `content` (required): Chapter content in markdown format
- `chapter_number` (optional): Chapter number (auto-assigned if not provided)

**Returns:** Success status and metadata about the added content

### update_chapter_content

**Description:** Update existing chapter content

**Parameters:**
- `book_title` (required): Title of the book
- `chapter_title` (required): Title of the chapter
- `new_content` (optional): New content (for full replacement)
- `new_title` (optional): New chapter title (for title-only update)

**Returns:** Success status and update details
