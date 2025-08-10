# Ragtime Gal User Guide
Generated on: 2025-08-10T09:29:08.897440

## Overview

Ragtime Gal is a comprehensive book writing assistant that combines:
- **Web Interface**: Upload and manage documents through a browser
- **VSCode Integration**: Access powerful writing tools directly in your editor
- **Vector Database**: Intelligent content search and analysis
- **AI-Powered Analysis**: Writing statistics, readability analysis, and more

## Getting Started

### Web Interface

1. **Access the web interface** at `http://localhost:5000`
2. **Upload your documents** using the upload form
   - Supported formats: Markdown (.md), PDF (.pdf)
   - Files are automatically processed and indexed
3. **Query your content** using the search interface
   - Enter natural language queries
   - Results show similar content with relevance scores

### VSCode Integration

1. **Install RooCode extension** in VSCode
2. **Configure MCP server** (see Configuration Guide)
3. **Access MCP tools** through the command palette
   - Search: `Ragtime Gal: Search Content`
   - Analysis: `Ragtime Gal: Analyze Writing`
   - Management: `Ragtime Gal: Manage Chapters`

## Features

### Content Search
- **Semantic Search**: Find content by meaning, not just keywords
- **Chapter Filtering**: Search within specific chapters or books
- **Similarity Scoring**: Results ranked by relevance
- **Context Preservation**: See surrounding content for better understanding

### Writing Analysis
- **Statistics**: Word count, character count, reading time
- **Readability**: Flesch-Kincaid scores and reading level
- **Style Analysis**: Sentence structure and writing patterns
- **Character Analysis**: Track character mentions and development

### Content Management
- **Add Chapters**: Create new content directly from VSCode
- **Update Content**: Modify existing chapters safely
- **Organize Structure**: Reorder chapters and manage book structure
- **Version Control**: Track changes and maintain content history

## Best Practices

### Document Organization
- Use clear, descriptive chapter titles
- Structure content with proper markdown headers
- Keep chapters at reasonable lengths (5,000-10,000 words)
- Use consistent naming conventions

### Effective Searching
- Use specific, descriptive queries
- Try different phrasings if initial results aren't helpful
- Use metadata filters to narrow results
- Review similarity scores to gauge relevance

### Content Management
- Make incremental changes rather than large rewrites
- Use descriptive commit messages when updating content
- Regularly backup your vector database
- Monitor system performance with large document collections
