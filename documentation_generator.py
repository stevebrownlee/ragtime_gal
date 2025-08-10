#!/usr/bin/env python3
"""
Phase 6: Documentation and Configuration Generator
Generates comprehensive documentation for MCP tools, API endpoints, and system configuration.
"""

import json
import os
import inspect
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DocumentationGenerator:
    """Generate comprehensive documentation for the RAG+MCP system."""

    def __init__(self, output_dir: str = "docs"):
        """Initialize documentation generator."""
        self.output_dir = output_dir
        self.ensure_output_directory()

    def ensure_output_directory(self):
        """Ensure output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_mcp_tools_documentation(self, mcp_manager) -> str:
        """Generate documentation for all MCP tools."""
        doc_content = []
        doc_content.append("# MCP Tools Documentation")
        doc_content.append(f"Generated on: {datetime.now().isoformat()}")
        doc_content.append("")
        doc_content.append("## Overview")
        doc_content.append("This document describes all available MCP (Model Context Protocol) tools for the Ragtime Gal book writing assistant.")
        doc_content.append("")

        # Get tool information
        try:
            # This would need to be adapted based on actual MCP manager interface
            tools_info = self._extract_mcp_tools_info(mcp_manager)

            for category, tools in tools_info.items():
                doc_content.append(f"## {category}")
                doc_content.append("")

                for tool in tools:
                    doc_content.append(f"### {tool['name']}")
                    doc_content.append("")
                    doc_content.append(f"**Description:** {tool['description']}")
                    doc_content.append("")

                    if tool.get('parameters'):
                        doc_content.append("**Parameters:**")
                        for param in tool['parameters']:
                            required = " (required)" if param.get('required') else " (optional)"
                            doc_content.append(f"- `{param['name']}`{required}: {param['description']}")
                        doc_content.append("")

                    if tool.get('returns'):
                        doc_content.append(f"**Returns:** {tool['returns']}")
                        doc_content.append("")

                    if tool.get('example'):
                        doc_content.append("**Example:**")
                        doc_content.append("```json")
                        doc_content.append(json.dumps(tool['example'], indent=2))
                        doc_content.append("```")
                        doc_content.append("")

        except Exception as e:
            logger.error(f"Error generating MCP tools documentation: {e}")
            doc_content.append("Error: Could not extract MCP tools information.")

        content = "\n".join(doc_content)

        # Write to file
        doc_file = os.path.join(self.output_dir, "mcp_tools.md")
        with open(doc_file, 'w') as f:
            f.write(content)

        return doc_file

    def _extract_mcp_tools_info(self, mcp_manager) -> Dict[str, List[Dict[str, Any]]]:
        """Extract MCP tools information from manager."""
        # This is a template - would need to be adapted based on actual MCP manager
        tools_info = {
            "Content Search Tools": [
                {
                    "name": "search_book_content",
                    "description": "Search for content within books using vector similarity",
                    "parameters": [
                        {"name": "query", "required": True, "description": "Search query text"},
                        {"name": "max_results", "required": False, "description": "Maximum number of results to return"},
                        {"name": "book_filter", "required": False, "description": "Filter results by specific book"},
                        {"name": "chapter_filter", "required": False, "description": "Filter results by specific chapter"},
                        {"name": "similarity_threshold", "required": False, "description": "Minimum similarity score (0.0-1.0)"}
                    ],
                    "returns": "List of matching content chunks with metadata and similarity scores",
                    "example": {
                        "query": "character development",
                        "max_results": 5,
                        "similarity_threshold": 0.7
                    }
                }
            ],
            "Chapter Management Tools": [
                {
                    "name": "get_chapter_info",
                    "description": "Get detailed information about a specific chapter",
                    "parameters": [
                        {"name": "book_title", "required": True, "description": "Title of the book"},
                        {"name": "chapter_title", "required": True, "description": "Title of the chapter"}
                    ],
                    "returns": "Chapter information including word count, content summary, and metadata"
                },
                {
                    "name": "list_all_chapters",
                    "description": "List all chapters across all books with optional filtering and sorting",
                    "parameters": [
                        {"name": "book_filter", "required": False, "description": "Filter by specific book"},
                        {"name": "sort_by", "required": False, "description": "Sort by: chapter_number, word_count, or title"}
                    ],
                    "returns": "List of chapters with metadata"
                }
            ],
            "Analysis Tools": [
                {
                    "name": "get_writing_statistics",
                    "description": "Get comprehensive writing statistics for books or chapters",
                    "parameters": [
                        {"name": "book_title", "required": False, "description": "Analyze specific book (optional)"},
                        {"name": "chapter_title", "required": False, "description": "Analyze specific chapter (optional)"}
                    ],
                    "returns": "Writing statistics including word count, reading time, and complexity metrics"
                },
                {
                    "name": "analyze_readability",
                    "description": "Analyze text readability using various metrics",
                    "parameters": [
                        {"name": "book_title", "required": False, "description": "Analyze specific book"},
                        {"name": "chapter_title", "required": False, "description": "Analyze specific chapter"}
                    ],
                    "returns": "Readability scores and reading level assessment"
                }
            ],
            "Content Management Tools": [
                {
                    "name": "add_chapter_content",
                    "description": "Add new chapter content to the vector database",
                    "parameters": [
                        {"name": "book_title", "required": True, "description": "Title of the book"},
                        {"name": "chapter_title", "required": True, "description": "Title of the chapter"},
                        {"name": "content", "required": True, "description": "Chapter content in markdown format"},
                        {"name": "chapter_number", "required": False, "description": "Chapter number (auto-assigned if not provided)"}
                    ],
                    "returns": "Success status and metadata about the added content"
                },
                {
                    "name": "update_chapter_content",
                    "description": "Update existing chapter content",
                    "parameters": [
                        {"name": "book_title", "required": True, "description": "Title of the book"},
                        {"name": "chapter_title", "required": True, "description": "Title of the chapter"},
                        {"name": "new_content", "required": False, "description": "New content (for full replacement)"},
                        {"name": "new_title", "required": False, "description": "New chapter title (for title-only update)"}
                    ],
                    "returns": "Success status and update details"
                }
            ]
        }

        return tools_info

    def generate_api_documentation(self, flask_app) -> str:
        """Generate documentation for Flask API endpoints."""
        doc_content = []
        doc_content.append("# Flask API Documentation")
        doc_content.append(f"Generated on: {datetime.now().isoformat()}")
        doc_content.append("")
        doc_content.append("## Overview")
        doc_content.append("This document describes all available REST API endpoints for the Ragtime Gal web interface.")
        doc_content.append("")

        # Extract routes from Flask app
        try:
            routes_info = self._extract_flask_routes(flask_app)

            for route in routes_info:
                doc_content.append(f"### {route['methods']} {route['rule']}")
                doc_content.append("")
                doc_content.append(f"**Description:** {route['description']}")
                doc_content.append("")

                if route.get('parameters'):
                    doc_content.append("**Parameters:**")
                    for param in route['parameters']:
                        doc_content.append(f"- `{param['name']}` ({param['type']}): {param['description']}")
                    doc_content.append("")

                if route.get('response'):
                    doc_content.append(f"**Response:** {route['response']}")
                    doc_content.append("")

                if route.get('example'):
                    doc_content.append("**Example:**")
                    doc_content.append("```bash")
                    doc_content.append(route['example'])
                    doc_content.append("```")
                    doc_content.append("")

        except Exception as e:
            logger.error(f"Error generating API documentation: {e}")
            doc_content.append("Error: Could not extract Flask routes information.")

        content = "\n".join(doc_content)

        # Write to file
        doc_file = os.path.join(self.output_dir, "api_endpoints.md")
        with open(doc_file, 'w') as f:
            f.write(content)

        return doc_file

    def _extract_flask_routes(self, flask_app) -> List[Dict[str, Any]]:
        """Extract Flask routes information."""
        # Template routes information
        routes_info = [
            {
                "rule": "/",
                "methods": "GET",
                "description": "Main web interface for document upload and querying",
                "response": "HTML page with upload form and query interface"
            },
            {
                "rule": "/upload",
                "methods": "POST",
                "description": "Upload documents (PDF or Markdown) to the vector database",
                "parameters": [
                    {"name": "file", "type": "file", "description": "Document file to upload"},
                    {"name": "book_title", "type": "string", "description": "Optional book title override"}
                ],
                "response": "JSON with upload status and document metadata",
                "example": "curl -X POST -F 'file=@chapter1.md' http://localhost:5000/upload"
            },
            {
                "rule": "/query",
                "methods": "POST",
                "description": "Query the vector database for similar content",
                "parameters": [
                    {"name": "query", "type": "string", "description": "Search query text"},
                    {"name": "max_results", "type": "integer", "description": "Maximum number of results (default: 5)"}
                ],
                "response": "JSON with search results and similarity scores",
                "example": "curl -X POST -H 'Content-Type: application/json' -d '{\"query\":\"character development\"}' http://localhost:5000/query"
            },
            {
                "rule": "/purge",
                "methods": "POST",
                "description": "Clear all documents from the vector database",
                "response": "JSON with purge status",
                "example": "curl -X POST http://localhost:5000/purge"
            },
            {
                "rule": "/health",
                "methods": "GET",
                "description": "Health check endpoint for monitoring",
                "response": "JSON with system health status",
                "example": "curl http://localhost:5000/health"
            }
        ]

        return routes_info

    def generate_configuration_guide(self) -> str:
        """Generate configuration and deployment guide."""
        doc_content = []
        doc_content.append("# Configuration and Deployment Guide")
        doc_content.append(f"Generated on: {datetime.now().isoformat()}")
        doc_content.append("")

        doc_content.append("## Environment Variables")
        doc_content.append("")
        doc_content.append("Create a `.env` file in the project root with the following variables:")
        doc_content.append("")
        doc_content.append("```bash")
        doc_content.append("# Flask Configuration")
        doc_content.append("FLASK_ENV=development")
        doc_content.append("FLASK_DEBUG=True")
        doc_content.append("SECRET_KEY=your-secret-key-here")
        doc_content.append("")
        doc_content.append("# Database Configuration")
        doc_content.append("CHROMA_DB_PATH=./chroma_db")
        doc_content.append("EMBEDDING_MODEL=mistral")
        doc_content.append("")
        doc_content.append("# MCP Server Configuration")
        doc_content.append("MCP_SERVER_NAME=ragtime-gal-mcp")
        doc_content.append("MCP_SERVER_VERSION=1.0.0")
        doc_content.append("BOOK_DIRECTORY=./books")
        doc_content.append("")
        doc_content.append("# Performance Configuration")
        doc_content.append("CHUNK_SIZE=7500")
        doc_content.append("CHUNK_OVERLAP=100")
        doc_content.append("MAX_UPLOAD_SIZE=16777216  # 16MB")
        doc_content.append("")
        doc_content.append("# Logging Configuration")
        doc_content.append("LOG_LEVEL=INFO")
        doc_content.append("LOG_FILE=logs/ragtime_gal.log")
        doc_content.append("```")
        doc_content.append("")

        doc_content.append("## Installation")
        doc_content.append("")
        doc_content.append("### Prerequisites")
        doc_content.append("- Python 3.8 or higher")
        doc_content.append("- Ollama installed and running")
        doc_content.append("- Git")
        doc_content.append("")
        doc_content.append("### Step-by-step Installation")
        doc_content.append("")
        doc_content.append("1. **Clone the repository:**")
        doc_content.append("```bash")
        doc_content.append("git clone <repository-url>")
        doc_content.append("cd ragtime-gal")
        doc_content.append("```")
        doc_content.append("")
        doc_content.append("2. **Install dependencies:**")
        doc_content.append("```bash")
        doc_content.append("pip install pipenv")
        doc_content.append("pipenv install")
        doc_content.append("pipenv shell")
        doc_content.append("```")
        doc_content.append("")
        doc_content.append("3. **Set up environment variables:**")
        doc_content.append("```bash")
        doc_content.append("cp .env.template .env")
        doc_content.append("# Edit .env with your configuration")
        doc_content.append("```")
        doc_content.append("")
        doc_content.append("4. **Start Ollama (if not already running):**")
        doc_content.append("```bash")
        doc_content.append("ollama serve")
        doc_content.append("ollama pull mistral  # Download embedding model")
        doc_content.append("```")
        doc_content.append("")
        doc_content.append("5. **Run the application:**")
        doc_content.append("```bash")
        doc_content.append("python app.py")
        doc_content.append("```")
        doc_content.append("")

        doc_content.append("## VSCode Integration")
        doc_content.append("")
        doc_content.append("### MCP Server Setup")
        doc_content.append("")
        doc_content.append("1. **Install RooCode extension** in VSCode")
        doc_content.append("")
        doc_content.append("2. **Configure MCP server** in VSCode settings:")
        doc_content.append("```json")
        doc_content.append("{")
        doc_content.append('  "roocode.mcpServers": {')
        doc_content.append('    "ragtime-gal": {')
        doc_content.append('      "command": "python",')
        doc_content.append('      "args": ["mcp_server.py"],')
        doc_content.append('      "cwd": "/path/to/ragtime-gal"')
        doc_content.append('    }')
        doc_content.append('  }')
        doc_content.append("}")
        doc_content.append("```")
        doc_content.append("")

        doc_content.append("## Troubleshooting")
        doc_content.append("")
        doc_content.append("### Common Issues")
        doc_content.append("")
        doc_content.append("#### 1. Ollama Connection Error")
        doc_content.append("**Problem:** Cannot connect to Ollama server")
        doc_content.append("")
        doc_content.append("**Solution:**")
        doc_content.append("- Ensure Ollama is running: `ollama serve`")
        doc_content.append("- Check if mistral model is available: `ollama list`")
        doc_content.append("- If not available: `ollama pull mistral`")
        doc_content.append("")
        doc_content.append("#### 2. Database Permission Error")
        doc_content.append("**Problem:** Cannot create or access Chroma database")
        doc_content.append("")
        doc_content.append("**Solution:**")
        doc_content.append("- Check directory permissions")
        doc_content.append("- Ensure CHROMA_DB_PATH directory is writable")
        doc_content.append("- Try running with elevated permissions if necessary")
        doc_content.append("")
        doc_content.append("#### 3. MCP Server Not Responding")
        doc_content.append("**Problem:** MCP tools not available in VSCode")
        doc_content.append("")
        doc_content.append("**Solution:**")
        doc_content.append("- Check MCP server logs for errors")
        doc_content.append("- Verify Python path and dependencies")
        doc_content.append("- Restart VSCode and RooCode extension")
        doc_content.append("- Check MCP server configuration in VSCode settings")
        doc_content.append("")
        doc_content.append("#### 4. Large File Upload Issues")
        doc_content.append("**Problem:** Cannot upload large documents")
        doc_content.append("")
        doc_content.append("**Solution:**")
        doc_content.append("- Increase MAX_UPLOAD_SIZE in .env")
        doc_content.append("- Check available disk space")
        doc_content.append("- Consider splitting large documents into chapters")
        doc_content.append("")

        doc_content.append("## Performance Optimization")
        doc_content.append("")
        doc_content.append("### Database Optimization")
        doc_content.append("- Adjust CHUNK_SIZE and CHUNK_OVERLAP for your content")
        doc_content.append("- Use SSD storage for better I/O performance")
        doc_content.append("- Monitor memory usage with large document collections")
        doc_content.append("")
        doc_content.append("### Query Optimization")
        doc_content.append("- Use specific queries rather than broad searches")
        doc_content.append("- Limit result count for better performance")
        doc_content.append("- Use metadata filters to narrow search scope")
        doc_content.append("")
        doc_content.append("### System Resources")
        doc_content.append("- Allocate sufficient RAM (minimum 4GB recommended)")
        doc_content.append("- Use multi-core CPU for better embedding performance")
        doc_content.append("- Monitor disk space usage")
        doc_content.append("")

        content = "\n".join(doc_content)

        # Write to file
        doc_file = os.path.join(self.output_dir, "configuration_guide.md")
        with open(doc_file, 'w') as f:
            f.write(content)

        return doc_file

    def generate_user_guide(self) -> str:
        """Generate user guide for the system."""
        doc_content = []
        doc_content.append("# Ragtime Gal User Guide")
        doc_content.append(f"Generated on: {datetime.now().isoformat()}")
        doc_content.append("")
        doc_content.append("## Overview")
        doc_content.append("")
        doc_content.append("Ragtime Gal is a comprehensive book writing assistant that combines:")
        doc_content.append("- **Web Interface**: Upload and manage documents through a browser")
        doc_content.append("- **VSCode Integration**: Access powerful writing tools directly in your editor")
        doc_content.append("- **Vector Database**: Intelligent content search and analysis")
        doc_content.append("- **AI-Powered Analysis**: Writing statistics, readability analysis, and more")
        doc_content.append("")

        doc_content.append("## Getting Started")
        doc_content.append("")
        doc_content.append("### Web Interface")
        doc_content.append("")
        doc_content.append("1. **Access the web interface** at `http://localhost:5000`")
        doc_content.append("2. **Upload your documents** using the upload form")
        doc_content.append("   - Supported formats: Markdown (.md), PDF (.pdf)")
        doc_content.append("   - Files are automatically processed and indexed")
        doc_content.append("3. **Query your content** using the search interface")
        doc_content.append("   - Enter natural language queries")
        doc_content.append("   - Results show similar content with relevance scores")
        doc_content.append("")

        doc_content.append("### VSCode Integration")
        doc_content.append("")
        doc_content.append("1. **Install RooCode extension** in VSCode")
        doc_content.append("2. **Configure MCP server** (see Configuration Guide)")
        doc_content.append("3. **Access MCP tools** through the command palette")
        doc_content.append("   - Search: `Ragtime Gal: Search Content`")
        doc_content.append("   - Analysis: `Ragtime Gal: Analyze Writing`")
        doc_content.append("   - Management: `Ragtime Gal: Manage Chapters`")
        doc_content.append("")

        doc_content.append("## Features")
        doc_content.append("")
        doc_content.append("### Content Search")
        doc_content.append("- **Semantic Search**: Find content by meaning, not just keywords")
        doc_content.append("- **Chapter Filtering**: Search within specific chapters or books")
        doc_content.append("- **Similarity Scoring**: Results ranked by relevance")
        doc_content.append("- **Context Preservation**: See surrounding content for better understanding")
        doc_content.append("")
        doc_content.append("### Writing Analysis")
        doc_content.append("- **Statistics**: Word count, character count, reading time")
        doc_content.append("- **Readability**: Flesch-Kincaid scores and reading level")
        doc_content.append("- **Style Analysis**: Sentence structure and writing patterns")
        doc_content.append("- **Character Analysis**: Track character mentions and development")
        doc_content.append("")
        doc_content.append("### Content Management")
        doc_content.append("- **Add Chapters**: Create new content directly from VSCode")
        doc_content.append("- **Update Content**: Modify existing chapters safely")
        doc_content.append("- **Organize Structure**: Reorder chapters and manage book structure")
        doc_content.append("- **Version Control**: Track changes and maintain content history")
        doc_content.append("")

        doc_content.append("## Best Practices")
        doc_content.append("")
        doc_content.append("### Document Organization")
        doc_content.append("- Use clear, descriptive chapter titles")
        doc_content.append("- Structure content with proper markdown headers")
        doc_content.append("- Keep chapters at reasonable lengths (5,000-10,000 words)")
        doc_content.append("- Use consistent naming conventions")
        doc_content.append("")
        doc_content.append("### Effective Searching")
        doc_content.append("- Use specific, descriptive queries")
        doc_content.append("- Try different phrasings if initial results aren't helpful")
        doc_content.append("- Use metadata filters to narrow results")
        doc_content.append("- Review similarity scores to gauge relevance")
        doc_content.append("")
        doc_content.append("### Content Management")
        doc_content.append("- Make incremental changes rather than large rewrites")
        doc_content.append("- Use descriptive commit messages when updating content")
        doc_content.append("- Regularly backup your vector database")
        doc_content.append("- Monitor system performance with large document collections")
        doc_content.append("")

        content = "\n".join(doc_content)

        # Write to file
        doc_file = os.path.join(self.output_dir, "user_guide.md")
        with open(doc_file, 'w') as f:
            f.write(content)

        return doc_file

    def generate_all_documentation(self, mcp_manager=None, flask_app=None) -> Dict[str, str]:
        """Generate all documentation files."""
        generated_files = {}

        try:
            generated_files['mcp_tools'] = self.generate_mcp_tools_documentation(mcp_manager)
            logger.info(f"Generated MCP tools documentation: {generated_files['mcp_tools']}")
        except Exception as e:
            logger.error(f"Failed to generate MCP tools documentation: {e}")

        try:
            generated_files['api_endpoints'] = self.generate_api_documentation(flask_app)
            logger.info(f"Generated API documentation: {generated_files['api_endpoints']}")
        except Exception as e:
            logger.error(f"Failed to generate API documentation: {e}")

        try:
            generated_files['configuration'] = self.generate_configuration_guide()
            logger.info(f"Generated configuration guide: {generated_files['configuration']}")
        except Exception as e:
            logger.error(f"Failed to generate configuration guide: {e}")

        try:
            generated_files['user_guide'] = self.generate_user_guide()
            logger.info(f"Generated user guide: {generated_files['user_guide']}")
        except Exception as e:
            logger.error(f"Failed to generate user guide: {e}")

        return generated_files


def main():
    """Test documentation generation."""
    print("Documentation Generator")
    print("=" * 30)

    # Create documentation generator
    doc_gen = DocumentationGenerator("docs")

    # Generate all documentation
    generated_files = doc_gen.generate_all_documentation()

    print("Generated documentation files:")
    for doc_type, file_path in generated_files.items():
        print(f"- {doc_type}: {file_path}")

    print("\nDocumentation generation completed!")


if __name__ == "__main__":
    main()