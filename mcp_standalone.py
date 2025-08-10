#!/usr/bin/env python3
"""
Standalone MCP server entry point for the Ragtime Gal RAG system.
This script runs the integrated MCP server with vector database access.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from shared_db import SharedDatabaseManager
from mcp_integration import MCPServerManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the standalone MCP server."""
    try:
        # Get book directory from environment or use current directory
        book_directory = os.getenv("BOOK_DIRECTORY", ".")
        logger.info(f"Using book directory: {book_directory}")

        # Initialize shared database manager
        logger.info("Initializing shared database manager...")
        db_manager = SharedDatabaseManager()

        # Initialize MCP server manager
        logger.info("Initializing MCP server manager...")
        mcp_manager = MCPServerManager(db_manager, book_directory)

        # Get the actual FastMCP server instance
        if mcp_manager.mcp_server is None:
            logger.error("Failed to create MCP server instance")
            return 1

        logger.info(f"MCP server created with {len(mcp_manager.mcp_server._tools) if hasattr(mcp_manager.mcp_server, '_tools') else 'unknown'} tools registered")
        logger.info("Starting MCP server...")

        # Run the FastMCP server directly - this will handle MCP protocol communication
        mcp_manager.mcp_server.run()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())