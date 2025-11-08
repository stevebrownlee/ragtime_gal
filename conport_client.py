"""
ConPort MCP Client Integration Module

This module provides a wrapper for ConPort MCP client functionality,
handling initialization and providing a clean interface for the application.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConPortClient:
    """
    ConPort MCP Client wrapper for feedback storage and retrieval.

    This class provides a simplified interface for storing and retrieving
    feedback data using ConPort's MCP tools.
    """

    def __init__(self, workspace_id: Optional[str] = None):
        """
        Initialize the ConPort client.

        Args:
            workspace_id: The workspace identifier for ConPort operations
        """
        self.workspace_id = workspace_id or os.getcwd()
        self.is_available = False
        self._initialize_client()

    def _initialize_client(self):
        """
        Initialize the MCP client connection.

        In a full implementation, this would establish connection to ConPort MCP server.
        For now, we'll simulate the connection and log operations.
        """
        try:
            # In a real implementation, this would connect to the MCP server
            # For now, we'll simulate availability
            logger.info(f"Initializing ConPort client for workspace: {self.workspace_id}")

            # Check if ConPort is available (simulate)
            # In reality, this would test the MCP connection
            self.is_available = True
            logger.info("ConPort client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ConPort client: {e}")
            self.is_available = False

    def log_custom_data(self, category: str, key: str, value: Any,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store custom data in ConPort.

        Args:
            category: Data category (e.g., "UserFeedback")
            key: Unique key for the data
            value: The data to store
            metadata: Optional metadata

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_available:
                logger.warning("ConPort client not available, storing data locally")
                return self._store_locally(category, key, value, metadata)

            # In a real implementation, this would use MCP tools:
            # result = use_mcp_tool(
            #     server_name="conport",
            #     tool_name="log_custom_data",
            #     arguments={
            #         "workspace_id": self.workspace_id,
            #         "category": category,
            #         "key": key,
            #         "value": value
            #     }
            # )

            # For now, simulate the storage and log the operation
            logger.info(f"Storing data in ConPort - Category: {category}, Key: {key}")
            logger.debug(f"Data: {json.dumps(value, indent=2, default=str)}")

            # Store locally as backup/simulation
            return self._store_locally(category, key, value, metadata)

        except Exception as e:
            logger.error(f"Error storing data in ConPort: {e}")
            return False

    def search_custom_data_value_fts(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search custom data using full-text search.

        Args:
            params: Search parameters including workspace_id, query_term, etc.

        Returns:
            List of matching entries
        """
        try:
            if not self.is_available:
                logger.warning("ConPort client not available, searching local data")
                return self._search_locally(params)

            # In a real implementation, this would use MCP tools:
            # result = use_mcp_tool(
            #     server_name="conport",
            #     tool_name="search_custom_data_value_fts",
            #     arguments=params
            # )

            # For now, simulate the search
            logger.info(f"Searching ConPort data with params: {params}")
            return self._search_locally(params)

        except Exception as e:
            logger.error(f"Error searching ConPort data: {e}")
            return []

    def _store_locally(self, category: str, key: str, value: Any,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store data locally as backup/simulation.

        Args:
            category: Data category
            key: Unique key
            value: Data to store
            metadata: Optional metadata

        Returns:
            bool: True if successful
        """
        try:
            # Create local storage directory
            storage_dir = os.path.join(os.getcwd(), '.conport_local', category)
            os.makedirs(storage_dir, exist_ok=True)

            # Prepare data with metadata
            storage_data = {
                'key': key,
                'value': value,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat(),
                'workspace_id': self.workspace_id
            }

            # Store as JSON file
            file_path = os.path.join(storage_dir, f"{key}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(storage_data, f, indent=2, default=str)

            logger.info(f"Data stored locally at: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error storing data locally: {e}")
            return False

    def _search_locally(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search local data as backup/simulation.

        Args:
            params: Search parameters

        Returns:
            List of matching entries
        """
        try:
            category_filter = params.get('category_filter', '')
            query_term = params.get('query_term', '').lower()
            limit = params.get('limit', 100)

            results = []
            storage_base = os.path.join(os.getcwd(), '.conport_local')

            if not os.path.exists(storage_base):
                return results

            # Search through categories
            categories_to_search = [category_filter] if category_filter else os.listdir(storage_base)

            for category in categories_to_search:
                category_path = os.path.join(storage_base, category)
                if not os.path.isdir(category_path):
                    continue

                # Search through files in category
                for filename in os.listdir(category_path):
                    if not filename.endswith('.json'):
                        continue

                    try:
                        file_path = os.path.join(category_path, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Simple text search in value
                        value_str = json.dumps(data.get('value', {}), default=str).lower()
                        if query_term in value_str:
                            results.append(data)

                        if len(results) >= limit:
                            break

                    except Exception as e:
                        logger.warning(f"Error reading file {filename}: {e}")
                        continue

                if len(results) >= limit:
                    break

            logger.info(f"Local search found {len(results)} results")
            return results[:limit]

        except Exception as e:
            logger.error(f"Error searching local data: {e}")
            return []

    def get_workspace_id(self) -> str:
        """Get the current workspace ID."""
        return self.workspace_id

    def is_client_available(self) -> bool:
        """Check if the ConPort client is available."""
        return self.is_available


# Global client instance
_conport_client = None

def get_conport_client(workspace_id: Optional[str] = None) -> ConPortClient:
    """
    Get or create the global ConPort client instance.

    Args:
        workspace_id: Optional workspace ID

    Returns:
        ConPortClient instance
    """
    global _conport_client

    if _conport_client is None:
        _conport_client = ConPortClient(workspace_id=workspace_id)

    return _conport_client

def initialize_conport_client(workspace_id: Optional[str] = None) -> ConPortClient:
    """
    Initialize the ConPort client.

    Args:
        workspace_id: Optional workspace ID

    Returns:
        ConPortClient instance
    """
    global _conport_client
    _conport_client = ConPortClient(workspace_id=workspace_id)
    return _conport_client