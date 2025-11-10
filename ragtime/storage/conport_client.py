"""
ConPort MCP Client Integration

Provides a wrapper for ConPort MCP client functionality with integrated
settings management and structured logging.

This module handles:
- Feedback storage and retrieval
- Custom data storage
- Full-text search capabilities
- Local storage fallback
"""

import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from ragtime.config.settings import get_settings
from ragtime.monitoring.logging import get_logger, log_error

logger = get_logger(__name__)


class ConPortClient:
    """
    ConPort MCP Client wrapper for feedback storage and retrieval.

    This class provides a simplified interface for storing and retrieving
    feedback data using ConPort's MCP tools with automatic fallback to
    local storage.
    """

    def __init__(self, workspace_id: Optional[str] = None):
        """
        Initialize the ConPort client.

        Args:
            workspace_id: The workspace identifier for ConPort operations.
                        If not provided, uses value from settings.
        """
        settings = get_settings()
        self.workspace_id = workspace_id or settings.conport_workspace_id
        self.is_available = False
        self._initialize_client()

    def _initialize_client(self):
        """
        Initialize the MCP client connection.

        In a full implementation, this would establish connection to ConPort MCP server.
        For now, we simulate the connection and log operations.
        """
        try:
            logger.info(
                "initializing_conport_client",
                workspace_id=self.workspace_id
            )

            # Check if ConPort is available (simulate)
            # In reality, this would test the MCP connection
            self.is_available = True

            logger.info(
                "conport_client_initialized",
                workspace_id=self.workspace_id,
                is_available=self.is_available
            )

        except Exception as e:
            log_error(
                logger,
                e,
                context={"operation": "initialize_client"}
            )
            self.is_available = False

    def log_custom_data(
        self,
        category: str,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store custom data in ConPort.

        Args:
            category: Data category (e.g., "UserFeedback")
            key: Unique key for the data
            value: The data to store (dict, list, or primitive)
            metadata: Optional metadata dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_available:
                logger.warning(
                    "conport_unavailable_using_local_storage",
                    category=category,
                    key=key
                )
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

            logger.info(
                "storing_data_in_conport",
                category=category,
                key=key,
                has_metadata=metadata is not None
            )

            # Store locally as backup/simulation
            return self._store_locally(category, key, value, metadata)

        except Exception as e:
            log_error(
                logger,
                e,
                context={
                    "operation": "log_custom_data",
                    "category": category,
                    "key": key
                }
            )
            return False

    def search_custom_data_value_fts(
        self,
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Search custom data using full-text search.

        Args:
            params: Search parameters including:
                - workspace_id: Workspace identifier
                - query_term: Search term
                - category_filter: Optional category filter
                - limit: Maximum results (default 100)

        Returns:
            List of matching entries with their data
        """
        try:
            if not self.is_available:
                logger.warning(
                    "conport_unavailable_searching_locally",
                    query_term=params.get('query_term', '')
                )
                return self._search_locally(params)

            # In a real implementation, this would use MCP tools:
            # result = use_mcp_tool(
            #     server_name="conport",
            #     tool_name="search_custom_data_value_fts",
            #     arguments=params
            # )

            logger.info(
                "searching_conport_data",
                query_term=params.get('query_term', ''),
                category_filter=params.get('category_filter', 'all')
            )

            return self._search_locally(params)

        except Exception as e:
            log_error(
                logger,
                e,
                context={
                    "operation": "search_custom_data_value_fts",
                    "params": params
                }
            )
            return []

    def _store_locally(
        self,
        category: str,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store data locally as backup/simulation.

        Creates a JSON file in .conport_local/<category>/<key>.json

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
            storage_dir = Path.cwd() / '.conport_local' / category
            storage_dir.mkdir(parents=True, exist_ok=True)

            # Prepare data with metadata
            storage_data = {
                'key': key,
                'value': value,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat(),
                'workspace_id': self.workspace_id
            }

            # Store as JSON file
            file_path = storage_dir / f"{key}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(storage_data, f, indent=2, default=str)

            logger.info(
                "data_stored_locally",
                category=category,
                key=key,
                file_path=str(file_path)
            )
            return True

        except Exception as e:
            log_error(
                logger,
                e,
                context={
                    "operation": "store_locally",
                    "category": category,
                    "key": key
                }
            )
            return False

    def _search_locally(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search local data as backup/simulation.

        Performs full-text search on JSON files in .conport_local/

        Args:
            params: Search parameters with query_term, category_filter, limit

        Returns:
            List of matching entries
        """
        try:
            category_filter = params.get('category_filter', '')
            query_term = params.get('query_term', '').lower()
            limit = params.get('limit', 100)

            results = []
            storage_base = Path.cwd() / '.conport_local'

            if not storage_base.exists():
                logger.debug(
                    "local_storage_not_found",
                    storage_base=str(storage_base)
                )
                return results

            # Determine categories to search
            if category_filter:
                categories_to_search = [category_filter]
            else:
                categories_to_search = [
                    d.name for d in storage_base.iterdir() if d.is_dir()
                ]

            # Search through categories
            for category in categories_to_search:
                category_path = storage_base / category
                if not category_path.is_dir():
                    continue

                # Search through files in category
                for file_path in category_path.glob('*.json'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Simple text search in value
                        value_str = json.dumps(
                            data.get('value', {}),
                            default=str
                        ).lower()

                        if query_term in value_str:
                            results.append(data)

                        if len(results) >= limit:
                            break

                    except Exception as e:
                        logger.warning(
                            "error_reading_local_file",
                            file_path=str(file_path),
                            error=str(e)
                        )
                        continue

                if len(results) >= limit:
                    break

            logger.info(
                "local_search_completed",
                results_found=len(results),
                query_term=query_term
            )
            return results[:limit]

        except Exception as e:
            log_error(
                logger,
                e,
                context={
                    "operation": "search_locally",
                    "params": params
                }
            )
            return []

    def get_workspace_id(self) -> str:
        """Get the current workspace ID."""
        return self.workspace_id

    def is_client_available(self) -> bool:
        """Check if the ConPort client is available."""
        return self.is_available


# Global client instance (singleton pattern)
_conport_client: Optional[ConPortClient] = None


def get_conport_client(workspace_id: Optional[str] = None) -> ConPortClient:
    """
    Get or create the global ConPort client instance.

    Uses singleton pattern to ensure only one client instance exists.

    Args:
        workspace_id: Optional workspace ID (uses settings if not provided)

    Returns:
        ConPortClient instance
    """
    global _conport_client

    if _conport_client is None:
        _conport_client = ConPortClient(workspace_id=workspace_id)

    return _conport_client


def initialize_conport_client(workspace_id: Optional[str] = None) -> ConPortClient:
    """
    Initialize or reinitialize the ConPort client.

    Useful for testing or when workspace changes.

    Args:
        workspace_id: Optional workspace ID (uses settings if not provided)

    Returns:
        ConPortClient instance
    """
    global _conport_client
    _conport_client = ConPortClient(workspace_id=workspace_id)
    return _conport_client


def reset_conport_client() -> None:
    """
    Reset the global ConPort client instance.

    Useful for testing or cleanup.
    """
    global _conport_client
    _conport_client = None