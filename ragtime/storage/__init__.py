"""
Storage Layer Package

Handles all data persistence operations including:
- ConPort integration for context management
- Vector database operations (ChromaDB)
- Session management

Usage:
    from ragtime.storage import get_conport_client, get_vector_db
    from ragtime.storage import ConPortClient, VectorDatabase
"""

from ragtime.storage.conport_client import (
    ConPortClient,
    get_conport_client,
    initialize_conport_client,
    reset_conport_client,
)

from ragtime.storage.vector_db import (
    VectorDatabase,
    get_vector_db,
    reset_vector_db,
)

__all__ = [
    # ConPort
    "ConPortClient",
    "get_conport_client",
    "initialize_conport_client",
    "reset_conport_client",
    # Vector Database
    "VectorDatabase",
    "get_vector_db",
    "reset_vector_db",
]