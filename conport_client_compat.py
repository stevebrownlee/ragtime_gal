"""
DEPRECATED: Backward Compatibility Shim for conport_client

This module provides backward compatibility for code importing from the old
location. New code should import from ragtime.storage instead:

    # Old (deprecated):
    from conport_client import ConPortClient, get_conport_client

    # New (recommended):
    from ragtime.storage import ConPortClient, get_conport_client

This compatibility shim will be removed in version 3.0.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'conport_client' is deprecated. "
    "Please update imports to 'from ragtime.storage import ConPortClient, get_conport_client'. "
    "This compatibility layer will be removed in version 3.0.",
    DeprecationWarning,
    stacklevel=2
)

# Import and re-export from new location
from ragtime.storage.conport_client import (
    ConPortClient,
    get_conport_client,
    initialize_conport_client,
    reset_conport_client,
)

__all__ = [
    "ConPortClient",
    "get_conport_client",
    "initialize_conport_client",
    "reset_conport_client",
]