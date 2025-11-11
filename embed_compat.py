"""
DEPRECATED: Backward Compatibility Shim for embed module

This module provides backward compatibility for code importing from the old
embed.py file. New code should import from ragtime.core instead:

    # Old (deprecated):
    from embed import embed, allowed_file

    # New (recommended):
    from ragtime.core.embeddings import embed_document, allowed_file

This compatibility shim will be removed in version 3.0.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'embed' is deprecated. "
    "Please update imports to 'from ragtime.core.embeddings import embed_document, allowed_file'. "
    "This compatibility layer will be removed in version 3.0.",
    DeprecationWarning,
    stacklevel=2
)

# Import and re-export from new location
from ragtime.core.embeddings import (
    embed,
    allowed_file,
    save_uploaded_file as save_file,
    embed_document,
    embed_documents_batch,
)

__all__ = [
    "embed",
    "allowed_file",
    "save_file",
    "embed_document",
    "embed_documents_batch",
]