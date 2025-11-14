"""
Backward Compatibility Shim for query_enhancer.py

This module maintains backward compatibility with code that imports from the root-level
query_enhancer.py. All functionality has been migrated to ragtime.services.query_enhancer
with enhancements:
- Structured logging with structlog
- Integration with centralized Settings
- Improved error handling and monitoring

Import from this module to maintain compatibility during the migration period.
After all dependent code is updated, this shim can be removed.
"""

import warnings
from ragtime.services.query_enhancer import (
    QueryEnhancer,
    create_query_enhancer
)

# Issue deprecation warning
warnings.warn(
    "Importing from root-level query_enhancer.py is deprecated. "
    "Please update imports to use ragtime.services.query_enhancer instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'QueryEnhancer',
    'create_query_enhancer'
]