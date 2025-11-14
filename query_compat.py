"""
Backward Compatibility Shim for query.py

This module maintains backward compatibility with code that imports from the root-level query.py.
All functionality has been migrated to ragtime.core.query_processor with enhancements:
- Centralized Settings management
- Structured logging with structlog
- Pydantic models for type safety
- Improved error handling

Import from this module to maintain compatibility during the migration period.
After all dependent code is updated, this shim can be removed.
"""

import warnings
from ragtime.core.query_processor import (
    # Main query functions
    query,
    query_with_feedback_optimization,
    get_query_enhancement_suggestions,
    get_feedback_summary,
    clear_optimization_cache,

    # Manager functions
    get_template_manager,
    get_context_manager,
    get_feedback_analyzer,
    get_query_enhancer,

    # Class
    QueryProcessor
)

# Issue deprecation warning
warnings.warn(
    "Importing from root-level query.py is deprecated. "
    "Please update imports to use ragtime.core.query_processor instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'query',
    'query_with_feedback_optimization',
    'get_query_enhancement_suggestions',
    'get_feedback_summary',
    'clear_optimization_cache',
    'get_template_manager',
    'get_context_manager',
    'get_feedback_analyzer',
    'get_query_enhancer',
    'QueryProcessor'
]