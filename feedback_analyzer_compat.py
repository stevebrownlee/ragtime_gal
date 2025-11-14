"""
Backward Compatibility Shim for feedback_analyzer.py

This module maintains backward compatibility with code that imports from the root-level
feedback_analyzer.py. All functionality has been migrated to ragtime.services.feedback_analyzer
with enhancements:
- Structured logging with structlog
- Integration with centralized Settings
- Improved error handling and monitoring

Import from this module to maintain compatibility during the migration period.
After all dependent code is updated, this shim can be removed.
"""

import warnings
from ragtime.services.feedback_analyzer import (
    FeedbackAnalyzer,
    create_feedback_analyzer
)

# Issue deprecation warning
warnings.warn(
    "Importing from root-level feedback_analyzer.py is deprecated. "
    "Please update imports to use ragtime.services.feedback_analyzer instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'FeedbackAnalyzer',
    'create_feedback_analyzer'
]