"""
Backward compatibility wrapper for template_manager.py

This module maintains compatibility with existing code that imports from
template_manager.py. All imports are redirected to the new unified
ragtime.utils.templates module.

Examples:
    >>> from template_manager import TemplateManager  # Old style
    >>> from ragtime.utils.templates import TemplateManager  # New style
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'template_manager' is deprecated. "
    "Use 'from ragtime.utils.templates import TemplateManager' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from ragtime.utils.templates import (
    TemplateManager,
    PromptManager,
)

__all__ = ['TemplateManager', 'PromptManager']