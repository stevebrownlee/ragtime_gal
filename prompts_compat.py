"""
Backward compatibility wrapper for prompts.py

This module maintains compatibility with existing code that imports from
prompts.py. All imports are redirected to the new unified
ragtime.utils.templates module.

Examples:
    >>> from prompts import get_template  # Old style
    >>> from ragtime.utils.templates import get_template  # New style
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'prompts' is deprecated. "
    "Use 'from ragtime.utils.templates import get_template, load_templates' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from ragtime.utils.templates import (
    get_templates_file,
    load_templates,
    get_template,
    PromptManager,
)

# Legacy constant for compatibility
DEFAULT_TEMPLATES = PromptManager.LEGACY_DEFAULT_TEMPLATES

__all__ = [
    'DEFAULT_TEMPLATES',
    'get_templates_file',
    'load_templates',
    'get_template',
    'PromptManager',
]