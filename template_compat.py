"""
Backward compatibility wrapper for template.py

This module maintains compatibility with existing code that imports from
template.py. All imports are redirected to the new unified
ragtime.utils.templates module.

Examples:
    >>> from template import load_html_template  # Old style
    >>> from ragtime.utils.templates import load_html_template  # New style
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'template' is deprecated. "
    "Use 'from ragtime.utils.templates import load_html_template, get_template_path' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from ragtime.utils.templates import (
    get_template_path,
    load_html_template,
    HTMLTemplateManager,
)

__all__ = ['get_template_path', 'load_html_template', 'HTMLTemplateManager']