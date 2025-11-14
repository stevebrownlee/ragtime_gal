"""
Backward compatibility wrapper for conversation management.

This module maintains compatibility with existing code that imports from the old
conversation module structure. All imports are redirected to the new unified
ragtime.services.conversation module.

Legacy import paths:
    - conversation.py -> ragtime.services.conversation
    - conversation_embedder.py -> ragtime.services.conversation
    - conversation_summarizer.py -> ragtime.services.conversation
    - enhanced_conversation.py -> ragtime.services.conversation

Examples:
    >>> from conversation import Conversation  # Old style
    >>> from ragtime.services.conversation import Conversation  # New style
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'conversation' is deprecated. "
    "Use 'from ragtime.services.conversation import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import all public classes and functions from the new location
from ragtime.services.conversation import (
    # Core classes
    Interaction,
    Conversation,
    EnhancedConversation,
    ConversationEmbedder,
    ConversationSummarizer,
    ConversationManager,

    # Session helpers
    get_conversation_from_session,
    update_conversation_in_session,
    clear_conversation_in_session,
    get_enhanced_conversation_from_session,
    update_enhanced_conversation_in_session,
)

__all__ = [
    'Interaction',
    'Conversation',
    'EnhancedConversation',
    'ConversationEmbedder',
    'ConversationSummarizer',
    'ConversationManager',
    'get_conversation_from_session',
    'update_conversation_in_session',
    'clear_conversation_in_session',
    'get_enhanced_conversation_from_session',
    'update_enhanced_conversation_in_session',
]