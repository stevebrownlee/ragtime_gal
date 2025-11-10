"""
Ragtime Gal - RAG Server with Feedback-Driven Optimization

A sophisticated RAG (Retrieval-Augmented Generation) server featuring:
- Document embedding and vector storage
- Intelligent query processing with enhancement
- User feedback collection and analysis
- Model fine-tuning based on feedback
- Conversation management
- ConPort integration for context management

Package Structure:
- api/: Flask API endpoints
- core/: Core RAG functionality (embeddings, retrieval, generation)
- models/: Pydantic models for type safety
- services/: Business logic services
- storage/: Storage layer (vector DB, ConPort, sessions)
- utils/: Utility functions
- monitoring/: Logging and monitoring
- config/: Configuration management

Usage:
    from ragtime import create_app, get_settings
    from ragtime.models import QueryRequest, DocumentMetadata
    from ragtime.services import FeedbackAnalyzer, QueryEnhancer
"""

__version__ = "2.0.0"
__author__ = "Ragtime Gal Team"

# Re-export commonly used items for convenience
from ragtime.config.settings import get_settings, Settings
from ragtime.monitoring.logging import get_logger, configure_logging

__all__ = [
    "__version__",
    "get_settings",
    "Settings",
    "get_logger",
    "configure_logging",
]