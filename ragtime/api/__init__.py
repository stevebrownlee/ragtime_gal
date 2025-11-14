"""
API Routes Package

Contains Flask Blueprint modules for all API endpoints:
- Document management (embed, collections, purge)
- Query processing (query, conversation)
- Feedback collection (submit, analytics)
- Training operations (data generation, fine-tuning)
- Health monitoring

Usage:
    from ragtime.api import register_all_routes
    register_all_routes(app)
"""

from ragtime.api.documents import documents_bp, register_documents_routes
from ragtime.api.queries import queries_bp, register_queries_routes
from ragtime.api.feedback import feedback_bp, register_feedback_routes
from ragtime.api.training import training_bp, register_training_routes
from ragtime.api.testing import testing_bp, register_testing_routes
from ragtime.api.health import health_bp, register_health_routes

__all__ = [
    # Documents
    "documents_bp",
    "register_documents_routes",
    # Queries
    "queries_bp",
    "register_queries_routes",
    # Feedback
    "feedback_bp",
    "register_feedback_routes",
    # Training
    "training_bp",
    "register_training_routes",
    # Testing
    "testing_bp",
    "register_testing_routes",
    # Health
    "health_bp",
    "register_health_routes",
]


def register_all_routes(app):
    """
    Register all API blueprints with the Flask app.

    Args:
        app: Flask application instance
    """
    register_documents_routes(app)
    register_queries_routes(app)
    register_feedback_routes(app)
    register_training_routes(app)
    register_testing_routes(app)
    register_health_routes(app)