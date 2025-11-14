"""
Health Check API Routes

This module provides system health monitoring:
- Basic health status endpoint
- ConPort availability checking
- Workspace information

Migrated from app.py to follow project maturity standards:
- Uses Flask Blueprints for modular route organization
- Integrates with centralized Settings
- Implements structured logging
"""

import structlog
from flask import Blueprint, jsonify

# Import from new structure
from ragtime.config.settings import get_settings

# Initialize structured logger
logger = structlog.get_logger(__name__)

# Create Blueprint
health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    System health check endpoint.

    Returns:
        JSON response with:
        - status: Overall system status ('healthy' or 'unhealthy')
        - conport_available: Whether ConPort client is available
        - workspace_id: Current workspace identifier
    """
    try:
        # Get ConPort client (temporary - will be injected later)
        from conport_client import get_conport_client
        settings = get_settings()
        conport_client = get_conport_client(settings.workspace_id)

        # Check ConPort availability
        conport_available = conport_client.is_client_available()
        workspace_id = conport_client.get_workspace_id()

        # Log health check
        logger.info(
            "health_check_performed",
            conport_available=conport_available,
            workspace_id=workspace_id
        )

        return jsonify({
            'status': 'healthy',
            'conport_available': conport_available,
            'workspace_id': workspace_id
        }), 200

    except Exception as e:
        logger.error(
            "health_check_error",
            error=str(e),
            exc_info=True
        )
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


# Helper function to register blueprint
def register_health_routes(app):
    """
    Register the health blueprint with the Flask app.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(health_bp)
    logger.info(
        "health_blueprint_registered",
        routes=['/health']
    )