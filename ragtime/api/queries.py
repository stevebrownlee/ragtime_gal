"""
Query Processing API Routes

This module handles query-related operations:
- Query processing with conversation context
- Conversation history management
- Conversation status tracking

Migrated from app.py to follow project maturity standards:
- Uses Flask Blueprints for modular route organization
- Integrates with centralized Settings
- Implements structured logging
- Uses migrated query processor and conversation modules
"""

import structlog
from flask import Blueprint, request, jsonify, session
import markdown

# Import from new structure
from ragtime.config.settings import get_settings
from ragtime.core.query_processor import query

# Imports from ragtime package
from ragtime.services.conversation import (
    get_enhanced_conversation_from_session as get_conversation_from_session,
    update_enhanced_conversation_in_session as update_conversation_in_session,
    clear_conversation_in_session
)

# Initialize structured logger
logger = structlog.get_logger(__name__)

# Create Blueprint
queries_bp = Blueprint('queries', __name__, url_prefix='/api/queries')


@queries_bp.route('/query', methods=['POST'])
def process_query():
    """
    Process a query against the vector database with conversation context.

    Expects JSON:
        - query: The question to ask
        - template (optional): Prompt template name
        - temperature (optional): LLM temperature (0-1)
        - collection (optional): Collection name to query

    Returns:
        JSON response with answer, conversation metadata
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            logger.warning("query_request_missing_query_parameter")
            return jsonify({'error': 'Missing query parameter'}), 400

        # Extract optional parameters
        template_name = data.get('template')
        temperature = data.get('temperature')
        collection_name = data.get('collection', "langchain")

        # Validate temperature if provided
        if temperature is not None:
            try:
                temperature = float(temperature)
                if temperature < 0 or temperature > 1:
                    return jsonify({
                        'error': 'Temperature must be between 0 and 1'
                    }), 400
            except ValueError:
                return jsonify({'error': 'Temperature must be a number'}), 400

        logger.info(
            "processing_query",
            query_length=len(data.get('query')),
            template=template_name,
            temperature=temperature,
            collection=collection_name
        )

        # Get conversation from session
        conversation = get_conversation_from_session(session)

        logger.debug(
            "conversation_retrieved",
            history_length=len(conversation.get_history())
        )

        # Call the query function with parameters
        try:
            response, document_ids, metadata = query(
                data.get('query'),
                template_name=template_name,
                temperature=temperature,
                conversation=conversation,
                collection_name=collection_name
            )

            # Check if response indicates an error
            if response.startswith("Error") or response.startswith("An error occurred"):
                logger.error(
                    "query_function_error",
                    response=response[:100]
                )
                return jsonify({'error': response}), 500

            # Add interaction to conversation history
            conversation.add_interaction(
                query=data.get('query'),
                response=response,
                document_ids=document_ids,
                metadata=metadata
            )

            # Update conversation in session
            update_conversation_in_session(session, conversation)

            logger.info(
                "query_processed_successfully",
                response_length=len(response),
                document_count=len(document_ids),
                new_history_length=len(conversation.get_history()),
                is_follow_up=metadata.get('is_follow_up', False)
            )

            # Convert markdown response to HTML
            html_response = markdown.markdown(
                response,
                extensions=['extra', 'codehilite', 'fenced_code', 'tables']
            )

            return jsonify({
                'message': html_response,
                'conversation_active': True,
                'is_follow_up': metadata.get('is_follow_up', False),
                'history_length': len(conversation.get_history())
            }), 200

        except Exception as query_error:
            logger.error(
                "query_execution_error",
                error=str(query_error),
                exc_info=True
            )
            return jsonify({
                'error': f'Error executing query: {str(query_error)}'
            }), 500

    except Exception as e:
        logger.error(
            "query_route_error",
            error=str(e),
            exc_info=True
        )
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@queries_bp.route('/clear-history', methods=['POST'])
def clear_history():
    """
    Clear the conversation history for the current session.

    Returns:
        JSON response confirming history cleared
    """
    try:
        logger.info("clearing_conversation_history")

        # Clear the conversation from the session
        clear_conversation_in_session(session)

        logger.info("conversation_history_cleared")

        return jsonify({
            'message': 'Conversation history cleared successfully',
            'conversation_active': False
        }), 200

    except Exception as e:
        logger.error(
            "clear_history_error",
            error=str(e),
            exc_info=True
        )
        return jsonify({
            'error': f'Error clearing conversation history: {str(e)}'
        }), 500


@queries_bp.route('/conversation-status', methods=['GET'])
def conversation_status():
    """
    Get the current status of the conversation.

    Returns:
        JSON response with conversation active status and history length
    """
    try:
        conversation = get_conversation_from_session(session)
        history_length = len(conversation.get_history())

        logger.debug(
            "conversation_status_retrieved",
            history_length=history_length,
            is_active=history_length > 0
        )

        return jsonify({
            'conversation_active': history_length > 0,
            'history_length': history_length
        }), 200

    except Exception as e:
        logger.error(
            "conversation_status_error",
            error=str(e),
            exc_info=True
        )
        return jsonify({
            'error': f'Error getting conversation status: {str(e)}'
        }), 500


# Helper function to register blueprint
def register_queries_routes(app):
    """
    Register the queries blueprint with the Flask app.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(queries_bp)
    logger.info(
        "queries_blueprint_registered",
        url_prefix=queries_bp.url_prefix
    )