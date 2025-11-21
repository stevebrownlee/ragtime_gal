"""
Feedback Collection API Routes

This module handles feedback-related operations:
- User feedback submission on query responses
- Feedback analytics and pattern analysis
- Quick feedback summaries

Migrated from app.py to follow project maturity standards:
- Uses Flask Blueprints for modular route organization
- Integrates with centralized Settings
- Implements structured logging
- Uses migrated feedback analyzer service
"""

import structlog
import uuid
import time
from flask import Blueprint, request, jsonify, session

# Import from new structure
from ragtime.config.settings import get_settings
from ragtime.services.feedback_analyzer import create_feedback_analyzer

# Imports from ragtime package
from ragtime.services.conversation import (
    get_enhanced_conversation_from_session as get_conversation_from_session
)

# Initialize structured logger
logger = structlog.get_logger(__name__)

# Create Blueprint
feedback_bp = Blueprint('feedback', __name__, url_prefix='/api/feedback')


@feedback_bp.route('', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback on query responses.

    Expects JSON:
        - rating: Integer 1-5
        - query: The original query
        - response: The response that was rated
        - relevance (optional): Relevance rating 1-5
        - completeness (optional): Completeness rating 1-5
        - length (optional): Length rating 1-5
        - comments (optional): Text comments

    Returns:
        JSON response with feedback ID and storage confirmation
    """
    try:
        data = request.get_json()
        if not data:
            logger.warning("feedback_request_no_data")
            return jsonify({'error': 'No data provided'}), 400

        # Validate required fields
        required_fields = ['rating', 'query', 'response']
        for field in required_fields:
            if field not in data:
                logger.warning(
                    "feedback_request_missing_field",
                    field=field
                )
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400

        # Validate rating range
        rating = data.get('rating')
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            logger.warning(
                "feedback_request_invalid_rating",
                rating=rating
            )
            return jsonify({
                'error': 'Rating must be an integer between 1 and 5'
            }), 400

        logger.info(
            "processing_feedback_submission",
            rating=rating,
            query_length=len(data['query']),
            has_comments=bool(data.get('comments'))
        )

        # Get conversation from session to access document_ids and metadata
        conversation = get_conversation_from_session(session)

        # Find the most recent interaction that matches the query/response
        matching_interaction = None
        if conversation.history:
            # Look for exact match or most recent interaction
            for interaction in reversed(conversation.history):
                if (interaction.query.strip() == data['query'].strip() and
                    interaction.response.strip() == data['response'].strip()):
                    matching_interaction = interaction
                    break

            # If no exact match, use the most recent interaction
            if not matching_interaction:
                matching_interaction = conversation.history[-1]

        # Prepare feedback data for ConPort storage
        feedback_data = {
            'feedback_id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'session_id': session.get('session_id', 'unknown'),
            'rating': rating,
            'query': data['query'],
            'response': data['response'],
            'detailed_feedback': {
                'relevance': data.get('relevance'),
                'completeness': data.get('completeness'),
                'length': data.get('length'),
                'comments': data.get('comments', '')
            },
            'document_ids': (
                matching_interaction.document_ids
                if matching_interaction else []
            ),
            'metadata': (
                matching_interaction.metadata
                if matching_interaction else {}
            ),
            'conversation_context': {
                'history_length': len(conversation.history),
                'is_follow_up': data.get('is_follow_up', False)
            }
        }

        # Store feedback in ConPort using custom_data
        try:
            # Get ConPort client (temporary - will be injected later)
            from conport_client import get_conport_client
            settings = get_settings()
            conport_client = get_conport_client(settings.workspace_id)

            logger.info(
                "storing_feedback_in_conport",
                feedback_id=feedback_data['feedback_id'],
                rating=rating
            )

            # Store feedback in ConPort
            success = conport_client.log_custom_data(
                category="UserFeedback",
                key=feedback_data['feedback_id'],
                value=feedback_data,
                metadata={
                    'rating': rating,
                    'timestamp': feedback_data['timestamp'],
                    'session_id': feedback_data['session_id']
                }
            )

            if success:
                logger.info(
                    "feedback_stored_successfully",
                    feedback_id=feedback_data['feedback_id'],
                    rating=rating,
                    relevance=data.get('relevance'),
                    completeness=data.get('completeness'),
                    length=data.get('length')
                )

                return jsonify({
                    'message': 'Feedback submitted successfully',
                    'feedback_id': feedback_data['feedback_id'],
                    'storage_method': (
                        'conport'
                        if conport_client.is_client_available()
                        else 'local_backup'
                    )
                }), 200
            else:
                logger.error("feedback_storage_failed")
                return jsonify({'error': 'Failed to store feedback'}), 500

        except Exception as storage_error:
            logger.error(
                "feedback_storage_error",
                error=str(storage_error),
                exc_info=True
            )
            return jsonify({
                'error': f'Error storing feedback: {str(storage_error)}'
            }), 500

    except Exception as e:
        logger.error(
            "feedback_route_error",
            error=str(e),
            exc_info=True
        )
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@feedback_bp.route('/analytics', methods=['GET'])
def get_feedback_analytics():
    """
    Get feedback analytics and insights.

    Query parameters:
        - days_back: Number of days to analyze (default: 30)
        - min_rating: Minimum rating filter (optional)

    Returns:
        JSON response with analytics summary, patterns, and recommendations
    """
    try:
        # Get query parameters
        days_back = request.args.get('days_back', 30, type=int)
        min_rating = request.args.get('min_rating', type=int)

        logger.info(
            "generating_feedback_analytics",
            days_back=days_back,
            min_rating=min_rating
        )

        # Get ConPort client (temporary - will be injected later)
        from conport_client import get_conport_client
        settings = get_settings()
        conport_client = get_conport_client(settings.workspace_id)

        # Create feedback analyzer
        analyzer = create_feedback_analyzer(
            conport_client=conport_client,
            workspace_id=conport_client.get_workspace_id()
        )

        # Get feedback data and analyze patterns
        feedback_data = analyzer.get_feedback_data(
            days_back=days_back,
            min_rating=min_rating
        )

        if not feedback_data:
            logger.info("no_feedback_data_available")
            return jsonify({
                'summary': {
                    'total_feedback': 0,
                    'average_rating': 0,
                    'message': 'No feedback data available'
                },
                'insights': {
                    'recommendations': 'Collect more feedback to generate insights'
                }
            }), 200

        # Analyze rating patterns
        analysis = analyzer.analyze_rating_patterns(feedback_data)

        # Get successful patterns
        patterns = analyzer.identify_successful_patterns(days_back=days_back)

        logger.info(
            "feedback_analytics_generated",
            total_feedback=analysis.get('total_feedback', 0),
            average_rating=analysis.get('average_rating', 0)
        )

        return jsonify({
            'summary': {
                'total_feedback': analysis.get('total_feedback', 0),
                'average_rating': round(analysis.get('average_rating', 0), 2),
                'rating_distribution': analysis.get('rating_distribution', {}),
                'high_rated_percentage': round(
                    analysis.get('high_rated_percentage', 0), 1
                )
            },
            'successful_patterns': analysis.get(
                'successful_query_characteristics', {}
            ),
            'problematic_patterns': analysis.get(
                'problematic_query_characteristics', {}
            ),
            'recommendations': patterns.get('recommendations', {}),
            'insights': {
                'analysis_period': f'{days_back} days',
                'data_quality': 'good' if len(feedback_data) > 10 else 'limited',
                'trends': (
                    'positive' if analysis.get('average_rating', 0) >= 3.5
                    else 'needs_improvement'
                )
            }
        }), 200

    except Exception as e:
        logger.error(
            "feedback_analytics_error",
            error=str(e),
            exc_info=True
        )
        return jsonify({
            'error': f'Error getting feedback analytics: {str(e)}'
        }), 500


@feedback_bp.route('/summary', methods=['GET'])
def get_feedback_summary():
    """
    Get a quick feedback summary.

    Query parameters:
        - days_back: Number of days to summarize (default: 7)

    Returns:
        JSON response with quick feedback summary
    """
    try:
        days_back = request.args.get('days_back', 7, type=int)

        logger.info(
            "generating_feedback_summary",
            days_back=days_back
        )

        # Get ConPort client (temporary - will be injected later)
        from conport_client import get_conport_client
        settings = get_settings()
        conport_client = get_conport_client(settings.workspace_id)

        # Create feedback analyzer
        analyzer = create_feedback_analyzer(
            conport_client=conport_client,
            workspace_id=conport_client.get_workspace_id()
        )

        summary = analyzer.get_feedback_summary(days_back=days_back)

        logger.info(
            "feedback_summary_generated",
            total_feedback=summary.get('total_feedback', 0),
            average_rating=summary.get('average_rating', 0),
            trend=summary.get('trend', 'unknown')
        )

        return jsonify(summary), 200

    except Exception as e:
        logger.error(
            "feedback_summary_error",
            error=str(e),
            exc_info=True
        )
        return jsonify({
            'error': f'Error getting feedback summary: {str(e)}'
        }), 500


# Helper function to register blueprint
def register_feedback_routes(app):
    """
    Register the feedback blueprint with the Flask app.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(feedback_bp)
    logger.info(
        "feedback_blueprint_registered",
        url_prefix=feedback_bp.url_prefix
    )