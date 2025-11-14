"""
A/B Testing API Routes

This module handles A/B testing operations for model comparison:
- Starting A/B tests between baseline and test models
- Tracking test metrics and user feedback
- Retrieving test results and recommendations

Migrated from app.py to follow project maturity standards:
- Uses Flask Blueprints for modular route organization
- Integrates with centralized Settings
- Implements structured logging
- Statistical analysis for model comparison
"""

import structlog
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify

# Import from new structure
from ragtime.config.settings import get_settings

# Initialize structured logger
logger = structlog.get_logger(__name__)

# Create Blueprint
testing_bp = Blueprint('testing', __name__, url_prefix='/api/testing')

# Global A/B test tracking dictionary
# TODO: Move to proper test management system in future iteration
ab_tests = {}


@testing_bp.route('/ab-test/start', methods=['POST'])
def start_ab_test():
    """
    Start an A/B test comparing two models.

    Expects JSON:
        - baseline_model: Name/path of baseline model (required)
        - test_model: Name/path of test model (required)
        - traffic_split: Fraction of traffic to test model (optional, default: 0.5)
        - duration_hours: Test duration in hours (optional, default: 24)
        - metrics: List of metrics to track (optional, default:
                   ['response_quality', 'relevance_score', 'user_satisfaction'])

    Returns:
        JSON response with test ID, status, and estimated completion time
    """
    try:
        data = request.get_json()
        if not data:
            logger.warning("ab_test_request_no_data")
            return jsonify({'error': 'No data provided'}), 400

        # Validate required parameters
        required_fields = ['baseline_model', 'test_model']
        for field in required_fields:
            if field not in data:
                logger.warning(
                    "ab_test_request_missing_field",
                    field=field
                )
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400

        baseline_model = data['baseline_model']
        test_model = data['test_model']
        traffic_split = data.get('traffic_split', 0.5)
        duration_hours = data.get('duration_hours', 24)
        metrics = data.get(
            'metrics',
            ['response_quality', 'relevance_score', 'user_satisfaction']
        )

        # Validate traffic split
        if not 0 < traffic_split < 1:
            logger.warning(
                "ab_test_invalid_traffic_split",
                traffic_split=traffic_split
            )
            return jsonify({
                'error': 'traffic_split must be between 0 and 1'
            }), 400

        # Generate test ID
        test_id = f"ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Calculate end time
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)

        logger.info(
            "starting_ab_test",
            test_id=test_id,
            baseline_model=baseline_model,
            test_model=test_model,
            traffic_split=traffic_split,
            duration_hours=duration_hours
        )

        # Initialize A/B test tracking
        ab_tests[test_id] = {
            'test_id': test_id,
            'status': 'active',
            'baseline_model': baseline_model,
            'test_model': test_model,
            'traffic_split': traffic_split,
            'start_time': start_time.isoformat(),
            'estimated_end_time': end_time.isoformat(),
            'duration_hours': duration_hours,
            'metrics_tracked': metrics,
            'baseline_metrics': {
                'query_count': 0,
                'ratings': [],
                'relevance_scores': [],
                'satisfaction_scores': []
            },
            'test_metrics': {
                'query_count': 0,
                'ratings': [],
                'relevance_scores': [],
                'satisfaction_scores': []
            }
        }

        logger.info(
            "ab_test_started",
            test_id=test_id,
            baseline_model=baseline_model,
            test_model=test_model,
            estimated_end_time=end_time.isoformat()
        )

        return jsonify({
            'success': True,
            'test_id': test_id,
            'status': 'active',
            'start_time': start_time.isoformat(),
            'estimated_end_time': end_time.isoformat(),
            'message': f'A/B test started. Will run for {duration_hours} hours.'
        }), 200

    except Exception as e:
        logger.error(
            "start_ab_test_error",
            error=str(e),
            exc_info=True
        )
        return jsonify({
            'error': f'Error starting A/B test: {str(e)}'
        }), 500


@testing_bp.route('/ab-test/<test_id>/results', methods=['GET'])
def get_ab_test_results(test_id):
    """
    Get results and analysis of an A/B test.

    Args:
        test_id: The A/B test identifier

    Returns:
        JSON response with:
        - Test configuration
        - Baseline and test model metrics
        - Statistical significance analysis
        - Deployment recommendation
    """
    try:
        if test_id not in ab_tests:
            logger.warning(
                "ab_test_not_found",
                test_id=test_id
            )
            return jsonify({
                'error': f'A/B test not found: {test_id}'
            }), 404

        test_info = ab_tests[test_id]

        logger.info(
            "retrieving_ab_test_results",
            test_id=test_id,
            status=test_info.get('status')
        )

        # Calculate average metrics
        baseline_metrics = test_info['baseline_metrics']
        test_metrics = test_info['test_metrics']

        def calc_avg(values):
            """Calculate average of a list of values"""
            return sum(values) / len(values) if values else 0

        baseline_avg = {
            'query_count': baseline_metrics['query_count'],
            'avg_rating': calc_avg(baseline_metrics['ratings']),
            'avg_relevance_score': calc_avg(baseline_metrics['relevance_scores']),
            'avg_satisfaction': calc_avg(baseline_metrics['satisfaction_scores'])
        }

        test_avg = {
            'query_count': test_metrics['query_count'],
            'avg_rating': calc_avg(test_metrics['ratings']),
            'avg_relevance_score': calc_avg(test_metrics['relevance_scores']),
            'avg_satisfaction': calc_avg(test_metrics['satisfaction_scores'])
        }

        # Statistical significance analysis
        # Simple check - for production, use proper statistical tests (t-test, chi-square)
        min_sample_size = 30
        has_enough_data = (
            baseline_metrics['query_count'] >= min_sample_size and
            test_metrics['query_count'] >= min_sample_size
        )

        improvement_threshold = 0.1  # 10% improvement required
        test_is_better = (
            test_avg['avg_rating'] > baseline_avg['avg_rating'] * (1 + improvement_threshold) and
            test_avg['avg_relevance_score'] > baseline_avg['avg_relevance_score'] * (1 + improvement_threshold)
        )

        # Determine test status based on time
        current_time = datetime.now()
        end_time = datetime.fromisoformat(test_info['estimated_end_time'])

        if current_time >= end_time:
            test_info['status'] = 'completed'
            logger.info(
                "ab_test_completed",
                test_id=test_id
            )

        # Generate recommendation
        recommendation = 'insufficient_data'
        if has_enough_data:
            if test_is_better:
                recommendation = 'deploy_test_model'
                logger.info(
                    "ab_test_recommends_deployment",
                    test_id=test_id,
                    test_model=test_info['test_model']
                )
            else:
                recommendation = 'keep_baseline'
                logger.info(
                    "ab_test_recommends_baseline",
                    test_id=test_id,
                    baseline_model=test_info['baseline_model']
                )
        else:
            logger.info(
                "ab_test_insufficient_data",
                test_id=test_id,
                baseline_count=baseline_metrics['query_count'],
                test_count=test_metrics['query_count'],
                min_required=min_sample_size
            )

        result = {
            'test_id': test_id,
            'status': test_info['status'],
            'baseline_model': test_info['baseline_model'],
            'test_model': test_info['test_model'],
            'baseline_metrics': baseline_avg,
            'test_metrics': test_avg,
            'statistical_significance': has_enough_data and test_is_better,
            'p_value': 0.05 if has_enough_data and test_is_better else 0.5,  # Simplified
            'recommendation': recommendation,
            'test_duration': {
                'start_time': test_info['start_time'],
                'end_time': test_info['estimated_end_time'],
                'status': test_info['status']
            },
            'analysis_notes': {
                'min_sample_size': min_sample_size,
                'improvement_threshold': improvement_threshold,
                'has_enough_data': has_enough_data,
                'test_is_better': test_is_better if has_enough_data else None
            }
        }

        logger.info(
            "ab_test_results_retrieved",
            test_id=test_id,
            recommendation=recommendation,
            baseline_queries=baseline_avg['query_count'],
            test_queries=test_avg['query_count']
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(
            "get_ab_test_results_error",
            test_id=test_id,
            error=str(e),
            exc_info=True
        )
        return jsonify({
            'error': f'Error getting A/B test results: {str(e)}'
        }), 500


# Helper function to register blueprint
def register_testing_routes(app):
    """
    Register the testing blueprint with the Flask app.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(testing_bp)
    logger.info(
        "testing_blueprint_registered",
        url_prefix=testing_bp.url_prefix
    )