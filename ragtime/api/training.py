"""
Training Data Generation and Model Fine-tuning API Routes

This module handles model training operations:
- Training data generation from user feedback
- Model fine-tuning job management
- Training job status tracking

Migrated from app.py to follow project maturity standards:
- Uses Flask Blueprints for modular route organization
- Integrates with centralized Settings
- Implements structured logging
- Background job processing with thread management
"""

import os
import structlog
import threading
from datetime import datetime
from flask import Blueprint, request, jsonify

# Import from new structure
from ragtime.config.settings import get_settings

# Imports from ragtime package
from ragtime.services.training_data_gen import create_training_data_generator
from ragtime.services.model_finetuner import create_model_finetuner
from ragtime.models.training import FineTuningConfig
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Initialize structured logger
logger = structlog.get_logger(__name__)

# Create Blueprint
training_bp = Blueprint('training', __name__, url_prefix='/api/training')

# Global job tracking dictionary
# TODO: Move to proper job queue/storage in future iteration
training_jobs = {}


def get_vector_db():
    """
    Helper to get vector database connection.

    TODO: This should be injected via dependency injection
    """
    try:
        settings = get_settings()
        embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url
        )
        return Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=embeddings
        )
    except Exception as e:
        logger.error(
            "vector_db_initialization_error",
            error=str(e),
            exc_info=True
        )
        raise


@training_bp.route('/generate-data', methods=['POST'])
def generate_training_data():
    """
    Generate training data from user feedback.

    Expects JSON (all optional):
        - min_positive_samples: Minimum positive pairs required (default: 50)
        - min_negative_samples: Minimum negative pairs required (default: 50)
        - include_hard_negatives: Whether to include hard negatives (default: true)
        - days_back: Days of feedback history to analyze (default: 90)
        - export_format: Output format 'csv' or 'json' (default: 'csv')

    Returns:
        JSON response with training data statistics and file path
    """
    try:
        data = request.get_json() or {}

        # Get parameters with defaults
        min_positive_samples = data.get('min_positive_samples', 50)
        min_negative_samples = data.get('min_negative_samples', 50)
        include_hard_negatives = data.get('include_hard_negatives', True)
        days_back = data.get('days_back', 90)
        export_format = data.get('export_format', 'csv')

        logger.info(
            "generating_training_data",
            days_back=days_back,
            include_hard_negatives=include_hard_negatives,
            min_positive_samples=min_positive_samples,
            min_negative_samples=min_negative_samples,
            export_format=export_format
        )

        # Get ConPort client and settings
        from conport_client import get_conport_client
        settings = get_settings()
        conport_client = get_conport_client(settings.workspace_id)

        # Create training data generator
        generator = create_training_data_generator(
            conport_client=conport_client,
            workspace_id=conport_client.get_workspace_id(),
            chroma_db=get_vector_db(),
            embedding_model=settings.embedding_model
        )

        # Map export format
        format_map = {
            'csv': 'sentence_transformers',
            'json': 'json'
        }
        output_format = format_map.get(export_format, 'sentence_transformers')

        training_dir = settings.training_data_path

        # Generate training data
        result = generator.generate_training_data(
            days_back=days_back,
            include_hard_negatives=include_hard_negatives,
            output_format=output_format,
            output_path=training_dir
        )

        # Check if generation was successful
        if not result.get('success'):
            error_msg = result.get('error', 'Unknown error during training data generation')
            logger.error(
                "training_data_generation_failed",
                error=error_msg
            )
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500

        # Check if we have enough pairs
        stats = result.get('data_statistics', {})
        positive_pairs = stats.get('positive_pairs', 0)
        negative_pairs = stats.get('negative_pairs', 0)

        if positive_pairs < min_positive_samples or negative_pairs < min_negative_samples:
            logger.warning(
                "insufficient_training_pairs",
                positive_pairs=positive_pairs,
                negative_pairs=negative_pairs,
                min_positive_samples=min_positive_samples,
                min_negative_samples=min_negative_samples
            )
            return jsonify({
                'success': False,
                'error': (
                    f'Insufficient training pairs generated. '
                    f'Got {positive_pairs} positive (need {min_positive_samples}), '
                    f'{negative_pairs} negative (need {min_negative_samples})'
                ),
                'statistics': stats
            }), 400

        # Get export details
        export_info = result.get('export', {})
        output_files = export_info.get('files', [])
        output_path = output_files[0] if output_files else 'unknown'

        # Extract statistics
        total_pairs = stats.get('total_training_pairs', 0)
        hard_negative_pairs = stats.get('hard_negatives', 0)
        source_feedback_count = stats.get('total_feedback_entries', 0)

        logger.info(
            "training_data_generated_successfully",
            total_pairs=total_pairs,
            positive_pairs=positive_pairs,
            negative_pairs=negative_pairs,
            hard_negative_pairs=hard_negative_pairs,
            output_path=output_path
        )

        return jsonify({
            'success': True,
            'training_data_path': output_path,
            'statistics': {
                'total_pairs': total_pairs,
                'positive_pairs': positive_pairs,
                'negative_pairs': negative_pairs,
                'hard_negative_pairs': hard_negative_pairs,
                'source_feedback_count': source_feedback_count
            }
        }), 200

    except Exception as e:
        logger.error(
            "training_data_generation_error",
            error=str(e),
            exc_info=True
        )
        return jsonify({
            'error': f'Error generating training data: {str(e)}'
        }), 500


@training_bp.route('/fine-tune', methods=['POST'])
def start_fine_tuning():
    """
    Start a model fine-tuning job in the background.

    Expects JSON:
        - training_data_path: Path to training data file (required)
        - base_model: Base model name (optional, from settings)
        - model_name_suffix: Suffix for output model name (optional, default: 'feedback')
        - config: Configuration object (optional):
            - batch_size: Training batch size (default: 16)
            - num_epochs: Number of epochs (default: 4)
            - learning_rate: Learning rate (default: 2e-5)
            - loss_function: Loss function name (default: 'CosineSimilarityLoss')
            - max_seq_length: Max sequence length (default: 512)
            - use_amp: Use automatic mixed precision (default: true)
            - validation_split: Validation split ratio (default: 0.2)

    Returns:
        JSON response with job ID and status (202 Accepted)
    """
    try:
        data = request.get_json()
        if not data or 'training_data_path' not in data:
            logger.warning("fine_tuning_request_missing_path")
            return jsonify({
                'error': 'Missing training_data_path parameter'
            }), 400

        training_data_path = data['training_data_path']
        if not os.path.exists(training_data_path):
            logger.warning(
                "training_data_file_not_found",
                path=training_data_path
            )
            return jsonify({
                'error': f'Training data file not found: {training_data_path}'
            }), 404

        # Get settings
        settings = get_settings()

        # Get configuration parameters
        config_data = data.get('config', {})
        config = FineTuningConfig(
            base_model_name=data.get('base_model', settings.base_model_name),
            output_model_path=settings.finetuned_models_path,
            batch_size=config_data.get('batch_size', settings.batch_size),
            num_epochs=config_data.get('num_epochs', settings.num_epochs),
            learning_rate=config_data.get('learning_rate', settings.learning_rate),
            loss_function=config_data.get('loss_function', settings.loss_function),
            max_seq_length=config_data.get('max_seq_length', settings.max_seq_length),
            use_amp=config_data.get('use_amp', settings.use_amp),
            validation_split=config_data.get('validation_split', 0.2)
        )

        model_name_suffix = data.get('model_name_suffix', 'feedback')

        # Generate job ID
        job_id = f"ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(
            "starting_fine_tuning_job",
            job_id=job_id,
            training_data_path=training_data_path,
            base_model=config.base_model_name,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs
        )

        # Initialize job tracking
        training_jobs[job_id] = {
            'job_id': job_id,
            'status': 'starting',
            'progress': 0,
            'start_time': datetime.now().isoformat(),
            'training_data_path': training_data_path,
            'config': {
                'base_model': config.base_model_name,
                'batch_size': config.batch_size,
                'num_epochs': config.num_epochs,
                'learning_rate': config.learning_rate
            }
        }

        # Define training function for background thread
        def run_training():
            """Background training function"""
            try:
                training_jobs[job_id]['status'] = 'running'
                logger.info(
                    "fine_tuning_job_running",
                    job_id=job_id
                )

                # Create fine-tuner
                finetuner = create_model_finetuner(config=config)

                # Load base model
                training_jobs[job_id]['progress'] = 10
                if not finetuner.load_base_model(config.base_model_name):
                    training_jobs[job_id]['status'] = 'failed'
                    training_jobs[job_id]['error'] = 'Failed to load base model'
                    logger.error(
                        "fine_tuning_failed_load_model",
                        job_id=job_id
                    )
                    return

                # Load training data
                training_jobs[job_id]['progress'] = 20
                train_examples, val_examples = finetuner.load_training_data(
                    training_data_path,
                    format_type='csv' if training_data_path.endswith('.csv') else 'json'
                )

                if not train_examples:
                    training_jobs[job_id]['status'] = 'failed'
                    training_jobs[job_id]['error'] = 'Failed to load training data'
                    logger.error(
                        "fine_tuning_failed_load_data",
                        job_id=job_id
                    )
                    return

                training_jobs[job_id]['progress'] = 30
                logger.info(
                    "fine_tuning_data_loaded",
                    job_id=job_id,
                    train_examples=len(train_examples),
                    val_examples=len(val_examples) if val_examples else 0
                )

                # Fine-tune model
                result = finetuner.fine_tune_model(
                    train_examples=train_examples,
                    val_examples=val_examples,
                    model_name_suffix=model_name_suffix
                )

                if result.get('success'):
                    training_jobs[job_id]['status'] = 'completed'
                    training_jobs[job_id]['progress'] = 100
                    training_jobs[job_id]['model_path'] = result['model_path']
                    training_jobs[job_id]['model_name'] = result['model_name']
                    training_jobs[job_id]['training_duration'] = result['training_duration']
                    training_jobs[job_id]['end_time'] = datetime.now().isoformat()

                    # Get final metrics from training history
                    if result.get('training_history'):
                        last_metrics = result['training_history'][-1]
                        training_jobs[job_id]['metrics'] = {
                            'final_train_loss': getattr(last_metrics, 'train_loss', 0),
                            'final_eval_score': getattr(last_metrics, 'eval_cosine_accuracy', 0)
                        }

                    logger.info(
                        "fine_tuning_job_completed",
                        job_id=job_id,
                        model_path=result['model_path'],
                        training_duration=result['training_duration']
                    )
                else:
                    training_jobs[job_id]['status'] = 'failed'
                    training_jobs[job_id]['error'] = result.get('error', 'Unknown error')
                    logger.error(
                        "fine_tuning_job_failed",
                        job_id=job_id,
                        error=result.get('error')
                    )

            except Exception as e:
                training_jobs[job_id]['status'] = 'failed'
                training_jobs[job_id]['error'] = str(e)
                logger.error(
                    "fine_tuning_job_exception",
                    job_id=job_id,
                    error=str(e),
                    exc_info=True
                )

        # Start training thread
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': 'running',
            'message': 'Fine-tuning job started'
        }), 202

    except Exception as e:
        logger.error(
            "start_fine_tuning_error",
            error=str(e),
            exc_info=True
        )
        return jsonify({
            'error': f'Error starting fine-tuning: {str(e)}'
        }), 500


@training_bp.route('/status/<job_id>', methods=['GET'])
def get_training_status(job_id):
    """
    Get the status of a fine-tuning job.

    Args:
        job_id: The training job ID

    Returns:
        JSON response with complete job status information
    """
    try:
        if job_id not in training_jobs:
            logger.warning(
                "training_job_not_found",
                job_id=job_id
            )
            return jsonify({
                'error': f'Job not found: {job_id}'
            }), 404

        job_info = training_jobs[job_id]

        logger.info(
            "training_status_retrieved",
            job_id=job_id,
            status=job_info.get('status'),
            progress=job_info.get('progress', 0)
        )

        return jsonify(job_info), 200

    except Exception as e:
        logger.error(
            "get_training_status_error",
            job_id=job_id,
            error=str(e),
            exc_info=True
        )
        return jsonify({
            'error': f'Error getting training status: {str(e)}'
        }), 500


# Helper function to register blueprint
def register_training_routes(app):
    """
    Register the training blueprint with the Flask app.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(training_bp)
    logger.info(
        "training_blueprint_registered",
        url_prefix=training_bp.url_prefix
    )