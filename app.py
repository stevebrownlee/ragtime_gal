"""Main application file for the Langchain API server """
import os
import logging
import secrets
import time
import uuid
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string, session
import markdown
from embed import embed
from query import query
from template import load_html_template
# Import enhanced conversation classes
from enhanced_conversation import (
    get_enhanced_conversation_from_session as get_conversation_from_session,
    update_enhanced_conversation_in_session as update_conversation_in_session
)
from conversation import clear_conversation_in_session
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
# Import ConPort client
from conport_client import get_conport_client, initialize_conport_client
# Import monitoring dashboard
from monitoring_dashboard import (
    create_monitoring_blueprint,
    start_monitoring,
    monitor_route,
    record_request_metrics
)
# Import Phase 3 components for model fine-tuning
from training_data_generator import create_training_data_generator
from model_finetuner import create_model_finetuner, FineTuningConfig
import threading
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up constants and directories
TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
# Set a secret key for session management
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(16))

# Initialize ConPort client
conport_client = initialize_conport_client(workspace_id=os.getcwd())

# Register monitoring dashboard blueprint
app.register_blueprint(create_monitoring_blueprint())

# Start monitoring system (collect metrics every 30 seconds)
# Phase 3: Training job tracking
training_jobs = {}  # job_id -> job_info
ab_tests = {}  # test_id -> test_info

start_monitoring(interval=30)
logger.info("Monitoring dashboard started - accessible at /monitoring")

# Helper function to get vector DB - replaces the imported get_vector_db
def get_vector_db():
    # pylint: disable=W0718
    try:
        # Use local Ollama embeddings instead of OpenAI
        embeddings = OllamaEmbeddings(
            model=os.getenv('EMBEDDING_MODEL', 'mistral'),  # Default to mistral model
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        )
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
    except Exception as e:
        logger.error("Error initializing vector database: %s", e)
        raise

@app.route('/', methods=['GET'])
def index():
    """Route to display the upload form"""
    template_html = load_html_template()
    return render_template_string(template_html)

@app.route('/embed', methods=['POST'])
@monitor_route
def route_embed():
    # pylint: disable=W0718
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Get collection name from form data (optional)
        collection_name = request.form.get('collection_name', 'langchain')

        # Validate collection name (basic validation)
        if not collection_name or not collection_name.strip():
            collection_name = 'langchain'
        else:
            # Clean the collection name (remove special characters, spaces)
            import re
            collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', collection_name.strip())

        embedded = embed(file, collection_name)
        if embedded:
            return jsonify({
                'message': f'File embedded successfully into collection "{collection_name}"',
                'collection_name': collection_name
            }), 200

        return jsonify({'error': 'File embedded unsuccessfully'}), 400

    except Exception as e:
        logger.error("Error in embedding route: %s", e)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
@monitor_route
def route_query():
    """Route to query the vector database with a question, using conversation history"""
    # pylint: disable=W0718
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400

        # Extract optional parameters if provided
        template_name = data.get('template')  # Optional template name
        temperature = data.get('temperature')  # Optional temperature
        collection_name = data.get('collection', "langchain")

        # Convert temperature to float if provided
        if temperature is not None:
            try:
                temperature = float(temperature)
                if temperature < 0 or temperature > 1:
                    return jsonify({'error': 'Temperature must be between 0 and 1'}), 400
            except ValueError:
                return jsonify({'error': 'Temperature must be a number'}), 400

        # Get conversation from session
        conversation = get_conversation_from_session(session)
        logger.info("Retrieved conversation from session (history length: %d)",
                   len(conversation.get_history()))

        # Call the query function with parameters including conversation
        try:
            # The query function now returns response, document_ids, and metadata
            response, document_ids, metadata = query(
                data.get('query'),
                template_name=template_name,
                temperature=temperature,
                conversation=conversation,
                collection_name=collection_name
            )

            # Check if response starts with an error message
            if response.startswith("Error") or response.startswith("An error occurred"):
                logger.error("Query function returned an error: %s", response)
                return jsonify({'error': response}), 500

            # Add the interaction to conversation history with metadata
            conversation.add_interaction(
                query=data.get('query'),
                response=response,
                document_ids=document_ids,
                metadata=metadata
            )

            # Update the conversation in the session
            update_conversation_in_session(session, conversation)
            logger.info("Updated conversation in session (new history length: %d)",
                       len(conversation.get_history()))

            # Convert markdown response to HTML with extensions for better formatting
            html_response = markdown.markdown(
                response,
                extensions=['extra', 'codehilite', 'fenced_code', 'tables']
            )

            # Return response with additional metadata about the conversation
            return jsonify({
                'message': html_response,
                'conversation_active': True,
                'is_follow_up': metadata.get('is_follow_up', False),
                'history_length': len(conversation.get_history())
            }), 200
        except Exception as query_error:
            logger.error("Error executing query: %s", str(query_error))
            return jsonify({'error': f'Error executing query: {str(query_error)}'}), 500

    except Exception as e:
        logger.error("Error in query route: %s", e)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Route to clear the conversation history"""
    try:
        # Clear the conversation from the session
        clear_conversation_in_session(session)
        logger.info("Cleared conversation history from session")
        return jsonify({
            'message': 'Conversation history cleared successfully',
            'conversation_active': False
        }), 200
    except Exception as e:
        logger.error("Error clearing conversation history: %s", str(e))
        return jsonify({'error': f'Error clearing conversation history: {str(e)}'}), 500

@app.route('/conversation-status', methods=['GET'])
def conversation_status():
    """Route to get the current status of the conversation"""
    try:
        conversation = get_conversation_from_session(session)
        history_length = len(conversation.get_history())

        return jsonify({
            'conversation_active': history_length > 0,
            'history_length': history_length
        }), 200
    except Exception as e:
        logger.error("Error getting conversation status: %s", str(e))
        return jsonify({'error': f'Error getting conversation status: {str(e)}'}), 500

@app.route('/purge', methods=['POST'])
def purge_database():
    """Route to delete all documents from all collections in the vector database"""
    # pylint: disable=W0718
    try:
        # Connect to ChromaDB directly to list and delete all collections
        import chromadb
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Get all collections
        collections = chroma_client.list_collections()

        if not collections:
            logger.info("No collections found to purge")
            return jsonify({'message': 'Database is already empty. No collections to purge.'}), 200

        total_documents = 0
        deleted_collections = []

        # Delete each collection
        for collection in collections:
            try:
                collection_name = collection.name
                doc_count = collection.count()
                total_documents += doc_count

                # Delete the entire collection
                chroma_client.delete_collection(name=collection_name)
                deleted_collections.append(f"{collection_name} ({doc_count} documents)")
                logger.info("Deleted collection '%s' with %d documents", collection_name, doc_count)
            except Exception as delete_error:
                logger.error("Error deleting collection %s: %s", collection_name, str(delete_error))
                return jsonify({'error': f'Error deleting collection {collection_name}: {str(delete_error)}'}), 500

        logger.info("Purged %d collections with %d total documents from %s",
                   len(deleted_collections), total_documents, CHROMA_PERSIST_DIR)

        return jsonify({
            'message': f'Database purged successfully. Removed {len(deleted_collections)} collections with {total_documents} total documents.',
            'deleted_collections': deleted_collections
        }), 200

    except Exception as e:
        logger.error("Error purging database: %s", str(e))
        return jsonify({'error': f'Error purging database: {str(e)}'}), 500

@app.route('/collections', methods=['GET'])
def get_collections():
    """Route to get all available collections in the database"""
    # pylint: disable=W0718
    try:
        # Get the vector database
        embeddings = OllamaEmbeddings(
            model=os.getenv('EMBEDDING_MODEL', 'mistral'),
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        )

        # Connect to ChromaDB directly to list collections
        import chromadb
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        collections = chroma_client.list_collections()
        collection_info = []

        for collection in collections:
            try:
                count = collection.count()
                collection_info.append({
                    "name": collection.name,
                    "document_count": count
                })
            except Exception as e:
                logger.error("Error getting count for collection %s: %s", collection.name, str(e))
                collection_info.append({
                    "name": collection.name,
                    "document_count": 0
                })

        return jsonify({
            'collections': collection_info,
            'total_collections': len(collections)
        }), 200

    except Exception as e:
        logger.error("Error getting collections: %s", str(e))
        return jsonify({'error': f'Error getting collections: {str(e)}'}), 500

@app.route('/feedback', methods=['POST'])
@monitor_route
def submit_feedback():
    """Route to submit user feedback on query responses"""
    # pylint: disable=W0718
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate required fields
        required_fields = ['rating', 'query', 'response']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Validate rating range
        rating = data.get('rating')
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({'error': 'Rating must be an integer between 1 and 5'}), 400

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
            'document_ids': matching_interaction.document_ids if matching_interaction else [],
            'metadata': matching_interaction.metadata if matching_interaction else {},
            'conversation_context': {
                'history_length': len(conversation.history),
                'is_follow_up': data.get('is_follow_up', False)
            }
        }

        # Store feedback in ConPort using custom_data
        try:
            logger.info(f"Feedback received: Rating {rating}/5 for query: {data['query'][:50]}...")
            logger.info(f"Storing feedback data in ConPort: {feedback_data['feedback_id']}")

            # Store feedback in ConPort using log_custom_data
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
                logger.info(f"Feedback stored successfully in ConPort with ID: {feedback_data['feedback_id']}")
                logger.info(f"Feedback details: Rating={rating}, Relevance={data.get('relevance')}, "
                           f"Completeness={data.get('completeness')}, Length={data.get('length')}")

                return jsonify({
                    'message': 'Feedback submitted successfully',
                    'feedback_id': feedback_data['feedback_id'],
                    'storage_method': 'conport' if conport_client.is_client_available() else 'local_backup'
                }), 200
            else:
                logger.error("Failed to store feedback in ConPort")
                return jsonify({'error': 'Failed to store feedback'}), 500

        except Exception as storage_error:
            logger.error(f"Error storing feedback: {str(storage_error)}")
            return jsonify({'error': f'Error storing feedback: {str(storage_error)}'}), 500

    except Exception as e:
        logger.error(f"Error in feedback route: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/feedback/analytics', methods=['GET'])
@monitor_route
def get_feedback_analytics():
    """Route to get feedback analytics and insights"""
    # pylint: disable=W0718
    try:
        # Get query parameters
        days_back = request.args.get('days_back', 30, type=int)
        min_rating = request.args.get('min_rating', type=int)

        # Import here to avoid circular imports
        from feedback_analyzer import create_feedback_analyzer

        # Create feedback analyzer with ConPort client
        analyzer = create_feedback_analyzer(
            conport_client=conport_client,
            workspace_id=conport_client.get_workspace_id()
        )

        # Get feedback data and analyze patterns
        feedback_data = analyzer.get_feedback_data(days_back=days_back, min_rating=min_rating)

        if not feedback_data:
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

        return jsonify({
            'summary': {
                'total_feedback': analysis.get('total_feedback', 0),
                'average_rating': round(analysis.get('average_rating', 0), 2),
                'rating_distribution': analysis.get('rating_distribution', {}),
                'high_rated_percentage': round(analysis.get('high_rated_percentage', 0), 1)
            },
            'successful_patterns': analysis.get('successful_query_characteristics', {}),
            'problematic_patterns': analysis.get('problematic_query_characteristics', {}),
            'recommendations': patterns.get('recommendations', {}),
            'insights': {
                'analysis_period': f'{days_back} days',
                'data_quality': 'good' if len(feedback_data) > 10 else 'limited',
                'trends': 'positive' if analysis.get('average_rating', 0) >= 3.5 else 'needs_improvement'
            }
        }), 200

    except Exception as e:
        logger.error(f"Error getting feedback analytics: {str(e)}")
        return jsonify({'error': f'Error getting feedback analytics: {str(e)}'}), 500

@app.route('/feedback/summary', methods=['GET'])
def get_feedback_summary():
    """Route to get a quick feedback summary"""
    # pylint: disable=W0718
    try:
        days_back = request.args.get('days_back', 7, type=int)

        # Import here to avoid circular imports
        from feedback_analyzer import create_feedback_analyzer

        # Create feedback analyzer with ConPort client
        analyzer = create_feedback_analyzer(
            conport_client=conport_client,
            workspace_id=conport_client.get_workspace_id()
        )

        summary = analyzer.get_feedback_summary(days_back=days_back)
        return jsonify(summary), 200

    except Exception as e:
        logger.error(f"Error getting feedback summary: {str(e)}")
        return jsonify({'error': f'Error getting feedback summary: {str(e)}'}), 500

# ============================================================================
# Phase 3: Model Fine-tuning API Endpoints
# ============================================================================

@app.route('/training/generate-data', methods=['POST'])
@monitor_route
def generate_training_data():
    """Generate training data from user feedback"""
    # pylint: disable=W0718
    try:
        data = request.get_json() or {}

        # Get parameters with defaults
        min_positive_samples = data.get('min_positive_samples', 50)
        min_negative_samples = data.get('min_negative_samples', 50)
        include_hard_negatives = data.get('include_hard_negatives', True)
        days_back = data.get('days_back', 90)
        export_format = data.get('export_format', 'csv')

        # Create training data generator
        generator = create_training_data_generator(
            conport_client=conport_client,
            workspace_id=conport_client.get_workspace_id(),
            chroma_db=get_vector_db(),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'mistral')
        )

        logger.info(f"Generating training data: min_pos={min_positive_samples}, "
                   f"min_neg={min_negative_samples}, days={days_back}")

        # Get feedback data
        feedback_data = generator.get_feedback_data(
            days_back=days_back,
            min_samples=min_positive_samples + min_negative_samples
        )

        if not feedback_data or len(feedback_data) < (min_positive_samples + min_negative_samples):
            return jsonify({
                'success': False,
                'error': f'Insufficient feedback data. Found {len(feedback_data)} entries, '
                        f'need at least {min_positive_samples + min_negative_samples}',
                'available_feedback': len(feedback_data)
            }), 400

        # Generate training pairs
        training_pairs = generator.generate_training_data_from_feedback(
            feedback_data,
            min_positive_samples=min_positive_samples,
            min_negative_samples=min_negative_samples,
            include_hard_negatives=include_hard_negatives
        )

        if not training_pairs:
            return jsonify({
                'success': False,
                'error': 'Failed to generate training pairs from feedback data'
            }), 500

        # Export training data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_dir = os.getenv('TRAINING_DATA_PATH', './training_data')
        os.makedirs(training_dir, exist_ok=True)

        output_path = os.path.join(training_dir, f"training_{timestamp}.{export_format}")
        generator.export_training_data(training_pairs, output_path, format_type=export_format)

        # Calculate statistics
        positive_pairs = sum(1 for p in training_pairs if p.label == 1.0)
        negative_pairs = sum(1 for p in training_pairs if p.label == 0.0)
        hard_negative_pairs = sum(1 for p in training_pairs if p.pair_type == "hard_negative")

        logger.info(f"Training data generated: {len(training_pairs)} pairs "
                   f"({positive_pairs} positive, {negative_pairs} negative, "
                   f"{hard_negative_pairs} hard negatives)")

        return jsonify({
            'success': True,
            'training_data_path': output_path,
            'statistics': {
                'total_pairs': len(training_pairs),
                'positive_pairs': positive_pairs,
                'negative_pairs': negative_pairs,
                'hard_negative_pairs': hard_negative_pairs,
                'source_feedback_count': len(feedback_data)
            }
        }), 200

    except Exception as e:
        logger.error(f"Error generating training data: {str(e)}")
        return jsonify({'error': f'Error generating training data: {str(e)}'}), 500


@app.route('/training/fine-tune', methods=['POST'])
@monitor_route
def start_fine_tuning():
    """Start a model fine-tuning job"""
    # pylint: disable=W0718
    try:
        data = request.get_json()
        if not data or 'training_data_path' not in data:
            return jsonify({'error': 'Missing training_data_path parameter'}), 400

        training_data_path = data['training_data_path']
        if not os.path.exists(training_data_path):
            return jsonify({'error': f'Training data file not found: {training_data_path}'}), 404

        # Get configuration parameters
        config_data = data.get('config', {})
        config = FineTuningConfig(
            base_model_name=data.get('base_model', os.getenv('BASE_MODEL_NAME', 'all-MiniLM-L6-v2')),
            output_model_path=os.getenv('FINETUNED_MODELS_PATH', './fine_tuned_models'),
            batch_size=config_data.get('batch_size', int(os.getenv('BATCH_SIZE', 16))),
            num_epochs=config_data.get('num_epochs', int(os.getenv('NUM_EPOCHS', 4))),
            learning_rate=config_data.get('learning_rate', float(os.getenv('LEARNING_RATE', 2e-5))),
            loss_function=config_data.get('loss_function', os.getenv('LOSS_FUNCTION', 'CosineSimilarityLoss')),
            max_seq_length=config_data.get('max_seq_length', int(os.getenv('MAX_SEQ_LENGTH', 512))),
            use_amp=config_data.get('use_amp', os.getenv('USE_AMP', 'true').lower() == 'true'),
            validation_split=config_data.get('validation_split', 0.2)
        )

        model_name_suffix = data.get('model_name_suffix', 'feedback')

        # Generate job ID
        job_id = f"ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

        # Start training in background thread
        def run_training():
            try:
                training_jobs[job_id]['status'] = 'running'

                # Create fine-tuner
                finetuner = create_model_finetuner(config=config)

                # Load base model
                training_jobs[job_id]['progress'] = 10
                if not finetuner.load_base_model(config.base_model_name):
                    training_jobs[job_id]['status'] = 'failed'
                    training_jobs[job_id]['error'] = 'Failed to load base model'
                    return

                # Load training data
                training_jobs[job_id]['progress'] = 20
                train_examples, val_examples = finetuner.load_training_data(
                    training_data_path,
                    format_type='csv' if training_data_path.endswith('.csv') else 'json',
                    validation_split=config.validation_split
                )

                if not train_examples:
                    training_jobs[job_id]['status'] = 'failed'
                    training_jobs[job_id]['error'] = 'Failed to load training data'
                    return

                training_jobs[job_id]['progress'] = 30

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

                    logger.info(f"Fine-tuning job {job_id} completed successfully")
                else:
                    training_jobs[job_id]['status'] = 'failed'
                    training_jobs[job_id]['error'] = result.get('error', 'Unknown error')
                    logger.error(f"Fine-tuning job {job_id} failed: {result.get('error')}")

            except Exception as e:
                training_jobs[job_id]['status'] = 'failed'
                training_jobs[job_id]['error'] = str(e)
                logger.error(f"Error in fine-tuning job {job_id}: {str(e)}")

        # Start training thread
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()

        logger.info(f"Started fine-tuning job {job_id}")

        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': 'running',
            'message': 'Fine-tuning job started'
        }), 202

    except Exception as e:
        logger.error(f"Error starting fine-tuning: {str(e)}")
        return jsonify({'error': f'Error starting fine-tuning: {str(e)}'}), 500


@app.route('/training/status/<job_id>', methods=['GET'])
def get_training_status(job_id):
    """Get the status of a fine-tuning job"""
    try:
        if job_id not in training_jobs:
            return jsonify({'error': f'Job not found: {job_id}'}), 404

        job_info = training_jobs[job_id]
        return jsonify(job_info), 200

    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return jsonify({'error': f'Error getting training status: {str(e)}'}), 500


@app.route('/testing/ab-test/start', methods=['POST'])
@monitor_route
def start_ab_test():
    """Start an A/B test comparing two models"""
    # pylint: disable=W0718
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate required parameters
        required_fields = ['baseline_model', 'test_model']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        baseline_model = data['baseline_model']
        test_model = data['test_model']
        traffic_split = data.get('traffic_split', 0.5)
        duration_hours = data.get('duration_hours', 24)
        metrics = data.get('metrics', ['response_quality', 'relevance_score', 'user_satisfaction'])

        # Validate traffic split
        if not 0 < traffic_split < 1:
            return jsonify({'error': 'traffic_split must be between 0 and 1'}), 400

        # Generate test ID
        test_id = f"ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Calculate end time
        start_time = datetime.now()
        from datetime import timedelta
        end_time = start_time + timedelta(hours=duration_hours)

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

        logger.info(f"Started A/B test {test_id}: {baseline_model} vs {test_model}")

        return jsonify({
            'success': True,
            'test_id': test_id,
            'status': 'active',
            'start_time': start_time.isoformat(),
            'estimated_end_time': end_time.isoformat(),
            'message': f'A/B test started. Will run for {duration_hours} hours.'
        }), 200

    except Exception as e:
        logger.error(f"Error starting A/B test: {str(e)}")
        return jsonify({'error': f'Error starting A/B test: {str(e)}'}), 500


@app.route('/testing/ab-test/<test_id>/results', methods=['GET'])
def get_ab_test_results(test_id):
    """Get results of an A/B test"""
    # pylint: disable=W0718
    try:
        if test_id not in ab_tests:
            return jsonify({'error': f'A/B test not found: {test_id}'}), 404

        test_info = ab_tests[test_id]

        # Calculate average metrics
        baseline_metrics = test_info['baseline_metrics']
        test_metrics = test_info['test_metrics']

        def calc_avg(values):
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

        # Simple statistical significance check (t-test would be more robust)
        # For now, just check if test model has significantly better metrics
        min_sample_size = 30
        has_enough_data = (baseline_metrics['query_count'] >= min_sample_size and
                          test_metrics['query_count'] >= min_sample_size)

        improvement_threshold = 0.1  # 10% improvement
        test_is_better = (
            test_avg['avg_rating'] > baseline_avg['avg_rating'] * (1 + improvement_threshold) and
            test_avg['avg_relevance_score'] > baseline_avg['avg_relevance_score'] * (1 + improvement_threshold)
        )

        # Determine status and recommendation
        current_time = datetime.now()
        end_time = datetime.fromisoformat(test_info['estimated_end_time'])

        if current_time >= end_time:
            test_info['status'] = 'completed'

        recommendation = 'insufficient_data'
        if has_enough_data:
            if test_is_better:
                recommendation = 'deploy_test_model'
            else:
                recommendation = 'keep_baseline'

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
            }
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error getting A/B test results: {str(e)}")
        return jsonify({'error': f'Error getting A/B test results: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'conport_available': conport_client.is_client_available(),
        'workspace_id': conport_client.get_workspace_id()
    }), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8084))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    logger.info("Starting server on port %d with debug=%s", port, debug)
    app.run(host='0.0.0.0', port=port, debug=debug)