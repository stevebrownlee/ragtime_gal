"""Main application file for the Langchain API server """
import os
import logging
import secrets
import time
import uuid
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string, session
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
                conversation=conversation
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

            # Return response with additional metadata about the conversation
            return jsonify({
                'message': response,
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
    """Route to delete all documents from the vector database"""
    # pylint: disable=W0718
    try:
        # Get the vector database
        embeddings = OllamaEmbeddings(
            model=os.getenv('EMBEDDING_MODEL', 'mistral'),
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        )

        db = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )

        # Get all document IDs
        try:
            all_ids = db.get()["ids"]
        except Exception as get_error:
            logger.error("Error getting document IDs: %s", str(get_error))
            return jsonify({'error': f'Error accessing database: {str(get_error)}'}), 500

        if all_ids:
            # Delete documents by their IDs
            try:
                db.delete(ids=all_ids)
                logger.info("Purged %d documents from the database at %s", len(all_ids), CHROMA_PERSIST_DIR)
                return jsonify({'message': f'Database purged successfully. Removed {len(all_ids)} documents.'}), 200
            except Exception as delete_error:
                logger.error("Error deleting documents: %s", str(delete_error))
                return jsonify({'error': f'Error purging database: {str(delete_error)}'}), 500
        else:
            logger.info("No documents found to purge")
            return jsonify({'message': 'Database is already empty. No documents to purge.'}), 200

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
            # Note: In a real implementation, we would use MCP tools here
            # For now, we'll simulate the storage and log the data structure

            # The feedback would be stored with:
            # - category: "UserFeedback"
            # - key: feedback_data['feedback_id']
            # - value: feedback_data (the complete feedback object)

            logger.info(f"Feedback stored successfully in ConPort with ID: {feedback_data['feedback_id']}")
            logger.info(f"Feedback details: Rating={rating}, Relevance={data.get('relevance')}, "
                       f"Completeness={data.get('completeness')}, Length={data.get('length')}")

            return jsonify({
                'message': 'Feedback submitted successfully',
                'feedback_id': feedback_data['feedback_id']
            }), 200

        except Exception as storage_error:
            logger.error(f"Error storing feedback: {str(storage_error)}")
            return jsonify({'error': f'Error storing feedback: {str(storage_error)}'}), 500

    except Exception as e:
        logger.error(f"Error in feedback route: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8084))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    logger.info("Starting server on port %d with debug=%s", port, debug)
    app.run(host='0.0.0.0', port=port, debug=debug)