"""Main application file for the Langchain API server """
import os
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
from embed import embed
from query import query
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from template import load_html_template

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

# Helper function to get vector DB - replaces the imported get_vector_db
def get_vector_db():
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
        logger.error(f"Error initializing vector database: {e}")
        raise

@app.route('/', methods=['GET'])
def index():
    """Route to display the upload form"""
    template_html = load_html_template()
    return render_template_string(template_html)

@app.route('/embed', methods=['POST'])
def route_embed():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        embedded = embed(file)
        if embedded:
            return jsonify({'message': 'File embedded successfully'}), 200

        return jsonify({'error': 'File embedded unsuccessfully'}), 400

    except Exception as e:
        logger.error(f"Error in embedding route: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
def route_query():
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

        # Call the query function with parameters
        response = query(
            data.get('query'),
            template_name=template_name,
            temperature=temperature
        )

        if response:
            return jsonify({'message': response}), 200

        return jsonify({'error': 'No response generated'}), 400

    except Exception as e:
        logger.error("Error in query route: %s", e)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/purge', methods=['POST'])
def purge_database():
    """Route to delete all documents from the vector database"""
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
        all_ids = db.get()["ids"]

        if all_ids:
            # Delete documents by their IDs
            db.delete(ids=all_ids)
            logger.info(f"Purged {len(all_ids)} documents from the database at {CHROMA_PERSIST_DIR}")
            return jsonify({'message': f'Database purged successfully. Removed {len(all_ids)} documents.'}), 200
        else:
            logger.info("No documents found to purge")
            return jsonify({'message': 'Database is already empty. No documents to purge.'}), 200

    except Exception as e:
        logger.error(f"Error purging database: {str(e)}")
        return jsonify({'error': f'Error purging database: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting server on port {port} with debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)