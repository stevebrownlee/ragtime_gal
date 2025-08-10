"""Main application file for the Langchain API server with MCP integration"""
import os
import logging
import secrets
import atexit
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

# Import integration modules
from shared_db import SharedDatabaseManager
from mcp_integration import MCPServerManager

# Import Phase 6 enhancements
from performance_optimizer import get_performance_optimizer, HealthChecker
from error_handler import (
    setup_global_error_handling, get_global_error_handler,
    error_handler_decorator, ErrorCategory, ErrorSeverity
)
from documentation_generator import DocumentationGenerator

# Load environment variables
load_dotenv()

# Set up Phase 6 global error handling
setup_global_error_handling(os.getenv('LOG_FILE', 'logs/ragtime_gal_errors.log'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up constants and directories
TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
BOOK_DIRECTORY = os.getenv('BOOK_DIRECTORY', '.')
MCP_SERVER_NAME = os.getenv('MCP_SERVER_NAME', 'Ragtime Gal MCP Server')

os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('docs', exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
# Set a secret key for session management
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(16))

# Initialize shared database manager
shared_db = SharedDatabaseManager(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_model=os.getenv('EMBEDDING_MODEL', 'mistral'),
    ollama_base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
)

# Initialize MCP server manager
mcp_manager = MCPServerManager(shared_db, BOOK_DIRECTORY)

# Initialize Phase 6 components
performance_optimizer = get_performance_optimizer(shared_db)
health_checker = HealthChecker(performance_optimizer)
error_handler = get_global_error_handler()
doc_generator = DocumentationGenerator('docs')

# Helper function to get vector DB - now uses shared database manager
def get_vector_db():
    """Get the shared vector database instance."""
    try:
        return shared_db.get_database()
    except Exception as e:
        logger.error("Error getting vector database: %s", e)
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

        embedded = embed(file)
        if embedded:
            return jsonify({'message': 'File embedded successfully'}), 200

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
        # Use shared database manager to purge
        deleted_count = shared_db.purge_database()

        if deleted_count > 0:
            return jsonify({'message': f'Database purged successfully. Removed {deleted_count} documents.'}), 200
        else:
            return jsonify({'message': 'Database is already empty. No documents to purge.'}), 200

    except Exception as e:
        logger.error("Error purging database: %s", str(e))
        return jsonify({'error': f'Error purging database: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
@error_handler_decorator(ErrorCategory.SYSTEM_RESOURCE, ErrorSeverity.MEDIUM)
def health_check():
    """Enhanced health check endpoint with Phase 6 monitoring"""
    try:
        # Check database connection
        db_healthy = shared_db.test_connection()

        # Check MCP server status
        mcp_healthy = mcp_manager.is_healthy()

        # Get database stats
        db_stats = shared_db.get_database_stats()

        # Get MCP server status
        mcp_status = mcp_manager.get_status()

        # Get Phase 6 health information
        comprehensive_health = health_checker.get_comprehensive_health()
        performance_stats = performance_optimizer.get_performance_stats()
        error_stats = error_handler.get_error_statistics()

        overall_status = 'healthy' if db_healthy and mcp_healthy else 'degraded'

        return jsonify({
            'status': overall_status,
            'database': {
                'healthy': db_healthy,
                'stats': db_stats
            },
            'mcp_server': {
                'healthy': mcp_healthy,
                'status': mcp_status
            },
            'performance': performance_stats,
            'system_health': comprehensive_health,
            'error_statistics': {
                'total_errors': error_stats['total_errors'],
                'categories': error_stats['categories'],
                'recent_errors_count': len(error_stats['recent_errors'])
            }
        }), 200

    except Exception as e:
        logger.error("Error in health check: %s", str(e))
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/mcp-status', methods=['GET'])
def mcp_status():
    """Get detailed MCP server status"""
    try:
        status = mcp_manager.get_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error("Error getting MCP status: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/performance-stats', methods=['GET'])
@error_handler_decorator(ErrorCategory.SYSTEM_RESOURCE, ErrorSeverity.LOW)
def performance_stats():
    """Get detailed performance statistics"""
    try:
        stats = performance_optimizer.get_performance_stats()
        return jsonify(stats), 200
    except Exception as e:
        logger.error("Error getting performance stats: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/error-statistics', methods=['GET'])
@error_handler_decorator(ErrorCategory.SYSTEM_RESOURCE, ErrorSeverity.LOW)
def error_statistics():
    """Get detailed error statistics"""
    try:
        stats = error_handler.get_error_statistics()
        return jsonify(stats), 200
    except Exception as e:
        logger.error("Error getting error statistics: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/clear-cache', methods=['POST'])
@error_handler_decorator(ErrorCategory.SYSTEM_RESOURCE, ErrorSeverity.MEDIUM)
def clear_cache():
    """Clear performance cache"""
    try:
        performance_optimizer.clear_cache()
        return jsonify({'message': 'Cache cleared successfully'}), 200
    except Exception as e:
        logger.error("Error clearing cache: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/clear-error-history', methods=['POST'])
@error_handler_decorator(ErrorCategory.SYSTEM_RESOURCE, ErrorSeverity.MEDIUM)
def clear_error_history():
    """Clear error history"""
    try:
        error_handler.clear_error_history()
        return jsonify({'message': 'Error history cleared successfully'}), 200
    except Exception as e:
        logger.error("Error clearing error history: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/generate-docs', methods=['POST'])
@error_handler_decorator(ErrorCategory.SYSTEM_RESOURCE, ErrorSeverity.MEDIUM)
def generate_documentation():
    """Generate comprehensive documentation"""
    try:
        generated_files = doc_generator.generate_all_documentation(mcp_manager, app)
        return jsonify({
            'message': 'Documentation generated successfully',
            'files': generated_files
        }), 200
    except Exception as e:
        logger.error("Error generating documentation: %s", str(e))
        return jsonify({'error': str(e)}), 500

def start_mcp_server():
    """Start the MCP server and Phase 6 components during Flask startup"""
    try:
        logger.info("Starting MCP server...")
        if mcp_manager.start():
            logger.info("MCP server started successfully")
        else:
            logger.error("Failed to start MCP server")

        # Start health monitoring
        logger.info("Starting health monitoring...")
        health_checker.start_monitoring()
        logger.info("Health monitoring started")

        # Log startup completion
        logger.info("Phase 6 enhanced RAG+MCP server startup completed")

    except Exception as e:
        error_handler.handle_error(e, ErrorCategory.MCP_SERVER, ErrorSeverity.CRITICAL)
        logger.error("Error starting MCP server: %s", str(e))

def shutdown_mcp_server():
    """Shutdown the MCP server and Phase 6 components during Flask shutdown"""
    try:
        logger.info("Shutting down Phase 6 enhanced server...")

        # Stop health monitoring
        health_checker.stop_monitoring()
        logger.info("Health monitoring stopped")

        # Shutdown MCP server
        logger.info("Shutting down MCP server...")
        if mcp_manager.stop():
            logger.info("MCP server stopped successfully")
        else:
            logger.error("Failed to stop MCP server gracefully")

        # Generate final performance report
        try:
            final_stats = performance_optimizer.get_performance_stats()
            logger.info("Final performance stats: %s", final_stats)
        except Exception as stats_error:
            logger.warning("Could not generate final performance stats: %s", stats_error)

        logger.info("Phase 6 enhanced server shutdown completed")

    except Exception as e:
        error_handler.handle_error(e, ErrorCategory.MCP_SERVER, ErrorSeverity.HIGH)
        logger.error("Error stopping MCP server: %s", str(e))

# Register shutdown handler
atexit.register(shutdown_mcp_server)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8084))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    logger.info("Starting unified RAG + MCP server on port %d with debug=%s", port, debug)

    # Start MCP server
    start_mcp_server()

    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        shutdown_mcp_server()