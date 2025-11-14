"""
Main application file for the Ragtime Gal RAG server.

This file initializes the Flask application and registers all API Blueprint modules.
Route implementations have been migrated to ragtime.api package for better organization.
"""

import os
import logging
import secrets
from dotenv import load_dotenv
from flask import Flask, render_template_string

# Import template loader
from template import load_html_template

# Import ConPort client
from conport_client import initialize_conport_client

# Import monitoring dashboard
from monitoring_dashboard import (
    create_monitoring_blueprint,
    start_monitoring
)

# Import API Blueprint registration
from ragtime.api import register_all_routes

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
logger.info("ConPort client initialized with workspace: %s", os.getcwd())

# Register monitoring dashboard blueprint
app.register_blueprint(create_monitoring_blueprint())
logger.info("Monitoring dashboard registered - accessible at /monitoring")

# Start monitoring system (collect metrics every 30 seconds)
start_monitoring(interval=30)

# Register all API routes from Blueprint modules
register_all_routes(app)
logger.info("All API Blueprint routes registered")


@app.route('/', methods=['GET'])
def index():
    """
    Root route - display the web UI for document upload and querying.

    Returns:
        Rendered HTML template with the application interface
    """
    template_html = load_html_template()
    return render_template_string(template_html)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8084))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    logger.info("=" * 60)
    logger.info("Starting Ragtime Gal RAG Server")
    logger.info("=" * 60)
    logger.info("Server port: %d", port)
    logger.info("Debug mode: %s", debug)
    logger.info("Workspace: %s", os.getcwd())
    logger.info("=" * 60)

    app.run(host='0.0.0.0', port=port, debug=debug)