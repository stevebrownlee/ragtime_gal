"""
Document Management API Routes

This module handles document-related operations:
- Document embedding into vector database
- Collection listing and management
- Database purge operations

Migrated from app.py to follow project maturity standards:
- Uses Flask Blueprints for modular route organization
- Integrates with centralized Settings
- Implements structured logging
- Uses migrated core modules
"""

import structlog
import re
from flask import Blueprint, request, jsonify
import chromadb
from langchain_ollama import OllamaEmbeddings

# Import from new structure
from ragtime.config.settings import get_settings
from ragtime.core.embeddings import embed_document

# Initialize structured logger
logger = structlog.get_logger(__name__)

# Create Blueprint
documents_bp = Blueprint('documents', __name__, url_prefix='/api/documents')


def get_vector_db_client():
    """Get ChromaDB client instance."""
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


@documents_bp.route('/embed', methods=['POST'])
def embed_document_route():
    """
    Embed a document into the vector database.

    Expects:
        - file: Document file (PDF or Markdown)
        - collection_name (optional): Target collection name

    Returns:
        JSON response with success/error message
    """
    try:
        if 'file' not in request.files:
            logger.warning("embed_request_missing_file")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.warning("embed_request_empty_filename")
            return jsonify({'error': 'No selected file'}), 400

        # Get collection name from form data (optional)
        collection_name = request.form.get('collection_name', 'langchain')

        # Validate and clean collection name
        if not collection_name or not collection_name.strip():
            collection_name = 'langchain'
        else:
            # Clean the collection name (remove special characters, spaces)
            collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', collection_name.strip())

        logger.info(
            "embedding_document",
            filename=file.filename,
            collection=collection_name
        )

        # Use the migrated embed_document function
        embedded = embed_document(file, collection_name)

        if embedded:
            logger.info(
                "document_embedded_successfully",
                filename=file.filename,
                collection=collection_name
            )
            return jsonify({
                'message': f'File embedded successfully into collection "{collection_name}"',
                'collection_name': collection_name
            }), 200

        logger.error(
            "document_embedding_failed",
            filename=file.filename,
            collection=collection_name
        )
        return jsonify({'error': 'File embedded unsuccessfully'}), 400

    except Exception as e:
        logger.error(
            "embed_route_error",
            error=str(e),
            exc_info=True
        )
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@documents_bp.route('/collections', methods=['GET'])
def list_collections():
    """
    List all available collections in the vector database.

    Returns:
        JSON response with collection information including document counts
    """
    try:
        settings = get_settings()

        logger.info("listing_collections")

        # Connect to ChromaDB
        chroma_client = get_vector_db_client()
        collections = chroma_client.list_collections()
        collection_info = []

        for collection in collections:
            try:
                count = collection.count()
                collection_info.append({
                    "name": collection.name,
                    "document_count": count
                })
                logger.debug(
                    "collection_info",
                    name=collection.name,
                    count=count
                )
            except Exception as e:
                logger.error(
                    "collection_count_error",
                    collection_name=collection.name,
                    error=str(e)
                )
                collection_info.append({
                    "name": collection.name,
                    "document_count": 0
                })

        logger.info(
            "collections_listed",
            total_collections=len(collections)
        )

        return jsonify({
            'collections': collection_info,
            'total_collections': len(collections)
        }), 200

    except Exception as e:
        logger.error(
            "list_collections_error",
            error=str(e),
            exc_info=True
        )
        return jsonify({'error': f'Error getting collections: {str(e)}'}), 500


@documents_bp.route('/purge', methods=['POST'])
def purge_database():
    """
    Delete all documents from all collections in the vector database.

    WARNING: This is a destructive operation that cannot be undone.

    Returns:
        JSON response with purge results
    """
    try:
        logger.warning("database_purge_requested")

        # Connect to ChromaDB
        chroma_client = get_vector_db_client()

        # Get all collections
        collections = chroma_client.list_collections()

        if not collections:
            logger.info("no_collections_to_purge")
            return jsonify({
                'message': 'Database is already empty. No collections to purge.'
            }), 200

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
                deleted_collections.append(
                    f"{collection_name} ({doc_count} documents)"
                )

                logger.info(
                    "collection_deleted",
                    collection_name=collection_name,
                    document_count=doc_count
                )

            except Exception as delete_error:
                logger.error(
                    "collection_deletion_error",
                    collection_name=collection_name,
                    error=str(delete_error),
                    exc_info=True
                )
                return jsonify({
                    'error': f'Error deleting collection {collection_name}: {str(delete_error)}'
                }), 500

        logger.warning(
            "database_purged",
            collections_deleted=len(deleted_collections),
            total_documents=total_documents
        )

        return jsonify({
            'message': f'Database purged successfully. Removed {len(deleted_collections)} collections with {total_documents} total documents.',
            'deleted_collections': deleted_collections
        }), 200

    except Exception as e:
        logger.error(
            "purge_database_error",
            error=str(e),
            exc_info=True
        )
        return jsonify({'error': f'Error purging database: {str(e)}'}), 500


# Helper function to register blueprint
def register_documents_routes(app):
    """
    Register the documents blueprint with the Flask app.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(documents_bp)
    logger.info(
        "documents_blueprint_registered",
        url_prefix=documents_bp.url_prefix
    )