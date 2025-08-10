#!/usr/bin/env python3
"""
Simple Phase 5 Test: Content Management Tools

This script tests the Phase 5 content management functionality by directly
testing the underlying methods rather than the MCP tool interface.
"""

import sys
import os
import logging
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared_db import SharedDatabaseManager
from mcp_integration import MCPServerManager
from metadata_utils import MetadataQueryManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_phase5_simple():
    """Simple test of Phase 5 content management functionality."""

    print("=" * 80)
    print("PHASE 5 CONTENT MANAGEMENT TOOLS - SIMPLE TEST")
    print("=" * 80)

    try:
        # Initialize components
        print("\n1. Initializing components...")
        shared_db = SharedDatabaseManager()
        mcp_manager = MCPServerManager(shared_db)
        metadata_manager = MetadataQueryManager()

        # Test database connection
        if not shared_db.test_connection():
            print("❌ Database connection failed!")
            return False

        print("✅ Database connection successful")

        # Get initial database state
        db = shared_db.get_database()
        initial_data = db.get()
        initial_count = len(initial_data.get("ids", []))
        print(f"📊 Initial database contains {initial_count} documents")

        # Test data
        test_book_title = "Phase 5 Test Book"
        test_content = """# Chapter 1: Test Chapter

This is a test chapter for Phase 5 content management tools.

It contains multiple paragraphs to test the chunking functionality.
The content is designed to be realistic but concise for testing purposes.

"This is some dialogue," said the character.

And this is the end of the test chapter."""

        # Test 1: Content Chunking
        print("\n" + "="*60)
        print("TEST 1: CONTENT CHUNKING")
        print("="*60)

        print("📝 Testing content chunking...")
        chunks = mcp_manager._chunk_content(test_content, test_book_title, 1, "Test Chapter")

        if chunks:
            print(f"✅ Content chunked successfully")
            print(f"   - Number of chunks: {len(chunks)}")
            print(f"   - Total words: {sum(chunk['metadata']['word_count'] for chunk in chunks)}")
            print(f"   - Sample chunk ID: {chunks[0]['id']}")
        else:
            print("❌ Content chunking failed")
            return False

        # Test 2: Add Content to Database
        print("\n" + "="*60)
        print("TEST 2: ADD CONTENT TO DATABASE")
        print("="*60)

        print("📝 Adding chunks to database...")
        try:
            documents = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            ids = [chunk["id"] for chunk in chunks]

            shared_db.add_documents(texts=documents, metadatas=metadatas, ids=ids)
            print("✅ Content added to database successfully")
        except Exception as e:
            print(f"❌ Failed to add content to database: {e}")
            return False

        # Test 3: Retrieve Content
        print("\n" + "="*60)
        print("TEST 3: RETRIEVE CONTENT")
        print("="*60)

        print("📝 Retrieving chapter content...")
        retrieved_chunks = metadata_manager.get_chapter_content(test_book_title, 1)

        if retrieved_chunks:
            print(f"✅ Content retrieved successfully")
            print(f"   - Retrieved chunks: {len(retrieved_chunks)}")
            print(f"   - Chapter title: {retrieved_chunks[0]['metadata'].get('chapter_title')}")
        else:
            print("❌ Failed to retrieve content")
            return False

        # Test 4: Delete Content
        print("\n" + "="*60)
        print("TEST 4: DELETE CONTENT")
        print("="*60)

        print("📝 Testing content deletion...")
        delete_result = mcp_manager._delete_chapter_content(test_book_title, 1)

        if delete_result.get("status") == "success":
            print("✅ Content deleted successfully")
            print(f"   - Chunks deleted: {delete_result.get('chunks_deleted')}")
        else:
            print(f"❌ Failed to delete content: {delete_result.get('error')}")
            return False

        # Test 5: Verify Deletion
        print("\n" + "="*60)
        print("TEST 5: VERIFY DELETION")
        print("="*60)

        print("📝 Verifying content was deleted...")
        remaining_chunks = metadata_manager.get_chapter_content(test_book_title, 1)

        if not remaining_chunks:
            print("✅ Content deletion verified - no chunks remain")
        else:
            print(f"❌ Deletion verification failed - {len(remaining_chunks)} chunks still exist")
            return False

        # Test 6: Check Database State
        print("\n" + "="*60)
        print("TEST 6: FINAL DATABASE STATE")
        print("="*60)

        final_data = db.get()
        final_count = len(final_data.get("ids", []))
        print(f"📊 Final database contains {final_count} documents")
        print(f"📈 Net change: {final_count - initial_count} documents")

        if final_count == initial_count:
            print("✅ Database state restored to initial state")
        else:
            print(f"⚠️ Database state changed by {final_count - initial_count} documents")

        print("\n" + "="*80)
        print("✅ PHASE 5 SIMPLE TEST COMPLETED SUCCESSFULLY!")
        print("="*80)

        print(f"\n📋 SUMMARY:")
        print(f"   ✅ Content Chunking - Working")
        print(f"   ✅ Database Addition - Working")
        print(f"   ✅ Content Retrieval - Working")
        print(f"   ✅ Content Deletion - Working")
        print(f"   ✅ Database Integration - Working")

        return True

    except Exception as e:
        print(f"\n❌ PHASE 5 SIMPLE TEST FAILED: {e}")
        logger.exception("Phase 5 simple test failed with exception")
        return False

if __name__ == "__main__":
    success = test_phase5_simple()
    sys.exit(0 if success else 1)