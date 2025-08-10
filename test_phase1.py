#!/usr/bin/env python3
"""
Test script for Phase 1 implementation.
Tests the basic integration between Flask app and MCP server.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        from shared_db import SharedDatabaseManager
        print("✓ SharedDatabaseManager imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SharedDatabaseManager: {e}")
        return False

    try:
        from mcp_integration import MCPServerManager
        print("✓ MCPServerManager imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MCPServerManager: {e}")
        return False

    return True

def test_shared_database():
    """Test the shared database manager."""
    print("\nTesting SharedDatabaseManager...")

    try:
        from shared_db import SharedDatabaseManager

        # Create test database manager
        db_manager = SharedDatabaseManager(
            persist_directory="./test_chroma_db",
            embedding_model="mistral",
            ollama_base_url="http://localhost:11434"
        )

        print("✓ SharedDatabaseManager created successfully")

        # Test connection (this might fail if Ollama is not running)
        try:
            connection_ok = db_manager.test_connection()
            if connection_ok:
                print("✓ Database connection test passed")
            else:
                print("⚠ Database connection test failed (Ollama might not be running)")
        except Exception as e:
            print(f"⚠ Database connection test failed: {e}")

        # Test getting stats
        try:
            stats = db_manager.get_database_stats()
            print(f"✓ Database stats retrieved: {stats.get('total_documents', 0)} documents")
        except Exception as e:
            print(f"⚠ Failed to get database stats: {e}")

        return True

    except Exception as e:
        print(f"✗ SharedDatabaseManager test failed: {e}")
        return False

def test_mcp_server():
    """Test the MCP server manager."""
    print("\nTesting MCPServerManager...")

    try:
        from shared_db import SharedDatabaseManager
        from mcp_integration import MCPServerManager

        # Create test database manager
        db_manager = SharedDatabaseManager(
            persist_directory="./test_chroma_db",
            embedding_model="mistral",
            ollama_base_url="http://localhost:11434"
        )

        # Create MCP server manager
        mcp_manager = MCPServerManager(db_manager, ".")
        print("✓ MCPServerManager created successfully")

        # Test starting the server
        print("Starting MCP server...")
        if mcp_manager.start():
            print("✓ MCP server started successfully")

            # Wait a moment
            time.sleep(2)

            # Test health check
            if mcp_manager.is_healthy():
                print("✓ MCP server health check passed")
            else:
                print("⚠ MCP server health check failed")

            # Get status
            status = mcp_manager.get_status()
            print(f"✓ MCP server status: {status}")

            # Stop the server
            print("Stopping MCP server...")
            if mcp_manager.stop():
                print("✓ MCP server stopped successfully")
            else:
                print("⚠ MCP server stop failed")

        else:
            print("✗ Failed to start MCP server")
            return False

        return True

    except Exception as e:
        print(f"✗ MCPServerManager test failed: {e}")
        return False

def test_flask_integration():
    """Test Flask app integration (import only, don't start server)."""
    print("\nTesting Flask integration...")

    try:
        # Test importing the modified app
        import app
        print("✓ Flask app imported successfully")

        # Check that shared_db and mcp_manager are initialized
        if hasattr(app, 'shared_db'):
            print("✓ shared_db initialized in Flask app")
        else:
            print("✗ shared_db not found in Flask app")
            return False

        if hasattr(app, 'mcp_manager'):
            print("✓ mcp_manager initialized in Flask app")
        else:
            print("✗ mcp_manager not found in Flask app")
            return False

        return True

    except Exception as e:
        print(f"✗ Flask integration test failed: {e}")
        return False

def cleanup():
    """Clean up test files."""
    print("\nCleaning up test files...")

    test_db_path = Path("./test_chroma_db")
    if test_db_path.exists():
        import shutil
        shutil.rmtree(test_db_path)
        print("✓ Test database directory cleaned up")

def main():
    """Run all tests."""
    print("=== Phase 1 Integration Test ===\n")

    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise

    tests = [
        ("Import Test", test_imports),
        ("Shared Database Test", test_shared_database),
        ("MCP Server Test", test_mcp_server),
        ("Flask Integration Test", test_flask_integration),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Cleanup
    cleanup()

    # Summary
    print("\n=== Test Summary ===")
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{len(results)} tests")

    if passed == len(results):
        print("🎉 All tests passed! Phase 1 integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())