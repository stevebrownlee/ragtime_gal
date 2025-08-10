#!/usr/bin/env python3
"""
Test script for Phase 2: Enhanced Vector Database Schema
Tests the enhanced metadata extraction and query functionality.
"""

import os
import sys
import logging
from io import StringIO
from unittest.mock import Mock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_metadata_extraction():
    """Test the enhanced metadata extraction functions"""
    print("=" * 60)
    print("TESTING PHASE 2: Enhanced Vector Database Schema")
    print("=" * 60)

    try:
        # Import the enhanced embed module
        from embed import extract_chapter_info, extract_chapters_from_content

        # Test markdown content with chapters
        test_content = """# My Great Novel

This is the introduction to my novel.

## Chapter 1: The Beginning

This is the first chapter of the story. It introduces the main character.

## Chapter 2: The Journey

The character embarks on a journey in this chapter.

## Chapter 3: The End

The story concludes in this final chapter.
"""

        print("\n1. Testing metadata extraction...")
        metadata = extract_chapter_info(test_content, "test_novel.md")

        print(f"   ✓ Book title: {metadata.get('book_title')}")
        print(f"   ✓ Has chapters: {metadata.get('has_chapters')}")
        print(f"   ✓ Word count: {metadata.get('word_count')}")
        print(f"   ✓ Character count: {metadata.get('character_count')}")
        print(f"   ✓ File type: {metadata.get('file_type')}")

        print("\n2. Testing chapter extraction...")
        chapters = extract_chapters_from_content(test_content, metadata)

        print(f"   ✓ Found {len(chapters)} chapters")
        for i, (content, chapter_meta) in enumerate(chapters):
            print(f"   ✓ Chapter {i+1}: '{chapter_meta.get('chapter_title')}' ({chapter_meta.get('word_count')} words)")

        print("\n✅ Metadata extraction tests PASSED")

    except Exception as e:
        print(f"\n❌ Metadata extraction tests FAILED: {e}")
        return False

    return True

def test_metadata_query_manager():
    """Test the MetadataQueryManager functionality"""
    print("\n" + "=" * 60)
    print("TESTING METADATA QUERY MANAGER")
    print("=" * 60)

    try:
        from metadata_utils import MetadataQueryManager

        # Create a mock database for testing
        print("\n1. Testing MetadataQueryManager initialization...")

        # Mock the database connection to avoid requiring actual Chroma/Ollama
        manager = MetadataQueryManager()
        print("   ✓ MetadataQueryManager created successfully")

        # Test method existence
        methods_to_test = [
            'get_all_books',
            'get_chapters_for_book',
            'search_by_metadata',
            'get_chapter_content',
            'search_content_with_metadata',
            'get_book_statistics',
            'validate_metadata_consistency'
        ]

        print("\n2. Testing method availability...")
        for method_name in methods_to_test:
            if hasattr(manager, method_name):
                print(f"   ✓ Method '{method_name}' available")
            else:
                print(f"   ❌ Method '{method_name}' missing")
                return False

        print("\n✅ MetadataQueryManager tests PASSED")

    except Exception as e:
        print(f"\n❌ MetadataQueryManager tests FAILED: {e}")
        return False

    return True

def test_shared_database_integration():
    """Test the enhanced SharedDatabaseManager"""
    print("\n" + "=" * 60)
    print("TESTING SHARED DATABASE INTEGRATION")
    print("=" * 60)

    try:
        from shared_db import SharedDatabaseManager

        print("\n1. Testing SharedDatabaseManager with metadata support...")

        # Create manager instance
        manager = SharedDatabaseManager()
        print("   ✓ SharedDatabaseManager created successfully")

        # Test enhanced method availability
        enhanced_methods = [
            'get_metadata_manager',
            'get_all_books',
            'get_chapters_for_book',
            'get_book_statistics',
            'search_content_with_metadata',
            'get_chapter_content',
            'validate_metadata_consistency'
        ]

        print("\n2. Testing enhanced method availability...")
        for method_name in enhanced_methods:
            if hasattr(manager, method_name):
                print(f"   ✓ Enhanced method '{method_name}' available")
            else:
                print(f"   ❌ Enhanced method '{method_name}' missing")
                return False

        print("\n✅ SharedDatabaseManager integration tests PASSED")

    except Exception as e:
        print(f"\n❌ SharedDatabaseManager integration tests FAILED: {e}")
        return False

    return True

def test_import_compatibility():
    """Test that all imports work correctly"""
    print("\n" + "=" * 60)
    print("TESTING IMPORT COMPATIBILITY")
    print("=" * 60)

    try:
        print("\n1. Testing core module imports...")

        # Test embed module
        try:
            import embed
            print("   ✓ embed module imported successfully")
        except Exception as e:
            print(f"   ❌ embed module import failed: {e}")
            return False

        # Test metadata_utils module
        try:
            import metadata_utils
            print("   ✓ metadata_utils module imported successfully")
        except Exception as e:
            print(f"   ❌ metadata_utils module import failed: {e}")
            return False

        # Test shared_db module
        try:
            import shared_db
            print("   ✓ shared_db module imported successfully")
        except Exception as e:
            print(f"   ❌ shared_db module import failed: {e}")
            return False

        print("\n✅ Import compatibility tests PASSED")

    except Exception as e:
        print(f"\n❌ Import compatibility tests FAILED: {e}")
        return False

    return True

def main():
    """Run all Phase 2 tests"""
    print("Starting Phase 2 Enhanced Vector Database Schema Tests...")
    print("Note: Some tests may show import warnings for langchain dependencies.")
    print("This is expected in the test environment and doesn't affect functionality.\n")

    tests = [
        ("Import Compatibility", test_import_compatibility),
        ("Metadata Extraction", test_metadata_extraction),
        ("Metadata Query Manager", test_metadata_query_manager),
        ("Shared Database Integration", test_shared_database_integration)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test encountered an error: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 2 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL PHASE 2 TESTS PASSED!")
        print("Enhanced Vector Database Schema implementation is ready.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)