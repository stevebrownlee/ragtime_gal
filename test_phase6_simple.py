#!/usr/bin/env python3
"""
Phase 6: Simple Testing Suite
Tests Phase 6 components without requiring full database dependencies.
"""

import sys
import os
import time
import logging
from typing import Dict, Any

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_performance_optimizer():
    """Test performance optimization components."""
    logger.info("Testing Performance Optimizer...")

    try:
        from performance_optimizer import QueryCache, PerformanceMonitor, BatchProcessor, MemoryOptimizer

        # Test QueryCache
        cache = QueryCache(max_size=5, ttl_seconds=60)
        test_query = {'query': 'test', 'n_results': 5}

        # Test cache miss
        result = cache.get(test_query)
        assert result is None, "Cache should be empty initially"

        # Test cache set and hit
        cache.set(test_query, {'results': ['test1', 'test2']})
        result = cache.get(test_query)
        assert result is not None, "Cache should return stored result"

        # Test cache stats
        stats = cache.get_stats()
        assert stats['size'] == 1, "Cache should have 1 item"

        logger.info("✅ QueryCache tests passed")

        # Test PerformanceMonitor
        monitor = PerformanceMonitor()
        monitor.record_query_time('test_op', 0.5)
        monitor.record_cache_hit('test_op', True)

        avg_time = monitor.get_average_time('test_op')
        assert avg_time == 0.5, "Average time should be 0.5"

        hit_rate = monitor.get_cache_hit_rate('test_op')
        assert hit_rate == 1.0, "Hit rate should be 100%"

        logger.info("✅ PerformanceMonitor tests passed")

        # Test BatchProcessor
        processor = BatchProcessor(batch_size=3)
        documents = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']

        def mock_processor(batch):
            return [f"processed_{doc}" for doc in batch]

        results = processor.process_documents_in_batches(documents, mock_processor)
        assert len(results) == 5, "Should process all documents"

        logger.info("✅ BatchProcessor tests passed")

        # Test MemoryOptimizer
        large_text = "This is a test sentence. " * 1000
        chunks = MemoryOptimizer.chunk_large_text(large_text, chunk_size=100, overlap=10)
        assert len(chunks) > 1, "Large text should be chunked"

        logger.info("✅ MemoryOptimizer tests passed")

        return True

    except Exception as e:
        logger.error(f"Performance optimizer test failed: {e}")
        return False

def test_error_handler():
    """Test error handling components."""
    logger.info("Testing Error Handler...")

    try:
        from error_handler import ErrorHandler, ErrorCategory, ErrorSeverity, error_handler_decorator

        # Test ErrorHandler
        handler = ErrorHandler("test_errors.log")

        # Test error handling
        test_error = ValueError("Test error")
        error_info = handler.handle_error(
            test_error,
            ErrorCategory.VALIDATION,
            ErrorSeverity.MEDIUM,
            {'test': True}
        )

        assert error_info['error_type'] == 'ValueError', "Error type should be ValueError"
        assert error_info['category'] == 'VALIDATION', "Category should be VALIDATION"
        assert error_info['severity'] == 'MEDIUM', "Severity should be MEDIUM"

        logger.info("✅ ErrorHandler basic tests passed")

        # Test error statistics
        stats = handler.get_error_statistics()
        assert stats['total_errors'] == 1, "Should have 1 error recorded"

        logger.info("✅ ErrorHandler statistics tests passed")

        # Test decorator
        @error_handler_decorator(ErrorCategory.QUERY_PROCESSING, ErrorSeverity.LOW)
        def test_function():
            raise RuntimeError("Test decorator error")

        try:
            test_function()
            assert False, "Function should have raised an exception"
        except RuntimeError:
            pass  # Expected

        logger.info("✅ Error decorator tests passed")

        # Clean up test log file
        if os.path.exists("test_errors.log"):
            os.remove("test_errors.log")

        return True

    except Exception as e:
        logger.error(f"Error handler test failed: {e}")
        return False

def test_documentation_generator():
    """Test documentation generation components."""
    logger.info("Testing Documentation Generator...")

    try:
        from documentation_generator import DocumentationGenerator

        # Test DocumentationGenerator
        doc_gen = DocumentationGenerator("test_docs")

        # Test configuration guide generation
        config_file = doc_gen.generate_configuration_guide()
        assert os.path.exists(config_file), "Configuration guide should be created"

        # Test user guide generation
        user_file = doc_gen.generate_user_guide()
        assert os.path.exists(user_file), "User guide should be created"

        logger.info("✅ Documentation generation tests passed")

        # Clean up test files
        import shutil
        if os.path.exists("test_docs"):
            shutil.rmtree("test_docs")

        return True

    except Exception as e:
        logger.error(f"Documentation generator test failed: {e}")
        return False

def test_integration():
    """Test integration between components."""
    logger.info("Testing Component Integration...")

    try:
        from performance_optimizer import QueryCache, PerformanceMonitor
        from error_handler import ErrorHandler, ErrorCategory, ErrorSeverity

        # Test integrated workflow
        cache = QueryCache()
        monitor = PerformanceMonitor()
        handler = ErrorHandler("integration_test.log")

        # Simulate a query workflow
        query_data = {'query': 'integration test', 'n_results': 10}

        # Simulate cache miss and query execution
        start_time = time.time()

        # Simulate some processing time
        time.sleep(0.1)

        end_time = time.time()
        duration = end_time - start_time

        # Record performance metrics
        monitor.record_query_time('integration_test', duration)
        monitor.record_cache_hit('integration_test', False)

        # Cache the result
        mock_result = {'documents': ['result1', 'result2']}
        cache.set(query_data, mock_result)

        # Test cache hit
        cached_result = cache.get(query_data)
        assert cached_result is not None, "Should get cached result"

        monitor.record_cache_hit('integration_test', True)

        # Test error handling in integration
        try:
            raise ConnectionError("Simulated connection error")
        except Exception as e:
            error_info = handler.handle_error(e, ErrorCategory.DATABASE, ErrorSeverity.HIGH)
            assert error_info['recovery_attempted'], "Recovery should be attempted"

        # Get final statistics
        perf_summary = monitor.get_performance_summary()
        error_stats = handler.get_error_statistics()

        assert 'integration_test' in perf_summary, "Performance data should be recorded"
        assert error_stats['total_errors'] > 0, "Error should be recorded"

        logger.info("✅ Integration tests passed")

        # Clean up
        if os.path.exists("integration_test.log"):
            os.remove("integration_test.log")

        return True

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False

def run_all_tests():
    """Run all Phase 6 tests."""
    logger.info("="*60)
    logger.info("PHASE 6 SIMPLE TEST SUITE")
    logger.info("="*60)

    test_results = {}

    # Run individual component tests
    test_results['performance_optimizer'] = test_performance_optimizer()
    test_results['error_handler'] = test_error_handler()
    test_results['documentation_generator'] = test_documentation_generator()
    test_results['integration'] = test_integration()

    # Generate summary
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)

    passed_tests = 0
    total_tests = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed_tests += 1

    logger.info("-" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("Phase 6: Testing and Optimization - COMPLETED SUCCESSFULLY")
    else:
        logger.info("⚠️  Some tests failed - review the logs above")

    logger.info("="*60)

    return passed_tests == total_tests

def main():
    """Main function to run Phase 6 simple tests."""
    print("Starting Phase 6: Simple Testing Suite")
    print("=" * 50)

    success = run_all_tests()

    if success:
        print("\n✅ Phase 6 implementation completed successfully!")
        print("All core components are working correctly.")
    else:
        print("\n❌ Phase 6 implementation has issues.")
        print("Check the logs above for details.")

    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)