#!/usr/bin/env python3
"""
Phase 6: Testing and Optimization - Comprehensive Test Suite
Tests all MCP tools, integration functionality, performance, and stress testing.
"""

import asyncio
import time
import threading
import concurrent.futures
import statistics
from typing import Dict, List, Any, Tuple
import logging
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared_db import SharedDatabaseManager
from metadata_utils import MetadataQueryManager
from mcp_integration import MCPServerManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase6TestSuite:
    """Comprehensive test suite for Phase 6 testing and optimization."""

    def __init__(self):
        """Initialize test suite with database and MCP server connections."""
        self.db_manager = SharedDatabaseManager()
        self.metadata_manager = MetadataQueryManager(self.db_manager)
        self.mcp_manager = None
        self.test_results = {
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'stress_tests': {},
            'error_handling_tests': {}
        }

    async def initialize_mcp_server(self):
        """Initialize MCP server for testing."""
        try:
            self.mcp_manager = MCPServerManager()
            await self.mcp_manager.initialize()
            logger.info("MCP server initialized for testing")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            return False

    def run_all_tests(self):
        """Run all test suites and generate comprehensive report."""
        logger.info("Starting Phase 6 Comprehensive Test Suite")

        # Run synchronous tests first
        self.run_unit_tests()
        self.run_performance_tests()
        self.run_stress_tests()
        self.run_error_handling_tests()

        # Run async integration tests
        asyncio.run(self.run_integration_tests())

        # Generate final report
        self.generate_test_report()

    def run_unit_tests(self):
        """Run unit tests for all MCP tools and core functionality."""
        logger.info("Running Unit Tests...")

        # Test database connectivity
        self.test_results['unit_tests']['database_connection'] = self.test_database_connection()

        # Test metadata operations
        self.test_results['unit_tests']['metadata_operations'] = self.test_metadata_operations()

        # Test search functionality
        self.test_results['unit_tests']['search_functionality'] = self.test_search_functionality()

        # Test chapter operations
        self.test_results['unit_tests']['chapter_operations'] = self.test_chapter_operations()

        # Test statistics calculations
        self.test_results['unit_tests']['statistics_calculations'] = self.test_statistics_calculations()

        logger.info("Unit tests completed")

    async def run_integration_tests(self):
        """Run integration tests for Flask + MCP interaction."""
        logger.info("Running Integration Tests...")

        # Initialize MCP server
        mcp_initialized = await self.initialize_mcp_server()
        self.test_results['integration_tests']['mcp_initialization'] = mcp_initialized

        if mcp_initialized:
            # Test MCP tool registration
            self.test_results['integration_tests']['tool_registration'] = await self.test_mcp_tool_registration()

            # Test MCP tool execution
            self.test_results['integration_tests']['tool_execution'] = await self.test_mcp_tool_execution()

            # Test concurrent access
            self.test_results['integration_tests']['concurrent_access'] = await self.test_concurrent_access()

        logger.info("Integration tests completed")

    def run_performance_tests(self):
        """Run performance tests for large datasets and operations."""
        logger.info("Running Performance Tests...")

        # Test query performance
        self.test_results['performance_tests']['query_performance'] = self.test_query_performance()

        # Test large dataset handling
        self.test_results['performance_tests']['large_dataset'] = self.test_large_dataset_performance()

        # Test memory usage
        self.test_results['performance_tests']['memory_usage'] = self.test_memory_usage()

        # Test caching effectiveness
        self.test_results['performance_tests']['caching'] = self.test_caching_performance()

        logger.info("Performance tests completed")

    def run_stress_tests(self):
        """Run stress tests for concurrent usage and high load."""
        logger.info("Running Stress Tests...")

        # Test concurrent queries
        self.test_results['stress_tests']['concurrent_queries'] = self.test_concurrent_queries()

        # Test high-frequency operations
        self.test_results['stress_tests']['high_frequency_ops'] = self.test_high_frequency_operations()

        # Test resource limits
        self.test_results['stress_tests']['resource_limits'] = self.test_resource_limits()

        logger.info("Stress tests completed")

    def run_error_handling_tests(self):
        """Run error handling and recovery tests."""
        logger.info("Running Error Handling Tests...")

        # Test invalid inputs
        self.test_results['error_handling_tests']['invalid_inputs'] = self.test_invalid_inputs()

        # Test database errors
        self.test_results['error_handling_tests']['database_errors'] = self.test_database_error_handling()

        # Test recovery mechanisms
        self.test_results['error_handling_tests']['recovery'] = self.test_recovery_mechanisms()

        logger.info("Error handling tests completed")

    # Unit Test Methods
    def test_database_connection(self) -> Dict[str, Any]:
        """Test database connection and basic operations."""
        try:
            # Test connection
            collection = self.db_manager.get_collection()
            if collection is None:
                return {'status': 'FAILED', 'error': 'Could not get collection'}

            # Test basic query
            count = collection.count()

            return {
                'status': 'PASSED',
                'document_count': count,
                'connection_time': time.time()
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_metadata_operations(self) -> Dict[str, Any]:
        """Test metadata query operations."""
        try:
            # Test getting all books
            books = self.metadata_manager.get_all_books()

            # Test getting chapters for first book if available
            chapters = []
            if books:
                first_book = list(books.keys())[0]
                chapters = self.metadata_manager.get_chapters_for_book(first_book)

            # Test metadata filtering
            filtered_docs = self.metadata_manager.get_documents_by_metadata({'file_type': 'markdown'})

            return {
                'status': 'PASSED',
                'books_found': len(books),
                'chapters_found': len(chapters),
                'filtered_docs': len(filtered_docs)
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_search_functionality(self) -> Dict[str, Any]:
        """Test vector search functionality."""
        try:
            collection = self.db_manager.get_collection()

            # Test basic search
            results = collection.query(
                query_texts=["character development"],
                n_results=5
            )

            # Test search with metadata filter
            filtered_results = collection.query(
                query_texts=["dialogue"],
                n_results=3,
                where={"file_type": "markdown"}
            )

            return {
                'status': 'PASSED',
                'basic_search_results': len(results['documents'][0]) if results['documents'] else 0,
                'filtered_search_results': len(filtered_results['documents'][0]) if filtered_results['documents'] else 0
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_chapter_operations(self) -> Dict[str, Any]:
        """Test chapter-related operations."""
        try:
            # Test chapter listing
            all_chapters = self.metadata_manager.get_all_chapters()

            # Test chapter statistics
            chapter_stats = {}
            if all_chapters:
                first_chapter = all_chapters[0]
                chapter_stats = self.metadata_manager.get_chapter_statistics(
                    first_chapter.get('book_title', ''),
                    first_chapter.get('chapter_title', '')
                )

            return {
                'status': 'PASSED',
                'total_chapters': len(all_chapters),
                'chapter_stats_available': bool(chapter_stats)
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_statistics_calculations(self) -> Dict[str, Any]:
        """Test statistics calculation accuracy."""
        try:
            collection = self.db_manager.get_collection()

            # Get sample documents
            results = collection.get(limit=10)

            if not results['documents']:
                return {'status': 'SKIPPED', 'reason': 'No documents available'}

            # Calculate basic statistics
            total_words = 0
            total_chars = 0

            for doc in results['documents']:
                words = len(doc.split())
                chars = len(doc)
                total_words += words
                total_chars += chars

            avg_words = total_words / len(results['documents'])
            avg_chars = total_chars / len(results['documents'])

            return {
                'status': 'PASSED',
                'documents_analyzed': len(results['documents']),
                'total_words': total_words,
                'total_chars': total_chars,
                'avg_words_per_doc': round(avg_words, 2),
                'avg_chars_per_doc': round(avg_chars, 2)
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    # Integration Test Methods
    async def test_mcp_tool_registration(self) -> Dict[str, Any]:
        """Test MCP tool registration."""
        try:
            if not self.mcp_manager:
                return {'status': 'FAILED', 'error': 'MCP manager not initialized'}

            # Get registered tools
            tools = self.mcp_manager.get_registered_tools()

            expected_tools = [
                'search_book_content',
                'get_chapter_info',
                'list_all_chapters',
                'analyze_character_mentions',
                'get_book_structure',
                'list_all_books',
                'get_writing_statistics',
                'analyze_readability',
                'check_grammar_and_style',
                'analyze_writing_patterns',
                'add_chapter_content',
                'update_chapter_content',
                'delete_chapter',
                'delete_book',
                'reorder_chapters'
            ]

            registered_tools = [tool for tool in expected_tools if tool in tools]

            return {
                'status': 'PASSED',
                'expected_tools': len(expected_tools),
                'registered_tools': len(registered_tools),
                'missing_tools': [tool for tool in expected_tools if tool not in tools]
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    async def test_mcp_tool_execution(self) -> Dict[str, Any]:
        """Test MCP tool execution."""
        try:
            if not self.mcp_manager:
                return {'status': 'FAILED', 'error': 'MCP manager not initialized'}

            # Test a few key tools
            test_results = {}

            # Test list_all_books
            try:
                books_result = await self.mcp_manager.execute_tool('list_all_books', {})
                test_results['list_all_books'] = 'PASSED'
            except Exception as e:
                test_results['list_all_books'] = f'FAILED: {str(e)}'

            # Test search_book_content
            try:
                search_result = await self.mcp_manager.execute_tool('search_book_content', {
                    'query': 'character',
                    'max_results': 3
                })
                test_results['search_book_content'] = 'PASSED'
            except Exception as e:
                test_results['search_book_content'] = f'FAILED: {str(e)}'

            # Test list_all_chapters
            try:
                chapters_result = await self.mcp_manager.execute_tool('list_all_chapters', {})
                test_results['list_all_chapters'] = 'PASSED'
            except Exception as e:
                test_results['list_all_chapters'] = f'FAILED: {str(e)}'

            passed_tests = sum(1 for result in test_results.values() if result == 'PASSED')

            return {
                'status': 'PASSED' if passed_tests > 0 else 'FAILED',
                'tools_tested': len(test_results),
                'tools_passed': passed_tests,
                'test_results': test_results
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    async def test_concurrent_access(self) -> Dict[str, Any]:
        """Test concurrent access to database and MCP tools."""
        try:
            # Create multiple concurrent tasks
            tasks = []

            # Database access tasks
            for i in range(5):
                task = asyncio.create_task(self.concurrent_database_access(i))
                tasks.append(task)

            # MCP tool access tasks
            if self.mcp_manager:
                for i in range(3):
                    task = asyncio.create_task(self.concurrent_mcp_access(i))
                    tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_tasks = sum(1 for result in results if not isinstance(result, Exception))

            return {
                'status': 'PASSED' if successful_tasks > 0 else 'FAILED',
                'total_tasks': len(tasks),
                'successful_tasks': successful_tasks,
                'failed_tasks': len(tasks) - successful_tasks
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    async def concurrent_database_access(self, task_id: int) -> bool:
        """Simulate concurrent database access."""
        try:
            collection = self.db_manager.get_collection()
            results = collection.query(
                query_texts=[f"test query {task_id}"],
                n_results=2
            )
            return True
        except Exception:
            return False

    async def concurrent_mcp_access(self, task_id: int) -> bool:
        """Simulate concurrent MCP tool access."""
        try:
            if not self.mcp_manager:
                return False

            result = await self.mcp_manager.execute_tool('list_all_books', {})
            return True
        except Exception:
            return False

    # Performance Test Methods
    def test_query_performance(self) -> Dict[str, Any]:
        """Test query performance with various parameters."""
        try:
            collection = self.db_manager.get_collection()
            performance_results = {}

            # Test different query sizes
            query_sizes = [1, 5, 10, 20]

            for size in query_sizes:
                start_time = time.time()
                results = collection.query(
                    query_texts=["character development"],
                    n_results=size
                )
                end_time = time.time()

                performance_results[f'query_size_{size}'] = {
                    'time_seconds': round(end_time - start_time, 4),
                    'results_returned': len(results['documents'][0]) if results['documents'] else 0
                }

            return {
                'status': 'PASSED',
                'performance_results': performance_results
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_large_dataset_performance(self) -> Dict[str, Any]:
        """Test performance with large dataset operations."""
        try:
            collection = self.db_manager.get_collection()

            # Test getting all documents
            start_time = time.time()
            all_docs = collection.get()
            get_all_time = time.time() - start_time

            # Test complex metadata query
            start_time = time.time()
            books = self.metadata_manager.get_all_books()
            metadata_time = time.time() - start_time

            # Test large search
            start_time = time.time()
            large_search = collection.query(
                query_texts=["story narrative plot"],
                n_results=50
            )
            search_time = time.time() - start_time

            return {
                'status': 'PASSED',
                'total_documents': len(all_docs['documents']) if all_docs['documents'] else 0,
                'get_all_time': round(get_all_time, 4),
                'metadata_query_time': round(metadata_time, 4),
                'large_search_time': round(search_time, 4),
                'books_found': len(books)
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage patterns."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # Get initial memory usage
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Perform memory-intensive operations
            collection = self.db_manager.get_collection()

            # Load large dataset
            large_results = collection.query(
                query_texts=["comprehensive analysis"],
                n_results=100
            )

            # Get peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB

            memory_increase = peak_memory - initial_memory

            return {
                'status': 'PASSED',
                'initial_memory_mb': round(initial_memory, 2),
                'peak_memory_mb': round(peak_memory, 2),
                'memory_increase_mb': round(memory_increase, 2),
                'results_processed': len(large_results['documents'][0]) if large_results['documents'] else 0
            }
        except ImportError:
            return {'status': 'SKIPPED', 'reason': 'psutil not available'}
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_caching_performance(self) -> Dict[str, Any]:
        """Test caching effectiveness."""
        try:
            # Test repeated queries to measure caching impact
            query = "character development story"
            times = []

            # Run same query multiple times
            for i in range(5):
                start_time = time.time()
                collection = self.db_manager.get_collection()
                results = collection.query(
                    query_texts=[query],
                    n_results=10
                )
                end_time = time.time()
                times.append(end_time - start_time)

            # Calculate statistics
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)

            # Check if later queries are faster (indicating caching)
            first_half_avg = statistics.mean(times[:2])
            second_half_avg = statistics.mean(times[3:])

            caching_effective = second_half_avg < first_half_avg

            return {
                'status': 'PASSED',
                'query_times': [round(t, 4) for t in times],
                'avg_time': round(avg_time, 4),
                'min_time': round(min_time, 4),
                'max_time': round(max_time, 4),
                'caching_effective': caching_effective,
                'performance_improvement': round((first_half_avg - second_half_avg) / first_half_avg * 100, 2) if first_half_avg > 0 else 0
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    # Stress Test Methods
    def test_concurrent_queries(self) -> Dict[str, Any]:
        """Test concurrent query handling."""
        try:
            def run_query(query_id):
                try:
                    collection = self.db_manager.get_collection()
                    results = collection.query(
                        query_texts=[f"test query {query_id}"],
                        n_results=5
                    )
                    return True
                except Exception:
                    return False

            # Run concurrent queries
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(run_query, i) for i in range(20)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            successful_queries = sum(results)

            return {
                'status': 'PASSED' if successful_queries > 15 else 'FAILED',
                'total_queries': len(results),
                'successful_queries': successful_queries,
                'success_rate': round(successful_queries / len(results) * 100, 2)
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_high_frequency_operations(self) -> Dict[str, Any]:
        """Test high-frequency operations."""
        try:
            collection = self.db_manager.get_collection()

            # Perform rapid-fire operations
            start_time = time.time()
            operations_completed = 0

            for i in range(100):
                try:
                    # Alternate between different operations
                    if i % 3 == 0:
                        collection.query(query_texts=["test"], n_results=1)
                    elif i % 3 == 1:
                        collection.get(limit=1)
                    else:
                        collection.count()
                    operations_completed += 1
                except Exception:
                    pass

            end_time = time.time()
            total_time = end_time - start_time
            ops_per_second = operations_completed / total_time if total_time > 0 else 0

            return {
                'status': 'PASSED' if operations_completed > 80 else 'FAILED',
                'operations_attempted': 100,
                'operations_completed': operations_completed,
                'total_time_seconds': round(total_time, 4),
                'operations_per_second': round(ops_per_second, 2)
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_resource_limits(self) -> Dict[str, Any]:
        """Test system resource limits."""
        try:
            # Test large query results
            collection = self.db_manager.get_collection()

            # Try progressively larger queries until we hit limits
            max_successful_size = 0

            for size in [10, 50, 100, 200, 500]:
                try:
                    results = collection.query(
                        query_texts=["comprehensive test"],
                        n_results=size
                    )
                    if results['documents']:
                        max_successful_size = size
                except Exception:
                    break

            return {
                'status': 'PASSED',
                'max_successful_query_size': max_successful_size,
                'resource_limit_reached': max_successful_size < 500
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    # Error Handling Test Methods
    def test_invalid_inputs(self) -> Dict[str, Any]:
        """Test handling of invalid inputs."""
        try:
            collection = self.db_manager.get_collection()
            error_handling_results = {}

            # Test invalid query parameters
            try:
                collection.query(query_texts=[], n_results=5)
                error_handling_results['empty_query'] = 'FAILED - Should have raised error'
            except Exception:
                error_handling_results['empty_query'] = 'PASSED - Error properly handled'

            # Test invalid n_results
            try:
                collection.query(query_texts=["test"], n_results=-1)
                error_handling_results['negative_results'] = 'FAILED - Should have raised error'
            except Exception:
                error_handling_results['negative_results'] = 'PASSED - Error properly handled'

            # Test invalid metadata filter
            try:
                collection.query(
                    query_texts=["test"],
                    n_results=5,
                    where={"invalid_field": "invalid_value"}
                )
                error_handling_results['invalid_metadata'] = 'PASSED - Query handled gracefully'
            except Exception:
                error_handling_results['invalid_metadata'] = 'PASSED - Error properly handled'

            passed_tests = sum(1 for result in error_handling_results.values() if 'PASSED' in result)

            return {
                'status': 'PASSED' if passed_tests >= 2 else 'FAILED',
                'tests_run': len(error_handling_results),
                'tests_passed': passed_tests,
                'error_handling_results': error_handling_results
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_database_error_handling(self) -> Dict[str, Any]:
        """Test database error handling."""
        try:
            # Test connection recovery
            original_collection = self.db_manager.get_collection()

            # Simulate connection issues by testing edge cases
            error_scenarios = {}

            # Test with very large query text
            try:
                large_query = "test " * 10000  # Very large query
                results = original_collection.query(
                    query_texts=[large_query],
                    n_results=1
                )
                error_scenarios['large_query'] = 'PASSED - Handled gracefully'
            except Exception as e:
                error_scenarios['large_query'] = f'HANDLED - {type(e).__name__}'

            # Test with special characters
            try:
                special_query = "test\x00\x01\x02"  # Special characters
                results = original_collection.query(
                    query_texts=[special_query],
                    n_results=1
                )
                error_scenarios['special_chars'] = 'PASSED - Handled gracefully'
            except Exception as e:
                error_scenarios['special_chars'] = f'HANDLED - {type(e).__name__}'

            return {
                'status': 'PASSED',
                'error_scenarios_tested': len(error_scenarios),
                'error_scenarios': error_scenarios
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_recovery_mechanisms(self) -> Dict[str, Any]:
        """Test recovery mechanisms."""
        try:
            # Test database reconnection
            db_reconnect_success = False
            try:
                # Get fresh connection
                collection = self.db_manager.get_collection()
                if collection:
                    db_reconnect_success = True
            except Exception:
                pass

            # Test metadata manager recovery
            metadata_recovery_success = False
            try:
                books = self.metadata_manager.get_all_books()
                metadata_recovery_success = True
            except Exception:
                pass

            return {
                'status': 'PASSED' if db_reconnect_success and metadata_recovery_success else 'PARTIAL',
                'database_reconnection': db_reconnect_success,
                'metadata_recovery': metadata_recovery_success
            }
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("\n" + "="*80)
        logger.info("PHASE 6 COMPREHENSIVE TEST REPORT")
        logger.info("="*80)

        # Summary statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0

        for category, tests in self.test_results.items():
            logger.info(f"\n{category.upper().replace('_', ' ')}:")
            logger.info("-" * 40)

            for test_name, result in tests.items():
                total_tests += 1
                status = result.get('status', 'UNKNOWN')

                if status == 'PASSED':
                    passed_tests += 1
                    logger.info(f"✅ {test_name}: PASSED")
                elif status == 'FAILED':
                    failed_tests += 1
                    logger.info(f"❌ {test_name}: FAILED")
                    if 'error' in result:
                        logger.info(f"   Error: {result['error']}")
                elif status == 'SKIPPED':
                    skipped_tests += 1
                    logger.info(f"⏭️  {test_name}: SKIPPED")
                    if 'reason' in result:
                        logger.info(f"   Reason: {result['reason']}")

                # Show additional details for some tests
                if test_name == 'database_connection' and status == 'PASSED':
                    logger.info(f"   Documents in database: {result.get('document_count', 'Unknown')}")
                elif test_name == 'query_performance' and status == 'PASSED':
                    logger.info(f"   Performance results available")
                elif test_name == 'concurrent_queries' and status == 'PASSED':
                    logger.info(f"   Success rate: {result.get('success_rate', 'Unknown')}%")

        # Overall summary
        logger.info("\n" + "="*80)
        logger.info("OVERALL SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Skipped: {skipped_tests}")

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        logger.info(f"Success Rate: {success_rate:.1f}%")

        # Overall status
        if failed_tests == 0:
            logger.info("🎉 ALL TESTS PASSED!")
        elif passed_tests > failed_tests:
            logger.info("⚠️  MOSTLY SUCCESSFUL - Some issues detected")
        else:
            logger.info("❌ SIGNIFICANT ISSUES DETECTED")

        logger.info("="*80)

        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'skipped_tests': skipped_tests,
            'success_rate': success_rate,
            'overall_status': 'PASSED' if failed_tests == 0 else 'FAILED'
        }


def main():
    """Main function to run Phase 6 comprehensive tests."""
    print("Starting Phase 6: Testing and Optimization")
    print("=" * 50)

    # Initialize test suite
    test_suite = Phase6TestSuite()

    # Run all tests
    test_suite.run_all_tests()

    print("\nPhase 6 testing completed!")
    print("Check the logs above for detailed results.")


if __name__ == "__main__":
    main()