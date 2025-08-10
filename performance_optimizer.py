#!/usr/bin/env python3
"""
Phase 6: Performance Optimization Module
Implements caching, query optimization, and performance monitoring for the RAG+MCP system.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple
from functools import wraps, lru_cache
from collections import defaultdict
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryCache:
    """Thread-safe query result cache with TTL support."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """Initialize cache with maximum size and time-to-live."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()

    def _generate_key(self, query_data: Dict[str, Any]) -> str:
        """Generate cache key from query parameters."""
        # Sort keys for consistent hashing
        sorted_data = json.dumps(query_data, sort_keys=True)
        return hashlib.md5(sorted_data.encode()).hexdigest()

    def get(self, query_data: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._generate_key(query_data)

        with self.lock:
            if key not in self.cache:
                return None

            # Check TTL
            if time.time() - self.access_times[key] > self.ttl_seconds:
                del self.cache[key]
                del self.access_times[key]
                return None

            # Update access time
            self.access_times[key] = time.time()
            return self.cache[key]

    def set(self, query_data: Dict[str, Any], result: Any) -> None:
        """Store result in cache."""
        key = self._generate_key(query_data)

        with self.lock:
            # Evict oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(),
                               key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]

            self.cache[key] = result
            self.access_times[key] = time.time()

    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'hit_rate': getattr(self, '_hit_rate', 0.0)
            }


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = defaultdict(list)
        self.lock = threading.RLock()

    def record_query_time(self, operation: str, duration: float,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record query execution time."""
        with self.lock:
            self.metrics[f"{operation}_times"].append({
                'duration': duration,
                'timestamp': time.time(),
                'metadata': metadata or {}
            })

    def record_cache_hit(self, operation: str, hit: bool) -> None:
        """Record cache hit/miss."""
        with self.lock:
            self.metrics[f"{operation}_cache"].append({
                'hit': hit,
                'timestamp': time.time()
            })

    def get_average_time(self, operation: str,
                        last_n: Optional[int] = None) -> float:
        """Get average execution time for an operation."""
        with self.lock:
            times_key = f"{operation}_times"
            if times_key not in self.metrics:
                return 0.0

            times = self.metrics[times_key]
            if last_n:
                times = times[-last_n:]

            if not times:
                return 0.0

            return sum(entry['duration'] for entry in times) / len(times)

    def get_cache_hit_rate(self, operation: str,
                          last_n: Optional[int] = None) -> float:
        """Get cache hit rate for an operation."""
        with self.lock:
            cache_key = f"{operation}_cache"
            if cache_key not in self.metrics:
                return 0.0

            cache_data = self.metrics[cache_key]
            if last_n:
                cache_data = cache_data[-last_n:]

            if not cache_data:
                return 0.0

            hits = sum(1 for entry in cache_data if entry['hit'])
            return hits / len(cache_data)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self.lock:
            summary = {}

            # Get all operation types
            operations = set()
            for key in self.metrics.keys():
                if key.endswith('_times'):
                    operations.add(key[:-6])  # Remove '_times' suffix

            for operation in operations:
                summary[operation] = {
                    'avg_time_seconds': self.get_average_time(operation),
                    'avg_time_last_10': self.get_average_time(operation, 10),
                    'cache_hit_rate': self.get_cache_hit_rate(operation),
                    'cache_hit_rate_last_10': self.get_cache_hit_rate(operation, 10),
                    'total_queries': len(self.metrics.get(f"{operation}_times", []))
                }

            return summary

    def clear_metrics(self) -> None:
        """Clear all performance metrics."""
        with self.lock:
            self.metrics.clear()


class OptimizedDatabaseManager:
    """Database manager with performance optimizations."""

    def __init__(self, base_db_manager):
        """Initialize with base database manager."""
        self.base_db_manager = base_db_manager
        self.query_cache = QueryCache()
        self.performance_monitor = PerformanceMonitor()

    def cached_query(self, query_texts: List[str], n_results: int = 10,
                    where: Optional[Dict[str, Any]] = None,
                    include_metadata: bool = True) -> Dict[str, Any]:
        """Execute query with caching."""
        # Create cache key
        cache_key = {
            'query_texts': query_texts,
            'n_results': n_results,
            'where': where,
            'include_metadata': include_metadata
        }

        # Check cache first
        cached_result = self.query_cache.get(cache_key)
        if cached_result is not None:
            self.performance_monitor.record_cache_hit('query', True)
            return cached_result

        # Cache miss - execute query
        self.performance_monitor.record_cache_hit('query', False)

        start_time = time.time()
        try:
            collection = self.base_db_manager.get_collection()

            # Build query parameters
            query_params = {
                'query_texts': query_texts,
                'n_results': n_results
            }

            if where:
                query_params['where'] = where

            if include_metadata:
                query_params['include'] = ['documents', 'metadatas', 'distances']

            result = collection.query(**query_params)

            # Cache the result
            self.query_cache.set(cache_key, result)

            return result

        finally:
            duration = time.time() - start_time
            self.performance_monitor.record_query_time('query', duration, {
                'n_results': n_results,
                'has_filter': bool(where)
            })

    def cached_get_documents(self, limit: Optional[int] = None,
                           where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get documents with caching."""
        cache_key = {
            'operation': 'get_documents',
            'limit': limit,
            'where': where
        }

        # Check cache
        cached_result = self.query_cache.get(cache_key)
        if cached_result is not None:
            self.performance_monitor.record_cache_hit('get_documents', True)
            return cached_result

        # Cache miss
        self.performance_monitor.record_cache_hit('get_documents', False)

        start_time = time.time()
        try:
            collection = self.base_db_manager.get_collection()

            query_params = {}
            if limit:
                query_params['limit'] = limit
            if where:
                query_params['where'] = where

            result = collection.get(**query_params)

            # Cache the result
            self.query_cache.set(cache_key, result)

            return result

        finally:
            duration = time.time() - start_time
            self.performance_monitor.record_query_time('get_documents', duration, {
                'limit': limit,
                'has_filter': bool(where)
            })

    def get_collection(self):
        """Get collection from base manager."""
        return self.base_db_manager.get_collection()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'cache_stats': self.query_cache.get_stats(),
            'performance_summary': self.performance_monitor.get_performance_summary()
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.query_cache.clear()

    def clear_metrics(self) -> None:
        """Clear performance metrics."""
        self.performance_monitor.clear_metrics()


def performance_timer(operation_name: str):
    """Decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                logger.debug(f"{operation_name} completed in {duration:.4f} seconds")
        return wrapper
    return decorator


class BatchProcessor:
    """Process operations in batches for better performance."""

    def __init__(self, batch_size: int = 50):
        """Initialize batch processor."""
        self.batch_size = batch_size

    def process_documents_in_batches(self, documents: List[str],
                                   processor_func, **kwargs) -> List[Any]:
        """Process documents in batches."""
        results = []

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_results = processor_func(batch, **kwargs)
            results.extend(batch_results)

        return results

    def batch_query(self, db_manager, queries: List[str],
                   n_results: int = 10) -> List[Dict[str, Any]]:
        """Execute multiple queries in optimized batches."""
        results = []

        # Group similar queries for better caching
        for query in queries:
            if hasattr(db_manager, 'cached_query'):
                result = db_manager.cached_query([query], n_results)
            else:
                collection = db_manager.get_collection()
                result = collection.query(query_texts=[query], n_results=n_results)
            results.append(result)

        return results


class MemoryOptimizer:
    """Optimize memory usage for large operations."""

    @staticmethod
    def chunk_large_text(text: str, chunk_size: int = 7500,
                        overlap: int = 100) -> List[str]:
        """Chunk large text efficiently."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(min(100, len(text) - end)):
                    if text[end + i] in '.!?':
                        end = end + i + 1
                        break

            chunk = text[start:end]
            chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break

        return chunks

    @staticmethod
    def optimize_query_results(results: Dict[str, Any],
                             max_results: Optional[int] = None) -> Dict[str, Any]:
        """Optimize query results to reduce memory usage."""
        if max_results and 'documents' in results:
            for key in results:
                if isinstance(results[key], list) and len(results[key]) > 0:
                    if isinstance(results[key][0], list):
                        results[key] = [sublist[:max_results] for sublist in results[key]]

        return results


class HealthChecker:
    """Health check utilities for system monitoring."""

    def __init__(self, db_manager):
        """Initialize health checker."""
        self.db_manager = db_manager

    def check_database_health(self) -> Dict[str, Any]:
        """Check database health and connectivity."""
        try:
            start_time = time.time()
            collection = self.db_manager.get_collection()

            if collection is None:
                return {
                    'status': 'UNHEALTHY',
                    'error': 'Could not get collection',
                    'response_time': None
                }

            # Test basic operations
            count = collection.count()
            response_time = time.time() - start_time

            # Test query
            test_query_start = time.time()
            test_result = collection.query(
                query_texts=["health check"],
                n_results=1
            )
            query_time = time.time() - test_query_start

            return {
                'status': 'HEALTHY',
                'document_count': count,
                'connection_time': round(response_time, 4),
                'query_time': round(query_time, 4),
                'timestamp': time.time()
            }

        except Exception as e:
            return {
                'status': 'UNHEALTHY',
                'error': str(e),
                'timestamp': time.time()
            }

    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            return {
                'status': 'HEALTHY',
                'memory_usage_mb': round(process.memory_info().rss / 1024 / 1024, 2),
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
                'timestamp': time.time()
            }

        except ImportError:
            return {
                'status': 'UNAVAILABLE',
                'error': 'psutil not available',
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'timestamp': time.time()
            }

    def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        return {
            'database': self.check_database_health(),
            'system_resources': self.check_system_resources(),
            'overall_status': 'HEALTHY'  # Will be determined by individual checks
        }


# Global performance optimizer instance
_performance_optimizer = None
_optimizer_lock = threading.Lock()

def get_performance_optimizer(db_manager):
    """Get or create global performance optimizer instance."""
    global _performance_optimizer

    with _optimizer_lock:
        if _performance_optimizer is None:
            _performance_optimizer = OptimizedDatabaseManager(db_manager)
        return _performance_optimizer


def main():
    """Test performance optimization features."""
    print("Performance Optimization Module")
    print("=" * 40)

    # Test cache
    cache = QueryCache(max_size=10, ttl_seconds=60)

    test_query = {'query': 'test', 'n_results': 5}
    print(f"Cache miss: {cache.get(test_query) is None}")

    cache.set(test_query, {'results': ['test1', 'test2']})
    print(f"Cache hit: {cache.get(test_query) is not None}")

    print(f"Cache stats: {cache.get_stats()}")

    # Test performance monitor
    monitor = PerformanceMonitor()
    monitor.record_query_time('test_operation', 0.5)
    monitor.record_cache_hit('test_operation', True)

    print(f"Performance summary: {monitor.get_performance_summary()}")

    print("\nPerformance optimization module ready!")


if __name__ == "__main__":
    main()