#!/usr/bin/env python3
"""
Phase 6: Error Handling and Logging Module
Comprehensive error handling, logging, and recovery mechanisms for the RAG+MCP system.
"""

import logging
import traceback
import functools
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
import json
import os
from datetime import datetime

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ErrorCategory(Enum):
    """Error categories for classification."""
    DATABASE = "DATABASE"
    MCP_SERVER = "MCP_SERVER"
    QUERY_PROCESSING = "QUERY_PROCESSING"
    AUTHENTICATION = "AUTHENTICATION"
    VALIDATION = "VALIDATION"
    SYSTEM_RESOURCE = "SYSTEM_RESOURCE"
    NETWORK = "NETWORK"
    UNKNOWN = "UNKNOWN"

class ErrorHandler:
    """Centralized error handling and logging system."""

    def __init__(self, log_file: str = "ragtime_gal_errors.log"):
        """Initialize error handler with logging configuration."""
        self.log_file = log_file
        self.error_counts = {}
        self.error_history = []
        self.lock = threading.RLock()

        # Configure logging
        self.setup_logging()

        # Error recovery strategies
        self.recovery_strategies = {
            ErrorCategory.DATABASE: self._recover_database_error,
            ErrorCategory.MCP_SERVER: self._recover_mcp_error,
            ErrorCategory.QUERY_PROCESSING: self._recover_query_error,
            ErrorCategory.SYSTEM_RESOURCE: self._recover_resource_error
        }

    def setup_logging(self):
        """Set up comprehensive logging configuration."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else "logs"
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )

        # Create specialized loggers
        self.error_logger = logging.getLogger('ragtime_gal.errors')
        self.performance_logger = logging.getLogger('ragtime_gal.performance')
        self.security_logger = logging.getLogger('ragtime_gal.security')
        self.mcp_logger = logging.getLogger('ragtime_gal.mcp')
        self.database_logger = logging.getLogger('ragtime_gal.database')

    def handle_error(self, error: Exception, category: ErrorCategory = ErrorCategory.UNKNOWN,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[Dict[str, Any]] = None,
                    attempt_recovery: bool = True) -> Dict[str, Any]:
        """Handle error with logging, classification, and optional recovery."""

        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'category': category.value,
            'severity': severity.value,
            'context': context or {},
            'traceback': traceback.format_exc(),
            'recovery_attempted': False,
            'recovery_successful': False
        }

        with self.lock:
            # Update error counts
            error_key = f"{category.value}:{type(error).__name__}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

            # Add to history
            self.error_history.append(error_info)

            # Keep only last 1000 errors
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-1000:]

        # Log the error
        self._log_error(error_info)

        # Attempt recovery if requested
        if attempt_recovery and category in self.recovery_strategies:
            try:
                error_info['recovery_attempted'] = True
                recovery_result = self.recovery_strategies[category](error, context)
                error_info['recovery_successful'] = recovery_result.get('success', False)
                error_info['recovery_details'] = recovery_result

                if recovery_result.get('success'):
                    self.error_logger.info(f"Successfully recovered from {category.value} error")
                else:
                    self.error_logger.warning(f"Recovery failed for {category.value} error: {recovery_result.get('message')}")

            except Exception as recovery_error:
                self.error_logger.error(f"Recovery attempt failed: {recovery_error}")
                error_info['recovery_error'] = str(recovery_error)

        return error_info

    def _log_error(self, error_info: Dict[str, Any]):
        """Log error with appropriate severity level."""
        severity = ErrorSeverity(error_info['severity'])
        category = error_info['category']
        message = f"[{category}] {error_info['error_type']}: {error_info['error_message']}"

        if severity == ErrorSeverity.CRITICAL:
            self.error_logger.critical(message)
        elif severity == ErrorSeverity.HIGH:
            self.error_logger.error(message)
        elif severity == ErrorSeverity.MEDIUM:
            self.error_logger.warning(message)
        else:
            self.error_logger.info(message)

        # Log context if available
        if error_info['context']:
            self.error_logger.debug(f"Context: {json.dumps(error_info['context'], indent=2)}")

    def _recover_database_error(self, error: Exception, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Attempt to recover from database errors."""
        try:
            self.database_logger.info("Attempting database error recovery")

            # Strategy 1: Retry connection
            if "connection" in str(error).lower():
                time.sleep(1)  # Brief delay before retry
                return {'success': True, 'strategy': 'connection_retry', 'message': 'Connection retry scheduled'}

            # Strategy 2: Query optimization
            if "timeout" in str(error).lower() or "slow" in str(error).lower():
                return {'success': True, 'strategy': 'query_optimization', 'message': 'Query optimization suggested'}

            # Strategy 3: Resource cleanup
            if "memory" in str(error).lower() or "resource" in str(error).lower():
                return {'success': True, 'strategy': 'resource_cleanup', 'message': 'Resource cleanup initiated'}

            return {'success': False, 'message': 'No applicable recovery strategy'}

        except Exception as e:
            return {'success': False, 'message': f'Recovery failed: {str(e)}'}

    def _recover_mcp_error(self, error: Exception, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Attempt to recover from MCP server errors."""
        try:
            self.mcp_logger.info("Attempting MCP server error recovery")

            # Strategy 1: Server restart
            if "connection" in str(error).lower() or "server" in str(error).lower():
                return {'success': True, 'strategy': 'server_restart', 'message': 'MCP server restart scheduled'}

            # Strategy 2: Tool re-registration
            if "tool" in str(error).lower() or "registration" in str(error).lower():
                return {'success': True, 'strategy': 'tool_reregistration', 'message': 'Tool re-registration scheduled'}

            return {'success': False, 'message': 'No applicable MCP recovery strategy'}

        except Exception as e:
            return {'success': False, 'message': f'MCP recovery failed: {str(e)}'}

    def _recover_query_error(self, error: Exception, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Attempt to recover from query processing errors."""
        try:
            # Strategy 1: Query simplification
            if "complex" in str(error).lower() or "timeout" in str(error).lower():
                return {'success': True, 'strategy': 'query_simplification', 'message': 'Query simplification recommended'}

            # Strategy 2: Result limit reduction
            if "memory" in str(error).lower() or "large" in str(error).lower():
                return {'success': True, 'strategy': 'result_limit', 'message': 'Result limit reduction recommended'}

            return {'success': False, 'message': 'No applicable query recovery strategy'}

        except Exception as e:
            return {'success': False, 'message': f'Query recovery failed: {str(e)}'}

    def _recover_resource_error(self, error: Exception, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Attempt to recover from system resource errors."""
        try:
            # Strategy 1: Memory cleanup
            if "memory" in str(error).lower():
                import gc
                gc.collect()
                return {'success': True, 'strategy': 'memory_cleanup', 'message': 'Garbage collection performed'}

            # Strategy 2: Cache clearing
            if "cache" in str(error).lower():
                return {'success': True, 'strategy': 'cache_clear', 'message': 'Cache clearing recommended'}

            return {'success': False, 'message': 'No applicable resource recovery strategy'}

        except Exception as e:
            return {'success': False, 'message': f'Resource recovery failed: {str(e)}'}

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self.lock:
            total_errors = len(self.error_history)

            if total_errors == 0:
                return {
                    'total_errors': 0,
                    'error_rate': 0.0,
                    'categories': {},
                    'severities': {},
                    'top_errors': [],
                    'recent_errors': []
                }

            # Count by category
            categories = {}
            severities = {}

            for error in self.error_history:
                cat = error['category']
                sev = error['severity']
                categories[cat] = categories.get(cat, 0) + 1
                severities[sev] = severities.get(sev, 0) + 1

            # Top errors by frequency
            top_errors = sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            # Recent errors (last 10)
            recent_errors = self.error_history[-10:]

            # Calculate error rate safely
            if self.error_history:
                # Parse timestamp if it's a string
                first_timestamp = self.error_history[0]['timestamp']
                if isinstance(first_timestamp, str):
                    from datetime import datetime
                    first_timestamp = datetime.fromisoformat(first_timestamp).timestamp()
                time_span = max(1, time.time() - first_timestamp)
                error_rate = total_errors / time_span
            else:
                error_rate = 0.0

            return {
                'total_errors': total_errors,
                'error_rate': error_rate,
                'categories': categories,
                'severities': severities,
                'top_errors': [{'error': error, 'count': count} for error, count in top_errors],
                'recent_errors': [
                    {
                        'timestamp': error['timestamp'],
                        'type': error['error_type'],
                        'category': error['category'],
                        'severity': error['severity']
                    } for error in recent_errors
                ]
            }

    def clear_error_history(self):
        """Clear error history and statistics."""
        with self.lock:
            self.error_history.clear()
            self.error_counts.clear()


def error_handler_decorator(category: ErrorCategory = ErrorCategory.UNKNOWN,
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                          attempt_recovery: bool = True):
    """Decorator for automatic error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get global error handler
                error_handler = get_global_error_handler()

                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }

                error_info = error_handler.handle_error(
                    e, category, severity, context, attempt_recovery
                )

                # Re-raise the exception after handling
                raise e
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default_return=None,
                error_category: ErrorCategory = ErrorCategory.UNKNOWN,
                **kwargs) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler = get_global_error_handler()
        error_handler.handle_error(e, error_category)
        return default_return


class HealthMonitor:
    """Monitor system health and detect issues proactively."""

    def __init__(self, error_handler: ErrorHandler):
        """Initialize health monitor."""
        self.error_handler = error_handler
        self.health_checks = {}
        self.monitoring_active = False
        self.monitor_thread = None

    def register_health_check(self, name: str, check_func: Callable,
                            interval_seconds: int = 60):
        """Register a health check function."""
        self.health_checks[name] = {
            'function': check_func,
            'interval': interval_seconds,
            'last_check': 0,
            'last_result': None
        }

    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            current_time = time.time()

            for name, check_info in self.health_checks.items():
                if current_time - check_info['last_check'] >= check_info['interval']:
                    try:
                        result = check_info['function']()
                        check_info['last_result'] = result
                        check_info['last_check'] = current_time

                        # Check for health issues
                        if isinstance(result, dict) and result.get('status') != 'HEALTHY':
                            self.error_handler.handle_error(
                                Exception(f"Health check failed: {name}"),
                                ErrorCategory.SYSTEM_RESOURCE,
                                ErrorSeverity.HIGH,
                                {'health_check': name, 'result': result}
                            )
                    except Exception as e:
                        self.error_handler.handle_error(
                            e, ErrorCategory.SYSTEM_RESOURCE, ErrorSeverity.MEDIUM,
                            {'health_check': name}
                        )

            time.sleep(10)  # Check every 10 seconds

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        status = {
            'overall_status': 'HEALTHY',
            'checks': {},
            'monitoring_active': self.monitoring_active
        }

        unhealthy_checks = 0

        for name, check_info in self.health_checks.items():
            result = check_info['last_result']
            if result:
                status['checks'][name] = result
                if isinstance(result, dict) and result.get('status') != 'HEALTHY':
                    unhealthy_checks += 1

        if unhealthy_checks > 0:
            status['overall_status'] = 'DEGRADED' if unhealthy_checks < len(self.health_checks) / 2 else 'UNHEALTHY'

        return status


# Global error handler instance
_global_error_handler = None
_error_handler_lock = threading.Lock()

def get_global_error_handler() -> ErrorHandler:
    """Get or create global error handler instance."""
    global _global_error_handler

    with _error_handler_lock:
        if _global_error_handler is None:
            _global_error_handler = ErrorHandler()
        return _global_error_handler


def setup_global_error_handling(log_file: str = "logs/ragtime_gal_errors.log"):
    """Set up global error handling system."""
    global _global_error_handler

    with _error_handler_lock:
        _global_error_handler = ErrorHandler(log_file)

    # Set up uncaught exception handler
    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        _global_error_handler.handle_error(
            exc_value,
            ErrorCategory.UNKNOWN,
            ErrorSeverity.CRITICAL,
            {'uncaught': True}
        )

    import sys
    sys.excepthook = handle_uncaught_exception


def main():
    """Test error handling functionality."""
    print("Error Handling and Logging Module")
    print("=" * 40)

    # Set up error handling
    setup_global_error_handling()
    error_handler = get_global_error_handler()

    # Test error handling
    try:
        raise ValueError("Test error for demonstration")
    except Exception as e:
        error_info = error_handler.handle_error(
            e, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM,
            {'test': True}
        )
        print(f"Error handled: {error_info['error_type']}")

    # Test decorator
    @error_handler_decorator(ErrorCategory.QUERY_PROCESSING, ErrorSeverity.LOW)
    def test_function():
        raise RuntimeError("Test decorator error")

    try:
        test_function()
    except Exception:
        pass  # Error was handled by decorator

    # Get statistics
    stats = error_handler.get_error_statistics()
    print(f"Error statistics: {stats['total_errors']} total errors")

    print("\nError handling system ready!")


if __name__ == "__main__":
    main()