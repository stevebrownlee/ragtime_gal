"""
Structured Logging Configuration

Sets up structlog for structured, context-rich logging throughout the application.
Supports both JSON (production) and console (development) output formats.
"""

import logging
import sys
from typing import Optional
from pathlib import Path

import structlog
from structlog.typing import EventDict, WrappedLogger

from ragtime.config.settings import get_settings


def add_app_context(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add application context to all log entries.

    Adds:
    - Application name
    - Environment
    - Service name
    """
    event_dict['app'] = 'ragtime-gal'
    event_dict['service'] = 'rag-server'
    return event_dict


def configure_logging(log_level: Optional[str] = None, log_format: Optional[str] = None) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format style ('json' or 'console')

    The configuration is loaded from settings if not explicitly provided.
    """
    settings = get_settings()

    # Use provided values or fall back to settings
    level = log_level or settings.log_level
    format_style = log_format or settings.log_format

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        stream=sys.stdout,
    )

    # Determine processors based on format
    if format_style == "json":
        renderer = structlog.processors.JSONRenderer()
    else:  # console
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )

    # Configure structlog
    structlog.configure(
        processors=[
            # Filtering
            structlog.stdlib.filter_by_level,
            # Add log level
            structlog.stdlib.add_log_level,
            # Add logger name
            structlog.stdlib.add_logger_name,
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            # Add application context
            add_app_context,
            # Stack info
            structlog.processors.StackInfoRenderer(),
            # Exception formatting
            structlog.processors.format_exc_info,
            # Unicode handling
            structlog.processors.UnicodeDecoder(),
            # Final renderer
            renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set up file handler if log_file is configured
    if settings.log_file:
        log_file_path = Path(settings.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter('%(message)s')
        )

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured structlog logger

    Usage:
        logger = get_logger(__name__)
        logger.info("processing_query", query_id=query_id, collection=collection)
    """
    return structlog.get_logger(name)


def log_function_call(logger: structlog.stdlib.BoundLogger, func_name: str, **kwargs) -> None:
    """
    Log a function call with parameters.

    Args:
        logger: Logger instance
        func_name: Function name
        **kwargs: Function parameters to log

    Usage:
        log_function_call(logger, "process_query", query_id=query_id, k=10)
    """
    logger.debug(
        "function_called",
        function=func_name,
        **kwargs
    )


def log_error(
    logger: structlog.stdlib.BoundLogger,
    error: Exception,
    context: Optional[dict] = None,
    **kwargs
) -> None:
    """
    Log an error with context.

    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Optional context dictionary
        **kwargs: Additional context

    Usage:
        try:
            process_query(query)
        except Exception as e:
            log_error(logger, e, context={"query_id": query_id})
    """
    error_context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    if context:
        error_context.update(context)

    error_context.update(kwargs)

    logger.error(
        "error_occurred",
        **error_context,
        exc_info=True
    )


def log_performance(
    logger: structlog.stdlib.BoundLogger,
    operation: str,
    duration_ms: float,
    **kwargs
) -> None:
    """
    Log performance metrics.

    Args:
        logger: Logger instance
        operation: Operation name
        duration_ms: Duration in milliseconds
        **kwargs: Additional metrics

    Usage:
        start = time.time()
        result = process_query(query)
        duration_ms = (time.time() - start) * 1000
        log_performance(logger, "process_query", duration_ms, items_retrieved=len(result))
    """
    logger.info(
        "performance_metric",
        operation=operation,
        duration_ms=duration_ms,
        **kwargs
    )


class LogContext:
    """
    Context manager for adding context to logs within a block.

    Usage:
        with LogContext(logger, query_id=query_id, session_id=session_id):
            logger.info("processing_query")
            # All logs within this block will include query_id and session_id
    """

    def __init__(self, logger: structlog.stdlib.BoundLogger, **context):
        self.logger = logger
        self.context = context
        self.bound_logger = None

    def __enter__(self):
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Context is automatically removed when exiting
        pass


# Initialize logging on module import
configure_logging()