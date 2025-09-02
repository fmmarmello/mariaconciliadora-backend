"""
Centralized error handling utilities for Maria Conciliadora application.

This module provides:
- Error response formatting
- Exception handling decorators
- Error logging integration
- Recovery mechanisms
- HTTP status code mapping
"""

import functools
import traceback
from typing import Dict, Any, Optional, Callable, Union, Tuple
from flask import jsonify, request
from werkzeug.exceptions import HTTPException
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
import psutil
import time
from datetime import datetime

from .exceptions import (
    BaseApplicationError, DatabaseError, DatabaseConnectionError,
    DatabaseTransactionError, DatabaseConstraintError, TimeoutError,
    ResourceLimitError, SystemError, ErrorSeverity
)
from .logging_config import get_logger, get_audit_logger

# Initialize loggers
logger = get_logger(__name__)
audit_logger = get_audit_logger()


class ErrorHandler:
    """Centralized error handler with logging and response formatting."""
    
    def __init__(self):
        self.error_counts = {}
        self.last_error_time = {}
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], int]:
        """
        Handle any exception and return formatted response with status code.
        
        Args:
            error: The exception to handle
            context: Additional context information
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        context = context or {}
        
        # Track error frequency
        self._track_error(error)
        
        # Handle different types of errors
        if isinstance(error, BaseApplicationError):
            return self._handle_application_error(error, context)
        elif isinstance(error, HTTPException):
            return self._handle_http_error(error, context)
        elif isinstance(error, SQLAlchemyError):
            return self._handle_database_error(error, context)
        else:
            return self._handle_unexpected_error(error, context)
    
    def _handle_application_error(self, error: BaseApplicationError, context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Handle custom application errors."""
        # Log the error
        log_level = self._get_log_level(error.severity)
        log_message = f"Application error: {error.message}"
        
        if context:
            log_message += f" | Context: {context}"
        
        getattr(logger, log_level)(log_message, exc_info=True)
        
        # Audit log for critical errors
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            audit_logger.log_database_operation(
                operation='error_handling',
                table='system',
                records_affected=0,
                success=False,
                error=error.message
            )
        
        # Prepare response
        response = error.to_dict()
        if context:
            response['context'] = context
        
        return response, error.status_code
    
    def _handle_http_error(self, error: HTTPException, context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Handle HTTP exceptions from Flask/Werkzeug."""
        logger.warning(f"HTTP error {error.code}: {error.description}")
        
        response = {
            'error': True,
            'error_code': f'HTTP_{error.code}',
            'message': error.description or 'Erro HTTP',
            'category': 'http',
            'severity': 'medium'
        }
        
        if context:
            response['context'] = context
        
        return response, error.code
    
    def _handle_database_error(self, error: SQLAlchemyError, context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Handle SQLAlchemy database errors."""
        logger.error(f"Database error: {str(error)}", exc_info=True)
        
        # Map specific database errors to custom exceptions
        if isinstance(error, OperationalError):
            if "connection" in str(error).lower():
                app_error = DatabaseConnectionError()
            else:
                app_error = DatabaseError("Database operational error")
        elif isinstance(error, IntegrityError):
            constraint = self._extract_constraint_name(str(error))
            app_error = DatabaseConstraintError(constraint)
        else:
            app_error = DatabaseError(f"Database error: {type(error).__name__}")
        
        return self._handle_application_error(app_error, context)
    
    def _handle_unexpected_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Handle unexpected errors."""
        error_id = self._generate_error_id()
        
        logger.critical(
            f"Unexpected error [{error_id}]: {str(error)}",
            exc_info=True,
            extra={'error_id': error_id, 'context': context}
        )
        
        # Audit log for unexpected errors
        audit_logger.log_database_operation(
            operation='unexpected_error',
            table='system',
            records_affected=0,
            success=False,
            error=f"[{error_id}] {str(error)}"
        )
        
        response = {
            'error': True,
            'error_code': 'SYSTEM_UNEXPECTED_ERROR',
            'message': 'Erro interno inesperado. Nossa equipe foi notificada.',
            'category': 'system',
            'severity': 'critical',
            'error_id': error_id
        }
        
        if context:
            response['context'] = context
        
        return response, 500
    
    def _track_error(self, error: Exception):
        """Track error frequency for monitoring."""
        error_type = type(error).__name__
        current_time = time.time()
        
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
            self.last_error_time[error_type] = current_time
        
        self.error_counts[error_type] += 1
        self.last_error_time[error_type] = current_time
        
        # Log warning if error frequency is high
        if self.error_counts[error_type] > 10:
            time_diff = current_time - self.last_error_time.get(f"{error_type}_warning", 0)
            if time_diff > 300:  # 5 minutes
                logger.warning(f"High frequency of {error_type}: {self.error_counts[error_type]} occurrences")
                self.last_error_time[f"{error_type}_warning"] = current_time
    
    def _get_log_level(self, severity: ErrorSeverity) -> str:
        """Map error severity to log level."""
        mapping = {
            ErrorSeverity.LOW: 'info',
            ErrorSeverity.MEDIUM: 'warning',
            ErrorSeverity.HIGH: 'error',
            ErrorSeverity.CRITICAL: 'critical'
        }
        return mapping.get(severity, 'error')
    
    def _extract_constraint_name(self, error_message: str) -> str:
        """Extract constraint name from database error message."""
        # This is a simplified implementation
        # In production, you might want more sophisticated parsing
        if 'UNIQUE constraint failed' in error_message:
            return 'unique_constraint'
        elif 'FOREIGN KEY constraint failed' in error_message:
            return 'foreign_key_constraint'
        elif 'NOT NULL constraint failed' in error_message:
            return 'not_null_constraint'
        else:
            return 'unknown_constraint'
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"ERR_{timestamp}_{hash(time.time()) % 10000:04d}"


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(func: Callable) -> Callable:
    """
    Decorator to handle errors in route functions.
    
    Usage:
        @app.route('/api/endpoint')
        @handle_errors
        def my_endpoint():
            # Your code here
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get request context for better error reporting
            context = {
                'endpoint': request.endpoint,
                'method': request.method,
                'url': request.url,
                'user_agent': request.headers.get('User-Agent'),
                'remote_addr': request.remote_addr
            }
            
            response_data, status_code = error_handler.handle_error(e, context)
            return jsonify(response_data), status_code
    
    return wrapper


def handle_service_errors(service_name: str):
    """
    Decorator to handle errors in service functions.
    
    Usage:
        @handle_service_errors('ai_service')
        def process_with_ai(data):
            # Your code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'service': service_name,
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                # Re-raise as application error for consistent handling
                if isinstance(e, BaseApplicationError):
                    raise
                else:
                    logger.error(f"Service error in {service_name}.{func.__name__}: {str(e)}", exc_info=True)
                    raise SystemError(f"Service error in {service_name}: {str(e)}")
        
        return wrapper
    return decorator


def with_database_transaction(func: Callable) -> Callable:
    """
    Decorator to handle database transactions with automatic rollback on errors.
    
    Usage:
        @with_database_transaction
        def create_records(data):
            # Your database operations here
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from src.models.user import db
        
        try:
            result = func(*args, **kwargs)
            db.session.commit()
            return result
        except Exception as e:
            db.session.rollback()
            logger.error(f"Database transaction rolled back in {func.__name__}: {str(e)}")
            
            # Convert to appropriate database error
            if isinstance(e, BaseApplicationError):
                raise
            else:
                raise DatabaseTransactionError(func.__name__)
    
    return wrapper


def with_timeout(timeout_seconds: int):
    """
    Decorator to add timeout to function execution.
    Cross-platform compatible (works on both Unix and Windows).

    Usage:
        @with_timeout(30)
        def long_running_operation():
            # Your code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import platform
            import threading
            logger.info(f"Platform system: {platform.system()}")

            if platform.system() == 'Windows':
                # Windows implementation using threading
                result = [None]
                exception = [None]

                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception[0] = e

                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(timeout_seconds)

                if thread.is_alive():
                    # Thread is still running, timeout occurred
                    raise TimeoutError(func.__name__, timeout_seconds)
                elif exception[0]:
                    # Exception occurred in thread
                    raise exception[0]
                else:
                    # Success
                    return result[0]
            else:
                # Unix implementation using signals
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError(func.__name__, timeout_seconds)

                # Set up timeout
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)

                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # Cancel timeout
                    return result
                except Exception as e:
                    signal.alarm(0)  # Cancel timeout
                    raise
                finally:
                    signal.signal(signal.SIGALRM, old_handler)

        return wrapper
    return decorator


def check_resource_limits(memory_limit=90, disk_limit=95, cpu_limit=95):
    """
    Check system resource limits and raise error if exceeded.
    
    Raises:
        ResourceLimitError: If resource limits are exceeded
    """
    # Check memory usage
    memory = psutil.virtual_memory()
    if memory.percent > memory_limit:
        raise ResourceLimitError(
            resource='memory',
            limit=f'{memory_limit}%',
            current=f'{memory.percent:.1f}%'
        )
    
    # Check disk usage
    disk = psutil.disk_usage('/')
    if disk.percent > disk_limit:
        raise ResourceLimitError(
            resource='disk',
            limit=f'{disk_limit}%',
            current=f'{disk.percent:.1f}%'
        )
    
    # Check CPU usage (average over last 1 minute)
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > cpu_limit:
        raise ResourceLimitError(
            resource='cpu',
            limit=f'{cpu_limit}%',
            current=f'{cpu_percent:.1f}%'
        )


def with_resource_check(_func=None, *, memory_limit=90, disk_limit=95, cpu_limit=95):
    """
    Decorator to check resource limits before function execution.
    
    Usage:
        @with_resource_check
        def regular_operation():
            # Your code here
            pass

        @with_resource_check(memory_limit=95)
        def resource_intensive_operation():
            # Your code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            check_resource_limits(memory_limit, disk_limit, cpu_limit)
            return func(*args, **kwargs)
        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def create_error_response(
    message: str,
    status_code: int = 500,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    suggestions: Optional[list] = None
) -> Tuple[Dict[str, Any], int]:
    """
    Create a standardized error response.
    
    Args:
        message: User-friendly error message
        status_code: HTTP status code
        error_code: Unique error code
        details: Additional error details
        suggestions: List of suggestions for the user
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        'error': True,
        'message': message,
        'error_code': error_code or f'ERROR_{status_code}',
        'details': details or {},
        'suggestions': suggestions or []
    }
    
    return response, status_code


def log_error_context(error: Exception, context: Dict[str, Any]):
    """
    Log error with additional context information.
    
    Args:
        error: The exception that occurred
        context: Additional context information
    """
    logger.error(
        f"Error: {str(error)}",
        exc_info=True,
        extra={
            'error_type': type(error).__name__,
            'context': context
        }
    )


# Recovery mechanisms
class RecoveryManager:
    """Manages error recovery mechanisms."""
    
    @staticmethod
    def retry_with_backoff(func: Callable, max_retries: int = 3, backoff_factor: float = 1.0):
        """
        Retry function with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            backoff_factor: Backoff multiplier
            
        Returns:
            Function result or raises last exception
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
        
        raise last_exception
    
    @staticmethod
    def fallback_on_error(primary_func: Callable, fallback_func: Callable):
        """
        Execute fallback function if primary function fails.
        
        Args:
            primary_func: Primary function to execute
            fallback_func: Fallback function to execute on error
            
        Returns:
            Result from primary or fallback function
        """
        try:
            return primary_func()
        except Exception as e:
            logger.warning(f"Primary function failed, using fallback: {str(e)}")
            return fallback_func()


# Global recovery manager instance
recovery_manager = RecoveryManager()