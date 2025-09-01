"""
Advanced Validation Middleware for Maria Conciliadora API endpoints.

This module provides:
- Flask middleware for advanced validation using the new validation pipeline
- Request/response validation decorators
- Input sanitization with advanced validation engine
- Comprehensive error handling and logging
"""

import functools
from flask import request, jsonify, g
from typing import Dict, Any, List, Optional, Callable
from werkzeug.exceptions import RequestEntityTooLarge

from .advanced_validation_engine import advanced_validation_engine
from .validation_result import ValidationSeverity
from .exceptions import ValidationError, AuthorizationError
from .logging_config import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class AdvancedValidationMiddleware:
    """Advanced validation middleware for Flask applications using the new validation pipeline."""

    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Initialize the advanced validation middleware with Flask app."""
        app.before_request(self.validate_request)
        app.after_request(self.add_validation_headers)

        # Register error handlers
        app.errorhandler(ValidationError)(self.handle_validation_error)
        app.errorhandler(RequestEntityTooLarge)(self.handle_request_too_large)

    def validate_request(self):
        """Validate incoming requests before processing using advanced validation."""
        try:
            # Skip validation for static files and health checks
            if request.endpoint in ['static', 'health', None]:
                return

            # Get request information
            client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
            content_length = request.content_length or 0
            headers = dict(request.headers)
            url = request.url
            method = request.method

            # Prepare validation context
            validation_context = {
                'client_ip': client_ip,
                'content_length': content_length,
                'headers': headers,
                'url': url,
                'method': method,
                'endpoint': request.endpoint,
                'source': 'api_middleware'
            }

            # Validate request structure and security
            validation_result = advanced_validation_engine.validate(
                {},  # Empty data for request-level validation
                profile='api_request',
                context=validation_context
            )

            if not validation_result.is_valid:
                logger.warning(f"Request validation failed from {client_ip}: {validation_result.errors}")
                audit_logger.log_security_event('request_validation_failed', {
                    'client_ip': client_ip,
                    'errors': validation_result.errors,
                    'url': url,
                    'method': method
                })

                return jsonify({
                    'success': False,
                    'error': 'Request validation failed',
                    'details': validation_result.errors,
                    'validation_summary': validation_result.get_summary()
                }), 400

            # Log warnings if any
            if validation_result.warnings:
                logger.info(f"Request validation warnings from {client_ip}: {validation_result.warnings}")

            # Store validation result in request context
            g.request_validation = validation_result

        except Exception as e:
            logger.error(f"Error in advanced request validation: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': 'Internal validation error'
            }), 500

    def add_validation_headers(self, response):
        """Add validation-related headers to responses."""
        # Add validation timing header if available
        if hasattr(g, 'request_validation') and g.request_validation:
            validation_result = g.request_validation
            if validation_result.validation_duration_ms:
                response.headers['X-Validation-Duration'] = f"{validation_result.validation_duration_ms:.2f}ms"

            # Add validation status header
            response.headers['X-Validation-Status'] = validation_result.status.value

        return response

    def handle_validation_error(self, error):
        """Handle validation errors with detailed information."""
        logger.warning(f"Advanced validation error: {error.message}")

        response_data = {
            'success': False,
            'error': error.user_message or error.message,
            'error_code': error.error_code,
            'validation_details': {
                'field': getattr(error, 'field_name', None),
                'severity': getattr(error, 'severity', ValidationSeverity.MEDIUM).value
            }
        }

        if hasattr(error, 'details'):
            response_data['details'] = error.details

        return jsonify(response_data), error.status_code

    def handle_request_too_large(self, error):
        """Handle request entity too large errors."""
        logger.warning(f"Request too large: {error}")
        return jsonify({
            'success': False,
            'error': 'Request entity too large',
            'max_size': '10MB'
        }), 413


def validate_api_data(data_type: str = 'generic', profile: str = None,
                     required_fields: List[str] = None, **validation_kwargs):
    """
    Decorator to validate API request data using the advanced validation engine.

    Args:
        data_type: Type of data being validated ('transaction', 'company_financial', etc.)
        profile: Validation profile to use
        required_fields: List of fields that must be present
        **validation_kwargs: Additional validation context
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Get request data
                if request.is_json:
                    data = request.get_json() or {}
                else:
                    data = request.form.to_dict()

                # Prepare validation context
                validation_context = {
                    'data_type': data_type,
                    'endpoint': request.endpoint,
                    'method': request.method,
                    'client_ip': request.environ.get('REMOTE_ADDR', 'unknown'),
                    'source': 'api_decorator',
                    **validation_kwargs
                }

                # Determine validation profile
                if not profile:
                    if data_type == 'transaction':
                        profile_to_use = 'transaction'
                    elif data_type == 'company_financial':
                        profile_to_use = 'company_financial'
                    else:
                        profile_to_use = 'api_request'

                # Validate data using advanced validation engine
                validation_result = advanced_validation_engine.validate(
                    data,
                    profile=profile_to_use,
                    context=validation_context
                )

                # Check required fields if specified
                if required_fields:
                    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
                    if missing_fields:
                        validation_result.add_error(f"Missing required fields: {', '.join(missing_fields)}",
                                                  severity=ValidationSeverity.HIGH)

                if not validation_result.is_valid:
                    logger.warning(f"API data validation failed: {validation_result.errors}")

                    return jsonify({
                        'success': False,
                        'error': 'Data validation failed',
                        'validation_errors': validation_result.errors,
                        'validation_warnings': validation_result.warnings,
                        'validation_summary': validation_result.get_summary()
                    }), 400

                # Log warnings if any
                if validation_result.warnings:
                    logger.info(f"API data validation warnings: {validation_result.warnings}")

                # Store validated data and validation result for use in the route
                g.validated_data = data
                g.validation_result = validation_result

                return func(*args, **kwargs)

            except Exception as e:
                logger.error(f"Error in API data validation decorator: {str(e)}", exc_info=True)
                return jsonify({
                    'success': False,
                    'error': 'Data validation error'
                }), 500

        return wrapper
    return decorator


def validate_financial_transaction():
    """
    Decorator specifically for validating financial transaction data.
    """
    return validate_api_data(
        data_type='transaction',
        profile='transaction',
        required_fields=['amount', 'date', 'description']
    )


def validate_company_financial_entry():
    """
    Decorator specifically for validating company financial entry data.
    """
    return validate_api_data(
        data_type='company_financial',
        profile='company_financial',
        required_fields=['amount', 'date', 'description', 'transaction_type']
    )


def validate_bulk_data(max_items: int = 1000, item_profile: str = 'transaction'):
    """
    Decorator to validate bulk data uploads.

    Args:
        max_items: Maximum number of items allowed
        item_profile: Validation profile for individual items
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Get request data
                if request.is_json:
                    data = request.get_json() or {}
                else:
                    data = request.form.to_dict()

                # Extract items to validate
                items = data.get('items', data.get('data', []))
                if not isinstance(items, list):
                    return jsonify({
                        'success': False,
                        'error': 'Expected list of items for bulk validation'
                    }), 400

                # Check item count limit
                if len(items) > max_items:
                    return jsonify({
                        'success': False,
                        'error': f'Too many items. Maximum allowed: {max_items}',
                        'received': len(items)
                    }), 400

                if len(items) == 0:
                    return jsonify({
                        'success': False,
                        'error': 'No items provided for validation'
                    }), 400

                # Prepare validation context
                validation_context = {
                    'bulk_operation': True,
                    'total_items': len(items),
                    'endpoint': request.endpoint,
                    'source': 'bulk_validation_decorator'
                }

                # Validate bulk data
                validation_result = advanced_validation_engine.validate_bulk(
                    items,
                    profile=item_profile,
                    context=validation_context,
                    parallel=True
                )

                if not validation_result.is_valid:
                    logger.warning(f"Bulk data validation failed: {len(validation_result.errors)} errors")

                    return jsonify({
                        'success': False,
                        'error': 'Bulk data validation failed',
                        'validation_summary': validation_result.get_summary(),
                        'total_errors': len(validation_result.errors),
                        'total_warnings': len(validation_result.warnings)
                    }), 400

                # Log validation summary
                summary = validation_result.get_summary()
                logger.info(f"Bulk validation completed: {summary['total_items']} items, "
                           f"{summary['items_with_errors']} with errors")

                # Store validated data and results
                g.validated_bulk_data = items
                g.bulk_validation_result = validation_result

                return func(*args, **kwargs)

            except Exception as e:
                logger.error(f"Error in bulk data validation decorator: {str(e)}", exc_info=True)
                return jsonify({
                    'success': False,
                    'error': 'Bulk validation error'
                }), 500

        return wrapper
    return decorator


def get_validation_metrics():
    """
    Decorator to collect and log validation metrics for monitoring.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = __import__('time').time()

            try:
                result = func(*args, **kwargs)

                # Collect validation metrics
                metrics = {
                    'endpoint': request.endpoint,
                    'method': request.method,
                    'processing_time': __import__('time').time() - start_time,
                    'client_ip': request.environ.get('REMOTE_ADDR', 'unknown')
                }

                # Add validation metrics if available
                if hasattr(g, 'validation_result') and g.validation_result:
                    val_result = g.validation_result
                    metrics.update({
                        'validation_status': val_result.status.value,
                        'validation_errors': len(val_result.errors),
                        'validation_warnings': len(val_result.warnings),
                        'validation_duration': val_result.validation_duration_ms
                    })

                elif hasattr(g, 'bulk_validation_result') and g.bulk_validation_result:
                    bulk_result = g.bulk_validation_result
                    summary = bulk_result.get_summary()
                    metrics.update({
                        'bulk_validation': True,
                        'total_items': summary.get('total_items', 0),
                        'items_with_errors': summary.get('items_with_errors', 0),
                        'bulk_validation_duration': bulk_result.validation_duration_ms
                    })

                # Log metrics for monitoring
                logger.info(f"Validation metrics: {metrics}")

                return result

            except Exception as e:
                # Log error metrics
                error_metrics = {
                    'endpoint': request.endpoint,
                    'method': request.method,
                    'error': str(e),
                    'processing_time': __import__('time').time() - start_time
                }
                logger.error(f"Validation error metrics: {error_metrics}")
                raise

        return wrapper
    return decorator


# Initialize middleware instance
advanced_validation_middleware = AdvancedValidationMiddleware()