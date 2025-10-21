"""
Validation middleware for Maria Conciliadora application.

This module provides:
- Request validation middleware
- Security validation decorators
- Input sanitization decorators
- Rate limiting middleware
- CSRF protection
- Security headers middleware
"""

import functools
from flask import request, jsonify, g
from typing import Dict, Any, List, Optional, Callable
from werkzeug.exceptions import RequestEntityTooLarge

from .validators import (
    validate_api_request, validate_input_security, sanitize_input,
    sanitize_filename, sanitize_path, rate_limiter, SecurityValidator,
    InputSanitizer, RequestValidator, BusinessRuleValidator
)
from .exceptions import ValidationError, AuthorizationError, BaseApplicationError
from .logging_config import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class ValidationMiddleware:
    """Comprehensive validation middleware for Flask applications."""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the validation middleware with Flask app."""
        app.before_request(self.validate_request)
        app.after_request(self.add_security_headers)
        
        # Register error handlers
        app.errorhandler(ValidationError)(self.handle_validation_error)
        app.errorhandler(RequestEntityTooLarge)(self.handle_request_too_large)
    
    def validate_request(self):
        """Validate incoming requests before processing."""
        try:
            # Skip validation for static files and health checks
            if request.endpoint in ['static', 'health', None]:
                return

            # Detect file upload endpoints
            is_file_upload = False
            if request.endpoint and ('upload' in request.endpoint or 'analyze' in request.endpoint):
                is_file_upload = True
                logger.info(f"Detected file upload endpoint: {request.endpoint}")

            # Get request information
            client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
            content_length = request.content_length or 0
            headers = dict(request.headers)
            url = request.url

            # Validate request
            validation_result = validate_api_request(
                request.get_json(silent=True) or {},
                headers,
                url,
                content_length,
                client_ip,
                is_file_upload=is_file_upload
            )

            if not validation_result.is_valid:
                logger.warning(f"Request validation failed from {client_ip}: {validation_result.errors}")
                logger.warning(f"Request headers: {headers}")
                logger.warning(f"Request method: {request.method}, URL: {url}")
                audit_logger.log_security_event('request_validation_failed', {
                    'client_ip': client_ip,
                    'errors': validation_result.errors,
                    'url': url,
                    'headers': headers
                })

                return jsonify({
                    'error': True,
                    'message': 'Request validation failed',
                    'details': validation_result.errors
                }), 400

            # Log warnings if any
            if validation_result.warnings:
                logger.info(f"Request validation warnings from {client_ip}: {validation_result.warnings}")

        except Exception as e:
            logger.error(f"Error in request validation: {str(e)}", exc_info=True)
            return jsonify({
                'error': True,
                'message': 'Internal validation error'
            }), 500
    
    def add_security_headers(self, response):
        """Add security headers to all responses."""
        # Content Security Policy
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data: https:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none';"
        )
        
        # Security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        
        # HSTS (only for HTTPS)
        if request.is_secure:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        return response
    
    def handle_validation_error(self, error):
        """Handle validation errors."""
        logger.warning(f"Validation error: {error.message}")
        return jsonify(error.to_dict()), error.status_code
    
    def handle_request_too_large(self, error):
        """Handle request entity too large errors."""
        logger.warning(f"Request too large: {error}")
        return jsonify({
            'error': True,
            'message': 'Request entity too large',
            'max_size': '10MB'
        }), 413


def validate_input_fields(*field_names: str):
    """
    Decorator to validate and sanitize specific input fields.
    
    Args:
        field_names: Names of fields to validate and sanitize
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Get request data
                if request.is_json:
                    # Use silent=True to avoid raising on empty/invalid JSON when Content-Type is application/json
                    data = request.get_json(silent=True) or {}
                else:
                    data = request.form.to_dict()
                
                # Validate and sanitize specified fields
                for field_name in field_names:
                    if field_name in data:
                        value = data[field_name]
                        
                        if isinstance(value, str):
                            # Security validation
                            security_result = validate_input_security(value, field_name)
                            if not security_result.is_valid:
                                logger.warning(f"Security validation failed for field {field_name}: {security_result.errors}")
                                return jsonify({
                                    'error': True,
                                    'message': f'Security validation failed for field {field_name}',
                                    'details': security_result.errors
                                }), 400
                            
                            # Sanitize input
                            data[field_name] = sanitize_input(value)
                
                # Store sanitized data for use in the route
                g.validated_data = data
                
                return func(*args, **kwargs)
                
            except BaseApplicationError as e:
                raise e
            except Exception as e:
                logger.error(f"Error in input validation decorator: {str(e)}", exc_info=True)
                return jsonify({
                    'error': True,
                    'message': 'Input validation error'
                }), 500
        
        return wrapper
    return decorator

def validate_file_upload(allowed_extensions: List[str], max_size_mb: int = 16):
    """
    Decorator to validate file uploads.

    Args:
        allowed_extensions: List of allowed file extensions
        max_size_mb: Maximum file size in MB
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Request files: {request.files}")
            try:
                if 'file' not in request.files:
                    return jsonify({
                        'error': True,
                        'message': 'No file provided'
                    }), 400

                file = request.files['file']

                if file.filename == '':
                    return jsonify({
                        'error': True,
                        'message': 'No file selected'
                    }), 400

                # Sanitize filename
                original_filename = file.filename
                sanitized_filename = sanitize_filename(original_filename)

                if sanitized_filename != original_filename:
                    logger.info(f"Filename sanitized: {original_filename} -> {sanitized_filename}")

                # Validate file extension
                file_ext = sanitized_filename.split('.')[-1].lower()
                if file_ext not in [ext.lower() for ext in allowed_extensions]:
                    return jsonify({
                        'error': True,
                        'message': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
                    }), 400

                # Store sanitized filename for use in the route
                g.sanitized_filename = sanitized_filename
                g.original_filename = original_filename

                return func(*args, **kwargs)

            except BaseApplicationError as e:
                raise e # Re-raise application errors to be handled by the main error handler
            except Exception as e:
                logger.error(f"Error in file upload validation: {str(e)}", exc_info=True)
                return jsonify({
                    'error': True,
                    'message': 'File validation error'
                }), 500

        return wrapper
    return decorator


def validate_financial_data(data_type: str = 'transaction'):
    """
    Decorator to validate financial data with business rules.
    
    Args:
        data_type: Type of financial data ('transaction' or 'company_financial')
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Get request data
                if request.is_json:
                    # Tolerate empty body with JSON content type
                    data = request.get_json(silent=True) or {}
                else:
                    data = request.form.to_dict()
                
                # Validate based on data type
                if data_type == 'transaction':
                    validation_result = BusinessRuleValidator.validate_financial_transaction(data)
                elif data_type == 'company_financial':
                    validation_result = BusinessRuleValidator.validate_company_financial_entry(data)
                else:
                    raise ValueError(f"Unknown data type: {data_type}")
                
                if not validation_result.is_valid:
                    logger.warning(f"Financial data validation failed: {validation_result.errors}")
                    return jsonify({
                        'error': True,
                        'message': 'Financial data validation failed',
                        'details': validation_result.errors
                    }), 400
                
                # Log warnings if any
                if validation_result.warnings:
                    logger.info(f"Financial data validation warnings: {validation_result.warnings}")
                    g.validation_warnings = validation_result.warnings
                
                return func(*args, **kwargs)
                
            except BaseApplicationError as e:
                raise e
            except Exception as e:
                logger.error(f"Error in financial data validation: {str(e)}", exc_info=True)
                return jsonify({
                    'error': True,
                    'message': 'Financial validation error'
                }), 500
        
        return wrapper
    return decorator


def rate_limit(max_requests: int = 100, window_minutes: int = 60):
    """
    Decorator to apply rate limiting to endpoints.
    
    Args:
        max_requests: Maximum number of requests allowed
        window_minutes: Time window in minutes
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
                
                # Check rate limit
                rate_result = rate_limiter.validate_rate_limit(
                    client_ip, max_requests, window_minutes
                )
                
                if not rate_result.is_valid:
                    logger.warning(f"Rate limit exceeded for {client_ip}: {rate_result.errors}")
                    audit_logger.log_security_event('rate_limit_exceeded', {
                        'client_ip': client_ip,
                        'max_requests': max_requests,
                        'window_minutes': window_minutes
                    })
                    
                    return jsonify({
                        'error': True,
                        'message': 'Rate limit exceeded',
                        'details': rate_result.errors
                    }), 429
                
                return func(*args, **kwargs)
                
            except BaseApplicationError as e:
                raise e
            except Exception as e:
                logger.error(f"Error in rate limiting: {str(e)}", exc_info=True)
                return jsonify({
                    'error': True,
                    'message': 'Rate limiting error'
                }), 500
        
        return wrapper
    return decorator


def require_content_type(*allowed_types: str):
    """
    Decorator to validate request content type.
    
    Args:
        allowed_types: Allowed content types
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            content_type = request.content_type
            
            if content_type:
                # Extract main content type (ignore charset and other parameters)
                main_type = content_type.split(';')[0].strip().lower()
                
                if main_type not in allowed_types:
                    logger.warning(f"Invalid content type: {content_type}")
                    return jsonify({
                        'error': True,
                        'message': f'Invalid content type. Allowed: {", ".join(allowed_types)}'
                    }), 415
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def sanitize_path_params(*param_names: str):
    """
    Decorator to sanitize path parameters.
    
    Args:
        param_names: Names of path parameters to sanitize
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Sanitize specified path parameters
                for param_name in param_names:
                    if param_name in kwargs:
                        original_value = kwargs[param_name]
                        
                        if isinstance(original_value, str):
                            # Security validation
                            security_result = validate_input_security(original_value, param_name)
                            if not security_result.is_valid:
                                logger.warning(f"Security validation failed for path param {param_name}: {security_result.errors}")
                                return jsonify({
                                    'error': True,
                                    'message': f'Invalid path parameter: {param_name}'
                                }), 400
                            
                            # Sanitize path parameter
                            if '/' in original_value or '\\' in original_value:
                                kwargs[param_name] = sanitize_path(original_value)
                            else:
                                kwargs[param_name] = sanitize_input(original_value)
                
                return func(*args, **kwargs)
                
            except BaseApplicationError as e:
                raise e
            except Exception as e:
                logger.error(f"Error in path parameter sanitization: {str(e)}", exc_info=True)
                return jsonify({
                    'error': True,
                    'message': 'Path parameter validation error'
                }), 500
        
        return wrapper
    return decorator


# CSRF Protection (basic implementation)
class CSRFProtection:
    """Basic CSRF protection implementation."""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize CSRF protection with Flask app."""
        app.before_request(self.validate_csrf)
    
    def validate_csrf(self):
        """Validate CSRF token for state-changing requests."""
        # Skip CSRF validation for safe methods and specific endpoints
        if request.method in ['GET', 'HEAD', 'OPTIONS']:
            return
        
        if request.endpoint in ['static', 'health']:
            return
        
        # For API endpoints, we rely on CORS and other security measures
        # In a full implementation, you would validate CSRF tokens here
        # For now, we log the request for audit purposes
        client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
        audit_logger.log_security_event('state_changing_request', {
            'client_ip': client_ip,
            'method': request.method,
            'endpoint': request.endpoint,
            'url': request.url
        })


# Initialize middleware instances
validation_middleware = ValidationMiddleware()
csrf_protection = CSRFProtection()
