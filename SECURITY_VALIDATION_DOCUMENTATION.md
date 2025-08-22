# Security Validation Documentation

## Overview

This document describes the comprehensive security validation and sanitization system implemented in the Maria Conciliadora application. The system provides multiple layers of protection against common security vulnerabilities and ensures data quality and integrity.

## Table of Contents

1. [Security Features](#security-features)
2. [Validation Components](#validation-components)
3. [Input Sanitization](#input-sanitization)
4. [Security Middleware](#security-middleware)
5. [Rate Limiting](#rate-limiting)
6. [File Upload Security](#file-upload-security)
7. [Business Rule Validation](#business-rule-validation)
8. [Security Headers](#security-headers)
9. [Usage Examples](#usage-examples)
10. [Testing](#testing)
11. [Configuration](#configuration)
12. [Best Practices](#best-practices)

## Security Features

### Implemented Protections

- **XSS (Cross-Site Scripting) Prevention**
- **SQL Injection Prevention**
- **Path Traversal Prevention**
- **Command Injection Prevention**
- **File Upload Security**
- **Rate Limiting**
- **Request Size Validation**
- **Input Sanitization**
- **Security Headers**
- **CSRF Protection**

### Security Layers

1. **Input Validation Layer**: Validates all incoming data
2. **Sanitization Layer**: Cleans and normalizes input data
3. **Business Logic Layer**: Enforces business rules and constraints
4. **File Security Layer**: Validates file uploads and content
5. **Rate Limiting Layer**: Prevents abuse and DoS attacks
6. **Response Security Layer**: Adds security headers

## Validation Components

### Core Validators

#### SecurityValidator
Detects various security threats in input data:

```python
from src.utils.validators import SecurityValidator

# Detect XSS attempts
is_xss = SecurityValidator.detect_xss('<script>alert("xss")</script>')

# Detect SQL injection
is_sql_injection = SecurityValidator.detect_sql_injection("'; DROP TABLE users; --")

# Detect path traversal
is_path_traversal = SecurityValidator.detect_path_traversal('../../../etc/passwd')

# Comprehensive security validation
result = SecurityValidator.validate_input_security(user_input, 'field_name')
```

#### InputSanitizer
Sanitizes input data to prevent attacks:

```python
from src.utils.validators import InputSanitizer

# Sanitize HTML content
clean_html = InputSanitizer.sanitize_html('<script>alert("xss")</script><p>Safe content</p>')

# Sanitize SQL input
clean_sql = InputSanitizer.sanitize_sql_input("'; DROP TABLE users; --")

# Sanitize filename
clean_filename = InputSanitizer.sanitize_filename('../../../malicious.exe')

# Sanitize file path
clean_path = InputSanitizer.sanitize_path('../../../etc/passwd')

# General input sanitization
clean_input = InputSanitizer.sanitize_general_input(user_input, max_length=1000)
```

#### BusinessRuleValidator
Validates financial data against business rules:

```python
from src.utils.validators import BusinessRuleValidator

# Validate financial transaction
transaction_data = {
    'amount': 1000.50,
    'date': '2023-12-01',
    'description': 'Office supplies',
    'bank_name': 'itau'
}

result = BusinessRuleValidator.validate_financial_transaction(transaction_data)

# Validate company financial entry
entry_data = {
    'amount': -500.00,
    'date': '2023-12-01',
    'description': 'Office rent',
    'transaction_type': 'expense',
    'category': 'rent'
}

result = BusinessRuleValidator.validate_company_financial_entry(entry_data)
```

## Input Sanitization

### Automatic Sanitization

All string inputs are automatically sanitized to prevent:

- HTML/JavaScript injection
- SQL injection attempts
- Path traversal attacks
- Command injection
- Control character injection

### Sanitization Functions

```python
from src.utils.validators import sanitize_input, sanitize_filename, sanitize_path

# Basic input sanitization
clean_input = sanitize_input(user_input, max_length=500, allow_html=False)

# Filename sanitization
safe_filename = sanitize_filename(uploaded_filename)

# Path sanitization
safe_path = sanitize_path(file_path)
```

## Security Middleware

### Validation Middleware

The validation middleware provides automatic security validation for all requests:

```python
from src.utils.validation_middleware import validation_middleware

# Initialize with Flask app
validation_middleware.init_app(app)
```

### Decorators

#### @validate_input_fields
Validates and sanitizes specific input fields:

```python
@validate_input_fields('username', 'email', 'description')
def create_user():
    # Fields are automatically validated and sanitized
    data = g.validated_data  # Access sanitized data
    # ... route logic
```

#### @validate_file_upload
Validates file uploads:

```python
@validate_file_upload(['pdf', 'xlsx'], max_size_mb=10)
def upload_document():
    filename = g.sanitized_filename  # Access sanitized filename
    # ... file processing logic
```

#### @validate_financial_data
Validates financial data with business rules:

```python
@validate_financial_data('transaction')
def process_transaction():
    # Financial data is validated against business rules
    warnings = g.validation_warnings  # Access validation warnings
    # ... processing logic
```

#### @rate_limit
Applies rate limiting to endpoints:

```python
@rate_limit(max_requests=100, window_minutes=60)
def api_endpoint():
    # Rate limiting is automatically enforced
    # ... endpoint logic
```

#### @require_content_type
Validates request content type:

```python
@require_content_type('application/json')
def json_endpoint():
    # Only JSON requests are allowed
    # ... endpoint logic
```

#### @sanitize_path_params
Sanitizes URL path parameters:

```python
@sanitize_path_params('user_id', 'document_id')
def get_user_document(user_id, document_id):
    # Path parameters are automatically sanitized
    # ... endpoint logic
```

## Rate Limiting

### Configuration

Rate limiting is applied per IP address with configurable limits:

```python
# Default limits
DEFAULT_RATE_LIMITS = {
    'file_upload': {'max_requests': 50, 'window_minutes': 60},
    'user_creation': {'max_requests': 10, 'window_minutes': 60},
    'api_read': {'max_requests': 200, 'window_minutes': 60},
    'api_write': {'max_requests': 100, 'window_minutes': 60}
}
```

### IP Blocking

IPs that exceed rate limits by 2x are temporarily blocked:

```python
from src.utils.validators import rate_limiter

# Unblock an IP address
rate_limiter.unblock_ip('192.168.1.100')
```

## File Upload Security

### Validation Layers

1. **Extension Validation**: Only allowed file extensions
2. **MIME Type Validation**: Validates actual file type
3. **Size Validation**: Enforces maximum file size
4. **Content Validation**: Scans for malicious content
5. **Filename Sanitization**: Prevents path traversal

### Enhanced File Validator

```python
from src.utils.validators import EnhancedFileValidator

validator = EnhancedFileValidator(
    allowed_extensions=['pdf', 'xlsx', 'ofx'],
    max_size_bytes=16 * 1024 * 1024,  # 16MB
    allowed_mime_types=['application/pdf', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
    check_content=True
)

result = validator.validate_file(file_path, filename)
```

### File Content Scanning

Files are scanned for:
- Embedded scripts
- Malicious JavaScript
- HTML content in non-HTML files
- Suspicious patterns

## Business Rule Validation

### Financial Constraints

- **Amount Limits**: -999,999,999.99 to 999,999,999.99
- **Date Limits**: 1900-01-01 to 30 days in the future
- **Description Length**: 1 to 500 characters
- **Suspicious Pattern Detection**: Flags test data, repeated characters, etc.

### Validation Rules

```python
# Transaction validation rules
MAX_TRANSACTION_AMOUNT = Decimal('999999999.99')
MIN_TRANSACTION_AMOUNT = Decimal('-999999999.99')
MIN_TRANSACTION_DATE = date(1900, 1, 1)
MAX_FUTURE_DATE_DAYS = 30

# Business logic validation
- Expense amounts should typically be negative
- Income amounts should typically be positive
- Large round numbers are flagged for review
- Suspicious descriptions are flagged
```

## Security Headers

### Implemented Headers

```http
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' https:; connect-src 'self' https:; frame-ancestors 'none';
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
Strict-Transport-Security: max-age=31536000; includeSubDomains (HTTPS only)
```

### Header Descriptions

- **CSP**: Prevents XSS by controlling resource loading
- **X-Content-Type-Options**: Prevents MIME type sniffing
- **X-Frame-Options**: Prevents clickjacking
- **X-XSS-Protection**: Enables browser XSS filtering
- **Referrer-Policy**: Controls referrer information
- **Permissions-Policy**: Restricts browser features
- **HSTS**: Enforces HTTPS connections

## Usage Examples

### Route Protection

```python
from flask import Blueprint, request, jsonify
from src.utils.validation_middleware import (
    validate_input_fields, validate_file_upload, rate_limit
)

@app.route('/api/users', methods=['POST'])
@rate_limit(max_requests=10, window_minutes=60)
@validate_input_fields('username', 'email')
def create_user():
    data = g.validated_data  # Sanitized input data
    # ... create user logic
    return jsonify({'success': True})

@app.route('/api/upload', methods=['POST'])
@rate_limit(max_requests=20, window_minutes=60)
@validate_file_upload(['pdf', 'xlsx'], max_size_mb=10)
def upload_file():
    filename = g.sanitized_filename
    # ... file processing logic
    return jsonify({'success': True})
```

### Manual Validation

```python
from src.utils.validators import validate_input_security, sanitize_input

def process_user_input(user_input):
    # Validate for security threats
    security_result = validate_input_security(user_input, 'user_input')
    
    if not security_result.is_valid:
        raise ValidationError(f"Security validation failed: {security_result.errors}")
    
    # Sanitize input
    clean_input = sanitize_input(user_input, max_length=1000)
    
    return clean_input
```

## Testing

### Unit Tests

Run security validation tests:

```bash
# Run all validation tests
pytest tests/unit/test_validation_security.py -v

# Run specific test class
pytest tests/unit/test_validation_security.py::TestSecurityValidator -v

# Run with coverage
pytest tests/unit/test_validation_security.py --cov=src.utils.validators
```

### Integration Tests

Run middleware integration tests:

```bash
# Run middleware tests
pytest tests/integration/test_validation_middleware.py -v

# Test specific functionality
pytest tests/integration/test_validation_middleware.py::TestValidationMiddlewareIntegration::test_malicious_input_detection -v
```

### Security Test Cases

The test suite includes:

- XSS attack detection (13+ test cases)
- SQL injection detection (15+ test cases)
- Path traversal detection (9+ test cases)
- Command injection detection (16+ test cases)
- File upload security
- Rate limiting enforcement
- Input sanitization
- Business rule validation

## Configuration

### Environment Variables

```bash
# Rate limiting configuration
RATE_LIMIT_ENABLED=true
RATE_LIMIT_STORAGE=memory  # or redis for production

# File upload limits
MAX_FILE_SIZE=16777216  # 16MB in bytes
ALLOWED_FILE_EXTENSIONS=pdf,xlsx,ofx,qfx

# Security features
SECURITY_HEADERS_ENABLED=true
CSRF_PROTECTION_ENABLED=true
INPUT_VALIDATION_ENABLED=true
```

### Application Configuration

```python
# config.py
class SecurityConfig:
    # Rate limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_STORAGE = 'memory'  # Use Redis in production
    
    # File uploads
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = ['pdf', 'xlsx', 'ofx', 'qfx']
    
    # Security features
    SECURITY_HEADERS_ENABLED = True
    CSRF_PROTECTION_ENABLED = True
    INPUT_VALIDATION_ENABLED = True
    
    # Validation strictness
    VALIDATION_STRICT_MODE = True
    SANITIZATION_ENABLED = True
```

## Best Practices

### Development Guidelines

1. **Always Validate Input**: Never trust user input
2. **Sanitize Early**: Clean input as soon as possible
3. **Use Decorators**: Apply validation decorators to routes
4. **Check Return Values**: Always check validation results
5. **Log Security Events**: Log all security-related events
6. **Test Thoroughly**: Include security tests in your test suite

### Security Checklist

- [ ] Input validation on all endpoints
- [ ] File upload restrictions in place
- [ ] Rate limiting configured
- [ ] Security headers enabled
- [ ] SQL injection prevention active
- [ ] XSS protection implemented
- [ ] Path traversal prevention active
- [ ] Business rules validated
- [ ] Error handling secure
- [ ] Logging configured for security events

### Production Considerations

1. **Use Redis for Rate Limiting**: Memory storage is not suitable for production
2. **Configure Proper CSP**: Adjust Content Security Policy for your needs
3. **Enable HTTPS**: Required for HSTS and secure cookies
4. **Monitor Security Logs**: Set up alerting for security events
5. **Regular Security Updates**: Keep dependencies updated
6. **Penetration Testing**: Regular security assessments

### Common Pitfalls

1. **Bypassing Validation**: Don't skip validation for "trusted" sources
2. **Client-Side Only Validation**: Always validate on the server
3. **Insufficient Logging**: Log security events for monitoring
4. **Weak Rate Limits**: Set appropriate limits for your use case
5. **Ignoring Warnings**: Address validation warnings promptly

## Monitoring and Alerting

### Security Events

The system logs the following security events:

- Failed validation attempts
- Rate limit violations
- Malicious input detection
- File upload violations
- Suspicious patterns

### Log Format

```json
{
  "timestamp": "2023-12-01T10:00:00Z",
  "level": "WARNING",
  "event_type": "security_validation_failed",
  "client_ip": "192.168.1.100",
  "endpoint": "/api/users",
  "details": {
    "validation_errors": ["XSS detected in username field"],
    "user_agent": "Mozilla/5.0...",
    "request_id": "req_123456"
  }
}
```

### Metrics to Monitor

- Validation failure rate
- Rate limit violations per IP
- File upload rejections
- Security header compliance
- Response time impact

## Troubleshooting

### Common Issues

1. **False Positives**: Adjust validation patterns if legitimate input is rejected
2. **Performance Impact**: Monitor response times and optimize if needed
3. **Rate Limit Issues**: Adjust limits based on usage patterns
4. **File Upload Problems**: Check file size limits and allowed types

### Debug Mode

Enable debug logging for detailed validation information:

```python
import logging
logging.getLogger('src.utils.validators').setLevel(logging.DEBUG)
```

### Validation Bypass (Development Only)

For development/testing, you can temporarily disable validation:

```python
# NEVER use in production
app.config['VALIDATION_ENABLED'] = False
```

## Updates and Maintenance

### Regular Tasks

1. **Update Security Patterns**: Add new attack patterns as they emerge
2. **Review Rate Limits**: Adjust based on usage patterns
3. **Update Dependencies**: Keep security libraries updated
4. **Review Logs**: Analyze security events regularly
5. **Test Security**: Run security tests with each deployment

### Version History

- **v1.0**: Initial implementation with basic validation
- **v1.1**: Added comprehensive security validation
- **v1.2**: Enhanced file upload security
- **v1.3**: Added business rule validation
- **v1.4**: Implemented rate limiting and security headers

---

For questions or issues related to security validation, please refer to the development team or create an issue in the project repository.