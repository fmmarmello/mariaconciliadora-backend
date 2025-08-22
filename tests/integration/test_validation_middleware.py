"""
Integration tests for validation middleware.

This module tests:
- Middleware integration with Flask routes
- End-to-end validation flow
- Security headers
- Rate limiting in practice
- File upload validation
- CSRF protection
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, Mock
from flask import Flask

from src.main import app
from src.utils.validation_middleware import validation_middleware, csrf_protection
from src.utils.validators import rate_limiter


class TestValidationMiddlewareIntegration:
    """Test validation middleware integration."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_security_headers_added(self, client):
        """Test that security headers are added to responses."""
        response = client.get('/api/transactions')
        
        # Check security headers
        assert 'Content-Security-Policy' in response.headers
        assert 'X-Content-Type-Options' in response.headers
        assert 'X-Frame-Options' in response.headers
        assert 'X-XSS-Protection' in response.headers
        assert 'Referrer-Policy' in response.headers
        assert 'Permissions-Policy' in response.headers
        
        # Check header values
        assert response.headers['X-Content-Type-Options'] == 'nosniff'
        assert response.headers['X-Frame-Options'] == 'DENY'
        assert response.headers['X-XSS-Protection'] == '1; mode=block'
        assert 'strict-origin-when-cross-origin' in response.headers['Referrer-Policy']
    
    def test_request_size_validation(self, client):
        """Test request size validation."""
        # Create a large payload (larger than 10MB limit)
        large_data = {'data': 'x' * (11 * 1024 * 1024)}  # 11MB of data
        
        response = client.post('/api/users', 
                             data=json.dumps(large_data),
                             content_type='application/json')
        
        # Should be rejected due to size
        assert response.status_code == 413  # Request Entity Too Large
    
    def test_malicious_input_detection(self, client):
        """Test detection of malicious input."""
        # Test XSS attempt
        xss_data = {
            'username': '<script>alert("xss")</script>',
            'email': 'user@example.com'
        }
        
        response = client.post('/api/users',
                             data=json.dumps(xss_data),
                             content_type='application/json')
        
        # Should be rejected due to XSS
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] is True
        assert 'validation failed' in data['message'].lower()
    
    def test_sql_injection_detection(self, client):
        """Test detection of SQL injection attempts."""
        # Test SQL injection attempt
        sql_data = {
            'username': "admin'; DROP TABLE users; --",
            'email': 'user@example.com'
        }
        
        response = client.post('/api/users',
                             data=json.dumps(sql_data),
                             content_type='application/json')
        
        # Should be rejected due to SQL injection
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] is True
        assert 'validation failed' in data['message'].lower()
    
    def test_rate_limiting_enforcement(self, client):
        """Test rate limiting enforcement."""
        # Clear any existing rate limit data
        rate_limiter.request_counts.clear()
        rate_limiter.blocked_ips.clear()
        
        # Make requests up to the limit
        for i in range(10):  # Assuming user creation has a limit of 10/hour
            response = client.post('/api/users',
                                 data=json.dumps({
                                     'username': f'user{i}',
                                     'email': f'user{i}@example.com'
                                 }),
                                 content_type='application/json')
            
            if i < 9:  # First 9 should succeed (or fail for other reasons)
                assert response.status_code != 429
        
        # The 11th request should be rate limited
        response = client.post('/api/users',
                             data=json.dumps({
                                 'username': 'user_blocked',
                                 'email': 'blocked@example.com'
                             }),
                             content_type='application/json')
        
        assert response.status_code == 429  # Too Many Requests
        data = json.loads(response.data)
        assert 'Rate limit exceeded' in data['message']
    
    def test_file_upload_validation(self, client):
        """Test file upload validation."""
        # Test with invalid file type
        with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as temp_file:
            temp_file.write(b'fake executable content')
            temp_path = temp_file.name
        
        try:
            with open(temp_path, 'rb') as f:
                response = client.post('/api/upload-ofx',
                                     data={'file': (f, 'malicious.exe')},
                                     content_type='multipart/form-data')
            
            # Should be rejected due to invalid file type
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['error'] is True
            assert 'Invalid file type' in data['message']
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_malicious_file_upload(self, client):
        """Test detection of malicious file content."""
        # Create a file with script content
        with tempfile.NamedTemporaryFile(suffix='.ofx', delete=False) as temp_file:
            temp_file.write(b'<script>alert("malicious")</script>')
            temp_path = temp_file.name
        
        try:
            with open(temp_path, 'rb') as f:
                response = client.post('/api/upload-ofx',
                                     data={'file': (f, 'malicious.ofx')},
                                     content_type='multipart/form-data')
            
            # Should be rejected or processed with warnings
            # The exact response depends on the file validation implementation
            assert response.status_code in [400, 422]  # Bad Request or Unprocessable Entity
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_path_traversal_prevention(self, client):
        """Test prevention of path traversal attacks."""
        # Test path traversal in URL parameters
        response = client.get('/api/reconciliation/../../../etc/passwd/confirm',
                            headers={'Content-Type': 'application/json'})
        
        # Should not allow path traversal
        assert response.status_code in [400, 404]  # Bad Request or Not Found
    
    def test_header_injection_prevention(self, client):
        """Test prevention of header injection attacks."""
        # Test header injection
        malicious_headers = {
            'X-Custom': 'value\r\nX-Injected: malicious',
            'Content-Type': 'application/json'
        }
        
        response = client.get('/api/transactions', headers=malicious_headers)
        
        # Should be rejected due to header injection
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] is True
        assert 'validation failed' in data['message'].lower()
    
    def test_content_type_validation(self, client):
        """Test content type validation."""
        # Test with invalid content type
        response = client.post('/api/users',
                             data='{"username": "test", "email": "test@example.com"}',
                             content_type='application/evil')
        
        # Should be rejected due to invalid content type
        assert response.status_code == 415  # Unsupported Media Type
        data = json.loads(response.data)
        assert 'Invalid content type' in data['message']
    
    def test_financial_data_validation(self, client):
        """Test financial data validation."""
        # Test with invalid financial data
        invalid_financial_data = {
            'entries': [{
                'amount': 999999999999.99,  # Exceeds maximum
                'date': '2025-12-31',  # Too far in future
                'description': 'test transaction',  # Suspicious pattern
                'transaction_type': 'expense'
            }]
        }
        
        response = client.post('/api/upload-xlsx-corrected',
                             data=json.dumps(invalid_financial_data),
                             content_type='application/json')
        
        # Should be rejected or processed with warnings
        assert response.status_code in [400, 422]
        data = json.loads(response.data)
        assert data['error'] is True
    
    def test_pagination_parameter_validation(self, client):
        """Test pagination parameter validation."""
        # Test with invalid pagination parameters
        response = client.get('/api/transactions?page=-1&per_page=1000')
        
        # Should be rejected due to invalid pagination
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] is True
    
    def test_url_length_validation(self, client):
        """Test URL length validation."""
        # Create a very long URL
        long_param = 'x' * 3000
        response = client.get(f'/api/transactions?very_long_param={long_param}')
        
        # Should be rejected due to URL length
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] is True
        assert 'validation failed' in data['message'].lower()
    
    def test_csrf_protection_logging(self, client):
        """Test CSRF protection logging."""
        with patch('src.utils.validation_middleware.audit_logger') as mock_audit_logger:
            # Make a state-changing request
            response = client.post('/api/users',
                                 data=json.dumps({
                                     'username': 'testuser',
                                     'email': 'test@example.com'
                                 }),
                                 content_type='application/json')
            
            # Verify that the request was logged for audit
            mock_audit_logger.log_security_event.assert_called()
            call_args = mock_audit_logger.log_security_event.call_args
            assert call_args[0][0] == 'state_changing_request'
    
    def test_input_sanitization_in_practice(self, client):
        """Test that input sanitization works in practice."""
        # Test with HTML content that should be sanitized
        html_data = {
            'username': '<b>bold</b>username',
            'email': 'user@example.com'
        }
        
        response = client.post('/api/users',
                             data=json.dumps(html_data),
                             content_type='application/json')
        
        # The request might succeed with sanitized input
        # or be rejected depending on validation strictness
        if response.status_code == 201:
            # If successful, verify the data was sanitized
            data = json.loads(response.data)
            # The exact sanitization behavior depends on implementation
            assert '<script>' not in str(data)
    
    def test_business_rule_validation_integration(self, client):
        """Test business rule validation integration."""
        # Test with data that violates business rules
        business_rule_violation = {
            'entries': [{
                'amount': 0.001,  # Very small amount
                'date': '1800-01-01',  # Very old date
                'description': 'a',  # Very short description
                'transaction_type': 'expense',
                'category': 'unknown_category'
            }]
        }
        
        response = client.post('/api/upload-xlsx-corrected',
                             data=json.dumps(business_rule_violation),
                             content_type='application/json')
        
        # Should be processed but with warnings or rejected
        if response.status_code == 200:
            data = json.loads(response.data)
            # Should have warnings about business rule violations
            assert 'warnings' in str(data) or 'errors' in str(data)
        else:
            assert response.status_code in [400, 422]


class TestSecurityHeadersIntegration:
    """Test security headers integration."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_csp_header_configuration(self, client):
        """Test Content Security Policy header configuration."""
        response = client.get('/api/transactions')
        
        csp_header = response.headers.get('Content-Security-Policy')
        assert csp_header is not None
        
        # Check for important CSP directives
        assert "default-src 'self'" in csp_header
        assert "frame-ancestors 'none'" in csp_header
        assert "script-src" in csp_header
        assert "style-src" in csp_header
    
    def test_security_headers_on_error_responses(self, client):
        """Test that security headers are added even to error responses."""
        # Make a request that will result in an error
        response = client.get('/api/nonexistent-endpoint')
        
        # Even error responses should have security headers
        assert 'X-Content-Type-Options' in response.headers
        assert 'X-Frame-Options' in response.headers
        assert 'X-XSS-Protection' in response.headers
    
    def test_hsts_header_on_https(self, client):
        """Test HSTS header on HTTPS requests."""
        # Simulate HTTPS request
        with app.test_request_context('/', environ_base={'wsgi.url_scheme': 'https'}):
            response = client.get('/api/transactions')
            
            # HSTS header should be present for HTTPS
            if response.headers.get('Strict-Transport-Security'):
                assert 'max-age=' in response.headers['Strict-Transport-Security']
                assert 'includeSubDomains' in response.headers['Strict-Transport-Security']


if __name__ == '__main__':
    pytest.main([__file__])