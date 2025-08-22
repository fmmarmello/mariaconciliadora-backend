"""
Validation utilities for Maria Conciliadora application.

This module provides:
- Input validation functions
- File validation utilities
- Data format validators
- Business rule validators
- Security validation (XSS, SQL injection prevention)
- Input sanitization utilities
- Integration with error handling system
"""

import re
import os
import mimetypes
import html
import urllib.parse
import hashlib
import magic
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from decimal import Decimal, InvalidOperation
import pandas as pd

from .exceptions import (
    ValidationError, RequiredFieldError, InvalidFormatError,
    FileSizeExceededError, InvalidFileFormatError, FileCorruptedError
)


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, is_valid: bool = True, errors: Optional[List[str]] = None, warnings: Optional[List[str]] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False


class BaseValidator:
    """Base class for all validators."""
    
    def __init__(self, field_name: str, required: bool = True):
        self.field_name = field_name
        self.required = required
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate a value and return result."""
        result = ValidationResult()
        
        # Check if field is required
        if self.required and (value is None or value == ''):
            raise RequiredFieldError(self.field_name)
        
        # If not required and empty, skip validation
        if not self.required and (value is None or value == ''):
            return result
        
        # Perform specific validation
        return self._validate_value(value)
    
    def _validate_value(self, value: Any) -> ValidationResult:
        """Override this method in subclasses."""
        return ValidationResult()


class StringValidator(BaseValidator):
    """Validator for string fields."""
    
    def __init__(self, field_name: str, min_length: int = 0, max_length: int = None, 
                 pattern: str = None, allowed_values: List[str] = None, **kwargs):
        super().__init__(field_name, **kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None
        self.allowed_values = allowed_values
    
    def _validate_value(self, value: Any) -> ValidationResult:
        result = ValidationResult()
        
        if not isinstance(value, str):
            raise InvalidFormatError(self.field_name, "string")
        
        # Check length
        if len(value) < self.min_length:
            result.add_error(f"Minimum length is {self.min_length} characters")
        
        if self.max_length and len(value) > self.max_length:
            result.add_error(f"Maximum length is {self.max_length} characters")
        
        # Check pattern
        if self.pattern and not self.pattern.match(value):
            result.add_error("Invalid format")
        
        # Check allowed values
        if self.allowed_values and value not in self.allowed_values:
            result.add_error(f"Must be one of: {', '.join(self.allowed_values)}")
        
        return result


class NumberValidator(BaseValidator):
    """Validator for numeric fields."""
    
    def __init__(self, field_name: str, min_value: Union[int, float] = None, 
                 max_value: Union[int, float] = None, decimal_places: int = None, **kwargs):
        super().__init__(field_name, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.decimal_places = decimal_places
    
    def _validate_value(self, value: Any) -> ValidationResult:
        result = ValidationResult()
        
        # Convert to number
        try:
            if isinstance(value, str):
                # Handle Brazilian decimal format (comma as decimal separator)
                value = value.replace(',', '.')
            
            if self.decimal_places is not None:
                num_value = Decimal(str(value))
            else:
                num_value = float(value)
        except (ValueError, InvalidOperation):
            raise InvalidFormatError(self.field_name, "number")
        
        # Check range
        if self.min_value is not None and num_value < self.min_value:
            result.add_error(f"Minimum value is {self.min_value}")
        
        if self.max_value is not None and num_value > self.max_value:
            result.add_error(f"Maximum value is {self.max_value}")
        
        # Check decimal places
        if self.decimal_places is not None:
            decimal_value = Decimal(str(num_value))
            if decimal_value.as_tuple().exponent < -self.decimal_places:
                result.add_error(f"Maximum {self.decimal_places} decimal places allowed")
        
        return result


class DateValidator(BaseValidator):
    """Validator for date fields."""
    
    def __init__(self, field_name: str, min_date: date = None, max_date: date = None, 
                 date_format: str = None, **kwargs):
        super().__init__(field_name, **kwargs)
        self.min_date = min_date
        self.max_date = max_date
        self.date_format = date_format or '%Y-%m-%d'
    
    def _validate_value(self, value: Any) -> ValidationResult:
        result = ValidationResult()
        
        # Convert to date
        if isinstance(value, str):
            try:
                date_value = datetime.strptime(value, self.date_format).date()
            except ValueError:
                raise InvalidFormatError(self.field_name, f"date in format {self.date_format}")
        elif isinstance(value, datetime):
            date_value = value.date()
        elif isinstance(value, date):
            date_value = value
        else:
            raise InvalidFormatError(self.field_name, "date")
        
        # Check range
        if self.min_date and date_value < self.min_date:
            result.add_error(f"Date must be after {self.min_date}")
        
        if self.max_date and date_value > self.max_date:
            result.add_error(f"Date must be before {self.max_date}")
        
        return result


class EmailValidator(BaseValidator):
    """Validator for email fields."""
    
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    def _validate_value(self, value: Any) -> ValidationResult:
        result = ValidationResult()
        
        if not isinstance(value, str):
            raise InvalidFormatError(self.field_name, "email string")
        
        if not self.EMAIL_PATTERN.match(value):
            raise InvalidFormatError(self.field_name, "valid email address")
        
        return result


class FileValidator:
    """Validator for file uploads."""
    
    def __init__(self, allowed_extensions: List[str], max_size_bytes: int, 
                 allowed_mime_types: List[str] = None):
        self.allowed_extensions = [ext.lower() for ext in allowed_extensions]
        self.max_size_bytes = max_size_bytes
        self.allowed_mime_types = allowed_mime_types or []
    
    def validate_file(self, file_path: str, filename: str) -> ValidationResult:
        """Validate an uploaded file."""
        result = ValidationResult()
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise ValidationError(f"File not found: {filename}")
        
        # Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in [f'.{ext}' for ext in self.allowed_extensions]:
            raise InvalidFileFormatError(filename, self.allowed_extensions)
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_size_bytes:
            raise FileSizeExceededError(filename, file_size, self.max_size_bytes)
        
        # Check MIME type if specified
        if self.allowed_mime_types:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type not in self.allowed_mime_types:
                result.add_error(f"Invalid file type. Allowed types: {', '.join(self.allowed_mime_types)}")
        
        # Additional file content validation
        try:
            self._validate_file_content(file_path, file_ext)
        except Exception as e:
            raise FileCorruptedError(filename)
        
        return result
    
    def _validate_file_content(self, file_path: str, file_ext: str):
        """Validate file content based on extension."""
        if file_ext in ['.xlsx', '.xls']:
            # Try to read Excel file
            pd.read_excel(file_path, nrows=1)
        elif file_ext in ['.ofx', '.qfx']:
            # Try to read OFX file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(100)  # Read first 100 characters
                if '<OFX>' not in content.upper() and 'OFXHEADER' not in content.upper():
                    raise ValueError("Invalid OFX format")


class TransactionValidator:
    """Validator for financial transaction data."""
    
    def __init__(self):
        self.amount_validator = NumberValidator('amount', min_value=-999999999, max_value=999999999, decimal_places=2)
        self.date_validator = DateValidator('date', min_date=date(1900, 1, 1), max_date=date.today())
        self.description_validator = StringValidator('description', min_length=1, max_length=500)
    
    def validate_transaction(self, transaction_data: Dict[str, Any]) -> ValidationResult:
        """Validate transaction data."""
        result = ValidationResult()
        
        # Validate required fields
        required_fields = ['amount', 'date', 'description']
        for field in required_fields:
            if field not in transaction_data or transaction_data[field] is None:
                result.add_error(f"Required field '{field}' is missing")
        
        if not result.is_valid:
            return result
        
        # Validate individual fields
        try:
            amount_result = self.amount_validator.validate(transaction_data['amount'])
            result.merge(amount_result)
        except ValidationError as e:
            result.add_error(f"Amount: {e.user_message}")
        
        try:
            date_result = self.date_validator.validate(transaction_data['date'])
            result.merge(date_result)
        except ValidationError as e:
            result.add_error(f"Date: {e.user_message}")
        
        try:
            desc_result = self.description_validator.validate(transaction_data['description'])
            result.merge(desc_result)
        except ValidationError as e:
            result.add_error(f"Description: {e.user_message}")
        
        # Business rule validations
        if transaction_data.get('amount') == 0:
            result.add_warning("Transaction amount is zero")
        
        return result


class CompanyFinancialValidator:
    """Validator for company financial data."""
    
    def __init__(self):
        self.amount_validator = NumberValidator('amount', decimal_places=2)
        self.date_validator = DateValidator('date', min_date=date(1900, 1, 1))
        self.description_validator = StringValidator('description', min_length=1, max_length=500)
        self.category_validator = StringValidator('category', max_length=100)
        self.transaction_type_validator = StringValidator('transaction_type', allowed_values=['income', 'expense'])
    
    def validate_entry(self, entry_data: Dict[str, Any]) -> ValidationResult:
        """Validate company financial entry."""
        result = ValidationResult()
        
        # Validate required fields
        required_fields = ['amount', 'date', 'description', 'transaction_type']
        for field in required_fields:
            if field not in entry_data or entry_data[field] is None:
                result.add_error(f"Required field '{field}' is missing")
        
        if not result.is_valid:
            return result
        
        # Validate individual fields
        validators = [
            ('amount', self.amount_validator),
            ('date', self.date_validator),
            ('description', self.description_validator),
            ('transaction_type', self.transaction_type_validator)
        ]
        
        for field_name, validator in validators:
            if field_name in entry_data:
                try:
                    field_result = validator.validate(entry_data[field_name])
                    result.merge(field_result)
                except ValidationError as e:
                    result.add_error(f"{field_name.title()}: {e.user_message}")
        
        # Optional field validation
        if 'category' in entry_data and entry_data['category']:
            try:
                cat_result = self.category_validator.validate(entry_data['category'])
                result.merge(cat_result)
            except ValidationError as e:
                result.add_error(f"Category: {e.user_message}")
        
        # Business rule validations
        if entry_data.get('transaction_type') == 'expense' and entry_data.get('amount', 0) > 0:
            result.add_warning("Expense amount is positive - this might be incorrect")
        
        if entry_data.get('transaction_type') == 'income' and entry_data.get('amount', 0) < 0:
            result.add_warning("Income amount is negative - this might be incorrect")
        
        return result


# Validation utility functions
def validate_request_data(data: Dict[str, Any], validators: Dict[str, BaseValidator]) -> ValidationResult:
    """
    Validate request data using provided validators.
    
    Args:
        data: Data to validate
        validators: Dictionary mapping field names to validators
        
    Returns:
        ValidationResult with all validation results
    """
    result = ValidationResult()
    
    for field_name, validator in validators.items():
        field_value = data.get(field_name)
        try:
            field_result = validator.validate(field_value)
            result.merge(field_result)
        except ValidationError as e:
            result.add_error(f"{field_name}: {e.user_message}")
    
    return result


def validate_file_upload(file_path: str, filename: str, file_type: str) -> ValidationResult:
    """
    Validate file upload based on file type.
    
    Args:
        file_path: Path to the uploaded file
        filename: Original filename
        file_type: Type of file ('ofx', 'xlsx', etc.)
        
    Returns:
        ValidationResult
    """
    # Define file type configurations
    file_configs = {
        'ofx': {
            'extensions': ['ofx', 'qfx'],
            'max_size': 16 * 1024 * 1024,  # 16MB
            'mime_types': ['application/x-ofx']
        },
        'xlsx': {
            'extensions': ['xlsx'],
            'max_size': 10 * 1024 * 1024,  # 10MB
            'mime_types': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
        }
    }
    
    if file_type not in file_configs:
        raise ValidationError(f"Unsupported file type: {file_type}")
    
    config = file_configs[file_type]
    validator = FileValidator(
        allowed_extensions=config['extensions'],
        max_size_bytes=config['max_size'],
        allowed_mime_types=config['mime_types']
    )
    
    return validator.validate_file(file_path, filename)


# Security Validation Classes
class SecurityValidator:
    """Comprehensive security validation utilities."""
    
    # XSS patterns to detect
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>.*?</iframe>',
        r'<object[^>]*>.*?</object>',
        r'<embed[^>]*>.*?</embed>',
        r'<link[^>]*>',
        r'<meta[^>]*>',
        r'<style[^>]*>.*?</style>',
        r'expression\s*\(',
        r'url\s*\(',
        r'@import',
        r'vbscript:',
        r'data:text/html'
    ]
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)',
        r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
        r'(\b(OR|AND)\s+[\'"]?\w+[\'"]?\s*=\s*[\'"]?\w+[\'"]?)',
        r'(--|#|/\*|\*/)',
        r'(\bxp_\w+)',
        r'(\bsp_\w+)',
        r'(\bEXEC\s*\()',
        r'(\bCAST\s*\()',
        r'(\bCONVERT\s*\()',
        r'(\bCHAR\s*\()',
        r'(\bASCII\s*\()',
        r'(\bSUBSTRING\s*\()',
        r'(\bLEN\s*\()',
        r'(\bLENGTH\s*\()',
        r'(\bCOUNT\s*\()',
        r'(\bSUM\s*\()',
        r'(\bAVG\s*\()',
        r'(\bMIN\s*\()',
        r'(\bMAX\s*\()',
        r'(\bGROUP\s+BY)',
        r'(\bORDER\s+BY)',
        r'(\bHAVING\b)',
        r'(\bLIMIT\b)',
        r'(\bOFFSET\b)',
        r'(\bINTO\s+OUTFILE)',
        r'(\bLOAD_FILE\s*\()',
        r'(\bINTO\s+DUMPFILE)',
        r'(\bBENCHMARK\s*\()',
        r'(\bSLEEP\s*\()',
        r'(\bWAITFOR\s+DELAY)',
        r'(\bpg_sleep\s*\()',
        r'(\bdbms_pipe\.receive_message)',
        r'(\bdbms_lock\.sleep)',
        r'(\bUTL_INADDR\.get_host_name)',
        r'(\bUTL_HTTP\.request)',
        r'(\bDBMS_XMLQUERY\.newcontext)',
        r'(\bextractvalue\s*\()',
        r'(\bupdatexml\s*\()',
        r'(\bxmltype\s*\()',
        r'(\bINFORMATION_SCHEMA)',
        r'(\bPERFORMANCE_SCHEMA)',
        r'(\bSYS\b)',
        r'(\bMYSQL\b)',
        r'(\bPG_)',
        r'(\bPOSTGRES)',
        r'(\bSQLITE_)',
        r'(\bORACLE)',
        r'(\bMSSQL)',
        r'(\bSQLSERVER)'
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\./',
        r'\.\.\\',
        r'%2e%2e%2f',
        r'%2e%2e%5c',
        r'%252e%252e%252f',
        r'%252e%252e%255c',
        r'\.\.%2f',
        r'\.\.%5c',
        r'%2e%2e/',
        r'%2e%2e\\',
        r'..%c0%af',
        r'..%c1%9c'
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r'[;&|`$()]',
        r'\b(cat|ls|dir|type|copy|move|del|rm|mkdir|rmdir|cd|pwd|whoami|id|ps|kill|chmod|chown|su|sudo|passwd|crontab|at|batch|nohup|screen|tmux)\b',
        r'(\||&&|\|\||;|`|\$\(|\${)',
        r'(>|>>|<|<<)',
        r'(\beval\b|\bexec\b|\bsystem\b|\bshell_exec\b|\bpassthru\b|\bpopen\b)'
    ]
    
    @classmethod
    def detect_xss(cls, value: str) -> bool:
        """Detect potential XSS attacks."""
        if not isinstance(value, str):
            return False
        
        value_lower = value.lower()
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE | re.DOTALL):
                return True
        return False
    
    @classmethod
    def detect_sql_injection(cls, value: str) -> bool:
        """Detect potential SQL injection attacks."""
        if not isinstance(value, str):
            return False
        
        value_upper = value.upper()
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_upper, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def detect_path_traversal(cls, value: str) -> bool:
        """Detect potential path traversal attacks."""
        if not isinstance(value, str):
            return False
        
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def detect_command_injection(cls, value: str) -> bool:
        """Detect potential command injection attacks."""
        if not isinstance(value, str):
            return False
        
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def validate_input_security(cls, value: str, field_name: str = "input") -> ValidationResult:
        """Comprehensive security validation for input."""
        result = ValidationResult()
        
        if not isinstance(value, str):
            return result
        
        # Check for XSS
        if cls.detect_xss(value):
            result.add_error(f"Potential XSS attack detected in {field_name}")
        
        # Check for SQL injection
        if cls.detect_sql_injection(value):
            result.add_error(f"Potential SQL injection detected in {field_name}")
        
        # Check for path traversal
        if cls.detect_path_traversal(value):
            result.add_error(f"Potential path traversal attack detected in {field_name}")
        
        # Check for command injection
        if cls.detect_command_injection(value):
            result.add_error(f"Potential command injection detected in {field_name}")
        
        return result


class InputSanitizer:
    """Comprehensive input sanitization utilities."""
    
    @staticmethod
    def sanitize_html(value: str, allowed_tags: List[str] = None) -> str:
        """Sanitize HTML input to prevent XSS."""
        if not isinstance(value, str):
            return str(value)
        
        # HTML escape all content by default
        sanitized = html.escape(value)
        
        # If specific tags are allowed, implement whitelist approach
        if allowed_tags:
            # This is a basic implementation - for production use a library like bleach
            for tag in allowed_tags:
                # Allow only specific safe tags
                if tag in ['b', 'i', 'u', 'strong', 'em']:
                    sanitized = re.sub(f'&lt;{tag}&gt;', f'<{tag}>', sanitized)
                    sanitized = re.sub(f'&lt;/{tag}&gt;', f'</{tag}>', sanitized)
        
        return sanitized
    
    @staticmethod
    def sanitize_sql_input(value: str) -> str:
        """Sanitize input to prevent SQL injection."""
        if not isinstance(value, str):
            return str(value)
        
        # Remove or escape dangerous SQL characters
        sanitized = value.replace("'", "''")  # Escape single quotes
        sanitized = re.sub(r'[;\\]', '', sanitized)  # Remove semicolons and backslashes
        sanitized = re.sub(r'--.*$', '', sanitized, flags=re.MULTILINE)  # Remove SQL comments
        sanitized = re.sub(r'/\*.*?\*/', '', sanitized, flags=re.DOTALL)  # Remove block comments
        
        return sanitized.strip()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal and other attacks."""
        if not isinstance(filename, str):
            return str(filename)
        
        # Remove path traversal attempts
        sanitized = re.sub(r'\.\./', '', filename)
        sanitized = re.sub(r'\.\.\\', '', sanitized)
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>:"|?*]', '', sanitized)
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        
        # Ensure it's not empty or just dots
        if not sanitized or sanitized in ['.', '..']:
            sanitized = 'file'
        
        return sanitized.strip()
    
    @staticmethod
    def sanitize_path(path: str) -> str:
        """Sanitize file path to prevent path traversal attacks."""
        if not isinstance(path, str):
            return str(path)
        
        # Normalize path separators
        sanitized = path.replace('\\', '/')
        
        # Remove path traversal attempts
        sanitized = re.sub(r'\.\./', '', sanitized)
        sanitized = re.sub(r'/+', '/', sanitized)  # Remove multiple slashes
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>:"|?*]', '', sanitized)
        
        # Ensure it doesn't start with /
        if sanitized.startswith('/'):
            sanitized = sanitized[1:]
        
        return sanitized.strip()
    
    @staticmethod
    def sanitize_general_input(value: str, max_length: int = 1000, allow_html: bool = False) -> str:
        """General purpose input sanitization."""
        if not isinstance(value, str):
            return str(value)
        
        if allow_html:
            # Allow basic HTML tags but escape dangerous content
            sanitized = InputSanitizer.sanitize_html(value, ['b', 'i', 'u', 'strong', 'em'])
        else:
            # Full HTML escape
            sanitized = html.escape(value)
        
        # Remove control characters except newlines and tabs
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize URL input."""
        if not isinstance(url, str):
            return str(url)
        
        # URL encode dangerous characters
        sanitized = urllib.parse.quote(url, safe=':/?#[]@!$&\'()*+,;=')
        
        # Ensure it starts with http:// or https://
        if not sanitized.startswith(('http://', 'https://')):
            if sanitized.startswith('//'):
                sanitized = 'https:' + sanitized
            elif not sanitized.startswith(('ftp://', 'mailto:')):
                sanitized = 'https://' + sanitized
        
        return sanitized


class EnhancedFileValidator(FileValidator):
    """Enhanced file validator with additional security checks."""
    
    def __init__(self, allowed_extensions: List[str], max_size_bytes: int,
                 allowed_mime_types: List[str] = None, check_content: bool = True):
        super().__init__(allowed_extensions, max_size_bytes, allowed_mime_types)
        self.check_content = check_content
    
    def validate_file(self, file_path: str, filename: str) -> ValidationResult:
        """Enhanced file validation with security checks."""
        result = super().validate_file(file_path, filename)
        
        if not result.is_valid:
            return result
        
        # Additional security validations
        try:
            # Sanitize filename
            sanitized_filename = InputSanitizer.sanitize_filename(filename)
            if sanitized_filename != filename:
                result.add_warning("Filename was sanitized for security")
            
            # Check file content type using python-magic if available
            if self.check_content:
                try:
                    import magic
                    file_type = magic.from_file(file_path, mime=True)
                    if self.allowed_mime_types and file_type not in self.allowed_mime_types:
                        result.add_error(f"File content type {file_type} doesn't match allowed types")
                except ImportError:
                    # python-magic not available, skip content type check
                    pass
                except Exception as e:
                    result.add_warning(f"Could not verify file content type: {str(e)}")
            
            # Check for embedded scripts in files
            if self._check_for_embedded_scripts(file_path):
                result.add_error("File contains potentially malicious embedded scripts")
            
            # Validate file hash for integrity
            file_hash = self._calculate_file_hash(file_path)
            if not file_hash:
                result.add_error("Could not calculate file hash for integrity check")
            
        except Exception as e:
            result.add_error(f"Security validation failed: {str(e)}")
        
        return result
    
    def _check_for_embedded_scripts(self, file_path: str) -> bool:
        """Check for embedded scripts in files."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read(8192)  # Read first 8KB
                content_str = content.decode('utf-8', errors='ignore').lower()
                
                # Check for script tags and javascript
                dangerous_patterns = [
                    '<script', 'javascript:', 'vbscript:', 'onload=', 'onerror=',
                    'eval(', 'document.', 'window.', 'alert(', 'confirm('
                ]
                
                for pattern in dangerous_patterns:
                    if pattern in content_str:
                        return True
                        
        except Exception:
            # If we can't read the file, assume it's safe
            pass
        
        return False
    
    def _calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculate SHA-256 hash of file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return None


def sanitize_input(value: str, max_length: int = 1000, allow_html: bool = False) -> str:
    """
    Sanitize input string to prevent injection attacks.
    
    Args:
        value: Input string to sanitize
        max_length: Maximum allowed length
        allow_html: Whether to allow basic HTML tags
        
    Returns:
        Sanitized string
    """
    return InputSanitizer.sanitize_general_input(value, max_length, allow_html)


def validate_input_security(value: str, field_name: str = "input") -> ValidationResult:
    """
    Validate input for security threats.
    
    Args:
        value: Input string to validate
        field_name: Name of the field being validated
        
    Returns:
        ValidationResult with security validation results
    """
    return SecurityValidator.validate_input_security(value, field_name)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent security issues.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    return InputSanitizer.sanitize_filename(filename)


def sanitize_path(path: str) -> str:
    """
    Sanitize file path to prevent path traversal attacks.
    
    Args:
        path: Original path
        
    Returns:
        Sanitized path
    """
    return InputSanitizer.sanitize_path(path)


def validate_pagination_params(page: Any, per_page: Any) -> Dict[str, int]:
    """
    Validate and normalize pagination parameters.
    
    Args:
        page: Page number
        per_page: Items per page
        
    Returns:
        Dictionary with validated page and per_page values
    """
    try:
        page = int(page) if page else 1
        per_page = int(per_page) if per_page else 20
    except (ValueError, TypeError):
        raise ValidationError("Page and per_page must be integers")
    
    if page < 1:
        raise ValidationError("Page must be greater than 0")
    
    if per_page < 1 or per_page > 100:
        raise ValidationError("Per page must be between 1 and 100")
    
    return {'page': page, 'per_page': per_page}


class RateLimitValidator:
    """Rate limiting validation utilities."""
    
    def __init__(self):
        self.request_counts = {}  # In production, use Redis or similar
        self.blocked_ips = set()
    
    def validate_rate_limit(self, identifier: str, max_requests: int = 100,
                          window_minutes: int = 60) -> ValidationResult:
        """Validate rate limiting for requests."""
        result = ValidationResult()
        
        if identifier in self.blocked_ips:
            result.add_error("IP address is temporarily blocked due to excessive requests")
            return result
        
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=window_minutes)
        
        # Clean old entries
        if identifier in self.request_counts:
            self.request_counts[identifier] = [
                req_time for req_time in self.request_counts[identifier]
                if req_time > window_start
            ]
        else:
            self.request_counts[identifier] = []
        
        # Check current request count
        current_count = len(self.request_counts[identifier])
        
        if current_count >= max_requests:
            result.add_error(f"Rate limit exceeded: {current_count}/{max_requests} requests in {window_minutes} minutes")
            # Block IP for repeated violations
            if current_count >= max_requests * 2:
                self.blocked_ips.add(identifier)
        else:
            # Add current request
            self.request_counts[identifier].append(current_time)
        
        return result
    
    def unblock_ip(self, identifier: str):
        """Unblock an IP address."""
        self.blocked_ips.discard(identifier)


class RequestValidator:
    """Request validation utilities."""
    
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_HEADER_SIZE = 8192  # 8KB
    MAX_URL_LENGTH = 2048
    
    @staticmethod
    def validate_request_size(content_length: int) -> ValidationResult:
        """Validate request content size."""
        result = ValidationResult()
        
        if content_length > RequestValidator.MAX_REQUEST_SIZE:
            result.add_error(f"Request size {content_length} exceeds maximum allowed size {RequestValidator.MAX_REQUEST_SIZE}")
        
        return result
    
    @staticmethod
    def validate_headers(headers: Dict[str, str]) -> ValidationResult:
        """Validate request headers."""
        result = ValidationResult()
        
        # Check total header size
        total_header_size = sum(len(k) + len(v) for k, v in headers.items())
        if total_header_size > RequestValidator.MAX_HEADER_SIZE:
            result.add_error(f"Total header size {total_header_size} exceeds maximum {RequestValidator.MAX_HEADER_SIZE}")
        
        # Validate individual headers
        for key, value in headers.items():
            # Check for header injection
            if '\n' in key or '\r' in key or '\n' in value or '\r' in value:
                result.add_error(f"Header injection detected in {key}")
            
            # Validate specific headers
            if key.lower() == 'content-type':
                if not RequestValidator._validate_content_type(value):
                    result.add_error(f"Invalid content-type: {value}")
            
            elif key.lower() == 'user-agent':
                if len(value) > 512:
                    result.add_error("User-Agent header too long")
        
        return result
    
    @staticmethod
    def validate_url(url: str) -> ValidationResult:
        """Validate request URL."""
        result = ValidationResult()
        
        if len(url) > RequestValidator.MAX_URL_LENGTH:
            result.add_error(f"URL length {len(url)} exceeds maximum {RequestValidator.MAX_URL_LENGTH}")
        
        # Check for URL injection attempts
        if SecurityValidator.detect_xss(url) or SecurityValidator.detect_sql_injection(url):
            result.add_error("Malicious content detected in URL")
        
        return result
    
    @staticmethod
    def _validate_content_type(content_type: str) -> bool:
        """Validate content-type header."""
        allowed_types = [
            'application/json',
            'application/x-www-form-urlencoded',
            'multipart/form-data',
            'text/plain',
            'application/octet-stream',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            'application/x-ofx'
        ]
        
        # Extract main content type (ignore charset and other parameters)
        main_type = content_type.split(';')[0].strip().lower()
        return main_type in allowed_types


class BusinessRuleValidator:
    """Enhanced business rule validation for financial data."""
    
    # Financial limits
    MAX_TRANSACTION_AMOUNT = Decimal('999999999.99')
    MIN_TRANSACTION_AMOUNT = Decimal('-999999999.99')
    MAX_DAILY_TRANSACTIONS = 10000
    MAX_MONTHLY_AMOUNT = Decimal('100000000.00')
    
    # Date limits
    MIN_TRANSACTION_DATE = date(1900, 1, 1)
    MAX_FUTURE_DATE_DAYS = 30
    
    @classmethod
    def validate_financial_transaction(cls, transaction_data: Dict[str, Any]) -> ValidationResult:
        """Comprehensive financial transaction validation."""
        result = ValidationResult()
        
        # Amount validation
        amount = transaction_data.get('amount')
        if amount is not None:
            try:
                amount_decimal = Decimal(str(amount))
                
                if amount_decimal > cls.MAX_TRANSACTION_AMOUNT:
                    result.add_error(f"Transaction amount exceeds maximum allowed: {cls.MAX_TRANSACTION_AMOUNT}")
                
                if amount_decimal < cls.MIN_TRANSACTION_AMOUNT:
                    result.add_error(f"Transaction amount below minimum allowed: {cls.MIN_TRANSACTION_AMOUNT}")
                
                # Check for suspicious round numbers
                if amount_decimal > 100000 and amount_decimal % 1000 == 0:
                    result.add_warning("Large round number transaction - please verify")
                
            except (ValueError, InvalidOperation):
                result.add_error("Invalid amount format")
        
        # Date validation
        transaction_date = transaction_data.get('date')
        if transaction_date:
            try:
                if isinstance(transaction_date, str):
                    transaction_date = datetime.fromisoformat(transaction_date).date()
                elif isinstance(transaction_date, datetime):
                    transaction_date = transaction_date.date()
                
                if transaction_date < cls.MIN_TRANSACTION_DATE:
                    result.add_error(f"Transaction date too old: {transaction_date}")
                
                max_future_date = date.today() + timedelta(days=cls.MAX_FUTURE_DATE_DAYS)
                if transaction_date > max_future_date:
                    result.add_error(f"Transaction date too far in future: {transaction_date}")
                
            except (ValueError, TypeError):
                result.add_error("Invalid date format")
        
        # Description validation
        description = transaction_data.get('description', '')
        if description:
            # Check for suspicious patterns
            suspicious_patterns = [
                r'\b(test|teste|debug)\b',
                r'\b(hack|exploit|inject)\b',
                r'^[0-9]+$',  # Only numbers
                r'^[a-z]$',   # Single character
                r'(.)\1{10,}' # Repeated characters
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, description.lower()):
                    result.add_warning(f"Suspicious description pattern detected: {description}")
                    break
        
        # Bank name validation
        bank_name = transaction_data.get('bank_name', '')
        if bank_name:
            valid_banks = [
                'caixa', 'sicoob', 'nubank', 'itau', 'bradesco',
                'santander', 'bb', 'banco do brasil', 'unknown'
            ]
            if bank_name.lower() not in valid_banks:
                result.add_warning(f"Unknown bank name: {bank_name}")
        
        return result
    
    @classmethod
    def validate_company_financial_entry(cls, entry_data: Dict[str, Any]) -> ValidationResult:
        """Validate company financial entry with business rules."""
        result = ValidationResult()
        
        # Use existing company financial validator first
        base_result = company_financial_validator.validate_entry(entry_data)
        result.merge(base_result)
        
        # Additional business validations
        transaction_type = entry_data.get('transaction_type')
        amount = entry_data.get('amount')
        
        if transaction_type and amount is not None:
            try:
                amount_decimal = Decimal(str(amount))
                
                # Business logic checks
                if transaction_type == 'expense' and amount_decimal > 0:
                    # Expenses should typically be negative or we should warn
                    result.add_warning("Expense with positive amount - please verify")
                
                if transaction_type == 'income' and amount_decimal < 0:
                    # Income should typically be positive or we should warn
                    result.add_warning("Income with negative amount - please verify")
                
                # Check for unusually large amounts
                if abs(amount_decimal) > 1000000:  # 1 million
                    result.add_warning("Unusually large amount - please verify")
                
            except (ValueError, InvalidOperation):
                pass  # Already handled by base validator
        
        # Category validation
        category = entry_data.get('category', '')
        if category:
            # Define valid categories
            valid_categories = [
                'office_supplies', 'travel', 'meals', 'utilities', 'rent',
                'salaries', 'taxes', 'insurance', 'marketing', 'equipment',
                'software', 'consulting', 'legal', 'accounting', 'other'
            ]
            
            if category.lower().replace(' ', '_') not in valid_categories:
                result.add_warning(f"Non-standard category: {category}")
        
        return result


def validate_api_request(request_data: Dict[str, Any], headers: Dict[str, str],
                        url: str, content_length: int = 0,
                        client_ip: str = None) -> ValidationResult:
    """
    Comprehensive API request validation.
    
    Args:
        request_data: Request payload data
        headers: Request headers
        url: Request URL
        content_length: Content length in bytes
        client_ip: Client IP address for rate limiting
        
    Returns:
        ValidationResult with all validation results
    """
    result = ValidationResult()
    
    # Request size validation
    size_result = RequestValidator.validate_request_size(content_length)
    result.merge(size_result)
    
    # Header validation
    header_result = RequestValidator.validate_headers(headers)
    result.merge(header_result)
    
    # URL validation
    url_result = RequestValidator.validate_url(url)
    result.merge(url_result)
    
    # Rate limiting (if IP provided)
    if client_ip:
        rate_limiter = RateLimitValidator()
        rate_result = rate_limiter.validate_rate_limit(client_ip)
        result.merge(rate_result)
    
    # Input security validation for all string values in request data
    if isinstance(request_data, dict):
        for key, value in request_data.items():
            if isinstance(value, str):
                security_result = validate_input_security(value, key)
                result.merge(security_result)
    
    return result


# Global validator instances
transaction_validator = TransactionValidator()
company_financial_validator = CompanyFinancialValidator()
rate_limiter = RateLimitValidator()
business_rule_validator = BusinessRuleValidator()