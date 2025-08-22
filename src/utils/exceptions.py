"""
Custom exception classes for Maria Conciliadora application.

This module defines a hierarchy of custom exceptions that provide:
- Structured error information
- Proper HTTP status codes
- User-friendly error messages
- Developer-friendly error details
- Integration with logging system
"""

from typing import Optional, Dict, Any, List
from enum import Enum


class ErrorCategory(Enum):
    """Categories of errors for better classification and handling."""
    VALIDATION = "validation"
    FILE_PROCESSING = "file_processing"
    DATABASE = "database"
    AI_SERVICE = "ai_service"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM = "system"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BaseApplicationError(Exception):
    """
    Base exception class for all application-specific errors.
    
    Provides structured error information including:
    - HTTP status code
    - Error category and severity
    - User-friendly and developer messages
    - Additional context data
    """
    
    def __init__(
        self,
        message: str,
        user_message: Optional[str] = None,
        status_code: int = 500,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.user_message = user_message or self._get_default_user_message()
        self.status_code = status_code
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.suggestions = suggestions or []
        self.error_code = error_code or self._generate_error_code()
    
    def _get_default_user_message(self) -> str:
        """Generate a default user-friendly message."""
        return "Ocorreu um erro interno. Tente novamente em alguns instantes."
    
    def _generate_error_code(self) -> str:
        """Generate a unique error code for this exception type."""
        return f"{self.category.value.upper()}_{self.__class__.__name__.upper()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            'error': True,
            'error_code': self.error_code,
            'message': self.user_message,
            'category': self.category.value,
            'severity': self.severity.value,
            'details': self.details,
            'suggestions': self.suggestions,
            'developer_message': self.message
        }


# Validation Errors
class ValidationError(BaseApplicationError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        user_message: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            user_message=user_message or "Dados inválidos fornecidos.",
            status_code=400,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )
        if field:
            self.details['field'] = field
        if value is not None:
            self.details['value'] = str(value)


class RequiredFieldError(ValidationError):
    """Raised when a required field is missing."""
    
    def __init__(self, field: str, **kwargs):
        super().__init__(
            message=f"Required field '{field}' is missing",
            field=field,
            user_message=f"O campo '{field}' é obrigatório.",
            **kwargs
        )


class InvalidFormatError(ValidationError):
    """Raised when data format is invalid."""
    
    def __init__(self, field: str, expected_format: str, **kwargs):
        super().__init__(
            message=f"Invalid format for field '{field}', expected: {expected_format}",
            field=field,
            user_message=f"Formato inválido para o campo '{field}'. Formato esperado: {expected_format}.",
            **kwargs
        )
        self.details['expected_format'] = expected_format


# File Processing Errors
class FileProcessingError(BaseApplicationError):
    """Base class for file processing errors."""
    
    def __init__(self, message: str, filename: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            status_code=422,
            category=ErrorCategory.FILE_PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        if filename:
            self.details['filename'] = filename


class FileNotFoundError(FileProcessingError):
    """Raised when a file is not found."""
    
    def __init__(self, filename: str, **kwargs):
        super().__init__(
            message=f"File not found: {filename}",
            filename=filename,
            user_message="Arquivo não encontrado.",
            status_code=404,
            **kwargs
        )


class InvalidFileFormatError(FileProcessingError):
    """Raised when file format is invalid or unsupported."""
    
    def __init__(self, filename: str, expected_formats: List[str], **kwargs):
        formats_str = ", ".join(expected_formats)
        super().__init__(
            message=f"Invalid file format for {filename}. Expected: {formats_str}",
            filename=filename,
            user_message=f"Formato de arquivo inválido. Formatos aceitos: {formats_str}",
            **kwargs
        )
        self.details['expected_formats'] = expected_formats


class FileSizeExceededError(FileProcessingError):
    """Raised when file size exceeds limits."""
    
    def __init__(self, filename: str, size: int, max_size: int, **kwargs):
        super().__init__(
            message=f"File size exceeded for {filename}: {size} bytes (max: {max_size} bytes)",
            filename=filename,
            user_message=f"Arquivo muito grande. Tamanho máximo permitido: {max_size // (1024*1024)}MB",
            **kwargs
        )
        self.details.update({
            'file_size': size,
            'max_size': max_size
        })


class FileCorruptedError(FileProcessingError):
    """Raised when file is corrupted or cannot be parsed."""
    
    def __init__(self, filename: str, **kwargs):
        super().__init__(
            message=f"File corrupted or cannot be parsed: {filename}",
            filename=filename,
            user_message="Arquivo corrompido ou não pode ser processado.",
            **kwargs
        )


class DuplicateFileError(FileProcessingError):
    """Raised when a duplicate file is detected."""
    
    def __init__(self, filename: str, original_upload_date: Optional[str] = None, **kwargs):
        super().__init__(
            message=f"Duplicate file detected: {filename}",
            filename=filename,
            user_message="Este arquivo já foi processado anteriormente.",
            status_code=409,
            **kwargs
        )
        if original_upload_date:
            self.details['original_upload_date'] = original_upload_date


# Database Errors
class DatabaseError(BaseApplicationError):
    """Base class for database-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            user_message="Erro no banco de dados. Tente novamente.",
            status_code=500,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    def __init__(self, **kwargs):
        super().__init__(
            message="Database connection failed",
            user_message="Não foi possível conectar ao banco de dados. Tente novamente em alguns instantes.",
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class DatabaseTransactionError(DatabaseError):
    """Raised when database transaction fails."""
    
    def __init__(self, operation: str, **kwargs):
        super().__init__(
            message=f"Database transaction failed during {operation}",
            user_message="Erro ao processar dados. A operação foi cancelada.",
            **kwargs
        )
        self.details['operation'] = operation


class DatabaseConstraintError(DatabaseError):
    """Raised when database constraint is violated."""
    
    def __init__(self, constraint: str, **kwargs):
        super().__init__(
            message=f"Database constraint violation: {constraint}",
            user_message="Violação de regra de integridade dos dados.",
            status_code=409,
            **kwargs
        )
        self.details['constraint'] = constraint


class RecordNotFoundError(DatabaseError):
    """Raised when a database record is not found."""
    
    def __init__(self, model: str, identifier: Any, **kwargs):
        super().__init__(
            message=f"{model} not found with identifier: {identifier}",
            user_message="Registro não encontrado.",
            status_code=404,
            **kwargs
        )
        self.details.update({
            'model': model,
            'identifier': str(identifier)
        })


# AI Service Errors
class AIServiceError(BaseApplicationError):
    """Base class for AI service errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            user_message="Erro no serviço de IA. Funcionalidade limitada disponível.",
            status_code=503,
            category=ErrorCategory.AI_SERVICE,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class AIServiceUnavailableError(AIServiceError):
    """Raised when AI service is unavailable."""
    
    def __init__(self, service_name: str, **kwargs):
        super().__init__(
            message=f"AI service unavailable: {service_name}",
            user_message="Serviço de IA temporariamente indisponível.",
            **kwargs
        )
        self.details['service_name'] = service_name


class AIServiceTimeoutError(AIServiceError):
    """Raised when AI service times out."""
    
    def __init__(self, service_name: str, timeout_seconds: int, **kwargs):
        super().__init__(
            message=f"AI service timeout: {service_name} ({timeout_seconds}s)",
            user_message="Tempo limite excedido para processamento com IA.",
            **kwargs
        )
        self.details.update({
            'service_name': service_name,
            'timeout_seconds': timeout_seconds
        })


class AIServiceQuotaExceededError(AIServiceError):
    """Raised when AI service quota is exceeded."""
    
    def __init__(self, service_name: str, **kwargs):
        super().__init__(
            message=f"AI service quota exceeded: {service_name}",
            user_message="Limite de uso do serviço de IA excedido. Tente novamente mais tarde.",
            status_code=429,
            **kwargs
        )
        self.details['service_name'] = service_name


# Business Logic Errors
class BusinessLogicError(BaseApplicationError):
    """Base class for business logic errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            status_code=422,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class InsufficientDataError(BusinessLogicError):
    """Raised when there's insufficient data for an operation."""
    
    def __init__(self, operation: str, minimum_required: int, actual: int, **kwargs):
        super().__init__(
            message=f"Insufficient data for {operation}: need {minimum_required}, got {actual}",
            user_message=f"Dados insuficientes para {operation}. Mínimo necessário: {minimum_required}",
            **kwargs
        )
        self.details.update({
            'operation': operation,
            'minimum_required': minimum_required,
            'actual': actual
        })


class ReconciliationError(BusinessLogicError):
    """Raised when reconciliation process fails."""
    
    def __init__(self, reason: str, **kwargs):
        super().__init__(
            message=f"Reconciliation failed: {reason}",
            user_message="Falha no processo de reconciliação.",
            **kwargs
        )
        self.details['reason'] = reason


# System Errors
class SystemError(BaseApplicationError):
    """Base class for system-level errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            user_message="Erro interno do sistema. Tente novamente.",
            status_code=500,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class TimeoutError(SystemError):
    """Raised when an operation times out."""
    
    def __init__(self, operation: str, timeout_seconds: int, **kwargs):
        super().__init__(
            message=f"Operation timeout: {operation} ({timeout_seconds}s)",
            user_message="Operação demorou muito para ser concluída. Tente novamente.",
            status_code=408,
            category=ErrorCategory.TIMEOUT,
            **kwargs
        )
        self.details.update({
            'operation': operation,
            'timeout_seconds': timeout_seconds
        })


class ResourceLimitError(SystemError):
    """Raised when system resource limits are exceeded."""
    
    def __init__(self, resource: str, limit: Any, current: Any, **kwargs):
        super().__init__(
            message=f"Resource limit exceeded for {resource}: {current} (limit: {limit})",
            user_message="Limite de recursos do sistema excedido. Tente novamente mais tarde.",
            status_code=429,
            category=ErrorCategory.RESOURCE_LIMIT,
            **kwargs
        )
        self.details.update({
            'resource': resource,
            'limit': str(limit),
            'current': str(current)
        })


# Authentication and Authorization Errors
class AuthenticationError(BaseApplicationError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            user_message="Falha na autenticação. Verifique suas credenciais.",
            status_code=401,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class AuthorizationError(BaseApplicationError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(
            message=message,
            user_message="Acesso negado. Você não tem permissão para esta operação.",
            status_code=403,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


# External Service Errors
class ExternalServiceError(BaseApplicationError):
    """Base class for external service errors."""
    
    def __init__(self, message: str, service_name: str, **kwargs):
        super().__init__(
            message=message,
            user_message="Serviço externo temporariamente indisponível.",
            status_code=503,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.details['service_name'] = service_name


class ExternalServiceTimeoutError(ExternalServiceError):
    """Raised when external service times out."""
    
    def __init__(self, service_name: str, timeout_seconds: int, **kwargs):
        super().__init__(
            message=f"External service timeout: {service_name} ({timeout_seconds}s)",
            service_name=service_name,
            **kwargs
        )
        self.details['timeout_seconds'] = timeout_seconds


class ExternalServiceUnavailableError(ExternalServiceError):
    """Raised when external service is unavailable."""
    
    def __init__(self, service_name: str, **kwargs):
        super().__init__(
            message=f"External service unavailable: {service_name}",
            service_name=service_name,
            **kwargs
        )