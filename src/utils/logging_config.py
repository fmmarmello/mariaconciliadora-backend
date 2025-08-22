"""
Centralized logging configuration for Maria Conciliadora application.

This module provides a comprehensive logging framework with:
- Different log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- File-based logging with rotation
- Console logging for development
- Audit logging for financial operations
- Environment-based configuration
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class LoggerConfig:
    """Centralized logging configuration class."""
    
    # Default configuration
    DEFAULT_LOG_LEVEL = 'INFO'
    DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    DEFAULT_BACKUP_COUNT = 5
    
    # Log directories
    LOG_DIR = 'logs'
    AUDIT_LOG_DIR = 'logs/audit'
    
    def __init__(self):
        """Initialize logging configuration."""
        self.log_level = os.getenv('LOG_LEVEL', self.DEFAULT_LOG_LEVEL).upper()
        self.log_format = os.getenv('LOG_FORMAT', self.DEFAULT_LOG_FORMAT)
        self.date_format = os.getenv('LOG_DATE_FORMAT', self.DEFAULT_DATE_FORMAT)
        self.console_logging = os.getenv('CONSOLE_LOGGING', 'true').lower() == 'true'
        self.file_logging = os.getenv('FILE_LOGGING', 'true').lower() == 'true'
        self.audit_logging = os.getenv('AUDIT_LOGGING', 'true').lower() == 'true'
        self.max_bytes = int(os.getenv('LOG_MAX_BYTES', self.DEFAULT_MAX_BYTES))
        self.backup_count = int(os.getenv('LOG_BACKUP_COUNT', self.DEFAULT_BACKUP_COUNT))
        
        # Create log directories
        self._create_log_directories()
        
        # Configure root logger
        self._configure_root_logger()
    
    def _create_log_directories(self):
        """Create log directories if they don't exist."""
        Path(self.LOG_DIR).mkdir(exist_ok=True)
        Path(self.AUDIT_LOG_DIR).mkdir(exist_ok=True)
    
    def _configure_root_logger(self):
        """Configure the root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(self.log_format, self.date_format)
        
        # Console handler
        if self.console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.log_level))
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.file_logging:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=os.path.join(self.LOG_DIR, 'maria_conciliadora.log'),
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, self.log_level))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance with the specified name."""
        return logging.getLogger(name)
    
    def get_audit_logger(self, name: str = 'audit') -> logging.Logger:
        """Get an audit logger instance."""
        audit_logger = logging.getLogger(f'audit.{name}')
        
        # Check if handlers are already configured
        if not audit_logger.handlers:
            # Create audit-specific formatter
            audit_formatter = logging.Formatter(
                '%(asctime)s - AUDIT - %(name)s - %(levelname)s - %(message)s',
                self.date_format
            )
            
            # Audit file handler with rotation
            audit_handler = logging.handlers.RotatingFileHandler(
                filename=os.path.join(self.AUDIT_LOG_DIR, f'{name}_audit.log'),
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            audit_handler.setLevel(logging.INFO)
            audit_handler.setFormatter(audit_formatter)
            audit_logger.addHandler(audit_handler)
            
            # Set level and prevent propagation to root logger
            audit_logger.setLevel(logging.INFO)
            audit_logger.propagate = False
        
        return audit_logger


class AuditLogger:
    """Specialized audit logger for financial operations."""
    
    def __init__(self, logger_config: LoggerConfig):
        self.config = logger_config
        self.financial_logger = logger_config.get_audit_logger('financial')
        self.upload_logger = logger_config.get_audit_logger('upload')
        self.reconciliation_logger = logger_config.get_audit_logger('reconciliation')
        self.ai_logger = logger_config.get_audit_logger('ai')
        self.database_logger = logger_config.get_audit_logger('database')
    
    def log_file_upload(self, filename: str, file_type: str, user_id: Optional[str] = None, 
                       file_size: Optional[int] = None, file_hash: Optional[str] = None):
        """Log file upload operations."""
        audit_data = {
            'operation': 'file_upload',
            'filename': filename,
            'file_type': file_type,
            'user_id': user_id,
            'file_size': file_size,
            'file_hash': file_hash,
            'timestamp': datetime.now().isoformat()
        }
        self.upload_logger.info(f"File upload started: {json.dumps(audit_data)}")
    
    def log_file_processing_result(self, filename: str, success: bool, 
                                 items_processed: int, duplicates_found: int,
                                 errors: Optional[list] = None):
        """Log file processing results."""
        audit_data = {
            'operation': 'file_processing_result',
            'filename': filename,
            'success': success,
            'items_processed': items_processed,
            'duplicates_found': duplicates_found,
            'errors': errors or [],
            'timestamp': datetime.now().isoformat()
        }
        level = 'info' if success else 'error'
        getattr(self.upload_logger, level)(f"File processing completed: {json.dumps(audit_data)}")
    
    def log_duplicate_detection(self, detection_type: str, duplicates_count: int, 
                              details: Optional[Dict] = None):
        """Log duplicate detection operations."""
        audit_data = {
            'operation': 'duplicate_detection',
            'detection_type': detection_type,
            'duplicates_count': duplicates_count,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        self.financial_logger.info(f"Duplicate detection: {json.dumps(audit_data)}")
    
    def log_reconciliation_operation(self, operation: str, matches_found: int, 
                                   status: str, details: Optional[Dict] = None):
        """Log reconciliation operations."""
        audit_data = {
            'operation': f'reconciliation_{operation}',
            'matches_found': matches_found,
            'status': status,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        self.reconciliation_logger.info(f"Reconciliation operation: {json.dumps(audit_data)}")
    
    def log_ai_operation(self, operation: str, input_count: int, success: bool,
                        model_used: Optional[str] = None, error: Optional[str] = None):
        """Log AI service operations."""
        audit_data = {
            'operation': f'ai_{operation}',
            'input_count': input_count,
            'success': success,
            'model_used': model_used,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        level = 'info' if success else 'error'
        getattr(self.ai_logger, level)(f"AI operation: {json.dumps(audit_data)}")
    
    def log_database_operation(self, operation: str, table: str, records_affected: int,
                             success: bool, error: Optional[str] = None):
        """Log database operations."""
        audit_data = {
            'operation': f'database_{operation}',
            'table': table,
            'records_affected': records_affected,
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        level = 'info' if success else 'error'
        getattr(self.database_logger, level)(f"Database operation: {json.dumps(audit_data)}")
    
    def log_financial_transaction(self, transaction_type: str, amount: float,
                                description: str, account_id: Optional[str] = None):
        """Log financial transaction processing."""
        audit_data = {
            'operation': 'financial_transaction',
            'transaction_type': transaction_type,
            'amount': amount,
            'description': description,
            'account_id': account_id,
            'timestamp': datetime.now().isoformat()
        }
        self.financial_logger.info(f"Financial transaction: {json.dumps(audit_data)}")


# Global instances
_logger_config = None
_audit_logger = None


def get_logger_config() -> LoggerConfig:
    """Get the global logger configuration instance."""
    global _logger_config
    if _logger_config is None:
        _logger_config = LoggerConfig()
    return _logger_config


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    config = get_logger_config()
    return config.get_logger(name)


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        config = get_logger_config()
        _audit_logger = AuditLogger(config)
    return _audit_logger


def setup_logging():
    """Initialize the logging system. Call this once at application startup."""
    get_logger_config()
    logger = get_logger(__name__)
    logger.info("Logging system initialized successfully")


# Convenience functions for common logging operations
def log_info(message: str, logger_name: str = 'maria_conciliadora'):
    """Log an info message."""
    logger = get_logger(logger_name)
    logger.info(message)


def log_warning(message: str, logger_name: str = 'maria_conciliadora'):
    """Log a warning message."""
    logger = get_logger(logger_name)
    logger.warning(message)


def log_error(message: str, logger_name: str = 'maria_conciliadora', exc_info: bool = False):
    """Log an error message."""
    logger = get_logger(logger_name)
    logger.error(message, exc_info=exc_info)


def log_debug(message: str, logger_name: str = 'maria_conciliadora'):
    """Log a debug message."""
    logger = get_logger(logger_name)
    logger.debug(message)


def log_critical(message: str, logger_name: str = 'maria_conciliadora'):
    """Log a critical message."""
    logger = get_logger(logger_name)
    logger.critical(message)