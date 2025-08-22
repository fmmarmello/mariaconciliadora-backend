#!/usr/bin/env python3
"""
Test script for the Maria Conciliadora logging framework.
This script tests all logging functionality including file logging, console logging, and audit logging.
"""

import os
import sys
import tempfile
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up environment variables for testing
os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['CONSOLE_LOGGING'] = 'true'
os.environ['FILE_LOGGING'] = 'true'
os.environ['AUDIT_LOGGING'] = 'true'

from src.utils.logging_config import (
    setup_logging, 
    get_logger, 
    get_audit_logger,
    log_info,
    log_warning,
    log_error,
    log_debug,
    log_critical
)

def test_basic_logging():
    """Test basic logging functionality."""
    print("=== Testing Basic Logging ===")
    
    # Initialize logging
    setup_logging()
    logger = get_logger('test_basic')
    
    # Test all log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    print("[OK] Basic logging test completed")

def test_convenience_functions():
    """Test convenience logging functions."""
    print("\n=== Testing Convenience Functions ===")
    
    log_debug("Debug message using convenience function")
    log_info("Info message using convenience function")
    log_warning("Warning message using convenience function")
    log_error("Error message using convenience function")
    log_critical("Critical message using convenience function")
    
    print("[OK] Convenience functions test completed")

def test_audit_logging():
    """Test audit logging functionality."""
    print("\n=== Testing Audit Logging ===")
    
    audit_logger = get_audit_logger()
    
    # Test file upload audit
    audit_logger.log_file_upload(
        filename="test_file.ofx",
        file_type="OFX",
        file_size=1024,
        file_hash="abc123def456"
    )
    
    # Test file processing result audit
    audit_logger.log_file_processing_result(
        filename="test_file.ofx",
        success=True,
        items_processed=50,
        duplicates_found=5,
        errors=[]
    )
    
    # Test duplicate detection audit
    audit_logger.log_duplicate_detection(
        detection_type="file_level",
        duplicates_count=1,
        details={"filename": "test_file.ofx"}
    )
    
    # Test reconciliation audit
    audit_logger.log_reconciliation_operation(
        operation="start",
        matches_found=25,
        status="completed",
        details={"bank_transactions": 100, "company_entries": 80}
    )
    
    # Test AI operation audit
    audit_logger.log_ai_operation(
        operation="categorization",
        input_count=50,
        success=True,
        model_used="gpt-4o-mini"
    )
    
    # Test database operation audit
    audit_logger.log_database_operation(
        operation="insert",
        table="transactions",
        records_affected=50,
        success=True
    )
    
    # Test financial transaction audit
    audit_logger.log_financial_transaction(
        transaction_type="credit",
        amount=1500.00,
        description="Test transaction",
        account_id="12345"
    )
    
    print("[OK] Audit logging test completed")

def test_error_logging():
    """Test error logging with exception info."""
    print("\n=== Testing Error Logging with Exceptions ===")
    
    logger = get_logger('test_error')
    
    try:
        # Intentionally cause an error
        result = 1 / 0
    except Exception as e:
        logger.error("Test error with exception info", exc_info=True)
        log_error("Test error using convenience function", exc_info=True)
    
    print("[OK] Error logging test completed")

def test_different_loggers():
    """Test multiple logger instances."""
    print("\n=== Testing Multiple Logger Instances ===")
    
    logger1 = get_logger('module1')
    logger2 = get_logger('module2')
    logger3 = get_logger('module3.submodule')
    
    logger1.info("Message from module1")
    logger2.info("Message from module2")
    logger3.info("Message from module3.submodule")
    
    print("[OK] Multiple loggers test completed")

def check_log_files():
    """Check if log files were created."""
    print("\n=== Checking Log Files ===")
    
    log_files = [
        'logs/maria_conciliadora.log',
        'logs/audit/financial_audit.log',
        'logs/audit/upload_audit.log',
        'logs/audit/reconciliation_audit.log',
        'logs/audit/ai_audit.log',
        'logs/audit/database_audit.log'
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            print(f"[OK] {log_file} exists ({size} bytes)")
        else:
            print(f"[MISSING] {log_file} not found")

def main():
    """Run all logging tests."""
    print("Maria Conciliadora Logging Framework Test")
    print("=" * 50)
    
    try:
        test_basic_logging()
        test_convenience_functions()
        test_audit_logging()
        test_error_logging()
        test_different_loggers()
        check_log_files()
        
        print("\n" + "=" * 50)
        print("[SUCCESS] All logging tests completed successfully!")
        print("\nCheck the 'logs/' directory for generated log files.")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())