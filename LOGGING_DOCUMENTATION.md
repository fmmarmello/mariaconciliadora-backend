# Maria Conciliadora - Logging Framework Documentation

## Overview

The Maria Conciliadora application uses a comprehensive logging framework that provides:

- **Centralized Configuration**: Single point of configuration for all logging
- **Multiple Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **File-based Logging**: Automatic log rotation to prevent disk space issues
- **Console Logging**: Real-time logging output for development
- **Audit Logging**: Specialized logging for financial operations and compliance
- **Environment-based Configuration**: Configurable via environment variables

## Quick Start

### Basic Usage

```python
from src.utils.logging_config import get_logger

# Get a logger instance
logger = get_logger(__name__)

# Log messages at different levels
logger.debug("Debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical system error")
```

### Convenience Functions

```python
from src.utils.logging_config import log_info, log_warning, log_error

# Use convenience functions for quick logging
log_info("Application started")
log_warning("Configuration file not found, using defaults")
log_error("Database connection failed", exc_info=True)
```

### Audit Logging

```python
from src.utils.logging_config import get_audit_logger

audit_logger = get_audit_logger()

# Log file upload
audit_logger.log_file_upload(
    filename="transactions.ofx",
    file_type="OFX",
    file_size=1024,
    file_hash="abc123"
)

# Log financial transaction
audit_logger.log_financial_transaction(
    transaction_type="credit",
    amount=1500.00,
    description="Payment received",
    account_id="12345"
)
```

## Configuration

### Environment Variables

Configure logging behavior using these environment variables in your `.env` file:

```bash
# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Log format string
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s

# Date format
LOG_DATE_FORMAT=%Y-%m-%d %H:%M:%S

# Enable/disable console logging
CONSOLE_LOGGING=true

# Enable/disable file logging
FILE_LOGGING=true

# Enable/disable audit logging
AUDIT_LOGGING=true

# Maximum log file size in bytes (default: 10MB)
LOG_MAX_BYTES=10485760

# Number of backup files to keep (default: 5)
LOG_BACKUP_COUNT=5
```

### Default Configuration

If environment variables are not set, the following defaults are used:

- **LOG_LEVEL**: INFO
- **CONSOLE_LOGGING**: true
- **FILE_LOGGING**: true
- **AUDIT_LOGGING**: true
- **LOG_MAX_BYTES**: 10MB
- **LOG_BACKUP_COUNT**: 5

## Log Files Structure

The logging system creates the following directory structure:

```
logs/
├── maria_conciliadora.log          # Main application log
├── maria_conciliadora.log.1        # Rotated log files
├── maria_conciliadora.log.2
└── audit/                          # Audit logs directory
    ├── financial_audit.log         # Financial operations
    ├── upload_audit.log            # File uploads
    ├── reconciliation_audit.log    # Reconciliation operations
    ├── ai_audit.log               # AI service operations
    └── database_audit.log         # Database operations
```

## Log Levels

### DEBUG
- Detailed diagnostic information
- Only visible when LOG_LEVEL=DEBUG
- Use for troubleshooting and development

### INFO
- General operational messages
- Confirms things are working as expected
- Default minimum level for production

### WARNING
- Something unexpected happened, but the application is still working
- Indicates potential issues that should be investigated

### ERROR
- A serious problem occurred
- The application couldn't perform a specific function
- Should be investigated and fixed

### CRITICAL
- A very serious error occurred
- The application may not be able to continue running
- Requires immediate attention

## Audit Logging

The audit logging system provides specialized logging for compliance and monitoring:

### File Operations
```python
# Log file upload
audit_logger.log_file_upload(filename, file_type, user_id, file_size, file_hash)

# Log processing results
audit_logger.log_file_processing_result(filename, success, items_processed, duplicates_found, errors)
```

### Financial Operations
```python
# Log financial transactions
audit_logger.log_financial_transaction(transaction_type, amount, description, account_id)

# Log duplicate detection
audit_logger.log_duplicate_detection(detection_type, duplicates_count, details)
```

### System Operations
```python
# Log reconciliation operations
audit_logger.log_reconciliation_operation(operation, matches_found, status, details)

# Log AI operations
audit_logger.log_ai_operation(operation, input_count, success, model_used, error)

# Log database operations
audit_logger.log_database_operation(operation, table, records_affected, success, error)
```

## Integration Examples

### Flask Route Logging

```python
from flask import Blueprint, request, jsonify
from src.utils.logging_config import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("File upload request received")
        
        # Process file upload
        filename = process_upload()
        
        # Audit log the upload
        audit_logger.log_file_upload(filename, "OFX")
        
        logger.info(f"File upload completed: {filename}")
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
```

### Service Class Logging

```python
from src.utils.logging_config import get_logger, get_audit_logger

class TransactionService:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.audit_logger = get_audit_logger()
    
    def process_transactions(self, transactions):
        self.logger.info(f"Processing {len(transactions)} transactions")
        
        try:
            # Process transactions
            processed = self._process_batch(transactions)
            
            # Audit log the operation
            self.audit_logger.log_database_operation(
                'insert', 'transactions', len(processed), True
            )
            
            self.logger.info(f"Successfully processed {len(processed)} transactions")
            return processed
            
        except Exception as e:
            self.logger.error(f"Transaction processing failed: {str(e)}", exc_info=True)
            self.audit_logger.log_database_operation(
                'insert', 'transactions', 0, False, str(e)
            )
            raise
```

## Best Practices

### 1. Use Appropriate Log Levels
- **DEBUG**: Detailed diagnostic information for developers
- **INFO**: General operational messages
- **WARNING**: Unexpected situations that don't prevent operation
- **ERROR**: Errors that prevent specific operations
- **CRITICAL**: Errors that may cause the application to stop

### 2. Include Context Information
```python
# Good: Include relevant context
logger.info(f"Processing file: {filename} ({file_size} bytes)")

# Bad: Vague message
logger.info("Processing file")
```

### 3. Use Exception Information
```python
try:
    risky_operation()
except Exception as e:
    # Include exception traceback
    logger.error("Operation failed", exc_info=True)
```

### 4. Avoid Logging Sensitive Information
```python
# Good: Log without sensitive data
logger.info(f"User login successful: user_id={user_id}")

# Bad: Don't log passwords or tokens
logger.info(f"User login: {username}:{password}")
```

### 5. Use Structured Logging for Audit Events
```python
# Use audit logger for compliance-related events
audit_logger.log_financial_transaction(
    transaction_type="credit",
    amount=amount,
    description=description,
    account_id=account_id
)
```

## Monitoring and Maintenance

### Log Rotation
- Log files automatically rotate when they reach the configured size limit
- Old log files are kept according to the `LOG_BACKUP_COUNT` setting
- Monitor disk space usage in production environments

### Log Analysis
- Use log aggregation tools like ELK Stack or Splunk for production
- Set up alerts for ERROR and CRITICAL level messages
- Monitor audit logs for compliance and security

### Performance Considerations
- Logging has minimal performance impact
- File I/O is handled efficiently with buffering
- Consider reducing log level in high-performance scenarios

## Troubleshooting

### Common Issues

#### 1. Log Files Not Created
- Check file permissions in the application directory
- Ensure the `logs/` directory can be created
- Verify `FILE_LOGGING=true` in environment variables

#### 2. No Console Output
- Check `CONSOLE_LOGGING=true` in environment variables
- Verify `LOG_LEVEL` is set appropriately

#### 3. Missing Audit Logs
- Ensure `AUDIT_LOGGING=true` in environment variables
- Check that audit logging methods are being called

#### 4. Log Files Too Large
- Reduce `LOG_MAX_BYTES` setting
- Increase `LOG_BACKUP_COUNT` if needed
- Consider raising the log level to reduce volume

### Testing the Logging System

Run the included test script to verify logging functionality:

```bash
cd mariaconciliadora-backend
python test_logging.py
```

This will test all logging features and create sample log files.

## Migration from Print Statements

When migrating existing code from `print()` statements to proper logging:

### Before
```python
print("Processing started")
print(f"Error: {error_message}")
```

### After
```python
logger.info("Processing started")
logger.error(f"Error: {error_message}")
```

### For User-Facing Messages
Keep `print()` statements for user interaction, but add logging:

```python
# User interaction
print("Enter your choice: ")
choice = input()

# Also log the interaction
logger.info(f"User selected option: {choice}")
```

## Security Considerations

1. **Sensitive Data**: Never log passwords, API keys, or personal information
2. **Log Access**: Restrict access to log files in production
3. **Log Retention**: Implement appropriate log retention policies
4. **Audit Trail**: Ensure audit logs are tamper-evident
5. **Compliance**: Follow relevant compliance requirements (PCI DSS, GDPR, etc.)

## Support

For questions or issues with the logging framework:

1. Check this documentation
2. Run the test script to verify functionality
3. Review the source code in `src/utils/logging_config.py`
4. Check environment variable configuration

---

*This documentation covers the comprehensive logging framework implemented for the Maria Conciliadora application. The framework provides production-ready logging with audit capabilities for financial applications.*