# Enhanced Data Validation Pipeline Documentation

## Overview

The Maria Conciliadora system now features a comprehensive, multi-layer validation pipeline that provides advanced data validation capabilities. This document describes the new validation system architecture, components, and usage.

## Architecture

The enhanced validation pipeline consists of several key components:

### 1. AdvancedValidationEngine
The central orchestration engine that coordinates multiple validation layers.

### 2. ValidationResult
Structured validation results with detailed error information, field-level results, and validation metadata.

### 3. BusinessRuleEngine
Configurable business rules for financial data validation with support for amount ranges, temporal consistency, and cross-field dependencies.

### 4. SchemaValidator
JSON schema-based validation for data structure validation with custom format validators.

### 5. Advanced Validation Middleware
Flask middleware that integrates validation into API endpoints with comprehensive error handling.

## Key Features

### Multi-Layer Validation
- **Schema Validation**: JSON schema validation for data structure
- **Business Rules**: Configurable business logic validation
- **Security Validation**: XSS, SQL injection, and other security checks
- **Cross-Field Validation**: Validation of relationships between fields
- **Temporal Consistency**: Date and time validation
- **Data Quality**: Pattern recognition and data quality checks

### Structured Results
- Detailed error messages with severity levels
- Field-level validation results
- Validation metadata and timestamps
- Recommendations for data quality improvements

### Extensibility
- Pluggable validation layers
- Custom business rules
- Configurable validation profiles
- Custom format validators

## Components

### AdvancedValidationEngine

```python
from src.utils.advanced_validation_engine import advanced_validation_engine

# Validate transaction data
result = advanced_validation_engine.validate(
    transaction_data,
    profile='transaction',
    context={'source': 'api'}
)

if not result.is_valid:
    print(f"Validation failed: {result.errors}")
```

#### Validation Profiles

- **transaction**: For bank transaction data
- **company_financial**: For company financial entries
- **bank_statement**: For complete bank statement validation
- **file_upload**: For file upload validation
- **api_request**: For general API request validation

#### Bulk Validation

```python
# Validate multiple items
result = advanced_validation_engine.validate_bulk(
    data_list,
    profile='transaction',
    parallel=True
)
```

### ValidationResult

```python
from src.utils.validation_result import ValidationResult, ValidationStatus

# Create validation result
result = ValidationResult()

# Add errors and warnings
result.add_error("Invalid amount", "amount", ValidationSeverity.HIGH)
result.add_warning("Suspicious pattern detected", "description")

# Check status
if result.status == ValidationStatus.FAIL:
    print("Validation failed")

# Get summary
summary = result.get_summary()
print(f"Errors: {summary['total_errors']}, Warnings: {summary['total_warnings']}")
```

### BusinessRuleEngine

```python
from src.utils.business_rule_engine import business_rule_engine, AmountRangeRule

# Add custom business rule
rule = AmountRangeRule(
    rule_id="large_transaction",
    name="Large Transaction Alert",
    category_field="transaction_type",
    amount_field="amount",
    min_amount=Decimal("10000.00"),
    category_values=["transfer"]
)

business_rule_engine.add_rule(rule)

# Validate with business rules
result = business_rule_engine.validate(data, "financial_transaction")
```

#### Built-in Rules

- **Amount Range Rules**: Validate amounts within specified ranges by category
- **Temporal Consistency Rules**: Ensure dates are within reasonable ranges
- **Cross-Field Dependency Rules**: Validate relationships between fields
- **Pattern Validation Rules**: Validate field values against patterns

### SchemaValidator

```python
from src.utils.schema_validator import schema_validator

# Validate against schema
result = schema_validator.validate(
    data,
    schema_name='transaction'
)

# Add custom schema
custom_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 2},
        "value": {"type": "number"}
    },
    "required": ["name"]
}

schema_validator.add_schema('custom', custom_schema)
```

#### Custom Format Validators

- **brazilian-date**: Validates DD/MM/YYYY format
- **currency-amount**: Validates currency amounts with Brazilian formatting
- **percentage**: Validates percentage values

### Advanced Validation Middleware

```python
from src.utils.advanced_validation_middleware import (
    validate_financial_transaction,
    validate_company_financial_entry,
    validate_bulk_data
)

# Decorate API endpoints
@app.route('/api/transactions', methods=['POST'])
@validate_financial_transaction()
def create_transaction():
    data = g.validated_data
    # Process validated data
    return jsonify({"success": True})

@app.route('/api/bulk-import', methods=['POST'])
@validate_bulk_data(max_items=1000)
def bulk_import():
    items = g.validated_bulk_data
    validation_result = g.bulk_validation_result
    # Process validated bulk data
    return jsonify({"success": True})
```

## Integration with Existing Services

### OFX Processor Integration

The OFX processor now uses the Advanced Validation Engine:

```python
# Enhanced validation in OFX processor
validation_result = processor.validate_transactions(transactions)
# Returns: {
#     'valid_transactions': [...],
#     'invalid_transactions': [...],
#     'warnings': [...],
#     'validation_summary': {...},
#     'validation_duration_ms': 150.5
# }
```

### XLSX Processor Integration

The XLSX processor validates each row using the advanced validation pipeline:

```python
# Each row is validated using AdvancedValidationEngine
entry = processor._process_single_row(row, index)
```

## Validation Rules

### Financial Transaction Rules

1. **Amount Validation**
   - Range validation by transaction type
   - Decimal precision validation
   - Suspicious pattern detection

2. **Date Validation**
   - Reasonable date ranges
   - Future date limits
   - Business day validation

3. **Description Validation**
   - Length limits
   - Pattern validation
   - Suspicious content detection

4. **Cross-Field Validation**
   - Amount sign consistency with transaction type
   - Category and department relationships
   - Bank name validation

### Company Financial Rules

1. **Expense vs Income Validation**
   - Amount sign validation
   - Category consistency
   - Department relationships

2. **Business Logic Validation**
   - Large amount warnings
   - Round number detection
   - Temporal consistency

## Error Handling

### Validation Error Types

- **CRITICAL**: System-breaking errors
- **HIGH**: Important validation failures
- **MEDIUM**: Standard validation issues
- **LOW**: Minor issues or warnings
- **INFO**: Informational messages

### Error Response Format

```json
{
  "success": false,
  "error": "Data validation failed",
  "validation_errors": [
    "Amount exceeds maximum allowed: 1000000.00"
  ],
  "validation_warnings": [
    "Transaction date is in the future"
  ],
  "validation_summary": {
    "is_valid": false,
    "status": "FAIL",
    "total_errors": 1,
    "total_warnings": 1,
    "fields_validated": 5,
    "validation_duration_ms": 45.2
  }
}
```

## Performance Considerations

### Parallel Validation
The system supports parallel validation for bulk operations:

```python
# Enable parallel processing for better performance
result = engine.validate_bulk(
    large_dataset,
    profile='transaction',
    parallel=True,
    max_workers=4
)
```

### Validation Caching
Consider implementing caching for frequently validated data patterns.

### Monitoring
Validation metrics are automatically collected:

- Validation duration
- Error rates by validation type
- Most common validation failures
- Performance bottlenecks

## Configuration

### Validation Profiles

Profiles can be customized by modifying the validation layers:

```python
# Create custom profile
engine.create_profile(
    'custom_profile',
    ['schema_validation', 'business_rules', 'security_validation']
)
```

### Rule Configuration

Business rules can be enabled/disabled dynamically:

```python
# Disable a specific rule
business_rule_engine.disable_rule('large_transaction_alert')

# Enable rule
business_rule_engine.enable_rule('temporal_consistency_check')
```

## Testing

Comprehensive tests are provided in `test_advanced_validation_engine.py`:

```bash
# Run validation tests
pytest tests/unit/test_advanced_validation_engine.py -v

# Run specific test
pytest tests/unit/test_advanced_validation_engine.py::TestAdvancedValidationEngine::test_transaction_validation_profile -v
```

## Migration Guide

### From Old Validation System

1. **Update Imports**
   ```python
   # Old
   from src.utils.validators import validate_transaction

   # New
   from src.utils.advanced_validation_engine import advanced_validation_engine
   ```

2. **Update Validation Calls**
   ```python
   # Old
   result = validate_transaction(data)

   # New
   result = advanced_validation_engine.validate(data, profile='transaction')
   ```

3. **Handle New Result Format**
   ```python
   # Old
   if not result.is_valid:
       errors = result.errors

   # New
   if not result.is_valid:
       errors = result.errors
       warnings = result.warnings
       summary = result.get_summary()
   ```

## Best Practices

1. **Use Appropriate Profiles**: Choose the right validation profile for your data type
2. **Handle Warnings**: Don't ignore warnings - they indicate potential data quality issues
3. **Monitor Performance**: Use validation metrics to identify bottlenecks
4. **Customize Rules**: Add business-specific rules as needed
5. **Test Thoroughly**: Validate your custom rules and profiles with comprehensive test data

## Troubleshooting

### Common Issues

1. **Validation Too Strict**
   - Adjust rule severity levels
   - Disable overly restrictive rules
   - Add exceptions for valid edge cases

2. **Performance Issues**
   - Use parallel validation for bulk operations
   - Disable unnecessary validation layers
   - Implement caching for repeated validations

3. **False Positives**
   - Review and adjust business rules
   - Add whitelist patterns for valid data
   - Update rule parameters based on business requirements

### Debug Mode

Enable debug logging to see detailed validation steps:

```python
import logging
logging.getLogger('src.utils.advanced_validation_engine').setLevel(logging.DEBUG)
```

## Future Enhancements

- Machine learning-based anomaly detection
- Custom validation rule marketplace
- Real-time validation metrics dashboard
- Integration with external validation services
- Advanced pattern recognition for fraud detection