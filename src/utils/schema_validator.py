"""
JSON Schema Validator for Maria Conciliadora system.

This module provides:
- JSON schema-based validation for transaction data structure
- Required field validation
- Data type constraints
- Format validation (dates, amounts, etc.)
- Custom validation rules
"""

import json
import re
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union
from decimal import Decimal, InvalidOperation
import jsonschema
from jsonschema import validate, ValidationError as JSONValidationError

from .validation_result import ValidationResult, ValidationSeverity, FieldValidationResult
from .logging_config import get_logger

logger = get_logger(__name__)


class SchemaValidator:
    """
    JSON Schema-based validator for structured data validation.
    """

    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.custom_validators: Dict[str, callable] = {}
        self._load_default_schemas()

    def _load_default_schemas(self):
        """Load default JSON schemas for common data types."""

        # Transaction data schema
        self.schemas['transaction'] = {
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string", "minLength": 1, "maxLength": 100},
                "date": {"type": "string", "format": "date"},
                "amount": {"type": "number", "minimum": -999999999, "maximum": 999999999},
                "description": {"type": "string", "minLength": 1, "maxLength": 500},
                "transaction_type": {"type": "string", "enum": ["credit", "debit", "income", "expense"]},
                "bank_name": {"type": "string", "minLength": 1, "maxLength": 50},
                "balance": {"type": ["number", "null"]},
                "category": {"type": ["string", "null"], "maxLength": 100},
                "reference": {"type": ["string", "null"], "maxLength": 100}
            },
            "required": ["date", "amount", "description", "transaction_type"]
        }

        # Company financial entry schema
        self.schemas['company_financial'] = {
            "type": "object",
            "properties": {
                "date": {"type": "string", "format": "date"},
                "description": {"type": "string", "minLength": 1, "maxLength": 500},
                "amount": {"type": "number", "minimum": -999999999, "maximum": 999999999},
                "category": {"type": ["string", "null"], "maxLength": 100},
                "cost_center": {"type": ["string", "null"], "maxLength": 100},
                "department": {"type": ["string", "null"], "maxLength": 100},
                "project": {"type": ["string", "null"], "maxLength": 100},
                "transaction_type": {"type": "string", "enum": ["income", "expense"]},
                "observations": {"type": ["string", "null"], "maxLength": 1000},
                "monthly_report_value": {"type": ["number", "null"]}
            },
            "required": ["date", "description", "amount", "transaction_type"]
        }

        # Bank statement schema
        self.schemas['bank_statement'] = {
            "type": "object",
            "properties": {
                "bank_name": {"type": "string", "minLength": 1, "maxLength": 50},
                "account_info": {
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string", "minLength": 1},
                        "routing_number": {"type": ["string", "null"]},
                        "account_type": {"type": ["string", "null"]},
                        "bank_id": {"type": ["string", "null"]}
                    }
                },
                "transactions": {
                    "type": "array",
                    "items": self.schemas['transaction']
                },
                "summary": {
                    "type": "object",
                    "properties": {
                        "total_transactions": {"type": "integer", "minimum": 0},
                        "total_credits": {"type": "number", "minimum": 0},
                        "total_debits": {"type": "number", "maximum": 0},
                        "balance": {"type": ["number", "null"]}
                    }
                }
            },
            "required": ["bank_name", "transactions", "summary"]
        }

        # File upload schema
        self.schemas['file_upload'] = {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "minLength": 1, "maxLength": 255},
                "file_type": {"type": "string", "enum": ["ofx", "xlsx", "csv"]},
                "file_size": {"type": "integer", "minimum": 1, "maximum": 16777216},  # 16MB
                "content_type": {"type": "string", "minLength": 1},
                "upload_timestamp": {"type": "string", "format": "date-time"}
            },
            "required": ["filename", "file_type", "file_size"]
        }

    def add_schema(self, schema_name: str, schema: Dict[str, Any]):
        """Add a custom JSON schema."""
        try:
            # Validate the schema itself
            jsonschema.Draft7Validator.check_schema(schema)
            self.schemas[schema_name] = schema
            logger.info(f"Added schema: {schema_name}")
        except Exception as e:
            logger.error(f"Invalid schema {schema_name}: {str(e)}")
            raise ValueError(f"Invalid JSON schema: {str(e)}")

    def remove_schema(self, schema_name: str):
        """Remove a schema."""
        if schema_name in self.schemas:
            del self.schemas[schema_name]
            logger.info(f"Removed schema: {schema_name}")

    def add_custom_validator(self, format_name: str, validator_function: callable):
        """Add a custom format validator."""
        self.custom_validators[format_name] = validator_function
        logger.info(f"Added custom validator: {format_name}")

    def validate(self, data: Any, schema_name: str,
                context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data against a JSON schema."""
        result = ValidationResult()
        result.set_validator_info("SchemaValidator", "1.0")

        if schema_name not in self.schemas:
            result.add_error(f"Schema '{schema_name}' not found")
            return result

        schema = self.schemas[schema_name]

        try:
            # Create validator with custom format validators
            validator = jsonschema.Draft7Validator(
                schema,
                format_checker=jsonschema.draft7_format_checker
            )

            # Add custom format validators
            for format_name, validator_func in self.custom_validators.items():
                validator.format_checker.checks(format_name)(validator_func)

            # Collect all validation errors
            errors = list(validator.iter_errors(data))

            for error in errors:
                # Extract field path
                field_path = '.'.join(str(x) for x in error.path) if error.path else 'root'

                # Map JSON schema error to our validation result
                severity = self._map_error_severity(error)
                result.add_error(str(error.message), field_path, severity)

                # Add additional context for certain error types
                if 'required' in str(error.message).lower():
                    result.add_recommendation(f"Field '{field_path}' is required but missing")
                elif 'format' in str(error.message).lower():
                    result.add_recommendation(f"Field '{field_path}' has invalid format")

        except Exception as e:
            logger.error(f"Schema validation error: {str(e)}")
            result.add_error(f"Schema validation failed: {str(e)}")

        # Perform additional custom validations
        self._perform_custom_validations(data, schema_name, result, context)

        return result

    def _map_error_severity(self, error: JSONValidationError) -> ValidationSeverity:
        """Map JSON schema validation error to our severity levels."""
        error_msg = str(error.message).lower()

        if any(keyword in error_msg for keyword in ['required', 'missing']):
            return ValidationSeverity.HIGH
        elif any(keyword in error_msg for keyword in ['format', 'invalid', 'type']):
            return ValidationSeverity.MEDIUM
        elif any(keyword in error_msg for keyword in ['maximum', 'minimum', 'length']):
            return ValidationSeverity.LOW
        else:
            return ValidationSeverity.MEDIUM

    def _perform_custom_validations(self, data: Any, schema_name: str,
                                  result: ValidationResult, context: Optional[Dict[str, Any]]):
        """Perform additional custom validations beyond JSON schema."""

        if schema_name == 'transaction':
            self._validate_transaction_data(data, result)
        elif schema_name == 'company_financial':
            self._validate_company_financial_data(data, result)
        elif schema_name == 'bank_statement':
            self._validate_bank_statement_data(data, result)

    def _validate_transaction_data(self, data: Dict[str, Any], result: ValidationResult):
        """Custom validations for transaction data."""

        # Validate amount precision (should not have more than 2 decimal places)
        if 'amount' in data:
            try:
                amount = Decimal(str(data['amount']))
                if amount.as_tuple().exponent < -2:
                    result.add_warning("Amount has more than 2 decimal places", 'amount')
            except (ValueError, InvalidOperation):
                pass

        # Validate date is not in future (with some tolerance)
        if 'date' in data:
            try:
                transaction_date = datetime.fromisoformat(data['date']).date()
                today = date.today()
                if transaction_date > today:
                    days_future = (transaction_date - today).days
                    if days_future > 30:
                        result.add_error("Transaction date is too far in the future", 'date')
                    else:
                        result.add_warning("Transaction date is in the future", 'date')
            except (ValueError, TypeError):
                pass

        # Validate transaction type consistency with amount sign
        if 'transaction_type' in data and 'amount' in data:
            trans_type = data['transaction_type'].lower()
            amount = data['amount']

            try:
                amount_val = float(amount)
                if trans_type in ['debit', 'expense'] and amount_val > 0:
                    result.add_warning("Debit/expense transaction has positive amount", 'amount')
                elif trans_type in ['credit', 'income'] and amount_val < 0:
                    result.add_warning("Credit/income transaction has negative amount", 'amount')
            except (ValueError, TypeError):
                pass

    def _validate_company_financial_data(self, data: Dict[str, Any], result: ValidationResult):
        """Custom validations for company financial data."""

        # Validate amount ranges for different transaction types
        if 'transaction_type' in data and 'amount' in data:
            trans_type = data['transaction_type'].lower()
            amount = data['amount']

            try:
                amount_val = Decimal(str(amount))

                if trans_type == 'expense' and amount_val > 0:
                    result.add_warning("Expense entry has positive amount - should typically be negative", 'amount')
                elif trans_type == 'income' and amount_val < 0:
                    result.add_warning("Income entry has negative amount - should typically be positive", 'amount')

                # Check for unusually large amounts
                if abs(amount_val) > 1000000:  # 1 million
                    result.add_warning("Unusually large amount detected", 'amount')

            except (ValueError, InvalidOperation):
                pass

        # Validate category consistency
        if 'category' in data and 'transaction_type' in data:
            category = str(data['category']).lower()
            trans_type = data['transaction_type'].lower()

            # Define expected categories for each transaction type
            expense_categories = ['office_supplies', 'travel', 'meals', 'utilities', 'rent',
                                'salaries', 'taxes', 'insurance', 'marketing', 'equipment']
            income_categories = ['sales', 'services', 'interest', 'dividends', 'other_income']

            if trans_type == 'expense' and category not in expense_categories:
                result.add_warning(f"Category '{category}' is not typical for expenses", 'category')
            elif trans_type == 'income' and category not in income_categories:
                result.add_warning(f"Category '{category}' is not typical for income", 'category')

    def _validate_bank_statement_data(self, data: Dict[str, Any], result: ValidationResult):
        """Custom validations for bank statement data."""

        # Validate transaction count consistency
        if 'transactions' in data and 'summary' in data:
            transactions = data['transactions']
            summary = data['summary']

            actual_count = len(transactions)
            reported_count = summary.get('total_transactions', 0)

            if actual_count != reported_count:
                result.add_error(f"Transaction count mismatch: {actual_count} vs {reported_count}",
                               'summary.total_transactions')

        # Validate balance calculations
        if 'transactions' in data and 'summary' in data:
            transactions = data['transactions']
            summary = data['summary']

            try:
                calculated_balance = sum(float(t.get('amount', 0)) for t in transactions)
                reported_balance = summary.get('balance')

                if reported_balance is not None:
                    reported_balance = float(reported_balance)
                    if abs(calculated_balance - reported_balance) > 0.01:  # Allow small rounding differences
                        result.add_warning("Calculated balance doesn't match reported balance",
                                         'summary.balance')
            except (ValueError, TypeError, AttributeError):
                pass

        # Validate bank name consistency
        if 'bank_name' in data and 'transactions' in data:
            bank_name = data['bank_name'].lower()
            transactions = data['transactions']

            # Check if transaction descriptions contain bank-specific patterns
            bank_patterns = {
                'itau': ['itau', '341'],
                'bradesco': ['bradesco', '237'],
                'santander': ['santander', '033'],
                'nubank': ['nubank', '260'],
                'sicoob': ['sicoob', '756'],
                'caixa': ['caixa', 'cef', '104'],
                'banco do brasil': ['banco do brasil', 'bb', '001']
            }

            if bank_name in bank_patterns:
                patterns = bank_patterns[bank_name]
                inconsistent_count = 0

                for transaction in transactions[:10]:  # Check first 10 transactions
                    description = str(transaction.get('description', '')).lower()
                    if not any(pattern in description for pattern in patterns):
                        inconsistent_count += 1

                if inconsistent_count > 5:  # More than half are inconsistent
                    result.add_warning("Transaction descriptions don't match specified bank",
                                     'bank_name')

    def validate_bulk(self, data_list: List[Any], schema_name: str,
                     context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate a list of data items against a schema."""
        result = ValidationResult()
        result.set_validator_info("SchemaValidator", "1.0")

        for i, data in enumerate(data_list):
            item_result = self.validate(data, schema_name, context)

            # Prefix field names with array index for bulk validation
            for field_name in item_result.field_results:
                prefixed_name = f"[{i}].{field_name}"
                item_result.field_results[prefixed_name] = item_result.field_results.pop(field_name)
                item_result.field_results[prefixed_name].field_name = prefixed_name

            result.merge(item_result)

        return result

    def get_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Get a schema by name."""
        return self.schemas.get(schema_name)

    def list_schemas(self) -> List[str]:
        """List all available schemas."""
        return list(self.schemas.keys())


# Custom format validators
def validate_brazilian_date(value: str) -> bool:
    """Validate Brazilian date format (DD/MM/YYYY)."""
    if not isinstance(value, str):
        return False

    pattern = r'^\d{2}/\d{2}/\d{4}$'
    if not re.match(pattern, value):
        return False

    try:
        datetime.strptime(value, '%d/%m/%Y')
        return True
    except ValueError:
        return False


def validate_currency_amount(value: str) -> bool:
    """Validate currency amount format."""
    if not isinstance(value, str):
        return True  # Let JSON schema handle type validation

    # Remove currency symbols and spaces
    cleaned = re.sub(r'[R$\s]', '', value)

    # Handle Brazilian decimal format
    if ',' in cleaned and '.' in cleaned:
        # Format like 1.234.567,89
        cleaned = cleaned.replace('.', '').replace(',', '.')
    elif ',' in cleaned:
        # Format like 1234,56
        cleaned = cleaned.replace(',', '.')

    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def validate_percentage(value: str) -> bool:
    """Validate percentage format."""
    if not isinstance(value, str):
        return True

    # Remove % symbol and spaces
    cleaned = re.sub(r'[%s]', '', value)

    try:
        percent = float(cleaned)
        return 0 <= percent <= 100
    except ValueError:
        return False


# Global schema validator instance
schema_validator = SchemaValidator()

# Add custom format validators
schema_validator.add_custom_validator('brazilian-date', validate_brazilian_date)
schema_validator.add_custom_validator('currency-amount', validate_currency_amount)
schema_validator.add_custom_validator('percentage', validate_percentage)