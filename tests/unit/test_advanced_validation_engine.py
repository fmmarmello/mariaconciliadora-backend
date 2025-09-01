"""
Tests for the Advanced Validation Engine.

This module provides comprehensive tests for:
- AdvancedValidationEngine functionality
- ValidationResult class
- BusinessRuleEngine
- SchemaValidator
- Integration with processors
"""

import pytest
import json
from datetime import datetime, date
from decimal import Decimal

from src.utils.advanced_validation_engine import AdvancedValidationEngine
from src.utils.validation_result import ValidationResult, ValidationStatus, ValidationSeverity, FieldValidationResult
from src.utils.business_rule_engine import BusinessRuleEngine, AmountRangeRule, TemporalConsistencyRule
from src.utils.schema_validator import SchemaValidator


class TestValidationResult:
    """Test cases for ValidationResult class."""

    def test_validation_result_creation(self):
        """Test basic ValidationResult creation."""
        result = ValidationResult()
        assert result.is_valid is True
        assert result.status == ValidationStatus.PASS
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_validation_result_with_errors(self):
        """Test ValidationResult with errors."""
        result = ValidationResult()
        result.add_error("Test error", "field1", ValidationSeverity.HIGH)

        assert result.is_valid is False
        assert result.status == ValidationStatus.FAIL
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error"

    def test_validation_result_merge(self):
        """Test merging ValidationResult instances."""
        result1 = ValidationResult()
        result1.add_error("Error 1")

        result2 = ValidationResult()
        result2.add_warning("Warning 1")

        result1.merge(result2)

        assert result1.is_valid is False
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 1

    def test_field_validation_result(self):
        """Test FieldValidationResult functionality."""
        field_result = FieldValidationResult("test_field")
        field_result.add_error("Field error", ValidationSeverity.MEDIUM)
        field_result.add_warning("Field warning")

        assert field_result.is_valid is False
        assert len(field_result.errors) == 1
        assert len(field_result.warnings) == 1
        assert field_result.severity == ValidationSeverity.MEDIUM

    def test_validation_result_serialization(self):
        """Test ValidationResult JSON serialization."""
        result = ValidationResult()
        result.add_error("Test error", "field1")
        result.add_warning("Test warning", "field2")
        result.set_validator_info("TestValidator", "1.0")

        # Test to_dict
        data = result.to_dict()
        assert data['is_valid'] is False
        assert data['status'] == 'FAIL'
        assert len(data['errors']) == 1
        assert len(data['warnings']) == 1

        # Test from_dict
        result2 = ValidationResult.from_dict(data)
        assert result2.is_valid == result.is_valid
        assert result2.status == result.status
        assert len(result2.errors) == len(result.errors)


class TestBusinessRuleEngine:
    """Test cases for BusinessRuleEngine."""

    def test_business_rule_engine_creation(self):
        """Test BusinessRuleEngine initialization."""
        engine = BusinessRuleEngine()
        assert len(engine.rules) > 0  # Should have default rules
        assert len(engine.rule_groups) > 0

    def test_amount_range_rule(self):
        """Test AmountRangeRule validation."""
        rule = AmountRangeRule(
            rule_id="test_amount",
            name="Test Amount Rule",
            category_field="type",
            amount_field="amount",
            min_amount=Decimal("10.00"),
            max_amount=Decimal("100.00"),
            category_values=["expense"]
        )

        # Valid data
        valid_data = {"type": "expense", "amount": Decimal("50.00")}
        result = rule.validate(valid_data)
        assert result.is_valid is True

        # Invalid data - below minimum
        invalid_data = {"type": "expense", "amount": Decimal("5.00")}
        result = rule.validate(invalid_data)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_temporal_consistency_rule(self):
        """Test TemporalConsistencyRule validation."""
        rule = TemporalConsistencyRule(
            rule_id="test_date",
            name="Test Date Rule",
            date_field="transaction_date",
            max_future_days=30,
            min_past_days=365
        )

        # Valid date
        today = date.today()
        valid_data = {"transaction_date": today.isoformat()}
        result = rule.validate(valid_data)
        assert result.is_valid is True

        # Invalid date - too old
        old_date = today.replace(year=today.year - 2)
        invalid_data = {"transaction_date": old_date.isoformat()}
        result = rule.validate(invalid_data)
        assert result.is_valid is False

    def test_business_rule_engine_validation(self):
        """Test full BusinessRuleEngine validation."""
        engine = BusinessRuleEngine()

        # Test transaction validation
        transaction_data = {
            "amount": Decimal("100.00"),
            "date": date.today().isoformat(),
            "description": "Test transaction",
            "transaction_type": "expense"
        }

        result = engine.validate(transaction_data, "financial_transaction")
        assert isinstance(result, ValidationResult)


class TestSchemaValidator:
    """Test cases for SchemaValidator."""

    def test_schema_validator_creation(self):
        """Test SchemaValidator initialization."""
        validator = SchemaValidator()
        assert len(validator.schemas) > 0  # Should have default schemas

    def test_transaction_schema_validation(self):
        """Test transaction schema validation."""
        validator = SchemaValidator()

        # Valid transaction data
        valid_transaction = {
            "date": date.today().isoformat(),
            "amount": 100.50,
            "description": "Test transaction",
            "transaction_type": "credit"
        }

        result = validator.validate(valid_transaction, "transaction")
        assert result.is_valid is True

        # Invalid transaction data - missing required field
        invalid_transaction = {
            "amount": 100.50,
            "description": "Test transaction"
            # Missing date and transaction_type
        }

        result = validator.validate(invalid_transaction, "transaction")
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_custom_schema(self):
        """Test custom schema validation."""
        validator = SchemaValidator()

        custom_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 2},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }

        validator.add_schema("custom", custom_schema)

        # Valid data
        valid_data = {"name": "John", "age": 30}
        result = validator.validate(valid_data, "custom")
        assert result.is_valid is True

        # Invalid data
        invalid_data = {"name": "A", "age": -5}  # Name too short, negative age
        result = validator.validate(invalid_data, "custom")
        assert result.is_valid is False


class TestAdvancedValidationEngine:
    """Test cases for AdvancedValidationEngine."""

    def test_advanced_validation_engine_creation(self):
        """Test AdvancedValidationEngine initialization."""
        engine = AdvancedValidationEngine()
        assert len(engine.validation_layers) > 0
        assert len(engine.validation_profiles) > 0

    def test_transaction_validation_profile(self):
        """Test transaction validation profile."""
        engine = AdvancedValidationEngine()

        transaction_data = {
            "date": date.today().isoformat(),
            "amount": 100.50,
            "description": "Test transaction",
            "transaction_type": "credit"
        }

        result = engine.validate(transaction_data, profile="transaction")
        assert isinstance(result, ValidationResult)

    def test_company_financial_validation_profile(self):
        """Test company financial validation profile."""
        engine = AdvancedValidationEngine()

        financial_data = {
            "date": date.today().isoformat(),
            "amount": 500.00,
            "description": "Office supplies",
            "transaction_type": "expense",
            "category": "supplies"
        }

        result = engine.validate(financial_data, profile="company_financial")
        assert isinstance(result, ValidationResult)

    def test_bulk_validation(self):
        """Test bulk validation functionality."""
        engine = AdvancedValidationEngine()

        transactions = [
            {
                "date": date.today().isoformat(),
                "amount": 100.50,
                "description": "Transaction 1",
                "transaction_type": "credit"
            },
            {
                "date": date.today().isoformat(),
                "amount": 200.75,
                "description": "Transaction 2",
                "transaction_type": "debit"
            }
        ]

        result = engine.validate_bulk(transactions, profile="transaction")
        assert isinstance(result, ValidationResult)
        assert result.metadata.get('bulk_validation') is True
        assert result.metadata.get('total_items') == 2

    def test_validation_with_context(self):
        """Test validation with additional context."""
        engine = AdvancedValidationEngine()

        data = {"amount": 100.00, "description": "Test"}
        context = {
            "source": "test",
            "user_id": "123",
            "schema_name": "transaction"
        }

        result = engine.validate(data, profile="transaction", context=context)
        assert isinstance(result, ValidationResult)

    def test_layer_management(self):
        """Test validation layer management."""
        engine = AdvancedValidationEngine()

        # Test layer stats
        stats = engine.get_validation_stats()
        assert 'total_layers' in stats
        assert 'enabled_layers' in stats
        assert 'total_profiles' in stats

        # Test layer enabling/disabling
        initial_enabled = stats['enabled_layers']

        # Disable a layer
        if engine.validation_layers:
            layer_name = list(engine.validation_layers.keys())[0]
            engine.disable_layer(layer_name)

            new_stats = engine.get_validation_stats()
            assert new_stats['enabled_layers'] == initial_enabled - 1

            # Re-enable
            engine.enable_layer(layer_name)
            final_stats = engine.get_validation_stats()
            assert final_stats['enabled_layers'] == initial_enabled


class TestIntegrationValidation:
    """Integration tests for validation pipeline."""

    def test_ofx_processor_integration(self):
        """Test OFX processor integration with advanced validation."""
        from src.services.ofx_processor import OFXProcessor

        processor = OFXProcessor()

        # Mock transaction data
        transactions = [
            {
                "transaction_id": "test-1",
                "date": date.today().isoformat(),
                "amount": 100.50,
                "description": "Test transaction",
                "transaction_type": "credit"
            }
        ]

        # This should work with the updated validate_transactions method
        result = processor.validate_transactions(transactions)

        # Check that it returns the expected structure
        assert isinstance(result, dict)
        assert 'valid_transactions' in result
        assert 'invalid_transactions' in result
        assert 'warnings' in result
        assert 'validation_summary' in result

    def test_xlsx_processor_integration(self):
        """Test XLSX processor integration with advanced validation."""
        from src.services.xlsx_processor import XLSXProcessor
        import pandas as pd

        processor = XLSXProcessor()

        # Create mock DataFrame
        df = pd.DataFrame({
            'data': [date.today().isoformat()],
            'description': ['Test transaction'],
            'valor': [100.50],
            'tipo': ['receita']
        })

        # Test row processing
        entry = processor._process_single_row(df.iloc[0], 0)

        # Should return either valid entry or None
        assert entry is None or isinstance(entry, dict)

    def test_validation_pipeline_completeness(self):
        """Test that the validation pipeline covers all required components."""
        engine = AdvancedValidationEngine()

        # Check that all expected layers are present
        expected_layers = [
            'schema_validation',
            'business_rules',
            'security_validation',
            'cross_field_validation',
            'temporal_consistency',
            'data_quality'
        ]

        for layer_name in expected_layers:
            assert layer_name in engine.validation_layers

        # Check that expected profiles are present
        expected_profiles = [
            'transaction',
            'company_financial',
            'bank_statement',
            'file_upload',
            'api_request'
        ]

        for profile_name in expected_profiles:
            assert profile_name in engine.validation_profiles


class TestValidationPerformance:
    """Performance tests for validation pipeline."""

    def test_validation_performance(self):
        """Test validation performance with larger datasets."""
        engine = AdvancedValidationEngine()

        # Create larger dataset
        transactions = []
        for i in range(100):
            transactions.append({
                "date": date.today().isoformat(),
                "amount": float(i * 10.5),
                "description": f"Transaction {i}",
                "transaction_type": "credit" if i % 2 == 0 else "debit"
            })

        import time
        start_time = time.time()

        result = engine.validate_bulk(transactions, profile="transaction", parallel=True)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time (less than 5 seconds for 100 items)
        assert duration < 5.0
        assert result.metadata.get('total_items') == 100

    def test_validation_error_handling(self):
        """Test validation error handling."""
        engine = AdvancedValidationEngine()

        # Test with invalid data types
        invalid_data = {
            "date": "invalid-date",
            "amount": "not-a-number",
            "description": None,
            "transaction_type": "invalid_type"
        }

        result = engine.validate(invalid_data, profile="transaction")

        # Should handle errors gracefully
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0


if __name__ == "__main__":
    pytest.main([__file__])