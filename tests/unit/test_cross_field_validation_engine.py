"""
Unit tests for CrossFieldValidationEngine.

This module contains comprehensive tests for the CrossFieldValidationEngine
including dependency validation, business rule enforcement, temporal consistency,
referential integrity checks, and conditional validation.
"""

import pytest
from datetime import datetime, date
from decimal import Decimal

from src.utils.cross_field_validation_engine import (
    CrossFieldValidationEngine,
    CrossFieldRule,
    ValidationCondition,
    DependencyType
)
from src.utils.validation_result import ValidationResult, ValidationSeverity


class TestCrossFieldValidationEngine:
    """Test cases for CrossFieldValidationEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = CrossFieldValidationEngine()

    def test_initialization(self):
        """Test engine initialization with default rules."""
        assert len(self.engine.rules) > 0
        assert len(self.engine.rule_groups) > 0
        assert 'financial_transaction' in self.engine.rule_groups

    def test_transaction_type_amount_consistency(self):
        """Test transaction type and amount consistency validation."""
        # Test debit transaction with negative amount (should pass)
        data = {
            'transaction_type': 'debit',
            'amount': -100.00
        }
        result = self.engine.validate(data, rule_group='financial_transaction')
        assert result.is_valid

        # Test debit transaction with positive amount (should fail)
        data = {
            'transaction_type': 'debit',
            'amount': 100.00
        }
        result = self.engine.validate(data, rule_group='financial_transaction')
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any('Amount sign inconsistent' in error for error in result.errors)

    def test_date_balance_temporal_consistency(self):
        """Test date and balance temporal consistency."""
        data = {
            'date': '2023-01-15',
            'balance': 5000.00
        }
        result = self.engine.validate(data, rule_group='financial_transaction')
        # This should pass as it's within reasonable temporal bounds
        assert result.is_valid or len(result.warnings) > 0

    def test_category_department_consistency(self):
        """Test category and department consistency."""
        # Valid combination
        data = {
            'category': 'software',
            'department': 'it'
        }
        result = self.engine.validate(data, rule_group='financial_transaction')
        assert result.is_valid

        # Invalid combination
        data = {
            'category': 'software',
            'department': 'sales'
        }
        result = self.engine.validate(data, rule_group='financial_transaction')
        assert len(result.warnings) > 0

    def test_amount_tax_range_consistency(self):
        """Test amount and tax range consistency."""
        data = {
            'amount': 1000.00,
            'tax_amount': 150.00  # 15% tax rate - reasonable
        }
        result = self.engine.validate(data, rule_group='financial_transaction')
        assert result.is_valid

        # Unreasonable tax amount
        data = {
            'amount': 100.00,
            'tax_amount': 80.00  # 80% tax rate - unreasonable
        }
        result = self.engine.validate(data, rule_group='financial_transaction')
        assert len(result.warnings) > 0

    def test_invoice_required_fields(self):
        """Test invoice required fields validation."""
        # Invoice without required fields
        data = {
            'transaction_type': 'invoice'
            # Missing invoice_number and supplier_name
        }
        result = self.engine.validate(data, rule_group='invoice_validation')
        assert not result.is_valid
        assert len(result.errors) > 0

        # Invoice with required fields
        data = {
            'transaction_type': 'invoice',
            'invoice_number': 'INV-001',
            'supplier_name': 'Test Supplier'
        }
        result = self.engine.validate(data, rule_group='invoice_validation')
        assert result.is_valid

    def test_cash_excluded_fields(self):
        """Test cash transaction excluded fields validation."""
        # Cash transaction with excluded fields
        data = {
            'transaction_type': 'cash',
            'check_number': '12345'
        }
        result = self.engine.validate(data, rule_group='invoice_validation')
        assert not result.is_valid
        assert len(result.errors) > 0

        # Cash transaction without excluded fields
        data = {
            'transaction_type': 'cash'
        }
        result = self.engine.validate(data, rule_group='invoice_validation')
        assert result.is_valid

    def test_bank_account_referential_integrity(self):
        """Test bank account referential integrity."""
        # This would need reference data in context
        data = {
            'bank_name': 'itau',
            'account_number': '12345-6'
        }
        context = {
            'reference_data': {
                'itau': {
                    'account_number': ['12345-6', '67890-1']
                }
            }
        }
        result = self.engine.validate(data, rule_group='banking_validation', context=context)
        assert result.is_valid

    def test_rule_management(self):
        """Test rule management functionality."""
        # Add a custom rule
        custom_rule = CrossFieldRule(
            rule_id="custom_test_rule",
            name="Custom Test Rule",
            dependency_type=DependencyType.VALUE_DEPENDENT,
            primary_field="test_field",
            dependent_fields=["dependent_field"],
            conditions=[
                ValidationCondition("test_field", "equals", "test_value")
            ],
            error_message="Custom validation failed"
        )

        self.engine.add_rule(custom_rule)
        assert "custom_test_rule" in self.engine.rules

        # Test the custom rule
        data = {
            'test_field': 'test_value',
            'dependent_field': 'should_be_present'
        }
        result = self.engine.validate(data)
        assert result.is_valid

        # Remove the custom rule
        self.engine.remove_rule("custom_test_rule")
        assert "custom_test_rule" not in self.engine.rules

    def test_rule_groups(self):
        """Test rule group functionality."""
        # Create a custom rule group
        self.engine.create_rule_group('test_group', ['transaction_type_amount_consistency'])

        assert 'test_group' in self.engine.rule_groups
        assert 'transaction_type_amount_consistency' in self.engine.rule_groups['test_group']

        # Test validation with the custom group
        data = {
            'transaction_type': 'debit',
            'amount': 100.00  # Should fail
        }
        result = self.engine.validate(data, rule_group='test_group')
        assert not result.is_valid

    def test_rule_enable_disable(self):
        """Test rule enable/disable functionality."""
        rule_id = 'transaction_type_amount_consistency'

        # Disable rule
        self.engine.disable_rule(rule_id)
        assert not self.engine.rules[rule_id].enabled

        # Test with disabled rule
        data = {
            'transaction_type': 'debit',
            'amount': 100.00
        }
        result = self.engine.validate(data, rule_group='financial_transaction')
        # Should not trigger the disabled rule
        disabled_rule_triggered = any('Amount sign inconsistent' in error for error in result.errors)
        assert not disabled_rule_triggered

        # Re-enable rule
        self.engine.enable_rule(rule_id)
        assert self.engine.rules[rule_id].enabled

    def test_complex_conditions(self):
        """Test complex validation conditions."""
        # Test multiple conditions
        data = {
            'field1': 'value1',
            'field2': 'value2',
            'amount': 100
        }

        # This would require adding a rule with multiple conditions
        # For now, test existing rules with complex scenarios
        result = self.engine.validate(data, rule_group='financial_transaction')
        assert isinstance(result, ValidationResult)

    def test_error_handling(self):
        """Test error handling in validation engine."""
        # Test with invalid data types
        data = {
            'transaction_type': None,
            'amount': 'invalid_amount'
        }
        result = self.engine.validate(data, rule_group='financial_transaction')
        # Should handle gracefully without crashing
        assert isinstance(result, ValidationResult)

    def test_performance(self):
        """Test validation engine performance."""
        import time

        # Test with multiple transactions
        transactions = [
            {'transaction_type': 'debit', 'amount': -100 * i}
            for i in range(100)
        ]

        start_time = time.time()
        for transaction in transactions:
            result = self.engine.validate(transaction, rule_group='financial_transaction')
            assert isinstance(result, ValidationResult)
        end_time = time.time()

        duration = end_time - start_time
        # Should complete within reasonable time (adjust threshold as needed)
        assert duration < 5.0  # 5 seconds for 100 validations

    def test_validation_result_structure(self):
        """Test that validation results have correct structure."""
        data = {
            'transaction_type': 'debit',
            'amount': 100.00
        }
        result = self.engine.validate(data, rule_group='financial_transaction')

        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'validation_duration_ms')

        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.is_valid, bool)


class TestValidationCondition:
    """Test cases for ValidationCondition."""

    def test_equals_condition(self):
        """Test equals condition."""
        condition = ValidationCondition("field", "equals", "value")

        assert condition.evaluate({"field": "value"})
        assert not condition.evaluate({"field": "other"})
        assert not condition.evaluate({"field": None})

    def test_not_equals_condition(self):
        """Test not_equals condition."""
        condition = ValidationCondition("field", "not_equals", "value")

        assert not condition.evaluate({"field": "value"})
        assert condition.evaluate({"field": "other"})
        assert condition.evaluate({"field": None})

    def test_contains_condition(self):
        """Test contains condition."""
        condition = ValidationCondition("field", "contains", "test")

        assert condition.evaluate({"field": "this is a test"})
        assert not condition.evaluate({"field": "no match"})
        assert not condition.evaluate({"field": None})

    def test_greater_than_condition(self):
        """Test greater_than condition."""
        condition = ValidationCondition("field", "greater_than", 10)

        assert condition.evaluate({"field": 15})
        assert not condition.evaluate({"field": 5})
        assert not condition.evaluate({"field": "invalid"})

    def test_less_than_condition(self):
        """Test less_than condition."""
        condition = ValidationCondition("field", "less_than", 10)

        assert condition.evaluate({"field": 5})
        assert not condition.evaluate({"field": 15})
        assert not condition.evaluate({"field": "invalid"})

    def test_case_insensitive(self):
        """Test case insensitive conditions."""
        condition = ValidationCondition("field", "equals", "VALUE", case_sensitive=False)

        assert condition.evaluate({"field": "value"})
        assert condition.evaluate({"field": "VALUE"})
        assert condition.evaluate({"field": "Value"})

    def test_null_conditions(self):
        """Test null-related conditions."""
        null_condition = ValidationCondition("field", "is_null")
        not_null_condition = ValidationCondition("field", "not_null")

        # Test is_null condition
        assert null_condition.evaluate({"field": None})  # None should be null
        assert not not_null_condition.evaluate({"field": None})  # None should not be not-null

        # Test not_null condition
        assert not_null_condition.evaluate({"field": "value"})  # Non-empty string should be not-null
        assert not_null_condition.evaluate({"field": ""})  # Empty string should be not-null
        assert not null_condition.evaluate({"field": "value"})  # Non-empty string should not be null
        assert not null_condition.evaluate({"field": ""})  # Empty string should not be null


class TestCrossFieldRule:
    """Test cases for CrossFieldRule."""

    def test_rule_creation(self):
        """Test cross-field rule creation."""
        rule = CrossFieldRule(
            rule_id="test_rule",
            name="Test Rule",
            dependency_type=DependencyType.VALUE_DEPENDENT,
            primary_field="primary",
            dependent_fields=["dependent"],
            conditions=[ValidationCondition("primary", "equals", "value")],
            error_message="Test error message"
        )

        assert rule.rule_id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.primary_field == "primary"
        assert rule.dependent_fields == ["dependent"]
        assert len(rule.conditions) == 1

    def test_rule_validation_success(self):
        """Test successful rule validation."""
        rule = CrossFieldRule(
            rule_id="test_rule",
            name="Test Rule",
            dependency_type=DependencyType.REQUIRED_IF,
            primary_field="type",
            dependent_fields=["required_field"],
            conditions=[ValidationCondition("type", "equals", "special")],
            error_message="Required field missing"
        )

        # Condition not met - should pass
        data = {"type": "normal"}
        result = rule.validate(data)
        assert result.is_valid

        # Condition met but field present - should pass
        data = {"type": "special", "required_field": "present"}
        result = rule.validate(data)
        assert result.is_valid

    def test_rule_validation_failure(self):
        """Test failed rule validation."""
        rule = CrossFieldRule(
            rule_id="test_rule",
            name="Test Rule",
            dependency_type=DependencyType.REQUIRED_IF,
            primary_field="type",
            dependent_fields=["required_field"],
            conditions=[ValidationCondition("type", "equals", "special")],
            error_message="Required field missing for {primary} type"
        )

        # Condition met but field missing - should fail
        data = {"type": "special"}
        result = rule.validate(data)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "Required field missing for special type" in result.errors[0]

    def test_disabled_rule(self):
        """Test disabled rule behavior."""
        rule = CrossFieldRule(
            rule_id="test_rule",
            name="Test Rule",
            dependency_type=DependencyType.REQUIRED_IF,
            primary_field="type",
            dependent_fields=["required_field"],
            conditions=[ValidationCondition("type", "equals", "special")],
            error_message="Required field missing",
            enabled=False
        )

        data = {"type": "special"}
        result = rule.validate(data)
        assert result.is_valid  # Should pass because rule is disabled


if __name__ == '__main__':
    pytest.main([__file__])