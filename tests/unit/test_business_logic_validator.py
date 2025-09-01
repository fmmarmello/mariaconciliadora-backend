"""
Unit tests for BusinessLogicValidator.

This module contains comprehensive tests for the BusinessLogicValidator
including financial transaction rules, amount range validation, date consistency,
account balance rules, and tax-related validation.
"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal

from src.utils.business_logic_validator import (
    BusinessLogicValidator,
    BusinessLogicRule,
    BusinessRuleType
)
from src.utils.validation_result import ValidationResult, ValidationSeverity


class TestBusinessLogicValidator:
    """Test cases for BusinessLogicValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = BusinessLogicValidator()

    def test_initialization(self):
        """Test validator initialization with default rules."""
        assert len(self.validator.rules) > 0
        assert len(self.validator.rule_groups) > 0
        assert 'transaction_validation' in self.validator.rule_groups

    def test_transaction_amount_sign_consistency(self):
        """Test transaction amount sign consistency."""
        # Valid debit transaction (negative amount)
        data = {
            'transaction_type': 'debit',
            'amount': -100.00
        }
        result = self.validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

        # Invalid debit transaction (positive amount)
        data = {
            'transaction_type': 'debit',
            'amount': 100.00
        }
        result = self.validator.validate(data, rule_group='transaction_validation')
        assert not result.is_valid
        assert len(result.errors) > 0

        # Valid credit transaction (positive amount)
        data = {
            'transaction_type': 'credit',
            'amount': 100.00
        }
        result = self.validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

    def test_transaction_amount_range_validation(self):
        """Test transaction amount range validation."""
        # Valid salary amount
        data = {
            'transaction_type': 'salary',
            'amount': 5000.00
        }
        result = self.validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

        # Invalid salary amount (too high)
        data = {
            'transaction_type': 'salary',
            'amount': 100000.00
        }
        result = self.validator.validate(data, rule_group='transaction_validation')
        assert len(result.warnings) > 0

        # Very small amount
        data = {
            'transaction_type': 'debit',
            'amount': 0.001
        }
        result = self.validator.validate(data, rule_group='transaction_validation')
        assert len(result.warnings) > 0

    def test_duplicate_transaction_detection(self):
        """Test duplicate transaction detection."""
        # This test would require historical data in context
        data = {
            'date': '2023-01-15',
            'amount': 100.00,
            'description': 'Test transaction'
        }

        # Without historical data, should pass
        result = self.validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

        # With duplicate in historical data
        context = {
            'recent_transactions': [{
                'date': '2023-01-15',
                'amount': 100.00,
                'description': 'Test transaction'
            }]
        }
        result = self.validator.validate(data, rule_group='transaction_validation', context=context)
        assert len(result.warnings) > 0

    def test_business_day_validation(self):
        """Test business day validation."""
        # Weekend date
        data = {
            'date': '2023-01-14',  # Saturday
            'transaction_type': 'debit',
            'amount': -100.00
        }
        result = self.validator.validate(data, rule_group='transaction_validation')
        assert len(result.warnings) > 0

        # Business day
        data = {
            'date': '2023-01-16',  # Monday
            'transaction_type': 'debit',
            'amount': -100.00
        }
        result = self.validator.validate(data, rule_group='transaction_validation')
        # Should not have weekend warnings
        weekend_warnings = [w for w in result.warnings if 'weekend' in w.lower()]
        assert len(weekend_warnings) == 0

    def test_transaction_date_sequence(self):
        """Test transaction date sequence validation."""
        current_data = {
            'date': '2023-01-15',
            'amount': -100.00
        }

        # With older previous transaction
        context = {
            'previous_transaction': {
                'date': '2023-01-10',
                'amount': -50.00
            }
        }
        result = self.validator.validate(current_data, rule_group='reconciliation_validation', context=context)
        assert result.is_valid

        # With newer previous transaction (error)
        context = {
            'previous_transaction': {
                'date': '2023-01-20',
                'amount': -50.00
            }
        }
        result = self.validator.validate(current_data, rule_group='reconciliation_validation', context=context)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_account_balance_consistency(self):
        """Test account balance consistency."""
        data = {
            'amount': -100.00,
            'balance': 900.00
        }

        # With correct previous balance
        context = {
            'previous_balance': 1000.00
        }
        result = self.validator.validate(data, rule_group='reconciliation_validation', context=context)
        assert result.is_valid

        # With incorrect previous balance
        context = {
            'previous_balance': 800.00
        }
        result = self.validator.validate(data, rule_group='reconciliation_validation', context=context)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_balance_reconciliation_check(self):
        """Test balance reconciliation check."""
        data = {
            'book_balance': 1000.00
        }

        # With matching bank balance
        context = {
            'reconciliation_data': {
                'bank_balance': 1000.00,
                'outstanding_checks': 0,
                'deposits_in_transit': 0
            }
        }
        result = self.validator.validate(data, rule_group='reconciliation_validation', context=context)
        assert result.is_valid

        # With non-matching balance
        context = {
            'reconciliation_data': {
                'bank_balance': 1200.00,
                'outstanding_checks': 0,
                'deposits_in_transit': 0
            }
        }
        result = self.validator.validate(data, rule_group='reconciliation_validation', context=context)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_tax_amount_calculation(self):
        """Test tax amount calculation validation."""
        # With correct tax calculation
        data = {
            'amount': 1000.00,
            'tax_amount': 150.00,
            'tax_rate': 0.15
        }
        result = self.validator.validate(data, rule_group='tax_compliance')
        assert result.is_valid

        # With incorrect tax calculation
        data = {
            'amount': 1000.00,
            'tax_amount': 100.00,  # Should be 150.00
            'tax_rate': 0.15
        }
        result = self.validator.validate(data, rule_group='tax_compliance')
        assert not result.is_valid
        assert len(result.errors) > 0

        # Without tax rate (estimate validation)
        data = {
            'amount': 100.00,
            'tax_amount': 80.00  # 80% tax rate - unreasonable
        }
        result = self.validator.validate(data, rule_group='tax_compliance')
        assert len(result.warnings) > 0

    def test_tax_exemption_validation(self):
        """Test tax exemption validation."""
        # Tax-exempt transaction with zero tax
        data = {
            'transaction_type': 'salary',
            'tax_exempt': True,
            'tax_amount': 0.00
        }
        result = self.validator.validate(data, rule_group='tax_compliance')
        assert result.is_valid

        # Tax-exempt transaction with non-zero tax (error)
        data = {
            'transaction_type': 'salary',
            'tax_exempt': True,
            'tax_amount': 50.00
        }
        result = self.validator.validate(data, rule_group='tax_compliance')
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_suspicious_transaction_detection(self):
        """Test suspicious transaction detection."""
        # Large round number
        data = {
            'amount': 10000.00,
            'description': 'Transfer'
        }
        result = self.validator.validate(data, rule_group='regulatory_compliance')
        assert len(result.warnings) > 0

        # Suspicious keywords
        data = {
            'amount': 500.00,
            'description': 'Urgent transfer needed'
        }
        result = self.validator.validate(data, rule_group='regulatory_compliance')
        assert len(result.warnings) > 0

    def test_regulatory_threshold(self):
        """Test regulatory reporting threshold validation."""
        # Below threshold
        data = {
            'amount': 5000.00
        }
        result = self.validator.validate(data, rule_group='regulatory_compliance')
        # Should not trigger regulatory warnings for smaller amounts
        regulatory_warnings = [w for w in result.warnings if 'threshold' in w.lower()]
        assert len(regulatory_warnings) == 0

        # Above threshold
        data = {
            'amount': 15000.00  # Above BACEN threshold
        }
        result = self.validator.validate(data, rule_group='regulatory_compliance')
        regulatory_warnings = [w for w in result.warnings if 'threshold' in w.lower()]
        assert len(regulatory_warnings) > 0

    def test_rule_management(self):
        """Test rule management functionality."""
        # Add a custom rule
        def custom_validation(data, context):
            result = ValidationResult()
            if data.get('custom_field') != 'expected_value':
                result.add_error("Custom field validation failed", 'custom_field')
            return result

        custom_rule = BusinessLogicRule(
            rule_id="custom_test_rule",
            name="Custom Test Rule",
            rule_type=BusinessRuleType.FINANCIAL_TRANSACTION,
            validation_function=custom_validation,
            description="Custom validation rule for testing"
        )

        self.validator.add_rule(custom_rule)
        assert "custom_test_rule" in self.validator.rules

        # Test the custom rule
        data = {'custom_field': 'expected_value'}
        result = self.validator.validate(data)
        assert result.is_valid

        # Test failure case
        data = {'custom_field': 'wrong_value'}
        result = self.validator.validate(data)
        assert not result.is_valid

        # Remove the custom rule
        self.validator.remove_rule("custom_test_rule")
        assert "custom_test_rule" not in self.validator.rules

    def test_rule_groups(self):
        """Test rule group functionality."""
        # Create a custom rule group
        self.validator.create_rule_group('test_group', ['transaction_amount_sign_consistency'])

        assert 'test_group' in self.validator.rule_groups
        assert 'transaction_amount_sign_consistency' in self.validator.rule_groups['test_group']

        # Test validation with the custom group
        data = {
            'transaction_type': 'debit',
            'amount': 100.00  # Should fail
        }
        result = self.validator.validate(data, rule_group='test_group')
        assert not result.is_valid

    def test_rule_enable_disable(self):
        """Test rule enable/disable functionality."""
        rule_id = 'transaction_amount_sign_consistency'

        # Disable rule
        self.validator.disable_rule(rule_id)
        assert not self.validator.rules[rule_id].enabled

        # Test with disabled rule
        data = {
            'transaction_type': 'debit',
            'amount': 100.00
        }
        result = self.validator.validate(data, rule_group='transaction_validation')
        # Should not trigger the disabled rule
        sign_errors = [e for e in result.errors if 'should have negative' in e or 'should have positive' in e]
        assert len(sign_errors) == 0

        # Re-enable rule
        self.validator.enable_rule(rule_id)
        assert self.validator.rules[rule_id].enabled

    def test_error_handling(self):
        """Test error handling in business logic validator."""
        # Test with invalid data types
        data = {
            'transaction_type': None,
            'amount': 'invalid_amount'
        }
        result = self.validator.validate(data, rule_group='transaction_validation')
        # Should handle gracefully without crashing
        assert isinstance(result, ValidationResult)

    def test_performance(self):
        """Test business logic validator performance."""
        import time

        # Test with multiple transactions
        transactions = [
            {'transaction_type': 'debit', 'amount': -100 * i, 'date': '2023-01-15'}
            for i in range(50)
        ]

        start_time = time.time()
        for transaction in transactions:
            result = self.validator.validate(transaction, rule_group='transaction_validation')
            assert isinstance(result, ValidationResult)
        end_time = time.time()

        duration = end_time - start_time
        # Should complete within reasonable time
        assert duration < 3.0  # 3 seconds for 50 validations

    def test_validation_result_structure(self):
        """Test that validation results have correct structure."""
        data = {
            'transaction_type': 'debit',
            'amount': 100.00
        }
        result = self.validator.validate(data, rule_group='transaction_validation')

        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'validation_duration')

        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.is_valid, bool)


class TestBusinessLogicRule:
    """Test cases for BusinessLogicRule."""

    def test_rule_creation(self):
        """Test business logic rule creation."""
        def dummy_validation(data, context):
            return ValidationResult()

        rule = BusinessLogicRule(
            rule_id="test_rule",
            name="Test Rule",
            rule_type=BusinessRuleType.FINANCIAL_TRANSACTION,
            validation_function=dummy_validation,
            description="Test rule description"
        )

        assert rule.rule_id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.rule_type == BusinessRuleType.FINANCIAL_TRANSACTION
        assert callable(rule.validation_function)

    def test_rule_validation_success(self):
        """Test successful rule validation."""
        def success_validation(data, context):
            result = ValidationResult()
            if data.get('test_field') == 'valid':
                return result
            result.add_error("Validation failed")
            return result

        rule = BusinessLogicRule(
            rule_id="test_rule",
            name="Test Rule",
            rule_type=BusinessRuleType.FINANCIAL_TRANSACTION,
            validation_function=success_validation,
            description="Test rule"
        )

        data = {'test_field': 'valid'}
        result = rule.validate(data)
        assert result.is_valid

    def test_rule_validation_failure(self):
        """Test failed rule validation."""
        def failure_validation(data, context):
            result = ValidationResult()
            result.add_error("Always fails", 'test_field', ValidationSeverity.HIGH)
            return result

        rule = BusinessLogicRule(
            rule_id="test_rule",
            name="Test Rule",
            rule_type=BusinessRuleType.FINANCIAL_TRANSACTION,
            validation_function=failure_validation,
            description="Test rule"
        )

        data = {'test_field': 'invalid'}
        result = rule.validate(data)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "Always fails" in result.errors[0]

    def test_disabled_rule(self):
        """Test disabled rule behavior."""
        def failure_validation(data, context):
            result = ValidationResult()
            result.add_error("Should not trigger")
            return result

        rule = BusinessLogicRule(
            rule_id="test_rule",
            name="Test Rule",
            rule_type=BusinessRuleType.FINANCIAL_TRANSACTION,
            validation_function=failure_validation,
            description="Test rule",
            enabled=False
        )

        data = {'test_field': 'value'}
        result = rule.validate(data)
        assert result.is_valid  # Should pass because rule is disabled

    def test_rule_with_context(self):
        """Test rule validation with context."""
        def context_validation(data, context):
            result = ValidationResult()
            if context and context.get('expected_value') == data.get('test_field'):
                return result
            result.add_error("Context validation failed")
            return result

        rule = BusinessLogicRule(
            rule_id="test_rule",
            name="Test Rule",
            rule_type=BusinessRuleType.FINANCIAL_TRANSACTION,
            validation_function=context_validation,
            description="Test rule with context"
        )

        data = {'test_field': 'expected'}
        context = {'expected_value': 'expected'}
        result = rule.validate(data, context)
        assert result.is_valid

        # Test with wrong context
        context = {'expected_value': 'wrong'}
        result = rule.validate(data, context)
        assert not result.is_valid


if __name__ == '__main__':
    pytest.main([__file__])