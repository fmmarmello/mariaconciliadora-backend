"""
CrossFieldValidationTestSuite - Comprehensive tests for cross-field validation and business logic

This module provides comprehensive tests for:
- Financial business rule testing
- Temporal consistency testing
- Referential integrity testing
- Bank-specific validation testing
- Regulatory compliance testing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from src.utils.business_logic_validator import (
    BusinessLogicValidator,
    BusinessLogicRule,
    BusinessRuleType
)
from src.utils.validation_result import ValidationResult, ValidationSeverity


class TestCrossFieldValidation:
    """Test cross-field validation functionality"""

    @pytest.fixture
    def validator(self):
        """Create BusinessLogicValidator instance for testing"""
        return BusinessLogicValidator()

    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for testing"""
        return {
            'id': 1,
            'description': 'Compra no supermercado Extra',
            'amount': -150.50,
            'date': '2024-01-15',
            'category': 'alimentacao',
            'type': 'debit',
            'balance': 850.00,
            'account_id': '12345-6',
            'bank_code': '001'
        }

    @pytest.fixture
    def sample_financial_data(self):
        """Create sample financial data for testing"""
        return {
            'company_id': '12345678000123',
            'transaction_date': '2024-01-15',
            'due_date': '2024-01-20',
            'amount': 1500.00,
            'tax_amount': 225.00,
            'tax_rate': 0.15,
            'supplier_id': '98765432000198',
            'payment_method': 'transfer',
            'department': 'IT',
            'cost_center': '001'
        }

    def test_amount_transaction_type_consistency(self, validator, sample_transaction_data):
        """Test amount and transaction type consistency"""
        # Valid debit transaction (negative amount)
        data = sample_transaction_data.copy()
        data.update({'type': 'debit', 'amount': -150.50})
        result = validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

        # Invalid debit transaction (positive amount)
        data = sample_transaction_data.copy()
        data.update({'type': 'debit', 'amount': 150.50})
        result = validator.validate(data, rule_group='transaction_validation')
        assert not result.is_valid
        assert len(result.errors) > 0

        # Valid credit transaction (positive amount)
        data = sample_transaction_data.copy()
        data.update({'type': 'credit', 'amount': 150.50})
        result = validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

    def test_balance_amount_consistency(self, validator, sample_transaction_data):
        """Test account balance and transaction amount consistency"""
        # Valid balance calculation
        data = sample_transaction_data.copy()
        data.update({
            'amount': -150.50,
            'balance': 850.00,
            'previous_balance': 1000.50
        })

        context = {'previous_balance': 1000.50}
        result = validator.validate(data, rule_group='reconciliation_validation', context=context)
        assert result.is_valid

        # Invalid balance calculation
        data = sample_transaction_data.copy()
        data.update({
            'amount': -150.50,
            'balance': 900.00,  # Should be 850.00
            'previous_balance': 1000.50
        })

        context = {'previous_balance': 1000.50}
        result = validator.validate(data, rule_group='reconciliation_validation', context=context)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_date_sequence_validation(self, validator, sample_transaction_data):
        """Test transaction date sequence validation"""
        # Valid date sequence
        current_data = sample_transaction_data.copy()
        current_data['date'] = '2024-01-15'

        context = {
            'previous_transaction': {
                'date': '2024-01-10',
                'amount': -50.00
            }
        }
        result = validator.validate(current_data, rule_group='reconciliation_validation', context=context)
        assert result.is_valid

        # Invalid date sequence (older than previous)
        current_data = sample_transaction_data.copy()
        current_data['date'] = '2024-01-05'  # Older than previous

        context = {
            'previous_transaction': {
                'date': '2024-01-10',
                'amount': -50.00
            }
        }
        result = validator.validate(current_data, rule_group='reconciliation_validation', context=context)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_category_amount_consistency(self, validator, sample_transaction_data):
        """Test category and amount consistency"""
        # Valid salary amount for salary category
        data = sample_transaction_data.copy()
        data.update({
            'category': 'salário',
            'amount': 5000.00,
            'type': 'credit'
        })
        result = validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

        # Invalid salary amount (too high)
        data = sample_transaction_data.copy()
        data.update({
            'category': 'salário',
            'amount': 100000.00,  # Unrealistic salary
            'type': 'credit'
        })
        result = validator.validate(data, rule_group='transaction_validation')
        assert len(result.warnings) > 0

        # Invalid utility amount (too high)
        data = sample_transaction_data.copy()
        data.update({
            'category': 'luz',
            'amount': -5000.00,  # Unrealistic utility bill
            'type': 'debit'
        })
        result = validator.validate(data, rule_group='transaction_validation')
        assert len(result.warnings) > 0

    def test_business_day_validation(self, validator, sample_transaction_data):
        """Test business day validation for transactions"""
        # Weekend transaction (Saturday)
        data = sample_transaction_data.copy()
        data['date'] = '2024-01-13'  # Saturday
        result = validator.validate(data, rule_group='transaction_validation')
        assert len(result.warnings) > 0

        # Business day transaction (Monday)
        data = sample_transaction_data.copy()
        data['date'] = '2024-01-15'  # Monday
        result = validator.validate(data, rule_group='transaction_validation')
        # Should not have weekend warnings
        weekend_warnings = [w for w in result.warnings if 'weekend' in w.lower() or 'sábado' in w.lower() or 'domingo' in w.lower()]
        assert len(weekend_warnings) == 0

    def test_duplicate_transaction_detection(self, validator, sample_transaction_data):
        """Test duplicate transaction detection"""
        # First transaction
        data1 = sample_transaction_data.copy()
        data1.update({
            'date': '2024-01-15',
            'amount': -150.50,
            'description': 'Compra Extra'
        })

        # Duplicate transaction
        data2 = data1.copy()

        # Without historical context, both should pass
        result1 = validator.validate(data1, rule_group='transaction_validation')
        result2 = validator.validate(data2, rule_group='transaction_validation')
        assert result1.is_valid
        assert result2.is_valid

        # With historical context, second should have warnings
        context = {'recent_transactions': [data1]}
        result2 = validator.validate(data2, rule_group='transaction_validation', context=context)
        assert len(result2.warnings) > 0

    def test_tax_calculation_validation(self, validator, sample_financial_data):
        """Test tax calculation validation"""
        # Valid tax calculation
        data = sample_financial_data.copy()
        data.update({
            'amount': 1000.00,
            'tax_amount': 150.00,
            'tax_rate': 0.15
        })
        result = validator.validate(data, rule_group='tax_compliance')
        assert result.is_valid

        # Invalid tax calculation
        data = sample_financial_data.copy()
        data.update({
            'amount': 1000.00,
            'tax_amount': 100.00,  # Should be 150.00
            'tax_rate': 0.15
        })
        result = validator.validate(data, rule_group='tax_compliance')
        assert not result.is_valid
        assert len(result.errors) > 0

        # Unreasonable tax rate
        data = sample_financial_data.copy()
        data.update({
            'amount': 100.00,
            'tax_amount': 80.00  # 80% tax rate - unreasonable
        })
        result = validator.validate(data, rule_group='tax_compliance')
        assert len(result.warnings) > 0

    def test_tax_exemption_validation(self, validator, sample_financial_data):
        """Test tax exemption validation"""
        # Valid tax-exempt transaction
        data = sample_financial_data.copy()
        data.update({
            'tax_exempt': True,
            'tax_amount': 0.00
        })
        result = validator.validate(data, rule_group='tax_compliance')
        assert result.is_valid

        # Invalid tax-exempt transaction (has tax)
        data = sample_financial_data.copy()
        data.update({
            'tax_exempt': True,
            'tax_amount': 50.00
        })
        result = validator.validate(data, rule_group='tax_compliance')
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_supplier_customer_validation(self, validator, sample_financial_data):
        """Test supplier/customer relationship validation"""
        # Valid supplier transaction
        data = sample_financial_data.copy()
        data.update({
            'supplier_id': '98765432000198',
            'amount': -1500.00,
            'type': 'debit'
        })
        result = validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

        # Invalid supplier transaction (positive amount for debit)
        data = sample_financial_data.copy()
        data.update({
            'supplier_id': '98765432000198',
            'amount': 1500.00,
            'type': 'debit'
        })
        result = validator.validate(data, rule_group='transaction_validation')
        assert not result.is_valid

    def test_department_cost_center_consistency(self, validator, sample_financial_data):
        """Test department and cost center consistency"""
        # Valid department-cost center combination
        data = sample_financial_data.copy()
        data.update({
            'department': 'IT',
            'cost_center': '001',
            'amount': -500.00
        })
        result = validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

        # Invalid department-cost center combination
        data = sample_financial_data.copy()
        data.update({
            'department': 'IT',
            'cost_center': '999',  # Invalid cost center for IT
            'amount': -500.00
        })
        result = validator.validate(data, rule_group='transaction_validation')
        # May have warnings for unusual combinations
        assert isinstance(result, ValidationResult)

    def test_payment_method_validation(self, validator, sample_financial_data):
        """Test payment method validation"""
        # Valid payment methods
        valid_methods = ['dinheiro', 'cartão', 'transferência', 'cheque', 'boleto']

        for method in valid_methods:
            data = sample_financial_data.copy()
            data['payment_method'] = method
            result = validator.validate(data, rule_group='transaction_validation')
            assert isinstance(result, ValidationResult)

        # Invalid payment method
        data = sample_financial_data.copy()
        data['payment_method'] = 'invalid_method'
        result = validator.validate(data, rule_group='transaction_validation')
        assert len(result.warnings) > 0

    def test_due_date_transaction_date_consistency(self, validator, sample_financial_data):
        """Test due date and transaction date consistency"""
        # Valid: transaction before due date
        data = sample_financial_data.copy()
        data.update({
            'transaction_date': '2024-01-15',
            'due_date': '2024-01-20'
        })
        result = validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

        # Invalid: transaction after due date
        data = sample_financial_data.copy()
        data.update({
            'transaction_date': '2024-01-25',
            'due_date': '2024-01-20'
        })
        result = validator.validate(data, rule_group='transaction_validation')
        assert len(result.warnings) > 0

    def test_account_bank_consistency(self, validator, sample_transaction_data):
        """Test account number and bank code consistency"""
        # Valid account-bank combination
        data = sample_transaction_data.copy()
        data.update({
            'account_id': '12345-6',
            'bank_code': '001'  # Banco do Brasil
        })
        result = validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

        # Invalid account format
        data = sample_transaction_data.copy()
        data.update({
            'account_id': 'invalid-account',
            'bank_code': '001'
        })
        result = validator.validate(data, rule_group='transaction_validation')
        assert len(result.warnings) > 0

    def test_suspicious_transaction_detection(self, validator, sample_transaction_data):
        """Test suspicious transaction detection"""
        # Large round number
        data = sample_transaction_data.copy()
        data.update({
            'amount': 10000.00,
            'description': 'Transferência'
        })
        result = validator.validate(data, rule_group='regulatory_compliance')
        assert len(result.warnings) > 0

        # Suspicious keywords
        data = sample_transaction_data.copy()
        data.update({
            'amount': 500.00,
            'description': 'Transferência urgente necessária'
        })
        result = validator.validate(data, rule_group='regulatory_compliance')
        assert len(result.warnings) > 0

        # Unusual timing
        data = sample_transaction_data.copy()
        data.update({
            'date': '2024-01-13',  # Saturday
            'amount': 50000.00
        })
        result = validator.validate(data, rule_group='regulatory_compliance')
        assert len(result.warnings) > 0

    def test_regulatory_threshold_validation(self, validator, sample_transaction_data):
        """Test regulatory reporting threshold validation"""
        # Below threshold
        data = sample_transaction_data.copy()
        data['amount'] = 5000.00
        result = validator.validate(data, rule_group='regulatory_compliance')
        # Should not trigger regulatory warnings for smaller amounts
        regulatory_warnings = [w for w in result.warnings if 'threshold' in w.lower() or 'regulatório' in w.lower()]
        assert len(regulatory_warnings) == 0

        # Above threshold
        data = sample_transaction_data.copy()
        data['amount'] = 15000.00  # Above BACEN threshold
        result = validator.validate(data, rule_group='regulatory_compliance')
        regulatory_warnings = [w for w in result.warnings if 'threshold' in w.lower() or 'regulatório' in w.lower()]
        assert len(regulatory_warnings) > 0

    def test_multi_currency_validation(self, validator, sample_transaction_data):
        """Test multi-currency transaction validation"""
        # Valid currency transaction
        data = sample_transaction_data.copy()
        data.update({
            'amount': 100.00,
            'currency': 'USD',
            'exchange_rate': 5.20,
            'amount_brl': 520.00
        })
        result = validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

        # Invalid currency calculation
        data = sample_transaction_data.copy()
        data.update({
            'amount': 100.00,
            'currency': 'USD',
            'exchange_rate': 5.20,
            'amount_brl': 500.00  # Should be 520.00
        })
        result = validator.validate(data, rule_group='transaction_validation')
        assert len(result.warnings) > 0

    def test_intercompany_transaction_validation(self, validator):
        """Test intercompany transaction validation"""
        # Valid intercompany transaction
        data = {
            'company_from': '12345678000123',
            'company_to': '98765432000198',
            'amount': 1000.00,
            'type': 'transfer',
            'description': 'Transferência intercompany'
        }
        result = validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

        # Invalid intercompany transaction (same company)
        data = {
            'company_from': '12345678000123',
            'company_to': '12345678000123',  # Same company
            'amount': 1000.00,
            'type': 'transfer'
        }
        result = validator.validate(data, rule_group='transaction_validation')
        assert len(result.warnings) > 0

    def test_budget_compliance_validation(self, validator, sample_financial_data):
        """Test budget compliance validation"""
        # Within budget
        data = sample_financial_data.copy()
        data.update({
            'amount': -500.00,
            'department': 'IT',
            'budget_limit': 1000.00,
            'budget_used': 300.00
        })

        context = {
            'budget_limits': {'IT': 1000.00},
            'budget_used': {'IT': 300.00}
        }
        result = validator.validate(data, rule_group='transaction_validation', context=context)
        assert result.is_valid

        # Over budget
        data = sample_financial_data.copy()
        data.update({
            'amount': -800.00,
            'department': 'IT',
            'budget_limit': 1000.00,
            'budget_used': 300.00
        })

        context = {
            'budget_limits': {'IT': 1000.00},
            'budget_used': {'IT': 300.00}
        }
        result = validator.validate(data, rule_group='transaction_validation', context=context)
        assert len(result.warnings) > 0

    def test_vendor_payment_validation(self, validator):
        """Test vendor payment validation"""
        # Valid vendor payment
        data = {
            'vendor_id': '98765432000198',
            'amount': -1500.00,
            'type': 'debit',
            'invoice_number': 'INV-2024-001',
            'due_date': '2024-01-20',
            'payment_date': '2024-01-15'
        }
        result = validator.validate(data, rule_group='transaction_validation')
        assert result.is_valid

        # Invalid vendor payment (payment before invoice)
        data = {
            'vendor_id': '98765432000198',
            'amount': -1500.00,
            'type': 'debit',
            'invoice_date': '2024-01-20',
            'payment_date': '2024-01-15'  # Payment before invoice
        }
        result = validator.validate(data, rule_group='transaction_validation')
        assert len(result.warnings) > 0

    def test_reconciliation_validation(self, validator):
        """Test account reconciliation validation"""
        # Valid reconciliation
        data = {
            'book_balance': 1000.00
        }

        context = {
            'reconciliation_data': {
                'bank_balance': 1000.00,
                'outstanding_checks': 0,
                'deposits_in_transit': 0
            }
        }
        result = validator.validate(data, rule_group='reconciliation_validation', context=context)
        assert result.is_valid

        # Invalid reconciliation
        data = {
            'book_balance': 1000.00
        }

        context = {
            'reconciliation_data': {
                'bank_balance': 1200.00,
                'outstanding_checks': 0,
                'deposits_in_transit': 0
            }
        }
        result = validator.validate(data, rule_group='reconciliation_validation', context=context)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_error_handling_edge_cases(self, validator):
        """Test error handling for edge cases"""
        # Test with None values
        data = {
            'amount': None,
            'type': None,
            'date': None
        }
        result = validator.validate(data, rule_group='transaction_validation')
        assert isinstance(result, ValidationResult)

        # Test with empty strings
        data = {
            'amount': '',
            'type': '',
            'date': ''
        }
        result = validator.validate(data, rule_group='transaction_validation')
        assert isinstance(result, ValidationResult)

        # Test with invalid data types
        data = {
            'amount': 'invalid_amount',
            'type': 123,
            'date': []
        }
        result = validator.validate(data, rule_group='transaction_validation')
        assert isinstance(result, ValidationResult)

    def test_performance_large_dataset(self, validator):
        """Test performance with large dataset"""
        import time

        # Create large dataset
        large_dataset = []
        for i in range(100):
            large_dataset.append({
                'id': i,
                'amount': float(i * 10),
                'type': 'debit' if i % 2 == 0 else 'credit',
                'date': '2024-01-15',
                'description': f'Transaction {i}'
            })

        start_time = time.time()

        for data in large_dataset:
            result = validator.validate(data, rule_group='transaction_validation')
            assert isinstance(result, ValidationResult)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time
        assert duration < 5.0  # 5 seconds for 100 validations

    def test_custom_rule_integration(self, validator):
        """Test custom rule integration"""
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

        validator.add_rule(custom_rule)

        # Test the custom rule
        data = {'custom_field': 'expected_value'}
        result = validator.validate(data)
        assert result.is_valid

        # Test failure case
        data = {'custom_field': 'wrong_value'}
        result = validator.validate(data)
        assert not result.is_valid

        # Clean up
        validator.remove_rule("custom_test_rule")

    def test_rule_group_management(self, validator):
        """Test rule group management"""
        # Create a custom rule group
        validator.create_rule_group('test_group', ['transaction_amount_sign_consistency'])

        assert 'test_group' in validator.rule_groups
        assert 'transaction_amount_sign_consistency' in validator.rule_groups['test_group']

        # Test validation with the custom group
        data = {
            'type': 'debit',
            'amount': 100.00  # Should fail
        }
        result = validator.validate(data, rule_group='test_group')
        assert not result.is_valid

    def test_rule_enable_disable_functionality(self, validator):
        """Test rule enable/disable functionality"""
        rule_id = 'transaction_amount_sign_consistency'

        # Disable rule
        validator.disable_rule(rule_id)
        assert not validator.rules[rule_id].enabled

        # Test with disabled rule
        data = {
            'type': 'debit',
            'amount': 100.00
        }
        result = validator.validate(data, rule_group='transaction_validation')
        # Should not trigger the disabled rule
        sign_errors = [e for e in result.errors if 'should have negative' in e or 'should have positive' in e]
        assert len(sign_errors) == 0

        # Re-enable rule
        validator.enable_rule(rule_id)
        assert validator.rules[rule_id].enabled


if __name__ == "__main__":
    pytest.main([__file__])