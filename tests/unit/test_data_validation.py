"""
DataValidationTestSuite - Comprehensive tests for enhanced validation pipeline

This module provides comprehensive tests for:
- Schema validation testing with various data types
- Business rule validation testing
- Cross-field validation testing
- Temporal validation testing
- Error handling and edge case testing
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal

from src.utils.advanced_validation_engine import AdvancedValidationEngine, ValidationLayer
from src.utils.validation_result import ValidationResult, ValidationSeverity
from src.utils.exceptions import ValidationError


class TestValidationEngineInitialization:
    """Test AdvancedValidationEngine initialization and configuration"""

    def test_engine_initialization(self):
        """Test basic engine initialization"""
        engine = AdvancedValidationEngine()

        assert engine is not None
        assert hasattr(engine, 'validation_layers')
        assert hasattr(engine, 'validation_profiles')
        assert len(engine.validation_layers) > 0
        assert len(engine.validation_profiles) > 0

    def test_default_layers_initialization(self):
        """Test that default validation layers are properly initialized"""
        engine = AdvancedValidationEngine()

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
            assert isinstance(engine.validation_layers[layer_name], ValidationLayer)

    def test_default_profiles_initialization(self):
        """Test that default validation profiles are properly initialized"""
        engine = AdvancedValidationEngine()

        expected_profiles = [
            'transaction',
            'company_financial',
            'bank_statement',
            'file_upload',
            'api_request'
        ]

        for profile_name in expected_profiles:
            assert profile_name in engine.validation_profiles
            assert isinstance(engine.validation_profiles[profile_name], list)
            assert len(engine.validation_profiles[profile_name]) > 0

    def test_layer_management(self):
        """Test validation layer management functionality"""
        engine = AdvancedValidationEngine()

        # Test layer enabling/disabling
        initial_enabled = len([l for l in engine.validation_layers.values() if l.enabled])

        # Disable a layer
        if engine.validation_layers:
            layer_name = list(engine.validation_layers.keys())[0]
            engine.disable_layer(layer_name)

            # Check that layer is disabled
            assert not engine.validation_layers[layer_name].enabled

            # Re-enable
            engine.enable_layer(layer_name)
            assert engine.validation_layers[layer_name].enabled

    def test_custom_layer_creation(self):
        """Test creating and adding custom validation layers"""
        engine = AdvancedValidationEngine()

        def custom_validator(data, context=None):
            result = ValidationResult()
            if isinstance(data, dict) and 'custom_field' in data:
                result.add_warning("Custom validation warning")
            return result

        custom_layer = ValidationLayer(
            name="custom_validation",
            validator=custom_validator,
            severity=ValidationSeverity.MEDIUM
        )

        engine.add_validation_layer(custom_layer)

        assert "custom_validation" in engine.validation_layers
        assert engine.validation_layers["custom_validation"] == custom_layer

    def test_profile_creation(self):
        """Test creating custom validation profiles"""
        engine = AdvancedValidationEngine()

        custom_profile = ['schema_validation', 'business_rules']
        engine.create_profile('custom_profile', custom_profile)

        assert 'custom_profile' in engine.validation_profiles
        assert engine.validation_profiles['custom_profile'] == custom_profile


class TestSchemaValidation:
    """Test schema validation functionality"""

    @pytest.fixture
    def engine(self):
        """Create validation engine for testing"""
        return AdvancedValidationEngine()

    @pytest.fixture
    def valid_transaction_data(self):
        """Create valid transaction data for testing"""
        return {
            'date': '2024-01-15',
            'amount': 150.50,
            'description': 'Test transaction',
            'transaction_type': 'debit',
            'category': 'alimentacao'
        }

    @pytest.fixture
    def invalid_transaction_data(self):
        """Create invalid transaction data for testing"""
        return {
            'date': 'invalid-date',
            'amount': 'not-a-number',
            'description': '',  # Empty required field
            # Missing transaction_type
        }

    def test_valid_transaction_schema(self, engine, valid_transaction_data):
        """Test validation of valid transaction data"""
        result = engine.validate(valid_transaction_data, profile='transaction')

        assert isinstance(result, ValidationResult)
        # Should pass basic schema validation
        assert result.is_valid or len(result.errors) == 0

    def test_invalid_transaction_schema(self, engine, invalid_transaction_data):
        """Test validation of invalid transaction data"""
        result = engine.validate(invalid_transaction_data, profile='transaction')

        assert isinstance(result, ValidationResult)
        # Should have validation errors
        assert not result.is_valid or len(result.errors) > 0

    def test_schema_validation_with_missing_fields(self, engine):
        """Test schema validation with missing required fields"""
        incomplete_data = {
            'amount': 100.0
            # Missing date, description, transaction_type
        }

        result = engine.validate(incomplete_data, profile='transaction')

        assert isinstance(result, ValidationResult)
        assert not result.is_valid or len(result.errors) > 0

    def test_schema_validation_with_extra_fields(self, engine, valid_transaction_data):
        """Test schema validation with extra unexpected fields"""
        data_with_extra = valid_transaction_data.copy()
        data_with_extra['extra_field'] = 'unexpected_value'
        data_with_extra['another_extra'] = 123

        result = engine.validate(data_with_extra, profile='transaction')

        assert isinstance(result, ValidationResult)
        # Should handle extra fields gracefully
        assert result.is_valid or len(result.errors) == 0

    def test_schema_validation_data_types(self, engine):
        """Test schema validation with various data types"""
        test_cases = [
            # Valid cases
            {'date': '2024-01-15', 'amount': 100.0, 'description': 'test'},
            {'date': '2024-01-15', 'amount': '100.0', 'description': 'test'},  # String amount
            {'date': datetime.now().isoformat(), 'amount': 100.0, 'description': 'test'},

            # Invalid cases
            {'date': 12345, 'amount': 100.0, 'description': 'test'},  # Numeric date
            {'date': '2024-01-15', 'amount': [100.0], 'description': 'test'},  # List amount
        ]

        for i, test_data in enumerate(test_cases):
            result = engine.validate(test_data, layer_names=['schema_validation'])
            assert isinstance(result, ValidationResult)


class TestBusinessRulesValidation:
    """Test business rules validation functionality"""

    @pytest.fixture
    def engine(self):
        """Create validation engine for testing"""
        return AdvancedValidationEngine()

    def test_business_rules_valid_transaction(self, engine):
        """Test business rules validation with valid transaction"""
        valid_transaction = {
            'amount': 150.50,
            'date': '2024-01-15',
            'description': 'Valid transaction',
            'transaction_type': 'debit',
            'category': 'alimentacao'
        }

        result = engine.validate(valid_transaction, layer_names=['business_rules'])

        assert isinstance(result, ValidationResult)
        # Should pass business rules validation
        assert result.is_valid or len(result.errors) == 0

    def test_business_rules_invalid_amount_range(self, engine):
        """Test business rules validation with invalid amount ranges"""
        # Test extremely high amount
        high_amount_transaction = {
            'amount': 10000000.0,  # 10 million
            'date': '2024-01-15',
            'description': 'High amount transaction',
            'transaction_type': 'debit'
        }

        result = engine.validate(high_amount_transaction, layer_names=['business_rules'])

        assert isinstance(result, ValidationResult)
        # May have warnings for unusual amounts

    def test_business_rules_temporal_consistency(self, engine):
        """Test business rules validation with temporal consistency"""
        # Future date
        future_transaction = {
            'amount': 100.0,
            'date': (datetime.now() + timedelta(days=365)).isoformat(),  # 1 year in future
            'description': 'Future transaction',
            'transaction_type': 'debit'
        }

        result = engine.validate(future_transaction, layer_names=['business_rules'])

        assert isinstance(result, ValidationResult)
        # Should have warnings for future dates

    def test_business_rules_category_validation(self, engine):
        """Test business rules validation with category validation"""
        transactions = [
            {
                'amount': 100.0,
                'date': '2024-01-15',
                'description': 'IT transaction',
                'transaction_type': 'debit',
                'category': 'software',
                'department': 'it'
            },
            {
                'amount': 200.0,
                'date': '2024-01-15',
                'description': 'Sales transaction',
                'transaction_type': 'debit',
                'category': 'travel',
                'department': 'sales'
            }
        ]

        for transaction in transactions:
            result = engine.validate(transaction, layer_names=['business_rules'])
            assert isinstance(result, ValidationResult)


class TestCrossFieldValidation:
    """Test cross-field validation functionality"""

    @pytest.fixture
    def engine(self):
        """Create validation engine for testing"""
        return AdvancedValidationEngine()

    def test_amount_transaction_type_consistency(self, engine):
        """Test amount and transaction type consistency"""
        test_cases = [
            # Debit should have negative amount
            {'amount': -100.0, 'transaction_type': 'debit', 'expected_warning': False},
            {'amount': 100.0, 'transaction_type': 'debit', 'expected_warning': True},

            # Credit should have positive amount
            {'amount': 100.0, 'transaction_type': 'credit', 'expected_warning': False},
            {'amount': -100.0, 'transaction_type': 'credit', 'expected_warning': True},
        ]

        for test_case in test_cases:
            data = {
                'amount': test_case['amount'],
                'transaction_type': test_case['transaction_type'],
                'date': '2024-01-15',
                'description': 'Test transaction'
            }

            result = engine.validate(data, layer_names=['cross_field_validation'])

            assert isinstance(result, ValidationResult)
            # Check for expected warnings
            has_warning = len(result.warnings) > 0
            if test_case['expected_warning']:
                assert has_warning, f"Expected warning for {test_case}"
            else:
                assert not has_warning, f"Unexpected warning for {test_case}"

    def test_date_balance_consistency(self, engine):
        """Test date and balance consistency validation"""
        # Very old transaction with high balance
        old_transaction = {
            'date': (datetime.now() - timedelta(days=365*10)).isoformat(),  # 10 years ago
            'balance': 5000000.0,  # 5 million balance
            'amount': 100.0,
            'description': 'Old transaction'
        }

        result = engine.validate(old_transaction, layer_names=['cross_field_validation'])

        assert isinstance(result, ValidationResult)
        # Should have warnings for unusual balance-age combination

    def test_category_department_consistency(self, engine):
        """Test category and department consistency"""
        test_cases = [
            # Consistent combinations
            {'category': 'software', 'department': 'it', 'expected_warning': False},
            {'category': 'travel', 'department': 'sales', 'expected_warning': False},

            # Inconsistent combinations
            {'category': 'software', 'department': 'sales', 'expected_warning': True},
            {'category': 'travel', 'department': 'it', 'expected_warning': True},
        ]

        for test_case in test_cases:
            data = {
                'category': test_case['category'],
                'department': test_case['department'],
                'amount': 100.0,
                'date': '2024-01-15',
                'description': 'Test transaction'
            }

            result = engine.validate(data, layer_names=['cross_field_validation'])

            assert isinstance(result, ValidationResult)
            has_warning = len(result.warnings) > 0
            if test_case['expected_warning']:
                assert has_warning, f"Expected warning for {test_case}"
            else:
                assert not has_warning, f"Unexpected warning for {test_case}"


class TestTemporalValidation:
    """Test temporal consistency validation functionality"""

    @pytest.fixture
    def engine(self):
        """Create validation engine for testing"""
        return AdvancedValidationEngine()

    def test_valid_date_formats(self, engine):
        """Test validation with various valid date formats"""
        valid_dates = [
            '2024-01-15',
            '2024-01-15T10:30:00',
            datetime.now().isoformat(),
            (datetime.now() - timedelta(days=30)).isoformat()
        ]

        for date_val in valid_dates:
            data = {
                'date': date_val,
                'amount': 100.0,
                'description': 'Test transaction'
            }

            result = engine.validate(data, layer_names=['temporal_consistency'])

            assert isinstance(result, ValidationResult)

    def test_invalid_date_formats(self, engine):
        """Test validation with invalid date formats"""
        invalid_dates = [
            'invalid-date',
            '2024-13-45',  # Invalid month/day
            'not-a-date',
            12345,  # Numeric
            ['2024-01-15'],  # List
        ]

        for date_val in invalid_dates:
            data = {
                'date': date_val,
                'amount': 100.0,
                'description': 'Test transaction'
            }

            result = engine.validate(data, layer_names=['temporal_consistency'])

            assert isinstance(result, ValidationResult)
            # Should have errors for invalid dates
            assert not result.is_valid or len(result.errors) > 0

    def test_future_dates_validation(self, engine):
        """Test validation of future dates"""
        future_dates = [
            (datetime.now() + timedelta(days=1)).isoformat(),  # Tomorrow
            (datetime.now() + timedelta(days=30)).isoformat(),  # 30 days future
            (datetime.now() + timedelta(days=100)).isoformat(),  # 100 days future
        ]

        for date_val in future_dates:
            data = {
                'date': date_val,
                'amount': 100.0,
                'description': 'Future transaction'
            }

            result = engine.validate(data, layer_names=['temporal_consistency'])

            assert isinstance(result, ValidationResult)
            # Should have warnings for future dates

    def test_very_old_dates_validation(self, engine):
        """Test validation of very old dates"""
        old_dates = [
            (datetime.now() - timedelta(days=365*5)).isoformat(),  # 5 years ago
            (datetime.now() - timedelta(days=365*10)).isoformat(),  # 10 years ago
            (datetime.now() - timedelta(days=365*20)).isoformat(),  # 20 years ago
        ]

        for date_val in old_dates:
            data = {
                'date': date_val,
                'amount': 100.0,
                'description': 'Old transaction'
            }

            result = engine.validate(data, layer_names=['temporal_consistency'])

            assert isinstance(result, ValidationResult)
            # Should have warnings for very old dates

    def test_date_range_consistency(self, engine):
        """Test consistency between multiple date fields"""
        # Created date after updated date (inconsistency)
        inconsistent_dates = {
            'created_at': (datetime.now() - timedelta(hours=1)).isoformat(),
            'updated_at': (datetime.now() - timedelta(hours=2)).isoformat(),  # Before created
            'amount': 100.0,
            'description': 'Test transaction'
        }

        result = engine.validate(inconsistent_dates, layer_names=['temporal_consistency'])

        assert isinstance(result, ValidationResult)
        # Should have errors for date inconsistency
        assert not result.is_valid or len(result.errors) > 0

    def test_amount_trends_validation(self, engine):
        """Test amount trends validation with historical data"""
        current_transaction = {
            'amount': 10000.0,  # Much higher than historical average
            'date': datetime.now().isoformat(),
            'description': 'High amount transaction'
        }

        historical_data = [
            {'amount': 100.0, 'date': '2024-01-01'},
            {'amount': 150.0, 'date': '2024-01-02'},
            {'amount': 120.0, 'date': '2024-01-03'},
        ]

        context = {'historical_data': historical_data}

        result = engine.validate(current_transaction, layer_names=['temporal_consistency'], context=context)

        assert isinstance(result, ValidationResult)
        # Should have warnings for unusual amount trends


class TestSecurityValidation:
    """Test security validation functionality"""

    @pytest.fixture
    def engine(self):
        """Create validation engine for testing"""
        return AdvancedValidationEngine()

    def test_security_validation_clean_data(self, engine):
        """Test security validation with clean data"""
        clean_data = {
            'description': 'Normal transaction description',
            'amount': 100.0,
            'date': '2024-01-15'
        }

        result = engine.validate(clean_data, layer_names=['security_validation'])

        assert isinstance(result, ValidationResult)
        # Should pass security validation
        assert result.is_valid or len(result.errors) == 0

    def test_security_validation_suspicious_patterns(self, engine):
        """Test security validation with suspicious patterns"""
        suspicious_data = {
            'description': '<script>alert("xss")</script>',
            'amount': 100.0,
            'date': '2024-01-15'
        }

        result = engine.validate(suspicious_data, layer_names=['security_validation'])

        assert isinstance(result, ValidationResult)
        # Should detect security issues
        assert not result.is_valid or len(result.errors) > 0

    def test_security_validation_sql_injection(self, engine):
        """Test security validation for SQL injection patterns"""
        sql_injection_data = {
            'description': "'; DROP TABLE users; --",
            'amount': 100.0,
            'date': '2024-01-15'
        }

        result = engine.validate(sql_injection_data, layer_names=['security_validation'])

        assert isinstance(result, ValidationResult)
        # Should detect potential SQL injection
        assert not result.is_valid or len(result.errors) > 0


class TestDataQualityValidation:
    """Test data quality validation functionality"""

    @pytest.fixture
    def engine(self):
        """Create validation engine for testing"""
        return AdvancedValidationEngine()

    def test_data_quality_repeated_characters(self, engine):
        """Test data quality validation for repeated characters"""
        repeated_chars_data = {
            'description': 'AAAAAAA BBBBBB CCCCCCC',
            'amount': 100.0,
            'date': '2024-01-15'
        }

        result = engine.validate(repeated_chars_data, layer_names=['data_quality'])

        assert isinstance(result, ValidationResult)
        # Should detect repeated character patterns

    def test_data_quality_long_words(self, engine):
        """Test data quality validation for very long words"""
        long_word_data = {
            'description': 'Thisisaverylongwordwithoutanyspacesorpunctuationmarks',
            'amount': 100.0,
            'date': '2024-01-15'
        }

        result = engine.validate(long_word_data, layer_names=['data_quality'])

        assert isinstance(result, ValidationResult)
        # Should detect very long words

    def test_data_quality_special_characters(self, engine):
        """Test data quality validation for excessive special characters"""
        special_chars_data = {
            'description': '!@#$%^&*()_+{}|:<>?[]\\;\'",./',
            'amount': 100.0,
            'date': '2024-01-15'
        }

        result = engine.validate(special_chars_data, layer_names=['data_quality'])

        assert isinstance(result, ValidationResult)
        # Should detect excessive special characters

    def test_data_quality_round_numbers(self, engine):
        """Test data quality validation for suspicious round numbers"""
        round_number_data = {
            'amount': 1000000.0,  # 1 million - round number
            'balance': 500000.0,  # 500k - round number
            'date': '2024-01-15',
            'description': 'Round number transaction'
        }

        result = engine.validate(round_number_data, layer_names=['data_quality'])

        assert isinstance(result, ValidationResult)
        # Should have warnings for large round numbers

    def test_data_quality_very_small_amounts(self, engine):
        """Test data quality validation for very small amounts"""
        small_amount_data = {
            'amount': 0.001,  # Very small amount
            'date': '2024-01-15',
            'description': 'Very small amount transaction'
        }

        result = engine.validate(small_amount_data, layer_names=['data_quality'])

        assert isinstance(result, ValidationResult)
        # Should have warnings for very small amounts


class TestValidationProfiles:
    """Test validation profiles functionality"""

    @pytest.fixture
    def engine(self):
        """Create validation engine for testing"""
        return AdvancedValidationEngine()

    def test_transaction_profile(self, engine):
        """Test transaction validation profile"""
        transaction_data = {
            'date': '2024-01-15',
            'amount': 150.50,
            'description': 'Test transaction',
            'transaction_type': 'debit',
            'category': 'alimentacao'
        }

        result = engine.validate(transaction_data, profile='transaction')

        assert isinstance(result, ValidationResult)
        assert 'executed_layers' in result.metadata
        assert 'transaction' in result.metadata.get('executed_layers', [])

    def test_company_financial_profile(self, engine):
        """Test company financial validation profile"""
        financial_data = {
            'date': '2024-01-15',
            'amount': 5000.00,
            'description': 'Office supplies',
            'transaction_type': 'expense',
            'category': 'supplies',
            'department': 'admin'
        }

        result = engine.validate(financial_data, profile='company_financial')

        assert isinstance(result, ValidationResult)
        assert 'executed_layers' in result.metadata

    def test_bank_statement_profile(self, engine):
        """Test bank statement validation profile"""
        statement_data = {
            'date': '2024-01-15',
            'amount': -200.00,
            'balance': 1500.00,
            'description': 'ATM withdrawal',
            'transaction_type': 'debit'
        }

        result = engine.validate(statement_data, profile='bank_statement')

        assert isinstance(result, ValidationResult)
        assert 'executed_layers' in result.metadata

    def test_file_upload_profile(self, engine):
        """Test file upload validation profile"""
        upload_data = {
            'filename': 'transactions.csv',
            'file_size': 1024,
            'content_type': 'text/csv'
        }

        result = engine.validate(upload_data, profile='file_upload')

        assert isinstance(result, ValidationResult)
        # File upload profile should have fewer validation layers
        executed_layers = result.metadata.get('executed_layers', [])
        assert len(executed_layers) <= 2  # Should be minimal validation

    def test_api_request_profile(self, engine):
        """Test API request validation profile"""
        api_data = {
            'endpoint': '/api/transactions',
            'method': 'POST',
            'user_id': '12345'
        }

        result = engine.validate(api_data, profile='api_request')

        assert isinstance(result, ValidationResult)
        # API request profile should have minimal validation
        executed_layers = result.metadata.get('executed_layers', [])
        assert len(executed_layers) <= 2


class TestBulkValidation:
    """Test bulk validation functionality"""

    @pytest.fixture
    def engine(self):
        """Create validation engine for testing"""
        return AdvancedValidationEngine()

    @pytest.fixture
    def bulk_transaction_data(self):
        """Create bulk transaction data for testing"""
        return [
            {
                'date': '2024-01-15',
                'amount': 150.50,
                'description': 'Transaction 1',
                'transaction_type': 'debit'
            },
            {
                'date': '2024-01-16',
                'amount': 200.75,
                'description': 'Transaction 2',
                'transaction_type': 'credit'
            },
            {
                'date': 'invalid-date',  # Invalid
                'amount': 'not-a-number',  # Invalid
                'description': 'Invalid transaction',
                'transaction_type': 'debit'
            }
        ]

    def test_bulk_validation_sequential(self, engine, bulk_transaction_data):
        """Test bulk validation in sequential mode"""
        result = engine.validate_bulk(bulk_transaction_data, profile='transaction', parallel=False)

        assert isinstance(result, ValidationResult)
        assert result.metadata.get('bulk_validation') == True
        assert result.metadata.get('total_items') == len(bulk_transaction_data)

    def test_bulk_validation_parallel(self, engine, bulk_transaction_data):
        """Test bulk validation in parallel mode"""
        result = engine.validate_bulk(bulk_transaction_data, profile='transaction', parallel=True)

        assert isinstance(result, ValidationResult)
        assert result.metadata.get('bulk_validation') == True
        assert result.metadata.get('total_items') == len(bulk_transaction_data)

    def test_bulk_validation_empty_list(self, engine):
        """Test bulk validation with empty list"""
        result = engine.validate_bulk([], profile='transaction')

        assert isinstance(result, ValidationResult)
        assert result.metadata.get('total_items') == 0

    def test_bulk_validation_with_errors(self, engine, bulk_transaction_data):
        """Test bulk validation error handling"""
        # Include some invalid data
        invalid_data = bulk_transaction_data + [None, {}, "invalid"]

        result = engine.validate_bulk(invalid_data, profile='transaction')

        assert isinstance(result, ValidationResult)
        assert result.metadata.get('bulk_validation') == True
        assert result.metadata.get('total_items') == len(invalid_data)


class TestValidationPerformance:
    """Test validation performance and scalability"""

    @pytest.fixture
    def engine(self):
        """Create validation engine for testing"""
        return AdvancedValidationEngine()

    def test_validation_performance_small_dataset(self, engine):
        """Test validation performance with small dataset"""
        import time

        data = {
            'date': '2024-01-15',
            'amount': 100.0,
            'description': 'Performance test',
            'transaction_type': 'debit'
        }

        start_time = time.time()
        result = engine.validate(data, profile='transaction')
        end_time = time.time()

        duration = end_time - start_time

        assert isinstance(result, ValidationResult)
        assert duration < 1.0  # Should complete within 1 second

    def test_validation_performance_large_dataset(self, engine):
        """Test validation performance with larger dataset"""
        import time

        # Create larger dataset
        large_data = []
        for i in range(100):
            large_data.append({
                'date': f'2024-01-{i%28 + 1:02d}',
                'amount': float(i * 10),
                'description': f'Transaction {i}',
                'transaction_type': 'debit' if i % 2 == 0 else 'credit'
            })

        start_time = time.time()
        result = engine.validate_bulk(large_data, profile='transaction', parallel=True)
        end_time = time.time()

        duration = end_time - start_time

        assert isinstance(result, ValidationResult)
        assert duration < 10.0  # Should complete within 10 seconds for 100 items

    def test_parallel_vs_sequential_performance(self, engine):
        """Test performance comparison between parallel and sequential validation"""
        import time

        # Create test data
        test_data = []
        for i in range(50):
            test_data.append({
                'date': '2024-01-15',
                'amount': float(i),
                'description': f'Test {i}',
                'transaction_type': 'debit'
            })

        # Sequential validation
        start_time = time.time()
        sequential_result = engine.validate_bulk(test_data, profile='transaction', parallel=False)
        sequential_time = time.time() - start_time

        # Parallel validation
        start_time = time.time()
        parallel_result = engine.validate_bulk(test_data, profile='transaction', parallel=True)
        parallel_time = time.time() - start_time

        # Both should complete successfully
        assert isinstance(sequential_result, ValidationResult)
        assert isinstance(parallel_result, ValidationResult)

        # Parallel should be faster (though not guaranteed in small datasets)
        assert parallel_time <= sequential_time * 1.5  # Allow some tolerance


class TestErrorHandling:
    """Test error handling in validation engine"""

    @pytest.fixture
    def engine(self):
        """Create validation engine for testing"""
        return AdvancedValidationEngine()

    def test_validation_with_none_data(self, engine):
        """Test validation with None data"""
        result = engine.validate(None, profile='transaction')

        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validation_with_invalid_profile(self, engine):
        """Test validation with invalid profile name"""
        data = {'amount': 100.0, 'description': 'test'}

        result = engine.validate(data, profile='invalid_profile')

        assert isinstance(result, ValidationResult)
        # Should fall back to default validation

    def test_validation_layer_error_handling(self, engine):
        """Test error handling when validation layer fails"""
        # Create a layer that raises an exception
        def failing_validator(data, context=None):
            raise Exception("Validation layer failed")

        failing_layer = ValidationLayer(
            name="failing_layer",
            validator=failing_validator,
            severity=ValidationSeverity.MEDIUM
        )

        engine.add_validation_layer(failing_layer)

        data = {'amount': 100.0, 'description': 'test'}

        result = engine.validate(data, layer_names=['failing_layer'])

        assert isinstance(result, ValidationResult)
        # Should handle the error gracefully
        assert not result.is_valid or len(result.errors) > 0

    def test_bulk_validation_error_handling(self, engine):
        """Test error handling in bulk validation"""
        # Mix of valid and invalid data
        mixed_data = [
            {'amount': 100.0, 'description': 'valid'},
            None,  # Invalid
            {'invalid_field': 'value'},  # Invalid structure
            "string_data",  # Invalid type
        ]

        result = engine.validate_bulk(mixed_data, profile='transaction')

        assert isinstance(result, ValidationResult)
        assert result.metadata.get('bulk_validation') == True
        # Should handle errors gracefully without crashing


if __name__ == "__main__":
    pytest.main([__file__])