import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.services.data_completeness_analyzer import DataCompletenessAnalyzer
from src.services.advanced_imputation_engine import AdvancedImputationEngine
from src.services.missing_data_handler import MissingDataHandler
from src.services.ofx_processor import OFXProcessor
from src.services.xlsx_processor import XLSXProcessor


class TestDataQualityWorkflow:
    """Integration tests for the complete data quality workflow"""

    @pytest.fixture
    def sample_transaction_data(self):
        """Create comprehensive sample transaction data with various missing patterns"""
        return pd.DataFrame({
            'id': range(1, 101),
            'date': [f'2023-01-{str(i).zfill(2)}' if i <= 31 else None for i in range(1, 101)],
            'amount': [100.0 + i if i % 5 != 0 else None for i in range(1, 101)],
            'description': [f'Transaction {i}' if i % 7 != 0 else None for i in range(1, 101)],
            'transaction_type': ['credit' if i % 2 == 0 else 'debit' for i in range(1, 101)],
            'balance': [1000.0 + i * 10 if i % 3 != 0 else None for i in range(1, 101)],
            'category': [f'Category_{i % 5}' if i % 11 != 0 else None for i in range(1, 101)],
            'cost_center': [f'CC_{i % 3}' if i % 13 != 0 else None for i in range(1, 101)]
        })

    @pytest.fixture
    def sample_company_financial_data(self):
        """Create sample company financial data"""
        return pd.DataFrame({
            'id': range(1, 51),
            'date': [f'2023-01-{str(i).zfill(2)}' if i <= 31 else f'2023-02-{str(i-31).zfill(2)}' for i in range(1, 51)],
            'description': [f'Expense {i}' if i % 4 != 0 else None for i in range(1, 51)],
            'amount': [500.0 + i * 25 if i % 6 != 0 else None for i in range(1, 51)],
            'category': [f'Category_{i % 4}' if i % 8 != 0 else None for i in range(1, 51)],
            'cost_center': [f'CostCenter_{i % 3}' if i % 10 != 0 else None for i in range(1, 51)],
            'department': [f'Dept_{i % 5}' if i % 12 != 0 else None for i in range(1, 51)],
            'transaction_type': ['expense' if i % 2 == 0 else 'income' for i in range(1, 51)]
        })

    def test_complete_data_quality_workflow_transaction_data(self, sample_transaction_data):
        """Test complete data quality workflow for transaction data"""
        # Step 1: Initialize components
        analyzer = DataCompletenessAnalyzer()
        imputer = AdvancedImputationEngine()
        handler = MissingDataHandler()

        # Step 2: Analyze completeness
        completeness_report = analyzer.generate_completeness_report(sample_transaction_data)

        assert completeness_report['dataset_completeness'] < 1.0  # Should have missing data
        assert len(completeness_report['field_completeness']) > 0
        assert 'critical_issues' in completeness_report

        # Step 3: Get recommendations
        recommendations = handler.get_imputation_recommendations(sample_transaction_data)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0  # Should have recommendations for missing data

        # Step 4: Apply comprehensive imputation
        result = handler.analyze_and_impute(sample_transaction_data)

        assert result.imputation_count > 0
        assert result.confidence_score > 0
        assert len(result.columns_affected) > 0

        # Step 5: Verify imputation quality
        quality_metrics = imputer.get_imputation_quality_metrics(
            sample_transaction_data, result.imputed_data
        )

        assert 'overall_metrics' in quality_metrics
        assert quality_metrics['overall_metrics']['imputation_success_rate'] > 0

        # Step 6: Verify no new missing values were introduced
        original_missing = sample_transaction_data.isnull().sum().sum()
        final_missing = result.imputed_data.isnull().sum().sum()

        assert final_missing <= original_missing

    def test_complete_data_quality_workflow_company_data(self, sample_company_financial_data):
        """Test complete data quality workflow for company financial data"""
        # Step 1: Initialize components
        analyzer = DataCompletenessAnalyzer()
        imputer = AdvancedImputationEngine()
        handler = MissingDataHandler()

        # Step 2: Analyze completeness
        completeness_report = analyzer.generate_completeness_report(sample_company_financial_data)

        assert completeness_report['dataset_completeness'] < 1.0
        assert 'cost_center' in completeness_report['field_completeness']
        assert 'department' in completeness_report['field_completeness']

        # Step 3: Apply statistical imputation first
        imputed_data, stat_info = imputer.impute_statistical(sample_company_financial_data)

        assert stat_info['imputation_count'] > 0
        assert not imputed_data.isnull().any().any()  # All missing values should be filled

        # Step 4: Apply comprehensive analysis and imputation
        result = handler.analyze_and_impute(sample_company_financial_data)

        assert result.imputation_count > 0
        assert result.strategy_used.value in ['statistical', 'knn', 'context_aware', 'auto']

    def test_ofx_processor_with_data_quality_integration(self, sample_transaction_data):
        """Test OFX processor integration with data quality components"""
        # Create a mock OFX processor with data quality enabled
        processor = OFXProcessor()
        processor.data_quality_enabled = True

        # Mock the parsing result
        mock_parse_result = {
            'bank_name': 'Test Bank',
            'account_info': {'account_id': '12345'},
            'transactions': sample_transaction_data.to_dict('records'),
            'summary': {
                'total_transactions': len(sample_transaction_data),
                'total_credits': sample_transaction_data[sample_transaction_data['amount'] > 0]['amount'].sum(),
                'total_debits': abs(sample_transaction_data[sample_transaction_data['amount'] < 0]['amount'].sum()),
                'balance': 1000.0
            }
        }

        # Test the data quality analysis method
        quality_result = processor._analyze_data_quality(mock_parse_result['transactions'])

        assert 'completeness_analysis' in quality_result
        assert 'recommendations' in quality_result
        assert 'data_quality_score' in quality_result

        # Verify data quality score is reasonable
        assert 0.0 <= quality_result['data_quality_score'] <= 1.0

    def test_xlsx_processor_with_data_quality_integration(self, sample_company_financial_data):
        """Test XLSX processor integration with data quality components"""
        # Create a mock XLSX processor with data quality enabled
        processor = XLSXProcessor()
        processor.data_quality_enabled = True

        # Test the data quality analysis method
        quality_result = processor._analyze_data_quality(sample_company_financial_data.to_dict('records'))

        assert 'completeness_analysis' in quality_result
        assert 'recommendations' in quality_result
        assert 'data_quality_score' in quality_result

        # Verify financial-specific quality checks
        assert 'critical_issues' in quality_result

    def test_cross_component_data_flow(self, sample_transaction_data):
        """Test data flow between different components"""
        # Step 1: Completeness analysis
        analyzer = DataCompletenessAnalyzer()
        completeness_report = analyzer.generate_completeness_report(sample_transaction_data)

        # Step 2: Use completeness info for imputation strategy selection
        handler = MissingDataHandler()
        recommendations = handler.get_imputation_recommendations(sample_transaction_data)

        # Step 3: Apply recommended imputation
        if recommendations:
            # Use the first recommendation's suggested strategy
            suggested_strategy = recommendations[0].get('suggested_strategy', 'auto')
            result = handler.analyze_and_impute(sample_transaction_data, strategy=suggested_strategy)

            assert result.imputation_count >= 0
            assert result.confidence_score >= 0

    def test_performance_under_load(self, sample_transaction_data):
        """Test system performance with larger datasets"""
        # Create a larger dataset by duplicating the sample
        large_dataset = pd.concat([sample_transaction_data] * 10, ignore_index=True)

        handler = MissingDataHandler()

        import time
        start_time = time.time()

        result = handler.analyze_and_impute(large_dataset)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should process within reasonable time (adjust threshold as needed)
        assert processing_time < 30  # 30 seconds max
        assert result.imputation_count > 0

    def test_error_recovery_and_robustness(self):
        """Test error recovery and system robustness"""
        handler = MissingDataHandler()

        # Test with completely empty data
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):  # Should raise ValidationError
            handler.analyze_and_impute(empty_df)

        # Test with data that has only null values
        null_data = pd.DataFrame({
            'col1': [None, None, None],
            'col2': [None, None, None]
        })

        result = handler.analyze_and_impute(null_data)
        assert result.imputation_count > 0  # Should still attempt imputation

    def test_data_consistency_after_imputation(self, sample_transaction_data):
        """Test that data consistency is maintained after imputation"""
        handler = MissingDataHandler()
        original_data = sample_transaction_data.copy()

        result = handler.analyze_and_impute(sample_transaction_data)

        # Check that the data structure is preserved
        assert len(result.imputed_data) == len(original_data)
        assert list(result.imputed_data.columns) == list(original_data.columns)

        # Check that numeric columns still contain numeric data
        for col in result.imputed_data.select_dtypes(include=[np.number]).columns:
            assert pd.api.types.is_numeric_dtype(result.imputed_data[col])

        # Check that no new NaN values were introduced (except where expected)
        for col in original_data.columns:
            if not original_data[col].isnull().any():
                # If original column had no missing values, imputed should also have none
                assert not result.imputed_data[col].isnull().any()

    def test_imputation_strategy_adaptation(self):
        """Test that imputation strategies adapt to different data patterns"""
        handler = MissingDataHandler()

        # Test 1: Data with high missing rate
        high_missing_data = pd.DataFrame({
            'col1': [1, None, None, None, None],
            'col2': [2, None, None, None, None]
        })

        result1 = handler.analyze_and_impute(high_missing_data)
        assert result1.imputation_count > 0

        # Test 2: Data with correlated missing patterns
        correlated_missing_data = pd.DataFrame({
            'col1': [1, 2, None, 4, None],
            'col2': [1, 2, None, 4, None],  # Correlated with col1
            'col3': [1, None, 3, None, 5]  # Independent
        })

        result2 = handler.analyze_and_impute(correlated_missing_data)
        assert result2.imputation_count > 0

        # Test 3: Time series data
        ts_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': [1, 2, None, 4, 5, None, 7, 8, None, 10]
        })

        result3 = handler.analyze_and_impute(ts_data)
        assert result3.imputation_count > 0

    def test_quality_metrics_computation(self, sample_transaction_data):
        """Test comprehensive quality metrics computation"""
        analyzer = DataCompletenessAnalyzer()
        imputer = AdvancedImputationEngine()
        handler = MissingDataHandler()

        # Generate completeness report
        completeness_report = analyzer.generate_completeness_report(sample_transaction_data)

        # Apply imputation
        result = handler.analyze_and_impute(sample_transaction_data)

        # Compute quality metrics
        quality_metrics = imputer.get_imputation_quality_metrics(
            sample_transaction_data, result.imputed_data
        )

        # Verify all expected metrics are present
        required_metrics = [
            'overall_metrics',
            'column_metrics',
            'data_integrity_checks'
        ]

        for metric in required_metrics:
            assert metric in quality_metrics

        # Verify overall metrics
        overall = quality_metrics['overall_metrics']
        assert 'original_missing_values' in overall
        assert 'final_missing_values' in overall
        assert 'imputation_success_rate' in overall

        # Verify imputation was successful
        assert overall['imputation_success_rate'] >= 0.0
        assert overall['final_missing_values'] <= overall['original_missing_values']

    def test_workflow_with_different_data_types(self):
        """Test workflow with different data types and missing patterns"""
        handler = MissingDataHandler()

        # Test with mixed data types
        mixed_data = pd.DataFrame({
            'numeric_col': [1.0, 2.0, None, 4.0, None],
            'string_col': ['A', 'B', None, 'D', None],
            'date_col': ['2023-01-01', '2023-01-02', None, '2023-01-04', None],
            'boolean_col': [True, False, None, True, None]
        })

        result = handler.analyze_and_impute(mixed_data)

        assert result.imputation_count > 0

        # Verify data types are preserved
        assert pd.api.types.is_numeric_dtype(result.imputed_data['numeric_col'])
        assert pd.api.types.is_string_dtype(result.imputed_data['string_col'])
        assert pd.api.types.is_datetime64_any_dtype(result.imputed_data['date_col']) or \
               result.imputed_data['date_col'].dtype == 'object'

    def test_long_running_workflow_timeout(self, sample_transaction_data):
        """Test that long-running operations don't hang indefinitely"""
        handler = MissingDataHandler()

        # This should complete within a reasonable time
        import time
        start_time = time.time()

        result = handler.analyze_and_impute(sample_transaction_data)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in less than 10 seconds for this dataset
        assert duration < 10.0
        assert result is not None

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets"""
        handler = MissingDataHandler()

        # Create a large dataset
        large_data = pd.DataFrame({
            'id': range(10000),
            'amount': [100.0 if i % 10 != 0 else None for i in range(10000)],
            'description': [f'Desc_{i}' if i % 15 != 0 else None for i in range(10000)],
            'category': [f'Cat_{i % 5}' if i % 20 != 0 else None for i in range(10000)]
        })

        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        result = handler.analyze_and_impute(large_data)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB for this operation)
        assert memory_increase < 500
        assert result.imputation_count > 0