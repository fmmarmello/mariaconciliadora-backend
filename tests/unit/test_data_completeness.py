"""
DataCompletenessTestSuite - Comprehensive tests for data completeness and imputation

This module provides comprehensive tests for:
- Missing data pattern analysis testing
- Imputation strategy testing (mean, median, KNN, regression)
- Quality assessment testing for imputed values
- Integration testing with validation pipeline
- Performance testing for large datasets
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.services.data_completeness_analyzer import DataCompletenessAnalyzer
from src.services.advanced_imputation_engine import AdvancedImputationEngine
from src.utils.exceptions import ValidationError


class TestDataCompletenessAnalyzer:
    """Test DataCompletenessAnalyzer functionality"""

    @pytest.fixture
    def analyzer(self):
        """Create DataCompletenessAnalyzer instance for testing"""
        return DataCompletenessAnalyzer()

    @pytest.fixture
    def sample_complete_data(self):
        """Create sample complete dataset"""
        return pd.DataFrame({
            'id': range(1, 101),
            'amount': np.random.normal(1000, 200, 100),
            'description': [f'Transaction {i}' for i in range(1, 101)],
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'category': np.random.choice(['food', 'transport', 'entertainment'], 100),
            'balance': np.random.normal(5000, 1000, 100)
        })

    @pytest.fixture
    def sample_incomplete_data(self):
        """Create sample dataset with missing values"""
        data = pd.DataFrame({
            'id': range(1, 101),
            'amount': np.random.normal(1000, 200, 100),
            'description': [f'Transaction {i}' for i in range(1, 101)],
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'category': np.random.choice(['food', 'transport', 'entertainment'], 100),
            'balance': np.random.normal(5000, 1000, 100)
        })

        # Introduce missing values
        np.random.seed(42)
        missing_mask = np.random.random(100) < 0.2  # 20% missing
        data.loc[missing_mask, 'amount'] = np.nan

        missing_mask = np.random.random(100) < 0.15  # 15% missing
        data.loc[missing_mask, 'balance'] = np.nan

        missing_mask = np.random.random(100) < 0.1  # 10% missing
        data.loc[missing_mask, 'category'] = np.nan

        return data

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert hasattr(analyzer, 'config')
        assert 'completeness_thresholds' in analyzer.config
        assert 'critical_fields' in analyzer.config

    def test_analyzer_custom_config(self):
        """Test analyzer with custom configuration"""
        custom_config = {
            'completeness_thresholds': {
                'field_level': 0.9,
                'record_level': 0.8,
                'dataset_level': 0.85
            },
            'critical_fields': ['date', 'amount', 'description']
        }

        analyzer = DataCompletenessAnalyzer(custom_config)
        assert analyzer.config['completeness_thresholds']['field_level'] == 0.9
        assert 'date' in analyzer.config['critical_fields']

    def test_field_completeness_complete_data(self, analyzer, sample_complete_data):
        """Test field completeness analysis on complete data"""
        result = analyzer.analyze_field_completeness(sample_complete_data, 'amount')

        assert result['field_name'] == 'amount'
        assert result['completeness_score'] == 1.0
        assert result['missing_records'] == 0
        assert result['missing_percentage'] == 0.0
        assert result['total_records'] == 100

    def test_field_completeness_incomplete_data(self, analyzer, sample_incomplete_data):
        """Test field completeness analysis on incomplete data"""
        result = analyzer.analyze_field_completeness(sample_incomplete_data, 'amount')

        assert result['field_name'] == 'amount'
        assert result['completeness_score'] < 1.0
        assert result['missing_records'] > 0
        assert result['missing_percentage'] > 0.0
        assert result['total_records'] == 100

    def test_field_completeness_nonexistent_field(self, analyzer, sample_complete_data):
        """Test field completeness analysis for nonexistent field"""
        result = analyzer.analyze_field_completeness(sample_complete_data, 'nonexistent_field')

        assert result['field_name'] == 'nonexistent_field'
        assert result['completeness_score'] == 0.0
        assert 'error' in result

    def test_field_completeness_numeric_metrics(self, analyzer, sample_complete_data):
        """Test field completeness with numeric field metrics"""
        result = analyzer.analyze_field_completeness(sample_complete_data, 'amount')

        # Should include numeric-specific metrics
        assert 'mean_value' in result
        assert 'median_value' in result
        assert 'std_value' in result
        assert 'min_value' in result
        assert 'max_value' in result
        assert 'zero_values' in result
        assert 'negative_values' in result

    def test_field_completeness_string_metrics(self, analyzer, sample_complete_data):
        """Test field completeness with string field metrics"""
        result = analyzer.analyze_field_completeness(sample_complete_data, 'description')

        # Should include string-specific metrics
        assert 'avg_length' in result
        assert 'empty_strings' in result
        assert 'most_common_value' in result

    def test_field_completeness_datetime_metrics(self, analyzer, sample_complete_data):
        """Test field completeness with datetime field metrics"""
        result = analyzer.analyze_field_completeness(sample_complete_data, 'date')

        # Should include datetime-specific metrics
        assert 'date_range_days' in result
        assert 'future_dates' in result
        assert 'past_dates' in result

    def test_record_completeness_analysis(self, analyzer, sample_incomplete_data):
        """Test record-level completeness analysis"""
        results = analyzer.analyze_record_completeness(sample_incomplete_data)

        assert isinstance(results, list)
        assert len(results) == len(sample_incomplete_data)

        # Check structure of first result
        first_result = results[0]
        assert 'record_index' in first_result
        assert 'completeness_score' in first_result
        assert 'total_fields' in first_result
        assert 'filled_fields' in first_result
        assert 'missing_fields' in first_result
        assert 'is_complete' in first_result

    def test_record_completeness_with_record_id(self, analyzer, sample_incomplete_data):
        """Test record completeness analysis with record ID field"""
        results = analyzer.analyze_record_completeness(sample_incomplete_data, 'id')

        assert isinstance(results, list)
        assert len(results) == len(sample_incomplete_data)

        # Check that record_id is used
        first_result = results[0]
        assert 'record_id' in first_result
        assert first_result['record_id'] == 1  # First ID should be 1

    def test_dataset_completeness_analysis(self, analyzer, sample_incomplete_data):
        """Test dataset-level completeness analysis"""
        result = analyzer.analyze_dataset_completeness(sample_incomplete_data)

        assert isinstance(result, dict)
        assert 'dataset_info' in result
        assert 'overall_completeness' in result
        assert 'field_completeness' in result
        assert 'critical_fields' in result
        assert 'missing_patterns' in result
        assert 'summary' in result

        # Check dataset info
        dataset_info = result['dataset_info']
        assert 'total_records' in dataset_info
        assert 'total_fields' in dataset_info
        assert 'total_cells' in dataset_info
        assert 'missing_cells' in dataset_info

        # Check overall completeness
        overall = result['overall_completeness']
        assert 'score' in overall
        assert 'percentage' in overall
        assert 'is_acceptable' in overall

    def test_critical_fields_analysis(self, analyzer, sample_incomplete_data):
        """Test critical fields completeness analysis"""
        result = analyzer.analyze_dataset_completeness(sample_incomplete_data)
        critical_fields = result['critical_fields']

        assert isinstance(critical_fields, dict)

        # Should include all critical fields from config
        for field in analyzer.config['critical_fields']:
            assert field in critical_fields
            field_analysis = critical_fields[field]
            assert 'completeness_score' in field_analysis
            assert 'is_critical' in field_analysis
            assert 'meets_threshold' in field_analysis

    def test_missing_patterns_identification(self, analyzer):
        """Test missing data pattern identification"""
        # Create data with correlated missing patterns
        data = pd.DataFrame({
            'field1': [1, 2, np.nan, 4, 5, np.nan],
            'field2': [1, 2, np.nan, 4, 5, np.nan],  # Correlated with field1
            'field3': [1, np.nan, 3, np.nan, 5, 6],  # Different pattern
            'field4': [1, 2, 3, 4, 5, 6]  # No missing
        })

        result = analyzer.analyze_dataset_completeness(data)
        patterns = result['missing_patterns']

        assert isinstance(patterns, list)
        # Should identify correlated missing patterns
        if patterns:  # Only if patterns are detected
            for pattern in patterns:
                assert 'fields' in pattern
                assert 'correlation' in pattern
                assert 'support' in pattern
                assert 'pattern_type' in pattern

    def test_completeness_trends_analysis(self, analyzer):
        """Test completeness trends analysis over time"""
        # Create historical data
        base_date = datetime.now() - timedelta(days=60)
        historical_data = []

        for i in range(10):
            date = base_date + timedelta(days=i*7)  # Weekly data

            # Create data with improving completeness
            completeness_factor = min(0.3 + i * 0.07, 1.0)  # Improving over time

            data = pd.DataFrame({
                'amount': [100.0 if np.random.random() < completeness_factor else np.nan for _ in range(50)],
                'description': [f'Test {j}' if np.random.random() < completeness_factor else np.nan for j in range(50)],
                'date': [date.strftime('%Y-%m-%d')] * 50
            })

            historical_data.append((date, data))

        result = analyzer.analyze_completeness_trends(historical_data)

        assert isinstance(result, dict)
        assert 'field_trends' in result
        assert 'overall_trends' in result
        assert 'improvement_areas' in result

        # Check overall trends
        overall_trends = result['overall_trends']
        assert len(overall_trends) == 10
        for trend in overall_trends:
            assert 'timestamp' in trend
            assert 'completeness_score' in trend
            assert 'missing_cells' in trend

    def test_completeness_report_generation(self, analyzer, sample_incomplete_data):
        """Test comprehensive completeness report generation"""
        report = analyzer.generate_completeness_report(sample_incomplete_data)

        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'dataset_analysis' in report
        assert 'record_analysis_summary' in report
        assert 'recommendations' in report

        # Check record analysis summary
        record_summary = report['record_analysis_summary']
        assert 'total_records' in record_summary
        assert 'complete_records' in record_summary
        assert 'incomplete_records' in record_summary
        assert 'worst_records' in record_summary

    def test_completeness_report_with_recommendations(self, analyzer, sample_incomplete_data):
        """Test completeness report with recommendations"""
        report = analyzer.generate_completeness_report(sample_incomplete_data, include_recommendations=True)

        recommendations = report['recommendations']
        assert isinstance(recommendations, list)

        if recommendations:  # If there are recommendations
            for rec in recommendations:
                assert 'priority' in rec
                assert 'category' in rec
                assert 'recommendation' in rec

    def test_ensure_dataframe_conversion(self, analyzer):
        """Test DataFrame conversion from different input types"""
        # Test with list of dicts
        data_list = [
            {'amount': 100.0, 'description': 'Test 1'},
            {'amount': 200.0, 'description': 'Test 2'}
        ]

        df = analyzer._ensure_dataframe(data_list)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

        # Test with existing DataFrame
        existing_df = pd.DataFrame({'col': [1, 2, 3]})
        result_df = analyzer._ensure_dataframe(existing_df)
        assert isinstance(result_df, pd.DataFrame)
        assert result_df is not existing_df  # Should be a copy

    def test_ensure_dataframe_invalid_input(self, analyzer):
        """Test DataFrame conversion with invalid input"""
        with pytest.raises(ValidationError):
            analyzer._ensure_dataframe("invalid_input")


class TestImputationStrategies:
    """Test various imputation strategies"""

    @pytest.fixture
    def imputer(self):
        """Create AdvancedImputationEngine instance for testing"""
        return AdvancedImputationEngine()

    @pytest.fixture
    def data_with_missing(self):
        """Create test data with missing values"""
        np.random.seed(42)
        data = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0],
            'categorical_col': ['A', 'B', np.nan, 'A', 'B', np.nan, 'A', 'B'],
            'amount': [100.0, 200.0, np.nan, 400.0, 500.0, np.nan, 700.0, 800.0],
            'balance': [1000.0, 1100.0, np.nan, 1300.0, 1400.0, np.nan, 1600.0, 1700.0]
        })
        return data

    def test_statistical_imputation_mean(self, imputer, data_with_missing):
        """Test statistical imputation with mean strategy"""
        data, info = imputer.impute_statistical(data_with_missing, ['numeric_col'], method='mean')

        assert 'method' in info
        assert info['method'] == 'mean'
        assert 'columns_imputed' in info
        assert 'numeric_col' in info['columns_imputed']

        # Check that missing values were filled
        assert not data['numeric_col'].isnull().any()

    def test_statistical_imputation_median(self, imputer, data_with_missing):
        """Test statistical imputation with median strategy"""
        data, info = imputer.impute_statistical(data_with_missing, ['numeric_col'], method='median')

        assert info['method'] == 'median'
        assert not data['numeric_col'].isnull().any()

    def test_statistical_imputation_auto_selection(self, imputer, data_with_missing):
        """Test statistical imputation with auto strategy selection"""
        data, info = imputer.impute_statistical(data_with_missing, ['numeric_col'], method='auto')

        assert 'method' in info
        assert info['method'] in ['mean', 'median']
        assert not data['numeric_col'].isnull().any()

    def test_knn_imputation(self, imputer, data_with_missing):
        """Test KNN imputation"""
        numeric_data = data_with_missing[['amount', 'balance']].copy()
        data, info = imputer.impute_knn(numeric_data)

        assert 'method' in info
        assert info['method'] == 'knn'
        assert 'columns_imputed' in info

        # Check that missing values were filled
        assert not data['amount'].isnull().any()
        assert not data['balance'].isnull().any()

    def test_regression_imputation(self, imputer, data_with_missing):
        """Test regression-based imputation"""
        data, info = imputer.impute_regression(data_with_missing, 'amount', ['balance'])

        assert 'method' in info
        assert info['method'] == 'regression'
        assert 'target_column' in info
        assert 'predictor_columns' in info

    def test_time_series_imputation(self, imputer):
        """Test time series imputation"""
        # Create time series data
        ts_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'value': [1, 2, np.nan, 4, 5, np.nan, 7, 8, np.nan, 10]
        })

        data, info = imputer.impute_time_series(ts_data, 'date', 'value')

        assert 'method' in info
        assert info['method'] == 'time_series'
        assert not data['value'].isnull().any()

    def test_context_aware_imputation(self, imputer, data_with_missing):
        """Test context-aware imputation"""
        data, info = imputer.impute_context_aware(data_with_missing, 'amount')

        assert 'method' in info
        assert info['method'] == 'context_aware'
        assert 'target_column' in info

    def test_auto_imputation_strategies(self, imputer, data_with_missing):
        """Test auto imputation with different strategies"""
        strategies = ['simple', 'intelligent', 'comprehensive']

        for strategy in strategies:
            data, summary = imputer.auto_impute(data_with_missing, strategy)

            assert 'strategy' in summary
            assert summary['strategy'] == strategy
            assert 'total_imputations' in summary
            assert 'methods_used' in summary

    def test_imputation_quality_metrics(self, imputer, data_with_missing):
        """Test imputation quality metrics calculation"""
        # First impute some data
        imputed_data, _ = imputer.impute_statistical(data_with_missing, ['numeric_col'])

        # Calculate quality metrics
        metrics = imputer.get_imputation_quality_metrics(data_with_missing, imputed_data)

        assert 'overall_metrics' in metrics
        assert 'column_metrics' in metrics
        assert 'data_integrity_checks' in metrics

        assert 'original_missing_values' in metrics['overall_metrics']
        assert 'final_missing_values' in metrics['overall_metrics']
        assert 'imputation_success_rate' in metrics['overall_metrics']

    def test_confidence_scoring(self, imputer, data_with_missing):
        """Test confidence scoring for different imputation methods"""
        # Test statistical imputation confidence
        _, info = imputer.impute_statistical(data_with_missing, ['numeric_col'])
        assert 'confidence_scores' in info
        assert 'numeric_col' in info['confidence_scores']
        assert 0.0 <= info['confidence_scores']['numeric_col'] <= 1.0

        # Test KNN imputation confidence
        numeric_data = data_with_missing[['amount', 'balance']].copy()
        _, info = imputer.impute_knn(numeric_data)
        assert 'confidence_scores' in info

    def test_imputation_error_handling(self, imputer):
        """Test error handling in imputation"""
        # Test with empty data
        empty_df = pd.DataFrame()
        data, info = imputer.impute_statistical(empty_df)
        assert data.empty
        assert 'error' in info

        # Test with nonexistent column
        data, info = imputer.impute_statistical(data_with_missing, ['nonexistent_column'])
        assert 'error' in info
        assert 'nonexistent_column' in info['error']


class TestIntegrationTesting:
    """Test integration between completeness analysis and imputation"""

    @pytest.fixture
    def analyzer(self):
        """Create DataCompletenessAnalyzer instance"""
        return DataCompletenessAnalyzer()

    @pytest.fixture
    def imputer(self):
        """Create AdvancedImputationEngine instance"""
        return AdvancedImputationEngine()

    @pytest.fixture
    def validation_engine(self):
        """Create validation engine instance"""
        from src.utils.advanced_validation_engine import AdvancedValidationEngine
        return AdvancedValidationEngine()

    def test_completeness_imputation_integration(self, analyzer, imputer):
        """Test integration between completeness analysis and imputation"""
        # Create data with missing values
        data = pd.DataFrame({
            'amount': [100.0, np.nan, 300.0, np.nan, 500.0],
            'description': ['Test 1', np.nan, 'Test 3', 'Test 4', np.nan],
            'date': ['2024-01-01', '2024-01-02', np.nan, '2024-01-04', '2024-01-05']
        })

        # First analyze completeness
        completeness_report = analyzer.generate_completeness_report(data)

        # Then perform imputation
        imputed_data, imputation_info = imputer.auto_impute(data, 'intelligent')

        # Verify integration
        assert completeness_report['dataset_analysis']['overall_completeness']['score'] < 1.0
        assert not imputed_data.isnull().any().any()  # No missing values after imputation

    def test_validation_pipeline_integration(self, analyzer, validation_engine):
        """Test integration with validation pipeline"""
        # Create data with missing critical fields
        data = {
            'amount': 100.0,
            # Missing description (critical field)
            'date': '2024-01-15'
        }

        # Test validation
        result = validation_engine.validate(data, profile='transaction')

        assert isinstance(result, ValidationResult)
        # Should detect missing critical field
        assert not result.is_valid or len(result.errors) > 0

    def test_end_to_end_data_quality_workflow(self, analyzer, imputer, validation_engine):
        """Test end-to-end data quality improvement workflow"""
        # Step 1: Create problematic data
        raw_data = pd.DataFrame({
            'amount': [100.0, np.nan, 300.0, np.nan, 500.0],
            'description': ['Test 1', np.nan, 'Test 3', 'Test 4', np.nan],
            'date': ['2024-01-01', '2024-01-02', np.nan, '2024-01-04', '2024-01-05'],
            'transaction_type': ['debit', 'credit', np.nan, 'debit', 'credit']
        })

        # Step 2: Analyze completeness
        completeness_before = analyzer.analyze_dataset_completeness(raw_data)

        # Step 3: Perform imputation
        imputed_data, _ = imputer.auto_impute(raw_data, 'comprehensive')

        # Step 4: Analyze completeness after imputation
        completeness_after = analyzer.analyze_dataset_completeness(imputed_data)

        # Step 5: Validate final data
        validation_results = []
        for _, row in imputed_data.iterrows():
            result = validation_engine.validate(row.to_dict(), profile='transaction')
            validation_results.append(result)

        # Verify improvements
        assert completeness_after['overall_completeness']['score'] > completeness_before['overall_completeness']['score']
        assert not imputed_data.isnull().any().any()

        # Check validation results
        valid_results = [r for r in validation_results if r.is_valid]
        assert len(valid_results) > 0  # At least some should be valid


class TestPerformanceAndScalability:
    """Test performance and scalability of completeness and imputation systems"""

    @pytest.fixture
    def analyzer(self):
        """Create DataCompletenessAnalyzer instance"""
        return DataCompletenessAnalyzer()

    @pytest.fixture
    def imputer(self):
        """Create AdvancedImputationEngine instance"""
        return AdvancedImputationEngine()

    def test_large_dataset_completeness_analysis(self, analyzer):
        """Test completeness analysis performance with large dataset"""
        # Create large dataset
        np.random.seed(42)
        large_data = pd.DataFrame({
            'id': range(10000),
            'amount': np.random.normal(1000, 200, 10000),
            'balance': np.random.normal(5000, 1000, 10000),
            'description': [f'Transaction {i}' for i in range(10000)],
            'date': pd.date_range('2020-01-01', periods=10000, freq='H'),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 10000)
        })

        # Introduce missing values
        missing_mask = np.random.random(10000) < 0.1  # 10% missing
        large_data.loc[missing_mask, 'amount'] = np.nan

        import time
        start_time = time.time()

        result = analyzer.analyze_dataset_completeness(large_data)

        duration = time.time() - start_time

        assert isinstance(result, dict)
        assert 'overall_completeness' in result
        assert duration < 30.0  # Should complete within 30 seconds

    def test_large_dataset_imputation(self, imputer):
        """Test imputation performance with large dataset"""
        # Create large dataset with missing values
        np.random.seed(42)
        large_data = pd.DataFrame({
            'amount': np.random.normal(1000, 200, 5000),
            'balance': np.random.normal(5000, 1000, 5000),
            'category': np.random.choice(['A', 'B', 'C'], 5000)
        })

        # Introduce missing values
        missing_mask = np.random.random(5000) < 0.2  # 20% missing
        large_data.loc[missing_mask, 'amount'] = np.nan

        import time
        start_time = time.time()

        imputed_data, info = imputer.auto_impute(large_data, 'intelligent')

        duration = time.time() - start_time

        assert not imputed_data.isnull().any().any()
        assert duration < 60.0  # Should complete within 1 minute
        assert 'total_imputations' in info

    def test_memory_efficiency(self, analyzer, imputer):
        """Test memory efficiency with large datasets"""
        # Create moderately large dataset
        np.random.seed(42)
        data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 10000),
            'col2': np.random.normal(0, 1, 10000),
            'col3': np.random.normal(0, 1, 10000),
            'col4': np.random.normal(0, 1, 10000),
            'col5': np.random.normal(0, 1, 10000)
        })

        # Add missing values
        for col in data.columns:
            missing_mask = np.random.random(10000) < 0.1
            data.loc[missing_mask, col] = np.nan

        # Test that operations complete without memory issues
        result = analyzer.analyze_dataset_completeness(data)
        assert result is not None

        imputed_data, _ = imputer.auto_impute(data, 'simple')
        assert not imputed_data.isnull().any().any()

    def test_scalability_comparison(self, analyzer):
        """Test scalability with increasing dataset sizes"""
        sizes = [100, 500, 1000, 2500]

        performance_results = []

        for size in sizes:
            # Create dataset of given size
            np.random.seed(42)
            data = pd.DataFrame({
                'amount': np.random.normal(1000, 200, size),
                'balance': np.random.normal(5000, 1000, size),
                'description': [f'Test {i}' for i in range(size)]
            })

            # Add missing values
            missing_mask = np.random.random(size) < 0.15
            data.loc[missing_mask, 'amount'] = np.nan

            import time
            start_time = time.time()

            result = analyzer.analyze_dataset_completeness(data)

            duration = time.time() - start_time

            performance_results.append({
                'size': size,
                'duration': duration,
                'efficiency': duration / size  # Time per record
            })

            assert result is not None
            assert duration < 10.0  # Each should complete within 10 seconds

        # Check that efficiency doesn't degrade significantly
        efficiencies = [r['efficiency'] for r in performance_results]
        max_efficiency = max(efficiencies)
        min_efficiency = min(efficiencies)

        # Efficiency should not vary by more than 5x
        assert max_efficiency / min_efficiency < 5.0


if __name__ == "__main__":
    pytest.main([__file__])