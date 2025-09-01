import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.services.missing_data_handler import MissingDataHandler, ImputationStrategy, ConfidenceLevel
from src.utils.exceptions import ValidationError


class TestMissingDataHandler:
    """Test suite for MissingDataHandler"""

    @pytest.fixture
    def handler(self):
        """Create a MissingDataHandler instance for testing"""
        return MissingDataHandler()

    @pytest.fixture
    def sample_incomplete_data(self):
        """Create sample incomplete transaction data"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'date': ['2023-01-01', None, '2023-01-03', '2023-01-04', None],
            'amount': [100.50, None, 300.00, None, 500.75],
            'description': ['Payment A', None, 'Income C', 'Expense D', None],
            'transaction_type': ['credit', 'debit', None, 'debit', 'credit'],
            'balance': [1000.00, None, 1200.00, None, 1400.00]
        })

    @pytest.fixture
    def sample_complete_data(self):
        """Create sample complete transaction data"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'amount': [100.50, 200.25, 300.00, 400.10, 500.75],
            'description': ['Payment A', 'Purchase B', 'Income C', 'Expense D', 'Payment E'],
            'transaction_type': ['credit', 'debit', 'credit', 'debit', 'credit'],
            'balance': [1000.00, 1100.00, 1200.00, 1300.00, 1400.00]
        })

    def test_initialization(self, handler):
        """Test handler initialization"""
        assert handler is not None
        assert hasattr(handler, 'config')
        assert hasattr(handler, 'completeness_analyzer')
        assert hasattr(handler, 'imputation_engine')
        assert hasattr(handler, 'imputation_history')
        assert hasattr(handler, 'completeness_reports')
        assert handler.config is not None

    def test_analyze_and_impute_auto_strategy(self, handler, sample_incomplete_data):
        """Test comprehensive analysis and imputation with auto strategy"""
        result = handler.analyze_and_impute(sample_incomplete_data)

        assert isinstance(result, object)
        assert hasattr(result, 'original_data')
        assert hasattr(result, 'imputed_data')
        assert hasattr(result, 'strategy_used')
        assert hasattr(result, 'confidence_score')
        assert hasattr(result, 'confidence_level')
        assert hasattr(result, 'imputation_count')
        assert hasattr(result, 'columns_affected')
        assert hasattr(result, 'quality_metrics')

        assert result.imputation_count > 0
        assert isinstance(result.strategy_used, ImputationStrategy)
        assert isinstance(result.confidence_level, ConfidenceLevel)

    def test_analyze_and_impute_specific_strategy(self, handler, sample_incomplete_data):
        """Test analysis and imputation with specific strategy"""
        result = handler.analyze_and_impute(sample_incomplete_data, strategy='statistical')

        assert result.strategy_used == ImputationStrategy.STATISTICAL
        assert result.imputation_count > 0

    def test_analyze_and_impute_target_columns(self, handler, sample_incomplete_data):
        """Test analysis and imputation with specific target columns"""
        result = handler.analyze_and_impute(
            sample_incomplete_data,
            strategy='statistical',
            target_columns=['amount', 'balance']
        )

        assert 'amount' in result.columns_affected or 'balance' in result.columns_affected

    def test_analyze_and_impute_complete_data(self, handler, sample_complete_data):
        """Test analysis and imputation with complete data"""
        result = handler.analyze_and_impute(sample_complete_data)

        assert result.imputation_count == 0
        assert result.confidence_score == 1.0
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_generate_completeness_report(self, handler, sample_incomplete_data):
        """Test completeness report generation"""
        report = handler.generate_completeness_report(sample_incomplete_data)

        assert hasattr(report, 'dataset_completeness')
        assert hasattr(report, 'field_completeness')
        assert hasattr(report, 'record_completeness')
        assert hasattr(report, 'missing_patterns')
        assert hasattr(report, 'critical_issues')
        assert hasattr(report, 'recommendations')
        assert hasattr(report, 'timestamp')

        assert report.dataset_completeness < 1.0  # Should have missing data

    def test_get_imputation_recommendations(self, handler, sample_incomplete_data):
        """Test imputation recommendations generation"""
        recommendations = handler.get_imputation_recommendations(sample_incomplete_data)

        assert isinstance(recommendations, list)
        if recommendations:  # Only if recommendations are generated
            for rec in recommendations:
                assert 'type' in rec
                assert 'priority' in rec
                assert 'recommendation' in rec

    def test_get_performance_summary(self, handler):
        """Test performance summary generation"""
        # First perform some imputations to have data
        handler.analyze_and_impute(pd.DataFrame({
            'col': [1, None, 3]
        }))

        summary = handler.get_performance_summary()

        assert 'total_imputations' in summary
        assert 'successful_imputations' in summary
        assert 'success_rate' in summary
        assert 'average_confidence' in summary
        assert 'strategy_usage' in summary

    def test_strategy_selection_auto(self, handler, sample_incomplete_data):
        """Test automatic strategy selection"""
        report = handler.generate_completeness_report(sample_incomplete_data)
        strategy = handler._select_optimal_strategy(report, sample_incomplete_data)

        assert isinstance(strategy, ImputationStrategy)
        assert strategy in [ImputationStrategy.STATISTICAL, ImputationStrategy.KNN,
                          ImputationStrategy.CONTEXT_AWARE, ImputationStrategy.AUTO]

    def test_strategy_selection_high_missing(self, handler):
        """Test strategy selection for high missing data"""
        # Create data with very high missing rate
        high_missing_data = pd.DataFrame({
            'col1': [None, None, None, None, 1],
            'col2': [None, None, None, None, 2]
        })

        report = handler.generate_completeness_report(high_missing_data)
        strategy = handler._select_optimal_strategy(report, high_missing_data)

        # Should select comprehensive approach for high missing data
        assert strategy == ImputationStrategy.AUTO

    def test_strategy_selection_critical_fields(self, handler):
        """Test strategy selection when critical fields are missing"""
        # Create data with missing critical fields
        critical_missing_data = pd.DataFrame({
            'date': [None, None, None, None, None],
            'amount': [100, 200, 300, 400, 500],
            'description': ['A', 'B', 'C', 'D', 'E']
        })

        report = handler.generate_completeness_report(critical_missing_data)
        strategy = handler._select_optimal_strategy(report, critical_missing_data)

        # Should select context-aware for critical field issues
        assert strategy == ImputationStrategy.CONTEXT_AWARE

    def test_strategy_selection_time_series(self, handler):
        """Test strategy selection for time series data"""
        # Create time series data
        ts_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': [1, 2, None, 4, 5, None, 7, 8, None, 10]
        })

        report = handler.generate_completeness_report(ts_data)
        strategy = handler._select_optimal_strategy(report, ts_data)

        # Should detect time series and select appropriate strategy
        # (This might be statistical or time series depending on detection)

    def test_confidence_level_calculation(self, handler):
        """Test confidence level determination"""
        # Test high confidence
        high_confidence = handler._determine_confidence_level(0.9)
        assert high_confidence == ConfidenceLevel.HIGH

        # Test medium confidence
        medium_confidence = handler._determine_confidence_level(0.7)
        assert medium_confidence == ConfidenceLevel.MEDIUM

        # Test low confidence
        low_confidence = handler._determine_confidence_level(0.4)
        assert low_confidence == ConfidenceLevel.LOW

    def test_overall_confidence_calculation(self, handler):
        """Test overall confidence score calculation"""
        # Test with imputation details
        imputation_details = {
            'confidence_scores': {'col1': 0.8, 'col2': 0.9},
            'confidence_score': 0.85
        }

        confidence = handler._calculate_overall_confidence(imputation_details)
        assert 0.0 <= confidence <= 1.0

        # Test with empty details
        confidence = handler._calculate_overall_confidence({})
        assert confidence == 0.5  # Default confidence

    def test_apply_imputation_strategy_statistical(self, handler, sample_incomplete_data):
        """Test applying statistical imputation strategy"""
        data, details = handler._apply_imputation_strategy(
            sample_incomplete_data, ImputationStrategy.STATISTICAL, ['amount']
        )

        assert isinstance(data, pd.DataFrame)
        assert isinstance(details, dict)
        assert 'method' in details

    def test_apply_imputation_strategy_knn(self, handler, sample_incomplete_data):
        """Test applying KNN imputation strategy"""
        data, details = handler._apply_imputation_strategy(
            sample_incomplete_data, ImputationStrategy.KNN, ['amount', 'balance']
        )

        assert isinstance(data, pd.DataFrame)
        assert isinstance(details, dict)

    def test_apply_imputation_strategy_auto(self, handler, sample_incomplete_data):
        """Test applying auto imputation strategy"""
        data, details = handler._apply_imputation_strategy(
            sample_incomplete_data, ImputationStrategy.AUTO
        )

        assert isinstance(data, pd.DataFrame)
        assert isinstance(details, dict)
        assert 'total_imputations' in details

    def test_validate_imputation_results(self, handler, sample_incomplete_data):
        """Test imputation result validation"""
        # First impute some data
        imputed_data, _ = handler.imputation_engine.impute_statistical(sample_incomplete_data, ['amount'])

        validation = handler._validate_imputation_results(
            sample_incomplete_data, imputed_data, {'method': 'statistical'}
        )

        assert 'data_integrity' in validation
        assert 'warnings' in validation
        assert 'errors' in validation
        assert isinstance(validation['data_integrity'], bool)

    def test_update_performance_metrics(self, handler):
        """Test performance metrics updating"""
        initial_total = handler.performance_metrics['total_imputations']

        # Create a mock result
        mock_result = Mock()
        mock_result.confidence_level = ConfidenceLevel.HIGH
        mock_result.strategy_used = ImputationStrategy.STATISTICAL
        mock_result.imputation_count = 5

        handler._update_performance_metrics(mock_result)

        assert handler.performance_metrics['total_imputations'] == initial_total + 1
        assert handler.performance_metrics['successful_imputations'] >= 0

    def test_is_time_series_data_detection(self, handler):
        """Test time series data detection"""
        # Test with date column
        ts_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': [1, 2, 3, 4, 5]
        })

        assert handler._is_time_series_data(ts_data) == True

        # Test without date column
        non_ts_data = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'value': [1, 2, 3]
        })

        assert handler._is_time_series_data(non_ts_data) == False

    def test_has_correlated_missing_detection(self, handler):
        """Test correlated missing data detection"""
        # Create data with correlated missing
        corr_data = pd.DataFrame({
            'col1': [1, 2, None, 4, None],
            'col2': [1, 2, None, 4, None]  # Perfect correlation
        })

        assert handler._has_correlated_missing(corr_data) == True

        # Create data without correlated missing
        no_corr_data = pd.DataFrame({
            'col1': [1, None, 3, None, 5],
            'col2': [None, 2, None, 4, None]  # No correlation
        })

        assert handler._has_correlated_missing(no_corr_data) == False

    def test_should_auto_impute_logic(self, handler):
        """Test auto-imputation decision logic"""
        # Test with good data (should impute)
        good_report = Mock()
        good_report.dataset_completeness = 0.8

        good_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'amount': [100, 200],
            'description': ['A', 'B']
        })

        assert handler._should_auto_impute(good_report, good_data) == True

        # Test with very incomplete data (should not auto-impute)
        bad_report = Mock()
        bad_report.dataset_completeness = 0.5

        assert handler._should_auto_impute(bad_report, good_data) == False

    def test_clear_history(self, handler):
        """Test history clearing functionality"""
        # Add some data to history
        handler.imputation_history.append(Mock())
        handler.completeness_reports.append(Mock())

        initial_history = len(handler.imputation_history)
        initial_reports = len(handler.completeness_reports)

        assert initial_history > 0
        assert initial_reports > 0

        # Clear history
        handler.clear_history()

        assert len(handler.imputation_history) == 0
        assert len(handler.completeness_reports) == 0
        assert handler.performance_metrics['total_imputations'] == 0

    def test_ensure_dataframe_conversion(self, handler):
        """Test DataFrame conversion functionality"""
        # Test with list of dicts
        data_list = [
            {'amount': 100, 'description': 'A'},
            {'amount': None, 'description': 'B'}
        ]

        result = handler._ensure_dataframe(data_list)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

        # Test with existing DataFrame
        df = pd.DataFrame({'col': [1, 2, 3]})
        result = handler._ensure_dataframe(df)
        assert isinstance(result, pd.DataFrame)

    def test_ensure_dataframe_invalid_input(self, handler):
        """Test DataFrame conversion with invalid input"""
        with pytest.raises(ValidationError):
            handler._ensure_dataframe("invalid_input")

    def test_config_customization(self):
        """Test handler with custom configuration"""
        custom_config = {
            'strategy_selection': {
                'min_confidence_threshold': 0.7,
                'max_imputation_ratio': 0.6
            }
        }

        handler = MissingDataHandler(custom_config)
        assert handler.config['strategy_selection']['min_confidence_threshold'] == 0.7

    @patch('src.services.missing_data_handler.get_logger')
    def test_logging_integration(self, mock_get_logger, handler, sample_incomplete_data):
        """Test that logging is properly integrated"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        handler.analyze_and_impute(sample_incomplete_data)

        mock_logger.info.assert_called()

    def test_error_handling_invalid_strategy(self, handler, sample_incomplete_data):
        """Test error handling for invalid imputation strategy"""
        with pytest.raises(ValidationError):
            handler.analyze_and_impute(sample_incomplete_data, strategy='invalid_strategy')

    def test_error_handling_empty_data(self, handler):
        """Test error handling with empty data"""
        empty_df = pd.DataFrame()

        with pytest.raises(ValidationError):
            handler.analyze_and_impute(empty_df)

    def test_imputation_history_tracking(self, handler, sample_incomplete_data):
        """Test that imputation results are properly tracked in history"""
        initial_history_length = len(handler.imputation_history)

        handler.analyze_and_impute(sample_incomplete_data)

        assert len(handler.imputation_history) == initial_history_length + 1

        # Check that the result is properly stored
        latest_result = handler.imputation_history[-1]
        assert hasattr(latest_result, 'imputation_count')
        assert hasattr(latest_result, 'strategy_used')

    def test_completeness_reports_tracking(self, handler, sample_incomplete_data):
        """Test that completeness reports are properly tracked"""
        initial_reports_length = len(handler.completeness_reports)

        handler.analyze_and_impute(sample_incomplete_data)

        assert len(handler.completeness_reports) == initial_reports_length + 1

        # Check that the report is properly stored
        latest_report = handler.completeness_reports[-1]
        assert hasattr(latest_report, 'dataset_completeness')
        assert hasattr(latest_report, 'timestamp')

    def test_get_field_specific_strategy(self, handler):
        """Test field-specific strategy recommendation"""
        # Test numeric field
        numeric_data = pd.DataFrame({'amount': [1, 2, 3]})
        strategy = handler._get_field_specific_strategy(numeric_data, 'amount')
        assert strategy in ['statistical', 'context_aware']

        # Test categorical field
        cat_data = pd.DataFrame({'category': ['A', 'B', 'C']})
        strategy = handler._get_field_specific_strategy(cat_data, 'category')
        assert strategy == 'statistical'

    def test_edge_case_single_column_data(self, handler):
        """Test handling of single column data"""
        single_col_data = pd.DataFrame({'amount': [100, None, 300]})

        result = handler.analyze_and_impute(single_col_data)

        assert result.imputation_count >= 0
        assert isinstance(result.strategy_used, ImputationStrategy)

    def test_edge_case_all_missing_data(self, handler):
        """Test handling of data where all values are missing"""
        all_missing_data = pd.DataFrame({
            'amount': [None, None, None],
            'description': [None, None, None]
        })

        result = handler.analyze_and_impute(all_missing_data)

        assert result.imputation_count > 0
        assert result.confidence_level == ConfidenceLevel.LOW  # Should have low confidence

    def test_memory_management_large_dataset(self, handler):
        """Test memory management with larger datasets"""
        # Create a moderately large dataset
        large_data = pd.DataFrame({
            'id': range(1000),
            'amount': [100 if i % 3 != 0 else None for i in range(1000)],
            'description': [f'Desc_{i}' if i % 4 != 0 else None for i in range(1000)]
        })

        result = handler.analyze_and_impute(large_data)

        assert result.imputation_count > 0
        assert result.imputation_count < 1000  # Should not impute everything

    def test_concurrent_access_simulation(self, handler, sample_incomplete_data):
        """Test behavior under simulated concurrent access"""
        import threading

        results = []

        def worker():
            result = handler.analyze_and_impute(sample_incomplete_data.copy())
            results.append(result)

        # Simulate concurrent access
        threads = []
        for _ in range(3):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 3
        for result in results:
            assert result.imputation_count >= 0