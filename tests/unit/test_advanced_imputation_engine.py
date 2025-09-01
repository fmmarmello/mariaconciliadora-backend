import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.services.advanced_imputation_engine import AdvancedImputationEngine
from src.utils.exceptions import ValidationError


class TestAdvancedImputationEngine:
    """Test suite for AdvancedImputationEngine"""

    @pytest.fixture
    def engine(self):
        """Create an AdvancedImputationEngine instance for testing"""
        return AdvancedImputationEngine()

    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'amount': [100.0, None, 300.0, None, 500.0],
            'balance': [1000.0, 1100.0, None, 1300.0, None],
            'description': ['Payment A', None, 'Income C', 'Expense D', None],
            'date': ['2023-01-01', '2023-01-02', None, '2023-01-04', '2023-01-05'],
            'category': ['A', 'B', None, 'A', 'B']
        })

    @pytest.fixture
    def complete_sample_data(self):
        """Create complete sample data for testing"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'amount': [100.0, 200.0, 300.0, 400.0, 500.0],
            'balance': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            'description': ['A', 'B', 'C', 'D', 'E'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'category': ['X', 'Y', 'X', 'Y', 'X']
        })

    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert hasattr(engine, 'config')
        assert hasattr(engine, 'imputation_history')
        assert hasattr(engine, 'confidence_scores')
        assert engine.config is not None

    def test_impute_statistical_mean(self, engine, sample_data_with_missing):
        """Test statistical imputation with mean strategy"""
        data, info = engine.impute_statistical(sample_data_with_missing, ['amount'], method='mean')

        assert 'method' in info
        assert 'columns_imputed' in info
        assert 'imputation_counts' in info
        assert info['method'] == 'mean'
        assert 'amount' in info['columns_imputed']

        # Check that missing values were filled
        assert not data['amount'].isnull().any()

    def test_impute_statistical_median(self, engine, sample_data_with_missing):
        """Test statistical imputation with median strategy"""
        data, info = engine.impute_statistical(sample_data_with_missing, ['amount'], method='median')

        assert info['method'] == 'median'
        assert not data['amount'].isnull().any()

    def test_impute_statistical_auto_selection(self, engine, sample_data_with_missing):
        """Test statistical imputation with auto strategy selection"""
        data, info = engine.impute_statistical(sample_data_with_missing, ['amount'], method='auto')

        assert 'method' in info
        assert not data['amount'].isnull().any()

    def test_impute_statistical_no_missing(self, engine, complete_sample_data):
        """Test statistical imputation when no missing values exist"""
        data, info = engine.impute_statistical(complete_sample_data, ['amount'])

        assert info['imputation_count'] == 0
        assert info['method'] == 'none_required'

    def test_impute_knn(self, engine, sample_data_with_missing):
        """Test KNN imputation"""
        # Select only numeric columns for KNN
        numeric_data = sample_data_with_missing[['amount', 'balance']].copy()
        data, info = engine.impute_knn(numeric_data)

        assert 'method' in info
        assert info['method'] == 'knn'
        assert 'columns_imputed' in info

        # Check that missing values were filled
        assert not data['amount'].isnull().any()
        assert not data['balance'].isnull().any()

    def test_impute_regression(self, engine, sample_data_with_missing):
        """Test regression-based imputation"""
        data, info = engine.impute_regression(sample_data_with_missing, 'amount', ['balance'])

        assert 'method' in info
        assert info['method'] == 'regression'
        assert 'target_column' in info
        assert 'predictor_columns' in info

    def test_impute_time_series(self, engine):
        """Test time series imputation"""
        # Create time series data
        ts_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': [1, 2, None, 4, 5, None, 7, 8, None, 10]
        })

        data, info = engine.impute_time_series(ts_data, 'date', 'value')

        assert 'method' in info
        assert info['method'] == 'time_series'
        assert not data['value'].isnull().any()

    def test_impute_context_aware(self, engine, sample_data_with_missing):
        """Test context-aware imputation"""
        data, info = engine.impute_context_aware(sample_data_with_missing, 'amount')

        assert 'method' in info
        assert info['method'] == 'context_aware'
        assert 'target_column' in info

    def test_auto_impute_simple(self, engine, sample_data_with_missing):
        """Test auto imputation with simple strategy"""
        data, summary = engine.auto_impute(sample_data_with_missing, 'simple')

        assert 'strategy' in summary
        assert summary['strategy'] == 'simple'
        assert 'total_imputations' in summary
        assert 'methods_used' in summary

    def test_auto_impute_intelligent(self, engine, sample_data_with_missing):
        """Test auto imputation with intelligent strategy"""
        data, summary = engine.auto_impute(sample_data_with_missing, 'intelligent')

        assert summary['strategy'] == 'intelligent'
        assert 'total_imputations' in summary
        assert 'methods_used' in summary

    def test_auto_impute_comprehensive(self, engine, sample_data_with_missing):
        """Test auto imputation with comprehensive strategy"""
        data, summary = engine.auto_impute(sample_data_with_missing, 'comprehensive')

        assert summary['strategy'] == 'comprehensive'
        assert 'total_imputations' in summary
        assert 'methods_used' in summary

    def test_get_imputation_quality_metrics(self, engine, sample_data_with_missing):
        """Test imputation quality metrics calculation"""
        # First impute some data
        imputed_data, _ = engine.impute_statistical(sample_data_with_missing, ['amount'])

        # Calculate quality metrics
        metrics = engine.get_imputation_quality_metrics(sample_data_with_missing, imputed_data)

        assert 'overall_metrics' in metrics
        assert 'column_metrics' in metrics
        assert 'data_integrity_checks' in metrics

        assert 'original_missing_values' in metrics['overall_metrics']
        assert 'final_missing_values' in metrics['overall_metrics']
        assert 'imputation_success_rate' in metrics['overall_metrics']

    def test_confidence_scoring_statistical(self, engine, sample_data_with_missing):
        """Test confidence scoring for statistical imputation"""
        _, info = engine.impute_statistical(sample_data_with_missing, ['amount'])

        assert 'confidence_scores' in info
        assert 'amount' in info['confidence_scores']
        assert 0.0 <= info['confidence_scores']['amount'] <= 1.0

    def test_confidence_scoring_knn(self, engine, sample_data_with_missing):
        """Test confidence scoring for KNN imputation"""
        numeric_data = sample_data_with_missing[['amount', 'balance']].copy()
        _, info = engine.impute_knn(numeric_data)

        assert 'confidence_scores' in info
        for col in ['amount', 'balance']:
            if col in info['confidence_scores']:
                assert 0.0 <= info['confidence_scores'][col] <= 1.0

    def test_error_handling_invalid_method(self, engine, sample_data_with_missing):
        """Test error handling for invalid imputation methods"""
        with pytest.raises(ValidationError):
            engine.impute_statistical(sample_data_with_missing, method='invalid_method')

    def test_error_handling_empty_data(self, engine):
        """Test error handling with empty data"""
        empty_df = pd.DataFrame()
        data, info = engine.impute_statistical(empty_df)

        assert data.empty
        assert 'error' in info

    def test_error_handling_nonexistent_column(self, engine, sample_data_with_missing):
        """Test error handling when requesting imputation for nonexistent column"""
        data, info = engine.impute_statistical(sample_data_with_missing, ['nonexistent_column'])

        assert 'error' in info
        assert 'nonexistent_column' in info['error']

    def test_ensure_dataframe_conversion(self, engine):
        """Test DataFrame conversion from different input types"""
        # Test with list of dicts
        data_list = [
            {'amount': 100, 'description': 'A'},
            {'amount': None, 'description': 'B'}
        ]

        result = engine._ensure_dataframe(data_list)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

        # Test with existing DataFrame
        df = pd.DataFrame({'col': [1, 2, 3]})
        result = engine._ensure_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        assert result is not df  # Should be a copy

    def test_ensure_dataframe_invalid_input(self, engine):
        """Test DataFrame conversion with invalid input"""
        with pytest.raises(ValidationError):
            engine._ensure_dataframe("invalid_input")

    def test_config_customization(self):
        """Test engine with custom configuration"""
        custom_config = {
            'statistical_methods': {
                'numeric_strategy': 'median',
                'categorical_strategy': 'most_frequent'
            },
            'knn_imputation': {
                'n_neighbors': 3
            }
        }

        engine = AdvancedImputationEngine(custom_config)
        assert engine.config['knn_imputation']['n_neighbors'] == 3
        assert engine.config['statistical_methods']['numeric_strategy'] == 'median'

    @patch('sklearn.impute.SimpleImputer')
    def test_statistical_imputation_fallback(self, mock_imputer, engine, sample_data_with_missing):
        """Test statistical imputation fallback behavior"""
        mock_imputer.side_effect = Exception("Imputation failed")

        data, info = engine.impute_statistical(sample_data_with_missing, ['amount'])

        # Should still return data and error info
        assert isinstance(data, pd.DataFrame)
        assert 'error' in info

    @patch('sklearn.impute.KNNImputer')
    def test_knn_imputation_fallback(self, mock_knn_imputer, engine, sample_data_with_missing):
        """Test KNN imputation fallback behavior"""
        mock_knn_imputer.side_effect = Exception("KNN imputation failed")

        numeric_data = sample_data_with_missing[['amount', 'balance']].copy()
        data, info = engine.impute_knn(numeric_data)

        # Should still return data and error info
        assert isinstance(data, pd.DataFrame)
        assert 'error' in info

    def test_calculate_statistical_confidence(self, engine):
        """Test statistical confidence calculation"""
        # Test with data that has missing values
        series_with_missing = pd.Series([1, 2, 3, None, 5])
        confidence = engine._calculate_statistical_confidence(series_with_missing, 1)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

        # Test with complete data
        complete_series = pd.Series([1, 2, 3, 4, 5])
        confidence = engine._calculate_statistical_confidence(complete_series, 0)

        assert confidence == 1.0

    def test_calculate_knn_confidence(self, engine):
        """Test KNN confidence calculation"""
        imputed_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        missing_mask = pd.Series([False, True, False, True, False])

        confidence = engine._calculate_knn_confidence(imputed_values, missing_mask)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_calculate_regression_confidence(self, engine):
        """Test regression confidence calculation"""
        predictions = np.array([1.0, 2.0, 3.0])
        training_target = pd.Series([1.1, 2.1, 3.1])

        confidence = engine._calculate_regression_confidence(predictions, training_target)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_calculate_time_series_confidence(self, engine):
        """Test time series confidence calculation"""
        # Create a smooth series (should have high confidence)
        smooth_series = pd.Series([1, 2, 3, 4, 5])
        confidence = engine._calculate_time_series_confidence(smooth_series)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_select_predictor_columns(self, engine, sample_data_with_missing):
        """Test predictor column selection for regression"""
        predictors = engine._select_predictor_columns(
            sample_data_with_missing, 'amount', sample_data_with_missing.dropna()
        )

        assert isinstance(predictors, list)
        # Should select numeric columns as predictors
        assert 'balance' in predictors

    def test_select_predictor_columns_no_correlation(self, engine):
        """Test predictor selection when no correlated columns exist"""
        # Create data with no correlation
        data = pd.DataFrame({
            'target': [1, 2, 3, 4, 5],
            'unrelated': [10, 20, 30, 40, 50]
        })

        predictors = engine._select_predictor_columns(data, 'target', data)

        # Should still return predictors if available
        assert isinstance(predictors, list)

    def test_choose_numeric_strategy_auto(self, engine):
        """Test automatic numeric strategy selection"""
        # Test with normal distribution (should choose mean)
        normal_data = pd.Series([1, 2, 3, 4, 5])
        strategy = engine._choose_numeric_strategy(normal_data, 'auto')
        assert strategy in ['mean', 'median']

        # Test with skewed data (should choose median)
        skewed_data = pd.Series([1, 1, 1, 1, 100])
        strategy = engine._choose_numeric_strategy(skewed_data, 'auto')
        assert strategy in ['mean', 'median']

    def test_choose_numeric_strategy_explicit(self, engine):
        """Test explicit numeric strategy selection"""
        data = pd.Series([1, 2, 3, 4, 5])
        strategy = engine._choose_numeric_strategy(data, 'median')
        assert strategy == 'median'

    @patch('src.services.advanced_imputation_engine.get_logger')
    def test_logging_integration(self, mock_get_logger, engine, sample_data_with_missing):
        """Test that logging is properly integrated"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        engine.impute_statistical(sample_data_with_missing, ['amount'])

        mock_logger.info.assert_called()

    def test_imputation_history_tracking(self, engine, sample_data_with_missing):
        """Test that imputation history is properly tracked"""
        initial_history_length = len(engine.imputation_history)

        engine.impute_statistical(sample_data_with_missing, ['amount'])

        assert len(engine.imputation_history) == initial_history_length + 1

    def test_scaler_initialization(self, engine):
        """Test that scalers are properly initialized"""
        assert hasattr(engine, 'scaler')
        assert engine.scaler is not None

    def test_imputer_initialization(self, engine):
        """Test that imputers are properly initialized"""
        assert hasattr(engine, 'statistical_imputer_numeric')
        assert hasattr(engine, 'statistical_imputer_categorical')
        assert hasattr(engine, 'knn_imputer')

    def test_edge_case_single_value_imputation(self, engine):
        """Test imputation with single values"""
        single_value_data = pd.DataFrame({
            'amount': [None],
            'description': ['Test']
        })

        data, info = engine.impute_statistical(single_value_data, ['amount'])

        assert not data['amount'].isnull().any()
        assert 'imputation_count' in info

    def test_edge_case_all_missing_imputation(self, engine):
        """Test imputation when all values are missing"""
        all_missing_data = pd.DataFrame({
            'amount': [None, None, None],
            'description': [None, None, None]
        })

        data, info = engine.impute_statistical(all_missing_data, ['amount'])

        assert not data['amount'].isnull().any()
        assert info['imputation_count'] == 3