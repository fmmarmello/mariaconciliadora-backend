#!/usr/bin/env python3
"""
Comprehensive error handling and edge case tests
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.services.model_manager import (
    ModelManager, RandomForestModel, XGBoostModel,
    LightGBMModel, BERTModel, KMeansModel
)
from src.services.feature_engineer import FeatureEngineer
from src.utils.exceptions import ValidationError, AIServiceError


@pytest.fixture
def mock_feature_engineer():
    """Mock FeatureEngineer for error testing"""
    mock_fe = Mock()
    return mock_fe


@pytest.fixture
def mock_model_selector():
    """Mock ModelSelector for error testing"""
    mock_ms = Mock()
    return mock_ms


@pytest.fixture
def model_manager(mock_feature_engineer, mock_model_selector):
    """ModelManager with mocked dependencies"""
    with patch('src.services.model_manager.FeatureEngineer', return_value=mock_feature_engineer), \
         patch('src.services.model_manager.ModelSelector', return_value=mock_model_selector):

        manager = ModelManager()
        return manager


class TestModelManagerErrorHandling:
    """Test error handling in ModelManager"""

    def test_train_model_invalid_model_type(self, model_manager):
        """Test training with invalid model type"""
        X = np.random.rand(10, 5)
        y = np.array(['class_a'] * 10)

        result = model_manager.train_model('invalid_model', X, y)

        assert result['success'] is False
        assert 'not found' in result['error']

    def test_train_model_with_none_data(self, model_manager):
        """Test training with None data"""
        result = model_manager.train_model('random_forest', None, None)

        assert result['success'] is False
        assert 'error' in result

    def test_train_model_with_empty_data(self, model_manager):
        """Test training with empty data"""
        X = np.array([])
        y = np.array([])

        result = model_manager.train_model('random_forest', X, y)

        assert result['success'] is False
        assert 'error' in result

    def test_train_model_with_nan_features(self, model_manager):
        """Test training with NaN features"""
        X = np.random.rand(20, 5)
        X[10:] = np.nan  # Introduce NaN values
        y = np.array(['class_a', 'class_b'] * 10)

        result = model_manager.train_model('random_forest', X, y)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result

    def test_train_model_with_inf_features(self, model_manager):
        """Test training with infinite features"""
        X = np.random.rand(20, 5)
        X[10:] = np.inf  # Introduce infinite values
        y = np.array(['class_a', 'class_b'] * 10)

        result = model_manager.train_model('random_forest', X, y)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result

    def test_predict_untrained_model(self, model_manager):
        """Test prediction with untrained model"""
        X = np.random.rand(5, 5)

        with pytest.raises(ValueError, match="not trained"):
            model_manager.predict('random_forest', X)

    def test_predict_with_wrong_dimensions(self, model_manager):
        """Test prediction with wrong input dimensions"""
        # Train with certain dimensions
        X_train = np.random.rand(20, 5)
        y_train = np.array(['class_a', 'class_b'] * 10)
        model_manager.train_model('random_forest', X_train, y_train)

        # Try to predict with different dimensions
        X_test = np.random.rand(5, 3)  # Different number of features

        with pytest.raises(Exception):  # Should raise some error
            model_manager.predict('random_forest', X_test)

    def test_predict_with_none_input(self, model_manager):
        """Test prediction with None input"""
        with pytest.raises(Exception):
            model_manager.predict('random_forest', None)

    def test_evaluate_untrained_model(self, model_manager):
        """Test evaluation of untrained model"""
        X = np.random.rand(10, 5)
        y = np.array(['class_a', 'class_b'] * 5)

        result = model_manager.evaluate_model('random_forest', X, y)

        assert 'error' in result
        assert 'not trained' in result['error']

    def test_evaluate_with_mismatched_data(self, model_manager):
        """Test evaluation with mismatched training/evaluation data"""
        # Train model
        X_train = np.random.rand(20, 5)
        y_train = np.array(['class_a', 'class_b'] * 10)
        model_manager.train_model('random_forest', X_train, y_train)

        # Try to evaluate with different feature count
        X_eval = np.random.rand(10, 3)  # Different number of features
        y_eval = np.array(['class_a', 'class_b'] * 5)

        result = model_manager.evaluate_model('random_forest', X_train, y_train, X_eval, y_eval)

        # Should handle gracefully or raise appropriate error
        assert isinstance(result, dict)

    def test_compare_models_empty_list(self, model_manager):
        """Test comparing with empty model list"""
        X = np.random.rand(20, 5)
        y = np.array(['class_a', 'class_b'] * 10)

        result = model_manager.compare_models(X, y, [])

        assert 'comparison_results' in result
        assert 'best_model' in result

    def test_compare_models_invalid_model(self, model_manager):
        """Test comparing with invalid model in list"""
        X = np.random.rand(20, 5)
        y = np.array(['class_a', 'class_b'] * 10)

        result = model_manager.compare_models(X, y, ['random_forest', 'invalid_model'])

        # Should handle invalid models gracefully
        assert isinstance(result, dict)
        assert 'comparison_results' in result

    def test_select_best_model_with_errors(self, model_manager, mock_model_selector):
        """Test model selection with errors"""
        mock_model_selector.select_best_model.side_effect = Exception("Selection error")

        X = np.random.rand(20, 5)
        y = np.array(['class_a', 'class_b'] * 10)

        result = model_manager.select_best_model(X, y)

        # Should fallback to simple selection
        assert 'best_model' in result
        assert 'recommendation' in result

    def test_process_data_with_invalid_transactions(self, model_manager, mock_feature_engineer):
        """Test processing data with invalid transactions"""
        mock_feature_engineer.create_comprehensive_features.side_effect = Exception("Processing error")

        invalid_transactions = [
            {'invalid_field': 'value'},
            None,
            {}
        ]

        X, y, feature_names = model_manager.process_data(invalid_transactions)

        # Should handle gracefully
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, list)

    def test_process_data_with_mixed_valid_invalid(self, model_manager, mock_feature_engineer):
        """Test processing data with mix of valid and invalid transactions"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(2, 5),
            ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        )

        mixed_transactions = [
            {'description': 'Valid transaction', 'amount': 100.0, 'date': '2024-01-01', 'category': 'test'},
            {'invalid': 'transaction'},
            {'description': 'Another valid', 'amount': 200.0, 'date': '2024-01-02', 'category': 'test'}
        ]

        X, y, feature_names = model_manager.process_data(mixed_transactions)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(feature_names) == 5

    def test_save_load_model_errors(self, model_manager, tmp_path):
        """Test save/load model error handling"""
        # Try to save untrained model
        success = model_manager.save_model('random_forest')
        assert success is False

        # Try to load non-existent model
        success = model_manager.load_model('random_forest', '/non/existent/path')
        assert success is False

    def test_get_model_info_invalid_model(self, model_manager):
        """Test getting info for invalid model"""
        info = model_manager.get_model_info('invalid_model')

        assert 'error' in info
        assert 'not found' in info['error']


class TestIndividualModelErrorHandling:
    """Test error handling in individual models"""

    def test_random_forest_with_single_class(self):
        """Test Random Forest with single class"""
        model = RandomForestModel({})

        X = np.random.rand(20, 5)
        y = np.array(['single_class'] * 20)

        result = model.train(X, y)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result

    def test_random_forest_with_few_samples(self):
        """Test Random Forest with very few samples"""
        model = RandomForestModel({})

        X = np.random.rand(5, 5)  # Very few samples
        y = np.array(['class_a', 'class_b', 'class_c', 'class_d', 'class_e'])

        result = model.train(X, y)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result

    def test_xgboost_with_extreme_values(self):
        """Test XGBoost with extreme values"""
        model = XGBoostModel({})

        X = np.random.rand(20, 5)
        X[10:] = 1e10  # Very large values
        y = np.array(['class_a', 'class_b'] * 10)

        result = model.train(X, y)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result

    def test_lightgbm_with_categorical_features(self):
        """Test LightGBM with categorical features"""
        model = LightGBMModel({})

        X = np.random.rand(20, 5)
        # Mix of continuous and categorical features
        X[:, 2] = np.random.choice([0, 1, 2], 20)  # Categorical column
        y = np.array(['class_a', 'class_b'] * 10)

        result = model.train(X, y)

        assert result['success'] is True

    def test_kmeans_with_high_dimensional_data(self):
        """Test KMeans with high-dimensional data"""
        config = {'n_clusters': 3}
        model = KMeansModel(config)

        X = np.random.rand(50, 100)  # High dimensional
        y = np.array(['dummy'] * 50)

        result = model.train(X, y)

        assert result['success'] is True
        assert result['n_clusters'] == 3

    def test_kmeans_with_few_clusters(self):
        """Test KMeans with very few data points per cluster"""
        config = {'n_clusters': 10}
        model = KMeansModel(config)

        X = np.random.rand(15, 5)  # Very few samples for many clusters
        y = np.array(['dummy'] * 15)

        result = model.train(X, y)

        assert result['success'] is True
        # Should adjust n_clusters based on data size

    def test_bert_model_without_texts(self):
        """Test BERT model without text data"""
        config = {'bert_config': {}}
        model = BERTModel(config)

        X = np.random.rand(10, 5)
        y = np.array(['class_a', 'class_b'] * 5)

        result = model.train(X, y)

        assert result['success'] is False
        assert 'BERT requires text data' in result['error']

    def test_bert_predict_without_texts(self):
        """Test BERT prediction without text data"""
        config = {'bert_config': {}}
        model = BERTModel(config)

        # Mock as trained
        model.is_trained = True

        X = np.random.rand(5, 5)

        with pytest.raises(ValueError, match="BERT prediction requires text data"):
            model.predict(X)


class TestFeatureEngineerErrorHandling:
    """Test error handling in FeatureEngineer"""

    def test_text_embeddings_with_empty_texts(self):
        """Test text embeddings with empty text list"""
        engineer = FeatureEngineer()

        embeddings = engineer.extract_text_embeddings([])

        assert embeddings.size == 0
        assert isinstance(embeddings, np.ndarray)

    def test_text_embeddings_with_none_texts(self):
        """Test text embeddings with None values"""
        engineer = FeatureEngineer()

        texts = ['valid text', None, '', 'another text']

        embeddings = engineer.extract_text_embeddings(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)

    def test_temporal_features_with_invalid_dates(self):
        """Test temporal features with invalid dates"""
        engineer = FeatureEngineer()

        invalid_dates = ['invalid-date', None, '', '2024-13-45', 'not-a-date']

        temporal_df = engineer.extract_temporal_features(invalid_dates)

        assert isinstance(temporal_df, pd.DataFrame)
        # Should handle invalid dates gracefully

    def test_temporal_features_empty_dates(self):
        """Test temporal features with empty date list"""
        engineer = FeatureEngineer()

        temporal_df = engineer.extract_temporal_features([])

        assert isinstance(temporal_df, pd.DataFrame)
        assert len(temporal_df) == 0

    def test_transaction_patterns_with_invalid_data(self):
        """Test transaction patterns with invalid transaction data"""
        engineer = FeatureEngineer()

        invalid_transactions = [
            {'invalid': 'data'},
            None,
            {},
            {'description': None, 'amount': 'invalid'},
            {'description': '', 'amount': None}
        ]

        pattern_df = engineer.extract_transaction_patterns(invalid_transactions)

        assert isinstance(pattern_df, pd.DataFrame)
        # Should handle invalid data gracefully

    def test_categorical_features_with_mixed_types(self):
        """Test categorical features with mixed data types"""
        engineer = FeatureEngineer()

        categories = ['string', 123, None, True, ['list'], {'dict': 'value'}]

        cat_df = engineer.extract_categorical_features(categories)

        assert isinstance(cat_df, pd.DataFrame)
        assert len(cat_df) == len(categories)

    def test_scaling_with_constant_features(self):
        """Test scaling with constant features"""
        engineer = FeatureEngineer()

        # Features with no variance
        features = np.ones((20, 5))  # All same values

        scaled_features, feature_names = engineer.scale_features(features)

        assert isinstance(scaled_features, np.ndarray)
        assert scaled_features.shape == features.shape

    def test_scaling_with_nan_features(self):
        """Test scaling with NaN features"""
        engineer = FeatureEngineer()

        features = np.random.rand(20, 5)
        features[10:] = np.nan

        scaled_features, feature_names = engineer.scale_features(features)

        assert isinstance(scaled_features, np.ndarray)
        # Should handle NaN values

    def test_feature_selection_with_few_features(self):
        """Test feature selection with very few features"""
        engineer = FeatureEngineer()

        X = np.random.rand(50, 3)  # Very few features
        y = np.random.choice([0, 1], 50)

        selected_features, selected_names = engineer.select_features(X, y)

        assert isinstance(selected_features, np.ndarray)
        assert selected_features.shape[0] == X.shape[0]

    def test_comprehensive_features_with_corrupted_data(self):
        """Test comprehensive features with corrupted transaction data"""
        engineer = FeatureEngineer()

        corrupted_data = [
            {'description': 'Valid', 'amount': 100.0, 'date': '2024-01-01', 'category': 'test'},
            {'description': None, 'amount': 'invalid', 'date': None, 'category': None},
            {'corrupted': 'data'},
            None
        ]

        features, feature_names = engineer.create_comprehensive_features(corrupted_data)

        assert isinstance(features, np.ndarray)
        assert isinstance(feature_names, list)
        # Should handle corrupted data gracefully


class TestHyperparameterOptimizationErrors:
    """Test error handling in hyperparameter optimization"""

    def test_optimization_with_invalid_model(self, model_manager):
        """Test optimization with invalid model type"""
        X = np.random.rand(50, 5)
        y = np.random.choice(['class_a', 'class_b'], 50)

        result = model_manager.optimize_hyperparameters('invalid_model', X, y, n_trials=3)

        assert result['success'] is False
        assert 'not found' in result['error']

    def test_optimization_with_insufficient_data(self, model_manager):
        """Test optimization with insufficient data"""
        X = np.random.rand(10, 5)  # Too few samples
        y = np.random.choice(['class_a', 'class_b'], 10)

        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=3)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result

    def test_optimization_with_single_class(self, model_manager):
        """Test optimization with single class target"""
        X = np.random.rand(50, 5)
        y = np.array(['single_class'] * 50)

        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=3)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result

    def test_optimization_with_high_cardinality_features(self, model_manager):
        """Test optimization with high cardinality categorical features"""
        X = np.random.rand(50, 5)
        # Make one column have many unique values
        X[:, 2] = np.random.choice(range(40), 50)  # High cardinality
        y = np.random.choice(['class_a', 'class_b'], 50)

        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=3)

        assert result['success'] is True

    def test_optimization_with_extreme_parameter_ranges(self, model_manager):
        """Test optimization with extreme parameter values"""
        X = np.random.rand(50, 5)
        y = np.random.choice(['class_a', 'class_b'], 50)

        # This should work with default parameter ranges
        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=3)

        assert result['success'] is True
        assert 'best_params' in result


class TestEdgeCases:
    """Test various edge cases"""

    def test_very_small_dataset(self, model_manager):
        """Test with very small dataset (minimum viable size)"""
        X = np.random.rand(2, 3)  # Minimal dataset
        y = np.array(['class_a', 'class_b'])

        result = model_manager.train_model('random_forest', X, y)

        # Should handle minimal dataset
        assert isinstance(result, dict)
        assert 'success' in result

    def test_single_feature_dataset(self, model_manager):
        """Test with single feature"""
        X = np.random.rand(20, 1)  # Single feature
        y = np.random.choice(['class_a', 'class_b'], 20)

        result = model_manager.train_model('random_forest', X, y)

        assert result['success'] is True

    def test_high_class_imbalance(self, model_manager):
        """Test with high class imbalance"""
        X = np.random.rand(100, 5)
        # Create severe imbalance: 95% one class, 5% another
        y = np.array(['majority'] * 95 + ['minority'] * 5)

        result = model_manager.train_model('random_forest', X, y)

        assert result['success'] is True

    def test_many_classes_few_samples(self, model_manager):
        """Test with many classes but few samples per class"""
        X = np.random.rand(20, 5)
        y = np.array([f'class_{i}' for i in range(10)] * 2)  # 10 classes, 2 samples each

        result = model_manager.train_model('random_forest', X, y)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result

    def test_long_text_descriptions(self, model_manager, mock_feature_engineer):
        """Test with very long text descriptions"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(5, 10),
            [f'feature_{i}' for i in range(10)]
        )

        long_text = 'A' * 10000  # Very long text
        data = [
            {'description': long_text, 'amount': 100.0, 'date': '2024-01-01', 'category': 'test'}
        ] * 5

        X, y, feature_names = model_manager.process_data(data)

        assert isinstance(X, np.ndarray)
        assert len(y) == 5

    def test_unicode_text_descriptions(self, model_manager, mock_feature_engineer):
        """Test with unicode text descriptions"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(3, 10),
            [f'feature_{i}' for i in range(10)]
        )

        unicode_texts = [
            'Compra café naïve résumé',
            'Pagamentoação ñoños',
            'Transferência naïve'
        ]

        data = [
            {'description': text, 'amount': 100.0, 'date': '2024-01-01', 'category': 'test'}
            for text in unicode_texts
        ]

        X, y, feature_names = model_manager.process_data(data)

        assert isinstance(X, np.ndarray)
        assert len(y) == 3

    def test_extreme_amount_values(self, model_manager, mock_feature_engineer):
        """Test with extreme amount values"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(5, 10),
            [f'feature_{i}' for i in range(10)]
        )

        extreme_amounts = [1e-10, 1e10, -1e10, 0, np.inf, -np.inf, np.nan]

        data = [
            {'description': 'Test', 'amount': amount, 'date': '2024-01-01', 'category': 'test'}
            for amount in extreme_amounts
        ]

        X, y, feature_names = model_manager.process_data(data)

        assert isinstance(X, np.ndarray)
        assert len(y) == len(extreme_amounts)

    def test_empty_strings_and_whitespace(self, model_manager, mock_feature_engineer):
        """Test with empty strings and whitespace"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(5, 10),
            [f'feature_{i}' for i in range(10)]
        )

        problematic_texts = ['', '   ', '\t\n', '\r\n\t']

        data = [
            {'description': text, 'amount': 100.0, 'date': '2024-01-01', 'category': 'test'}
            for text in problematic_texts
        ]

        X, y, feature_names = model_manager.process_data(data)

        assert isinstance(X, np.ndarray)
        assert len(y) == len(problematic_texts)

    def test_duplicate_transactions(self, model_manager, mock_feature_engineer):
        """Test with duplicate transactions"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(6, 10),
            [f'feature_{i}' for i in range(10)]
        )

        duplicate_data = [
            {'description': 'Same transaction', 'amount': 100.0, 'date': '2024-01-01', 'category': 'test'}
        ] * 6  # Same transaction repeated

        X, y, feature_names = model_manager.process_data(duplicate_data)

        assert isinstance(X, np.ndarray)
        assert len(y) == 6
        # All labels should be the same
        assert all(label == y[0] for label in y)

    def test_mixed_data_types_in_transactions(self, model_manager, mock_feature_engineer):
        """Test with mixed data types in transaction fields"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(5, 10),
            [f'feature_{i}' for i in range(10)]
        )

        mixed_data = [
            {'description': 'String desc', 'amount': 100.0, 'date': '2024-01-01', 'category': 'test'},
            {'description': 123, 'amount': '200.5', 'date': datetime(2024, 1, 2), 'category': None},
            {'description': True, 'amount': None, 'date': None, 'category': []},
            {'description': {}, 'amount': set(), 'date': 'invalid', 'category': 'test'},
            {'description': lambda x: x, 'amount': complex(1, 2), 'date': '2024-01-05', 'category': 'test'}
        ]

        X, y, feature_names = model_manager.process_data(mixed_data)

        assert isinstance(X, np.ndarray)
        assert len(y) == 5
        # Should handle mixed types gracefully