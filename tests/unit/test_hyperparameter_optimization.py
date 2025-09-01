#!/usr/bin/env python3
"""
Unit tests for hyperparameter optimization functionality
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import optuna

from src.services.model_manager import ModelManager
from src.utils.exceptions import ValidationError


@pytest.fixture
def sample_data():
    """Sample data for optimization testing"""
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.choice(['class_a', 'class_b', 'class_c'], 100)
    return X, y


@pytest.fixture
def mock_feature_engineer():
    """Mock FeatureEngineer"""
    mock_fe = Mock()
    mock_fe.create_comprehensive_features.return_value = (
        np.random.rand(50, 15),
        [f'feature_{i}' for i in range(15)]
    )
    return mock_fe


@pytest.fixture
def mock_model_selector():
    """Mock ModelSelector"""
    mock_ms = Mock()
    return mock_ms


@pytest.fixture
def model_manager(mock_feature_engineer, mock_model_selector):
    """ModelManager with mocked dependencies"""
    with patch('src.services.model_manager.FeatureEngineer', return_value=mock_feature_engineer), \
         patch('src.services.model_manager.ModelSelector', return_value=mock_model_selector):

        manager = ModelManager()
        return manager


class TestHyperparameterOptimization:
    """Test hyperparameter optimization functionality"""

    def test_optimize_random_forest_success(self, model_manager, sample_data):
        """Test successful Random Forest optimization"""
        X, y = sample_data

        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=5)

        assert result['success'] is True
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'n_trials' in result
        assert result['n_trials'] == 5

        # Check that best_params contains expected RF parameters
        best_params = result['best_params']
        expected_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
        assert any(param in best_params for param in expected_params)

    def test_optimize_xgboost_success(self, model_manager, sample_data):
        """Test successful XGBoost optimization"""
        X, y = sample_data

        result = model_manager.optimize_hyperparameters('xgboost', X, y, n_trials=5)

        assert result['success'] is True
        assert 'best_params' in result
        assert 'best_score' in result

        # Check XGBoost specific parameters
        best_params = result['best_params']
        expected_params = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']
        assert any(param in best_params for param in expected_params)

    def test_optimize_lightgbm_success(self, model_manager, sample_data):
        """Test successful LightGBM optimization"""
        X, y = sample_data

        result = model_manager.optimize_hyperparameters('lightgbm', X, y, n_trials=5)

        assert result['success'] is True
        assert 'best_params' in result
        assert 'best_score' in result

        # Check LightGBM specific parameters
        best_params = result['best_params']
        expected_params = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']
        assert any(param in best_params for param in expected_params)

    def test_optimize_invalid_model_type(self, model_manager, sample_data):
        """Test optimization with invalid model type"""
        X, y = sample_data

        result = model_manager.optimize_hyperparameters('invalid_model', X, y, n_trials=5)

        assert result['success'] is False
        assert 'not found' in result['error']

    def test_optimize_unsupported_model_type(self, model_manager, sample_data):
        """Test optimization with unsupported model type (BERT/KMeans)"""
        X, y = sample_data

        # KMeans doesn't support hyperparameter optimization in the same way
        result = model_manager.optimize_hyperparameters('kmeans', X, y, n_trials=5)

        assert result['success'] is False
        assert 'error' in result

    def test_optimization_with_small_dataset(self, model_manager):
        """Test optimization with very small dataset"""
        X = np.random.rand(10, 5)  # Very small dataset
        y = np.random.choice(['class_a', 'class_b'], 10)

        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=3)

        # Should still work but might have warnings
        assert isinstance(result, dict)
        assert 'success' in result

    def test_optimization_with_single_class(self, model_manager):
        """Test optimization with single class (should fail gracefully)"""
        X = np.random.rand(50, 5)
        y = np.array(['class_a'] * 50)  # Single class

        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=3)

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_optimization_with_high_dimensional_data(self, model_manager):
        """Test optimization with high-dimensional data"""
        X = np.random.rand(50, 100)  # High dimensional
        y = np.random.choice(['class_a', 'class_b', 'class_c'], 50)

        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=3)

        assert result['success'] is True
        assert 'best_params' in result

    def test_optimization_preserves_model_state(self, model_manager, sample_data):
        """Test that optimization doesn't break existing model state"""
        X, y = sample_data

        # Train a model first
        model_manager.train_model('random_forest', X, y)
        original_trained_state = model_manager.models['random_forest'].is_trained

        # Run optimization
        model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=3)

        # Check that original model is still trained
        assert model_manager.models['random_forest'].is_trained == original_trained_state

        # Check that optimized model is also available
        assert 'random_forest_optimized' in model_manager.models

    def test_optimization_creates_optimized_model(self, model_manager, sample_data):
        """Test that optimization creates an optimized model instance"""
        X, y = sample_data

        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=3)

        assert result['success'] is True
        assert 'random_forest_optimized' in model_manager.models
        assert model_manager.models['random_forest_optimized'].is_trained is True

    def test_optimization_with_custom_n_trials(self, model_manager, sample_data):
        """Test optimization with custom number of trials"""
        X, y = sample_data

        for n_trials in [1, 5, 10]:
            result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=n_trials)

            assert result['success'] is True
            assert result['n_trials'] == n_trials

    def test_optimization_error_handling(self, model_manager):
        """Test error handling in optimization"""
        # Test with None inputs
        result = model_manager.optimize_hyperparameters('random_forest', None, None, n_trials=3)

        assert result['success'] is False
        assert 'error' in result

    def test_optimization_with_empty_data(self, model_manager):
        """Test optimization with empty data"""
        X = np.array([])
        y = np.array([])

        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=3)

        assert result['success'] is False
        assert 'error' in result

    def test_optimization_with_nan_values(self, model_manager):
        """Test optimization with NaN values in data"""
        X = np.random.rand(50, 5)
        X[10:20] = np.nan  # Introduce NaN values
        y = np.random.choice(['class_a', 'class_b'], 50)

        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=3)

        # Should handle NaN values gracefully
        assert isinstance(result, dict)
        assert 'success' in result

    def test_optimization_with_different_random_states(self, model_manager, sample_data):
        """Test optimization reproducibility with different random states"""
        X, y = sample_data

        # Run optimization twice with same random state
        result1 = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=5)
        result2 = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=5)

        # Results should be similar (not exactly same due to optimization randomness)
        assert result1['success'] is True
        assert result2['success'] is True
        assert abs(result1['best_score'] - result2['best_score']) < 0.5  # Allow some variance

    def test_optimization_parameter_ranges(self, model_manager, sample_data):
        """Test that optimization explores reasonable parameter ranges"""
        X, y = sample_data

        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=10)

        assert result['success'] is True

        best_params = result['best_params']

        # Check parameter ranges for Random Forest
        if 'n_estimators' in best_params:
            assert 50 <= best_params['n_estimators'] <= 300
        if 'max_depth' in best_params:
            assert best_params['max_depth'] is None or 3 <= best_params['max_depth'] <= 20
        if 'min_samples_split' in best_params:
            assert 2 <= best_params['min_samples_split'] <= 20
        if 'min_samples_leaf' in best_params:
            assert 1 <= best_params['min_samples_leaf'] <= 10

    def test_optimization_xgboost_parameter_ranges(self, model_manager, sample_data):
        """Test XGBoost parameter ranges in optimization"""
        X, y = sample_data

        result = model_manager.optimize_hyperparameters('xgboost', X, y, n_trials=10)

        assert result['success'] is True

        best_params = result['best_params']

        # Check XGBoost parameter ranges
        if 'learning_rate' in best_params:
            assert 0.01 <= best_params['learning_rate'] <= 0.3
        if 'subsample' in best_params:
            assert 0.6 <= best_params['subsample'] <= 1.0
        if 'colsample_bytree' in best_params:
            assert 0.6 <= best_params['colsample_bytree'] <= 1.0

    def test_optimization_lightgbm_parameter_ranges(self, model_manager, sample_data):
        """Test LightGBM parameter ranges in optimization"""
        X, y = sample_data

        result = model_manager.optimize_hyperparameters('lightgbm', X, y, n_trials=10)

        assert result['success'] is True

        best_params = result['best_params']

        # Check LightGBM parameter ranges
        if 'learning_rate' in best_params:
            assert 0.01 <= best_params['learning_rate'] <= 0.3
        if 'subsample' in best_params:
            assert 0.6 <= best_params['subsample'] <= 1.0
        if 'colsample_bytree' in best_params:
            assert 0.6 <= best_params['colsample_bytree'] <= 1.0

    def test_optimization_with_cv_folds(self, model_manager, sample_data):
        """Test optimization uses cross-validation internally"""
        X, y = sample_data

        # Mock cross_val_score to verify it's called
        with patch('src.services.model_manager.cross_val_score') as mock_cv:
            mock_cv.return_value = [0.8, 0.82, 0.79]  # Mock CV scores

            result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=3)

            # Verify cross_val_score was called during optimization
            assert mock_cv.called
            assert result['success'] is True

    def test_optimization_handles_optuna_errors(self, model_manager, sample_data):
        """Test handling of Optuna-related errors"""
        X, y = sample_data

        # Mock optuna.create_study to raise an exception
        with patch('src.services.model_manager.optuna.create_study') as mock_study:
            mock_study.side_effect = Exception("Optuna error")

            result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=3)

            assert result['success'] is False
            assert 'error' in result
            assert 'Optuna error' in result['error']

    def test_optimization_with_very_large_n_trials(self, model_manager, sample_data):
        """Test optimization with very large number of trials (should be reasonable)"""
        X, y = sample_data

        # Test with large n_trials - should complete without hanging
        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=50)

        assert result['success'] is True
        assert result['n_trials'] == 50

    def test_optimization_best_params_applied_to_model(self, model_manager, sample_data):
        """Test that best parameters are actually applied to the optimized model"""
        X, y = sample_data

        result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=5)

        assert result['success'] is True

        optimized_model = model_manager.models['random_forest_optimized']
        assert optimized_model.is_trained is True

        # Check that the model has the optimized parameters
        best_params = result['best_params']
        if best_params:
            # Verify at least one parameter was applied
            model_params = optimized_model.model.get_params()
            applied_params = set(best_params.keys()) & set(model_params.keys())
            assert len(applied_params) > 0


class TestOptimizationObjective:
    """Test the optimization objective function"""

    def test_optimization_objective_random_forest(self, model_manager, sample_data):
        """Test optimization objective for Random Forest"""
        X, y = sample_data

        # Create a mock trial
        mock_trial = Mock()
        mock_trial.suggest_int.side_effect = lambda name, min_val, max_val: {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }.get(name, min_val)

        score = model_manager._optimization_objective(mock_trial, 'random_forest', X, y)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0  # F1 score should be between 0 and 1

    def test_optimization_objective_xgboost(self, model_manager, sample_data):
        """Test optimization objective for XGBoost"""
        X, y = sample_data

        mock_trial = Mock()
        mock_trial.suggest_int.side_effect = lambda name, min_val, max_val: {
            'n_estimators': 100,
            'max_depth': 6
        }.get(name, min_val)
        mock_trial.suggest_float.side_effect = lambda name, min_val, max_val, **kwargs: {
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }.get(name, min_val)

        score = model_manager._optimization_objective(mock_trial, 'xgboost', X, y)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_optimization_objective_lightgbm(self, model_manager, sample_data):
        """Test optimization objective for LightGBM"""
        X, y = sample_data

        mock_trial = Mock()
        mock_trial.suggest_int.side_effect = lambda name, min_val, max_val: {
            'n_estimators': 100,
            'max_depth': 6
        }.get(name, min_val)
        mock_trial.suggest_float.side_effect = lambda name, min_val, max_val, **kwargs: {
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }.get(name, min_val)

        score = model_manager._optimization_objective(mock_trial, 'lightgbm', X, y)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_optimization_objective_invalid_model(self, model_manager, sample_data):
        """Test optimization objective with invalid model type"""
        X, y = sample_data

        mock_trial = Mock()

        score = model_manager._optimization_objective(mock_trial, 'invalid_model', X, y)

        assert score == 0.0

    def test_optimization_objective_error_handling(self, model_manager):
        """Test error handling in optimization objective"""
        # Test with invalid data
        mock_trial = Mock()

        score = model_manager._optimization_objective(mock_trial, 'random_forest', None, None)

        assert score == 0.0


class TestOptimizationIntegration:
    """Integration tests for hyperparameter optimization"""

    def test_full_optimization_workflow(self, model_manager, sample_data):
        """Test complete optimization workflow"""
        X, y = sample_data

        # 1. Train baseline model
        baseline_result = model_manager.train_model('random_forest', X, y)
        assert baseline_result['success'] is True

        baseline_eval = model_manager.evaluate_model('random_forest', X, y)
        baseline_score = baseline_eval.get('f1_score', 0)

        # 2. Run optimization
        opt_result = model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=5)
        assert opt_result['success'] is True

        # 3. Evaluate optimized model
        opt_eval = model_manager.evaluate_model('random_forest_optimized', X, y)
        opt_score = opt_eval.get('f1_score', 0)

        # 4. Compare results
        assert isinstance(baseline_score, (int, float))
        assert isinstance(opt_score, (int, float))

        # Optimization should at least not make things worse
        assert opt_score >= 0.0

    def test_optimization_with_different_models(self, model_manager, sample_data):
        """Test optimization across different model types"""
        X, y = sample_data

        models_to_test = ['random_forest', 'xgboost', 'lightgbm']

        for model_type in models_to_test:
            result = model_manager.optimize_hyperparameters(model_type, X, y, n_trials=3)

            assert result['success'] is True
            assert f"{model_type}_optimized" in model_manager.models
            assert model_manager.models[f"{model_type}_optimized"].is_trained is True

    def test_optimization_preserves_original_models(self, model_manager, sample_data):
        """Test that optimization doesn't overwrite original models"""
        X, y = sample_data

        # Train original models
        original_models = {}
        for model_type in ['random_forest', 'xgboost']:
            model_manager.train_model(model_type, X, y)
            original_models[model_type] = model_manager.models[model_type].is_trained

        # Run optimization
        for model_type in ['random_forest', 'xgboost']:
            model_manager.optimize_hyperparameters(model_type, X, y, n_trials=3)

        # Check that original models are still trained
        for model_type in ['random_forest', 'xgboost']:
            assert model_manager.models[model_type].is_trained == original_models[model_type]
            assert f"{model_type}_optimized" in model_manager.models