import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.services.model_selector import (
    ModelSelector, ModelPerformanceMetrics, ModelComparisonResult, ABTestResult
)
from src.services.model_manager import ModelManager
from src.utils.exceptions import ValidationError


class TestModelPerformanceMetrics:
    """Test cases for ModelPerformanceMetrics data class"""

    def test_initialization(self):
        """Test ModelPerformanceMetrics initialization"""
        metrics = ModelPerformanceMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            auc_roc=0.91,
            training_time=45.2,
            prediction_time=1.5
        )

        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.82
        assert metrics.recall == 0.88
        assert metrics.f1_score == 0.85
        assert metrics.auc_roc == 0.91
        assert metrics.training_time == 45.2
        assert metrics.prediction_time == 1.5

    def test_to_dict(self):
        """Test conversion to dictionary"""
        metrics = ModelPerformanceMetrics(
            accuracy=0.85, precision=0.82, recall=0.88, f1_score=0.85
        )
        metrics_dict = metrics.to_dict()

        assert metrics_dict['accuracy'] == 0.85
        assert metrics_dict['precision'] == 0.82
        assert metrics_dict['recall'] == 0.88
        assert metrics_dict['f1_score'] == 0.85


class TestModelComparisonResult:
    """Test cases for ModelComparisonResult data class"""

    def test_initialization(self):
        """Test ModelComparisonResult initialization"""
        metrics = ModelPerformanceMetrics(accuracy=0.85, f1_score=0.82)
        result = ModelComparisonResult(
            model_name='random_forest',
            performance_metrics=metrics,
            rank=1,
            relative_score=0.85,
            recommendation_score=0.88,
            data_characteristics={'n_samples': 1000},
            timestamp='2024-01-01T10:00:00'
        )

        assert result.model_name == 'random_forest'
        assert result.rank == 1
        assert result.relative_score == 0.85
        assert result.recommendation_score == 0.88

    def test_to_dict(self):
        """Test conversion to dictionary"""
        metrics = ModelPerformanceMetrics(accuracy=0.85, f1_score=0.82)
        result = ModelComparisonResult(
            model_name='xgboost',
            performance_metrics=metrics,
            rank=2,
            relative_score=0.82,
            recommendation_score=0.85,
            data_characteristics={},
            timestamp='2024-01-01T10:00:00'
        )
        result_dict = result.to_dict()

        assert result_dict['model_name'] == 'xgboost'
        assert result_dict['rank'] == 2
        assert result_dict['performance_metrics']['accuracy'] == 0.85


class TestModelSelector:
    """Test cases for ModelSelector class"""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock ModelManager"""
        mock_mm = Mock(spec=ModelManager)
        mock_mm.models = {
            'random_forest': Mock(),
            'xgboost': Mock(),
            'lightgbm': Mock()
        }
        mock_mm.config = {
            'model_configs': {
                'random_forest': {},
                'xgboost': {},
                'lightgbm': {}
            }
        }
        return mock_mm

    @pytest.fixture
    def model_selector(self, mock_model_manager):
        """Create ModelSelector instance"""
        return ModelSelector(mock_model_manager)

    def test_initialization(self, model_selector):
        """Test ModelSelector initialization"""
        assert model_selector.model_manager is not None
        assert isinstance(model_selector.config, dict)
        assert 'model_selection_criteria' in model_selector.config
        assert 'performance_history_file' in model_selector.config

    def test_analyze_data_characteristics(self, model_selector):
        """Test data characteristics analysis"""
        X = np.random.rand(100, 5)
        y = np.random.choice(['class_a', 'class_b'], 100)

        characteristics = model_selector.analyze_data_characteristics(X, y)

        assert characteristics['n_samples'] == 100
        assert characteristics['n_features'] == 5
        assert characteristics['n_classes'] == 2
        assert 'class_distribution' in characteristics
        assert 'data_complexity' in characteristics
        assert 'timestamp' in characteristics

    def test_analyze_data_characteristics_with_feature_names(self, model_selector):
        """Test data characteristics analysis with feature names"""
        X = np.random.rand(50, 3)
        y = np.random.choice(['A', 'B', 'C'], 50)
        feature_names = ['feature_1', 'feature_2', 'feature_3']

        characteristics = model_selector.analyze_data_characteristics(X, y, feature_names)

        assert characteristics['n_samples'] == 50
        assert characteristics['n_features'] == 3
        assert characteristics['n_classes'] == 3
        assert characteristics['feature_names'] == feature_names

    def test_recommend_models_based_on_data_small_dataset(self, model_selector):
        """Test model recommendations for small dataset"""
        data_characteristics = {
            'n_samples': 500,
            'n_features': 10,
            'n_classes': 2,
            'data_complexity': {'class_imbalance_ratio': 1.2}
        }

        recommendations = model_selector._recommend_models_based_on_data(data_characteristics)

        # Should prefer simpler models for small datasets
        assert 'random_forest' in recommendations
        assert 'xgboost' in recommendations

    def test_recommend_models_based_on_data_large_features(self, model_selector):
        """Test model recommendations for high-dimensional data"""
        data_characteristics = {
            'n_samples': 2000,
            'n_features': 100,
            'n_classes': 2,
            'data_complexity': {'class_imbalance_ratio': 1.0}
        }

        recommendations = model_selector._recommend_models_based_on_data(data_characteristics)

        # Should include LightGBM for high-dimensional data
        assert 'lightgbm' in recommendations

    def test_recommend_models_based_on_data_imbalanced(self, model_selector):
        """Test model recommendations for imbalanced data"""
        data_characteristics = {
            'n_samples': 2000,
            'n_features': 20,
            'n_classes': 2,
            'data_complexity': {'class_imbalance_ratio': 5.0}
        }

        recommendations = model_selector._recommend_models_based_on_data(data_characteristics)

        # Should prioritize XGBoost for imbalanced data
        assert recommendations[0] == 'xgboost'

    @patch('src.services.model_selector.ModelSelector._perform_cross_validation')
    def test_compare_models_comprehensive(self, mock_cv, model_selector, mock_model_manager):
        """Test comprehensive model comparison"""
        # Mock model training
        mock_model_manager.train_model.return_value = {'success': True}
        mock_model_manager.predict.return_value = np.array(['A'] * 80 + ['B'] * 20)
        mock_model_manager.predict_proba.return_value = np.random.rand(100, 2)

        # Mock cross-validation
        mock_cv.return_value = [0.85, 0.82, 0.88, 0.86, 0.84]

        X = np.random.rand(100, 5)
        y = np.random.choice(['A', 'B'], 100)
        models_to_compare = ['random_forest', 'xgboost']

        results = model_selector.compare_models_comprehensive(X, y, models_to_compare)

        assert len(results) == 2
        assert all(isinstance(r, ModelComparisonResult) for r in results)
        assert all(r.performance_metrics.f1_score > 0 for r in results)
        assert results[0].rank == 1  # Best model should be rank 1

    def test_calculate_recommendation_score(self, model_selector):
        """Test recommendation score calculation"""
        metrics = ModelPerformanceMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            cross_val_std=0.03,
            training_time=30.0
        )

        data_characteristics = {
            'n_samples': 1000,
            'n_features': 20,
            'data_complexity': {'class_imbalance_ratio': 1.5}
        }

        score = model_selector._calculate_recommendation_score(metrics, data_characteristics)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_select_best_from_comparison(self, model_selector):
        """Test best model selection from comparison results"""
        metrics1 = ModelPerformanceMetrics(accuracy=0.85, f1_score=0.82)
        metrics2 = ModelPerformanceMetrics(accuracy=0.88, f1_score=0.85)

        result1 = ModelComparisonResult(
            model_name='model_a', performance_metrics=metrics1,
            rank=2, relative_score=0.82, recommendation_score=0.80,
            data_characteristics={}, timestamp=''
        )
        result2 = ModelComparisonResult(
            model_name='model_b', performance_metrics=metrics2,
            rank=1, relative_score=0.85, recommendation_score=0.85,
            data_characteristics={}, timestamp=''
        )

        comparison_results = [result1, result2]
        best_result = model_selector._select_best_from_comparison(comparison_results)

        assert best_result.model_name == 'model_b'
        assert best_result.recommendation_score == 0.85

    def test_generate_model_recommendation(self, model_selector):
        """Test model recommendation generation"""
        metrics = ModelPerformanceMetrics(
            accuracy=0.88, precision=0.85, recall=0.90, f1_score=0.87,
            cross_val_std=0.02, training_time=25.0
        )

        best_result = ModelComparisonResult(
            model_name='xgboost', performance_metrics=metrics,
            rank=1, relative_score=0.87, recommendation_score=0.89,
            data_characteristics={'n_samples': 1000}, timestamp=''
        )

        all_results = [best_result]
        data_characteristics = {'n_samples': 1000, 'n_features': 20}

        recommendation = model_selector._generate_model_recommendation(
            best_result, all_results, data_characteristics
        )

        assert recommendation['recommended_model'] == 'xgboost'
        assert 'confidence_level' in recommendation
        assert 'performance_score' in recommendation
        assert 'key_advantages' in recommendation

    @patch('src.services.model_selector.train_test_split')
    def test_perform_ab_test(self, mock_train_test_split, model_selector, mock_model_manager):
        """Test A/B testing functionality"""
        # Mock train_test_split
        X_a, X_b = np.random.rand(50, 5), np.random.rand(50, 5)
        y_a, y_b = np.random.choice(['A', 'B'], 50), np.random.choice(['A', 'B'], 50)
        mock_train_test_split.return_value = (X_a, X_b, y_a, y_b)

        # Mock model evaluation
        mock_model_manager.evaluate_model.side_effect = [
            {'f1_score': 0.85, 'accuracy': 0.82},  # Model A results
            {'f1_score': 0.88, 'accuracy': 0.85}   # Model B results
        ]

        X = np.random.rand(100, 5)
        y = np.random.choice(['A', 'B'], 100)

        ab_result = model_selector.perform_ab_test('random_forest', 'xgboost', X, y)

        assert 'winner' in ab_result
        assert 'confidence_level' in ab_result
        assert 'performance_difference' in ab_result
        assert ab_result['winner'] == 'xgboost'  # Should win based on mocked results

    def test_get_performance_history(self, model_selector):
        """Test performance history retrieval"""
        # Add some mock history
        mock_history = {
            'selection_1': {
                'best_model': 'xgboost',
                'timestamp': (datetime.now() - timedelta(days=5)).isoformat(),
                'comparison_results': []
            },
            'selection_2': {
                'best_model': 'random_forest',
                'timestamp': (datetime.now() - timedelta(days=15)).isoformat(),
                'comparison_results': []
            }
        }
        model_selector.performance_history = mock_history

        # Test getting all history
        history = model_selector.get_performance_history()
        assert 'total_entries' in history
        assert history['total_entries'] == 2

        # Test getting history for specific model
        rf_history = model_selector.get_performance_history('random_forest')
        assert rf_history['model'] == 'random_forest'
        assert len(rf_history['history']) == 1

        # Test getting recent history
        recent_history = model_selector.get_performance_history(days=7)
        assert recent_history['total_entries'] == 1  # Only the 5-day old entry

    def test_analyze_model_stability(self, model_selector):
        """Test model stability analysis"""
        # Create mock results with different CV std values
        metrics1 = ModelPerformanceMetrics(
            accuracy=0.85, f1_score=0.82, cross_val_std=0.02, cross_val_mean=0.84
        )
        metrics2 = ModelPerformanceMetrics(
            accuracy=0.83, f1_score=0.80, cross_val_std=0.08, cross_val_mean=0.82
        )

        result1 = ModelComparisonResult(
            model_name='stable_model', performance_metrics=metrics1,
            rank=1, relative_score=0.84, recommendation_score=0.86,
            data_characteristics={}, timestamp=''
        )
        result2 = ModelComparisonResult(
            model_name='unstable_model', performance_metrics=metrics2,
            rank=2, relative_score=0.82, recommendation_score=0.78,
            data_characteristics={}, timestamp=''
        )

        comparison_results = [result1, result2]
        stability = model_selector._analyze_model_stability(comparison_results)

        assert stability['stable_model']['stability_level'] == 'high'
        assert stability['unstable_model']['stability_level'] == 'medium'

    def test_analyze_model_efficiency(self, model_selector):
        """Test model efficiency analysis"""
        # Create mock results with different training times
        metrics1 = ModelPerformanceMetrics(
            accuracy=0.85, f1_score=0.82, training_time=15.0
        )
        metrics2 = ModelPerformanceMetrics(
            accuracy=0.83, f1_score=0.80, training_time=45.0
        )

        result1 = ModelComparisonResult(
            model_name='fast_model', performance_metrics=metrics1,
            rank=1, relative_score=0.84, recommendation_score=0.86,
            data_characteristics={}, timestamp=''
        )
        result2 = ModelComparisonResult(
            model_name='slow_model', performance_metrics=metrics2,
            rank=2, relative_score=0.82, recommendation_score=0.78,
            data_characteristics={}, timestamp=''
        )

        comparison_results = [result1, result2]
        efficiency = model_selector._analyze_model_efficiency(comparison_results)

        assert efficiency['fast_model']['efficiency_level'] == 'high'
        assert efficiency['slow_model']['efficiency_level'] == 'medium'

    def test_generate_performance_report(self, model_selector):
        """Test performance report generation"""
        # Create mock comparison results
        metrics = ModelPerformanceMetrics(
            accuracy=0.85, precision=0.82, recall=0.88, f1_score=0.85,
            cross_val_mean=0.84, cross_val_std=0.03, training_time=25.0
        )

        result = ModelComparisonResult(
            model_name='test_model', performance_metrics=metrics,
            rank=1, relative_score=0.85, recommendation_score=0.87,
            data_characteristics={'n_samples': 1000, 'n_features': 20}, timestamp=''
        )

        data_characteristics = {'n_samples': 1000, 'n_features': 20}
        report = model_selector.generate_performance_report([result], data_characteristics)

        assert report['summary']['best_model'] == 'test_model'
        assert report['summary']['total_models_compared'] == 1
        assert 'model_rankings' in report
        assert 'performance_analysis' in report
        assert 'recommendations' in report

    @patch('builtins.open')
    @patch('os.path.exists')
    def test_export_comparison_report(self, mock_exists, mock_open, model_selector):
        """Test comparison report export"""
        mock_exists.return_value = True

        # Create mock file handle
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Create mock comparison result
        metrics = ModelPerformanceMetrics(accuracy=0.85, f1_score=0.82)
        result = ModelComparisonResult(
            model_name='test_model', performance_metrics=metrics,
            rank=1, relative_score=0.85, recommendation_score=0.87,
            data_characteristics={}, timestamp=''
        )

        success = model_selector.export_comparison_report([result], 'test_report.json')

        assert success is True
        mock_open.assert_called_once()
        mock_file.write.assert_called_once()

    def test_select_best_model_error_handling(self, model_selector):
        """Test error handling in model selection"""
        X = np.random.rand(10, 2)  # Very small dataset
        y = np.random.choice(['A', 'B'], 10)

        # This should handle the error gracefully
        result = model_selector.select_best_model(X, y)

        # Should return some result even with small dataset
        assert isinstance(result, dict)
        assert 'best_model' in result or 'error' in result

    def test_ab_test_error_handling(self, model_selector):
        """Test error handling in A/B testing"""
        X = np.array([])  # Empty array
        y = np.array([])

        result = model_selector.perform_ab_test('model_a', 'model_b', X, y)

        assert 'error' in result

    def test_persistent_data_operations(self, model_selector):
        """Test persistent data save/load operations"""
        # Add some test data
        model_selector.performance_history = {'test': 'data'}
        model_selector.ab_tests = {'test_ab': 'ab_data'}

        # Mock the save operation
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            model_selector._save_persistent_data()

            # Should have called open for each file
            assert mock_open.call_count == 3  # performance_history, ab_tests, recommendations


class TestABTestResult:
    """Test cases for ABTestResult data class"""

    def test_initialization(self):
        """Test ABTestResult initialization"""
        ab_result = ABTestResult(
            test_id='test_123',
            model_a='random_forest',
            model_b='xgboost',
            winner='xgboost',
            confidence_level=0.85,
            statistical_significance=True,
            performance_difference={'f1_diff': 0.05},
            sample_size=1000,
            test_duration='24 hours',
            timestamp='2024-01-01T10:00:00'
        )

        assert ab_result.test_id == 'test_123'
        assert ab_result.model_a == 'random_forest'
        assert ab_result.model_b == 'xgboost'
        assert ab_result.winner == 'xgboost'
        assert ab_result.confidence_level == 0.85
        assert ab_result.statistical_significance is True

    def test_to_dict(self):
        """Test conversion to dictionary"""
        ab_result = ABTestResult(
            test_id='test_456',
            model_a='lightgbm',
            model_b='bert',
            winner='bert',
            confidence_level=0.92,
            statistical_significance=True,
            performance_difference={'accuracy_diff': 0.03},
            sample_size=500,
            test_duration='12 hours',
            timestamp='2024-01-01T10:00:00'
        )

        result_dict = ab_result.to_dict()

        assert result_dict['test_id'] == 'test_456'
        assert result_dict['winner'] == 'bert'
        assert result_dict['confidence_level'] == 0.92


if __name__ == '__main__':
    pytest.main([__file__])