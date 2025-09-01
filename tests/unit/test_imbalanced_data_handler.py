import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from collections import Counter

from src.services.imbalanced_data_handler import ImbalancedDataHandler
from src.utils.exceptions import ValidationError


class TestImbalancedDataHandler:
    """Test cases for ImbalancedDataHandler class"""

    @pytest.fixture
    def handler(self):
        """Create handler instance"""
        config = {
            'smote_config': {'random_state': 42},
            'synthetic_config': {'methods': ['vae'], 'sample_size_ratio': 0.5}
        }
        return ImbalancedDataHandler(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample imbalanced dataset"""
        np.random.seed(42)

        # Create imbalanced dataset
        n_samples = 1000
        n_features = 5

        # Majority class (80%)
        X_maj = np.random.normal(0, 1, (800, n_features))
        y_maj = np.ones(800)

        # Minority class (20%)
        X_min = np.random.normal(2, 1, (200, n_features))
        y_min = np.zeros(200)

        # Combine
        X = np.vstack([X_maj, X_min])
        y = np.hstack([y_maj, y_min])

        return X, y
        np.random.seed(42)

        # Create imbalanced dataset
        n_samples = 1000
        n_features = 5

        # Majority class (80%)
        X_maj = np.random.normal(0, 1, (800, n_features))
        y_maj = np.ones(800)

        # Minority class (20%)
        X_min = np.random.normal(2, 1, (200, n_features))
        y_min = np.zeros(200)

        # Combine
        X = np.vstack([X_maj, X_min])
        y = np.hstack([y_maj, y_min])

        return X, y

    def test_initialization(self, handler):
        """Test handler initialization"""
        assert handler.config is not None
        assert handler.smote_handler is not None
        assert handler.synthetic_generator is not None
        assert handler.balancing_history == []
        assert handler.performance_metrics == {}

    def test_handle_imbalanced_data_auto_strategy(self, handler, sample_data):
        """Test automatic strategy selection"""
        X, y = sample_data

        X_balanced, y_balanced = handler.handle_imbalanced_data(X, y, strategy='auto')

        assert X_balanced.shape[0] >= X.shape[0]
        assert y_balanced.shape[0] >= y.shape[0]
        assert X_balanced.shape[0] == y_balanced.shape[0]

    def test_handle_imbalanced_data_smote_strategy(self, handler, sample_data):
        """Test SMOTE strategy"""
        X, y = sample_data

        X_balanced, y_balanced = handler.handle_imbalanced_data(X, y, strategy='smote')

        assert X_balanced.shape[0] >= X.shape[0]
        assert y_balanced.shape[0] >= y.shape[0]

        # Check that balancing improved the ratio
        original_ratio = max(Counter(y).values()) / min(Counter(y).values())
        balanced_ratio = max(Counter(y_balanced).values()) / min(Counter(y_balanced).values())
        assert balanced_ratio <= original_ratio

    def test_handle_imbalanced_data_synthetic_strategy(self, handler, sample_data):
        """Test synthetic data strategy"""
        X, y = sample_data

        X_balanced, y_balanced = handler.handle_imbalanced_data(X, y, strategy='synthetic')

        assert X_balanced.shape[0] >= X.shape[0]
        assert y_balanced.shape[0] >= y.shape[0]

    def test_handle_imbalanced_data_hybrid_strategy(self, handler, sample_data):
        """Test hybrid strategy"""
        X, y = sample_data

        X_balanced, y_balanced = handler.handle_imbalanced_data(X, y, strategy='hybrid')

        assert X_balanced.shape[0] >= X.shape[0]
        assert y_balanced.shape[0] >= y.shape[0]

    def test_handle_balanced_data(self, handler):
        """Test handling of already balanced data"""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 5))
        y = np.array([0, 1] * 50)  # Perfectly balanced

        X_balanced, y_balanced = handler.handle_imbalanced_data(X, y, strategy='auto')

        # Should return original data unchanged
        assert np.array_equal(X_balanced, X)
        assert np.array_equal(y_balanced, y)

    def test_invalid_strategy(self, handler, sample_data):
        """Test invalid strategy handling"""
        X, y = sample_data

        with pytest.raises(ValidationError):
            handler.handle_imbalanced_data(X, y, strategy='invalid_strategy')

    def test_select_optimal_strategy_low_imbalance(self, handler):
        """Test strategy selection for low imbalance"""
        # Mock imbalance info
        handler.imbalance_info = {
            'severity': 'low',
            'imbalance_ratio': 1.5
        }

        strategy = handler._select_optimal_strategy(None, None)
        assert strategy in ['classic', 'smote']

    def test_select_optimal_strategy_high_imbalance(self, handler):
        """Test strategy selection for high imbalance"""
        # Mock imbalance info
        handler.imbalance_info = {
            'severity': 'high',
            'imbalance_ratio': 5.0
        }

        strategy = handler._select_optimal_strategy(None, None)
        assert strategy in ['adasyn', 'svm', 'kmeans', 'synthetic', 'hybrid']

    def test_get_balancing_recommendations(self, handler, sample_data):
        """Test balancing recommendations"""
        X, y = sample_data

        recommendations = handler.get_balancing_recommendations(X, y)

        assert 'imbalance_analysis' in recommendations
        assert 'recommended_strategies' in recommendations
        assert 'expected_outcomes' in recommendations
        assert 'implementation_notes' in recommendations

        assert len(recommendations['recommended_strategies']) > 0

    def test_evaluate_balancing_effectiveness(self, handler, sample_data):
        """Test balancing effectiveness evaluation"""
        X, y = sample_data

        # Create balanced data (simulate)
        X_balanced = np.vstack([X, X[:100]])  # Duplicate some samples
        y_balanced = np.hstack([y, y[:100]])

        evaluation = handler.evaluate_balancing_effectiveness(
            X, y, X, y, X_balanced, y_balanced
        )

        assert 'original_performance' in evaluation
        assert 'balanced_performance' in evaluation
        assert 'improvement_metrics' in evaluation
        assert 'confusion_matrices' in evaluation

    def test_get_balancing_history(self, handler, sample_data):
        """Test balancing history retrieval"""
        X, y = sample_data

        # Perform balancing operation
        handler.handle_imbalanced_data(X, y, strategy='smote')

        history = handler.get_balancing_history()
        assert len(history) > 0

        last_entry = history[-1]
        assert 'strategy' in last_entry
        assert 'timestamp' in last_entry
        assert 'original_samples' in last_entry
        assert 'balanced_samples' in last_entry

    def test_get_performance_metrics(self, handler, sample_data):
        """Test performance metrics calculation"""
        X, y = sample_data

        # Perform multiple balancing operations
        handler.handle_imbalanced_data(X, y, strategy='smote')
        handler.handle_imbalanced_data(X, y, strategy='hybrid')

        metrics = handler.get_performance_metrics()

        assert 'total_operations' in metrics
        assert 'average_processing_time' in metrics
        assert 'average_imbalance_reduction' in metrics
        assert 'strategy_usage' in metrics
        assert 'most_used_strategy' in metrics

        assert metrics['total_operations'] == 2

    def test_empty_data_handling(self, handler):
        """Test handling of empty data"""
        X = np.array([])
        y = np.array([])

        # Should handle gracefully
        X_balanced, y_balanced = handler.handle_imbalanced_data(X, y, strategy='smote')

        assert X_balanced.shape[0] == 0
        assert y_balanced.shape[0] == 0

    def test_single_sample_handling(self, handler):
        """Test handling of single sample data"""
        X = np.array([[1, 2, 3]])
        y = np.array([0])

        # Should handle gracefully
        X_balanced, y_balanced = handler.handle_imbalanced_data(X, y, strategy='smote')

        # May return original data if SMOTE cannot be applied
        assert X_balanced.shape[0] >= 1
        assert y_balanced.shape[0] >= 1

    @patch('src.services.imbalanced_data_handler.ImbalancedDataHandler._apply_smote_strategy')
    def test_smote_strategy_error_handling(self, mock_smote, handler, sample_data):
        """Test error handling in SMOTE strategy"""
        X, y = sample_data

        mock_smote.side_effect = Exception("SMOTE failed")

        # Should fallback gracefully
        X_balanced, y_balanced = handler.handle_imbalanced_data(X, y, strategy='smote')

        # Should return original data on failure
        assert X_balanced.shape[0] == X.shape[0]
        assert y_balanced.shape[0] == y.shape[0]

    def test_target_balance_ratio(self, handler, sample_data):
        """Test target balance ratio parameter"""
        X, y = sample_data

        X_balanced, y_balanced = handler.handle_imbalanced_data(
            X, y, strategy='smote', target_ratio=0.8
        )

        assert X_balanced.shape[0] >= X.shape[0]
        assert y_balanced.shape[0] >= y.shape[0]

    def test_save_load_state(self, handler, sample_data, tmp_path):
        """Test saving and loading handler state"""
        X, y = sample_data

        # Perform some operations
        handler.handle_imbalanced_data(X, y, strategy='smote')

        # Save state
        filepath = tmp_path / "handler_state.pkl"
        handler.save_balancing_state(str(filepath))

        # Create new handler and load state
        new_handler = ImbalancedDataHandler(handler.config)
        new_handler.load_balancing_state(str(filepath))

        # Check that state was loaded
        assert len(new_handler.balancing_history) == len(handler.balancing_history)
        assert new_handler.imbalance_info == handler.imbalance_info

    def test_performance_tracking(self, handler, sample_data):
        """Test performance tracking functionality"""
        X, y = sample_data

        initial_history_length = len(handler.balancing_history)

        handler.handle_imbalanced_data(X, y, strategy='smote')

        # Check that history was updated
        assert len(handler.balancing_history) == initial_history_length + 1

        # Check history entry structure
        entry = handler.balancing_history[-1]
        required_fields = ['strategy', 'timestamp', 'original_samples',
                          'balanced_samples', 'processing_time']

        for field in required_fields:
            assert field in entry