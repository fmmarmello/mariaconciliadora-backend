import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from collections import Counter

from src.services.smote_implementation import SMOTEImplementation
from src.utils.exceptions import ValidationError


class TestSMOTEImplementation:
    """Test cases for SMOTEImplementation class"""

    @pytest.fixture
    def smote_handler(self):
        """Create SMOTE handler instance"""
        config = {'random_state': 42}
        return SMOTEImplementation(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample imbalanced dataset"""
        np.random.seed(42)

        # Create imbalanced dataset
        n_samples = 1000
        n_features = 5

        # Majority class (70%)
        X_maj = np.random.normal(0, 1, (700, n_features))
        y_maj = np.ones(700)

        # Minority class (30%)
        X_min = np.random.normal(2, 1, (300, n_features))
        y_min = np.zeros(300)

        # Combine
        X = np.vstack([X_maj, X_min])
        y = np.hstack([y_maj, y_min])

        return X, y

    def test_initialization(self, smote_handler):
        """Test SMOTE handler initialization"""
        assert smote_handler.config is not None
        assert 'smote_variants' in dir(smote_handler)
        assert 'hybrid_methods' in dir(smote_handler)
        assert smote_handler.is_fitted == False

    def test_detect_imbalance(self, smote_handler, sample_data):
        """Test imbalance detection"""
        X, y = sample_data

        imbalance_info = smote_handler.detect_imbalance(X, y)

        assert 'imbalance_ratio' in imbalance_info
        assert 'severity' in imbalance_info
        assert 'requires_balancing' in imbalance_info
        assert imbalance_info['imbalance_ratio'] > 1.0
        assert imbalance_info['requires_balancing'] == True

    def test_detect_imbalance_balanced_data(self, smote_handler):
        """Test imbalance detection on balanced data"""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 5))
        y = np.array([0, 1] * 50)  # Perfectly balanced

        imbalance_info = smote_handler.detect_imbalance(X, y)

        assert imbalance_info['imbalance_ratio'] == 1.0
        assert imbalance_info['requires_balancing'] == False

    def test_apply_smote_classic(self, smote_handler, sample_data):
        """Test classic SMOTE application"""
        X, y = sample_data

        X_resampled, y_resampled = smote_handler.apply_smote(X, y, method='classic')

        assert X_resampled.shape[0] > X.shape[0]
        assert y_resampled.shape[0] > y.shape[0]
        assert X_resampled.shape[0] == y_resampled.shape[0]

        # Check that minority class was oversampled
        original_minority = Counter(y)[0]
        resampled_minority = Counter(y_resampled)[0]
        assert resampled_minority > original_minority

    def test_apply_smote_borderline(self, smote_handler, sample_data):
        """Test BorderlineSMOTE application"""
        X, y = sample_data

        X_resampled, y_resampled = smote_handler.apply_smote(X, y, method='borderline')

        assert X_resampled.shape[0] >= X.shape[0]
        assert y_resampled.shape[0] >= y.shape[0]

    def test_apply_smote_adasyn(self, smote_handler, sample_data):
        """Test ADASYN application"""
        X, y = sample_data

        X_resampled, y_resampled = smote_handler.apply_smote(X, y, method='adasyn')

        assert X_resampled.shape[0] >= X.shape[0]
        assert y_resampled.shape[0] >= y.shape[0]

    def test_apply_smote_auto_method(self, smote_handler, sample_data):
        """Test automatic method selection"""
        X, y = sample_data

        X_resampled, y_resampled = smote_handler.apply_smote(X, y, method='auto')

        assert X_resampled.shape[0] >= X.shape[0]
        assert y_resampled.shape[0] >= y.shape[0]

    def test_compare_smote_methods(self, smote_handler, sample_data):
        """Test SMOTE methods comparison"""
        X, y = sample_data

        results = smote_handler.compare_smote_methods(X, y, methods=['classic', 'borderline'])

        assert 'classic' in results
        assert 'borderline' in results
        assert results['classic']['success'] == True
        assert results['borderline']['success'] == True

    def test_invalid_method(self, smote_handler, sample_data):
        """Test invalid SMOTE method handling"""
        X, y = sample_data

        with pytest.raises(ValidationError):
            smote_handler.apply_smote(X, y, method='invalid_method')

    def test_get_method_info(self, smote_handler):
        """Test method information retrieval"""
        info = smote_handler.get_method_info()

        assert 'classic' in info
        assert 'borderline' in info
        assert 'adasyn' in info
        assert 'svm' in info
        assert 'kmeans' in info

        # Check structure
        classic_info = info['classic']
        assert 'name' in classic_info
        assert 'description' in classic_info
        assert 'best_for' in classic_info
        assert 'complexity' in classic_info

    def test_performance_tracking(self, smote_handler, sample_data):
        """Test performance tracking"""
        X, y = sample_data

        # Apply SMOTE
        smote_handler.apply_smote(X, y, method='classic')

        # Check performance history
        history = smote_handler.get_performance_history()
        assert len(history) > 0

        last_entry = history[-1]
        assert 'method' in last_entry
        assert 'timestamp' in last_entry
        assert 'original_samples' in last_entry
        assert 'resampled_samples' in last_entry
        assert last_entry['method'] == 'classic'

    def test_optimize_smote_parameters(self, smote_handler, sample_data):
        """Test SMOTE parameter optimization"""
        X, y = sample_data

        param_grid = {
            'k_neighbors': [3, 5],
            'sampling_strategy': ['auto', 'minority']
        }

        results = smote_handler.optimize_smote_parameters(X, y, method='classic', param_grid=param_grid)

        assert 'best_params' in results
        assert 'best_score' in results
        assert results['method'] == 'classic'

    @patch('src.services.smote_implementation.stats.ks_2samp')
    def test_kolmogorov_smirnov_test(self, mock_ks, smote_handler):
        """Test Kolmogorov-Smirnov test"""
        mock_ks.return_value = (0.5, 0.05)

        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(0.5, 1, 100)

        result = smote_handler._kolmogorov_smirnov_test(data1, data2)

        assert 'statistic' in result
        assert 'p_value' in result
        mock_ks.assert_called_once()

    @patch('src.services.smote_implementation.stats.mannwhitneyu')
    def test_mann_whitney_u_test(self, mock_mwu, smote_handler):
        """Test Mann-Whitney U test"""
        mock_mwu.return_value = (500, 0.05)

        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(0.5, 1, 100)

        result = smote_handler._mann_whitney_u_test(data1, data2)

        assert 'statistic' in result
        assert 'p_value' in result
        mock_mwu.assert_called_once()

    def test_empty_data_handling(self, smote_handler):
        """Test handling of empty data"""
        X = np.array([])
        y = np.array([])

        imbalance_info = smote_handler.detect_imbalance(X, y)
        assert imbalance_info == {}

    def test_single_class_data(self, smote_handler):
        """Test handling of single class data"""
        X = np.random.normal(0, 1, (100, 5))
        y = np.ones(100)  # All same class

        imbalance_info = smote_handler.detect_imbalance(X, y)
        assert imbalance_info['imbalance_ratio'] == 1.0
        assert imbalance_info['requires_balancing'] == False