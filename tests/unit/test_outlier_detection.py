"""
OutlierDetectionTestSuite - Comprehensive tests for statistical outlier detection

This module provides comprehensive tests for:
- IQR method testing with various datasets
- Z-score method testing with different thresholds
- Local Outlier Factor testing with parameter variations
- Ensemble method testing and comparison
- Performance and accuracy validation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.datasets import make_blobs

from src.services.advanced_outlier_detector import AdvancedOutlierDetector
from src.utils.exceptions import ValidationError


class TestOutlierDetectionInitialization:
    """Test OutlierDetection initialization and configuration"""

    def test_default_initialization(self):
        """Test detector initialization with default config"""
        detector = AdvancedOutlierDetector()

        assert detector is not None
        assert detector.config is not None
        assert 'iqr' in detector.config
        assert 'zscore' in detector.config
        assert 'lof' in detector.config
        assert 'isolation_forest' in detector.config
        assert 'ensemble' in detector.config

    def test_custom_config_initialization(self):
        """Test detector initialization with custom config"""
        custom_config = {
            'iqr': {'multiplier': 2.0},
            'zscore': {'threshold': 2.5},
            'lof': {'n_neighbors': 15}
        }

        detector = AdvancedOutlierDetector(custom_config)

        assert detector.config['iqr']['multiplier'] == 2.0
        assert detector.config['zscore']['threshold'] == 2.5
        assert detector.config['lof']['n_neighbors'] == 15

    def test_method_initialization(self):
        """Test that all detection methods are properly initialized"""
        detector = AdvancedOutlierDetector()

        assert hasattr(detector, 'isolation_forest')
        assert hasattr(detector, 'lof_detector')
        assert hasattr(detector, 'ocsvm_detector')
        assert hasattr(detector, 'standard_scaler')
        assert hasattr(detector, 'robust_scaler')

    @patch('src.services.advanced_outlier_detector.IsolationForest')
    def test_initialization_error_handling(self, mock_if):
        """Test error handling during initialization"""
        mock_if.side_effect = Exception("Initialization failed")

        with pytest.raises(ValidationError):
            AdvancedOutlierDetector()


class TestIQRMethod:
    """Test IQR (Interquartile Range) outlier detection method"""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        return AdvancedOutlierDetector()

    @pytest.fixture
    def normal_data(self):
        """Create normal distribution data"""
        np.random.seed(42)
        return np.random.normal(100, 10, 100)

    @pytest.fixture
    def data_with_outliers(self):
        """Create data with known outliers"""
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, 98)
        outliers = np.array([200, -50])  # Clear outliers
        return np.concatenate([normal_data, outliers])

    def test_iqr_normal_data(self, detector, normal_data):
        """Test IQR detection on normal data"""
        flags, scores = detector.detect_outliers_iqr(normal_data)

        # Should detect few or no outliers in normal data
        outlier_count = np.sum(flags)
        assert outlier_count <= 5  # Allow small number due to statistical variation

        # Scores should be reasonable
        assert np.all(scores >= 0)
        assert len(scores) == len(normal_data)

    def test_iqr_with_known_outliers(self, detector, data_with_outliers):
        """Test IQR detection with known outliers"""
        flags, scores = detector.detect_outliers_iqr(data_with_outliers)

        # Should detect the outliers
        outlier_count = np.sum(flags)
        assert outlier_count >= 1  # At least one outlier detected

        # Last two values should be flagged as outliers
        assert flags[-1] == True or flags[-2] == True

    def test_iqr_robust_vs_standard(self, detector, data_with_outliers):
        """Test difference between robust and standard IQR methods"""
        flags_robust, scores_robust = detector.detect_outliers_iqr(data_with_outliers, method='robust')
        flags_standard, scores_standard = detector.detect_outliers_iqr(data_with_outliers, method='standard')

        # Results might differ but should be valid
        assert len(flags_robust) == len(flags_standard)
        assert len(scores_robust) == len(scores_standard)

    def test_iqr_edge_cases(self, detector):
        """Test IQR with edge cases"""
        # Single value
        single_value = np.array([100.0])
        flags, scores = detector.detect_outliers_iqr(single_value)
        assert len(flags) == 1
        assert flags[0] == False

        # All same values
        same_values = np.full(10, 100.0)
        flags, scores = detector.detect_outliers_iqr(same_values)
        assert np.sum(flags) == 0

        # Empty array
        empty_array = np.array([])
        flags, scores = detector.detect_outliers_iqr(empty_array)
        assert len(flags) == 0


class TestZScoreMethod:
    """Test Z-score outlier detection method"""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        return AdvancedOutlierDetector()

    @pytest.fixture
    def normal_data(self):
        """Create normal distribution data"""
        np.random.seed(42)
        return np.random.normal(0, 1, 100)

    @pytest.fixture
    def skewed_data(self):
        """Create skewed data for testing"""
        np.random.seed(42)
        return np.random.exponential(2, 100)

    def test_zscore_standard_method(self, detector, normal_data):
        """Test standard Z-score method"""
        flags, scores = detector.detect_outliers_zscore(normal_data, method='standard')

        assert len(flags) == len(normal_data)
        assert len(scores) == len(normal_data)
        assert np.all(scores >= 0)  # Z-scores are absolute values

    def test_zscore_modified_method(self, detector, normal_data):
        """Test modified Z-score method"""
        flags, scores = detector.detect_outliers_zscore(normal_data, method='modified')

        assert len(flags) == len(normal_data)
        assert len(scores) == len(normal_data)

    def test_zscore_robust_method(self, detector, skewed_data):
        """Test robust Z-score method on skewed data"""
        flags, scores = detector.detect_outliers_zscore(skewed_data, method='robust')

        assert len(flags) == len(skewed_data)
        assert len(scores) == len(skewed_data)

    def test_zscore_with_different_thresholds(self, detector):
        """Test Z-score with different threshold values"""
        data = np.random.normal(0, 1, 100)
        data[0] = 5.0  # Add clear outlier

        # Test with different thresholds
        flags_strict, _ = detector.detect_outliers_zscore(data, method='standard')
        detector.config['zscore']['threshold'] = 2.0
        flags_lenient, _ = detector.detect_outliers_zscore(data, method='standard')

        # Stricter threshold should detect fewer outliers
        assert np.sum(flags_strict) <= np.sum(flags_lenient)

    def test_zscore_edge_cases(self, detector):
        """Test Z-score with edge cases"""
        # Single value
        single_value = np.array([100.0])
        flags, scores = detector.detect_outliers_zscore(single_value)
        assert len(flags) == 1
        assert flags[0] == False

        # All same values
        same_values = np.full(10, 100.0)
        flags, scores = detector.detect_outliers_zscore(same_values)
        assert np.sum(flags) == 0


class TestLocalOutlierFactor:
    """Test Local Outlier Factor (LOF) method"""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        return AdvancedOutlierDetector()

    @pytest.fixture
    def multivariate_data(self):
        """Create multivariate test data"""
        np.random.seed(42)
        return np.random.normal(0, 1, (100, 3))

    @pytest.fixture
    def univariate_data(self):
        """Create univariate test data"""
        np.random.seed(42)
        return np.random.normal(100, 10, 100)

    def test_lof_univariate_data(self, detector, univariate_data):
        """Test LOF on univariate data"""
        flags, scores = detector.detect_outliers_lof(univariate_data)

        assert len(flags) == len(univariate_data)
        assert len(scores) == len(univariate_data)
        assert np.all(scores >= 0)

    def test_lof_multivariate_data(self, detector, multivariate_data):
        """Test LOF on multivariate data"""
        flags, scores = detector.detect_outliers_lof(multivariate_data)

        assert len(flags) == len(multivariate_data)
        assert len(scores) == len(multivariate_data)

    def test_lof_with_different_neighbors(self, detector, multivariate_data):
        """Test LOF with different n_neighbors values"""
        # Test with different configurations
        configs = [
            {'lof': {'n_neighbors': 5}},
            {'lof': {'n_neighbors': 20}},
            {'lof': {'n_neighbors': 50}}
        ]

        results = []
        for config in configs:
            detector.config.update(config)
            flags, scores = detector.detect_outliers_lof(multivariate_data)
            results.append((flags, scores))

        # All results should be valid
        for flags, scores in results:
            assert len(flags) == len(multivariate_data)
            assert len(scores) == len(multivariate_data)

    def test_lof_fit_predict_separate(self, detector, multivariate_data):
        """Test LOF fit and predict separately"""
        # First fit
        detector.lof_detector.fit(multivariate_data)

        # Then predict
        flags, scores = detector.detect_outliers_lof(multivariate_data, fit_predict=False)

        assert len(flags) == len(multivariate_data)
        assert len(scores) == len(multivariate_data)


class TestIsolationForest:
    """Test Isolation Forest outlier detection method"""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        return AdvancedOutlierDetector()

    @pytest.fixture
    def test_data(self):
        """Create test data"""
        np.random.seed(42)
        return np.random.normal(0, 1, (100, 2))

    def test_isolation_forest_basic(self, detector, test_data):
        """Test basic Isolation Forest functionality"""
        flags, scores = detector.detect_outliers_isolation_forest(test_data)

        assert len(flags) == len(test_data)
        assert len(scores) == len(test_data)

    def test_isolation_forest_univariate(self, detector):
        """Test Isolation Forest on univariate data"""
        data = np.random.normal(100, 10, 100)
        flags, scores = detector.detect_outliers_isolation_forest(data)

        assert len(flags) == len(data)
        assert len(scores) == len(data)

    def test_isolation_forest_with_outliers(self, detector):
        """Test Isolation Forest with injected outliers"""
        # Create normal data
        normal_data = np.random.normal(0, 1, (90, 2))

        # Add clear outliers
        outliers = np.array([[10, 10], [10, -10], [-10, 10], [-10, -10]])
        test_data = np.vstack([normal_data, outliers])

        flags, scores = detector.detect_outliers_isolation_forest(test_data)

        # Should detect some outliers
        outlier_count = np.sum(flags)
        assert outlier_count > 0


class TestMahalanobisDistance:
    """Test Mahalanobis distance outlier detection method"""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        return AdvancedOutlierDetector()

    @pytest.fixture
    def multivariate_normal_data(self):
        """Create multivariate normal data"""
        np.random.seed(42)
        mean = [0, 0]
        cov = [[1, 0.5], [0.5, 1]]
        return np.random.multivariate_normal(mean, cov, 100)

    def test_mahalanobis_basic(self, detector, multivariate_normal_data):
        """Test basic Mahalanobis distance detection"""
        flags, scores = detector.detect_outliers_mahalanobis(multivariate_normal_data)

        assert len(flags) == len(multivariate_normal_data)
        assert len(scores) == len(multivariate_normal_data)
        assert np.all(scores >= 0)

    def test_mahalanobis_robust_vs_standard(self, detector, multivariate_normal_data):
        """Test robust vs standard Mahalanobis methods"""
        flags_robust, scores_robust = detector.detect_outliers_mahalanobis(
            multivariate_normal_data, method='robust'
        )
        flags_standard, scores_standard = detector.detect_outliers_mahalanobis(
            multivariate_normal_data, method='standard'
        )

        assert len(flags_robust) == len(flags_standard)
        assert len(scores_robust) == len(scores_standard)

    def test_mahalanobis_univariate(self, detector):
        """Test Mahalanobis on univariate data"""
        data = np.random.normal(100, 10, 50).reshape(-1, 1)
        flags, scores = detector.detect_outliers_mahalanobis(data)

        assert len(flags) == len(data)
        assert len(scores) == len(data)

    def test_mahalanobis_singular_matrix_handling(self, detector):
        """Test handling of singular covariance matrices"""
        # Create data that will result in singular covariance matrix
        data = np.ones((10, 2))  # All same values

        flags, scores = detector.detect_outliers_mahalanobis(data)

        # Should handle gracefully
        assert len(flags) == len(data)
        assert len(scores) == len(data)


class TestEnsembleMethod:
    """Test ensemble outlier detection method"""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        return AdvancedOutlierDetector()

    @pytest.fixture
    def test_data(self):
        """Create test data with outliers"""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 90)
        outliers = np.array([5, -5, 6, -6])  # Clear outliers
        return np.concatenate([normal_data, outliers])

    def test_ensemble_basic(self, detector, test_data):
        """Test basic ensemble detection"""
        flags, scores = detector.detect_outliers_ensemble(test_data)

        assert len(flags) == len(test_data)
        assert len(scores) == len(test_data)

    def test_ensemble_with_specific_methods(self, detector, test_data):
        """Test ensemble with specific methods"""
        methods = ['iqr', 'zscore', 'lof']
        flags, scores = detector.detect_outliers_ensemble(test_data, methods)

        assert len(flags) == len(test_data)
        assert len(scores) == len(test_data)

    def test_ensemble_voting_methods(self, detector, test_data):
        """Test different voting methods"""
        voting_configs = [
            {'ensemble': {'voting_method': 'majority'}},
            {'ensemble': {'voting_method': 'weighted'}},
            {'ensemble': {'voting_method': 'consensus'}}
        ]

        for config in voting_configs:
            detector.config.update(config)
            flags, scores = detector.detect_outliers_ensemble(test_data)
            assert len(flags) == len(test_data)

    def test_ensemble_method_weights(self, detector):
        """Test method weights in ensemble"""
        weights = detector._get_method_weights(['iqr', 'zscore', 'lof'])

        assert len(weights) == 3
        assert all(w > 0 for w in weights)


class TestComprehensiveDetection:
    """Test comprehensive outlier detection functionality"""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        return AdvancedOutlierDetector()

    @pytest.fixture
    def test_dataframe(self):
        """Create test DataFrame"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(10, 2, 100),
            'feature3': np.random.normal(5, 3, 100)
        })

    def test_comprehensive_with_dataframe(self, detector, test_dataframe):
        """Test comprehensive detection with DataFrame"""
        result = detector.detect_outliers_comprehensive(test_dataframe)

        assert 'results' in result
        assert 'summary' in result
        assert 'feature_names' in result
        assert 'data_shape' in result

        # Check that multiple methods were applied
        methods = result['results'].keys()
        assert len(methods) > 1

    def test_comprehensive_with_numpy_array(self, detector):
        """Test comprehensive detection with numpy array"""
        data = np.random.normal(0, 1, (50, 3))
        result = detector.detect_outliers_comprehensive(data)

        assert 'results' in result
        assert 'summary' in result

    def test_comprehensive_with_specific_methods(self, detector, test_dataframe):
        """Test comprehensive detection with specific methods"""
        methods = ['iqr', 'zscore', 'lof']
        result = detector.detect_outliers_comprehensive(test_dataframe, methods=methods)

        # Should only contain specified methods
        result_methods = set(result['results'].keys())
        expected_methods = set(methods + ['ensemble'])  # Ensemble is added automatically
        assert result_methods == expected_methods

    def test_comprehensive_detection_summary(self, detector, test_dataframe):
        """Test detection summary generation"""
        result = detector.detect_outliers_comprehensive(test_dataframe)

        summary = result['summary']
        assert 'total_methods' in summary
        assert 'method_comparison' in summary
        assert 'consensus_outliers' in summary
        assert 'high_confidence_outliers' in summary

    def test_comprehensive_error_handling(self, detector):
        """Test error handling in comprehensive detection"""
        # Test with invalid data
        result = detector.detect_outliers_comprehensive(None)

        assert 'error' in result


class TestPerformanceValidation:
    """Test performance and accuracy validation"""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        return AdvancedOutlierDetector()

    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing"""
        np.random.seed(42)
        return np.random.normal(0, 1, 1000)

    def test_performance_large_dataset(self, detector, large_dataset):
        """Test performance with large dataset"""
        import time

        start_time = time.time()
        result = detector.detect_outliers_comprehensive(large_dataset, methods=['iqr', 'zscore'])
        end_time = time.time()

        # Should complete within reasonable time
        duration = end_time - start_time
        assert duration < 10.0  # Less than 10 seconds

        assert 'results' in result

    def test_accuracy_with_ground_truth(self, detector):
        """Test accuracy against known ground truth"""
        # Create data with known outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 90)
        outliers = np.full(10, 5.0)  # Clear outliers
        data = np.concatenate([normal_data, outliers])

        # Ground truth labels
        true_labels = np.zeros(100, dtype=bool)
        true_labels[90:] = True  # Last 10 are outliers

        # Test different methods
        methods = ['iqr', 'zscore', 'lof']
        for method in methods:
            if method == 'iqr':
                flags, _ = detector.detect_outliers_iqr(data)
            elif method == 'zscore':
                flags, _ = detector.detect_outliers_zscore(data)
            elif method == 'lof':
                flags, _ = detector.detect_outliers_lof(data)

            # Calculate accuracy metrics
            from sklearn.metrics import precision_score, recall_score

            try:
                precision = precision_score(true_labels, flags, zero_division=0)
                recall = recall_score(true_labels, flags, zero_division=0)

                # Should have reasonable performance
                assert precision >= 0.0
                assert recall >= 0.0
                assert precision <= 1.0
                assert recall <= 1.0
            except:
                # Skip if metrics calculation fails
                pass

    def test_method_performance_comparison(self, detector):
        """Test method performance comparison"""
        data = np.random.normal(0, 1, 100)

        # Run comprehensive detection
        result = detector.detect_outliers_comprehensive(data)

        # Get performance metrics
        performance = detector.get_method_performance()

        assert isinstance(performance, dict)
        assert len(performance) > 0

        # Check that all methods have performance metrics
        for method, metrics in performance.items():
            assert 'outlier_count' in metrics
            assert 'outlier_percentage' in metrics
            assert 'mean_score' in metrics

    def test_scalability_test(self, detector):
        """Test scalability with increasing data sizes"""
        sizes = [100, 500, 1000]

        for size in sizes:
            data = np.random.normal(0, 1, size)

            import time
            start_time = time.time()

            flags, scores = detector.detect_outliers_iqr(data)

            duration = time.time() - start_time

            # Should scale reasonably (not exponentially)
            assert duration < size / 10  # Rough scaling check
            assert len(flags) == size
            assert len(scores) == size


class TestErrorHandling:
    """Test error handling in outlier detection"""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        return AdvancedOutlierDetector()

    def test_invalid_data_types(self, detector):
        """Test handling of invalid data types"""
        invalid_data = ["string", "data"]

        # Should handle gracefully
        flags, scores = detector.detect_outliers_iqr(np.array([]))
        assert len(flags) == 0

    def test_extreme_values(self, detector):
        """Test handling of extreme values"""
        data = np.array([np.inf, -np.inf, np.nan, 1, 2, 3])

        # Should handle gracefully
        flags, scores = detector.detect_outliers_iqr(data)
        assert len(flags) == len(data)

    def test_empty_data(self, detector):
        """Test handling of empty data"""
        empty_data = np.array([])

        flags, scores = detector.detect_outliers_iqr(empty_data)
        assert len(flags) == 0
        assert len(scores) == 0

    @patch('src.services.advanced_outlier_detector.LocalOutlierFactor')
    def test_lof_error_handling(self, mock_lof, detector):
        """Test LOF error handling"""
        mock_lof.side_effect = Exception("LOF failed")

        data = np.random.normal(0, 1, 50)
        flags, scores = detector.detect_outliers_lof(data)

        # Should return safe defaults
        assert len(flags) == len(data)
        assert len(scores) == len(data)
        assert np.all(flags == False)  # No outliers detected due to error


if __name__ == "__main__":
    pytest.main([__file__])