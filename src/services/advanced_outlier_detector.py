"""
Advanced Outlier Detection System for Maria Conciliadora

This module provides comprehensive statistical outlier detection methods
for financial transaction data, including multiple detection algorithms
and contextual analysis capabilities.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import MinCovDet
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta
import logging

from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError

logger = get_logger(__name__)


class AdvancedOutlierDetector:
    """
    Multi-method outlier detection framework with various statistical approaches
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced outlier detector

        Args:
            config: Configuration dictionary for detection parameters
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize detection methods
        self._initialize_methods()

        # Results storage
        self.detection_results = {}
        self.method_scores = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for outlier detection"""
        return {
            'iqr': {
                'multiplier': 1.5,
                'method': 'robust'  # 'robust' or 'standard'
            },
            'zscore': {
                'threshold': 3.0,
                'method': 'modified'  # 'standard', 'modified', or 'robust'
            },
            'lof': {
                'n_neighbors': 20,
                'contamination': 'auto',
                'algorithm': 'auto'
            },
            'isolation_forest': {
                'n_estimators': 100,
                'contamination': 'auto',
                'random_state': 42
            },
            'one_class_svm': {
                'kernel': 'rbf',
                'nu': 0.1,
                'gamma': 'scale'
            },
            'mahalanobis': {
                'method': 'robust',  # 'robust' or 'standard'
                'threshold': 3.0
            },
            'ensemble': {
                'voting_method': 'majority',  # 'majority', 'weighted', 'consensus'
                'min_agreement': 0.6
            },
            'performance': {
                'batch_size': 1000,
                'parallel_processing': True,
                'memory_efficient': True
            }
        }

    def _initialize_methods(self):
        """Initialize all detection method instances"""
        try:
            # Isolation Forest (existing integration)
            self.isolation_forest = IsolationForest(
                n_estimators=self.config['isolation_forest']['n_estimators'],
                contamination=self.config['isolation_forest']['contamination'],
                random_state=self.config['isolation_forest']['random_state']
            )

            # Local Outlier Factor
            self.lof_detector = LocalOutlierFactor(
                n_neighbors=self.config['lof']['n_neighbors'],
                contamination=self.config['lof']['contamination'],
                algorithm=self.config['lof']['algorithm']
            )

            # One-Class SVM
            self.ocsvm_detector = OneClassSVM(
                kernel=self.config['one_class_svm']['kernel'],
                nu=self.config['one_class_svm']['nu'],
                gamma=self.config['one_class_svm']['gamma']
            )

            # Scalers for preprocessing
            self.standard_scaler = StandardScaler()
            self.robust_scaler = RobustScaler()

            self.logger.info("Advanced outlier detection methods initialized")

        except Exception as e:
            self.logger.error(f"Error initializing detection methods: {str(e)}")
            raise ValidationError(f"Failed to initialize outlier detection methods: {str(e)}")

    def detect_outliers_iqr(self, data: np.ndarray, method: str = 'robust') -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using Interquartile Range method

        Args:
            data: Input data array
            method: 'robust' or 'standard' IQR calculation

        Returns:
            Tuple of (outlier_flags, outlier_scores)
        """
        try:
            if method == 'robust':
                # Use robust statistics
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
            else:
                # Use standard method
                q1 = np.quantile(data, 0.25)
                q3 = np.quantile(data, 0.75)

            iqr = q3 - q1
            multiplier = self.config['iqr']['multiplier']

            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            # Calculate outlier scores (distance from bounds)
            outlier_scores = np.zeros(len(data))
            outlier_flags = np.zeros(len(data), dtype=bool)

            for i, value in enumerate(data):
                if value < lower_bound:
                    outlier_flags[i] = True
                    outlier_scores[i] = (lower_bound - value) / (iqr + 1e-10)
                elif value > upper_bound:
                    outlier_flags[i] = True
                    outlier_scores[i] = (value - upper_bound) / (iqr + 1e-10)

            return outlier_flags, outlier_scores

        except Exception as e:
            self.logger.error(f"Error in IQR outlier detection: {str(e)}")
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))

    def detect_outliers_zscore(self, data: np.ndarray, method: str = 'modified') -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using Z-score method

        Args:
            data: Input data array
            method: 'standard', 'modified', or 'robust' Z-score calculation

        Returns:
            Tuple of (outlier_flags, outlier_scores)
        """
        try:
            threshold = self.config['zscore']['threshold']

            if method == 'robust':
                # Use median and MAD (Median Absolute Deviation)
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                z_scores = 0.6745 * (data - median) / (mad + 1e-10)
            elif method == 'modified':
                # Use modified Z-score
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                z_scores = 0.6745 * (data - median) / (mad + 1e-10)
            else:
                # Standard Z-score
                mean = np.mean(data)
                std = np.std(data)
                z_scores = (data - mean) / (std + 1e-10)

            # Identify outliers
            outlier_flags = np.abs(z_scores) > threshold
            outlier_scores = np.abs(z_scores)

            return outlier_flags, outlier_scores

        except Exception as e:
            self.logger.error(f"Error in Z-score outlier detection: {str(e)}")
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))

    def detect_outliers_lof(self, data: np.ndarray, fit_predict: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using Local Outlier Factor

        Args:
            data: Input data array (2D for multivariate)
            fit_predict: Whether to fit and predict in one step

        Returns:
            Tuple of (outlier_flags, outlier_scores)
        """
        try:
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            if fit_predict:
                # Fit and predict
                outlier_scores = self.lof_detector.fit_predict(data)
            else:
                # Only predict on fitted data
                outlier_scores = self.lof_detector._predict(data)

            # LOF returns -1 for outliers, 1 for inliers
            outlier_flags = outlier_scores == -1

            # Convert scores to positive values (higher = more outlier-like)
            if hasattr(self.lof_detector, 'negative_outlier_factor_'):
                outlier_scores = -self.lof_detector.negative_outlier_factor_
            else:
                outlier_scores = np.abs(outlier_scores)

            return outlier_flags, outlier_scores

        except Exception as e:
            self.logger.error(f"Error in LOF outlier detection: {str(e)}")
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))

    def detect_outliers_isolation_forest(self, data: np.ndarray, fit_predict: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using Isolation Forest

        Args:
            data: Input data array (2D for multivariate)
            fit_predict: Whether to fit and predict in one step

        Returns:
            Tuple of (outlier_flags, outlier_scores)
        """
        try:
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            if fit_predict:
                outlier_scores = self.isolation_forest.fit_predict(data)
            else:
                outlier_scores = self.isolation_forest.predict(data)

            # Isolation Forest returns -1 for outliers, 1 for inliers
            outlier_flags = outlier_scores == -1

            # Get anomaly scores (higher = more outlier-like)
            if hasattr(self.isolation_forest, 'score_samples'):
                anomaly_scores = -self.isolation_forest.score_samples(data)
            else:
                anomaly_scores = np.abs(outlier_scores)

            return outlier_flags, anomaly_scores

        except Exception as e:
            self.logger.error(f"Error in Isolation Forest outlier detection: {str(e)}")
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))

    def detect_outliers_ocsvm(self, data: np.ndarray, fit_predict: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using One-Class SVM

        Args:
            data: Input data array (2D for multivariate)
            fit_predict: Whether to fit and predict in one step

        Returns:
            Tuple of (outlier_flags, outlier_scores)
        """
        try:
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            if fit_predict:
                outlier_scores = self.ocsvm_detector.fit_predict(data)
            else:
                outlier_scores = self.ocsvm_detector.predict(data)

            # One-Class SVM returns -1 for outliers, 1 for inliers
            outlier_flags = outlier_scores == -1

            # Get decision function scores (higher = more outlier-like)
            if hasattr(self.ocsvm_detector, 'decision_function'):
                decision_scores = -self.ocsvm_detector.decision_function(data)
            else:
                decision_scores = np.abs(outlier_scores)

            return outlier_flags, decision_scores

        except Exception as e:
            self.logger.error(f"Error in One-Class SVM outlier detection: {str(e)}")
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))

    def detect_outliers_mahalanobis(self, data: np.ndarray, method: str = 'robust') -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using Mahalanobis distance

        Args:
            data: Input data array (must be 2D for multivariate)
            method: 'robust' or 'standard' covariance estimation

        Returns:
            Tuple of (outlier_flags, outlier_scores)
        """
        try:
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            n_samples, n_features = data.shape
            threshold = self.config['mahalanobis']['threshold']

            if method == 'robust':
                # Use Minimum Covariance Determinant for robust covariance estimation
                try:
                    robust_cov = MinCovDet().fit(data)
                    mean = robust_cov.location_
                    cov_matrix = robust_cov.covariance_
                except:
                    # Fallback to standard method if MCD fails
                    mean = np.mean(data, axis=0)
                    cov_matrix = np.cov(data.T)
            else:
                # Standard covariance estimation
                mean = np.mean(data, axis=0)
                cov_matrix = np.cov(data.T)

            # Handle singular covariance matrix
            try:
                inv_cov_matrix = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                # Add small regularization term
                reg_term = 1e-6 * np.eye(n_features)
                inv_cov_matrix = np.linalg.inv(cov_matrix + reg_term)

            # Calculate Mahalanobis distances
            mahalanobis_distances = np.zeros(n_samples)

            for i in range(n_samples):
                diff = data[i] - mean
                mahalanobis_distances[i] = mahalanobis(diff, mean, inv_cov_matrix)

            # Identify outliers
            outlier_flags = mahalanobis_distances > threshold
            outlier_scores = mahalanobis_distances

            return outlier_flags, outlier_scores

        except Exception as e:
            self.logger.error(f"Error in Mahalanobis outlier detection: {str(e)}")
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))

    def detect_outliers_ensemble(self, data: np.ndarray, methods: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble outlier detection using multiple methods

        Args:
            data: Input data array
            methods: List of methods to use (if None, uses all available)

        Returns:
            Tuple of (outlier_flags, confidence_scores)
        """
        try:
            if methods is None:
                methods = ['iqr', 'zscore', 'lof', 'isolation_forest', 'mahalanobis']

            if data.ndim == 1:
                data = data.reshape(-1, 1)

            n_samples = len(data)
            method_results = {}
            method_scores = {}

            # Apply each method
            for method in methods:
                try:
                    if method == 'iqr':
                        flags, scores = self.detect_outliers_iqr(data.flatten())
                    elif method == 'zscore':
                        flags, scores = self.detect_outliers_zscore(data.flatten())
                    elif method == 'lof':
                        flags, scores = self.detect_outliers_lof(data)
                    elif method == 'isolation_forest':
                        flags, scores = self.detect_outliers_isolation_forest(data)
                    elif method == 'ocsvm':
                        flags, scores = self.detect_outliers_ocsvm(data)
                    elif method == 'mahalanobis':
                        flags, scores = self.detect_outliers_mahalanobis(data)
                    else:
                        continue

                    method_results[method] = flags
                    method_scores[method] = scores

                except Exception as e:
                    self.logger.warning(f"Error with method {method}: {str(e)}")
                    method_results[method] = np.zeros(n_samples, dtype=bool)
                    method_scores[method] = np.zeros(n_samples)

            # Ensemble voting
            voting_method = self.config['ensemble']['voting_method']
            min_agreement = self.config['ensemble']['min_agreement']

            if voting_method == 'majority':
                # Simple majority voting
                votes = np.column_stack(list(method_results.values()))
                vote_counts = np.sum(votes, axis=1)
                outlier_flags = vote_counts >= (len(methods) // 2 + 1)

            elif voting_method == 'weighted':
                # Weighted voting based on method reliability
                weights = self._get_method_weights(methods)
                votes = np.column_stack(list(method_results.values()))
                weighted_votes = votes * weights
                outlier_flags = np.sum(weighted_votes, axis=1) >= min_agreement

            else:  # consensus
                # Require minimum agreement
                votes = np.column_stack(list(method_results.values()))
                vote_counts = np.sum(votes, axis=1)
                outlier_flags = vote_counts >= (len(methods) * min_agreement)

            # Calculate confidence scores
            confidence_scores = np.mean(np.column_stack(list(method_scores.values())), axis=1)

            return outlier_flags, confidence_scores

        except Exception as e:
            self.logger.error(f"Error in ensemble outlier detection: {str(e)}")
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))

    def _get_method_weights(self, methods: List[str]) -> np.ndarray:
        """Get weights for different detection methods"""
        # Default weights (can be made configurable)
        weight_map = {
            'iqr': 1.0,
            'zscore': 1.0,
            'lof': 1.2,
            'isolation_forest': 1.3,
            'ocsvm': 1.1,
            'mahalanobis': 1.4
        }

        weights = [weight_map.get(method, 1.0) for method in methods]
        return np.array(weights)

    def detect_outliers_comprehensive(self, data: Union[np.ndarray, pd.DataFrame],
                                    methods: List[str] = None,
                                    return_details: bool = True) -> Dict[str, Any]:
        """
        Comprehensive outlier detection using multiple methods

        Args:
            data: Input data (numpy array or pandas DataFrame)
            methods: List of methods to use
            return_details: Whether to return detailed results

        Returns:
            Dictionary with detection results
        """
        try:
            # Convert to numpy array if needed
            if isinstance(data, pd.DataFrame):
                data_array = data.values
                feature_names = data.columns.tolist()
            else:
                data_array = np.array(data)
                feature_names = [f"feature_{i}" for i in range(data_array.shape[1] if data_array.ndim > 1 else 1)]

            if data_array.ndim == 1:
                data_array = data_array.reshape(-1, 1)

            if methods is None:
                methods = ['iqr', 'zscore', 'lof', 'isolation_forest', 'mahalanobis', 'ensemble']

            results = {}

            # Apply each method
            for method in methods:
                try:
                    if method == 'ensemble':
                        flags, scores = self.detect_outliers_ensemble(data_array, methods[:-1])  # Exclude ensemble itself
                    elif method == 'iqr':
                        flags, scores = self.detect_outliers_iqr(data_array.flatten())
                    elif method == 'zscore':
                        flags, scores = self.detect_outliers_zscore(data_array.flatten())
                    elif method == 'lof':
                        flags, scores = self.detect_outliers_lof(data_array)
                    elif method == 'isolation_forest':
                        flags, scores = self.detect_outliers_isolation_forest(data_array)
                    elif method == 'ocsvm':
                        flags, scores = self.detect_outliers_ocsvm(data_array)
                    elif method == 'mahalanobis':
                        flags, scores = self.detect_outliers_mahalanobis(data_array)
                    else:
                        continue

                    results[method] = {
                        'outlier_flags': flags,
                        'outlier_scores': scores,
                        'outlier_count': np.sum(flags),
                        'outlier_percentage': np.sum(flags) / len(flags) * 100
                    }

                except Exception as e:
                    self.logger.warning(f"Error with method {method}: {str(e)}")
                    results[method] = {
                        'outlier_flags': np.zeros(len(data_array), dtype=bool),
                        'outlier_scores': np.zeros(len(data_array)),
                        'outlier_count': 0,
                        'outlier_percentage': 0.0,
                        'error': str(e)
                    }

            # Store results
            self.detection_results = results

            if return_details:
                return {
                    'results': results,
                    'summary': self._generate_detection_summary(results),
                    'feature_names': feature_names,
                    'data_shape': data_array.shape
                }
            else:
                return results

        except Exception as e:
            self.logger.error(f"Error in comprehensive outlier detection: {str(e)}")
            return {'error': str(e)}

    def _generate_detection_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of detection results"""
        summary = {
            'total_methods': len(results),
            'method_comparison': {},
            'consensus_outliers': None,
            'high_confidence_outliers': None
        }

        # Method comparison
        for method, result in results.items():
            summary['method_comparison'][method] = {
                'outlier_count': result['outlier_count'],
                'outlier_percentage': result['outlier_percentage']
            }

        # Find consensus outliers (detected by multiple methods)
        if len(results) > 1:
            outlier_flags = np.column_stack([result['outlier_flags'] for result in results.values()])
            consensus_flags = np.sum(outlier_flags, axis=1) >= 2  # At least 2 methods agree
            summary['consensus_outliers'] = {
                'count': np.sum(consensus_flags),
                'percentage': np.sum(consensus_flags) / len(consensus_flags) * 100,
                'indices': np.where(consensus_flags)[0].tolist()
            }

        # High confidence outliers (high scores across methods)
        if len(results) > 1:
            outlier_scores = np.column_stack([result['outlier_scores'] for result in results.values()])
            avg_scores = np.mean(outlier_scores, axis=1)
            high_confidence_threshold = np.percentile(avg_scores, 95)
            high_confidence_flags = avg_scores > high_confidence_threshold
            summary['high_confidence_outliers'] = {
                'count': np.sum(high_confidence_flags),
                'percentage': np.sum(high_confidence_flags) / len(high_confidence_flags) * 100,
                'indices': np.where(high_confidence_flags)[0].tolist()
            }

        return summary

    def get_method_performance(self, true_labels: np.ndarray = None) -> Dict[str, Any]:
        """
        Evaluate performance of different detection methods

        Args:
            true_labels: Ground truth outlier labels (if available)

        Returns:
            Dictionary with performance metrics
        """
        if not self.detection_results:
            return {'error': 'No detection results available'}

        performance = {}

        for method, result in self.detection_results.items():
            perf = {
                'outlier_count': result['outlier_count'],
                'outlier_percentage': result['outlier_percentage'],
                'mean_score': np.mean(result['outlier_scores']),
                'std_score': np.std(result['outlier_scores']),
                'max_score': np.max(result['outlier_scores'])
            }

            if true_labels is not None:
                # Calculate classification metrics if ground truth is available
                from sklearn.metrics import precision_score, recall_score, f1_score

                try:
                    precision = precision_score(true_labels, result['outlier_flags'], zero_division=0)
                    recall = recall_score(true_labels, result['outlier_flags'], zero_division=0)
                    f1 = f1_score(true_labels, result['outlier_flags'], zero_division=0)

                    perf.update({
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    })
                except Exception as e:
                    perf['classification_error'] = str(e)

            performance[method] = perf

        return performance