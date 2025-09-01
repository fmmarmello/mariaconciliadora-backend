import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from imblearn.over_sampling import (
    SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE, KMeansSMOTE
)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import logging
from collections import Counter

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError

logger = get_logger(__name__)


class SMOTEImplementation:
    """
    Advanced SMOTE implementation with multiple variants for handling imbalanced datasets.

    Supports:
    - Classic SMOTE for oversampling minority classes
    - BorderlineSMOTE for samples near decision boundaries
    - ADASYN for adaptive synthetic sampling
    - SVMSMOTE for support vector machine-based sampling
    - KMeansSMOTE for cluster-based sampling
    - Hybrid approaches combining SMOTE with undersampling
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SMOTE implementation

        Args:
            config: Configuration for SMOTE variants
        """
        self.config = config
        self.logger = get_logger(__name__)

        # SMOTE variants
        self.smote_variants = {
            'classic': SMOTE,
            'borderline': BorderlineSMOTE,
            'adasyn': ADASYN,
            'svm': SVMSMOTE,
            'kmeans': KMeansSMOTE
        }

        # Hybrid approaches
        self.hybrid_methods = {
            'smote_enn': SMOTEENN,
            'smote_tomek': SMOTETomek
        }

        # Scalers and transformers
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Performance tracking
        self.performance_history = []

        self.logger.info("SMOTEImplementation initialized")

    def detect_imbalance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Detect and assess imbalance severity in the dataset

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary with imbalance assessment
        """
        try:
            class_counts = Counter(y)
            total_samples = len(y)
            n_classes = len(class_counts)

            # Calculate imbalance metrics
            minority_class = min(class_counts, key=class_counts.get)
            majority_class = max(class_counts, key=class_counts.get)

            minority_count = class_counts[minority_class]
            majority_count = class_counts[majority_class]

            imbalance_ratio = majority_count / minority_count
            minority_percentage = (minority_count / total_samples) * 100

            # Calculate imbalance severity
            if imbalance_ratio < 2:
                severity = 'low'
            elif imbalance_ratio < 5:
                severity = 'moderate'
            elif imbalance_ratio < 10:
                severity = 'high'
            else:
                severity = 'extreme'

            assessment = {
                'n_classes': n_classes,
                'class_distribution': dict(class_counts),
                'minority_class': minority_class,
                'majority_class': majority_class,
                'minority_count': minority_count,
                'majority_count': majority_count,
                'imbalance_ratio': imbalance_ratio,
                'minority_percentage': minority_percentage,
                'severity': severity,
                'requires_balancing': imbalance_ratio >= 2,
                'recommended_methods': self._recommend_methods(severity, n_classes)
            }

            self.logger.info(f"Imbalance detected: ratio={imbalance_ratio:.2f}, severity={severity}")
            return assessment

        except Exception as e:
            self.logger.error(f"Error detecting imbalance: {str(e)}")
            return {}

    def _recommend_methods(self, severity: str, n_classes: int) -> List[str]:
        """Recommend appropriate SMOTE methods based on imbalance severity"""
        recommendations = []

        if severity == 'low':
            recommendations.extend(['classic', 'borderline'])
        elif severity == 'moderate':
            recommendations.extend(['classic', 'borderline', 'adasyn'])
        elif severity == 'high':
            recommendations.extend(['borderline', 'adasyn', 'svm', 'kmeans'])
        else:  # extreme
            recommendations.extend(['adasyn', 'svm', 'kmeans', 'hybrid'])

        # Add hybrid methods for multi-class problems
        if n_classes > 2:
            recommendations.append('hybrid')

        return list(set(recommendations))  # Remove duplicates

    def apply_smote(self, X: np.ndarray, y: np.ndarray,
                   method: str = 'auto',
                   sampling_strategy: Union[str, Dict] = 'auto',
                   k_neighbors: int = 5,
                   **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE variant to balance the dataset

        Args:
            X: Feature matrix
            y: Target labels
            method: SMOTE method ('classic', 'borderline', 'adasyn', 'svm', 'kmeans', 'auto')
            sampling_strategy: Sampling strategy for SMOTE
            k_neighbors: Number of nearest neighbors
            **kwargs: Additional parameters for SMOTE variants

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        try:
            # Detect imbalance if method is auto
            if method == 'auto':
                imbalance_info = self.detect_imbalance(X, y)
                recommended_methods = imbalance_info.get('recommended_methods', ['classic'])
                method = recommended_methods[0] if recommended_methods else 'classic'

            # Select SMOTE variant
            if method in self.smote_variants:
                smote_class = self.smote_variants[method]
            elif method in self.hybrid_methods:
                smote_class = self.hybrid_methods[method]
            else:
                raise ValidationError(f"Unknown SMOTE method: {method}")

            # Configure SMOTE parameters
            smote_params = {
                'sampling_strategy': sampling_strategy,
                'random_state': 42,
                'k_neighbors': k_neighbors,
                **kwargs
            }

            # Special handling for different variants
            if method == 'kmeans':
                smote_params['cluster_balance_threshold'] = kwargs.get('cluster_balance_threshold', 0.1)
            elif method == 'svm':
                smote_params['svm_estimator'] = kwargs.get('svm_estimator', SVC(kernel='rbf'))

            # Apply SMOTE
            smote = smote_class(**smote_params)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            self.logger.info(f"Applied {method} SMOTE: {X.shape[0]} -> {X_resampled.shape[0]} samples")

            # Track performance
            self._track_performance(X, y, X_resampled, y_resampled, method)

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(f"Error applying {method} SMOTE: {str(e)}")
            raise ValidationError(f"SMOTE application failed: {str(e)}")

    def apply_borderline_smote(self, X: np.ndarray, y: np.ndarray,
                              kind: str = 'borderline-1',
                              **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply BorderlineSMOTE variant

        Args:
            X: Feature matrix
            y: Target labels
            kind: BorderlineSMOTE kind ('borderline-1' or 'borderline-2')
            **kwargs: Additional parameters

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        return self.apply_smote(X, y, method='borderline', kind=kind, **kwargs)

    def apply_adasyn(self, X: np.ndarray, y: np.ndarray,
                    n_neighbors: int = 5,
                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ADASYN (Adaptive Synthetic Sampling)

        Args:
            X: Feature matrix
            y: Target labels
            n_neighbors: Number of nearest neighbors
            **kwargs: Additional parameters

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        return self.apply_smote(X, y, method='adasyn', n_neighbors=n_neighbors, **kwargs)

    def apply_svm_smote(self, X: np.ndarray, y: np.ndarray,
                       svm_estimator=None,
                       **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SVMSMOTE (SVM-based SMOTE)

        Args:
            X: Feature matrix
            y: Target labels
            svm_estimator: SVM estimator for boundary detection
            **kwargs: Additional parameters

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if svm_estimator is None:
            svm_estimator = SVC(kernel='rbf', random_state=42)

        return self.apply_smote(X, y, method='svm', svm_estimator=svm_estimator, **kwargs)

    def apply_kmeans_smote(self, X: np.ndarray, y: np.ndarray,
                          cluster_balance_threshold: float = 0.1,
                          **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply KMeansSMOTE (Cluster-based SMOTE)

        Args:
            X: Feature matrix
            y: Target labels
            cluster_balance_threshold: Threshold for cluster balancing
            **kwargs: Additional parameters

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        return self.apply_smote(X, y, method='kmeans',
                               cluster_balance_threshold=cluster_balance_threshold, **kwargs)

    def apply_hybrid_method(self, X: np.ndarray, y: np.ndarray,
                           method: str = 'smote_enn',
                           **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply hybrid method combining SMOTE with undersampling

        Args:
            X: Feature matrix
            y: Target labels
            method: Hybrid method ('smote_enn' or 'smote_tomek')
            **kwargs: Additional parameters

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        return self.apply_smote(X, y, method=method, **kwargs)

    def compare_smote_methods(self, X: np.ndarray, y: np.ndarray,
                             methods: List[str] = None,
                             test_size: float = 0.2) -> Dict[str, Any]:
        """
        Compare different SMOTE methods on the same dataset

        Args:
            X: Feature matrix
            y: Target labels
            methods: List of methods to compare
            test_size: Test set size for evaluation

        Returns:
            Dictionary with comparison results
        """
        try:
            if methods is None:
                methods = ['classic', 'borderline', 'adasyn']

            results = {}

            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            for method in methods:
                try:
                    # Apply SMOTE
                    X_resampled, y_resampled = self.apply_smote(X_train, y_train, method=method)

                    # Calculate metrics
                    original_dist = Counter(y_train)
                    resampled_dist = Counter(y_resampled)

                    results[method] = {
                        'original_distribution': dict(original_dist),
                        'resampled_distribution': dict(resampled_dist),
                        'original_samples': len(y_train),
                        'resampled_samples': len(y_resampled),
                        'imbalance_ratio_before': max(original_dist.values()) / min(original_dist.values()),
                        'imbalance_ratio_after': max(resampled_dist.values()) / min(resampled_dist.values()),
                        'success': True
                    }

                except Exception as e:
                    results[method] = {
                        'success': False,
                        'error': str(e)
                    }

            return results

        except Exception as e:
            self.logger.error(f"Error comparing SMOTE methods: {str(e)}")
            return {}

    def _track_performance(self, X_orig: np.ndarray, y_orig: np.ndarray,
                          X_resampled: np.ndarray, y_resampled: np.ndarray,
                          method: str):
        """Track performance of SMOTE application"""
        try:
            performance_entry = {
                'method': method,
                'timestamp': pd.Timestamp.now(),
                'original_samples': len(X_orig),
                'resampled_samples': len(X_resampled),
                'original_classes': len(set(y_orig)),
                'resampled_classes': len(set(y_resampled)),
                'imbalance_ratio_before': max(Counter(y_orig).values()) / min(Counter(y_orig).values()),
                'imbalance_ratio_after': max(Counter(y_resampled).values()) / min(Counter(y_resampled).values())
            }

            self.performance_history.append(performance_entry)

        except Exception as e:
            self.logger.warning(f"Error tracking performance: {str(e)}")

    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance history of SMOTE applications"""
        return self.performance_history

    def optimize_smote_parameters(self, X: np.ndarray, y: np.ndarray,
                                method: str = 'classic',
                                param_grid: Dict[str, List] = None) -> Dict[str, Any]:
        """
        Optimize SMOTE parameters using grid search

        Args:
            X: Feature matrix
            y: Target labels
            method: SMOTE method to optimize
            param_grid: Parameter grid for optimization

        Returns:
            Dictionary with optimization results
        """
        try:
            if param_grid is None:
                param_grid = {
                    'k_neighbors': [3, 5, 7],
                    'sampling_strategy': ['auto', 'minority', 'not minority']
                }

            best_score = -1
            best_params = {}

            # Simple grid search (can be enhanced with cross-validation)
            for k_neighbors in param_grid.get('k_neighbors', [5]):
                for sampling_strategy in param_grid.get('sampling_strategy', ['auto']):
                    try:
                        # Apply SMOTE with current parameters
                        X_resampled, y_resampled = self.apply_smote(
                            X, y, method=method,
                            k_neighbors=k_neighbors,
                            sampling_strategy=sampling_strategy
                        )

                        # Calculate score (imbalance reduction)
                        original_ratio = max(Counter(y).values()) / min(Counter(y).values())
                        resampled_ratio = max(Counter(y_resampled).values()) / min(Counter(y_resampled).values())

                        score = original_ratio - resampled_ratio

                        if score > best_score:
                            best_score = score
                            best_params = {
                                'k_neighbors': k_neighbors,
                                'sampling_strategy': sampling_strategy
                            }

                    except Exception as e:
                        continue

            return {
                'best_params': best_params,
                'best_score': best_score,
                'method': method
            }

        except Exception as e:
            self.logger.error(f"Error optimizing SMOTE parameters: {str(e)}")
            return {}

    def get_method_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available SMOTE methods"""
        return {
            'classic': {
                'name': 'Classic SMOTE',
                'description': 'Standard Synthetic Minority Oversampling Technique',
                'best_for': 'General imbalance problems',
                'complexity': 'Low'
            },
            'borderline': {
                'name': 'BorderlineSMOTE',
                'description': 'SMOTE for samples near decision boundaries',
                'best_for': 'Complex decision boundaries',
                'complexity': 'Medium'
            },
            'adasyn': {
                'name': 'ADASYN',
                'description': 'Adaptive Synthetic Sampling',
                'best_for': 'Adaptive sampling based on difficulty',
                'complexity': 'Medium'
            },
            'svm': {
                'name': 'SVMSMOTE',
                'description': 'SVM-based SMOTE',
                'best_for': 'Non-linear decision boundaries',
                'complexity': 'High'
            },
            'kmeans': {
                'name': 'KMeansSMOTE',
                'description': 'Cluster-based SMOTE',
                'best_for': 'Clustered minority classes',
                'complexity': 'High'
            },
            'hybrid': {
                'name': 'Hybrid Methods',
                'description': 'SMOTE combined with undersampling',
                'best_for': 'Severe imbalance with noise',
                'complexity': 'High'
            }
        }