import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from .smote_implementation import SMOTEImplementation
from .synthetic_data_generator import AdvancedSyntheticDataGenerator


class ImbalancedDataHandler:
    """
    Intelligent handler for imbalanced datasets with automatic strategy selection,
    performance monitoring, and hybrid approaches combining SMOTE with undersampling.

    Features:
    - Automatic imbalance detection and severity assessment
    - Strategy selection based on dataset characteristics
    - Hybrid approaches combining SMOTE with undersampling
    - Performance monitoring and adjustment
    - Integration with existing training pipelines
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the imbalanced data handler

        Args:
            config: Configuration for imbalance handling
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize components
        self.smote_handler = SMOTEImplementation(config.get('smote_config', {}))
        self.synthetic_generator = AdvancedSyntheticDataGenerator(config.get('synthetic_config', {}))

        # State tracking
        self.imbalance_info = {}
        self.balancing_history = []
        self.performance_metrics = {}

        # Scaler for preprocessing
        self.scaler = StandardScaler()

        self.logger.info("ImbalancedDataHandler initialized")

    def handle_imbalanced_data(self, X: np.ndarray, y: np.ndarray,
                              strategy: str = 'auto',
                              target_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main method to handle imbalanced data with automatic strategy selection

        Args:
            X: Feature matrix
            y: Target labels
            strategy: Balancing strategy ('auto', 'smote', 'synthetic', 'hybrid', 'undersample')
            target_ratio: Target balance ratio (1.0 = perfect balance)

        Returns:
            Tuple of (X_balanced, y_balanced)
        """
        try:
            start_time = time.time()

            # Detect imbalance
            self.imbalance_info = self.smote_handler.detect_imbalance(X, y)

            if not self.imbalance_info.get('requires_balancing', False):
                self.logger.info("Data does not require balancing")
                return X, y

            # Select strategy if auto
            if strategy == 'auto':
                strategy = self._select_optimal_strategy(X, y)

            self.logger.info(f"Selected balancing strategy: {strategy}")

            # Apply selected strategy
            if strategy == 'smote':
                X_balanced, y_balanced = self._apply_smote_strategy(X, y, target_ratio)
            elif strategy == 'synthetic':
                X_balanced, y_balanced = self._apply_synthetic_strategy(X, y, target_ratio)
            elif strategy == 'hybrid':
                X_balanced, y_balanced = self._apply_hybrid_strategy(X, y, target_ratio)
            elif strategy == 'undersample':
                X_balanced, y_balanced = self._apply_undersampling_strategy(X, y, target_ratio)
            else:
                raise ValidationError(f"Unknown balancing strategy: {strategy}")

            # Track performance
            processing_time = time.time() - start_time
            self._track_balancing_performance(X, y, X_balanced, y_balanced, strategy, processing_time)

            self.logger.info(f"Balancing completed: {X.shape[0]} -> {X_balanced.shape[0]} samples")
            return X_balanced, y_balanced

        except Exception as e:
            self.logger.error(f"Error handling imbalanced data: {str(e)}")
            raise ValidationError(f"Failed to handle imbalanced data: {str(e)}")

    def _select_optimal_strategy(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        Select the optimal balancing strategy based on data characteristics

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Optimal strategy name
        """
        try:
            severity = self.imbalance_info.get('severity', 'moderate')
            n_samples = X.shape[0]
            n_features = X.shape[1]
            n_classes = len(set(y))

            # Strategy selection logic
            if severity == 'low':
                return 'smote'  # Simple SMOTE for minor imbalances
            elif severity == 'moderate':
                if n_samples > 10000:
                    return 'hybrid'  # Hybrid for larger datasets
                else:
                    return 'smote'  # SMOTE for moderate imbalances
            elif severity == 'high':
                if n_features > 50:
                    return 'synthetic'  # Advanced generative models for high-dimensional data
                else:
                    return 'hybrid'  # Hybrid approach for high imbalance
            else:  # extreme
                if n_samples > 50000:
                    return 'undersample'  # Undersampling for very large datasets
                else:
                    return 'synthetic'  # Generative models for extreme imbalance

        except Exception as e:
            self.logger.warning(f"Error selecting optimal strategy: {str(e)}, using default")
            return 'smote'

    def _apply_smote_strategy(self, X: np.ndarray, y: np.ndarray,
                             target_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE-based balancing strategy"""
        try:
            # Choose SMOTE variant based on data characteristics
            severity = self.imbalance_info.get('severity', 'moderate')

            if severity == 'low':
                method = 'classic'
            elif severity == 'moderate':
                method = 'borderline'
            elif severity == 'high':
                method = 'adasyn'
            else:  # extreme
                method = 'svm'

            # Apply SMOTE
            X_balanced, y_balanced = self.smote_handler.apply_smote(
                X, y, method=method, sampling_strategy='auto'
            )

            return X_balanced, y_balanced

        except Exception as e:
            self.logger.error(f"Error applying SMOTE strategy: {str(e)}")
            return X, y

    def _apply_synthetic_strategy(self, X: np.ndarray, y: np.ndarray,
                                 target_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply synthetic data generation strategy"""
        try:
            # Prepare data for synthetic generation
            # For now, we'll use the existing synthetic generator
            # In production, this would be enhanced for conditional generation

            # Convert to DataFrame for synthetic generator
            if len(X.shape) == 2:
                df = pd.DataFrame(X)
                df['target'] = y
            else:
                df = pd.DataFrame({'feature': X, 'target': y})

            # Generate synthetic data
            synthetic_data = self.synthetic_generator.generate_synthetic_data(df)

            if synthetic_data is not None and not synthetic_data.empty:
                # Combine original and synthetic data
                combined_data = pd.concat([df, synthetic_data], ignore_index=True)

                # Extract features and target
                if 'target' in combined_data.columns:
                    y_balanced = combined_data['target'].values
                    X_balanced = combined_data.drop('target', axis=1).values
                else:
                    X_balanced = combined_data.values
                    y_balanced = y  # Keep original labels if no target in synthetic

                return X_balanced, y_balanced
            else:
                self.logger.warning("Synthetic data generation failed, returning original data")
                return X, y

        except Exception as e:
            self.logger.error(f"Error applying synthetic strategy: {str(e)}")
            return X, y

    def _apply_hybrid_strategy(self, X: np.ndarray, y: np.ndarray,
                              target_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply hybrid balancing strategy (SMOTE + undersampling)"""
        try:
            from imblearn.combine import SMOTEENN

            # Apply SMOTEENN (SMOTE + Edited Nearest Neighbors)
            smote_enn = SMOTEENN(random_state=42)
            X_balanced, y_balanced = smote_enn.fit_resample(X, y)

            self.logger.info(f"Hybrid strategy applied: SMOTE + ENN")
            return X_balanced, y_balanced

        except Exception as e:
            self.logger.error(f"Error applying hybrid strategy: {str(e)}")
            # Fallback to simple SMOTE
            return self._apply_smote_strategy(X, y, target_ratio)

    def _apply_undersampling_strategy(self, X: np.ndarray, y: np.ndarray,
                                     target_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply undersampling strategy for very large datasets"""
        try:
            from imblearn.under_sampling import RandomUnderSampler

            # Apply random undersampling
            rus = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X, y)

            self.logger.info(f"Undersampling strategy applied")
            return X_balanced, y_balanced

        except Exception as e:
            self.logger.error(f"Error applying undersampling strategy: {str(e)}")
            return X, y

    def _track_balancing_performance(self, X_orig: np.ndarray, y_orig: np.ndarray,
                                    X_balanced: np.ndarray, y_balanced: np.ndarray,
                                    strategy: str, processing_time: float):
        """Track performance of balancing operation"""
        try:
            original_dist = Counter(y_orig)
            balanced_dist = Counter(y_balanced)

            performance_entry = {
                'timestamp': datetime.now(),
                'strategy': strategy,
                'original_samples': len(X_orig),
                'balanced_samples': len(X_balanced),
                'original_distribution': dict(original_dist),
                'balanced_distribution': dict(balanced_dist),
                'original_imbalance_ratio': max(original_dist.values()) / min(original_dist.values()),
                'balanced_imbalance_ratio': max(balanced_dist.values()) / min(balanced_dist.values()),
                'processing_time': processing_time,
                'data_characteristics': {
                    'n_features': X_orig.shape[1],
                    'n_classes': len(original_dist)
                }
            }

            self.balancing_history.append(performance_entry)

        except Exception as e:
            self.logger.warning(f"Error tracking balancing performance: {str(e)}")

    def evaluate_balancing_effectiveness(self, X_train: np.ndarray, y_train: np.ndarray,
                                        X_test: np.ndarray, y_test: np.ndarray,
                                        X_balanced: np.ndarray, y_balanced: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of balancing on model performance

        Args:
            X_train: Original training features
            y_train: Original training labels
            X_test: Test features
            y_test: Test labels
            X_balanced: Balanced training features
            y_balanced: Balanced training labels

        Returns:
            Dictionary with evaluation results
        """
        try:
            # Train models on original and balanced data
            original_model = RandomForestClassifier(n_estimators=100, random_state=42)
            balanced_model = RandomForestClassifier(n_estimators=100, random_state=42)

            # Train on original data
            original_model.fit(X_train, y_train)
            original_pred = original_model.predict(X_test)

            # Train on balanced data
            balanced_model.fit(X_balanced, y_balanced)
            balanced_pred = balanced_model.predict(X_test)

            # Calculate metrics
            original_report = classification_report(y_test, original_pred, output_dict=True)
            balanced_report = classification_report(y_test, balanced_pred, output_dict=True)

            evaluation_results = {
                'original_performance': original_report,
                'balanced_performance': balanced_report,
                'improvement_metrics': {},
                'confusion_matrices': {
                    'original': confusion_matrix(y_test, original_pred).tolist(),
                    'balanced': confusion_matrix(y_test, balanced_pred).tolist()
                }
            }

            # Calculate improvements
            for metric in ['precision', 'recall', 'f1-score']:
                if metric in original_report['weighted avg'] and metric in balanced_report['weighted avg']:
                    original_score = original_report['weighted avg'][metric]
                    balanced_score = balanced_report['weighted avg'][metric]
                    improvement = balanced_score - original_score
                    evaluation_results['improvement_metrics'][f'{metric}_improvement'] = improvement

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error evaluating balancing effectiveness: {str(e)}")
            return {}

    def get_balancing_recommendations(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Get recommendations for balancing strategy based on data analysis

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary with recommendations
        """
        try:
            # Analyze data characteristics
            imbalance_info = self.smote_handler.detect_imbalance(X, y)

            recommendations = {
                'imbalance_analysis': imbalance_info,
                'recommended_strategies': [],
                'expected_outcomes': {},
                'implementation_notes': []
            }

            severity = imbalance_info.get('severity', 'moderate')
            n_samples = X.shape[0]
            n_features = X.shape[1]

            # Generate recommendations based on characteristics
            if severity == 'low':
                recommendations['recommended_strategies'].append({
                    'strategy': 'smote',
                    'method': 'classic',
                    'priority': 'high',
                    'reason': 'Minor imbalance, simple SMOTE sufficient'
                })
            elif severity == 'moderate':
                recommendations['recommended_strategies'].extend([
                    {
                        'strategy': 'smote',
                        'method': 'borderline',
                        'priority': 'high',
                        'reason': 'Moderate imbalance, borderline SMOTE effective'
                    },
                    {
                        'strategy': 'hybrid',
                        'method': 'smote_enn',
                        'priority': 'medium',
                        'reason': 'Hybrid approach for better noise handling'
                    }
                ])
            elif severity == 'high':
                recommendations['recommended_strategies'].extend([
                    {
                        'strategy': 'smote',
                        'method': 'adasyn',
                        'priority': 'high',
                        'reason': 'High imbalance, ADASYN adapts to difficulty'
                    },
                    {
                        'strategy': 'synthetic',
                        'method': 'vae',
                        'priority': 'medium',
                        'reason': 'Generative models for complex distributions'
                    }
                ])
            else:  # extreme
                recommendations['recommended_strategies'].extend([
                    {
                        'strategy': 'synthetic',
                        'method': 'gan',
                        'priority': 'high',
                        'reason': 'Extreme imbalance, advanced generative models needed'
                    },
                    {
                        'strategy': 'hybrid',
                        'method': 'smote_svm',
                        'priority': 'medium',
                        'reason': 'SVM-based SMOTE for complex boundaries'
                    }
                ])

            # Add implementation notes
            if n_samples > 100000:
                recommendations['implementation_notes'].append(
                    "Large dataset detected - consider undersampling or batch processing"
                )

            if n_features > 100:
                recommendations['implementation_notes'].append(
                    "High-dimensional data - generative models may be more effective than traditional SMOTE"
                )

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return {}

    def get_balancing_history(self) -> List[Dict[str, Any]]:
        """Get history of balancing operations"""
        return self.balancing_history

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics"""
        try:
            if not self.balancing_history:
                return {}

            # Calculate aggregate metrics
            total_operations = len(self.balancing_history)
            avg_processing_time = np.mean([op['processing_time'] for op in self.balancing_history])
            avg_imbalance_reduction = np.mean([
                op['original_imbalance_ratio'] - op['balanced_imbalance_ratio']
                for op in self.balancing_history
            ])

            strategy_counts = Counter([op['strategy'] for op in self.balancing_history])

            return {
                'total_operations': total_operations,
                'average_processing_time': avg_processing_time,
                'average_imbalance_reduction': avg_imbalance_reduction,
                'strategy_usage': dict(strategy_counts),
                'most_used_strategy': strategy_counts.most_common(1)[0][0] if strategy_counts else None
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}

    def save_balancing_state(self, filepath: str):
        """Save the current state of the balancing handler"""
        try:
            state = {
                'config': self.config,
                'imbalance_info': self.imbalance_info,
                'balancing_history': self.balancing_history,
                'performance_metrics': self.performance_metrics
            }

            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            self.logger.info(f"Balancing state saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving balancing state: {str(e)}")

    def load_balancing_state(self, filepath: str):
        """Load balancing state from file"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.config = state.get('config', self.config)
            self.imbalance_info = state.get('imbalance_info', {})
            self.balancing_history = state.get('balancing_history', [])
            self.performance_metrics = state.get('performance_metrics', {})

            self.logger.info(f"Balancing state loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading balancing state: {str(e)}")