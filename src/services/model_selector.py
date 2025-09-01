import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# ML libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError, AIServiceError
from src.services.model_manager import ModelManager

logger = get_logger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Data class for comprehensive model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    cross_val_scores: List[float] = None
    cross_val_mean: Optional[float] = None
    cross_val_std: Optional[float] = None
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    memory_usage: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelComparisonResult:
    """Data class for model comparison results"""
    model_name: str
    performance_metrics: ModelPerformanceMetrics
    rank: int
    relative_score: float
    recommendation_score: float
    data_characteristics: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ABTestResult:
    """Data class for A/B testing results"""
    test_id: str
    model_a: str
    model_b: str
    winner: str
    confidence_level: float
    statistical_significance: bool
    performance_difference: Dict[str, float]
    sample_size: int
    test_duration: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelSelector:
    """
    Advanced Model Selection and Comparison System

    Provides comprehensive model selection, comparison, benchmarking,
    and recommendation capabilities with historical tracking and A/B testing.
    """

    def __init__(self, model_manager: ModelManager, config: Optional[Dict[str, Any]] = None):
        self.model_manager = model_manager
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize components
        self.performance_history = {}
        self.ab_tests = {}
        self.model_recommendations = {}
        self.data_characteristics_cache = {}

        # Load existing data if available
        self._load_persistent_data()

        self.logger.info("ModelSelector initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'performance_history_file': 'models/performance_history.json',
            'ab_tests_file': 'models/ab_tests.json',
            'recommendations_file': 'models/recommendations.json',
            'min_samples_for_comparison': 100,
            'cross_validation_folds': 5,
            'ab_test_confidence_level': 0.95,
            'performance_tracking_enabled': True,
            'auto_recommendation_enabled': True,
            'historical_analysis_window_days': 30,
            'model_selection_criteria': {
                'primary_metric': 'f1_score',
                'secondary_metrics': ['accuracy', 'precision', 'recall'],
                'weights': {'f1_score': 0.4, 'accuracy': 0.3, 'precision': 0.15, 'recall': 0.15}
            }
        }

    def _load_persistent_data(self):
        """Load persistent data from disk"""
        try:
            # Load performance history
            if os.path.exists(self.config['performance_history_file']):
                with open(self.config['performance_history_file'], 'r') as f:
                    self.performance_history = json.load(f)

            # Load A/B tests
            if os.path.exists(self.config['ab_tests_file']):
                with open(self.config['ab_tests_file'], 'r') as f:
                    self.ab_tests = json.load(f)

            # Load recommendations
            if os.path.exists(self.config['recommendations_file']):
                with open(self.config['recommendations_file'], 'r') as f:
                    self.model_recommendations = json.load(f)

        except Exception as e:
            self.logger.warning(f"Error loading persistent data: {str(e)}")

    def _save_persistent_data(self):
        """Save persistent data to disk"""
        try:
            os.makedirs('models', exist_ok=True)

            # Save performance history
            with open(self.config['performance_history_file'], 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)

            # Save A/B tests
            with open(self.config['ab_tests_file'], 'w') as f:
                json.dump(self.ab_tests, f, indent=2, default=str)

            # Save recommendations
            with open(self.config['recommendations_file'], 'w') as f:
                json.dump(self.model_recommendations, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving persistent data: {str(e)}")

    def analyze_data_characteristics(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Analyze data characteristics to inform model selection

        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Feature names

        Returns:
            Dictionary of data characteristics
        """
        try:
            characteristics = {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
                'class_distribution': {},
                'feature_types': {},
                'data_complexity': {},
                'timestamp': datetime.now().isoformat()
            }

            # Class distribution
            unique_classes, class_counts = np.unique(y, return_counts=True)
            for cls, count in zip(unique_classes, class_counts):
                characteristics['class_distribution'][str(cls)] = {
                    'count': int(count),
                    'percentage': float(count / len(y) * 100)
                }

            # Feature analysis
            if feature_names:
                characteristics['feature_names'] = feature_names

            # Data complexity metrics
            characteristics['data_complexity'] = {
                'class_imbalance_ratio': float(max(class_counts) / min(class_counts)),
                'feature_to_sample_ratio': float(X.shape[1] / X.shape[0]),
                'sparse_features': float(np.count_nonzero(X == 0) / X.size),
                'categorical_features_estimate': 0  # Would need more sophisticated detection
            }

            # Cache the analysis
            data_hash = hashlib.md5(f"{X.shape}_{y.shape}".encode()).hexdigest()[:8]
            self.data_characteristics_cache[data_hash] = characteristics

            return characteristics

        except Exception as e:
            self.logger.error(f"Error analyzing data characteristics: {str(e)}")
            return {'error': str(e)}

    def select_best_model(self, X: np.ndarray, y: np.ndarray,
                         candidate_models: Optional[List[str]] = None,
                         data_characteristics: Optional[Dict[str, Any]] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Advanced model selection based on data characteristics and performance metrics

        Args:
            X: Feature matrix
            y: Target labels
            candidate_models: List of model names to consider
            data_characteristics: Pre-computed data characteristics
            **kwargs: Additional arguments

        Returns:
            Selection result with recommendations
        """
        try:
            if candidate_models is None:
                candidate_models = ['random_forest', 'xgboost', 'lightgbm']

            if data_characteristics is None:
                data_characteristics = self.analyze_data_characteristics(X, y)

            # Analyze data to determine best candidates
            recommended_models = self._recommend_models_based_on_data(data_characteristics)

            # If BERT is recommended and we have text data, add it
            if 'bert' in recommended_models and 'texts' in kwargs:
                if 'bert' not in candidate_models:
                    candidate_models.append('bert')

            # Perform comprehensive comparison
            comparison_results = self.compare_models_comprehensive(
                X, y, candidate_models, data_characteristics, **kwargs
            )

            if not comparison_results:
                return {'error': 'No comparison results available'}

            # Select best model based on weighted criteria
            best_model_result = self._select_best_from_comparison(comparison_results)

            # Generate detailed recommendation
            recommendation = self._generate_model_recommendation(
                best_model_result, comparison_results, data_characteristics
            )

            result = {
                'best_model': best_model_result.model_name,
                'recommendation': recommendation,
                'comparison_results': [result.to_dict() for result in comparison_results],
                'data_characteristics': data_characteristics,
                'selection_criteria': self.config['model_selection_criteria'],
                'timestamp': datetime.now().isoformat()
            }

            # Store in performance history
            if self.config['performance_tracking_enabled']:
                self._store_performance_history(result)

            return result

        except Exception as e:
            self.logger.error(f"Error in advanced model selection: {str(e)}")
            return {'error': str(e)}

    def _recommend_models_based_on_data(self, data_characteristics: Dict[str, Any]) -> List[str]:
        """Recommend models based on data characteristics"""
        recommendations = []

        n_samples = data_characteristics.get('n_samples', 0)
        n_features = data_characteristics.get('n_features', 0)
        n_classes = data_characteristics.get('n_classes', 2)
        class_imbalance = data_characteristics.get('data_complexity', {}).get('class_imbalance_ratio', 1.0)

        # Small dataset - prefer simpler models
        if n_samples < 1000:
            recommendations.extend(['random_forest', 'xgboost'])
        else:
            recommendations.extend(['xgboost', 'lightgbm', 'random_forest'])

        # High-dimensional data - prefer models that handle it well
        if n_features > 100:
            recommendations.insert(0, 'lightgbm')  # LightGBM handles high dimensions well

        # Imbalanced data - prefer models with good imbalance handling
        if class_imbalance > 5.0:
            recommendations.insert(0, 'xgboost')  # XGBoost handles imbalance well

        # Multi-class problems
        if n_classes > 2:
            recommendations.extend(['random_forest', 'xgboost'])

        return list(set(recommendations))  # Remove duplicates while preserving order

    def compare_models_comprehensive(self, X: np.ndarray, y: np.ndarray,
                                   models_to_compare: List[str],
                                   data_characteristics: Dict[str, Any],
                                   **kwargs) -> List[ModelComparisonResult]:
        """
        Comprehensive model comparison with detailed benchmarking

        Args:
            X: Feature matrix
            y: Target labels
            models_to_compare: List of model names to compare
            data_characteristics: Data characteristics
            **kwargs: Additional arguments

        Returns:
            List of comparison results
        """
        comparison_results = []
        texts = kwargs.get('texts', [])

        for model_name in models_to_compare:
            try:
                self.logger.info(f"Evaluating {model_name} for comprehensive comparison")

                # Measure training time
                import time
                start_time = time.time()

                # Train model
                if model_name == 'bert' and texts:
                    train_result = self.model_manager.train_model(model_name, X, y, texts=texts)
                else:
                    train_result = self.model_manager.train_model(model_name, X, y)

                training_time = time.time() - start_time

                if not train_result.get('success', False):
                    self.logger.warning(f"Failed to train {model_name}: {train_result.get('error', 'Unknown error')}")
                    continue

                # Measure prediction time
                start_time = time.time()
                predictions = self.model_manager.predict(model_name, X)
                prediction_time = time.time() - start_time

                # Calculate comprehensive metrics
                metrics = self._calculate_comprehensive_metrics(
                    model_name, X, y, predictions, training_time, prediction_time
                )

                # Calculate cross-validation scores
                cv_scores = self._perform_cross_validation(model_name, X, y, texts)

                # Update metrics with CV results
                metrics.cross_val_scores = cv_scores
                metrics.cross_val_mean = float(np.mean(cv_scores))
                metrics.cross_val_std = float(np.std(cv_scores))

                # Calculate recommendation score
                recommendation_score = self._calculate_recommendation_score(metrics, data_characteristics)

                # Create comparison result
                result = ModelComparisonResult(
                    model_name=model_name,
                    performance_metrics=metrics,
                    rank=0,  # Will be set after all models are evaluated
                    relative_score=metrics.f1_score,  # Primary metric
                    recommendation_score=recommendation_score,
                    data_characteristics=data_characteristics,
                    timestamp=datetime.now().isoformat()
                )

                comparison_results.append(result)

            except Exception as e:
                self.logger.error(f"Error comparing {model_name}: {str(e)}")
                continue

        # Sort by recommendation score and assign ranks
        comparison_results.sort(key=lambda x: x.recommendation_score, reverse=True)
        for i, result in enumerate(comparison_results):
            result.rank = i + 1

        return comparison_results

    def _calculate_comprehensive_metrics(self, model_name: str, X: np.ndarray, y: np.ndarray,
                                       predictions: np.ndarray, training_time: float,
                                       prediction_time: float) -> ModelPerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            # Basic classification metrics
            accuracy = accuracy_score(y, predictions)
            precision = precision_score(y, predictions, average='weighted', zero_division=0)
            recall = recall_score(y, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y, predictions, average='weighted', zero_division=0)

            # Confusion matrix and classification report
            conf_matrix = confusion_matrix(y, predictions).tolist()
            class_report = classification_report(y, predictions, output_dict=True, zero_division=0)

            # AUC-ROC if binary classification
            auc_roc = None
            try:
                if len(np.unique(y)) == 2:
                    probabilities = self.model_manager.predict_proba(model_name, X)
                    if probabilities is not None and len(probabilities.shape) > 1:
                        auc_roc = roc_auc_score(y, probabilities[:, 1])
            except:
                pass

            # Feature importance if available
            feature_importance = None
            try:
                model_info = self.model_manager.get_model_info(model_name)
                if 'training_metadata' in model_info and 'feature_importances' in model_info['training_metadata']:
                    feature_importance = model_info['training_metadata']['feature_importances']
            except:
                pass

            return ModelPerformanceMetrics(
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
                auc_roc=float(auc_roc) if auc_roc is not None else None,
                training_time=float(training_time),
                prediction_time=float(prediction_time),
                feature_importance=feature_importance,
                confusion_matrix=conf_matrix,
                classification_report=class_report
            )

        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {str(e)}")
            return ModelPerformanceMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0
            )

    def _perform_cross_validation(self, model_name: str, X: np.ndarray, y: np.ndarray,
                                texts: List[str] = None, cv_folds: int = 5) -> List[float]:
        """Perform cross-validation for a model"""
        try:
            if model_name == 'bert' and texts:
                # Special handling for BERT with text data
                return self._bert_cross_validation(texts, y, cv_folds)
            else:
                # Standard cross-validation
                model = self.model_manager.models.get(model_name)
                if model and hasattr(model, 'model') and model.model:
                    scores = cross_val_score(
                        model.model, X, y, cv=cv_folds,
                        scoring='f1_weighted'
                    )
                    return scores.tolist()
                else:
                    # Fallback: simple train/test split multiple times
                    scores = []
                    for _ in range(cv_folds):
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        # Quick training and evaluation
                        temp_model = self.model_manager.model_registry[model_name](
                            self.model_manager.config['model_configs'].get(model_name, {})
                        )
                        temp_model.train(X_train, y_train)
                        predictions = temp_model.predict(X_test)
                        score = f1_score(y_test, predictions, average='weighted', zero_division=0)
                        scores.append(score)
                    return scores

        except Exception as e:
            self.logger.error(f"Error in cross-validation for {model_name}: {str(e)}")
            return [0.0] * cv_folds

    def _bert_cross_validation(self, texts: List[str], labels: List[str], cv_folds: int) -> List[float]:
        """Cross-validation for BERT model"""
        try:
            scores = []
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

            for train_idx, val_idx in kf.split(texts):
                train_texts = [texts[i] for i in train_idx]
                train_labels = [labels[i] for i in train_idx]
                val_texts = [texts[i] for i in val_idx]
                val_labels = [labels[i] for i in val_idx]

                # Train BERT model
                result = self.model_manager.models['bert'].bert_classifier.train(
                    train_texts, train_labels, val_texts, val_labels
                )

                if result.get('success') and 'evaluation_metrics' in result:
                    score = result['evaluation_metrics'].get('eval_f1', 0.0)
                    scores.append(score)
                else:
                    scores.append(0.0)

            return scores

        except Exception as e:
            self.logger.error(f"Error in BERT cross-validation: {str(e)}")
            return [0.0] * cv_folds

    def _calculate_recommendation_score(self, metrics: ModelPerformanceMetrics,
                                       data_characteristics: Dict[str, Any]) -> float:
        """Calculate recommendation score based on metrics and data characteristics"""
        try:
            weights = self.config['model_selection_criteria']['weights']

            # Base score from weighted metrics
            base_score = (
                metrics.f1_score * weights.get('f1_score', 0.4) +
                metrics.accuracy * weights.get('accuracy', 0.3) +
                metrics.precision * weights.get('precision', 0.15) +
                metrics.recall * weights.get('recall', 0.15)
            )

            # Adjust for cross-validation stability
            if metrics.cross_val_std is not None:
                stability_bonus = max(0, 1.0 - metrics.cross_val_std) * 0.1
                base_score += stability_bonus

            # Adjust for training efficiency
            if metrics.training_time is not None and metrics.training_time > 0:
                efficiency_bonus = min(0.05, 10.0 / metrics.training_time)  # Bonus for faster training
                base_score += efficiency_bonus

            # Adjust for data characteristics
            n_samples = data_characteristics.get('n_samples', 1000)
            if n_samples < 100:
                base_score *= 0.9  # Penalty for very small datasets
            elif n_samples > 10000:
                base_score *= 1.05  # Bonus for large datasets

            return float(base_score)

        except Exception as e:
            self.logger.error(f"Error calculating recommendation score: {str(e)}")
            return 0.0

    def _select_best_from_comparison(self, comparison_results: List[ModelComparisonResult]) -> ModelComparisonResult:
        """Select the best model from comparison results"""
        if not comparison_results:
            raise ValueError("No comparison results available")

        # Sort by recommendation score
        sorted_results = sorted(comparison_results, key=lambda x: x.recommendation_score, reverse=True)
        return sorted_results[0]

    def _generate_model_recommendation(self, best_result: ModelComparisonResult,
                                      all_results: List[ModelComparisonResult],
                                      data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed model recommendation"""
        try:
            recommendation = {
                'recommended_model': best_result.model_name,
                'confidence_level': 'high' if best_result.recommendation_score > 0.8 else 'medium',
                'performance_score': best_result.performance_metrics.f1_score,
                'improvement_over_baseline': 0.0,
                'key_advantages': [],
                'considerations': [],
                'alternative_models': []
            }

            # Calculate improvement over baseline (worst performing model)
            if len(all_results) > 1:
                worst_result = min(all_results, key=lambda x: x.performance_metrics.f1_score)
                improvement = best_result.performance_metrics.f1_score - worst_result.performance_metrics.f1_score
                recommendation['improvement_over_baseline'] = improvement

            # Key advantages
            metrics = best_result.performance_metrics
            if metrics.f1_score > 0.8:
                recommendation['key_advantages'].append('High overall performance')
            if metrics.cross_val_std and metrics.cross_val_std < 0.05:
                recommendation['key_advantages'].append('Stable cross-validation performance')
            if metrics.training_time and metrics.training_time < 60:
                recommendation['key_advantages'].append('Fast training time')

            # Considerations based on data characteristics
            n_samples = data_characteristics.get('n_samples', 0)
            if n_samples < 100:
                recommendation['considerations'].append('Small dataset may lead to overfitting')
            if data_characteristics.get('data_complexity', {}).get('class_imbalance_ratio', 1.0) > 5.0:
                recommendation['considerations'].append('Consider techniques for imbalanced data')

            # Alternative models
            for result in all_results[1:4]:  # Top 3 alternatives
                recommendation['alternative_models'].append({
                    'model': result.model_name,
                    'performance_score': result.performance_metrics.f1_score,
                    'rank': result.rank
                })

            return recommendation

        except Exception as e:
            self.logger.error(f"Error generating model recommendation: {str(e)}")
            return {'error': str(e)}

    def perform_ab_test(self, model_a: str, model_b: str, X: np.ndarray, y: np.ndarray,
                       test_duration_hours: int = 24, **kwargs) -> Dict[str, Any]:
        """
        Perform A/B testing between two models

        Args:
            model_a: Name of first model
            model_b: Name of second model
            X: Feature matrix
            y: Target labels
            test_duration_hours: Duration of the test in hours
            **kwargs: Additional arguments

        Returns:
            A/B test results
        """
        try:
            test_id = f"ab_test_{model_a}_vs_{model_b}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.logger.info(f"Starting A/B test: {model_a} vs {model_b}")

            # Split data for A/B testing
            X_a, X_b, y_a, y_b = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

            # Evaluate both models
            results_a = self.model_manager.evaluate_model(model_a, X_a, y_a)
            results_b = self.model_manager.evaluate_model(model_b, X_b, y_b)

            if 'error' in results_a or 'error' in results_b:
                return {'error': 'Failed to evaluate models for A/B testing'}

            # Calculate performance difference
            performance_diff = {
                'accuracy_diff': results_b.get('accuracy', 0) - results_a.get('accuracy', 0),
                'f1_diff': results_b.get('f1_score', 0) - results_a.get('f1_score', 0),
                'precision_diff': results_b.get('precision', 0) - results_a.get('precision', 0),
                'recall_diff': results_b.get('recall', 0) - results_a.get('recall', 0)
            }

            # Determine winner based on primary metric
            primary_metric = self.config['model_selection_criteria']['primary_metric']
            metric_a = results_a.get(primary_metric, 0)
            metric_b = results_b.get(primary_metric, 0)

            winner = model_b if metric_b > metric_a else model_a
            confidence_level = abs(metric_b - metric_a) / max(metric_a, metric_b, 0.001)

            # Statistical significance (simplified)
            statistical_significance = confidence_level > 0.05

            # Create A/B test result
            ab_result = ABTestResult(
                test_id=test_id,
                model_a=model_a,
                model_b=model_b,
                winner=winner,
                confidence_level=float(confidence_level),
                statistical_significance=statistical_significance,
                performance_difference=performance_diff,
                sample_size=len(X),
                test_duration=f"{test_duration_hours} hours",
                timestamp=datetime.now().isoformat()
            )

            # Store result
            self.ab_tests[test_id] = ab_result.to_dict()
            self._save_persistent_data()

            return {
                'test_id': test_id,
                'winner': winner,
                'confidence_level': confidence_level,
                'statistical_significance': statistical_significance,
                'performance_difference': performance_diff,
                'model_a_results': results_a,
                'model_b_results': results_b,
                'recommendation': f"Use {winner} for better performance"
            }

        except Exception as e:
            self.logger.error(f"Error performing A/B test: {str(e)}")
            return {'error': str(e)}

    def get_performance_history(self, model_name: Optional[str] = None,
                               days: int = 30) -> Dict[str, Any]:
        """
        Get historical performance data

        Args:
            model_name: Specific model name (optional)
            days: Number of days to look back

        Returns:
            Historical performance data
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            if model_name:
                # Get history for specific model
                model_history = []
                for entry in self.performance_history.values():
                    if (entry.get('best_model') == model_name and
                        datetime.fromisoformat(entry.get('timestamp', '2000-01-01')) > cutoff_date):
                        model_history.append(entry)

                return {
                    'model': model_name,
                    'history': model_history,
                    'count': len(model_history),
                    'average_performance': self._calculate_average_performance(model_history)
                }
            else:
                # Get overall history
                recent_history = []
                for entry in self.performance_history.values():
                    if datetime.fromisoformat(entry.get('timestamp', '2000-01-01')) > cutoff_date:
                        recent_history.append(entry)

                return {
                    'total_entries': len(recent_history),
                    'model_distribution': self._analyze_model_distribution(recent_history),
                    'performance_trends': self._analyze_performance_trends(recent_history),
                    'recent_history': recent_history[-10:]  # Last 10 entries
                }

        except Exception as e:
            self.logger.error(f"Error getting performance history: {str(e)}")
            return {'error': str(e)}

    def _calculate_average_performance(self, history: List[Dict]) -> Dict[str, float]:
        """Calculate average performance from history"""
        if not history:
            return {}

        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        averages = {}

        for metric in metrics:
            values = []
            for entry in history:
                comparison_results = entry.get('comparison_results', [])
                for result in comparison_results:
                    perf = result.get('performance_metrics', {})
                    if metric in perf:
                        values.append(perf[metric])

            if values:
                averages[metric] = float(np.mean(values))

        return averages

    def _analyze_model_distribution(self, history: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of selected models"""
        distribution = {}

        for entry in history:
            model = entry.get('best_model')
            if model:
                distribution[model] = distribution.get(model, 0) + 1

        return distribution

    def _analyze_performance_trends(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if not history:
            return {}

        # Sort by timestamp
        sorted_history = sorted(history, key=lambda x: x.get('timestamp', '2000-01-01'))

        trends = {
            'performance_over_time': [],
            'model_preference_trends': []
        }

        for entry in sorted_history:
            timestamp = entry.get('timestamp')
            best_model = entry.get('best_model')

            # Performance trend
            comparison_results = entry.get('comparison_results', [])
            if comparison_results:
                best_result = max(comparison_results,
                                key=lambda x: x.get('performance_metrics', {}).get('f1_score', 0))
                perf = best_result.get('performance_metrics', {})
                trends['performance_over_time'].append({
                    'timestamp': timestamp,
                    'f1_score': perf.get('f1_score', 0),
                    'model': best_result.get('model_name')
                })

            # Model preference trend
            trends['model_preference_trends'].append({
                'timestamp': timestamp,
                'preferred_model': best_model
            })

        return trends

    def _store_performance_history(self, result: Dict[str, Any]):
        """Store performance result in history"""
        try:
            history_id = f"selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.performance_history[history_id] = result
            self._save_persistent_data()
        except Exception as e:
            self.logger.error(f"Error storing performance history: {str(e)}")

    def generate_performance_report(self, comparison_results: List[ModelComparisonResult],
                                  data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive performance report

        Args:
            comparison_results: Model comparison results
            data_characteristics: Data characteristics

        Returns:
            Performance report
        """
        try:
            report = {
                'summary': {
                    'total_models_compared': len(comparison_results),
                    'best_model': comparison_results[0].model_name if comparison_results else None,
                    'data_characteristics': data_characteristics,
                    'generated_at': datetime.now().isoformat()
                },
                'model_rankings': [],
                'performance_analysis': {},
                'recommendations': []
            }

            # Model rankings
            for result in comparison_results:
                ranking = {
                    'rank': result.rank,
                    'model': result.model_name,
                    'f1_score': result.performance_metrics.f1_score,
                    'accuracy': result.performance_metrics.accuracy,
                    'cross_val_mean': result.performance_metrics.cross_val_mean,
                    'training_time': result.performance_metrics.training_time,
                    'recommendation_score': result.recommendation_score
                }
                report['model_rankings'].append(ranking)

            # Performance analysis
            report['performance_analysis'] = {
                'best_performer': comparison_results[0].model_name if comparison_results else None,
                'performance_range': {
                    'min_f1': min(r.performance_metrics.f1_score for r in comparison_results),
                    'max_f1': max(r.performance_metrics.f1_score for r in comparison_results),
                    'avg_f1': np.mean([r.performance_metrics.f1_score for r in comparison_results])
                },
                'stability_analysis': self._analyze_model_stability(comparison_results),
                'efficiency_analysis': self._analyze_model_efficiency(comparison_results)
            }

            # Recommendations
            report['recommendations'] = self._generate_report_recommendations(comparison_results, data_characteristics)

            return report

        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            return {'error': str(e)}

    def _analyze_model_stability(self, comparison_results: List[ModelComparisonResult]) -> Dict[str, Any]:
        """Analyze model stability based on cross-validation"""
        stability = {}

        for result in comparison_results:
            model_name = result.model_name
            cv_std = result.performance_metrics.cross_val_std

            if cv_std is not None:
                if cv_std < 0.05:
                    stability_level = 'high'
                elif cv_std < 0.10:
                    stability_level = 'medium'
                else:
                    stability_level = 'low'

                stability[model_name] = {
                    'stability_level': stability_level,
                    'cv_std': cv_std,
                    'cv_mean': result.performance_metrics.cross_val_mean
                }

        return stability

    def _analyze_model_efficiency(self, comparison_results: List[ModelComparisonResult]) -> Dict[str, Any]:
        """Analyze model efficiency"""
        efficiency = {}

        training_times = [r.performance_metrics.training_time for r in comparison_results
                         if r.performance_metrics.training_time is not None]

        if training_times:
            avg_time = np.mean(training_times)
            for result in comparison_results:
                time = result.performance_metrics.training_time
                if time is not None:
                    if time <= avg_time * 0.5:
                        efficiency_level = 'high'
                    elif time <= avg_time * 1.5:
                        efficiency_level = 'medium'
                    else:
                        efficiency_level = 'low'

                    efficiency[result.model_name] = {
                        'efficiency_level': efficiency_level,
                        'training_time': time,
                        'relative_to_average': time / avg_time
                    }

        return efficiency

    def _generate_report_recommendations(self, comparison_results: List[ModelComparisonResult],
                                       data_characteristics: Dict[str, Any]) -> List[str]:
        """Generate report recommendations"""
        recommendations = []

        if not comparison_results:
            return recommendations

        best_model = comparison_results[0]

        # Primary recommendation
        recommendations.append(f"Use {best_model.model_name} as the primary model (F1: {best_model.performance_metrics.f1_score:.3f})")

        # Stability recommendations
        stability_analysis = self._analyze_model_stability(comparison_results)
        stable_models = [model for model, stats in stability_analysis.items()
                        if stats['stability_level'] == 'high']

        if stable_models:
            recommendations.append(f"Consider {', '.join(stable_models)} for stable performance across different data splits")

        # Efficiency recommendations
        efficiency_analysis = self._analyze_model_efficiency(comparison_results)
        efficient_models = [model for model, stats in efficiency_analysis.items()
                           if stats['efficiency_level'] == 'high']

        if efficient_models:
            recommendations.append(f"For fast training, consider {', '.join(efficient_models)}")

        # Data-specific recommendations
        n_samples = data_characteristics.get('n_samples', 0)
        if n_samples < 500:
            recommendations.append("With small dataset, consider using cross-validation and avoiding complex models")
        elif n_samples > 10000:
            recommendations.append("With large dataset, complex models may provide better performance")

        return recommendations

    def get_ab_test_results(self, test_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get A/B test results

        Args:
            test_id: Specific test ID (optional)

        Returns:
            A/B test results
        """
        if test_id:
            result = self.ab_tests.get(test_id)
            return result if result else {'error': f'Test {test_id} not found'}
        else:
            return {
                'total_tests': len(self.ab_tests),
                'recent_tests': list(self.ab_tests.values())[-5:],  # Last 5 tests
                'all_tests': list(self.ab_tests.keys())
            }

    def export_comparison_report(self, comparison_results: List[ModelComparisonResult],
                               filepath: str) -> bool:
        """
        Export comparison results to file

        Args:
            comparison_results: Model comparison results
            filepath: Export file path

        Returns:
            True if export successful
        """
        try:
            report = {
                'export_timestamp': datetime.now().isoformat(),
                'comparison_results': [result.to_dict() for result in comparison_results],
                'summary': {
                    'total_models': len(comparison_results),
                    'best_model': comparison_results[0].model_name if comparison_results else None,
                    'performance_range': {
                        'min_f1': min(r.performance_metrics.f1_score for r in comparison_results) if comparison_results else 0,
                        'max_f1': max(r.performance_metrics.f1_score for r in comparison_results) if comparison_results else 0
                    }
                }
            }

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info(f"Comparison report exported to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting comparison report: {str(e)}")
            return False