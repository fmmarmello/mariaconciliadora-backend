import os
import json
import pickle
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# ML libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
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
from src.services.feature_engineer import FeatureEngineer
from src.services.bert_service import BERTTextClassifier
from src.services.data_augmentation_pipeline import DataAugmentationPipeline

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all ML models
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_trained = False
        self.label_encoder = None
        self.feature_names = []
        self.training_metadata = {}

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass

    def save(self, filepath: str) -> bool:
        """Save model to file"""
        try:
            save_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'config': self.config,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'training_metadata': self.training_metadata,
                'model_type': self.__class__.__name__
            }

            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)

            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load(self, filepath: str) -> bool:
        """Load model from file"""
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)

            self.model = save_data['model']
            self.label_encoder = save_data['label_encoder']
            self.config = save_data['config']
            self.is_trained = save_data['is_trained']
            self.feature_names = save_data['feature_names']
            self.training_metadata = save_data['training_metadata']

            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


class KMeansModel(BaseModel):
    """K-Means clustering model"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.n_clusters = config.get('n_clusters', 10)

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        try:
            # For clustering, we ignore y and determine n_clusters from data
            n_samples = X.shape[0]
            self.n_clusters = min(self.n_clusters, n_samples // 2, 20)

            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )

            cluster_labels = self.model.fit_predict(X)

            # Store training metadata
            self.training_metadata = {
                'n_clusters': self.n_clusters,
                'n_samples': n_samples,
                'n_features': X.shape[1],
                'inertia': self.model.inertia_,
                'training_timestamp': datetime.now().isoformat()
            }

            self.is_trained = True

            return {
                'success': True,
                'n_clusters': self.n_clusters,
                'inertia': self.model.inertia_,
                'cluster_centers': self.model.cluster_centers_.tolist()
            }

        except Exception as e:
            logger.error(f"Error training KMeans: {str(e)}")
            return {'success': False, 'error': str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # KMeans doesn't have probabilities, return cluster distances as proxy
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")

        distances = self.model.transform(X)
        # Convert distances to probabilities (inverse relationship)
        probabilities = 1 / (1 + distances)
        # Normalize to sum to 1
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'kmeans',
            'n_clusters': self.n_clusters,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata
        }


class RandomForestModel(BaseModel):
    """Random Forest classifier"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        try:
            # Encode labels if needed
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)

            self.model = RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 10),
                min_samples_split=self.config.get('min_samples_split', 5),
                min_samples_leaf=self.config.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )

            self.model.fit(X, y_encoded)

            # Store training metadata
            self.training_metadata = {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'n_features': X.shape[1],
                'n_classes': len(self.label_encoder.classes_),
                'feature_importances': self.model.feature_importances_.tolist(),
                'training_timestamp': datetime.now().isoformat()
            }

            self.is_trained = True

            return {
                'success': True,
                'n_estimators': self.model.n_estimators,
                'feature_importances': self.model.feature_importances_.tolist()
            }

        except Exception as e:
            logger.error(f"Error training Random Forest: {str(e)}")
            return {'success': False, 'error': str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'random_forest',
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata
        }


class XGBoostModel(BaseModel):
    """XGBoost classifier"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        try:
            # Encode labels if needed
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)

            self.model = xgb.XGBClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )

            self.model.fit(X, y_encoded)

            # Store training metadata
            self.training_metadata = {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'learning_rate': self.model.learning_rate,
                'n_features': X.shape[1],
                'n_classes': len(self.label_encoder.classes_),
                'feature_importances': self.model.feature_importances_.tolist(),
                'training_timestamp': datetime.now().isoformat()
            }

            self.is_trained = True

            return {
                'success': True,
                'n_estimators': self.model.n_estimators,
                'feature_importances': self.model.feature_importances_.tolist()
            }

        except Exception as e:
            logger.error(f"Error training XGBoost: {str(e)}")
            return {'success': False, 'error': str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions.astype(int))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'xgboost',
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata
        }


class LightGBMModel(BaseModel):
    """LightGBM classifier"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        try:
            # Encode labels if needed
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)

            self.model = lgb.LGBMClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

            self.model.fit(X, y_encoded)

            # Store training metadata
            self.training_metadata = {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'learning_rate': self.model.learning_rate,
                'n_features': X.shape[1],
                'n_classes': len(self.label_encoder.classes_),
                'feature_importances': self.model.feature_importances_.tolist(),
                'training_timestamp': datetime.now().isoformat()
            }

            self.is_trained = True

            return {
                'success': True,
                'n_estimators': self.model.n_estimators,
                'feature_importances': self.model.feature_importances_.tolist()
            }

        except Exception as e:
            logger.error(f"Error training LightGBM: {str(e)}")
            return {'success': False, 'error': str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions.astype(int))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'lightgbm',
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata
        }


class BERTModel(BaseModel):
    """BERT-based classifier"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bert_classifier = BERTTextClassifier(config.get('bert_config', {}))

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        try:
            # For BERT, we expect text data in kwargs
            texts = kwargs.get('texts', [])
            if not texts:
                return {'success': False, 'error': 'BERT requires text data for training'}

            result = self.bert_classifier.train(texts, y)

            if result['success']:
                self.is_trained = True
                self.label_encoder = self.bert_classifier.label_encoder

                # Store training metadata
                self.training_metadata = {
                    'training_result': result,
                    'n_texts': len(texts),
                    'n_labels': len(set(y)),
                    'training_timestamp': datetime.now().isoformat()
                }

            return result

        except Exception as e:
            logger.error(f"Error training BERT: {str(e)}")
            return {'success': False, 'error': str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")

        # For BERT, we expect text data
        texts = getattr(X, 'texts', None)
        if texts is None:
            raise ValueError("BERT prediction requires text data")

        return self.bert_classifier.predict(texts)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")

        # For BERT, we expect text data
        texts = getattr(X, 'texts', None)
        if texts is None:
            raise ValueError("BERT prediction requires text data")

        return self.bert_classifier.predict_proba(texts)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'bert',
            'is_trained': self.is_trained,
            'bert_info': self.bert_classifier.get_model_info(),
            'training_metadata': self.training_metadata
        }


class ModelManager:
    """
    Unified Model Manager for orchestrating multiple ML algorithms
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize components
        self.models = {}
        self.feature_engineer = FeatureEngineer(self.config.get('feature_engineer_config', {}))
        self.data_augmentation_pipeline = DataAugmentationPipeline(self.config.get('data_augmentation_config', {}))
        self.model_versions = {}
        self.performance_history = {}
        self.fallback_models = []

        # Initialize ModelSelector (lazy loading to avoid circular imports)
        self._model_selector = None

        # Model registry
        self.model_registry = {
            'kmeans': KMeansModel,
            'random_forest': RandomForestModel,
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'bert': BERTModel
        }

        # Initialize default models
        self._initialize_default_models()

        self.logger.info("ModelManager initialized")

    @property
    def model_selector(self):
        """Lazy load ModelSelector to avoid circular imports"""
        if self._model_selector is None:
            from src.services.model_selector import ModelSelector
            self._model_selector = ModelSelector(self)
        return self._model_selector

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'model_dir': 'models',
            'cache_dir': 'cache',
            'auto_save': True,
            'enable_fallback': True,
            'performance_tracking': True,
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'random_state': 42,
            'feature_engineer_config': {},
            'data_augmentation_config': {
                'general': {
                    'augmentation_ratio': 1.5,  # 1.5x augmentation for training
                    'random_seed': 42
                },
                'text_augmentation': {
                    'enabled': True,
                    'strategies': ['synonym_replacement']
                },
                'numerical_augmentation': {
                    'enabled': True,
                    'strategies': ['gaussian_noise']
                },
                'quality_control': {
                    'enabled': True
                }
            },
            'model_configs': {
                'kmeans': {'n_clusters': 10},
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                },
                'lightgbm': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                },
                'bert': {
                    'bert_config': {}
                }
            }
        }

    def _initialize_default_models(self):
        """Initialize default model instances"""
        for model_name, model_class in self.model_registry.items():
            try:
                config = self.config['model_configs'].get(model_name, {})
                self.models[model_name] = model_class(config)
                self.logger.debug(f"Initialized {model_name} model")
            except Exception as e:
                self.logger.warning(f"Failed to initialize {model_name}: {str(e)}")

    def select_best_model(self, X: np.ndarray, y: np.ndarray,
                         candidate_models: Optional[List[str]] = None,
                         cv_folds: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Advanced model selection using ModelSelector

        Args:
            X: Feature matrix
            y: Target labels
            candidate_models: List of model names to consider
            cv_folds: Number of cross-validation folds
            **kwargs: Additional arguments (e.g., texts for BERT)

        Returns:
            Advanced selection result with recommendations
        """
        try:
            self.logger.info("Using advanced ModelSelector for model selection")

            # Use the advanced ModelSelector
            result = self.model_selector.select_best_model(
                X, y, candidate_models, **kwargs
            )

            self.logger.info(f"ModelSelector selected {result.get('best_model')} as best model")
            return result

        except Exception as e:
            self.logger.error(f"Error in advanced model selection: {str(e)}")
            # Fallback to simple selection
            return self._simple_model_selection(X, y, candidate_models)

    def _simple_model_selection(self, X: np.ndarray, y: np.ndarray,
                               candidate_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Simple fallback model selection
        """
        if candidate_models is None:
            candidate_models = ['random_forest', 'xgboost', 'lightgbm']

        best_model = 'random_forest'
        best_score = 0

        self.logger.info(f"Fallback: Evaluating {len(candidate_models)} models")

        for model_name in candidate_models:
            try:
                if model_name not in self.models:
                    continue

                # Simple evaluation
                result = self.evaluate_model(model_name, X, y)
                if 'f1_score' in result:
                    score = result['f1_score']
                    if score > best_score:
                        best_score = score
                        best_model = model_name

            except Exception as e:
                self.logger.warning(f"Error evaluating {model_name}: {str(e)}")
                continue

        return {
            'best_model': best_model,
            'recommendation': f'Selected {best_model} using simple evaluation',
            'performance_score': best_score
        }

    def _evaluate_bert_model(self, texts: List[str], labels: List[str], cv_folds: int) -> float:
        """Evaluate BERT model using cross-validation"""
        try:
            # Simple evaluation - split data and train/evaluate
            split_idx = int(0.8 * len(texts))
            train_texts = texts[:split_idx]
            train_labels = labels[:split_idx]
            val_texts = texts[split_idx:]
            val_labels = labels[split_idx:]

            result = self.models['bert'].bert_classifier.train(
                train_texts, train_labels, val_texts, val_labels
            )

            if result['success'] and 'evaluation_metrics' in result:
                return result['evaluation_metrics'].get('eval_f1', 0.0)

            return 0.0

        except Exception as e:
            self.logger.error(f"Error evaluating BERT: {str(e)}")
            return 0.0

    def train_model_with_augmentation(self, model_name: str, transactions: List[Dict],
                                     target_column: str = 'category',
                                     use_augmentation: bool = True,
                                     augmentation_config: Optional[Dict[str, Any]] = None,
                                     optimize: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Train a model with integrated data augmentation

        Args:
            model_name: Name of the model to train
            transactions: Raw transaction data
            target_column: Target column name
            use_augmentation: Whether to use data augmentation
            augmentation_config: Custom augmentation configuration
            optimize: Whether to perform hyperparameter optimization
            **kwargs: Additional arguments

        Returns:
            Training result with augmentation information
        """
        try:
            self.logger.info(f"Training {model_name} with data augmentation (enabled: {use_augmentation})")

            # Apply data augmentation if enabled
            augmented_data = transactions
            augmentation_report = None

            if use_augmentation:
                self.logger.info("Applying data augmentation to training data")

                # Use custom config if provided
                if augmentation_config:
                    temp_pipeline = DataAugmentationPipeline(augmentation_config)
                    aug_data, augmentation_report = temp_pipeline.augment_dataset(transactions, 'transaction')
                else:
                    aug_data, augmentation_report = self.data_augmentation_pipeline.augment_dataset(transactions, 'transaction')

                augmented_data = aug_data.to_dict('records') if hasattr(aug_data, 'to_dict') else aug_data

                self.logger.info(f"Data augmentation completed. Original: {len(transactions)}, Augmented: {len(augmented_data)}")

            # Process augmented data into features
            X, y, feature_names = self.process_data(augmented_data, target_column)

            if X.size == 0:
                return {'success': False, 'error': 'No valid data after processing'}

            # Extract texts for BERT if needed
            texts = None
            if model_name == 'bert':
                texts = [entry.get('description', '') for entry in augmented_data]

            # Train the model
            training_result = self.train_model(model_name, X, y, optimize=optimize, texts=texts, **kwargs)

            # Add augmentation information to result
            if training_result['success']:
                training_result['augmentation_info'] = {
                    'augmentation_used': use_augmentation,
                    'original_data_size': len(transactions),
                    'augmented_data_size': len(augmented_data),
                    'augmentation_report': augmentation_report
                }

            return training_result

        except Exception as e:
            self.logger.error(f"Error in training with augmentation: {str(e)}")
            return {'success': False, 'error': str(e)}

    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray,
                    optimize: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Train a specific model

        Args:
            model_name: Name of the model to train
            X: Feature matrix
            y: Target labels
            optimize: Whether to perform hyperparameter optimization
            **kwargs: Additional arguments (e.g., texts for BERT)

        Returns:
            Training result dictionary
        """
        if model_name not in self.models:
            return {'success': False, 'error': f'Model {model_name} not found'}

        try:
            self.logger.info(f"Training {model_name} model")

            model = self.models[model_name]

            # Special handling for BERT
            if model_name == 'bert':
                texts = kwargs.get('texts', [])
                if not texts:
                    return {'success': False, 'error': 'BERT requires text data'}
                result = model.train(X, y, texts=texts)
            else:
                result = model.train(X, y)

            if result['success']:
                # Store model version info
                version = self._create_model_version(model_name)
                self.model_versions[model_name] = version

                # Auto-save if enabled
                if self.config['auto_save']:
                    self.save_model(model_name)

                # Evaluate performance
                if self.config['performance_tracking']:
                    performance = self.evaluate_model(model_name, X, y, **kwargs)
                    self.performance_history[model_name] = performance

            return result

        except Exception as e:
            self.logger.error(f"Error training {model_name}: {str(e)}")
            return {'success': False, 'error': str(e)}

    def predict(self, model_name: str, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions using a trained model

        Args:
            model_name: Name of the model to use
            X: Feature matrix
            **kwargs: Additional arguments

        Returns:
            Predictions array
        """
        if model_name not in self.models:
            raise ValueError(f'Model {model_name} not found')

        model = self.models[model_name]

        if not model.is_trained:
            # Try to load from disk
            if not self.load_model(model_name):
                raise ValueError(f'Model {model_name} not trained and could not be loaded')

        try:
            return model.predict(X)
        except Exception as e:
            self.logger.error(f"Error predicting with {model_name}: {str(e)}")

            # Try fallback if enabled
            if self.config['enable_fallback']:
                return self._fallback_predict(X, **kwargs)

            raise

    def predict_proba(self, model_name: str, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict probabilities using a trained model

        Args:
            model_name: Name of the model to use
            X: Feature matrix
            **kwargs: Additional arguments

        Returns:
            Prediction probabilities array
        """
        if model_name not in self.models:
            raise ValueError(f'Model {model_name} not found')

        model = self.models[model_name]

        if not model.is_trained:
            if not self.load_model(model_name):
                raise ValueError(f'Model {model_name} not trained and could not be loaded')

        try:
            return model.predict_proba(X)
        except Exception as e:
            self.logger.error(f"Error predicting probabilities with {model_name}: {str(e)}")

            # Try fallback
            if self.config['enable_fallback']:
                return self._fallback_predict_proba(X, **kwargs)

            raise

    def _fallback_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Fallback prediction using available models"""
        for fallback_model in self.fallback_models:
            try:
                if fallback_model in self.models and self.models[fallback_model].is_trained:
                    self.logger.info(f"Using fallback model: {fallback_model}")
                    return self.models[fallback_model].predict(X)
            except Exception as e:
                self.logger.warning(f"Fallback model {fallback_model} failed: {str(e)}")
                continue

        # Final fallback: return most common class or zeros
        self.logger.warning("All models failed, using default fallback")
        if hasattr(X, 'shape') and X.shape[0] > 0:
            return np.array(['outros'] * X.shape[0])
        return np.array(['outros'])

    def _fallback_predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Fallback probability prediction"""
        for fallback_model in self.fallback_models:
            try:
                if fallback_model in self.models and self.models[fallback_model].is_trained:
                    self.logger.info(f"Using fallback model for probabilities: {fallback_model}")
                    return self.models[fallback_model].predict_proba(X)
            except Exception as e:
                self.logger.warning(f"Fallback model {fallback_model} failed: {str(e)}")
                continue

        # Final fallback: uniform probabilities
        n_samples = X.shape[0] if hasattr(X, 'shape') else 1
        return np.full((n_samples, 10), 0.1)  # Assume 10 classes

    def evaluate_model(self, model_name: str, X: np.ndarray, y: np.ndarray,
                      X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Evaluate model performance

        Args:
            model_name: Name of the model to evaluate
            X: Training features
            y: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            **kwargs: Additional arguments

        Returns:
            Performance metrics dictionary
        """
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}

        model = self.models[model_name]

        if not model.is_trained:
            return {'error': f'Model {model_name} not trained'}

        try:
            # Use test data if provided, otherwise use training data
            eval_X = X_test if X_test is not None else X
            eval_y = y_test if y_test is not None else y

            predictions = model.predict(eval_X)

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(eval_y, predictions),
                'precision': precision_score(eval_y, predictions, average='weighted', zero_division=0),
                'recall': recall_score(eval_y, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(eval_y, predictions, average='weighted', zero_division=0),
                'classification_report': classification_report(eval_y, predictions, output_dict=True, zero_division=0),
                'confusion_matrix': confusion_matrix(eval_y, predictions).tolist(),
                'evaluation_timestamp': datetime.now().isoformat()
            }

            # Add probabilities if available
            try:
                probabilities = model.predict_proba(eval_X)
                if len(np.unique(eval_y)) == 2:  # Binary classification
                    metrics['auc_roc'] = roc_auc_score(eval_y, probabilities[:, 1])
            except:
                pass

            return metrics

        except Exception as e:
            self.logger.error(f"Error evaluating {model_name}: {str(e)}")
            return {'error': str(e)}

    def compare_models(self, X: np.ndarray, y: np.ndarray,
                      models_to_compare: Optional[List[str]] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Advanced model comparison using ModelSelector

        Args:
            X: Feature matrix
            y: Target labels
            models_to_compare: List of model names to compare
            **kwargs: Additional arguments

        Returns:
            Advanced comparison results with detailed benchmarking
        """
        try:
            self.logger.info(f"Using advanced ModelSelector for model comparison")

            # Analyze data characteristics
            data_characteristics = self.model_selector.analyze_data_characteristics(X, y)

            # Perform comprehensive comparison
            comparison_results = self.model_selector.compare_models_comprehensive(
                X, y, models_to_compare, data_characteristics, **kwargs
            )

            # Generate performance report
            report = self.model_selector.generate_performance_report(
                comparison_results, data_characteristics
            )

            # Convert results to expected format
            legacy_format = {
                'comparison_results': {},
                'best_model': comparison_results[0].model_name if comparison_results else None,
                'best_score': comparison_results[0].performance_metrics.f1_score if comparison_results else 0,
                'recommendation': report.get('summary', {}).get('best_model', 'No recommendation available'),
                'detailed_comparison': [result.to_dict() for result in comparison_results],
                'performance_report': report
            }

            # Add legacy format for backward compatibility
            for result in comparison_results:
                legacy_format['comparison_results'][result.model_name] = {
                    'performance': result.performance_metrics.to_dict(),
                    'rank': result.rank,
                    'recommendation_score': result.recommendation_score
                }

            return legacy_format

        except Exception as e:
            self.logger.error(f"Error in advanced model comparison: {str(e)}")
            # Fallback to simple comparison
            return self._simple_model_comparison(X, y, models_to_compare, **kwargs)

    def _simple_model_comparison(self, X: np.ndarray, y: np.ndarray,
                                models_to_compare: Optional[List[str]] = None,
                                **kwargs) -> Dict[str, Any]:
        """
        Simple fallback model comparison
        """
        if models_to_compare is None:
            models_to_compare = ['random_forest', 'xgboost', 'lightgbm']

        comparison_results = {}

        for model_name in models_to_compare:
            try:
                # Quick train and evaluate
                result = self.train_model(model_name, X, y, **kwargs)
                if result['success']:
                    performance = self.evaluate_model(model_name, X, y, **kwargs)
                    comparison_results[model_name] = {
                        'training_result': result,
                        'performance': performance
                    }
            except Exception as e:
                self.logger.warning(f"Error comparing {model_name}: {str(e)}")
                comparison_results[model_name] = {'error': str(e)}

        # Determine best model
        best_model = None
        best_score = 0

        for model_name, results in comparison_results.items():
            if 'performance' in results and 'f1_score' in results['performance']:
                score = results['performance']['f1_score']
                if score > best_score:
                    best_score = score
                    best_model = model_name

        return {
            'comparison_results': comparison_results,
            'best_model': best_model,
            'best_score': best_score,
            'recommendation': f"Recommended model: {best_model} (F1: {best_score:.4f})"
        }

    def optimize_hyperparameters(self, model_name: str, X: np.ndarray, y: np.ndarray,
                               n_trials: int = 50, **kwargs) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a model using Optuna

        Args:
            model_name: Name of the model to optimize
            X: Feature matrix
            y: Target labels
            n_trials: Number of optimization trials
            **kwargs: Additional arguments

        Returns:
            Optimization results
        """
        if model_name not in self.models:
            return {'success': False, 'error': f'Model {model_name} not found'}

        try:
            self.logger.info(f"Starting hyperparameter optimization for {model_name}")

            def objective(trial):
                return self._optimization_objective(trial, model_name, X, y)

            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner()
            )

            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            best_score = study.best_value

            # Train model with best parameters
            model = self.models[model_name]
            if hasattr(model, 'model') and model.model:
                model.model.set_params(**best_params)
                result = model.train(X, y)

                if result['success']:
                    # Save optimized model
                    self.models[f"{model_name}_optimized"] = model

            return {
                'success': True,
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': len(study.trials),
                'optimization_history': {
                    'best_score': best_score,
                    'best_params': best_params,
                    'n_trials': len(study.trials)
                }
            }

        except Exception as e:
            self.logger.error(f"Error optimizing {model_name}: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _optimization_objective(self, trial, model_name: str, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for hyperparameter optimization"""
        try:
            model = self.models[model_name]

            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            else:
                return 0.0

            # Create temporary model with suggested parameters
            temp_model = self.model_registry[model_name]({**self.config['model_configs'][model_name], **params})

            # Quick cross-validation
            scores = cross_val_score(temp_model.model, X, y, cv=3, scoring='f1_weighted')
            return np.mean(scores)

        except Exception as e:
            self.logger.warning(f"Error in optimization objective: {str(e)}")
            return 0.0

    def _create_model_version(self, model_name: str) -> Dict[str, Any]:
        """Create version information for a model"""
        timestamp = datetime.now().isoformat()
        version_hash = hashlib.md5(f"{model_name}_{timestamp}".encode()).hexdigest()[:8]

        return {
            'model_name': model_name,
            'version': version_hash,
            'timestamp': timestamp,
            'config': self.models[model_name].config
        }

    def save_model(self, model_name: str, custom_path: Optional[str] = None) -> bool:
        """
        Save a trained model to disk

        Args:
            model_name: Name of the model to save
            custom_path: Custom save path (optional)

        Returns:
            True if saved successfully
        """
        if model_name not in self.models:
            return False

        try:
            model_dir = custom_path or os.path.join(self.config['model_dir'], model_name)
            os.makedirs(model_dir, exist_ok=True)

            filepath = os.path.join(model_dir, 'model.pkl')
            success = self.models[model_name].save(filepath)

            if success and model_name in self.model_versions:
                # Save version info
                version_file = os.path.join(model_dir, 'version.json')
                with open(version_file, 'w') as f:
                    json.dump(self.model_versions[model_name], f, indent=2)

            return success

        except Exception as e:
            self.logger.error(f"Error saving {model_name}: {str(e)}")
            return False

    def load_model(self, model_name: str, custom_path: Optional[str] = None) -> bool:
        """
        Load a model from disk

        Args:
            model_name: Name of the model to load
            custom_path: Custom load path (optional)

        Returns:
            True if loaded successfully
        """
        if model_name not in self.models:
            return False

        try:
            model_dir = custom_path or os.path.join(self.config['model_dir'], model_name)
            filepath = os.path.join(model_dir, 'model.pkl')

            if not os.path.exists(filepath):
                return False

            success = self.models[model_name].load(filepath)

            if success:
                # Load version info if available
                version_file = os.path.join(model_dir, 'version.json')
                if os.path.exists(version_file):
                    with open(version_file, 'r') as f:
                        self.model_versions[model_name] = json.load(f)

            return success

        except Exception as e:
            self.logger.error(f"Error loading {model_name}: {str(e)}")
            return False

    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about models

        Args:
            model_name: Specific model name (optional)

        Returns:
            Model information dictionary
        """
        if model_name:
            if model_name not in self.models:
                return {'error': f'Model {model_name} not found'}

            model = self.models[model_name]
            info = model.get_model_info()
            info['version'] = self.model_versions.get(model_name)
            info['performance'] = self.performance_history.get(model_name)
            return info

        # Return info for all models
        all_info = {}
        for name, model in self.models.items():
            info = model.get_model_info()
            info['version'] = self.model_versions.get(name)
            info['performance'] = self.performance_history.get(name)
            all_info[name] = info

        return all_info

    def create_fallback_chain(self, primary_models: List[str]):
        """
        Create a fallback chain of models

        Args:
            primary_models: List of primary model names in order of preference
        """
        self.fallback_models = primary_models.copy()
        self.logger.info(f"Created fallback chain: {self.fallback_models}")

    def process_data(self, transactions: List[Dict], target_column: str = 'category') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Process raw transaction data into features and labels

        Args:
            transactions: List of transaction dictionaries
            target_column: Name of the target column

        Returns:
            Tuple of (features, labels, feature_names)
        """
        try:
            self.logger.info(f"Processing {len(transactions)} transactions")

            # Use feature engineer to create comprehensive features
            X, feature_names = self.feature_engineer.create_comprehensive_features(
                transactions, target_column=target_column
            )

            # Extract labels
            labels = [entry.get(target_column, 'outros') for entry in transactions]

            # Convert to numpy arrays
            if hasattr(X, 'toarray'):  # Sparse matrix
                X = X.toarray()

            y = np.array(labels)

            self.logger.info(f"Processed data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, feature_names

        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            return np.array([]), np.array([]), []

    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance information from feature engineer

        Returns:
            Feature importance dictionary
        """
        return self.feature_engineer.get_feature_importance()

    def get_augmentation_stats(self) -> Dict[str, Any]:
        """
        Get data augmentation statistics and metrics

        Returns:
            Augmentation statistics dictionary
        """
        return self.data_augmentation_pipeline.get_augmentation_stats()

    def compare_models_with_augmentation(self, transactions: List[Dict],
                                       target_column: str = 'category',
                                       models_to_compare: Optional[List[str]] = None,
                                       use_augmentation: bool = True,
                                       augmentation_config: Optional[Dict[str, Any]] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Compare models with integrated data augmentation

        Args:
            transactions: Raw transaction data
            target_column: Target column name
            models_to_compare: List of model names to compare
            use_augmentation: Whether to use data augmentation
            augmentation_config: Custom augmentation configuration
            **kwargs: Additional arguments

        Returns:
            Model comparison results with augmentation information
        """
        try:
            self.logger.info(f"Comparing models with augmentation (enabled: {use_augmentation})")

            # Apply data augmentation if enabled
            training_data = transactions
            augmentation_report = None

            if use_augmentation:
                if augmentation_config:
                    temp_pipeline = DataAugmentationPipeline(augmentation_config)
                    aug_data, augmentation_report = temp_pipeline.augment_dataset(transactions, 'transaction')
                else:
                    aug_data, augmentation_report = self.data_augmentation_pipeline.augment_dataset(transactions, 'transaction')

                training_data = aug_data.to_dict('records') if hasattr(aug_data, 'to_dict') else aug_data

            # Process data into features
            X, y, feature_names = self.process_data(training_data, target_column)

            if X.size == 0:
                return {'error': 'No valid data after processing'}

            # Extract texts for BERT if needed
            texts = None
            if models_to_compare and 'bert' in models_to_compare:
                texts = [entry.get('description', '') for entry in training_data]

            # Perform model comparison
            comparison_result = self.compare_models(X, y, models_to_compare, texts=texts, **kwargs)

            # Add augmentation information
            comparison_result['augmentation_info'] = {
                'augmentation_used': use_augmentation,
                'original_data_size': len(transactions),
                'augmented_data_size': len(training_data),
                'augmentation_report': augmentation_report
            }

            return comparison_result

        except Exception as e:
            self.logger.error(f"Error in model comparison with augmentation: {str(e)}")
            return {'error': str(e)}

    def augment_training_data(self, transactions: List[Dict],
                            data_type: str = 'transaction',
                            augmentation_config: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Augment training data using the data augmentation pipeline

        Args:
            transactions: Raw transaction data
            data_type: Type of data ('transaction', 'company_financial', 'mixed')
            augmentation_config: Custom augmentation configuration

        Returns:
            Tuple of (augmented_data, augmentation_report)
        """
        try:
            self.logger.info(f"Augmenting {len(transactions)} {data_type} records")

            if augmentation_config:
                temp_pipeline = DataAugmentationPipeline(augmentation_config)
                augmented_data, report = temp_pipeline.augment_dataset(transactions, data_type)
            else:
                augmented_data, report = self.data_augmentation_pipeline.augment_dataset(transactions, data_type)

            augmented_records = augmented_data.to_dict('records') if hasattr(augmented_data, 'to_dict') else augmented_data

            self.logger.info(f"Augmentation completed. Original: {len(transactions)}, Augmented: {len(augmented_records)}")
            return augmented_records, report

        except Exception as e:
            self.logger.error(f"Error augmenting training data: {str(e)}")
            return transactions, {'error': str(e)}

    # Advanced ModelSelector methods

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
        return self.model_selector.perform_ab_test(model_a, model_b, X, y, test_duration_hours, **kwargs)

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
        return self.model_selector.get_performance_history(model_name, days)

    def get_ab_test_results(self, test_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get A/B test results

        Args:
            test_id: Specific test ID (optional)

        Returns:
            A/B test results
        """
        return self.model_selector.get_ab_test_results(test_id)

    def analyze_data_characteristics(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Analyze data characteristics for model selection

        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Feature names

        Returns:
            Data characteristics analysis
        """
        return self.model_selector.analyze_data_characteristics(X, y, feature_names)

    def generate_performance_report(self, comparison_results: List,
                                  data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive performance report

        Args:
            comparison_results: Model comparison results
            data_characteristics: Data characteristics

        Returns:
            Performance report
        """
        # Convert legacy format to ModelComparisonResult if needed
        if comparison_results and not hasattr(comparison_results[0], 'performance_metrics'):
            # Convert from legacy format
            from src.services.model_selector import ModelComparisonResult, ModelPerformanceMetrics
            converted_results = []
            for i, result in enumerate(comparison_results):
                if isinstance(result, dict) and 'performance' in result:
                    perf_data = result['performance']
                    metrics = ModelPerformanceMetrics(
                        accuracy=perf_data.get('accuracy', 0),
                        precision=perf_data.get('precision', 0),
                        recall=perf_data.get('recall', 0),
                        f1_score=perf_data.get('f1_score', 0),
                        auc_roc=perf_data.get('auc_roc'),
                        cross_val_scores=perf_data.get('cross_val_scores'),
                        cross_val_mean=perf_data.get('cross_val_mean'),
                        cross_val_std=perf_data.get('cross_val_std'),
                        training_time=perf_data.get('training_time'),
                        prediction_time=perf_data.get('prediction_time'),
                        confusion_matrix=perf_data.get('confusion_matrix'),
                        classification_report=perf_data.get('classification_report')
                    )
                    converted_result = ModelComparisonResult(
                        model_name=result.get('model_name', f'model_{i}'),
                        performance_metrics=metrics,
                        rank=result.get('rank', i+1),
                        relative_score=perf_data.get('f1_score', 0),
                        recommendation_score=result.get('recommendation_score', 0),
                        data_characteristics=data_characteristics,
                        timestamp=datetime.now().isoformat()
                    )
                    converted_results.append(converted_result)
            comparison_results = converted_results

        return self.model_selector.generate_performance_report(comparison_results, data_characteristics)

    def export_comparison_report(self, comparison_results: List,
                               filepath: str) -> bool:
        """
        Export comparison results to file

        Args:
            comparison_results: Model comparison results
            filepath: Export file path

        Returns:
            True if export successful
        """
        # Convert to ModelComparisonResult format if needed
        if comparison_results and not hasattr(comparison_results[0], 'performance_metrics'):
            # Convert from legacy format
            from src.services.model_selector import ModelComparisonResult, ModelPerformanceMetrics
            converted_results = []
            for i, result in enumerate(comparison_results):
                if isinstance(result, dict) and 'performance' in result:
                    perf_data = result['performance']
                    metrics = ModelPerformanceMetrics(
                        accuracy=perf_data.get('accuracy', 0),
                        precision=perf_data.get('precision', 0),
                        recall=perf_data.get('recall', 0),
                        f1_score=perf_data.get('f1_score', 0),
                        auc_roc=perf_data.get('auc_roc'),
                        cross_val_scores=perf_data.get('cross_val_scores'),
                        cross_val_mean=perf_data.get('cross_val_mean'),
                        cross_val_std=perf_data.get('cross_val_std'),
                        training_time=perf_data.get('training_time'),
                        prediction_time=perf_data.get('prediction_time'),
                        confusion_matrix=perf_data.get('confusion_matrix'),
                        classification_report=perf_data.get('classification_report')
                    )
                    converted_result = ModelComparisonResult(
                        model_name=result.get('model_name', f'model_{i}'),
                        performance_metrics=metrics,
                        rank=result.get('rank', i+1),
                        relative_score=perf_data.get('f1_score', 0),
                        recommendation_score=result.get('recommendation_score', 0),
                        data_characteristics={},
                        timestamp=datetime.now().isoformat()
                    )
                    converted_results.append(converted_result)
            comparison_results = converted_results

        return self.model_selector.export_comparison_report(comparison_results, filepath)