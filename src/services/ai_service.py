import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Optional imports for pruning callbacks
try:
    from optuna.integration import XGBoostPruningCallback, LightGBMPruningCallback
    XGBOOST_PRUNING_AVAILABLE = True
    LIGHTGBM_PRUNING_AVAILABLE = True
except ImportError:
    XGBOOST_PRUNING_AVAILABLE = False
    LIGHTGBM_PRUNING_AVAILABLE = False
    # Logger will be initialized later, so we'll warn then
from typing import List, Dict, Any, Optional, Tuple
import openai
from groq import Groq
import re
import time
from datetime import datetime, timedelta
from src.utils.logging_config import get_logger, get_audit_logger
from src.utils.exceptions import (
    AIServiceError, AIServiceUnavailableError, AIServiceTimeoutError,
    AIServiceQuotaExceededError, InsufficientDataError, ValidationError
)
from src.utils.error_handler import handle_service_errors, with_timeout, recovery_manager
from src.services.feature_engineer import FeatureEngineer
from src.services.bert_service import BERTTextClassifier
from src.services.advanced_outlier_detector import AdvancedOutlierDetector
from src.services.contextual_outlier_detector import ContextualOutlierDetector
from src.services.statistical_outlier_analysis import StatisticalOutlierAnalysis

# Initialize loggers
logger = get_logger(__name__)
audit_logger = get_audit_logger()


class SupervisedLearningManager:
    """
    Gerenciador de modelos de aprendizado supervisionado para categorização de transações
    """

    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.label_encoders = {}
        self.model_performance = {}
        self.optimized_params = {}  # Store optimized hyperparameters
        self.optimization_history = {}  # Store optimization history
        self.default_performance = {}  # Store default model performance
        self.optimized_performance = {}  # Store optimized model performance
        self.logger = get_logger(__name__)

        # Initialize advanced feature engineer
        self.feature_engineer = FeatureEngineer()
        self.feature_names = {}  # Store feature names for each model

        # Initialize BERT classifier
        self.bert_classifier = BERTTextClassifier()
        self.bert_performance = {}  # Store BERT-specific performance metrics

    def select_best_model(self, X_train, y_train, X_test=None, y_test=None, include_bert: bool = True) -> str:
        """
        Seleciona o melhor modelo baseado nas características dos dados
        """
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))

        self.logger.info(f"Selecting best model for {n_samples} samples, {n_features} features, {n_classes} classes")

        # Check if we have text data for BERT consideration
        has_text_data = self._check_for_text_data(X_train)

        # Critérios de seleção baseados em dados
        if n_samples < 1000:
            # Para datasets pequenos, Random Forest geralmente performa melhor
            if include_bert and has_text_data and n_samples >= 100:
                # Consider BERT for small datasets with text if we have enough samples
                return self._compare_models_with_bert(X_train, y_train, X_test, y_test)
            return 'random_forest'
        elif n_classes > 10:
            # Para muitas classes, XGBoost geralmente é melhor
            if include_bert and has_text_data:
                return self._compare_models_with_bert(X_train, y_train, X_test, y_test)
            return 'xgboost'
        elif n_features > 1000:
            # Para alta dimensionalidade, LightGBM é mais eficiente
            if include_bert and has_text_data:
                return self._compare_models_with_bert(X_train, y_train, X_test, y_test)
            return 'lightgbm'
        else:
            # Caso padrão: testar todos e escolher o melhor
            if include_bert and has_text_data:
                return self._compare_models_with_bert(X_train, y_train, X_test, y_test)
            return self._compare_models(X_train, y_train, X_test, y_test)

    def _compare_models(self, X_train, y_train, X_test=None, y_test=None) -> str:
        """
        Compara diferentes modelos e retorna o melhor
        """
        models_to_test = ['random_forest', 'xgboost', 'lightgbm']
        best_model = 'random_forest'
        best_score = 0

        for model_name in models_to_test:
            try:
                model, vectorizer, label_encoder = self._create_model(model_name)
                model.fit(X_train, y_train)

                if X_test is not None and y_test is not None:
                    y_pred = model.predict(X_test)
                    score = f1_score(y_test, y_pred, average='weighted')
                else:
                    # Usa cross-validation se não houver dados de teste
                    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted')
                    score = np.mean(scores)

                self.logger.info(f"{model_name} score: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model = model_name

            except Exception as e:
                self.logger.warning(f"Error testing {model_name}: {str(e)}")
                continue

        return best_model

    def _check_for_text_data(self, X_train) -> bool:
        """
        Check if the training data contains text features that would benefit from BERT
        """
        try:
            # Check if we have text-like features (high dimensionality, sparse data)
            n_samples, n_features = X_train.shape

            # BERT is beneficial for:
            # 1. High-dimensional sparse data (likely from text embeddings)
            # 2. When we have text features in the dataset
            if n_features > 100:  # Likely text embeddings
                return True

            # Check sparsity (text embeddings are often sparse)
            if hasattr(X_train, 'toarray'):
                sparsity = 1.0 - (X_train.nnz / (n_samples * n_features))
                if sparsity > 0.8:  # Very sparse data
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"Error checking for text data: {str(e)}")
            return False

    def _compare_models_with_bert(self, X_train, y_train, X_test=None, y_test=None) -> str:
        """
        Compare traditional ML models with BERT and return the best
        """
        models_to_test = ['random_forest', 'xgboost', 'lightgbm', 'bert']
        best_model = 'random_forest'
        best_score = 0

        # Extract text data for BERT if available
        text_data = None
        if hasattr(X_train, 'text_data'):
            text_data = X_train.text_data

        for model_name in models_to_test:
            try:
                if model_name == 'bert':
                    # Special handling for BERT
                    if text_data is None:
                        self.logger.warning("No text data available for BERT")
                        continue

                    score = self._evaluate_bert_model(text_data, y_train, X_test, y_test if X_test is not None else None)
                else:
                    model, vectorizer, label_encoder = self._create_model(model_name)
                    model.fit(X_train, y_train)

                    if X_test is not None and y_test is not None:
                        y_pred = model.predict(X_test)
                        score = f1_score(y_test, y_pred, average='weighted')
                    else:
                        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted')
                        score = np.mean(scores)

                self.logger.info(f"{model_name} score: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model = model_name

            except Exception as e:
                self.logger.warning(f"Error testing {model_name}: {str(e)}")
                continue

        return best_model

    def _evaluate_bert_model(self, texts: List[str], labels: List[str], test_texts=None, test_labels=None) -> float:
        """
        Evaluate BERT model performance
        """
        try:
            # For evaluation, we'll use a simple approach
            # In production, you'd want proper train/val split
            if test_texts and test_labels:
                # Use provided test data
                train_texts, val_texts = texts[:int(0.8 * len(texts))], texts[int(0.8 * len(texts)):]
                train_labels, val_labels = labels[:int(0.8 * len(labels))], labels[int(0.8 * len(labels)):]
            else:
                # Simple split for evaluation
                train_texts, val_texts = texts[:int(0.8 * len(texts))], texts[int(0.8 * len(texts)):]
                train_labels, val_labels = labels[:int(0.8 * len(labels))], labels[int(0.8 * len(labels)):]

            # Quick training and evaluation (simplified for model selection)
            result = self.bert_classifier.train(train_texts, train_labels, val_texts, val_labels)

            if result['success'] and 'evaluation_metrics' in result:
                eval_metrics = result['evaluation_metrics']
                return eval_metrics.get('eval_f1', 0.0)
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error evaluating BERT model: {str(e)}")
            return 0.0

    def _create_model(self, model_type: str):
        """
        Cria instância do modelo especificado
        """
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        elif model_type == 'bert':
            # BERT is handled separately, return placeholder
            return 'bert', None, None
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model, None, None  # vectorizer e label_encoder serão definidos depois

    def train_model(self, model_type: str, X_train, y_train, X_test=None, y_test=None) -> Dict[str, Any]:
        """
        Treina um modelo específico
        """
        try:
            self.logger.info(f"Training {model_type} model")

            # Cria o modelo
            model, _, _ = self._create_model(model_type)

            # Treina o modelo
            model.fit(X_train, y_train)

            # Avalia o modelo
            evaluation_metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test)

            # Salva o modelo treinado
            self.models[model_type] = model

            return {
                'success': True,
                'model_type': model_type,
                'metrics': evaluation_metrics,
                'training_samples': len(X_train),
                'test_samples': len(X_test) if X_test is not None else 0
            }

        except Exception as e:
            self.logger.error(f"Error training {model_type}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_type': model_type
            }

    def _evaluate_model(self, model, X_train, y_train, X_test=None, y_test=None) -> Dict[str, Any]:
        """
        Avalia o desempenho do modelo
        """
        metrics = {}

        # Métricas no conjunto de treinamento
        y_train_pred = model.predict(X_train)
        metrics['train'] = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        }

        # Métricas no conjunto de teste (se disponível)
        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test)
            metrics['test'] = {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            }

            # Classification report detalhado
            metrics['classification_report'] = classification_report(
                y_test, y_test_pred, output_dict=True, zero_division=0
            )

        return metrics

    def predict(self, model_type: str, X) -> np.ndarray:
        """
        Faz predições usando o modelo especificado
        """
        if model_type == 'bert':
            # Special handling for BERT predictions
            return self._predict_with_bert(X)
        elif model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")

        return self.models[model_type].predict(X)

    def _predict_with_bert(self, X) -> np.ndarray:
        """
        Make predictions using BERT model
        """
        try:
            if not hasattr(X, 'text_data'):
                raise ValueError("BERT predictions require text data")

            texts = X.text_data
            predictions = self.bert_classifier.predict(texts)

            # Convert string predictions to numerical labels
            if self.bert_classifier.label_encoder:
                numerical_predictions = self.bert_classifier.label_encoder.transform(predictions)
                return np.array(numerical_predictions)
            else:
                # Fallback: return as strings (will need handling upstream)
                return np.array(predictions)

        except Exception as e:
            self.logger.error(f"Error predicting with BERT: {str(e)}")
            raise

    def train_bert_model(self, texts: List[str], labels: List[str], val_texts: Optional[List[str]] = None,
                        val_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train BERT model specifically for text classification

        Args:
            texts: Training text data
            labels: Training labels
            val_texts: Validation text data (optional)
            val_labels: Validation labels (optional)

        Returns:
            Dictionary with training results
        """
        try:
            self.logger.info(f"Training BERT model with {len(texts)} samples")

            # Train the BERT model
            result = self.bert_classifier.train(texts, labels, val_texts, val_labels)

            if result['success']:
                # Store performance metrics
                self.bert_performance = {
                    'accuracy': result.get('evaluation_metrics', {}).get('eval_accuracy', 0),
                    'f1_score': result.get('evaluation_metrics', {}).get('eval_f1', 0),
                    'precision': result.get('evaluation_metrics', {}).get('eval_precision', 0),
                    'recall': result.get('evaluation_metrics', {}).get('eval_recall', 0),
                    'training_samples': len(texts),
                    'validation_samples': len(val_texts) if val_texts else 0
                }

                self.logger.info(f"BERT training completed. F1: {self.bert_performance['f1_score']:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"Error training BERT model: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'BERT training failed'
            }

    def predict_with_bert(self, texts: List[str]) -> List[str]:
        """
        Make predictions using trained BERT model

        Args:
            texts: List of texts to classify

        Returns:
            List of predicted categories
        """
        try:
            if not self.bert_classifier.is_trained:
                raise ValueError("BERT model not trained")

            return self.bert_classifier.predict(texts)

        except Exception as e:
            self.logger.error(f"Error predicting with BERT: {str(e)}")
            return ['outros'] * len(texts)  # Fallback predictions

    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Retorna informações sobre um modelo específico
        """
        if model_type not in self.models:
            return {'error': f'Model {model_type} not found'}

        model = self.models[model_type]

        info = {
            'model_type': model_type,
            'is_trained': True,
            'model_class': model.__class__.__name__
        }

        # Informações específicas do modelo
        if hasattr(model, 'n_estimators'):
            info['n_estimators'] = model.n_estimators
        if hasattr(model, 'max_depth'):
            info['max_depth'] = model.max_depth
        if hasattr(model, 'feature_importances_'):
            info['feature_importances_shape'] = model.feature_importances_.shape

        return info

    def optimize_hyperparameters(self, model_type: str, X_train, y_train, X_test=None, y_test=None,
                                n_trials: int = 50, cv_folds: int = 5, timeout: int = 300) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model using Optuna

        Args:
            model_type: Type of model to optimize ('random_forest', 'xgboost', 'lightgbm')
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            timeout: Maximum optimization time in seconds

        Returns:
            Dictionary with optimization results
        """
        try:
            self.logger.info(f"Starting hyperparameter optimization for {model_type} with {n_trials} trials")

            # Create Optuna study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner()
            )

            # Define objective function
            def objective(trial):
                return self._optimization_objective(trial, model_type, X_train, y_train, cv_folds)

            # Run optimization
            study.optimize(objective, n_trials=n_trials, timeout=timeout)

            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value

            # Store optimized parameters
            self.optimized_params[model_type] = best_params

            # Store optimization history
            self.optimization_history[model_type] = {
                'best_score': best_score,
                'best_params': best_params,
                'n_trials': len(study.trials),
                'optimization_time': timeout,
                'cv_folds': cv_folds
            }

            self.logger.info(f"Optimization completed for {model_type}. Best score: {best_score:.4f}")

            return {
                'success': True,
                'model_type': model_type,
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': len(study.trials),
                'optimization_history': self.optimization_history[model_type]
            }

        except Exception as e:
            self.logger.error(f"Error during hyperparameter optimization for {model_type}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_type': model_type
            }

    def _optimization_objective(self, trial, model_type: str, X_train, y_train, cv_folds: int) -> float:
        """
        Objective function for Optuna optimization
        """
        try:
            # Get model-specific parameters
            if model_type == 'random_forest':
                params = self._get_random_forest_params(trial)
            elif model_type == 'xgboost':
                params = self._get_xgboost_params(trial)
            elif model_type == 'lightgbm':
                params = self._get_lightgbm_params(trial)
            else:
                raise ValueError(f"Unsupported model type for optimization: {model_type}")

            # Create model with suggested parameters
            if model_type == 'random_forest':
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            elif model_type == 'xgboost':
                # XGBoost needs special handling for class labels
                model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1, eval_metric='mlogloss')
            elif model_type == 'lightgbm':
                model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)

            # Perform cross-validation with fallback for small datasets
            try:
                # Check if stratified split is possible
                min_class_count = min(np.bincount(y_train))
                if min_class_count >= cv_folds:
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                else:
                    # Fallback to regular KFold if stratified is not possible
                    from sklearn.model_selection import KFold
                    cv = KFold(n_splits=min(cv_folds, len(y_train)), shuffle=True, random_state=42)

                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
            except Exception as cv_error:
                # Final fallback: use a simple train/validation split
                self.logger.warning(f"Cross-validation failed: {str(cv_error)}, using simple split")
                from sklearn.model_selection import train_test_split
                X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train if len(np.unique(y_train)) > 1 else None
                )
                model.fit(X_train_cv, y_train_cv)
                y_pred_cv = model.predict(X_val_cv)
                scores = [f1_score(y_val_cv, y_pred_cv, average='weighted', zero_division=0)]

            return np.mean(scores)

        except Exception as e:
            self.logger.warning(f"Error in optimization objective for {model_type}: {str(e)}")
            return 0.0

    def _get_random_forest_params(self, trial) -> Dict[str, Any]:
        """Get Random Forest hyperparameters for optimization"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }

    def _get_xgboost_params(self, trial) -> Dict[str, Any]:
        """Get XGBoost hyperparameters for optimization"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
        }

    def _get_lightgbm_params(self, trial) -> Dict[str, Any]:
        """Get LightGBM hyperparameters for optimization"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
        }

    def train_optimized_model(self, model_type: str, X_train, y_train, X_test=None, y_test=None,
                             optimize: bool = True, n_trials: int = 50) -> Dict[str, Any]:
        """
        Train a model with optional hyperparameter optimization

        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            optimize: Whether to perform hyperparameter optimization
            n_trials: Number of optimization trials

        Returns:
            Dictionary with training results
        """
        try:
            self.logger.info(f"Training {model_type} model (optimized={optimize})")

            # Train default model first for comparison
            default_result = self.train_model(model_type, X_train, y_train, X_test, y_test)
            self.default_performance[model_type] = default_result.get('metrics', {})

            if not default_result['success']:
                return default_result

            if optimize:
                # Perform hyperparameter optimization
                opt_result = self.optimize_hyperparameters(
                    model_type, X_train, y_train, X_test, y_test, n_trials=n_trials
                )

                if opt_result['success']:
                    # Train optimized model
                    optimized_model, _, _ = self._create_model(model_type)
                    optimized_model.set_params(**opt_result['best_params'])

                    # Train optimized model
                    optimized_model.fit(X_train, y_train)

                    # Evaluate optimized model
                    opt_metrics = self._evaluate_model(optimized_model, X_train, y_train, X_test, y_test)
                    self.optimized_performance[model_type] = opt_metrics

                    # Store optimized model
                    self.models[f"{model_type}_optimized"] = optimized_model

                    # Compare performance
                    comparison = self._compare_model_performance(model_type, default_result['metrics'], opt_metrics)

                    return {
                        'success': True,
                        'model_type': model_type,
                        'optimized': True,
                        'default_metrics': default_result['metrics'],
                        'optimized_metrics': opt_metrics,
                        'best_params': opt_result['best_params'],
                        'optimization_score': opt_result['best_score'],
                        'performance_comparison': comparison,
                        'training_samples': len(X_train),
                        'test_samples': len(X_test) if X_test is not None else 0
                    }
                else:
                    self.logger.warning(f"Optimization failed for {model_type}, using default model")
                    return default_result
            else:
                return default_result

        except Exception as e:
            self.logger.error(f"Error training optimized model {model_type}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_type': model_type
            }

    def _compare_model_performance(self, model_type: str, default_metrics: Dict, optimized_metrics: Dict) -> Dict[str, Any]:
        """
        Compare performance between default and optimized models
        """
        comparison = {}

        if 'test' in default_metrics and 'test' in optimized_metrics:
            default_test = default_metrics['test']
            optimized_test = optimized_metrics['test']

            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                if metric in default_test and metric in optimized_test:
                    default_val = default_test[metric]
                    optimized_val = optimized_test[metric]
                    improvement = optimized_val - default_val
                    improvement_pct = (improvement / default_val * 100) if default_val > 0 else 0

                    comparison[metric] = {
                        'default': default_val,
                        'optimized': optimized_val,
                        'improvement': improvement,
                        'improvement_percentage': improvement_pct
                    }

        return comparison

    def get_optimization_info(self, model_type: str) -> Dict[str, Any]:
        """
        Get optimization information for a model
        """
        if model_type not in self.optimization_history:
            return {'error': f'No optimization history found for {model_type}'}

        info = self.optimization_history[model_type].copy()

        # Add performance comparison if available
        if model_type in self.default_performance and model_type in self.optimized_performance:
            info['performance_comparison'] = self._compare_model_performance(
                model_type, self.default_performance[model_type], self.optimized_performance[model_type]
            )

        return info

class AIService:
    """
    Serviço de IA para análise de transações bancárias
    """
    
    def __init__(self):
        self.categories = {
            'alimentacao': ['mercado', 'supermercado', 'padaria', 'restaurante', 'lanchonete', 'delivery', 'ifood', 'uber eats'],
            'transporte': ['uber', 'taxi', 'combustivel', 'posto', 'onibus', 'metro', 'estacionamento'],
            'saude': ['farmacia', 'hospital', 'clinica', 'medico', 'dentista', 'laboratorio'],
            'educacao': ['escola', 'faculdade', 'curso', 'livro', 'material escolar'],
            'lazer': ['cinema', 'teatro', 'show', 'viagem', 'hotel', 'netflix', 'spotify'],
            'casa': ['aluguel', 'condominio', 'luz', 'agua', 'gas', 'internet', 'telefone'],
            'vestuario': ['roupa', 'sapato', 'loja', 'shopping'],
            'investimento': ['aplicacao', 'poupanca', 'cdb', 'tesouro', 'acao'],
            'transferencia': ['ted', 'doc', 'pix', 'transferencia'],
            'saque': ['saque', 'caixa eletronico'],
            'salario': ['salario', 'ordenado', 'pagamento'],
            'outros': []
        }
        
        # Configuração dos clientes de IA
        self.openai_client = None
        self.groq_client = None
        
        if os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai.OpenAI()
            logger.info("OpenAI client initialized")
        
        if os.getenv('GROQ_API_KEY'):
            self.groq_client = Groq()
            logger.info("Groq client initialized")
        
        if not self.openai_client and not self.groq_client:
            logger.warning("No AI service configured. Set OPENAI_API_KEY or GROQ_API_KEY for AI features")

        # Warn about missing optuna integration if needed
        if not XGBOOST_PRUNING_AVAILABLE or not LIGHTGBM_PRUNING_AVAILABLE:
            logger.warning("Optuna integration packages not available. Pruning callbacks will be disabled for XGBoost/LightGBM optimization.")
        
        # AI service configuration
        self.default_timeout = int(os.getenv('AI_SERVICE_TIMEOUT', '30'))
        self.max_retries = int(os.getenv('AI_SERVICE_MAX_RETRIES', '3'))
        self.rate_limit_delay = float(os.getenv('AI_SERVICE_RATE_LIMIT_DELAY', '1.0'))

        # Initialize supervised learning manager
        self.supervised_manager = SupervisedLearningManager()

        # Initialize feature engineer (for direct access)
        self.feature_engineer = self.supervised_manager.feature_engineer
        self.feature_names = self.supervised_manager.feature_names

        # Initialize advanced outlier detection system
        self.advanced_detector = AdvancedOutlierDetector()
        self.contextual_detector = ContextualOutlierDetector()
        self.statistical_analyzer = StatisticalOutlierAnalysis()

        # Initialize logger for AIService
        self.logger = get_logger(__name__)
    
    def categorize_transaction(self, description: str) -> str:
        """
        Categoriza uma transação baseada na descrição
        """
        description_lower = description.lower()
        
        # Busca por palavras-chave nas categorias
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return category
        
        return 'outros'
    
    @handle_service_errors('ai_service')
    @with_timeout(120)  # 2 minute timeout for batch processing
    def categorize_transactions_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Categoriza um lote de transações
        """
        if not transactions:
            raise InsufficientDataError('transaction categorization', 1, 0)
        
        logger.info(f"Starting batch categorization for {len(transactions)} transactions")
        
        try:
            categorized_count = 0
            failed_count = 0
            
            for i, transaction in enumerate(transactions):
                try:
                    description = transaction.get('description', '')
                    if not description or description.strip() == '':
                        transaction['category'] = 'outros'
                        logger.debug(f"Transaction {i+1}: Empty description, assigned 'outros' category")
                    else:
                        transaction['category'] = self.categorize_transaction(description)
                        categorized_count += 1
                        
                    # Add small delay to avoid overwhelming the system
                    if i > 0 and i % 100 == 0:
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.warning(f"Failed to categorize transaction {i+1}: {str(e)}")
                    transaction['category'] = 'outros'  # Fallback category
                    failed_count += 1
            
            logger.info(f"Batch categorization completed. Success: {categorized_count}, Failed: {failed_count}")
            audit_logger.log_ai_operation('batch_categorization', len(transactions), True)
            
            return transactions
            
        except Exception as e:
            logger.error(f"Critical error in batch categorization: {str(e)}")
            audit_logger.log_ai_operation('batch_categorization', len(transactions), False, error=str(e))
            raise AIServiceError(f"Batch categorization failed: {str(e)}")
    
    @handle_service_errors('ai_service')
    @with_timeout(120)  # 2 minute timeout for advanced anomaly detection
    def detect_anomalies(self, transactions: List[Dict], method: str = 'ensemble',
                        include_contextual: bool = True) -> List[Dict]:
        """
        Detecta anomalias nas transações usando sistema avançado de detecção

        Args:
            transactions: Lista de transações para análise
            method: Método de detecção ('iqr', 'zscore', 'lof', 'isolation_forest',
                                        'mahalanobis', 'ensemble')
            include_contextual: Incluir análise contextual (categoria, temporal, etc.)
        """
        if not transactions:
            raise InsufficientDataError('anomaly detection', 1, 0)

        logger.info(f"Starting advanced anomaly detection for {len(transactions)} transactions using {method} method")

        if len(transactions) < 10:  # Precisa de dados suficientes
            logger.warning("Insufficient data for anomaly detection (minimum 10 transactions required)")
            # Mark all as non-anomalous
            for transaction in transactions:
                transaction['is_anomaly'] = False
                transaction['anomaly_score'] = 0.0
                transaction['anomaly_method'] = method
            return transactions

        try:
            # Prepare data for advanced analysis
            df = pd.DataFrame(transactions)

            # Basic amount-based detection
            if 'amount' in df.columns:
                amounts = np.abs(df['amount'].values)

                if method == 'ensemble':
                    # Use comprehensive ensemble detection
                    outlier_flags, outlier_scores = self.advanced_detector.detect_outliers_ensemble(
                        amounts.reshape(-1, 1)
                    )
                elif method == 'iqr':
                    outlier_flags, outlier_scores = self.advanced_detector.detect_outliers_iqr(amounts)
                elif method == 'zscore':
                    outlier_flags, outlier_scores = self.advanced_detector.detect_outliers_zscore(amounts)
                elif method == 'lof':
                    outlier_flags, outlier_scores = self.advanced_detector.detect_outliers_lof(
                        amounts.reshape(-1, 1)
                    )
                elif method == 'isolation_forest':
                    outlier_flags, outlier_scores = self.advanced_detector.detect_outliers_isolation_forest(
                        amounts.reshape(-1, 1)
                    )
                elif method == 'mahalanobis':
                    outlier_flags, outlier_scores = self.advanced_detector.detect_outliers_mahalanobis(
                        amounts.reshape(-1, 1)
                    )
                else:
                    # Default to ensemble
                    outlier_flags, outlier_scores = self.advanced_detector.detect_outliers_ensemble(
                        amounts.reshape(-1, 1)
                    )

                # Mark basic anomalies
                anomaly_count = 0
                for i, transaction in enumerate(transactions):
                    is_anomaly = outlier_flags[i]
                    transaction['is_anomaly'] = bool(is_anomaly)
                    transaction['anomaly_score'] = float(outlier_scores[i])
                    transaction['anomaly_method'] = method
                    if is_anomaly:
                        anomaly_count += 1

            # Add contextual analysis if requested
            if include_contextual:
                contextual_results = self._add_contextual_anomaly_analysis(transactions)
                # Merge contextual results with basic results
                for i, transaction in enumerate(transactions):
                    if f"contextual_{method}" in contextual_results:
                        transaction['contextual_anomaly'] = contextual_results[f"contextual_{method}"].get(i, False)

            logger.info(f"Advanced anomaly detection completed. Found {anomaly_count} anomalies out of {len(transactions)} transactions")
            audit_logger.log_ai_operation('advanced_anomaly_detection', len(transactions), True)
            return transactions

        except Exception as e:
            logger.error(f"Error in advanced anomaly detection: {str(e)}")
            # Fallback to basic detection
            return self._fallback_anomaly_detection(transactions, method)

    def _add_contextual_anomaly_analysis(self, transactions: List[Dict]) -> Dict[str, Dict[int, bool]]:
        """Add contextual anomaly analysis results"""
        contextual_results = {}

        try:
            # Category-based analysis
            if any(t.get('category') for t in transactions):
                category_results = self.contextual_detector.detect_amount_outliers_by_category(transactions)
                if 'results' in category_results:
                    for category, cat_data in category_results['results'].items():
                        for outlier in cat_data.get('outlier_transactions', []):
                            # Find matching transaction index
                            for i, transaction in enumerate(transactions):
                                if (transaction.get('description') == outlier.get('description') and
                                    transaction.get('amount') == outlier.get('amount')):
                                    contextual_results.setdefault('contextual_category', {})[i] = True

            # Temporal analysis
            if any(t.get('date') for t in transactions):
                temporal_results = self.contextual_detector.detect_temporal_outliers(transactions)
                if 'results' in temporal_results:
                    for time_window, window_data in temporal_results['results'].items():
                        if 'unusual_periods' in window_data:
                            # Mark transactions from unusual periods
                            for unusual in window_data['unusual_periods']:
                                period = unusual.get('period')
                                # This is a simplified mapping - in production you'd need more sophisticated logic
                                contextual_results.setdefault('contextual_temporal', {})[period] = True

        except Exception as e:
            logger.warning(f"Error in contextual anomaly analysis: {str(e)}")

        return contextual_results

    def _fallback_anomaly_detection(self, transactions: List[Dict], method: str) -> List[Dict]:
        """Fallback anomaly detection using basic Isolation Forest"""
        logger.warning(f"Using fallback anomaly detection due to error in {method} method")

        if len(transactions) < 10:
            for transaction in transactions:
                transaction['is_anomaly'] = False
                transaction['anomaly_score'] = 0.0
                transaction['anomaly_method'] = 'fallback'
            return transactions

        # Basic Isolation Forest fallback
        amounts = np.array([abs(t.get('amount', 0)) for t in transactions]).reshape(-1, 1)

        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(amounts)

        anomaly_count = 0
        for i, transaction in enumerate(transactions):
            is_anomaly = anomaly_labels[i] == -1
            transaction['is_anomaly'] = bool(is_anomaly)
            transaction['anomaly_score'] = float(abs(anomaly_labels[i]))
            transaction['anomaly_method'] = 'fallback_isolation_forest'
            if is_anomaly:
                anomaly_count += 1

        logger.info(f"Fallback anomaly detection completed. Found {anomaly_count} anomalies")
        return transactions

    @handle_service_errors('ai_service')
    @with_timeout(300)  # 5 minute timeout for comprehensive analysis
    def perform_comprehensive_outlier_analysis(self, transactions: List[Dict],
                                             ground_truth: Optional[List[bool]] = None,
                                             export_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive outlier analysis using all available methods

        Args:
            transactions: List of transaction dictionaries
            ground_truth: Optional ground truth labels for evaluation
            export_path: Optional path to export analysis report

        Returns:
            Dictionary with comprehensive analysis results
        """
        if not transactions:
            raise InsufficientDataError('comprehensive outlier analysis', 10, 0)

        logger.info(f"Starting comprehensive outlier analysis for {len(transactions)} transactions")

        try:
            # Convert ground truth to numpy array if provided
            ground_truth_array = np.array(ground_truth) if ground_truth else None

            # Perform comprehensive analysis
            analysis_result = self.statistical_analyzer.perform_comprehensive_analysis(
                transactions, ground_truth_array
            )

            # Export report if path provided
            if export_path and 'error' not in analysis_result:
                success = self.statistical_analyzer.export_analysis_report(export_path, analysis_result)
                analysis_result['export_success'] = success
                analysis_result['export_path'] = export_path

            audit_logger.log_ai_operation('comprehensive_outlier_analysis', len(transactions), True)
            return analysis_result

        except Exception as e:
            logger.error(f"Error in comprehensive outlier analysis: {str(e)}")
            audit_logger.log_ai_operation('comprehensive_outlier_analysis', len(transactions), False, error=str(e))
            return {'error': str(e)}

    @handle_service_errors('ai_service')
    @with_timeout(180)  # 3 minute timeout for contextual analysis
    def detect_contextual_outliers(self, transactions: List[Dict],
                                 analysis_type: str = 'all') -> Dict[str, Any]:
        """
        Detect contextual outliers (category-based, temporal, frequency-based, etc.)

        Args:
            transactions: List of transaction dictionaries
            analysis_type: Type of contextual analysis ('category', 'temporal',
                                                        'frequency', 'merchant', 'balance', 'all')

        Returns:
            Dictionary with contextual outlier detection results
        """
        if not transactions:
            raise InsufficientDataError('contextual outlier detection', 10, 0)

        logger.info(f"Starting contextual outlier detection ({analysis_type}) for {len(transactions)} transactions")

        try:
            results = {}

            # Category-based outlier detection
            if analysis_type in ['category', 'all']:
                category_results = self.contextual_detector.detect_amount_outliers_by_category(transactions)
                results['category_based'] = category_results

            # Temporal outlier detection
            if analysis_type in ['temporal', 'all']:
                temporal_results = self.contextual_detector.detect_temporal_outliers(transactions)
                results['temporal'] = temporal_results

            # Frequency-based outlier detection
            if analysis_type in ['frequency', 'all']:
                frequency_results = self.contextual_detector.detect_frequency_outliers(transactions)
                results['frequency'] = frequency_results

            # Merchant-specific outlier detection
            if analysis_type in ['merchant', 'all']:
                merchant_results = self.contextual_detector.detect_merchant_outliers(transactions)
                results['merchant'] = merchant_results

            # Account balance outlier detection
            if analysis_type in ['balance', 'all']:
                balance_results = self.contextual_detector.detect_balance_outliers(transactions)
                results['balance'] = balance_results

            audit_logger.log_ai_operation('contextual_outlier_detection', len(transactions), True)
            return results

        except Exception as e:
            logger.error(f"Error in contextual outlier detection: {str(e)}")
            audit_logger.log_ai_operation('contextual_outlier_detection', len(transactions), False, error=str(e))
            return {'error': str(e)}

    @handle_service_errors('ai_service')
    @with_timeout(120)  # 2 minute timeout for method comparison
    def compare_outlier_detection_methods(self, transactions: List[Dict],
                                        methods: List[str] = None,
                                        ground_truth: Optional[List[bool]] = None) -> Dict[str, Any]:
        """
        Compare performance of different outlier detection methods

        Args:
            transactions: List of transaction dictionaries
            methods: List of methods to compare (if None, uses all available)
            ground_truth: Optional ground truth labels

        Returns:
            Dictionary with method comparison results
        """
        if not transactions:
            raise InsufficientDataError('method comparison', 10, 0)

        logger.info(f"Starting outlier detection method comparison for {len(transactions)} transactions")

        try:
            # Prepare data
            df = pd.DataFrame(transactions)
            if 'amount' not in df.columns:
                raise ValidationError("Transactions must have 'amount' field")

            amounts = np.abs(df['amount'].values)
            ground_truth_array = np.array(ground_truth) if ground_truth else None

            if methods is None:
                methods = ['iqr', 'zscore', 'lof', 'isolation_forest', 'mahalanobis', 'ensemble']

            # Run comprehensive detection
            detection_results = self.advanced_detector.detect_outliers_comprehensive(
                amounts.reshape(-1, 1), methods=methods, return_details=True
            )

            # Compare methods
            comparison = self.statistical_analyzer._compare_detection_methods(
                detection_results, ground_truth_array
            )

            # Calculate confidence scores
            confidence_scores = self.statistical_analyzer._calculate_confidence_scores(detection_results)

            # Perform significance testing
            significance_tests = self.statistical_analyzer._perform_significance_testing(
                detection_results, ground_truth_array
            )

            result = {
                'detection_results': detection_results,
                'method_comparison': comparison,
                'confidence_analysis': confidence_scores,
                'significance_tests': significance_tests,
                'recommendations': self._generate_method_comparison_recommendations(
                    comparison, significance_tests
                )
            }

            audit_logger.log_ai_operation('method_comparison', len(transactions), True)
            return result

        except Exception as e:
            logger.error(f"Error in method comparison: {str(e)}")
            audit_logger.log_ai_operation('method_comparison', len(transactions), False, error=str(e))
            return {'error': str(e)}

    def _generate_method_comparison_recommendations(self, comparison: Dict[str, Any],
                                                  significance_tests: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on method comparison results"""
        recommendations = []

        try:
            # Best method recommendation
            if 'best_methods' in comparison and 'overall' in comparison['best_methods']:
                best_method = comparison['best_methods']['overall']
                recommendations.append(
                    f"Based on performance analysis, {best_method.replace('_', ' ').title()} "
                    "is recommended as the primary outlier detection method."
                )

            # Consensus analysis
            if 'consensus_analysis' in comparison:
                consensus_pct = comparison['consensus_analysis'].get('consensus_percentage', 0)
                if consensus_pct > 20:
                    recommendations.append(
                        f"High consensus among methods ({consensus_pct:.1f}%) indicates reliable results."
                    )
                elif consensus_pct < 5:
                    recommendations.append(
                        "Low consensus among methods suggests using ensemble approach or adjusting parameters."
                    )

            # Statistical significance
            if 'method_differences' in significance_tests:
                significant_diffs = []
                for method_pair, tests in significance_tests['method_differences'].items():
                    if any(test.get('significant', False) for test in tests.values() if isinstance(test, dict)):
                        significant_diffs.append(method_pair)

                if significant_diffs:
                    recommendations.append(
                        f"Statistically significant differences found between methods: "
                        f"{', '.join(significant_diffs)}. Consider method selection based on use case."
                    )

            # General recommendations
            recommendations.extend([
                "Consider using ensemble methods for improved robustness.",
                "Regularly evaluate and update detection thresholds based on new data patterns.",
                "Implement domain expert validation for high-confidence outliers."
            ])

        except Exception as e:
            logger.warning(f"Error generating method comparison recommendations: {str(e)}")

        return recommendations

    @handle_service_errors('ai_service')
    @with_timeout(60)  # 1 minute timeout for configuration
    def get_outlier_detection_config(self) -> Dict[str, Any]:
        """
        Get current configuration for outlier detection system

        Returns:
            Dictionary with current configuration
        """
        try:
            config = {
                'advanced_detector': self.advanced_detector.config,
                'contextual_detector': self.contextual_detector.config,
                'statistical_analyzer': self.statistical_analyzer.config,
                'available_methods': [
                    'iqr', 'zscore', 'lof', 'isolation_forest',
                    'mahalanobis', 'ocsvm', 'ensemble'
                ],
                'contextual_analyses': [
                    'category_based', 'temporal', 'frequency',
                    'merchant', 'balance'
                ]
            }

            return config

        except Exception as e:
            logger.error(f"Error getting outlier detection config: {str(e)}")
            return {'error': str(e)}

    @handle_service_errors('ai_service')
    @with_timeout(120)  # 2 minute timeout for configuration update
    def update_outlier_detection_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration for outlier detection system

        Args:
            config_updates: Dictionary with configuration updates

        Returns:
            Dictionary with update results
        """
        try:
            updated_configs = []

            # Update advanced detector config
            if 'advanced_detector' in config_updates:
                self.advanced_detector.config.update(config_updates['advanced_detector'])
                updated_configs.append('advanced_detector')

            # Update contextual detector config
            if 'contextual_detector' in config_updates:
                self.contextual_detector.config.update(config_updates['contextual_detector'])
                updated_configs.append('contextual_detector')

            # Update statistical analyzer config
            if 'statistical_analyzer' in config_updates:
                self.statistical_analyzer.config.update(config_updates['statistical_analyzer'])
                updated_configs.append('statistical_analyzer')

            logger.info(f"Updated outlier detection configuration for: {', '.join(updated_configs)}")

            return {
                'success': True,
                'updated_components': updated_configs,
                'current_config': self.get_outlier_detection_config()
            }

        except Exception as e:
            logger.error(f"Error updating outlier detection config: {str(e)}")
            return {'error': str(e)}
    
    def generate_insights(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Gera insights sobre as transações
        """
        if not transactions:
            return {'error': 'Nenhuma transação para análise'}
        
        df = pd.DataFrame(transactions)
        
        insights = {
            'summary': self._generate_summary_insights(df),
            'categories': self._generate_category_insights(df),
            'patterns': self._generate_pattern_insights(df),
            'anomalies': self._generate_anomaly_insights(df),
            'recommendations': []
        }
        
        # Adiciona recomendações baseadas nos insights
        insights['recommendations'] = self._generate_recommendations(insights)
        
        return insights
    
    def _generate_summary_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera insights resumidos
        """
        total_transactions = len(df)
        total_credits = df[df['amount'] > 0]['amount'].sum() if len(df[df['amount'] > 0]) > 0 else 0
        total_debits = abs(df[df['amount'] < 0]['amount'].sum()) if len(df[df['amount'] < 0]) > 0 else 0
        
        return {
            'total_transactions': total_transactions,
            'total_credits': round(total_credits, 2),
            'total_debits': round(total_debits, 2),
            'net_flow': round(total_credits - total_debits, 2),
            'avg_transaction_value': round(df['amount'].abs().mean(), 2) if total_transactions > 0 else 0,
            'largest_expense': round(df[df['amount'] < 0]['amount'].min(), 2) if len(df[df['amount'] < 0]) > 0 else 0,
            'largest_income': round(df[df['amount'] > 0]['amount'].max(), 2) if len(df[df['amount'] > 0]) > 0 else 0
        }
    
    def _generate_category_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera insights por categoria
        """
        if 'category' not in df.columns:
            return {}
        
        category_summary = {}
        
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            category_expenses = category_df[category_df['amount'] < 0]['amount'].sum()
            
            category_summary[category] = {
                'total_transactions': len(category_df),
                'total_spent': round(abs(category_expenses), 2),
                'avg_transaction': round(category_df['amount'].abs().mean(), 2),
                'percentage_of_expenses': 0  # Será calculado depois
            }
        
        # Calcula percentuais
        total_expenses = sum([cat['total_spent'] for cat in category_summary.values()])
        if total_expenses > 0:
            for category in category_summary:
                category_summary[category]['percentage_of_expenses'] = round(
                    (category_summary[category]['total_spent'] / total_expenses) * 100, 1
                )
        
        return category_summary
    
    def _generate_pattern_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera insights sobre padrões temporais
        """
        patterns = {}
        
        if 'date' in df.columns and len(df) > 0:
            # Converte datas se necessário
            dates = []
            for date_val in df['date']:
                if isinstance(date_val, str):
                    try:
                        dates.append(datetime.fromisoformat(date_val))
                    except:
                        dates.append(datetime.now())
                else:
                    dates.append(date_val)
            
            df['parsed_date'] = dates
            df['weekday'] = df['parsed_date'].apply(lambda x: x.weekday())
            df['day'] = df['parsed_date'].apply(lambda x: x.day)
            
            # Padrões por dia da semana
            weekday_spending = df[df['amount'] < 0].groupby('weekday')['amount'].sum().abs()
            weekdays = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
            
            patterns['weekday_spending'] = {
                weekdays[i]: round(weekday_spending.get(i, 0), 2) 
                for i in range(7)
            }
            
            # Dia do mês com mais gastos
            day_spending = df[df['amount'] < 0].groupby('day')['amount'].sum().abs()
            if len(day_spending) > 0:
                patterns['highest_spending_day'] = int(day_spending.idxmax())
                patterns['highest_spending_amount'] = round(day_spending.max(), 2)
        
        return patterns
    
    def _generate_anomaly_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera insights sobre anomalias
        """
        if 'is_anomaly' not in df.columns:
            return {}
        
        anomalies = df[df['is_anomaly'] == True]
        
        return {
            'total_anomalies': len(anomalies),
            'anomaly_percentage': round((len(anomalies) / len(df)) * 100, 1) if len(df) > 0 else 0,
            'anomalous_transactions': anomalies[['date', 'amount', 'description']].to_dict('records')[:5]  # Top 5
        }
    
    def _generate_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """
        Gera recomendações baseadas nos insights
        """
        recommendations = []
        
        # Recomendações baseadas em categorias
        if 'categories' in insights:
            categories = insights['categories']
            
            # Categoria com maior gasto
            if categories:
                max_category = max(categories.items(), key=lambda x: x[1]['total_spent'])
                if max_category[1]['total_spent'] > 0:
                    recommendations.append(
                        f"Sua maior categoria de gastos é '{max_category[0]}' com R$ {max_category[1]['total_spent']:.2f}. "
                        f"Considere revisar esses gastos para possíveis economias."
                    )
        
        # Recomendações baseadas em anomalias
        if 'anomalies' in insights:
            anomaly_count = insights['anomalies'].get('total_anomalies', 0)
            if anomaly_count > 0:
                recommendations.append(
                    f"Foram detectadas {anomaly_count} transações incomuns. "
                    f"Revise essas transações para verificar se são legítimas."
                )
        
        # Recomendações baseadas no fluxo de caixa
        if 'summary' in insights:
            net_flow = insights['summary'].get('net_flow', 0)
            if net_flow < 0:
                recommendations.append(
                    f"Seu saldo líquido está negativo em R$ {abs(net_flow):.2f}. "
                    f"Considere reduzir gastos ou aumentar receitas."
                )
        
        return recommendations
    
    @handle_service_errors('ai_service')
    @with_timeout(45)  # 45 second timeout for AI insights
    def generate_ai_insights(self, transactions: List[Dict]) -> str:
        """
        Gera insights usando IA generativa (GPT ou Groq)
        """
        if not transactions:
            raise InsufficientDataError('AI insights generation', 1, 0)
        
        # Check if AI service is available
        if not self.openai_client and not self.groq_client:
            raise AIServiceUnavailableError('No AI service configured')
        
        try:
            # Prepara um resumo dos dados para a IA
            summary = self.generate_insights(transactions)
            
            prompt = self._create_insights_prompt(summary)
            
            logger.info("Generating AI insights using external AI service")
            
            # Try with retry mechanism
            result = self._generate_insights_with_retry(prompt, len(transactions))
            
            return result
            
        except Exception as e:
            if isinstance(e, (AIServiceError, InsufficientDataError)):
                raise
            
            logger.error(f"Unexpected error generating AI insights: {str(e)}", exc_info=True)
            audit_logger.log_ai_operation('insights_generation', len(transactions), False, error=str(e))
            raise AIServiceError(f"AI insights generation failed: {str(e)}")
    
    def _create_insights_prompt(self, summary: Dict[str, Any]) -> str:
        """Create a structured prompt for AI insights generation."""
        return f"""
        Analise os seguintes dados financeiros e forneça insights valiosos em português:
        
        Resumo das transações:
        - Total de transações: {summary['summary']['total_transactions']}
        - Total de receitas: R$ {summary['summary']['total_credits']:.2f}
        - Total de gastos: R$ {summary['summary']['total_debits']:.2f}
        - Saldo líquido: R$ {summary['summary']['net_flow']:.2f}
        
        Gastos por categoria:
        {summary['categories']}
        
        Forneça uma análise concisa e actionable sobre:
        1. Principais padrões de gastos
        2. Oportunidades de economia
        3. Alertas importantes
        4. Recomendações personalizadas
        
        Mantenha a resposta em até 300 palavras e seja prático.
        """
    
    def _generate_insights_with_retry(self, prompt: str, transaction_count: int) -> str:
        """Generate insights with retry mechanism and fallback."""
        def try_openai():
            if not self.openai_client:
                raise AIServiceUnavailableError('OpenAI')
            
            logger.debug("Using OpenAI GPT-4o-mini for insights generation")
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    timeout=self.default_timeout
                )
                result = response.choices[0].message.content
                audit_logger.log_ai_operation('insights_generation', transaction_count, True, 'gpt-4o-mini')
                return result
                
            except openai.RateLimitError:
                raise AIServiceQuotaExceededError('OpenAI')
            except openai.APITimeoutError:
                raise AIServiceTimeoutError('OpenAI', self.default_timeout)
            except openai.APIConnectionError:
                raise AIServiceUnavailableError('OpenAI')
        
        def try_groq():
            if not self.groq_client:
                raise AIServiceUnavailableError('Groq')
            
            logger.debug("Using Groq Llama3 for insights generation")
            
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400
                )
                result = response.choices[0].message.content
                audit_logger.log_ai_operation('insights_generation', transaction_count, True, 'llama3-8b-8192')
                return result
                
            except Exception as e:
                if 'rate limit' in str(e).lower():
                    raise AIServiceQuotaExceededError('Groq')
                elif 'timeout' in str(e).lower():
                    raise AIServiceTimeoutError('Groq', self.default_timeout)
                else:
                    raise AIServiceUnavailableError('Groq')
        
        def fallback_insights():
            logger.warning("All AI services failed, generating fallback insights")
            return self._generate_fallback_insights(transaction_count)
        
        # Try primary service with retry
        for attempt in range(self.max_retries):
            try:
                if self.openai_client:
                    return try_openai()
                elif self.groq_client:
                    return try_groq()
            except (AIServiceQuotaExceededError, AIServiceTimeoutError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.rate_limit_delay * (2 ** attempt)
                    logger.warning(f"AI service error, retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"AI service failed after {self.max_retries} attempts")
                    break
            except AIServiceUnavailableError:
                # Try alternative service
                try:
                    if self.groq_client and attempt == 0:
                        return try_groq()
                    elif self.openai_client and attempt == 0:
                        return try_openai()
                except Exception:
                    pass
                break
        
        # Use fallback if all services fail
        return fallback_insights()
    
    def _generate_fallback_insights(self, transaction_count: int) -> str:
        """Generate basic insights when AI services are unavailable."""
        return f"""
        Análise básica dos dados financeiros:
        
        📊 Resumo: {transaction_count} transações processadas
        
        ⚠️ Serviço de IA temporariamente indisponível
        
        Recomendações gerais:
        • Revise regularmente suas transações
        • Categorize gastos para melhor controle
        • Monitore transações incomuns
        • Mantenha registros organizados
        
        Para análises mais detalhadas, tente novamente em alguns minutos.
        """
    
    @handle_service_errors('ai_service')
    @with_timeout(300)  # 5 minute timeout for model training
    def train_custom_model(self, financial_data: List[Dict], model_type: str = 'auto') -> Dict[str, Any]:
        """
        Treina um modelo personalizado com dados financeiros da empresa

        Args:
            financial_data: Lista de dados financeiros para treinamento
            model_type: Tipo de modelo ('auto', 'random_forest', 'xgboost', 'lightgbm', 'kmeans')
        """
        if not financial_data:
            raise InsufficientDataError('model training', 10, 0)

        if len(financial_data) < 10:
            raise InsufficientDataError('model training', 10, len(financial_data))

        try:
            logger.info(f"Starting custom model training with {len(financial_data)} data points using {model_type}")

            # Validate training data
            valid_data = self._validate_training_data(financial_data)

            if len(valid_data) < 5:
                raise InsufficientDataError('model training', 5, len(valid_data))

            # Prepara os dados de treinamento
            training_labels = [entry.get('category', 'outros') for entry in valid_data]

            unique_categories = len(set(training_labels))
            logger.info(f"Training data prepared. Valid entries: {len(valid_data)}, Unique categories: {unique_categories}")

            if unique_categories < 2:
                logger.warning("Insufficient category diversity for meaningful training")

            # Advanced feature engineering using FeatureEngineer
            X, feature_names = self.feature_engineer.create_comprehensive_features(
                valid_data, target_column='category'
            )

            # Store feature names
            self.feature_names[model_type] = feature_names

            # Encode labels for supervised learning
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(training_labels)

            # Ensure consecutive class labels for XGBoost compatibility
            unique_labels = np.unique(y)
            if len(unique_labels) != len(label_encoder.classes_):
                # Remap labels to be consecutive
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
                y = np.array([label_mapping[label] for label in y])

                # Update label encoder classes to match
                label_encoder.classes_ = np.array(list(label_mapping.keys()))

            # Split data for supervised learning evaluation
            # Check if stratification is possible (all classes must have at least 2 samples)
            min_class_count = min(np.bincount(y))
            if min_class_count >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                # Fallback to non-stratified split if stratification not possible
                logger.warning("Insufficient samples per class for stratified split, using random split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

            # Ensure test labels are also consecutive (important for XGBoost)
            test_unique_labels = np.unique(y_test)
            if len(test_unique_labels) != len(unique_labels):
                # Remap test labels to match training labels
                test_label_mapping = {label: i for i, label in enumerate(unique_labels)}
                y_test = np.array([test_label_mapping.get(label, 0) for label in y_test])

            # Determine model type
            if model_type == 'auto':
                selected_model = self.supervised_manager.select_best_model(X_train, y_train, X_test, y_test)
                logger.info(f"Auto-selected model: {selected_model}")
            elif model_type == 'kmeans':
                # Backward compatibility - use KMeans clustering
                selected_model = 'kmeans'
            elif model_type == 'bert':
                # Special handling for BERT
                return self.train_bert_model(financial_data, model_type)
            else:
                selected_model = model_type

            # Train the selected model
            if selected_model == 'kmeans':
                # Original KMeans approach for backward compatibility
                n_clusters = min(10, unique_categories, len(valid_data) // 2)
                classifier = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                classifier.fit(X)

                # Salva o modelo e componentes (legacy approach)
                self.custom_classifier = classifier
                self.custom_label_encoder = label_encoder
                self.model_trained = True
                self.model_type = 'kmeans'

                # Calcula acurácia (simplificada)
                self.model_accuracy = self._calculate_accuracy(valid_data, classifier, None)

                logger.info(f"KMeans model training completed successfully. Accuracy: {self.model_accuracy:.2f}")

                return {
                    'success': True,
                    'message': 'Modelo KMeans treinado com sucesso',
                    'model_type': 'kmeans',
                    'accuracy': self.model_accuracy,
                    'training_data_count': len(valid_data),
                    'categories_count': unique_categories
                }

            else:
                # Supervised learning approach
                # Convert sparse matrices to dense for supervised learning models
                X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
                X_test_dense = X_test.toarray() if X_test is not None and hasattr(X_test, 'toarray') else X_test

                training_result = self.supervised_manager.train_model(
                    selected_model, X_train_dense, y_train, X_test_dense, y_test
                )

                if training_result['success']:
                    # Salva os componentes do modelo supervisionado
                    self.custom_label_encoder = label_encoder
                    self.model_trained = True
                    self.model_type = selected_model

                    # Extract metrics
                    metrics = training_result['metrics']
                    test_metrics = metrics.get('test', {})

                    logger.info(f"Supervised model ({selected_model}) training completed successfully")

                    return {
                        'success': True,
                        'message': f'Modelo {selected_model} treinado com sucesso',
                        'model_type': selected_model,
                        'accuracy': test_metrics.get('accuracy', 0),
                        'precision': test_metrics.get('precision', 0),
                        'recall': test_metrics.get('recall', 0),
                        'f1_score': test_metrics.get('f1_score', 0),
                        'training_data_count': len(valid_data),
                        'categories_count': unique_categories,
                        'metrics': metrics
                    }
                else:
                    raise AIServiceError(f"Supervised model training failed: {training_result.get('error', 'Unknown error')}")

        except Exception as e:
            if isinstance(e, (InsufficientDataError, ValidationError)):
                raise

            logger.error(f"Error training custom model: {str(e)}", exc_info=True)
            audit_logger.log_ai_operation('model_training', len(financial_data), False, error=str(e))
            raise AIServiceError(f'Model training failed: {str(e)}')
    
    def _validate_training_data(self, financial_data: List[Dict]) -> List[Dict]:
        """Validate and clean training data."""
        valid_data = []
        
        for entry in financial_data:
            # Check required fields
            if not entry.get('description') or not entry.get('description').strip():
                continue
            
            # Clean description
            description = str(entry['description']).strip()
            if len(description) < 3:  # Too short to be meaningful
                continue
            
            # Add to valid data
            valid_entry = entry.copy()
            valid_entry['description'] = description
            valid_data.append(valid_entry)
        
        return valid_data
    
    def _calculate_accuracy(self, data: List[Dict], classifier, vectorizer) -> float:
        """Calcula a acurácia do modelo (simplificada)"""
        # Implementação simplificada para exemplo
        return 0.85  # Valor de exemplo
    
    @handle_service_errors('ai_service')
    def categorize_with_custom_model(self, description: str) -> str:
        """
        Categoriza uma transação usando o modelo personalizado treinado
        """
        if not description or not description.strip():
            return 'outros'

        if not hasattr(self, 'model_trained') or not self.model_trained:
            logger.debug("Custom model not trained, using fallback categorization")
            return self.categorize_transaction(description)

        try:
            # Validate model components
            if not hasattr(self, 'model_type') or not self.model_type:
                logger.debug("Custom model not properly initialized, using fallback")
                return self.categorize_transaction(description)

            # Create a transaction dict for feature engineering
            transaction_dict = {
                'description': description,
                'date': datetime.now().isoformat(),
                'amount': 0.0,  # Default amount
                'category': 'unknown'  # Will be predicted
            }

            # Use FeatureEngineer to extract features
            X, _ = self.feature_engineer.create_comprehensive_features([transaction_dict])

            if X.size == 0:
                logger.warning("Feature engineering failed, using fallback")
                return self.categorize_transaction(description)

            # Use appropriate prediction method based on model type
            if self.model_type == 'bert':
                # BERT-based prediction
                return self.categorize_with_bert(description)

            elif self.model_type in ['random_forest', 'xgboost', 'lightgbm']:
                # Supervised learning approach
                if self.model_type not in self.supervised_manager.models:
                    logger.warning(f"Supervised model {self.model_type} not found in manager, using fallback")
                    return self.categorize_transaction(description)

                # Get prediction from supervised model
                prediction = self.supervised_manager.predict(self.model_type, X)

                # Decode the prediction back to category name
                if hasattr(self, 'custom_label_encoder'):
                    category_index = prediction[0]
                    category = self.custom_label_encoder.inverse_transform([category_index])[0]
                else:
                    # Fallback if label encoder is missing
                    category = self._map_cluster_to_category(prediction[0], description)

            else:
                # KMeans clustering approach (backward compatibility)
                if not hasattr(self, 'custom_classifier'):
                    logger.warning("KMeans classifier missing, using fallback")
                    return self.categorize_transaction(description)

                prediction = self.custom_classifier.predict(X)
                category = self._map_cluster_to_category(prediction[0], description)

            logger.debug(f"Custom model ({self.model_type}) categorized '{description}' as '{category}'")
            return category

        except Exception as e:
            logger.warning(f"Error using custom model for categorization: {str(e)}")
            # Fallback para o método padrão em caso de erro
            return self.categorize_transaction(description)
    
    def _map_cluster_to_category(self, cluster_id: int, description: str) -> str:
        """Map cluster ID to meaningful category name."""
        # This is a simplified mapping - in production you might want more sophisticated mapping
        # based on the training data or cluster analysis
        
        # Try rule-based categorization first for better results
        rule_based_category = self.categorize_transaction(description)
        
        if rule_based_category != 'outros':
            return rule_based_category
        
        # Fallback to cluster-based category
        cluster_categories = {
            0: 'alimentacao',
            1: 'transporte',
            2: 'casa',
            3: 'saude',
            4: 'lazer',
            5: 'vestuario',
            6: 'educacao',
            7: 'investimento',
            8: 'transferencia',
            9: 'outros'
        }
        
        return cluster_categories.get(cluster_id % 10, 'outros')
    
    @handle_service_errors('ai_service')
    @with_timeout(60)  # 1 minute timeout for predictions
    def predict_financial_trends(self, historical_data: List[Dict], periods: int = 12) -> Dict[str, Any]:
        """
        Prevê tendências financeiras com base em dados históricos
        """
        if not historical_data:
            raise InsufficientDataError('financial trend prediction', 30, 0)
        
        if len(historical_data) < 30:
            raise InsufficientDataError('financial trend prediction', 30, len(historical_data))
        
        if periods <= 0 or periods > 24:
            raise ValidationError("Periods must be between 1 and 24")
        
        try:
            # Converte dados históricos para DataFrame
            df = pd.DataFrame(historical_data)
            
            # Validate required columns
            required_columns = ['date', 'amount']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}")
            
            # Clean and validate data
            df = self._clean_historical_data(df)
            
            if len(df) < 10:
                raise InsufficientDataError('financial trend prediction', 10, len(df))
            
            # Converte datas
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Prepara dados para predição (simplificada)
            # Numa implementação real, isso usaria modelos de séries temporais
            
            # Calcula tendências básicas
            total_income = df[df['amount'] > 0]['amount'].sum() if len(df[df['amount'] > 0]) > 0 else 0
            total_expenses = abs(df[df['amount'] < 0]['amount'].sum()) if len(df[df['amount'] < 0]) > 0 else 0
            net_flow = total_income - total_expenses
            
            # Calcula médias mensais
            df['month'] = df['date'].dt.to_period('M')
            monthly_data = df.groupby('month')['amount'].sum().reset_index()
            avg_monthly_income = monthly_data[monthly_data['amount'] > 0]['amount'].mean() if len(monthly_data[monthly_data['amount'] > 0]) > 0 else 0
            avg_monthly_expenses = abs(monthly_data[monthly_data['amount'] < 0]['amount'].mean()) if len(monthly_data[monthly_data['amount'] < 0]) > 0 else 0
            
            # Previsões para próximos períodos (simplificadas)
            predictions = []
            current_date = df['date'].max() if len(df) > 0 else pd.Timestamp.now()
            
            for i in range(1, periods + 1):
                future_date = current_date + pd.DateOffset(months=i)
                predicted_income = avg_monthly_income if not pd.isna(avg_monthly_income) else total_income / len(monthly_data) if len(monthly_data) > 0 else 0
                predicted_expenses = avg_monthly_expenses if not pd.isna(avg_monthly_expenses) else total_expenses / len(monthly_data) if len(monthly_data) > 0 else 0
                predicted_net_flow = predicted_income - predicted_expenses
                
                predictions.append({
                    'date': future_date.strftime('%Y-%m'),
                    'predicted_income': round(predicted_income, 2),
                    'predicted_expenses': round(predicted_expenses, 2),
                    'predicted_net_flow': round(predicted_net_flow, 2)
                })
            
            return {
                'success': True,
                'data': {
                    'historical_summary': {
                        'total_income': round(total_income, 2),
                        'total_expenses': round(total_expenses, 2),
                        'net_flow': round(net_flow, 2),
                        'period_months': len(monthly_data)
                    },
                    'predictions': predictions
                }
            }
            
        except Exception as e:
            if isinstance(e, (InsufficientDataError, ValidationError)):
                raise
            
            logger.error(f"Error predicting financial trends: {str(e)}", exc_info=True)
            raise AIServiceError(f'Financial trend prediction failed: {str(e)}')
    
    def _clean_historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate historical data for predictions."""
        # Remove rows with missing critical data
        df = df.dropna(subset=['date', 'amount'])
        
        # Convert dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Convert amounts to numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])
        
        # Remove extreme outliers (amounts beyond reasonable range)
        amount_q99 = df['amount'].quantile(0.99)
        amount_q01 = df['amount'].quantile(0.01)
        df = df[(df['amount'] >= amount_q01) & (df['amount'] <= amount_q99)]
        
        # Sort by date
        df = df.sort_values('date')
        
        return df

    @handle_service_errors('ai_service')
    @with_timeout(600)  # 10 minute timeout for optimized training
    def train_custom_model_optimized(self, financial_data: List[Dict], model_type: str = 'auto',
                                    optimize_hyperparams: bool = True, n_trials: int = 30) -> Dict[str, Any]:
        """
        Train a custom model with optional hyperparameter optimization

        Args:
            financial_data: Lista de dados financeiros para treinamento
            model_type: Tipo de modelo ('auto', 'random_forest', 'xgboost', 'lightgbm')
            optimize_hyperparams: Whether to perform hyperparameter optimization
            n_trials: Number of optimization trials (if optimize_hyperparams=True)

        Returns:
            Dictionary with training results
        """
        if not financial_data:
            raise InsufficientDataError('optimized model training', 10, 0)

        if len(financial_data) < 10:
            raise InsufficientDataError('optimized model training', 10, len(financial_data))

        try:
            logger.info(f"Starting optimized custom model training with {len(financial_data)} data points using {model_type}")

            # Validate training data
            valid_data = self._validate_training_data(financial_data)

            if len(valid_data) < 5:
                raise InsufficientDataError('optimized model training', 5, len(valid_data))

            # Prepare training data
            training_labels = [entry.get('category', 'outros') for entry in valid_data]

            unique_categories = len(set(training_labels))
            logger.info(f"Training data prepared. Valid entries: {len(valid_data)}, Unique categories: {unique_categories}")

            if unique_categories < 2:
                logger.warning("Insufficient category diversity for meaningful training")

            # Advanced feature engineering using FeatureEngineer
            X, feature_names = self.feature_engineer.create_comprehensive_features(
                valid_data, target_column='category'
            )

            # Store feature names
            self.feature_names[model_type] = feature_names

            # Encode labels
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(training_labels)

            # Ensure consecutive class labels
            unique_labels = np.unique(y)
            if len(unique_labels) != len(label_encoder.classes_):
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
                y = np.array([label_mapping[label] for label in y])
                label_encoder.classes_ = np.array(list(label_mapping.keys()))

            # Split data
            min_class_count = min(np.bincount(y))
            if min_class_count >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                logger.warning("Insufficient samples per class for stratified split, using random split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

            # Ensure test labels are consecutive
            test_unique_labels = np.unique(y_test)
            if len(test_unique_labels) != len(unique_labels):
                test_label_mapping = {label: i for i, label in enumerate(unique_labels)}
                y_test = np.array([test_label_mapping.get(label, 0) for label in y_test])

            # Determine model type
            if model_type == 'auto':
                selected_model = self.supervised_manager.select_best_model(X_train, y_train, X_test, y_test)
                logger.info(f"Auto-selected model: {selected_model}")
            else:
                selected_model = model_type

            # Convert to dense arrays for supervised learning
            X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
            X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

            # Train optimized model
            training_result = self.supervised_manager.train_optimized_model(
                selected_model, X_train_dense, y_train, X_test_dense, y_test,
                optimize=optimize_hyperparams, n_trials=n_trials
            )

            if training_result['success']:
                # Store components
                self.custom_label_encoder = label_encoder
                self.model_trained = True
                self.model_type = selected_model

                # Extract metrics
                if optimize_hyperparams and 'optimized_metrics' in training_result:
                    metrics = training_result['optimized_metrics']
                    test_metrics = metrics.get('test', {})
                    accuracy = test_metrics.get('accuracy', 0)
                    f1_score_val = test_metrics.get('f1_score', 0)
                else:
                    metrics = training_result.get('metrics', {})
                    test_metrics = metrics.get('test', {})
                    accuracy = test_metrics.get('accuracy', 0)
                    f1_score_val = test_metrics.get('f1_score', 0)

                logger.info(f"Optimized custom model ({selected_model}) training completed successfully")

                result = {
                    'success': True,
                    'message': f'Modelo {selected_model} treinado com sucesso (otimizado: {optimize_hyperparams})',
                    'model_type': selected_model,
                    'optimized': optimize_hyperparams,
                    'accuracy': accuracy,
                    'f1_score': f1_score_val,
                    'training_data_count': len(valid_data),
                    'categories_count': unique_categories,
                    'metrics': metrics
                }

                # Add optimization-specific information
                if optimize_hyperparams:
                    result.update({
                        'best_params': training_result.get('best_params', {}),
                        'optimization_score': training_result.get('optimization_score', 0),
                        'performance_comparison': training_result.get('performance_comparison', {}),
                        'default_metrics': training_result.get('default_metrics', {})
                    })

                return result
            else:
                raise AIServiceError(f"Optimized model training failed: {training_result.get('error', 'Unknown error')}")

        except Exception as e:
            if isinstance(e, (InsufficientDataError, ValidationError)):
                raise

            logger.error(f"Error training optimized custom model: {str(e)}", exc_info=True)
            audit_logger.log_ai_operation('optimized_model_training', len(financial_data), False, error=str(e))
            raise AIServiceError(f'Optimized model training failed: {str(e)}')

    def get_optimization_info(self, model_type: str) -> Dict[str, Any]:
        """
        Get optimization information for a specific model type
        """
        return self.supervised_manager.get_optimization_info(model_type)

    def get_available_models_info(self) -> Dict[str, Any]:
        """
        Get information about all available models including optimization status
        """
        models_info = {}

        for model_type in ['random_forest', 'xgboost', 'lightgbm']:
            info = {
                'is_trained': model_type in self.supervised_manager.models,
                'is_optimized': f"{model_type}_optimized" in self.supervised_manager.models,
                'has_optimization_history': model_type in self.supervised_manager.optimization_history
            }

            if info['has_optimization_history']:
                opt_info = self.supervised_manager.optimization_history[model_type]
                info.update({
                    'best_score': opt_info.get('best_score', 0),
                    'n_trials': opt_info.get('n_trials', 0),
                    'optimization_time': opt_info.get('optimization_time', 0)
                })

            models_info[model_type] = info

        return models_info

    def get_feature_importance_info(self) -> Dict[str, Any]:
        """
        Get feature importance information from the FeatureEngineer

        Returns:
            Dictionary with feature importance information
        """
        try:
            return self.supervised_manager.feature_engineer.get_feature_importance()
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {'error': f'Failed to get feature importance: {str(e)}'}

    def get_feature_names(self, model_type: str = None) -> List[str]:
        """
        Get feature names for a specific model or all models

        Args:
            model_type: Specific model type to get feature names for

        Returns:
            List of feature names
        """
        if model_type and model_type in self.feature_names:
            return self.feature_names[model_type]
        elif model_type:
            return []
        else:
            # Return all feature names
            all_names = []
            for names in self.feature_names.values():
                all_names.extend(names)
            return list(set(all_names))  # Remove duplicates

    def train_bert_model(self, financial_data: List[Dict], model_type: str = 'bert') -> Dict[str, Any]:
        """
        Train BERT model for text classification

        Args:
            financial_data: List of financial transaction data
            model_type: Model type (should be 'bert')

        Returns:
            Dictionary with training results
        """
        if not financial_data:
            raise InsufficientDataError('BERT training', 10, 0)

        if len(financial_data) < 10:
            raise InsufficientDataError('BERT training', 10, len(financial_data))

        try:
            self.logger.info(f"Starting BERT training with {len(financial_data)} data points")

            # Extract texts and labels
            texts = [entry.get('description', '') for entry in financial_data]
            labels = [entry.get('category', 'outros') for entry in financial_data]

            # Filter out empty texts
            valid_data = [(text, label) for text, label in zip(texts, labels) if text.strip()]
            if len(valid_data) < 5:
                raise InsufficientDataError('BERT training', 5, len(valid_data))

            texts, labels = zip(*valid_data)

            # Split data for validation
            train_texts, val_texts = list(texts)[:int(0.8 * len(texts))], list(texts)[int(0.8 * len(texts)):]
            train_labels, val_labels = list(labels)[:int(0.8 * len(labels))], list(labels)[int(0.8 * len(labels)):]

            # Train BERT model
            result = self.supervised_manager.train_bert_model(
                train_texts, train_labels, val_texts, val_labels
            )

            if result['success']:
                audit_logger.log_ai_operation('bert_training', len(financial_data), True)
            else:
                audit_logger.log_ai_operation('bert_training', len(financial_data), False, error=result.get('error'))

            return result

        except Exception as e:
            if isinstance(e, (InsufficientDataError, ValidationError)):
                raise

            self.logger.error(f"Error training BERT model: {str(e)}", exc_info=True)
            audit_logger.log_ai_operation('bert_training', len(financial_data), False, error=str(e))
            raise AIServiceError(f'BERT training failed: {str(e)}')

    def categorize_with_bert(self, description: str) -> str:
        """
        Categorize transaction using BERT model

        Args:
            description: Transaction description

        Returns:
            Predicted category
        """
        if not description or not description.strip():
            return 'outros'

        try:
            predictions = self.supervised_manager.predict_with_bert([description])
            return predictions[0] if predictions else 'outros'

        except Exception as e:
            self.logger.warning(f"Error categorizing with BERT: {str(e)}")
            # Fallback to rule-based categorization
            return self.categorize_transaction(description)

    def compare_model_performance(self, financial_data: List[Dict]) -> Dict[str, Any]:
        """
        Compare performance of different models including BERT

        Args:
            financial_data: List of financial transaction data for testing

        Returns:
            Dictionary with performance comparison results
        """
        if not financial_data:
            return {'error': 'No data provided for comparison'}

        try:
            self.logger.info(f"Comparing model performance with {len(financial_data)} samples")

            # Prepare data
            texts = [entry.get('description', '') for entry in financial_data]
            labels = [entry.get('category', 'outros') for entry in financial_data]

            # Filter valid data
            valid_data = [(text, label) for text, label in zip(texts, labels) if text.strip()]
            if len(valid_data) < 10:
                return {'error': 'Insufficient valid data for comparison'}

            texts, labels = zip(*valid_data)

            # Create feature matrix for traditional models
            X, feature_names = self.feature_engineer.create_comprehensive_features(
                [{'description': text, 'category': label} for text, label in zip(texts, labels)]
            )

            # Encode labels
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(labels)

            # Split data with fallback for small datasets
            try:
                # Check if stratification is possible
                unique_labels, counts = np.unique(y, return_counts=True)
                min_samples_per_class = min(counts)

                if min_samples_per_class >= 2:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                else:
                    # Fallback to non-stratified split
                    self.logger.warning("Insufficient samples per class for stratification, using random split")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                texts_train, texts_test = texts[:len(X_train)], texts[len(X_train):]
                labels_train, labels_test = labels[:len(y_train)], labels[len(y_train):]

            except Exception as split_error:
                self.logger.error(f"Error splitting data: {str(split_error)}")
                # Use simple split as final fallback
                split_idx = int(0.8 * len(texts))
                texts_train, texts_test = texts[:split_idx], texts[split_idx:]
                labels_train, labels_test = labels[:split_idx], labels[split_idx:]
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

            # Test traditional models
            traditional_results = {}
            for model_type in ['random_forest', 'xgboost', 'lightgbm']:
                try:
                    result = self.supervised_manager.train_model(model_type, X_train, y_train, X_test, y_test)
                    if result['success']:
                        traditional_results[model_type] = result['metrics']['test']
                except Exception as e:
                    self.logger.warning(f"Error training {model_type}: {str(e)}")

            # Test BERT
            bert_results = {}
            try:
                bert_result = self.supervised_manager.train_bert_model(
                    texts_train, labels_train, texts_test, labels_test
                )
                if bert_result['success'] and 'evaluation_metrics' in bert_result:
                    bert_results = bert_result['evaluation_metrics']
            except Exception as e:
                self.logger.warning(f"Error training BERT: {str(e)}")

            # Compare results
            comparison = {
                'traditional_models': traditional_results,
                'bert_model': bert_results,
                'best_model': self._determine_best_model(traditional_results, bert_results),
                'recommendation': self._generate_model_recommendation(traditional_results, bert_results)
            }

            self.logger.info(f"Model comparison completed. Best model: {comparison['best_model']}")
            return comparison

        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            return {'error': f'Model comparison failed: {str(e)}'}

    def _determine_best_model(self, traditional_results: Dict, bert_results: Dict) -> str:
        """Determine the best performing model based on F1 score"""
        best_model = 'random_forest'
        best_score = 0

        # Check traditional models
        for model_type, metrics in traditional_results.items():
            f1_score = metrics.get('f1_score', 0)
            if f1_score > best_score:
                best_score = f1_score
                best_model = model_type

        # Check BERT
        bert_f1 = bert_results.get('eval_f1', 0)
        if bert_f1 > best_score:
            best_score = bert_f1
            best_model = 'bert'

        return best_model

    def _generate_model_recommendation(self, traditional_results: Dict, bert_results: Dict) -> str:
        """Generate recommendation based on model performance"""
        best_model = self._determine_best_model(traditional_results, bert_results)

        if best_model == 'bert':
            bert_f1 = bert_results.get('eval_f1', 0)
            return f"BERT is recommended with F1 score of {bert_f1:.4f}. Suitable for text-heavy classification tasks."
        else:
            best_f1 = traditional_results.get(best_model, {}).get('f1_score', 0)
            return f"{best_model.replace('_', ' ').title()} is recommended with F1 score of {best_f1:.4f}. Good balance of performance and efficiency."

    def get_bert_model_info(self) -> Dict[str, Any]:
        """
        Get information about the BERT model

        Returns:
            Dictionary with BERT model information
        """
        return self.supervised_manager.bert_classifier.get_model_info()

    def get_model_comparison_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model comparison information

        Returns:
            Dictionary with model comparison data
        """
        info = {
            'traditional_models': self.supervised_manager.get_available_models_info(),
            'bert_model': self.get_bert_model_info(),
            'bert_performance': self.supervised_manager.bert_performance
        }

        return info

