#!/usr/bin/env python3
"""
Unit tests for ModelManager and individual ML models
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.services.model_manager import (
    ModelManager, RandomForestModel, XGBoostModel,
    LightGBMModel, BERTModel, KMeansModel
)
from src.services.feature_engineer import FeatureEngineer
from src.services.model_selector import ModelSelector
from src.utils.exceptions import ValidationError, AIServiceError


@pytest.fixture
def sample_training_data():
    """Sample training data for testing"""
    return [
        {'description': 'Compra no supermercado Extra', 'amount': 150.0, 'date': '2024-01-01', 'category': 'alimentacao'},
        {'description': 'Pagamento de conta de luz', 'amount': 200.0, 'date': '2024-01-02', 'category': 'casa'},
        {'description': 'Transferência PIX recebida', 'amount': 500.0, 'date': '2024-01-03', 'category': 'transferencia'},
        {'description': 'Compra de remédio na farmácia', 'amount': 80.0, 'date': '2024-01-04', 'category': 'saude'},
        {'description': 'Pagamento de aluguel', 'amount': 1200.0, 'date': '2024-01-05', 'category': 'casa'},
        {'description': 'Compra no shopping', 'amount': 300.0, 'date': '2024-01-06', 'category': 'vestuario'},
        {'description': 'Restaurante jantar', 'amount': 120.0, 'date': '2024-01-07', 'category': 'alimentacao'},
        {'description': 'Combustível posto', 'amount': 250.0, 'date': '2024-01-08', 'category': 'transporte'},
        {'description': 'Salário depositado', 'amount': 3500.0, 'date': '2024-01-09', 'category': 'salario'},
        {'description': 'Compra Netflix', 'amount': 39.90, 'date': '2024-01-10', 'category': 'lazer'}
    ]


@pytest.fixture
def mock_feature_engineer():
    """Mock FeatureEngineer for testing"""
    mock_fe = Mock(spec=FeatureEngineer)
    mock_fe.create_comprehensive_features.return_value = (
        np.random.rand(10, 20),  # X
        [f'feature_{i}' for i in range(20)]  # feature_names
    )
    return mock_fe


@pytest.fixture
def mock_model_selector():
    """Mock ModelSelector for testing"""
    mock_ms = Mock(spec=ModelSelector)
    mock_ms.select_best_model.return_value = {
        'best_model': 'random_forest',
        'recommendation': 'Random Forest is recommended for this dataset'
    }
    return mock_ms


@pytest.fixture
def model_manager(mock_feature_engineer, mock_model_selector):
    """ModelManager instance with mocked dependencies"""
    with patch('src.services.model_manager.FeatureEngineer', return_value=mock_feature_engineer), \
         patch('src.services.model_manager.ModelSelector', return_value=mock_model_selector):

        manager = ModelManager()
        return manager


class TestRandomForestModel:
    """Unit tests for RandomForestModel"""

    def test_initialization(self):
        """Test RandomForestModel initialization"""
        config = {'n_estimators': 50, 'max_depth': 10}
        model = RandomForestModel(config)

        assert model.config == config
        assert model.model is None
        assert not model.is_trained
        assert model.label_encoder is None

    def test_train_success(self):
        """Test successful training"""
        config = {'n_estimators': 10, 'max_depth': 5}
        model = RandomForestModel(config)

        X = np.random.rand(20, 5)
        y = np.array(['class_a', 'class_b'] * 10)

        result = model.train(X, y)

        assert result['success'] is True
        assert model.is_trained is True
        assert model.model is not None
        assert 'n_estimators' in result

    def test_predict_without_training(self):
        """Test prediction without training raises error"""
        model = RandomForestModel({})

        X = np.random.rand(5, 5)

        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(X)

    def test_predict_proba_without_training(self):
        """Test predict_proba without training raises error"""
        model = RandomForestModel({})

        X = np.random.rand(5, 5)

        with pytest.raises(ValueError, match="Model not trained"):
            model.predict_proba(X)

    def test_train_with_insufficient_data(self):
        """Test training with insufficient data"""
        model = RandomForestModel({})

        X = np.random.rand(2, 5)  # Very small dataset
        y = np.array(['class_a', 'class_b'])

        result = model.train(X, y)

        # Should still succeed but with warnings
        assert result['success'] is True

    def test_get_model_info(self):
        """Test getting model information"""
        config = {'n_estimators': 100}
        model = RandomForestModel(config)

        info = model.get_model_info()

        assert info['model_type'] == 'random_forest'
        assert info['is_trained'] is False
        assert 'training_metadata' in info


class TestXGBoostModel:
    """Unit tests for XGBoostModel"""

    def test_initialization(self):
        """Test XGBoostModel initialization"""
        config = {'n_estimators': 50, 'learning_rate': 0.1}
        model = XGBoostModel(config)

        assert model.config == config
        assert model.model is None
        assert not model.is_trained

    def test_train_success(self):
        """Test successful training"""
        config = {'n_estimators': 10, 'max_depth': 3}
        model = XGBoostModel(config)

        X = np.random.rand(20, 5)
        y = np.array(['class_a', 'class_b'] * 10)

        result = model.train(X, y)

        assert result['success'] is True
        assert model.is_trained is True
        assert model.model is not None

    def test_predict_returns_correct_format(self):
        """Test prediction returns correct format"""
        config = {'n_estimators': 10}
        model = XGBoostModel(config)

        # Train first
        X_train = np.random.rand(20, 5)
        y_train = np.array(['class_a', 'class_b'] * 10)
        model.train(X_train, y_train)

        # Test prediction
        X_test = np.random.rand(5, 5)
        predictions = model.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 5
        assert all(pred in ['class_a', 'class_b'] for pred in predictions)


class TestLightGBMModel:
    """Unit tests for LightGBMModel"""

    def test_initialization(self):
        """Test LightGBMModel initialization"""
        config = {'n_estimators': 50, 'learning_rate': 0.1}
        model = LightGBMModel(config)

        assert model.config == config
        assert model.model is None
        assert not model.is_trained

    def test_train_success(self):
        """Test successful training"""
        config = {'n_estimators': 10, 'max_depth': 3}
        model = LightGBMModel(config)

        X = np.random.rand(20, 5)
        y = np.array(['class_a', 'class_b'] * 10)

        result = model.train(X, y)

        assert result['success'] is True
        assert model.is_trained is True
        assert model.model is not None


class TestKMeansModel:
    """Unit tests for KMeansModel"""

    def test_initialization(self):
        """Test KMeansModel initialization"""
        config = {'n_clusters': 5}
        model = KMeansModel(config)

        assert model.config == config
        assert model.n_clusters == 5
        assert model.model is None
        assert not model.is_trained

    def test_train_clustering(self):
        """Test KMeans training for clustering"""
        config = {'n_clusters': 3}
        model = KMeansModel(config)

        X = np.random.rand(30, 5)
        y = np.array(['dummy'] * 30)  # y is ignored in clustering

        result = model.train(X, y)

        assert result['success'] is True
        assert model.is_trained is True
        assert result['n_clusters'] == 3
        assert 'inertia' in result

    def test_predict_clustering(self):
        """Test KMeans prediction returns cluster labels"""
        config = {'n_clusters': 3}
        model = KMeansModel(config)

        # Train
        X_train = np.random.rand(30, 5)
        y_train = np.array(['dummy'] * 30)
        model.train(X_train, y_train)

        # Predict
        X_test = np.random.rand(10, 5)
        predictions = model.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
        assert all(0 <= pred < 3 for pred in predictions)

    def test_predict_proba_clustering(self):
        """Test KMeans predict_proba returns distance-based probabilities"""
        config = {'n_clusters': 3}
        model = KMeansModel(config)

        # Train
        X_train = np.random.rand(30, 5)
        y_train = np.array(['dummy'] * 30)
        model.train(X_train, y_train)

        # Predict probabilities
        X_test = np.random.rand(5, 5)
        probabilities = model.predict_proba(X_test)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (5, 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1


class TestBERTModel:
    """Unit tests for BERTModel"""

    @patch('src.services.model_manager.BERTTextClassifier')
    def test_initialization(self, mock_bert_classifier):
        """Test BERTModel initialization"""
        mock_classifier = Mock()
        mock_bert_classifier.return_value = mock_classifier

        config = {'bert_config': {'model_name': 'bert-base-uncased'}}
        model = BERTModel(config)

        assert model.config == config
        assert model.bert_classifier == mock_classifier
        mock_bert_classifier.assert_called_once_with({'model_name': 'bert-base-uncased'})

    @patch('src.services.model_manager.BERTTextClassifier')
    def test_train_with_texts(self, mock_bert_classifier):
        """Test BERT training with text data"""
        mock_classifier = Mock()
        mock_classifier.train.return_value = {'success': True, 'training_metrics': {}}
        mock_bert_classifier.return_value = mock_classifier

        model = BERTModel({})
        X = np.random.rand(10, 5)
        y = np.array(['class_a', 'class_b'] * 5)
        texts = ['text 1', 'text 2'] * 5

        result = model.train(X, y, texts=texts)

        assert result['success'] is True
        mock_classifier.train.assert_called_once_with(texts, y)

    @patch('src.services.model_manager.BERTTextClassifier')
    def test_train_without_texts_fails(self, mock_bert_classifier):
        """Test BERT training fails without text data"""
        mock_classifier = Mock()
        mock_bert_classifier.return_value = mock_classifier

        model = BERTModel({})
        X = np.random.rand(10, 5)
        y = np.array(['class_a', 'class_b'] * 5)

        result = model.train(X, y)

        assert result['success'] is False
        assert 'BERT requires text data' in result['error']

    @patch('src.services.model_manager.BERTTextClassifier')
    def test_predict_with_texts(self, mock_bert_classifier):
        """Test BERT prediction with text data"""
        mock_classifier = Mock()
        mock_classifier.predict.return_value = np.array(['class_a', 'class_b'])
        mock_bert_classifier.return_value = mock_classifier

        model = BERTModel({})
        model.is_trained = True

        # Create mock data with texts attribute
        X = type('MockData', (), {'texts': ['text 1', 'text 2']})()

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 2
        mock_classifier.predict.assert_called_once_with(['text 1', 'text 2'])

    @patch('src.services.model_manager.BERTTextClassifier')
    def test_predict_without_texts_fails(self, mock_bert_classifier):
        """Test BERT prediction fails without text data"""
        mock_classifier = Mock()
        mock_bert_classifier.return_value = mock_classifier

        model = BERTModel({})
        model.is_trained = True

        X = np.random.rand(5, 5)

        with pytest.raises(ValueError, match="BERT prediction requires text data"):
            model.predict(X)


class TestModelManager:
    """Unit tests for ModelManager"""

    def test_initialization(self, mock_feature_engineer, mock_model_selector):
        """Test ModelManager initialization"""
        with patch('src.services.model_manager.FeatureEngineer', return_value=mock_feature_engineer):

            manager = ModelManager()

            assert isinstance(manager.models, dict)
            assert 'random_forest' in manager.models
            assert 'xgboost' in manager.models
            assert 'lightgbm' in manager.models
            assert 'bert' in manager.models
            assert 'kmeans' in manager.models

    def test_process_data(self, model_manager, sample_training_data, mock_feature_engineer):
        """Test data processing"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(10, 15),
            [f'feature_{i}' for i in range(15)]
        )

        X, y, feature_names = model_manager.process_data(sample_training_data)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, list)
        assert len(feature_names) == 15
        assert X.shape[0] == 10

    def test_train_model_success(self, model_manager):
        """Test successful model training"""
        X = np.random.rand(20, 5)
        y = np.array(['class_a', 'class_b'] * 10)

        result = model_manager.train_model('random_forest', X, y)

        assert result['success'] is True
        assert 'random_forest' in model_manager.models
        assert model_manager.models['random_forest'].is_trained is True

    def test_train_model_invalid_type(self, model_manager):
        """Test training with invalid model type"""
        X = np.random.rand(20, 5)
        y = np.array(['class_a', 'class_b'] * 10)

        result = model_manager.train_model('invalid_model', X, y)

        assert result['success'] is False
        assert 'not found' in result['error']

    def test_predict_success(self, model_manager):
        """Test successful prediction"""
        # Train model first
        X = np.random.rand(20, 5)
        y = np.array(['class_a', 'class_b'] * 10)
        model_manager.train_model('random_forest', X, y)

        # Test prediction
        X_test = np.random.rand(5, 5)
        predictions = model_manager.predict('random_forest', X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 5

    def test_predict_untrained_model(self, model_manager):
        """Test prediction with untrained model"""
        X = np.random.rand(5, 5)

        with pytest.raises(ValueError, match="not trained"):
            model_manager.predict('random_forest', X)

    def test_evaluate_model(self, model_manager):
        """Test model evaluation"""
        # Train model first
        X = np.random.rand(20, 5)
        y = np.array(['class_a', 'class_b'] * 10)
        model_manager.train_model('random_forest', X, y)

        # Evaluate
        evaluation = model_manager.evaluate_model('random_forest', X, y)

        assert 'accuracy' in evaluation
        assert 'f1_score' in evaluation
        assert 'precision' in evaluation
        assert 'recall' in evaluation
        assert isinstance(evaluation['accuracy'], (int, float))

    def test_select_best_model(self, model_manager, mock_model_selector):
        """Test model selection"""
        X = np.random.rand(20, 5)
        y = np.array(['class_a', 'class_b'] * 10)

        result = model_manager.select_best_model(X, y)

        assert 'best_model' in result
        assert 'recommendation' in result
        mock_model_selector.select_best_model.assert_called_once()

    def test_compare_models(self, model_manager):
        """Test model comparison"""
        X = np.random.rand(30, 5)
        y = np.array(['class_a', 'class_b'] * 15)

        result = model_manager.compare_models(X, y, ['random_forest', 'xgboost'])

        assert 'comparison_results' in result
        assert 'best_model' in result

    def test_get_model_info(self, model_manager):
        """Test getting model information"""
        info = model_manager.get_model_info()

        assert isinstance(info, dict)
        assert 'random_forest' in info
        assert 'xgboost' in info

    def test_get_model_info_specific(self, model_manager):
        """Test getting specific model information"""
        info = model_manager.get_model_info('random_forest')

        assert isinstance(info, dict)
        assert 'model_type' in info
        assert 'is_trained' in info

    def test_get_model_info_invalid(self, model_manager):
        """Test getting info for invalid model"""
        info = model_manager.get_model_info('invalid_model')

        assert 'error' in info
        assert 'not found' in info['error']

    def test_save_load_model(self, model_manager, tmp_path):
        """Test model saving and loading"""
        # Train model first
        X = np.random.rand(20, 5)
        y = np.array(['class_a', 'class_b'] * 10)
        model_manager.train_model('random_forest', X, y)

        # Save model
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        save_path = str(model_dir / "random_forest")

        success = model_manager.save_model('random_forest', save_path)
        assert success is True

        # Create new manager and load model
        with patch('src.services.model_manager.FeatureEngineer'):

            new_manager = ModelManager()
            load_success = new_manager.load_model('random_forest', save_path)

            assert load_success is True
            assert new_manager.models['random_forest'].is_trained is True

    def test_fallback_prediction(self, model_manager):
        """Test fallback prediction mechanism"""
        # Create scenario where primary model fails
        X = np.random.rand(5, 5)

        # Mock the primary model to be untrained
        model_manager.models['random_forest'].is_trained = False

        # Train a fallback model
        X_train = np.random.rand(20, 5)
        y_train = np.array(['class_a', 'class_b'] * 10)
        model_manager.train_model('xgboost', X_train, y_train)
        model_manager.create_fallback_chain(['xgboost'])

        # Should use fallback
        predictions = model_manager.predict('random_forest', X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 5

    def test_process_data_empty_input(self, model_manager):
        """Test processing empty data"""
        X, y, feature_names = model_manager.process_data([])

        assert X.size == 0
        assert y.size == 0
        assert feature_names == []

    def test_process_data_with_target_column(self, model_manager, sample_training_data):
        """Test processing data with specific target column"""
        X, y, feature_names = model_manager.process_data(sample_training_data, target_column='category')

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(y) == len(sample_training_data)
        assert all(category in y for category in ['alimentacao', 'casa', 'transferencia', 'saude', 'vestuario', 'transporte', 'salario', 'lazer'])


class TestModelManagerErrorHandling:
    """Test error handling in ModelManager"""

    def test_train_model_with_corrupted_data(self, model_manager):
        """Test training with corrupted data"""
        # Create data that might cause issues
        X = np.full((10, 5), np.nan)  # All NaN values
        y = np.array(['class_a'] * 10)

        result = model_manager.train_model('random_forest', X, y)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result

    def test_predict_with_wrong_dimensions(self, model_manager):
        """Test prediction with wrong input dimensions"""
        # Train with certain dimensions
        X_train = np.random.rand(20, 5)
        y_train = np.array(['class_a', 'class_b'] * 10)
        model_manager.train_model('random_forest', X_train, y_train)

        # Try to predict with different dimensions
        X_test = np.random.rand(5, 3)  # Different number of features

        with pytest.raises(Exception):  # Should raise some error
            model_manager.predict('random_forest', X_test)

    def test_evaluate_untrained_model(self, model_manager):
        """Test evaluating untrained model"""
        X = np.random.rand(10, 5)
        y = np.array(['class_a', 'class_b'] * 5)

        result = model_manager.evaluate_model('random_forest', X, y)

        assert 'error' in result
        assert 'not trained' in result['error']

    def test_compare_models_empty_list(self, model_manager):
        """Test comparing with empty model list"""
        X = np.random.rand(20, 5)
        y = np.array(['class_a', 'class_b'] * 10)

        result = model_manager.compare_models(X, y, [])

        assert 'comparison_results' in result
        assert 'best_model' in result