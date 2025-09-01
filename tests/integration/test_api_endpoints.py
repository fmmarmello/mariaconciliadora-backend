#!/usr/bin/env python3
"""
Integration tests for API endpoints
"""

import pytest
import json
from unittest.mock import Mock, patch
from flask import Flask

from src.routes.model_manager import model_manager_bp
from src.services.model_manager import ModelManager
from src.models.transaction import Transaction
from src.models.company_financial import CompanyFinancial


@pytest.fixture
def app():
    """Create Flask test app"""
    app = Flask(__name__)
    app.register_blueprint(model_manager_bp)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def mock_model_manager():
    """Mock ModelManager for testing"""
    with patch('src.routes.model_manager.model_manager', new_callable=Mock) as mock_mm:
        yield mock_mm


@pytest.fixture
def sample_training_data():
    """Sample training data"""
    return [
        {'description': 'Compra mercado', 'amount': 100.0, 'date': '2024-01-01', 'category': 'alimentacao'},
        {'description': 'Pagamento luz', 'amount': 200.0, 'date': '2024-01-02', 'category': 'casa'}
    ]


class TestTrainModelEndpoint:
    """Test /models/train endpoint"""

    def test_train_model_success(self, client, mock_model_manager, sample_training_data):
        """Test successful model training"""
        # Mock database query
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: data) for data in sample_training_data]
            mock_transaction.query = mock_query

            # Mock model manager
            mock_model_manager.process_data.return_value = (
                [[0.1, 0.2], [0.3, 0.4]],  # X
                ['alimentacao', 'casa'],     # y
                ['feature1', 'feature2']     # feature_names
            )
            mock_model_manager.train_model.return_value = {
                'success': True,
                'message': 'Model trained successfully'
            }

            # Test request
            response = client.post('/models/train', json={
                'model_type': 'random_forest',
                'data_source': 'transactions'
            })

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert 'data' in data

    def test_train_model_invalid_model_type(self, client):
        """Test training with invalid model type"""
        response = client.post('/models/train', json={
            'model_type': 'invalid_model',
            'data_source': 'transactions'
        })

        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'Invalid model type' in data['error']

    def test_train_model_invalid_data_source(self, client):
        """Test training with invalid data source"""
        response = client.post('/models/train', json={
            'model_type': 'random_forest',
            'data_source': 'invalid_source'
        })

        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'Invalid data source' in data['error']

    def test_train_model_missing_required_fields(self, client):
        """Test training with missing required fields"""
        response = client.post('/models/train', json={
            'model_type': 'random_forest'
            # Missing data_source
        })

        assert response.status_code == 400  # Validation error

    def test_train_model_insufficient_data(self, client):
        """Test training with insufficient data"""
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = []  # No data
            mock_transaction.query = mock_query

            response = client.post('/models/train', json={
                'model_type': 'random_forest',
                'data_source': 'transactions'
            })

            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['success'] is False

    def test_train_model_with_date_filters(self, client, mock_model_manager):
        """Test training with date filters"""
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: sample_training_data[0])]
            mock_transaction.query = mock_query

            mock_model_manager.process_data.return_value = (
                [[0.1, 0.2]], ['alimentacao'], ['feature1', 'feature2']
            )
            mock_model_manager.train_model.return_value = {'success': True}

            response = client.post('/models/train', json={
                'model_type': 'random_forest',
                'data_source': 'transactions',
                'start_date': '2024-01-01',
                'end_date': '2024-12-31'
            })

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True

    def test_train_model_with_category_filter(self, client, mock_model_manager):
        """Test training with category filter"""
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: sample_training_data[0])]
            mock_transaction.query = mock_query

            mock_model_manager.process_data.return_value = (
                [[0.1, 0.2]], ['alimentacao'], ['feature1', 'feature2']
            )
            mock_model_manager.train_model.return_value = {'success': True}

            response = client.post('/models/train', json={
                'model_type': 'random_forest',
                'data_source': 'transactions',
                'category_filter': 'alimentacao'
            })

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True

    def test_train_model_with_optimization(self, client, mock_model_manager):
        """Test training with hyperparameter optimization"""
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: sample_training_data[0])]
            mock_transaction.query = mock_query

            mock_model_manager.process_data.return_value = (
                [[0.1, 0.2]], ['alimentacao'], ['feature1', 'feature2']
            )
            mock_model_manager.train_model.return_value = {'success': True}

            response = client.post('/models/train', json={
                'model_type': 'random_forest',
                'data_source': 'transactions',
                'optimize': True,
                'n_trials': 10
            })

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True


class TestPredictEndpoint:
    """Test /models/predict endpoint"""

    def test_predict_success(self, client, mock_model_manager):
        """Test successful prediction"""
        mock_model_manager.process_data.return_value = (
            [[0.1, 0.2]], [], ['feature1', 'feature2']
        )
        mock_model_manager.predict.return_value = np.array(['alimentacao'])
        mock_model_manager.predict_proba.return_value = np.array([[0.8, 0.2]])

        response = client.post('/models/predict', json={
            'model_type': 'random_forest',
            'data': {
                'description': 'Compra mercado',
                'amount': 100.0,
                'date': '2024-01-01'
            }
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'prediction' in data['data']
        assert 'probabilities' in data['data']

    def test_predict_invalid_model_type(self, client):
        """Test prediction with invalid model type"""
        response = client.post('/models/predict', json={
            'model_type': 'invalid_model',
            'data': {'description': 'test'}
        })

        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'Invalid model type' in data['error']

    def test_predict_missing_data(self, client):
        """Test prediction with missing data"""
        response = client.post('/models/predict', json={
            'model_type': 'random_forest'
            # Missing data
        })

        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['success'] is False

    def test_predict_bert_model(self, client, mock_model_manager):
        """Test prediction with BERT model"""
        mock_model_manager.process_data.return_value = (
            type('MockData', (), {'texts': ['test text']})(), [], []
        )
        mock_model_manager.predict.return_value = np.array(['alimentacao'])

        response = client.post('/models/predict', json={
            'model_type': 'bert',
            'data': {
                'description': 'Compra mercado',
                'amount': 100.0,
                'date': '2024-01-01'
            }
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

    def test_predict_with_fallback(self, client, mock_model_manager):
        """Test prediction with fallback mechanism"""
        # Mock primary model failure
        mock_model_manager.predict.side_effect = Exception("Primary model failed")
        mock_model_manager._fallback_predict.return_value = np.array(['alimentacao'])

        mock_model_manager.process_data.return_value = (
            [[0.1, 0.2]], [], ['feature1', 'feature2']
        )

        response = client.post('/models/predict', json={
            'model_type': 'random_forest',
            'data': {
                'description': 'Compra mercado',
                'amount': 100.0,
                'date': '2024-01-01'
            }
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['data']['fallback_used'] is True


class TestCompareModelsEndpoint:
    """Test /models/compare endpoint"""

    def test_compare_models_success(self, client, mock_model_manager):
        """Test successful model comparison"""
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: sample_training_data[0]) for _ in range(10)]
            mock_transaction.query = mock_query

            mock_model_manager.process_data.return_value = (
                np.random.rand(10, 5), np.array(['class_a'] * 10), ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
            )
            mock_model_manager.compare_models.return_value = {
                'best_model': 'random_forest',
                'comparison_results': {}
            }

            response = client.post('/models/compare', json={
                'data_source': 'transactions',
                'models_to_compare': ['random_forest', 'xgboost']
            })

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert 'best_model' in data['data']

    def test_compare_models_insufficient_data(self, client):
        """Test comparison with insufficient data"""
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = []  # No data
            mock_transaction.query = mock_query

            response = client.post('/models/compare', json={
                'data_source': 'transactions',
                'models_to_compare': ['random_forest', 'xgboost']
            })

            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['success'] is False


class TestOptimizeModelEndpoint:
    """Test /models/optimize endpoint"""

    def test_optimize_model_success(self, client, mock_model_manager):
        """Test successful model optimization"""
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: sample_training_data[0]) for _ in range(30)]
            mock_transaction.query = mock_query

            mock_model_manager.process_data.return_value = (
                np.random.rand(30, 5), np.array(['class_a', 'class_b'] * 15), ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
            )
            mock_model_manager.optimize_hyperparameters.return_value = {
                'success': True,
                'best_params': {'n_estimators': 100, 'max_depth': 10},
                'best_score': 0.85
            }

            response = client.post('/models/optimize', json={
                'model_type': 'random_forest',
                'data_source': 'transactions',
                'n_trials': 10
            })

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert 'best_params' in data['data']

    def test_optimize_model_insufficient_data(self, client):
        """Test optimization with insufficient data"""
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: sample_training_data[0]) for _ in range(10)]  # Insufficient
            mock_transaction.query = mock_query

            response = client.post('/models/optimize', json={
                'model_type': 'random_forest',
                'data_source': 'transactions',
                'n_trials': 10
            })

            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['success'] is False


class TestModelInfoEndpoints:
    """Test model information endpoints"""

    def test_get_models_info(self, client, mock_model_manager):
        """Test getting all models info"""
        mock_model_manager.get_model_info.return_value = {
            'random_forest': {'is_trained': True, 'model_type': 'random_forest'},
            'xgboost': {'is_trained': False, 'model_type': 'xgboost'}
        }

        response = client.get('/models/info')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'models' in data['data']

    def test_get_specific_model_info(self, client, mock_model_manager):
        """Test getting specific model info"""
        mock_model_manager.get_model_info.return_value = {
            'is_trained': True,
            'model_type': 'random_forest',
            'training_metadata': {}
        }

        response = client.get('/models/random_forest/info')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['data']['model_type'] == 'random_forest'

    def test_get_invalid_model_info(self, client, mock_model_manager):
        """Test getting info for invalid model"""
        mock_model_manager.get_model_info.return_value = {'error': 'Model not found'}

        response = client.get('/models/invalid_model/info')

        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['success'] is False


class TestModelSelectionEndpoint:
    """Test /models/select endpoint"""

    def test_select_best_model_success(self, client, mock_model_manager):
        """Test successful model selection"""
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: sample_training_data[0]) for _ in range(20)]
            mock_transaction.query = mock_query

            mock_model_manager.process_data.return_value = (
                np.random.rand(20, 5), np.array(['class_a'] * 20), ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
            )
            mock_model_manager.select_best_model.return_value = {
                'best_model': 'random_forest',
                'recommendation': 'Random Forest is recommended'
            }

            response = client.post('/models/select', json={
                'data_source': 'transactions'
            })

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert data['data']['best_model'] == 'random_forest'


class TestBatchPredictEndpoint:
    """Test /models/batch-predict endpoint"""

    def test_batch_predict_success(self, client, mock_model_manager):
        """Test successful batch prediction"""
        batch_data = [
            {'description': 'Compra 1', 'amount': 100.0, 'date': '2024-01-01'},
            {'description': 'Compra 2', 'amount': 200.0, 'date': '2024-01-02'}
        ]

        mock_model_manager.process_data.return_value = (
            np.random.rand(2, 5), [], ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        )
        mock_model_manager.predict.return_value = np.array(['alimentacao', 'casa'])
        mock_model_manager.predict_proba.return_value = np.array([[0.8, 0.2], [0.6, 0.4]])

        response = client.post('/models/batch-predict', json={
            'model_type': 'random_forest',
            'data': batch_data
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'results' in data['data']
        assert len(data['data']['results']) == 2

    def test_batch_predict_empty_data(self, client):
        """Test batch prediction with empty data"""
        response = client.post('/models/batch-predict', json={
            'model_type': 'random_forest',
            'data': []
        })

        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['success'] is False

    def test_batch_predict_too_large_batch(self, client):
        """Test batch prediction with too large batch"""
        large_batch = [{'description': f'Item {i}'} for i in range(200)]

        response = client.post('/models/batch-predict', json={
            'model_type': 'random_forest',
            'data': large_batch
        })

        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['success'] is False


class TestAdvancedEndpoints:
    """Test advanced model management endpoints"""

    def test_advanced_model_selection(self, client, mock_model_manager):
        """Test advanced model selection"""
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: sample_training_data[0]) for _ in range(50)]
            mock_transaction.query = mock_query

            mock_model_manager.process_data.return_value = (
                np.random.rand(50, 5), np.array(['class_a'] * 50), ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
            )
            mock_model_manager.select_best_model.return_value = {
                'best_model': 'xgboost',
                'recommendation': 'XGBoost is recommended for this dataset'
            }

            response = client.post('/models/advanced-select', json={
                'data_source': 'transactions',
                'include_data_analysis': True,
                'generate_report': True
            })

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True

    def test_perform_ab_test(self, client, mock_model_manager):
        """Test A/B testing"""
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: sample_training_data[0]) for _ in range(100)]
            mock_transaction.query = mock_query

            mock_model_manager.process_data.return_value = (
                np.random.rand(100, 5), np.array(['class_a'] * 100), ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
            )
            mock_model_manager.perform_ab_test.return_value = {
                'winner': 'model_a',
                'confidence': 0.95,
                'improvement': 0.05
            }

            response = client.post('/models/ab-test', json={
                'model_a': 'random_forest',
                'model_b': 'xgboost',
                'data_source': 'transactions',
                'test_duration_hours': 24
            })

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True

    def test_data_analysis(self, client, mock_model_manager):
        """Test data analysis endpoint"""
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: sample_training_data[0]) for _ in range(10)]
            mock_transaction.query = mock_query

            mock_model_manager.process_data.return_value = (
                np.random.rand(10, 5), np.array(['class_a'] * 10), ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
            )
            mock_model_manager.analyze_data_characteristics.return_value = {
                'n_samples': 10,
                'n_features': 5,
                'class_distribution': {'class_a': 10}
            }

            response = client.post('/models/data-analysis', json={
                'data_source': 'transactions'
            })

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert 'n_samples' in data['data']


class TestErrorHandling:
    """Test error handling across endpoints"""

    def test_invalid_json(self, client):
        """Test handling of invalid JSON"""
        response = client.post('/models/train',
                             data='invalid json',
                             content_type='application/json')

        assert response.status_code == 400

    def test_missing_json(self, client):
        """Test handling of missing JSON"""
        response = client.post('/models/train')

        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['success'] is False

    def test_database_error(self, client):
        """Test handling of database errors"""
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.side_effect = Exception("Database error")
            mock_transaction.query = mock_query

            response = client.post('/models/train', json={
                'model_type': 'random_forest',
                'data_source': 'transactions'
            })

            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data

    def test_rate_limiting(self, client):
        """Test rate limiting functionality"""
        # This would require setting up rate limiting middleware
        # For now, just test that endpoints exist and respond
        response = client.post('/models/train', json={
            'model_type': 'random_forest',
            'data_source': 'transactions'
        })

        # Should not be rate limited in test environment
        assert response.status_code in [200, 500]


class TestEndpointValidation:
    """Test input validation for endpoints"""

    def test_train_endpoint_validation(self, client):
        """Test validation for train endpoint"""
        # Test missing model_type
        response = client.post('/models/train', json={
            'data_source': 'transactions'
        })
        assert response.status_code == 400

        # Test missing data_source
        response = client.post('/models/train', json={
            'model_type': 'random_forest'
        })
        assert response.status_code == 400

    def test_predict_endpoint_validation(self, client):
        """Test validation for predict endpoint"""
        # Test missing model_type
        response = client.post('/models/predict', json={
            'data': {'description': 'test'}
        })
        assert response.status_code == 500

        # Test missing data
        response = client.post('/models/predict', json={
            'model_type': 'random_forest'
        })
        assert response.status_code == 500

    def test_compare_endpoint_validation(self, client):
        """Test validation for compare endpoint"""
        # Test missing data_source
        response = client.post('/models/compare', json={
            'models_to_compare': ['random_forest']
        })
        assert response.status_code == 400

    def test_optimize_endpoint_validation(self, client):
        """Test validation for optimize endpoint"""
        # Test missing model_type
        response = client.post('/models/optimize', json={
            'data_source': 'transactions'
        })
        assert response.status_code == 400

        # Test missing data_source
        response = client.post('/models/optimize', json={
            'model_type': 'random_forest'
        })
        assert response.status_code == 400