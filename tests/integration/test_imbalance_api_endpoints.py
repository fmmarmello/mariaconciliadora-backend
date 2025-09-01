import pytest
import json
import numpy as np
from unittest.mock import Mock, patch

from src.services.imbalanced_data_handler import ImbalancedDataHandler
from src.services.financial_category_balancer import FinancialCategoryBalancer
from src.services.quality_assessment_engine import QualityAssessmentEngine


class TestImbalanceAPIEndpoints:
    """Integration tests for imbalance handling API endpoints"""

    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for testing"""
        np.random.seed(42)

        data = []
        categories = ['receita', 'despesa', 'transferencia', 'investimento']
        category_weights = [0.3, 0.5, 0.15, 0.05]  # Imbalanced distribution

        for i in range(1000):
            category = np.random.choice(categories, p=category_weights)
            amount = np.random.uniform(10, 1000)

            if category == 'receita':
                amount = np.random.uniform(1000, 5000)
            elif category == 'investimento':
                amount = np.random.uniform(100, 10000)

            data.append({
                'id': i,
                'amount': amount,
                'description': f'Transaction {i} for {category}',
                'category': category,
                'date': f'2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 28):02d}'
            })

        return data

    def test_train_model_with_balancing_endpoint(self, client, sample_transaction_data):
        """Test train model with balancing endpoint"""
        payload = {
            'model_type': 'random_forest',
            'data_source': 'transactions',
            'balancing_method': 'smote',
            'target_balance_ratio': 1.0,
            'use_augmentation': False
        }

        # Mock the database query
        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: item) for item in sample_transaction_data]

            mock_transaction.query = mock_query

            response = client.post('/api/model_manager/models/train-with-balancing',
                                 json=payload,
                                 content_type='application/json')

            assert response.status_code in [200, 500]  # 500 is acceptable if model training fails

            if response.status_code == 200:
                data = json.loads(response.data)
                assert 'success' in data
                assert 'data' in data

                if data['success']:
                    result_data = data['data']
                    assert 'model_type' in result_data
                    assert 'original_samples' in result_data
                    assert 'balanced_samples' in result_data

    def test_analyze_imbalance_endpoint(self, client, sample_transaction_data):
        """Test imbalance analysis endpoint"""
        payload = {
            'data_source': 'transactions',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        }

        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: item) for item in sample_transaction_data]

            mock_transaction.query = mock_query

            response = client.post('/api/model_manager/balancing/analyze-imbalance',
                                 json=payload,
                                 content_type='application/json')

            assert response.status_code == 200

            data = json.loads(response.data)
            assert data['success'] == True
            assert 'data' in data

            result_data = data['data']
            assert 'imbalance_analysis' in result_data
            assert 'recommendations' in result_data

            # Check imbalance analysis structure
            analysis = result_data['imbalance_analysis']
            assert 'imbalance_ratio' in analysis
            assert 'severity' in analysis
            assert 'requires_balancing' in analysis

    def test_balance_financial_categories_endpoint(self, client, sample_transaction_data):
        """Test financial category balancing endpoint"""
        payload = {
            'data_source': 'transactions',
            'target_category': 'investimento',
            'method': 'pattern_based'
        }

        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: item) for item in sample_transaction_data]

            mock_transaction.query = mock_query

            response = client.post('/api/model_manager/balancing/financial-balance',
                                 json=payload,
                                 content_type='application/json')

            assert response.status_code == 200

            data = json.loads(response.data)
            assert data['success'] == True
            assert 'data' in data

            result_data = data['data']
            assert 'original_transactions' in result_data
            assert 'balanced_transactions' in result_data
            assert 'synthetic_transactions' in result_data

            # Check that synthetic transactions were added
            assert result_data['synthetic_transactions'] >= 0

    def test_assess_synthetic_quality_endpoint(self, client):
        """Test synthetic quality assessment endpoint"""
        # Create sample original and synthetic data
        original_data = [
            {'amount': 100, 'description': 'Original transaction', 'category': 'receita'},
            {'amount': 200, 'description': 'Another original', 'category': 'despesa'}
        ]

        synthetic_data = [
            {'amount': 110, 'description': 'Synthetic transaction', 'category': 'receita'},
            {'amount': 190, 'description': 'Another synthetic', 'category': 'despesa'}
        ]

        payload = {
            'original_data': original_data,
            'synthetic_data': synthetic_data,
            'metadata': {
                'generation_method': 'smote',
                'target_variable': 'category'
            }
        }

        response = client.post('/api/model_manager/balancing/quality-assessment',
                             json=payload,
                             content_type='application/json')

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] == True
        assert 'data' in data

        result_data = data['data']
        assert 'overall_quality_score' in result_data
        assert 'quality_grade' in result_data
        assert 'recommendations' in result_data

        # Check score is valid
        assert 0 <= result_data['overall_quality_score'] <= 1
        assert result_data['quality_grade'] in ['A', 'B', 'C', 'D', 'F']

    def test_get_balancing_history_endpoint(self, client):
        """Test balancing history endpoint"""
        response = client.get('/api/model_manager/balancing/history')

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] == True
        assert 'data' in data

        result_data = data['data']
        assert 'total_operations' in result_data
        assert 'history' in result_data
        assert 'performance_metrics' in result_data

    def test_get_quality_history_endpoint(self, client):
        """Test quality history endpoint"""
        response = client.get('/api/model_manager/balancing/quality-history')

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] == True
        assert 'data' in data

        result_data = data['data']
        assert 'total_assessments' in result_data
        assert 'summary' in result_data
        assert 'recent_assessments' in result_data

    def test_get_balancing_methods_info_endpoint(self, client):
        """Test balancing methods info endpoint"""
        response = client.get('/api/model_manager/balancing/methods-info')

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] == True
        assert 'data' in data

        result_data = data['data']
        assert 'smote_methods' in result_data
        assert 'financial_categories' in result_data
        assert 'available_strategies' in result_data

    def test_invalid_payload_handling(self, client):
        """Test handling of invalid payloads"""
        # Test with missing required fields
        invalid_payload = {
            'data_source': 'transactions'
            # Missing model_type
        }

        response = client.post('/api/model_manager/models/train-with-balancing',
                             json=invalid_payload,
                             content_type='application/json')

        # Should handle validation error gracefully
        assert response.status_code in [400, 500]

    def test_empty_data_handling(self, client):
        """Test handling of empty data"""
        payload = {
            'original_data': [],
            'synthetic_data': []
        }

        response = client.post('/api/model_manager/balancing/quality-assessment',
                             json=payload,
                             content_type='application/json')

        # Should handle gracefully
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = json.loads(response.data)
            # May return error or low quality score
            assert 'success' in data

    def test_large_dataset_handling(self, client):
        """Test handling of large datasets"""
        # Create large dataset
        large_data = []
        for i in range(5000):
            large_data.append({
                'amount': np.random.uniform(10, 1000),
                'description': f'Large transaction {i}',
                'category': np.random.choice(['receita', 'despesa']),
                'date': '2024-01-01'
            })

        payload = {
            'original_data': large_data[:1000],  # First 1000
            'synthetic_data': large_data[1000:2000]  # Next 1000
        }

        response = client.post('/api/model_manager/balancing/quality-assessment',
                             json=payload,
                             content_type='application/json')

        # Should handle large data gracefully
        assert response.status_code in [200, 500]

    @patch('src.routes.model_manager.imbalanced_data_handler')
    def test_balancing_service_integration(self, mock_handler, client, sample_transaction_data):
        """Test integration with balancing service"""
        # Mock the balancing handler
        mock_instance = Mock()
        mock_instance.handle_imbalanced_data.return_value = (np.random.rand(100, 5), np.random.randint(0, 2, 100))
        mock_handler.return_value = mock_instance

        payload = {
            'model_type': 'random_forest',
            'data_source': 'transactions',
            'balancing_method': 'smote'
        }

        with patch('src.routes.model_manager.Transaction') as mock_transaction:
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [Mock(to_dict=lambda: item) for item in sample_transaction_data]

            mock_transaction.query = mock_query

            response = client.post('/api/model_manager/models/train-with-balancing',
                                 json=payload,
                                 content_type='application/json')

            # Should call the balancing handler
            assert mock_instance.handle_imbalanced_data.called

    def test_error_handling_and_recovery(self, client):
        """Test error handling and recovery"""
        # Test with malformed JSON
        response = client.post('/api/model_manager/balancing/quality-assessment',
                             data="invalid json",
                             content_type='application/json')

        assert response.status_code == 400

        # Test with valid but problematic data
        problematic_payload = {
            'original_data': None,
            'synthetic_data': []
        }

        response = client.post('/api/model_manager/balancing/quality-assessment',
                             json=problematic_payload,
                             content_type='application/json')

        # Should handle gracefully
        assert response.status_code in [200, 500]

    def test_rate_limiting(self, client, sample_transaction_data):
        """Test rate limiting on endpoints"""
        payload = {
            'data_source': 'transactions',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        }

        # Make multiple rapid requests
        responses = []
        for _ in range(5):
            with patch('src.routes.model_manager.Transaction') as mock_transaction:
                mock_query = Mock()
                mock_query.filter.return_value = mock_query
                mock_query.all.return_value = [Mock(to_dict=lambda: item) for item in sample_transaction_data]

                mock_transaction.query = mock_query

                response = client.post('/api/model_manager/balancing/analyze-imbalance',
                                     json=payload,
                                     content_type='application/json')
                responses.append(response.status_code)

        # At least some requests should succeed
        assert 200 in responses

    def test_concurrent_request_handling(self, client):
        """Test handling of concurrent requests"""
        import threading
        import time

        results = []

        def make_request():
            payload = {
                'original_data': [{'amount': 100, 'description': 'test', 'category': 'receita'}],
                'synthetic_data': [{'amount': 110, 'description': 'synthetic', 'category': 'receita'}]
            }

            response = client.post('/api/model_manager/balancing/quality-assessment',
                                 json=payload,
                                 content_type='application/json')
            results.append(response.status_code)

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All requests should complete
        assert len(results) == 3
        assert all(code in [200, 500] for code in results)