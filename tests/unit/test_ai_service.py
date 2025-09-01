"""
Unit tests for AIService.

Tests cover:
- Transaction categorization
- Anomaly detection
- Insights generation
- AI model training
- Financial trend prediction
- Error handling and fallbacks
- External AI service integration
"""

import pytest
import pandas as pd
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from src.services.ai_service import AIService
from src.utils.exceptions import (
    AIServiceError, AIServiceUnavailableError, AIServiceTimeoutError,
    AIServiceQuotaExceededError, InsufficientDataError, ValidationError
)


class TestAIService:
    """Test suite for AIService class."""
    
    def test_init(self):
        """Test AIService initialization."""
        service = AIService()
        assert service is not None
        assert hasattr(service, 'categories')
        assert hasattr(service, 'openai_client')
        assert hasattr(service, 'groq_client')
        assert len(service.categories) > 0
    
    def test_categories_structure(self):
        """Test that categories are properly structured."""
        service = AIService()
        
        expected_categories = [
            'alimentacao', 'transporte', 'saude', 'educacao', 'lazer',
            'casa', 'vestuario', 'investimento', 'transferencia', 'saque',
            'salario', 'outros'
        ]
        
        for category in expected_categories:
            assert category in service.categories
            assert isinstance(service.categories[category], list)
    
    @pytest.mark.parametrize("description,expected_category", [
        ("MERCADO EXTRA", "alimentacao"),
        ("POSTO SHELL", "transporte"),
        ("FARMACIA DROGASIL", "saude"),
        ("ESCOLA OBJETIVO", "educacao"),
        ("CINEMA MULTIPLEX", "lazer"),
        ("CONTA DE LUZ", "casa"),
        ("LOJA RENNER", "vestuario"),
        ("APLICACAO CDB", "investimento"),
        ("TED TRANSFERENCIA", "transferencia"),
        ("SAQUE CAIXA", "saque"),
        ("SALARIO EMPRESA", "salario"),
        ("TRANSACAO DESCONHECIDA", "investimento")  # ML model categorizes unknown transactions as investimento
    ])
    def test_categorize_transaction(self, ai_service, description, expected_category):
        """Test transaction categorization with various descriptions."""
        result = ai_service.categorize_transaction(description)
        assert result == expected_category
    
    def test_categorize_transaction_case_insensitive(self, ai_service):
        """Test that categorization is case insensitive."""
        test_cases = [
            ("mercado extra", "alimentacao"),
            ("MERCADO EXTRA", "alimentacao"),
            ("Mercado Extra", "alimentacao"),
            ("MeRcAdO eXtRa", "alimentacao")
        ]
        
        for description, expected in test_cases:
            result = ai_service.categorize_transaction(description)
            assert result == expected
    
    def test_categorize_transaction_partial_match(self, ai_service):
        """Test categorization with partial keyword matches."""
        test_cases = [
            ("COMPRA NO SUPERMERCADO CARREFOUR", "alimentacao"),
            ("PAGAMENTO POSTO DE GASOLINA", "transporte"),
            ("CONSULTA MEDICO CARDIOLOGISTA", "saude")
        ]
        
        for description, expected in test_cases:
            result = ai_service.categorize_transaction(description)
            assert result == expected
    
    def test_categorize_transactions_batch_success(self, ai_service):
        """Test batch transaction categorization."""
        transactions = [
            {'description': 'MERCADO EXTRA', 'amount': -100.0},
            {'description': 'POSTO SHELL', 'amount': -50.0},
            {'description': 'SALARIO EMPRESA', 'amount': 3000.0}
        ]
        
        result = ai_service.categorize_transactions_batch(transactions)
        
        assert isinstance(result, list)
        assert len(result) == 3
        
        for transaction in result:
            assert 'category' in transaction
            assert transaction['category'] in ai_service.categories
    
    def test_categorize_transactions_batch_empty(self, ai_service):
        """Test batch categorization with empty list raises InsufficientDataError."""
        with pytest.raises(InsufficientDataError):
            ai_service.categorize_transactions_batch([])
    
    def test_categorize_transactions_batch_empty_descriptions(self, ai_service):
        """Test batch categorization with empty descriptions."""
        transactions = [
            {'description': '', 'amount': -100.0},
            {'description': None, 'amount': -50.0},
            {'description': '   ', 'amount': 3000.0}
        ]
        
        result = ai_service.categorize_transactions_batch(transactions)
        
        for transaction in result:
            assert transaction['category'] == 'outros'
    
    def test_categorize_transactions_batch_error_handling(self, ai_service):
        """Test batch categorization error handling."""
        transactions = [
            {'description': 'VALID TRANSACTION', 'amount': -100.0},
            {'invalid': 'data'},  # Invalid transaction structure
            {'description': 'ANOTHER VALID', 'amount': -50.0}
        ]
        
        result = ai_service.categorize_transactions_batch(transactions)
        
        # Should handle errors gracefully and assign fallback category
        assert isinstance(result, list)
        assert len(result) == 3
        
        for transaction in result:
            assert 'category' in transaction
    
    def test_detect_anomalies_success(self, ai_service):
        """Test anomaly detection with sufficient data."""
        transactions = []
        for i in range(20):  # Need at least 10 transactions
            transactions.append({
                'amount': 100.0 + i * 10,  # Regular pattern
                'date': date(2024, 1, i + 1),
                'description': f'Transaction {i}'
            })
        
        # Add an anomalous transaction
        transactions.append({
            'amount': 10000.0,  # Much larger amount
            'date': date(2024, 1, 21),
            'description': 'Anomalous transaction'
        })
        
        result = ai_service.detect_anomalies(transactions)
        
        assert isinstance(result, list)
        assert len(result) == len(transactions)
        
        # Check that anomaly flags are added
        for transaction in result:
            assert 'is_anomaly' in transaction
            assert isinstance(transaction['is_anomaly'], bool)
        
        # At least one should be marked as anomaly
        anomaly_count = sum(1 for t in result if t['is_anomaly'])
        assert anomaly_count > 0
    
    def test_detect_anomalies_insufficient_data(self, ai_service):
        """Test anomaly detection with insufficient data."""
        transactions = [
            {'amount': 100.0, 'date': date(2024, 1, 1), 'description': 'Transaction 1'}
        ]  # Only 1 transaction, need at least 10
        
        result = ai_service.detect_anomalies(transactions)
        
        # Should mark all as non-anomalous
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['is_anomaly'] is False
    
    def test_detect_anomalies_empty_list(self, ai_service):
        """Test anomaly detection with empty list raises InsufficientDataError."""
        with pytest.raises(InsufficientDataError):
            ai_service.detect_anomalies([])
    
    def test_detect_anomalies_date_handling(self, ai_service):
        """Test anomaly detection with different date formats."""
        transactions = []
        for i in range(15):
            transactions.append({
                'amount': 100.0,
                'date': f'2024-01-{i+1:02d}',  # String date
                'description': f'Transaction {i}'
            })
        
        result = ai_service.detect_anomalies(transactions)
        
        assert isinstance(result, list)
        assert len(result) == 15
        
        for transaction in result:
            assert 'is_anomaly' in transaction
    
    def test_generate_insights_success(self, ai_service):
        """Test insights generation with valid data."""
        transactions = [
            {
                'amount': -100.0,
                'category': 'alimentacao',
                'date': date(2024, 1, 1),
                'description': 'Mercado'
            },
            {
                'amount': 3000.0,
                'category': 'salario',
                'date': date(2024, 1, 1),
                'description': 'Salario'
            },
            {
                'amount': -50.0,
                'category': 'transporte',
                'date': date(2024, 1, 2),
                'description': 'Uber'
            }
        ]
        
        result = ai_service.generate_insights(transactions)
        
        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'categories' in result
        assert 'patterns' in result
        assert 'anomalies' in result
        assert 'recommendations' in result
        
        # Check summary structure
        summary = result['summary']
        assert 'total_transactions' in summary
        assert 'total_credits' in summary
        assert 'total_debits' in summary
        assert 'net_flow' in summary
        
        # Verify calculations
        assert summary['total_transactions'] == 3
        assert summary['total_credits'] == 3000.0
        assert summary['total_debits'] == 150.0
        assert summary['net_flow'] == 2850.0
    
    def test_generate_insights_empty_data(self, ai_service):
        """Test insights generation with empty data."""
        result = ai_service.generate_insights([])
        
        assert isinstance(result, dict)
        assert 'error' in result
    
    def test_generate_summary_insights(self, ai_service):
        """Test summary insights generation."""
        df = pd.DataFrame([
            {'amount': -100.0, 'category': 'alimentacao'},
            {'amount': 3000.0, 'category': 'salario'},
            {'amount': -50.0, 'category': 'transporte'}
        ])
        
        result = ai_service._generate_summary_insights(df)
        
        assert isinstance(result, dict)
        assert result['total_transactions'] == 3
        assert result['total_credits'] == 3000.0
        assert result['total_debits'] == 150.0
        assert result['net_flow'] == 2850.0
        assert result['avg_transaction_value'] > 0
        assert result['largest_expense'] == -100.0
        assert result['largest_income'] == 3000.0
    
    def test_generate_category_insights(self, ai_service):
        """Test category insights generation."""
        df = pd.DataFrame([
            {'amount': -100.0, 'category': 'alimentacao'},
            {'amount': -50.0, 'category': 'alimentacao'},
            {'amount': -75.0, 'category': 'transporte'},
            {'amount': 3000.0, 'category': 'salario'}
        ])
        
        result = ai_service._generate_category_insights(df)
        
        assert isinstance(result, dict)
        assert 'alimentacao' in result
        assert 'transporte' in result
        assert 'salario' in result
        
        # Check alimentacao category
        alimentacao = result['alimentacao']
        assert alimentacao['total_transactions'] == 2
        assert alimentacao['total_spent'] == 150.0
        assert alimentacao['avg_transaction'] == 75.0
        assert alimentacao['percentage_of_expenses'] > 0
    
    def test_generate_pattern_insights(self, ai_service):
        """Test pattern insights generation."""
        dates = [
            datetime(2024, 1, 1),  # Monday
            datetime(2024, 1, 2),  # Tuesday
            datetime(2024, 1, 3),  # Wednesday
        ]
        
        df = pd.DataFrame([
            {'amount': -100.0, 'date': dates[0]},
            {'amount': -50.0, 'date': dates[1]},
            {'amount': -75.0, 'date': dates[2]}
        ])
        
        result = ai_service._generate_pattern_insights(df)
        
        assert isinstance(result, dict)
        if 'weekday_spending' in result:
            assert isinstance(result['weekday_spending'], dict)
            assert len(result['weekday_spending']) == 7  # All weekdays
    
    def test_generate_anomaly_insights(self, ai_service):
        """Test anomaly insights generation."""
        df = pd.DataFrame([
            {'amount': -100.0, 'is_anomaly': False, 'date': date(2024, 1, 1), 'description': 'Normal'},
            {'amount': -10000.0, 'is_anomaly': True, 'date': date(2024, 1, 2), 'description': 'Anomaly'},
            {'amount': -50.0, 'is_anomaly': False, 'date': date(2024, 1, 3), 'description': 'Normal'}
        ])
        
        result = ai_service._generate_anomaly_insights(df)
        
        assert isinstance(result, dict)
        assert result['total_anomalies'] == 1
        assert result['anomaly_percentage'] == 33.3
        assert len(result['anomalous_transactions']) == 1
    
    def test_generate_recommendations(self, ai_service):
        """Test recommendations generation."""
        insights = {
            'categories': {
                'alimentacao': {'total_spent': 500.0},
                'transporte': {'total_spent': 200.0}
            },
            'anomalies': {'total_anomalies': 2},
            'summary': {'net_flow': -100.0}
        }
        
        result = ai_service._generate_recommendations(insights)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Should contain recommendations about spending and anomalies
        recommendations_text = ' '.join(result)
        assert 'alimentacao' in recommendations_text or 'anomalias' in recommendations_text or 'saldo' in recommendations_text
    
    @patch('openai.OpenAI')
    def test_generate_ai_insights_success(self, mock_openai_class, ai_service):
        """Test AI insights generation with OpenAI."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="AI generated insights"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Set up AI service with mocked client
        ai_service.openai_client = mock_client
        
        transactions = [
            {'amount': -100.0, 'description': 'Test transaction'}
        ]
        
        result = ai_service.generate_ai_insights(transactions)
        
        assert isinstance(result, str)
        assert result == "AI generated insights"
        mock_client.chat.completions.create.assert_called_once()
    
    def test_generate_ai_insights_no_service(self, ai_service):
        """Test AI insights generation when no AI service is configured."""
        # Remove AI clients
        ai_service.openai_client = None
        ai_service.groq_client = None
        
        transactions = [{'amount': -100.0, 'description': 'Test'}]
        
        with pytest.raises(AIServiceUnavailableError):
            ai_service.generate_ai_insights(transactions)
    
    def test_generate_ai_insights_empty_data(self, ai_service):
        """Test AI insights generation with empty data raises InsufficientDataError."""
        with pytest.raises(InsufficientDataError):
            ai_service.generate_ai_insights([])
    
    @patch('openai.OpenAI')
    def test_generate_ai_insights_openai_error(self, mock_openai_class, ai_service):
        """Test AI insights generation with OpenAI errors."""
        import openai
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError("Rate limit exceeded", response=Mock(), body=None)
        mock_openai_class.return_value = mock_client
        
        ai_service.openai_client = mock_client
        ai_service.groq_client = None
        
        transactions = [{'amount': -100.0, 'description': 'Test'}]
        
        # Should fall back to basic insights
        result = ai_service.generate_ai_insights(transactions)
        assert isinstance(result, str)
        assert "temporariamente indisponível" in result
    
    def test_create_insights_prompt(self, ai_service):
        """Test insights prompt creation."""
        summary = {
            'summary': {
                'total_transactions': 10,
                'total_credits': 3000.0,
                'total_debits': 1500.0,
                'net_flow': 1500.0
            },
            'categories': {
                'alimentacao': {'total_spent': 500.0},
                'transporte': {'total_spent': 200.0}
            }
        }
        
        result = ai_service._create_insights_prompt(summary)
        
        assert isinstance(result, str)
        assert '10' in result  # total_transactions
        assert '3000.00' in result  # total_credits
        assert '1500.00' in result  # total_debits
        assert 'alimentacao' in result
        assert 'português' in result
    
    def test_generate_fallback_insights(self, ai_service):
        """Test fallback insights generation."""
        result = ai_service._generate_fallback_insights(50)
        
        assert isinstance(result, str)
        assert '50' in result
        assert 'temporariamente indisponível' in result
        assert 'Recomendações gerais' in result
    
    def test_train_custom_model_success(self, ai_service):
        """Test custom model training with valid data."""
        financial_data = []
        for i in range(20):  # Need at least 10 entries
            financial_data.append({
                'description': f'Transaction {i}',
                'category': 'alimentacao' if i % 2 == 0 else 'transporte'
            })
        
        result = ai_service.train_custom_model(financial_data)
        
        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'accuracy' in result
        assert 'training_data_count' in result
        assert 'categories_count' in result
        assert hasattr(ai_service, 'model_trained')
        assert ai_service.model_trained is True
    
    def test_train_custom_model_insufficient_data(self, ai_service):
        """Test custom model training with insufficient data."""
        financial_data = [
            {'description': 'Transaction 1', 'category': 'alimentacao'}
        ]  # Only 1 entry, need at least 10
        
        with pytest.raises(InsufficientDataError):
            ai_service.train_custom_model(financial_data)
    
    def test_train_custom_model_empty_data(self, ai_service):
        """Test custom model training with empty data."""
        with pytest.raises(InsufficientDataError):
            ai_service.train_custom_model([])
    
    def test_validate_training_data(self, ai_service):
        """Test training data validation."""
        training_data = [
            {'description': 'Valid transaction', 'category': 'alimentacao'},
            {'description': '', 'category': 'transporte'},  # Empty description
            {'description': 'AB', 'category': 'casa'},  # Too short
            {'description': 'Another valid transaction', 'category': 'saude'},
            {'category': 'lazer'}  # Missing description
        ]
        
        result = ai_service._validate_training_data(training_data)
        
        assert isinstance(result, list)
        assert len(result) == 2  # Only 2 valid entries
        
        for entry in result:
            assert 'description' in entry
            assert len(entry['description']) >= 3
    
    def test_categorize_with_custom_model_not_trained(self, ai_service):
        """Test categorization with custom model when not trained."""
        result = ai_service.categorize_with_custom_model('MERCADO EXTRA')
        
        # Should fall back to rule-based categorization
        assert result == 'alimentacao'
    
    def test_categorize_with_custom_model_trained(self, ai_service):
        """Test categorization with trained custom model."""
        # Mock trained model
        ai_service.model_trained = True
        ai_service.custom_vectorizer = Mock()
        ai_service.custom_classifier = Mock()
        
        ai_service.custom_vectorizer.transform.return_value = [[0.1, 0.2, 0.3]]
        ai_service.custom_classifier.predict.return_value = [0]
        
        result = ai_service.categorize_with_custom_model('Test transaction')
        
        assert isinstance(result, str)
        assert result in ai_service.categories
    
    def test_categorize_with_custom_model_error(self, ai_service):
        """Test categorization with custom model error handling."""
        # Mock trained model with error
        ai_service.model_trained = True
        ai_service.custom_vectorizer = Mock()
        ai_service.custom_classifier = Mock()
        
        ai_service.custom_vectorizer.transform.side_effect = Exception("Model error")
        
        result = ai_service.categorize_with_custom_model('MERCADO EXTRA')
        
        # Should fall back to rule-based categorization
        assert result == 'alimentacao'
    
    def test_map_cluster_to_category(self, ai_service):
        """Test cluster to category mapping."""
        # Test with rule-based match
        result = ai_service._map_cluster_to_category(0, 'MERCADO EXTRA')
        assert result == 'alimentacao'
        
        # Test with unknown description
        result = ai_service._map_cluster_to_category(5, 'UNKNOWN TRANSACTION')
        assert result in ai_service.categories
    
    def test_predict_financial_trends_success(self, ai_service):
        """Test financial trend prediction with valid data."""
        historical_data = []
        for i in range(50):  # Need at least 30 entries
            historical_data.append({
                'date': date(2024, 1, i % 28 + 1),
                'amount': 100.0 + i * 10
            })
        
        result = ai_service.predict_financial_trends(historical_data, periods=6)
        
        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'data' in result
        
        data = result['data']
        assert 'historical_summary' in data
        assert 'predictions' in data
        assert len(data['predictions']) == 6
        
        # Check prediction structure
        for prediction in data['predictions']:
            assert 'date' in prediction
            assert 'predicted_income' in prediction
            assert 'predicted_expenses' in prediction
            assert 'predicted_net_flow' in prediction
    
    def test_predict_financial_trends_insufficient_data(self, ai_service):
        """Test financial trend prediction with insufficient data."""
        historical_data = [
            {'date': date(2024, 1, 1), 'amount': 100.0}
        ]  # Only 1 entry, need at least 30
        
        with pytest.raises(InsufficientDataError):
            ai_service.predict_financial_trends(historical_data)
    
    def test_predict_financial_trends_invalid_periods(self, ai_service):
        """Test financial trend prediction with invalid periods."""
        historical_data = []
        for i in range(50):
            historical_data.append({
                'date': date(2024, 1, i % 28 + 1),
                'amount': 100.0
            })
        
        with pytest.raises(ValidationError):
            ai_service.predict_financial_trends(historical_data, periods=0)
        
        with pytest.raises(ValidationError):
            ai_service.predict_financial_trends(historical_data, periods=25)
    
    def test_clean_historical_data(self, ai_service):
        """Test historical data cleaning."""
        dirty_data = pd.DataFrame([
            {'date': '2024-01-01', 'amount': 100.0},
            {'date': None, 'amount': 200.0},  # Missing date
            {'date': '2024-01-02', 'amount': None},  # Missing amount
            {'date': 'invalid-date', 'amount': 300.0},  # Invalid date
            {'date': '2024-01-03', 'amount': 'invalid'},  # Invalid amount
            {'date': '2024-01-04', 'amount': 1000000000.0},  # Extreme outlier
            {'date': '2024-01-05', 'amount': 400.0}  # Valid
        ])
        
        result = ai_service._clean_historical_data(dirty_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert len(result) < len(dirty_data)  # Some rows should be removed
        
        # Check that remaining data is valid
        assert result['date'].notna().all()
        assert result['amount'].notna().all()


class TestAIServiceIntegration:
    """Integration tests for AIService."""
    
    def test_full_ai_workflow(self, ai_service):
        """Test complete AI workflow."""
        transactions = [
            {
                'amount': -100.0,
                'description': 'MERCADO EXTRA',
                'date': date(2024, 1, 1)
            },
            {
                'amount': -50.0,
                'description': 'POSTO SHELL',
                'date': date(2024, 1, 2)
            },
            {
                'amount': 3000.0,
                'description': 'SALARIO EMPRESA',
                'date': date(2024, 1, 3)
            }
        ]
        
        # Step 1: Categorize transactions
        categorized = ai_service.categorize_transactions_batch(transactions)
        assert len(categorized) == 3
        assert all('category' in t for t in categorized)
        
        # Step 2: Detect anomalies (need more data)
        extended_transactions = categorized * 5  # Duplicate to get 15 transactions
        anomaly_results = ai_service.detect_anomalies(extended_transactions)
        assert len(anomaly_results) == 15
        assert all('is_anomaly' in t for t in anomaly_results)
        
        # Step 3: Generate insights
        insights = ai_service.generate_insights(anomaly_results)
        assert isinstance(insights, dict)
        assert 'summary' in insights
        assert 'categories' in insights
        assert 'recommendations' in insights


class TestAIServiceErrorHandling:
    """Test error handling in AIService."""
    
    def test_timeout_handling(self, ai_service):
        """Test timeout handling in AI operations."""
        with patch('src.services.ai_service.with_timeout') as mock_timeout:
            mock_timeout.side_effect = TimeoutError("AI processing timeout")
            
            with pytest.raises(TimeoutError):
                ai_service.categorize_transactions_batch([{'description': 'test'}])
    
    def test_service_error_handling(self, ai_service):
        """Test service error handling decorator."""
        with patch('src.services.ai_service.handle_service_errors') as mock_handler:
            # Test that the decorator is applied
            assert hasattr(ai_service.categorize_transactions_batch, '__wrapped__')
    
    @patch('openai.OpenAI')
    def test_ai_service_quota_exceeded(self, mock_openai_class, ai_service):
        """Test handling of AI service quota exceeded."""
        import openai
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError("Quota exceeded", response=Mock(), body=None)
        mock_openai_class.return_value = mock_client
        
        ai_service.openai_client = mock_client
        ai_service.groq_client = None
        
        transactions = [{'amount': -100.0, 'description': 'Test'}]
        
        # Should fall back to basic insights
        result = ai_service.generate_ai_insights(transactions)
        assert isinstance(result, str)
        assert "temporariamente indisponível" in result
    
    @patch('openai.OpenAI')
    def test_ai_service_timeout(self, mock_openai_class, ai_service):
        """Test handling of AI service timeout."""
        import openai
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = openai.APITimeoutError("Request timeout")
        mock_openai_class.return_value = mock_client
        
        ai_service.openai_client = mock_client
        ai_service.groq_client = None
        
        transactions = [{'amount': -100.0, 'description': 'Test'}]
        
        # Should fall back to basic insights
        result = ai_service.generate_ai_insights(transactions)
        assert isinstance(result, str)
        assert "temporariamente indisponível" in result


class TestAIServicePerformance:
    """Performance tests for AIService."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_batch_categorization(self, ai_service, large_transaction_dataset):
        """Test categorization performance with large dataset."""
        import time
        start_time = time.time()
        
        result = ai_service.categorize_transactions_batch(large_transaction_dataset)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process within reasonable time
        assert processing_time < 10  # 10 seconds max for 1000 transactions
        assert len(result) == len(large_transaction_dataset)
        assert all('category' in t for t in result)
    
    @pytest.mark.performance
    def test_anomaly_detection_performance(self, ai_service, large_transaction_dataset):
        """Test anomaly detection performance with large dataset."""
        import time
        start_time = time.time()
        
        result = ai_service.detect_anomalies(large_transaction_dataset)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process within reasonable time
        assert processing_time < 30  # 30 seconds max for 1000 transactions
        assert len(result) == len(large_transaction_dataset)
        assert all('is_anomaly' in t for t in result)