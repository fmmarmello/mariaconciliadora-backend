"""
Integration tests for API endpoints.

Tests cover:
- File upload endpoints (OFX/XLSX)
- Transaction management endpoints
- Financial data endpoints
- Reconciliation endpoints
- AI service endpoints
- Error handling and validation
- Authentication and authorization
- Data integrity across requests
"""

import pytest
import json
import tempfile
import os
from datetime import date, datetime
from unittest.mock import patch, Mock
from src.models.transaction import Transaction, UploadHistory, ReconciliationRecord
from src.models.company_financial import CompanyFinancial


class TestFileUploadEndpoints:
    """Test file upload API endpoints."""
    
    @pytest.mark.integration
    def test_upload_ofx_success(self, client, database, temp_ofx_file):
        """Test successful OFX file upload."""
        with open(temp_ofx_file, 'rb') as f:
            response = client.post(
                '/api/upload-ofx',
                data={'file': (f, 'test.ofx')},
                content_type='multipart/form-data'
            )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'message' in data
        assert 'data' in data
        
        # Verify response structure
        response_data = data['data']
        assert 'bank_name' in response_data
        assert 'account_info' in response_data
        assert 'items_imported' in response_data
        assert 'duplicates_found' in response_data
        assert 'summary' in response_data
    
    @pytest.mark.integration
    def test_upload_ofx_no_file(self, client, database):
        """Test OFX upload without file."""
        response = client.post('/api/upload-ofx')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        
        assert data['error'] is True
        assert 'message' in data
    
    @pytest.mark.integration
    def test_upload_ofx_invalid_format(self, client, database):
        """Test OFX upload with invalid file format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"Not an OFX file")
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                response = client.post(
                    '/api/upload-ofx',
                    data={'file': (f, 'test.txt')},
                    content_type='multipart/form-data'
                )
            
            assert response.status_code == 422
            data = json.loads(response.data)
            
            assert data['error'] is True
            assert 'formato' in data['message'].lower()
        
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.integration
    def test_upload_ofx_duplicate_file(self, client, database, temp_ofx_file):
        """Test OFX upload with duplicate file."""
        # First upload
        with open(temp_ofx_file, 'rb') as f:
            response1 = client.post(
                '/api/upload-ofx',
                data={'file': (f, 'test.ofx')},
                content_type='multipart/form-data'
            )
        
        assert response1.status_code == 200
        
        # Second upload (duplicate)
        with open(temp_ofx_file, 'rb') as f:
            response2 = client.post(
                '/api/upload-ofx',
                data={'file': (f, 'test.ofx')},
                content_type='multipart/form-data'
            )
        
        assert response2.status_code == 409
        data = json.loads(response2.data)
        
        assert data['error'] is True
        assert 'duplicado' in data['message'].lower() or 'processado' in data['message'].lower()
    
    @pytest.mark.integration
    def test_upload_xlsx_success(self, client, database, temp_xlsx_file):
        """Test successful XLSX file upload."""
        with open(temp_xlsx_file, 'rb') as f:
            response = client.post(
                '/api/upload-xlsx',
                data={'file': (f, 'test.xlsx')},
                content_type='multipart/form-data'
            )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'message' in data
        assert 'data' in data
        
        # Verify response structure
        response_data = data['data']
        assert 'items_imported' in response_data
        assert 'items_incomplete' in response_data
        assert 'duplicates_found' in response_data
        assert 'duplicate_details' in response_data
    
    @pytest.mark.integration
    def test_upload_xlsx_invalid_format(self, client, database):
        """Test XLSX upload with invalid file format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"Not an XLSX file")
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                response = client.post(
                    '/api/upload-xlsx',
                    data={'file': (f, 'test.txt')},
                    content_type='multipart/form-data'
                )
            
            assert response.status_code == 400
            data = json.loads(response.data)
            
            assert 'error' in data
            assert 'xlsx' in data['error'].lower()
        
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.integration
    def test_upload_xlsx_corrected_data(self, client, database):
        """Test upload of corrected XLSX data."""
        corrected_data = {
            'entries': [
                {
                    'date': '2024-01-15',
                    'description': 'Corrected transaction 1',
                    'amount': -100.0,
                    'category': 'alimentacao',
                    'transaction_type': 'expense'
                },
                {
                    'date': '2024-01-16',
                    'description': 'Corrected transaction 2',
                    'amount': 200.0,
                    'category': 'receita',
                    'transaction_type': 'income'
                }
            ]
        }
        
        response = client.post(
            '/api/upload-xlsx-corrected',
            data=json.dumps(corrected_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert data['data']['items_imported'] == 2


class TestTransactionEndpoints:
    """Test transaction management API endpoints."""
    
    @pytest.mark.integration
    def test_get_transactions_success(self, client, database, create_test_transactions, db_session):
        """Test successful transaction retrieval."""
        # Create test transactions
        transactions = create_test_transactions(5, db_session)
        
        response = client.get('/api/transactions')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'data' in data
        
        response_data = data['data']
        assert 'transactions' in response_data
        assert 'total_count' in response_data
        assert 'page' in response_data
        assert 'per_page' in response_data
        assert 'total_pages' in response_data
        
        assert len(response_data['transactions']) <= 5
        assert response_data['total_count'] >= 5
    
    @pytest.mark.integration
    def test_get_transactions_with_filters(self, client, database, db_session):
        """Test transaction retrieval with filters."""
        # Create test transactions with specific attributes
        transaction1 = Transaction(
            bank_name='BANCO_A',
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST TRANSACTION',
            transaction_type='debit',
            category='alimentacao'
        )
        transaction2 = Transaction(
            bank_name='BANCO_B',
            account_id='67890',
            date=date(2024, 1, 16),
            amount=200.0,
            description='ANOTHER TRANSACTION',
            transaction_type='credit',
            category='salario'
        )
        
        db_session.add(transaction1)
        db_session.add(transaction2)
        db_session.commit()
        
        # Test bank filter
        response = client.get('/api/transactions?bank=BANCO_A')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        transactions = data['data']['transactions']
        assert all(t['bank_name'] == 'BANCO_A' for t in transactions)
        
        # Test category filter
        response = client.get('/api/transactions?category=alimentacao')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        transactions = data['data']['transactions']
        assert all(t['category'] == 'alimentacao' for t in transactions)
        
        # Test transaction type filter
        response = client.get('/api/transactions?type=credit')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        transactions = data['data']['transactions']
        assert all(t['transaction_type'] == 'credit' for t in transactions)
    
    @pytest.mark.integration
    def test_get_transactions_with_date_range(self, client, database, db_session):
        """Test transaction retrieval with date range filters."""
        # Create transactions with different dates
        transaction1 = Transaction(
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 10),
            amount=-100.0,
            description='OLD TRANSACTION',
            transaction_type='debit'
        )
        transaction2 = Transaction(
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 20),
            amount=-200.0,
            description='NEW TRANSACTION',
            transaction_type='debit'
        )
        
        db_session.add(transaction1)
        db_session.add(transaction2)
        db_session.commit()
        
        # Test date range filter
        response = client.get('/api/transactions?start_date=2024-01-15&end_date=2024-01-25')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        transactions = data['data']['transactions']
        for transaction in transactions:
            transaction_date = datetime.fromisoformat(transaction['date']).date()
            assert date(2024, 1, 15) <= transaction_date <= date(2024, 1, 25)
    
    @pytest.mark.integration
    def test_get_transactions_pagination(self, client, database, create_test_transactions, db_session):
        """Test transaction pagination."""
        # Create many test transactions
        transactions = create_test_transactions(25, db_session)
        
        # Test first page
        response = client.get('/api/transactions?page=1&per_page=10')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['data']['page'] == 1
        assert data['data']['per_page'] == 10
        assert len(data['data']['transactions']) <= 10
        assert data['data']['total_pages'] >= 3
        
        # Test second page
        response = client.get('/api/transactions?page=2&per_page=10')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['data']['page'] == 2
        assert len(data['data']['transactions']) <= 10
    
    @pytest.mark.integration
    def test_get_transactions_invalid_pagination(self, client, database):
        """Test transaction retrieval with invalid pagination parameters."""
        # Test invalid page
        response = client.get('/api/transactions?page=0')
        assert response.status_code == 400
        
        # Test invalid per_page
        response = client.get('/api/transactions?per_page=200')
        assert response.status_code == 400
    
    @pytest.mark.integration
    def test_get_summary(self, client, database, db_session):
        """Test transaction summary endpoint."""
        # Create test transactions
        transaction1 = Transaction(
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='EXPENSE',
            transaction_type='debit',
            category='alimentacao'
        )
        transaction2 = Transaction(
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 16),
            amount=300.0,
            description='INCOME',
            transaction_type='credit',
            category='salario'
        )
        
        db_session.add(transaction1)
        db_session.add(transaction2)
        db_session.commit()
        
        response = client.get('/api/summary')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'data' in data
        
        summary_data = data['data']
        assert 'overview' in summary_data
        assert 'banks' in summary_data
        assert 'categories' in summary_data
        assert 'recent_transactions' in summary_data
        
        # Verify overview calculations
        overview = summary_data['overview']
        assert overview['total_transactions'] >= 2
        assert overview['total_credits'] >= 300.0
        assert overview['total_debits'] >= 100.0
        assert overview['net_flow'] >= 200.0
    
    @pytest.mark.integration
    def test_get_upload_history(self, client, database, db_session):
        """Test upload history endpoint."""
        # Create test upload history
        upload1 = UploadHistory(
            filename='test1.ofx',
            bank_name='TEST_BANK',
            transactions_count=10,
            status='success'
        )
        upload2 = UploadHistory(
            filename='test2.xlsx',
            bank_name='xlsx_file',
            transactions_count=5,
            status='success'
        )
        
        db_session.add(upload1)
        db_session.add(upload2)
        db_session.commit()
        
        response = client.get('/api/upload-history')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'data' in data
        assert 'uploads' in data['data']
        
        uploads = data['data']['uploads']
        assert len(uploads) >= 2
        
        # Verify upload structure
        for upload in uploads:
            assert 'filename' in upload
            assert 'bank_name' in upload
            assert 'transactions_count' in upload
            assert 'status' in upload
            assert 'upload_date' in upload


class TestInsightsEndpoints:
    """Test insights and AI API endpoints."""
    
    @pytest.mark.integration
    def test_get_insights(self, client, database, db_session):
        """Test insights generation endpoint."""
        # Create test transactions
        transactions = []
        for i in range(10):
            transaction = Transaction(
                bank_name='TEST_BANK',
                account_id='12345',
                date=date(2024, 1, i + 1),
                amount=-100.0 - i * 10,
                description=f'TRANSACTION {i}',
                transaction_type='debit',
                category='alimentacao' if i % 2 == 0 else 'transporte'
            )
            transactions.append(transaction)
            db_session.add(transaction)
        
        db_session.commit()
        
        response = client.get('/api/insights')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'data' in data
        
        insights = data['data']
        assert 'summary' in insights
        assert 'categories' in insights
        assert 'patterns' in insights
        assert 'recommendations' in insights
    
    @pytest.mark.integration
    def test_get_insights_with_filters(self, client, database, db_session):
        """Test insights generation with filters."""
        # Create transactions for different banks
        transaction1 = Transaction(
            bank_name='BANCO_A',
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='BANCO A TRANSACTION',
            transaction_type='debit',
            category='alimentacao'
        )
        transaction2 = Transaction(
            bank_name='BANCO_B',
            account_id='67890',
            date=date(2024, 1, 16),
            amount=-200.0,
            description='BANCO B TRANSACTION',
            transaction_type='debit',
            category='transporte'
        )
        
        db_session.add(transaction1)
        db_session.add(transaction2)
        db_session.commit()
        
        # Test insights for specific bank
        response = client.get('/api/insights?bank=BANCO_A')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        insights = data['data']
        
        # Should only include BANCO_A transactions
        assert insights['summary']['total_transactions'] >= 1
    
    @pytest.mark.integration
    @patch('src.services.ai_service.AIService.generate_ai_insights')
    def test_get_ai_insights(self, mock_ai_insights, client, database, db_session):
        """Test AI insights generation endpoint."""
        # Mock AI service response
        mock_ai_insights.return_value = "AI generated insights for your transactions"
        
        # Create test transactions
        transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST TRANSACTION',
            transaction_type='debit',
            category='alimentacao'
        )
        db_session.add(transaction)
        db_session.commit()
        
        response = client.get('/api/ai-insights')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'data' in data
        assert 'ai_insights' in data['data']
        assert data['data']['ai_insights'] == "AI generated insights for your transactions"
        
        mock_ai_insights.assert_called_once()
    
    @pytest.mark.integration
    def test_get_insights_insufficient_data(self, client, database):
        """Test insights generation with insufficient data."""
        response = client.get('/api/insights')
        assert response.status_code == 400
        data = json.loads(response.data)
        
        assert data['error'] is True
        assert 'insuficientes' in data['message'].lower() or 'insufficient' in data['message'].lower()


class TestCompanyFinancialEndpoints:
    """Test company financial data API endpoints."""
    
    @pytest.mark.integration
    def test_get_company_financial(self, client, database, db_session):
        """Test company financial data retrieval."""
        # Create test financial entries
        entry1 = CompanyFinancial(
            date=date(2024, 1, 15),
            description='DESPESA TESTE',
            amount=-100.0,
            category='alimentacao',
            transaction_type='expense'
        )
        entry2 = CompanyFinancial(
            date=date(2024, 1, 16),
            description='RECEITA TESTE',
            amount=200.0,
            category='vendas',
            transaction_type='income'
        )
        
        db_session.add(entry1)
        db_session.add(entry2)
        db_session.commit()
        
        response = client.get('/api/company-financial')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'data' in data
        
        response_data = data['data']
        assert 'entries' in response_data
        assert 'total_count' in response_data
        
        entries = response_data['entries']
        assert len(entries) >= 2
        
        # Verify entry structure
        for entry in entries:
            assert 'date' in entry
            assert 'description' in entry
            assert 'amount' in entry
            assert 'transaction_type' in entry
    
    @pytest.mark.integration
    def test_get_company_financial_with_filters(self, client, database, db_session):
        """Test company financial data retrieval with filters."""
        # Create test entries
        expense_entry = CompanyFinancial(
            date=date(2024, 1, 15),
            description='DESPESA',
            amount=-100.0,
            category='alimentacao',
            transaction_type='expense'
        )
        income_entry = CompanyFinancial(
            date=date(2024, 1, 16),
            description='RECEITA',
            amount=200.0,
            category='vendas',
            transaction_type='income'
        )
        
        db_session.add(expense_entry)
        db_session.add(income_entry)
        db_session.commit()
        
        # Test type filter
        response = client.get('/api/company-financial?type=expense')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        entries = data['data']['entries']
        assert all(entry['transaction_type'] == 'expense' for entry in entries)
        
        # Test category filter
        response = client.get('/api/company-financial?category=alimentacao')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        entries = data['data']['entries']
        assert all(entry['category'] == 'alimentacao' for entry in entries)
    
    @pytest.mark.integration
    def test_get_company_financial_summary(self, client, database, db_session):
        """Test company financial summary endpoint."""
        # Create test entries
        entry1 = CompanyFinancial(
            date=date(2024, 1, 15),
            description='DESPESA 1',
            amount=-100.0,
            category='alimentacao',
            transaction_type='expense'
        )
        entry2 = CompanyFinancial(
            date=date(2024, 1, 16),
            description='RECEITA 1',
            amount=300.0,
            category='vendas',
            transaction_type='income'
        )
        
        db_session.add(entry1)
        db_session.add(entry2)
        db_session.commit()
        
        response = client.get('/api/company-financial/summary')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'data' in data
        
        summary_data = data['data']
        assert 'overview' in summary_data
        assert 'categories' in summary_data
        assert 'recent_entries' in summary_data
        
        # Verify overview calculations
        overview = summary_data['overview']
        assert overview['total_entries'] >= 2
        assert overview['total_income'] >= 300.0
        assert overview['total_expenses'] >= 100.0
        assert overview['net_flow'] >= 200.0


class TestAITrainingEndpoints:
    """Test AI training and prediction API endpoints."""
    
    @pytest.mark.integration
    @patch('src.services.ai_service.AIService.train_custom_model')
    def test_train_ai_model(self, mock_train, client, database, db_session):
        """Test AI model training endpoint."""
        # Mock training response
        mock_train.return_value = {
            'success': True,
            'accuracy': 0.85,
            'training_data_count': 20,
            'categories_count': 5
        }
        
        # Create training data
        for i in range(20):
            entry = CompanyFinancial(
                date=date(2024, 1, i + 1),
                description=f'TRAINING ENTRY {i}',
                amount=-100.0 - i,
                category='alimentacao' if i % 2 == 0 else 'transporte',
                transaction_type='expense'
            )
            db_session.add(entry)
        
        db_session.commit()
        
        response = client.post('/api/ai/train')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'accuracy' in data
        assert data['accuracy'] == 0.85
        
        mock_train.assert_called_once()
    
    @pytest.mark.integration
    @patch('src.services.ai_service.AIService.categorize_with_custom_model')
    def test_categorize_financial_with_ai(self, mock_categorize, client, database):
        """Test AI categorization endpoint."""
        # Mock categorization response
        mock_categorize.return_value = 'alimentacao'
        
        request_data = {
            'description': 'MERCADO EXTRA COMPRAS'
        }
        
        response = client.post(
            '/api/ai/categorize-financial',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'data' in data
        assert data['data']['category'] == 'alimentacao'
        
        mock_categorize.assert_called_once_with('MERCADO EXTRA COMPRAS')
    
    @pytest.mark.integration
    @patch('src.services.ai_service.AIService.predict_financial_trends')
    def test_get_financial_predictions(self, mock_predict, client, database, db_session):
        """Test financial predictions endpoint."""
        # Mock prediction response
        mock_predict.return_value = {
            'success': True,
            'data': {
                'historical_summary': {
                    'total_income': 5000.0,
                    'total_expenses': 3000.0,
                    'net_flow': 2000.0,
                    'period_months': 6
                },
                'predictions': [
                    {
                        'date': '2024-07',
                        'predicted_income': 1000.0,
                        'predicted_expenses': 600.0,
                        'predicted_net_flow': 400.0
                    }
                ]
            }
        }
        
        # Create historical data
        for i in range(50):
            entry = CompanyFinancial(
                date=date(2024, 1, i % 28 + 1),
                description=f'HISTORICAL ENTRY {i}',
                amount=100.0 + i * 10,
                transaction_type='income' if i % 2 == 0 else 'expense'
            )
            db_session.add(entry)
        
        db_session.commit()
        
        response = client.get('/api/ai/predictions?periods=6')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'data' in data
        
        prediction_data = data['data']
        assert 'historical_summary' in prediction_data
        assert 'predictions' in prediction_data
        
        mock_predict.assert_called_once()


class TestReconciliationEndpoints:
    """Test reconciliation API endpoints."""
    
    @pytest.mark.integration
    def test_start_reconciliation(self, client, database, db_session):
        """Test reconciliation start endpoint."""
        # Create test data
        bank_transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='PAGAMENTO FORNECEDOR',
            transaction_type='debit'
        )
        
        company_entry = CompanyFinancial(
            date=date(2024, 1, 15),
            description='PAGAMENTO FORNECEDOR',
            amount=-100.0,
            transaction_type='expense'
        )
        
        db_session.add(bank_transaction)
        db_session.add(company_entry)
        db_session.commit()
        
        response = client.post('/api/reconciliation')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'message' in data
        assert 'data' in data
        
        reconciliation_data = data['data']
        assert 'matches_count' in reconciliation_data
        assert 'bank_transactions_count' in reconciliation_data
        assert 'company_entries_count' in reconciliation_data
    
    @pytest.mark.integration
    def test_get_pending_reconciliations(self, client, database, db_session):
        """Test pending reconciliations endpoint."""
        # Create test reconciliation record
        bank_transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST',
            transaction_type='debit'
        )
        
        company_entry = CompanyFinancial(
            date=date(2024, 1, 15),
            description='TEST',
            amount=-100.0,
            transaction_type='expense'
        )
        
        db_session.add(bank_transaction)
        db_session.add(company_entry)
        db_session.commit()
        
        reconciliation_record = ReconciliationRecord(
            bank_transaction_id=bank_transaction.id,
            company_entry_id=company_entry.id,
            match_score=0.85,
            status='pending'
        )
        
        db_session.add(reconciliation_record)
        db_session.commit()
        
        response = client.get('/api/reconciliation/pending')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'data' in data
        assert 'records' in data['data']
        
        records = data['data']['records']
        assert len(records) >= 1
        
        # Verify record structure
        for record in records:
            assert 'id' in record
            assert 'match_score' in record
            assert 'status' in record
            assert record['status'] == 'pending'
    
    @pytest.mark.integration
    def test_confirm_reconciliation(self, client, database, db_session):
        """Test reconciliation confirmation endpoint."""
        # Create test reconciliation record
        bank_transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST',
            transaction_type='debit'
        )
        
        company_entry = CompanyFinancial(
            date=date(2024, 1, 15),
            description='TEST',
            amount=-100.0,
            transaction_type='expense'
        )
        
        db_session.add(bank_transaction)
        db_session.add(company_entry)
        db_session.commit()
        
        reconciliation_record = ReconciliationRecord(
            bank_transaction_id=bank_transaction.id,
            company_entry_id=company_entry.id,
            match_score=0.85,
            status='pending'
        )
        
        db_session.add(reconciliation_record)
        db_session.commit()
        
        # Confirm reconciliation
        response = client.post(f'/api/reconciliation/{reconciliation_record.id}/confirm')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'message' in data
        assert data['data']['reconciliation_id'] == reconciliation_record.id
        
        # Verify status changed in database
        db_session.refresh(reconciliation_record)
        assert reconciliation_record.status == 'confirmed'
    
    @pytest.mark.integration
    def test_reject_reconciliation(self, client, database, db_session):
        """Test reconciliation rejection endpoint."""
        # Create test reconciliation record
        bank_transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST',
            transaction_type='debit'
        )
        
        company_entry = CompanyFinancial(
            date=date(2024, 1, 15),
            description='TEST',
            amount=-100.0,
            transaction_type='expense'
        )
        
        db_session.add(bank_transaction)
        db_session.add(company_entry)
        db_session.commit()
        
        reconciliation_record = ReconciliationRecord(
            bank_transaction_id=bank_transaction.id,
            company_entry_id=company_entry.id,
            match_score=0.85,
            status='pending'
        )
        
        db_session.add(reconciliation_record)
        db_session.commit()
        
        # Reject reconciliation
        response = client.post(f'/api/reconciliation/{reconciliation_record.id}/reject')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'message' in data
        assert data['data']['reconciliation_id'] == reconciliation_record.id
        
        # Verify status changed in database
        db_session.refresh(reconciliation_record)
        assert reconciliation_record.status == 'rejected'
    
    @pytest.mark.integration
    def test_get_reconciliation_report(self, client, database, db_session):
        """Test reconciliation report endpoint."""
        # Create test reconciliation records
        bank_transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST',
            transaction_type='debit'
        )
        
        company_entry = CompanyFinancial(
            date=date(2024, 1, 15),
            description='TEST',
            amount=-100.0,
            transaction_type='expense'
        )
        
        db_session.add(bank_transaction)
        db_session.add(company_entry)
        db_session.commit()
        
        # Create confirmed reconciliation
        confirmed_record = ReconciliationRecord(
            bank_transaction_id=bank_transaction.id,
            company_entry_id=company_entry.id,
            match_score=0.85,
            status='confirmed'
        )
        
        db_session.add(confirmed_record)
        db_session.commit()
        
        response = client.get('/api/reconciliation/report')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'data' in data
        
        report_data = data['data']
        assert 'summary' in report_data
        assert 'financials' in report_data
        assert 'recent_activity' in report_data
        assert 'generated_at' in report_data
        
        # Verify summary structure
        summary = report_data['summary']
        assert 'total_records' in summary
        assert 'confirmed' in summary
        assert 'pending' in summary
        assert 'rejected' in summary
        assert 'reconciliation_rate' in summary


class TestErrorHandlingEndpoints:
    """Test error handling in API endpoints."""
    
    @pytest.mark.integration
    def test_invalid_json_request(self, client, database):
        """Test handling of invalid JSON requests."""
        response = client.post(
            '/api/upload-xlsx-corrected',
            data='invalid json',
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    @pytest.mark.integration
    def test_missing_required_fields(self, client, database):
        """Test handling of missing required fields."""
        # Test categorization without description
        response = client.post(
            '/api/ai/categorize-financial',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    @pytest.mark.integration
    def test_invalid_reconciliation_id(self, client, database):
        """Test handling of invalid reconciliation IDs."""
        # Test with non-existent ID
        response = client.post('/api/reconciliation/99999/confirm')
        assert response.status_code == 404
        
        # Test with invalid ID format
        response = client.post('/api/reconciliation/invalid/confirm')
        assert response.status_code == 404
    
    @pytest.mark.integration
    def test_database_error_handling(self, client, database):
        """Test handling of database errors."""
        # This would require mocking database failures
        # For now, we test that the endpoints handle empty databases gracefully
        
        response = client.get('/api/transactions')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['data']['total_count'] == 0
    
    @pytest.mark.integration
    def test_file_size_limit(self, client, database):
        """Test file size limit handling."""
        # Create a large file (simulate)
        large_content = b"x" * (20 * 1024 * 1024)  # 20MB
        
        with tempfile.NamedTemporaryFile(suffix='.ofx', delete=False) as f:
            f.write(large_content)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                response = client.post(
                    '/api/upload-ofx',
                    data={'file': (f, 'large.ofx')},
                    content_type='multipart/form-data'
                )
            
            # Should handle large files gracefully (either accept or reject with proper error)
            assert response.status_code in [200, 413, 422]
            
            if response.status_code != 200:
                data = json.loads(response.data)
                assert 'error' in data or data.get('success') is False
        
        finally:
            os.unlink(temp_path)


class TestSecurityEndpoints:
    """Test security aspects of API endpoints."""
    
    @pytest.mark.integration
    def test_sql_injection_protection(self, client, database, db_session):
        """Test protection against SQL injection attacks."""
        # Create a test transaction
        transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST TRANSACTION',
            transaction_type='debit'
        )
        db_session.add(transaction)
        db_session.commit()
        
        # Test SQL injection in query parameters
        malicious_queries = [
            "'; DROP TABLE transactions; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM transactions --",
            "<script>alert('xss')</script>"
        ]
        
        for malicious_query in malicious_queries:
            response = client.get(f'/api/transactions?bank={malicious_query}')
            
            # Should not cause server error and should handle gracefully
            assert response.status_code in [200, 400]
            
            if response.status_code == 200:
                data = json.loads(response.data)
                assert data['success'] is True
                # Should return empty results or filter properly
    
    @pytest.mark.integration
    def test_xss_protection(self, client, database):
        """Test protection against XSS attacks."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for payload in xss_payloads:
            # Test in corrected data upload
            corrected_data = {
                'entries': [
                    {
                        'date': '2024-01-15',
                        'description': payload,
                        'amount': -100.0,
                        'category': 'test',
                        'transaction_type': 'expense'
                    }
                ]
            }
            
            response = client.post(
                '/api/upload-xlsx-corrected',
                data=json.dumps(corrected_data),
                content_type='application/json'
            )
            
            # Should handle XSS attempts gracefully
            assert response.status_code in [200, 400]
    
    @pytest.mark.integration
    def test_file_type_validation(self, client, database):
        """Test file type validation security."""
        # Test with potentially dangerous file types
        dangerous_files = [
            ('malicious.exe', b'MZ\x90\x00'),  # Executable
            ('script.php', b'<?php echo "test"; ?>'),  # PHP script
            ('test.html', b'<html><script>alert("xss")</script></html>'),  # HTML with script
        ]
        
        for filename, content in dangerous_files:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(content)
                temp_path = f.name
            
            try:
                with open(temp_path, 'rb') as f:
                    response = client.post(
                        '/api/upload-ofx',
                        data={'file': (f, filename)},
                        content_type='multipart/form-data'
                    )
                
                # Should reject dangerous file types
                assert response.status_code in [400, 422]
                data = json.loads(response.data)
                assert data.get('success') is False or 'error' in data
            
            finally:
                os.unlink(temp_path)


class TestPerformanceEndpoints:
    """Test performance aspects of API endpoints."""
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_large_dataset_performance(self, client, database, create_test_transactions, db_session):
        """Test API performance with large datasets."""
        # Create a large number of transactions
        transactions = create_test_transactions(1000, db_session)
        
        import time
        
        # Test transaction listing performance
        start_time = time.time()
        response = client.get('/api/transactions?per_page=100')
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 5  # Should respond within 5 seconds
        
        # Test summary generation performance
        start_time = time.time()
        response = client.get('/api/summary')
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 10  # Should respond within 10 seconds
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_concurrent_requests(self, client, database):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get('/api/summary')
            results.append(response.status_code)
        
        # Create multiple threads to simulate concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 10
        
        # Should handle concurrent requests reasonably fast
        assert (end_time - start_time) < 30  # 30 seconds for 10 concurrent requests