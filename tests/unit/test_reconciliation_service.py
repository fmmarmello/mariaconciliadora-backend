"""
Unit tests for ReconciliationService.

Tests cover:
- Transaction matching algorithms
- Match scoring and validation
- Reconciliation record management
- Financial accuracy validation
- Error handling and edge cases
- Performance with large datasets
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from src.services.reconciliation_service import ReconciliationService
from src.models.transaction import Transaction, ReconciliationRecord
from src.models.company_financial import CompanyFinancial
from src.utils.exceptions import (
    ReconciliationError, DatabaseError, RecordNotFoundError,
    InsufficientDataError, ValidationError
)


class TestReconciliationService:
    """Test suite for ReconciliationService class."""
    
    def test_init(self):
        """Test ReconciliationService initialization."""
        service = ReconciliationService()
        assert service is not None
    
    def test_find_matches_success(self, reconciliation_service, db_session):
        """Test successful match finding between bank transactions and company entries."""
        # Create test bank transactions
        bank_transactions = [
            Transaction(
                id=1,
                bank_name='TEST_BANK',
                account_id='12345',
                date=date(2024, 1, 15),
                amount=-100.0,
                description='PAGAMENTO FORNECEDOR A'
            ),
            Transaction(
                id=2,
                bank_name='TEST_BANK',
                account_id='12345',
                date=date(2024, 1, 16),
                amount=2500.0,
                description='RECEITA VENDAS'
            )
        ]
        
        # Create test company entries
        company_entries = [
            CompanyFinancial(
                id=1,
                date=date(2024, 1, 15),
                amount=-100.0,
                description='PAGAMENTO FORNECEDOR A',
                transaction_type='expense'
            ),
            CompanyFinancial(
                id=2,
                date=date(2024, 1, 16),
                amount=2500.0,
                description='RECEITA VENDAS',
                transaction_type='income'
            ),
            CompanyFinancial(
                id=3,
                date=date(2024, 1, 17),
                amount=-50.0,
                description='DESPESA SEM MATCH',
                transaction_type='expense'
            )
        ]
        
        matches = reconciliation_service.find_matches(bank_transactions, company_entries)
        
        assert isinstance(matches, list)
        assert len(matches) == 2  # Should find 2 matches
        
        for match in matches:
            assert 'bank_transaction' in match
            assert 'company_entry' in match
            assert 'match_score' in match
            assert match['match_score'] >= 0.7  # Minimum threshold
    
    def test_find_matches_empty_bank_transactions(self, reconciliation_service):
        """Test match finding with empty bank transactions raises InsufficientDataError."""
        company_entries = [
            CompanyFinancial(
                id=1,
                date=date(2024, 1, 15),
                amount=-100.0,
                description='TEST',
                transaction_type='expense'
            )
        ]
        
        with pytest.raises(InsufficientDataError):
            reconciliation_service.find_matches([], company_entries)
    
    def test_find_matches_empty_company_entries(self, reconciliation_service):
        """Test match finding with empty company entries raises InsufficientDataError."""
        bank_transactions = [
            Transaction(
                id=1,
                bank_name='TEST_BANK',
                account_id='12345',
                date=date(2024, 1, 15),
                amount=-100.0,
                description='TEST'
            )
        ]
        
        with pytest.raises(InsufficientDataError):
            reconciliation_service.find_matches(bank_transactions, [])
    
    def test_find_matches_no_matches(self, reconciliation_service):
        """Test match finding when no matches are found."""
        bank_transactions = [
            Transaction(
                id=1,
                bank_name='TEST_BANK',
                account_id='12345',
                date=date(2024, 1, 15),
                amount=-100.0,
                description='BANK TRANSACTION'
            )
        ]
        
        company_entries = [
            CompanyFinancial(
                id=1,
                date=date(2024, 2, 15),  # Different month
                amount=-200.0,  # Different amount
                description='COMPANY ENTRY',  # Different description
                transaction_type='expense'
            )
        ]
        
        matches = reconciliation_service.find_matches(bank_transactions, company_entries)
        
        assert isinstance(matches, list)
        assert len(matches) == 0
    
    def test_validate_transactions_for_matching(self, reconciliation_service):
        """Test validation of bank transactions for matching."""
        valid_transaction = Transaction(
            id=1,
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='VALID TRANSACTION'
        )
        
        invalid_transactions = [
            Transaction(id=None, date=date(2024, 1, 15), amount=-100.0, description='No ID'),
            Transaction(id=2, date=None, amount=-100.0, description='No date'),
            Transaction(id=3, date=date(2024, 1, 15), amount=None, description='No amount'),
            Transaction(id=4, date=date(2024, 1, 15), amount=-100.0, description=None),
            Transaction(id=5, date=date(2024, 1, 15), amount=0.001, description='Too small')  # Very small amount
        ]
        
        all_transactions = [valid_transaction] + invalid_transactions
        
        result = reconciliation_service._validate_transactions_for_matching(all_transactions)
        
        assert isinstance(result, list)
        assert len(result) == 1  # Only valid transaction should remain
        assert result[0] == valid_transaction
    
    def test_validate_entries_for_matching(self, reconciliation_service):
        """Test validation of company entries for matching."""
        valid_entry = CompanyFinancial(
            id=1,
            date=date(2024, 1, 15),
            amount=-100.0,
            description='VALID ENTRY',
            transaction_type='expense'
        )
        
        invalid_entries = [
            CompanyFinancial(id=None, date=date(2024, 1, 15), amount=-100.0, description='No ID'),
            CompanyFinancial(id=2, date=None, amount=-100.0, description='No date'),
            CompanyFinancial(id=3, date=date(2024, 1, 15), amount=None, description='No amount'),
            CompanyFinancial(id=4, date=date(2024, 1, 15), amount=-100.0, description=None),
            CompanyFinancial(id=5, date=date(2024, 1, 15), amount=0.001, description='Too small')
        ]
        
        all_entries = [valid_entry] + invalid_entries
        
        result = reconciliation_service._validate_entries_for_matching(all_entries)
        
        assert isinstance(result, list)
        assert len(result) == 1  # Only valid entry should remain
        assert result[0] == valid_entry
    
    def test_calculate_match_score_perfect_match(self, reconciliation_service):
        """Test match score calculation for perfect match."""
        bank_transaction = Transaction(
            id=1,
            date=date(2024, 1, 15),
            amount=-100.0,
            description='PAGAMENTO FORNECEDOR A'
        )
        
        company_entry = CompanyFinancial(
            id=1,
            date=date(2024, 1, 15),
            amount=-100.0,
            description='PAGAMENTO FORNECEDOR A',
            transaction_type='expense'
        )
        
        score = reconciliation_service._calculate_match_score(bank_transaction, company_entry)
        
        assert isinstance(score, float)
        assert score >= 0.9  # Should be very high for perfect match
        assert score <= 1.0
    
    def test_calculate_match_score_partial_match(self, reconciliation_service):
        """Test match score calculation for partial match."""
        bank_transaction = Transaction(
            id=1,
            date=date(2024, 1, 15),
            amount=-100.0,
            description='PAGAMENTO FORNECEDOR A'
        )
        
        company_entry = CompanyFinancial(
            id=1,
            date=date(2024, 1, 16),  # Different date
            amount=-100.50,  # Slightly different amount
            description='PAGAMENTO FORNECEDOR B',  # Similar description
            transaction_type='expense'
        )
        
        score = reconciliation_service._calculate_match_score(bank_transaction, company_entry)
        
        assert isinstance(score, float)
        assert 0.0 <= score < 0.9  # Should be lower for partial match
    
    def test_calculate_match_score_no_match(self, reconciliation_service):
        """Test match score calculation for no match."""
        bank_transaction = Transaction(
            id=1,
            date=date(2024, 1, 15),
            amount=-100.0,
            description='BANK TRANSACTION'
        )
        
        company_entry = CompanyFinancial(
            id=1,
            date=date(2024, 2, 15),  # Very different date
            amount=500.0,  # Very different amount
            description='COMPANY ENTRY',  # Very different description
            transaction_type='income'
        )
        
        score = reconciliation_service._calculate_match_score(bank_transaction, company_entry)
        
        assert isinstance(score, float)
        assert score < 0.3  # Should be very low for no match
    
    def test_calculate_description_similarity_exact_match(self, reconciliation_service):
        """Test description similarity calculation for exact match."""
        similarity = reconciliation_service._calculate_description_similarity(
            'PAGAMENTO FORNECEDOR A',
            'PAGAMENTO FORNECEDOR A'
        )
        
        assert similarity == 0.8  # Exact match returns 0.8
    
    def test_calculate_description_similarity_contained(self, reconciliation_service):
        """Test description similarity calculation for contained strings."""
        similarity = reconciliation_service._calculate_description_similarity(
            'PAGAMENTO FORNECEDOR',
            'PAGAMENTO FORNECEDOR A LTDA'
        )
        
        assert similarity == 0.8  # One contained in other returns 0.8
    
    def test_calculate_description_similarity_common_words(self, reconciliation_service):
        """Test description similarity calculation for common words."""
        similarity = reconciliation_service._calculate_description_similarity(
            'PAGAMENTO FORNECEDOR A',
            'PAGAMENTO FORNECEDOR B'
        )
        
        assert 0.0 < similarity < 0.8  # Should have some similarity due to common words
    
    def test_calculate_description_similarity_no_match(self, reconciliation_service):
        """Test description similarity calculation for no match."""
        similarity = reconciliation_service._calculate_description_similarity(
            'COMPLETELY DIFFERENT',
            'TOTALLY UNRELATED'
        )
        
        assert similarity == 0.0  # No common words
    
    def test_validate_match_success(self, reconciliation_service):
        """Test match validation for valid match."""
        bank_transaction = Transaction(
            id=1,
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST'
        )
        
        company_entry = CompanyFinancial(
            id=1,
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST',
            transaction_type='expense'
        )
        
        match_data = {
            'bank_transaction': bank_transaction,
            'company_entry': company_entry,
            'match_score': 0.85
        }
        
        result = reconciliation_service._validate_match(match_data)
        assert result is True
    
    def test_validate_match_low_score(self, reconciliation_service):
        """Test match validation for low score match."""
        bank_transaction = Transaction(
            id=1,
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST'
        )
        
        company_entry = CompanyFinancial(
            id=1,
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST',
            transaction_type='expense'
        )
        
        match_data = {
            'bank_transaction': bank_transaction,
            'company_entry': company_entry,
            'match_score': 0.5  # Below threshold
        }
        
        result = reconciliation_service._validate_match(match_data)
        assert result is False
    
    def test_validate_match_large_date_difference(self, reconciliation_service):
        """Test match validation for large date difference."""
        bank_transaction = Transaction(
            id=1,
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST'
        )
        
        company_entry = CompanyFinancial(
            id=1,
            date=date(2024, 3, 15),  # 60 days difference
            amount=-100.0,
            description='TEST',
            transaction_type='expense'
        )
        
        match_data = {
            'bank_transaction': bank_transaction,
            'company_entry': company_entry,
            'match_score': 0.85
        }
        
        result = reconciliation_service._validate_match(match_data)
        assert result is False
    
    def test_validate_match_large_amount_difference(self, reconciliation_service):
        """Test match validation for large amount difference."""
        bank_transaction = Transaction(
            id=1,
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST'
        )
        
        company_entry = CompanyFinancial(
            id=1,
            date=date(2024, 1, 15),
            amount=-200.0,  # 100% difference
            description='TEST',
            transaction_type='expense'
        )
        
        match_data = {
            'bank_transaction': bank_transaction,
            'company_entry': company_entry,
            'match_score': 0.85
        }
        
        result = reconciliation_service._validate_match(match_data)
        assert result is False
    
    @patch('src.models.user.db.session')
    def test_create_reconciliation_records_success(self, mock_session, reconciliation_service):
        """Test successful creation of reconciliation records."""
        bank_transaction = Transaction(id=1, date=date(2024, 1, 15), amount=-100.0, description='TEST')
        company_entry = CompanyFinancial(id=1, date=date(2024, 1, 15), amount=-100.0, description='TEST')
        
        matches = [
            {
                'bank_transaction': bank_transaction,
                'company_entry': company_entry,
                'match_score': 0.85
            }
        ]
        
        # Mock database operations
        mock_session.add = Mock()
        mock_session.commit = Mock()
        
        with patch('src.models.transaction.ReconciliationRecord') as mock_record_class:
            mock_record = Mock()
            mock_record_class.return_value = mock_record
            
            with patch('src.models.transaction.ReconciliationRecord.query') as mock_query:
                mock_query.filter_by.return_value.first.return_value = None  # No existing record
                
                result = reconciliation_service.create_reconciliation_records(matches)
                
                assert isinstance(result, list)
                assert len(result) == 1
                mock_session.add.assert_called_once()
    
    def test_create_reconciliation_records_empty_matches(self, reconciliation_service):
        """Test creation of reconciliation records with empty matches raises InsufficientDataError."""
        with pytest.raises(InsufficientDataError):
            reconciliation_service.create_reconciliation_records([])
    
    def test_validate_match_for_record_creation_success(self, reconciliation_service):
        """Test match validation for record creation."""
        bank_transaction = Mock()
        bank_transaction.id = 1
        
        company_entry = Mock()
        company_entry.id = 1
        
        match = {
            'bank_transaction': bank_transaction,
            'company_entry': company_entry,
            'match_score': 0.85
        }
        
        result = reconciliation_service._validate_match_for_record_creation(match)
        assert result is True
    
    def test_validate_match_for_record_creation_missing_keys(self, reconciliation_service):
        """Test match validation for record creation with missing keys."""
        incomplete_match = {
            'bank_transaction': Mock(),
            # Missing company_entry and match_score
        }
        
        result = reconciliation_service._validate_match_for_record_creation(incomplete_match)
        assert result is False
    
    def test_validate_match_for_record_creation_invalid_score(self, reconciliation_service):
        """Test match validation for record creation with invalid score."""
        bank_transaction = Mock()
        bank_transaction.id = 1
        
        company_entry = Mock()
        company_entry.id = 1
        
        match = {
            'bank_transaction': bank_transaction,
            'company_entry': company_entry,
            'match_score': 1.5  # Invalid score > 1
        }
        
        result = reconciliation_service._validate_match_for_record_creation(match)
        assert result is False
    
    @patch('src.models.transaction.ReconciliationRecord.query')
    def test_get_pending_reconciliations(self, mock_query, reconciliation_service):
        """Test getting pending reconciliation records."""
        mock_records = [Mock(), Mock(), Mock()]
        mock_query.filter_by.return_value.all.return_value = mock_records
        
        result = reconciliation_service.get_pending_reconciliations()
        
        assert result == mock_records
        mock_query.filter_by.assert_called_once_with(status='pending')
    
    @patch('src.models.transaction.ReconciliationRecord.query')
    @patch('src.models.user.db.session')
    def test_confirm_reconciliation_success(self, mock_session, mock_query, reconciliation_service):
        """Test successful reconciliation confirmation."""
        mock_record = Mock()
        mock_record.status = 'pending'
        mock_record.match_score = 0.85
        mock_query.get.return_value = mock_record
        
        result = reconciliation_service.confirm_reconciliation(1)
        
        assert result is True
        assert mock_record.status == 'confirmed'
        mock_query.get.assert_called_once_with(1)
    
    def test_confirm_reconciliation_invalid_id(self, reconciliation_service):
        """Test reconciliation confirmation with invalid ID raises ValidationError."""
        with pytest.raises(ValidationError):
            reconciliation_service.confirm_reconciliation(0)
        
        with pytest.raises(ValidationError):
            reconciliation_service.confirm_reconciliation(-1)
    
    @patch('src.models.transaction.ReconciliationRecord.query')
    def test_confirm_reconciliation_not_found(self, mock_query, reconciliation_service):
        """Test reconciliation confirmation with non-existent record raises RecordNotFoundError."""
        mock_query.get.return_value = None
        
        with pytest.raises(RecordNotFoundError):
            reconciliation_service.confirm_reconciliation(999)
    
    @patch('src.models.transaction.ReconciliationRecord.query')
    def test_confirm_reconciliation_already_confirmed(self, mock_query, reconciliation_service):
        """Test reconciliation confirmation for already confirmed record."""
        mock_record = Mock()
        mock_record.status = 'confirmed'
        mock_query.get.return_value = mock_record
        
        result = reconciliation_service.confirm_reconciliation(1)
        
        assert result is True  # Should return True without error
    
    @patch('src.models.transaction.ReconciliationRecord.query')
    def test_confirm_reconciliation_rejected_record(self, mock_query, reconciliation_service):
        """Test reconciliation confirmation for rejected record raises ReconciliationError."""
        mock_record = Mock()
        mock_record.status = 'rejected'
        mock_query.get.return_value = mock_record
        
        with pytest.raises(ReconciliationError):
            reconciliation_service.confirm_reconciliation(1)
    
    @patch('src.models.transaction.ReconciliationRecord.query')
    @patch('src.models.user.db.session')
    def test_reject_reconciliation_success(self, mock_session, mock_query, reconciliation_service):
        """Test successful reconciliation rejection."""
        mock_record = Mock()
        mock_record.status = 'pending'
        mock_record.match_score = 0.85
        mock_query.get.return_value = mock_record
        
        result = reconciliation_service.reject_reconciliation(1)
        
        assert result is True
        assert mock_record.status == 'rejected'
        mock_query.get.assert_called_once_with(1)
    
    def test_reject_reconciliation_invalid_id(self, reconciliation_service):
        """Test reconciliation rejection with invalid ID raises ValidationError."""
        with pytest.raises(ValidationError):
            reconciliation_service.reject_reconciliation(0)
    
    @patch('src.models.transaction.ReconciliationRecord.query')
    def test_reject_reconciliation_confirmed_record(self, mock_query, reconciliation_service):
        """Test reconciliation rejection for confirmed record raises ReconciliationError."""
        mock_record = Mock()
        mock_record.status = 'confirmed'
        mock_query.get.return_value = mock_record
        
        with pytest.raises(ReconciliationError):
            reconciliation_service.reject_reconciliation(1)
    
    @patch('src.models.transaction.ReconciliationRecord.query')
    def test_get_reconciliation_report(self, mock_query, reconciliation_service):
        """Test reconciliation report generation."""
        # Mock query results
        mock_query.count.return_value = 100
        mock_query.filter_by.return_value.count.side_effect = [60, 30, 10]  # confirmed, pending, rejected
        
        with patch.object(reconciliation_service, '_calculate_financial_metrics') as mock_financial:
            mock_financial.return_value = {
                'total_reconciled_value': 10000.0,
                'reconciled_credits': 8000.0,
                'reconciled_debits': 2000.0,
                'net_reconciled_flow': 6000.0
            }
            
            with patch.object(reconciliation_service, '_calculate_average_match_score') as mock_avg:
                mock_avg.return_value = 0.85
                
                with patch.object(reconciliation_service, '_get_recent_reconciliation_activity') as mock_activity:
                    mock_activity.return_value = {
                        'confirmed_last_7_days': 10,
                        'rejected_last_7_days': 2,
                        'total_activity_last_7_days': 12
                    }
                    
                    result = reconciliation_service.get_reconciliation_report()
                    
                    assert isinstance(result, dict)
                    assert 'summary' in result
                    assert 'financials' in result
                    assert 'recent_activity' in result
                    assert 'generated_at' in result
                    
                    summary = result['summary']
                    assert summary['total_records'] == 100
                    assert summary['confirmed'] == 60
                    assert summary['pending'] == 30
                    assert summary['rejected'] == 10
                    assert summary['reconciliation_rate'] == 0.6
                    assert summary['average_match_score'] == 0.85
    
    def test_calculate_financial_metrics(self, reconciliation_service):
        """Test financial metrics calculation."""
        # Mock confirmed reconciliation records
        mock_records = []
        
        # Create mock records with bank transactions
        for i, amount in enumerate([100.0, -50.0, 200.0, -75.0]):
            mock_record = Mock()
            mock_record.id = i + 1
            mock_record.bank_transaction = Mock()
            mock_record.bank_transaction.amount = amount
            mock_records.append(mock_record)
        
        with patch('src.models.transaction.ReconciliationRecord.query') as mock_query:
            mock_query.filter_by.return_value.all.return_value = mock_records
            
            result = reconciliation_service._calculate_financial_metrics()
            
            assert isinstance(result, dict)
            assert result['total_reconciled_value'] == 425.0  # Sum of absolute values
            assert result['reconciled_credits'] == 300.0  # 100 + 200
            assert result['reconciled_debits'] == 125.0  # 50 + 75
            assert result['net_reconciled_flow'] == 175.0  # 300 - 125
    
    def test_calculate_average_match_score(self, reconciliation_service):
        """Test average match score calculation."""
        mock_records = []
        scores = [0.85, 0.90, 0.75, 0.95]
        
        for i, score in enumerate(scores):
            mock_record = Mock()
            mock_record.match_score = score
            mock_records.append(mock_record)
        
        with patch('src.models.transaction.ReconciliationRecord.query') as mock_query:
            mock_query.filter_by.return_value.all.return_value = mock_records
            
            result = reconciliation_service._calculate_average_match_score()
            
            expected_avg = sum(scores) / len(scores)
            assert abs(result - expected_avg) < 0.001
    
    def test_calculate_average_match_score_no_records(self, reconciliation_service):
        """Test average match score calculation with no records."""
        with patch('src.models.transaction.ReconciliationRecord.query') as mock_query:
            mock_query.filter_by.return_value.all.return_value = []
            
            result = reconciliation_service._calculate_average_match_score()
            
            assert result is None
    
    def test_get_recent_reconciliation_activity(self, reconciliation_service):
        """Test recent reconciliation activity calculation."""
        with patch('src.models.transaction.ReconciliationRecord.query') as mock_query:
            # Mock filter chain for confirmed records
            mock_confirmed_query = Mock()
            mock_confirmed_query.count.return_value = 5
            
            # Mock filter chain for rejected records
            mock_rejected_query = Mock()
            mock_rejected_query.count.return_value = 2
            
            mock_query.filter.return_value = mock_confirmed_query
            mock_query.filter.return_value = mock_rejected_query
            
            # Need to handle the chained filter calls
            def filter_side_effect(*args):
                if 'confirmed' in str(args):
                    return mock_confirmed_query
                elif 'rejected' in str(args):
                    return mock_rejected_query
                return Mock()
            
            mock_query.filter.side_effect = filter_side_effect
            
            result = reconciliation_service._get_recent_reconciliation_activity()
            
            assert isinstance(result, dict)
            assert 'confirmed_last_7_days' in result
            assert 'rejected_last_7_days' in result
            assert 'total_activity_last_7_days' in result


class TestReconciliationServiceIntegration:
    """Integration tests for ReconciliationService."""
    
    def test_full_reconciliation_workflow(self, reconciliation_service, db_session, assert_financial_accuracy):
        """Test complete reconciliation workflow."""
        # Create test data in database
        bank_transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='PAGAMENTO FORNECEDOR A'
        )
        db_session.add(bank_transaction)
        
        company_entry = CompanyFinancial(
            date=date(2024, 1, 15),
            amount=-100.0,
            description='PAGAMENTO FORNECEDOR A',
            transaction_type='expense'
        )
        db_session.add(company_entry)
        db_session.commit()
        
        # Step 1: Find matches
        matches = reconciliation_service.find_matches([bank_transaction], [company_entry])
        assert len(matches) == 1
        
        # Step 2: Create reconciliation records
        records = reconciliation_service.create_reconciliation_records(matches)
        assert len(records) == 1
        
        # Step 3: Confirm reconciliation
        record_id = records[0].id
        success = reconciliation_service.confirm_reconciliation(record_id)
        assert success is True
        
        # Step 4: Generate report
        report = reconciliation_service.get_reconciliation_report()
        assert report['summary']['confirmed'] >= 1
        
        # Verify financial accuracy
        assert_financial_accuracy(100.0, abs(matches[0]['bank_transaction'].amount))
        assert_financial_accuracy(100.0, abs(matches[0]['company_entry'].amount))


class TestReconciliationServiceErrorHandling:
    """Test error handling in ReconciliationService."""
    
    def test_timeout_handling(self, reconciliation_service):
        """Test timeout handling in reconciliation operations."""
        with patch('src.services.reconciliation_service.with_timeout') as mock_timeout:
            mock_timeout.side_effect = TimeoutError("Reconciliation timeout")
            
            with pytest.raises(TimeoutError):
                reconciliation_service.find_matches([Mock()], [Mock()])
    
    def test_service_error_handling(self, reconciliation_service):
        """Test service error handling decorator."""
        with patch('src.services.reconciliation_service.handle_service_errors') as mock_handler:
            # Test that the decorator is applied
            assert hasattr(reconciliation_service.find_matches, '__wrapped__')
    
    def test_database_transaction_handling(self, reconciliation_service):
        """Test database transaction handling decorator."""
        with patch('src.services.reconciliation_service.with_database_transaction') as mock_db_tx:
            # Test that the decorator is applied
            assert hasattr(reconciliation_service.create_reconciliation_records, '__wrapped__')
    
    def test_database_error_handling(self, reconciliation_service):
        """Test database error handling."""
        matches = [
            {
                'bank_transaction': Mock(id=1),
                'company_entry': Mock(id=1),
                'match_score': 0.85
            }
        ]
        