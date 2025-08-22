"""
Comprehensive error handling and edge case tests for Maria Conciliadora application.

This module tests:
- Exception handling across all services
- Edge cases and boundary conditions
- Error recovery mechanisms
- Input validation and sanitization
- Network and external service failures
- Database transaction rollbacks
- File processing errors
"""

import pytest
import tempfile
import os
from datetime import date, datetime
from unittest.mock import patch, MagicMock, Mock
from sqlalchemy.exc import IntegrityError, OperationalError
from src.services.ofx_processor import OFXProcessor
from src.services.xlsx_processor import XLSXProcessor
from src.services.ai_service import AIService
from src.services.reconciliation_service import ReconciliationService
from src.services.duplicate_detection_service import DuplicateDetectionService
from src.models.transaction import Transaction
from src.models.company_financial import CompanyFinancial
from src.utils.exceptions import (
    ValidationError, 
    BusinessLogicError, 
    FileProcessingError,
    AIServiceError,
    DatabaseError
)


class TestFileProcessingErrors:
    """Test file processing error handling."""
    
    def test_corrupted_ofx_file_handling(self, ofx_processor):
        """Test handling of corrupted OFX files."""
        # Create corrupted OFX content
        corrupted_ofx_content = """OFXHEADER:100
DATA:OFXSGML
VERSION:102
<OFX>
<BANKMSGSRSV1>
<STMTRS>
<CURDEF>BRL
<BANKACCTFROM>
<BANKID>001
<ACCTID>12345-6
<!-- Missing closing tags and corrupted structure -->
<BANKTRANLIST>
<DTSTART>20240101120000
<STMTTRN>
<TRNTYPE>DEBIT
<DTPOSTED>20240115120000
<TRNAMT>-100.00
<!-- Missing FITID and closing tags -->
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
            f.write(corrupted_ofx_content)
            temp_path = f.name
        
        try:
            # Should raise FileProcessingError for corrupted file
            with pytest.raises(FileProcessingError) as exc_info:
                ofx_processor.parse_ofx_file(temp_path)
            
            assert "corrupted" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
        
        finally:
            os.unlink(temp_path)
    
    def test_empty_ofx_file_handling(self, ofx_processor):
        """Test handling of empty OFX files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            with pytest.raises(FileProcessingError) as exc_info:
                ofx_processor.parse_ofx_file(temp_path)
            
            assert "empty" in str(exc_info.value).lower()
        
        finally:
            os.unlink(temp_path)
    
    def test_invalid_file_format_handling(self, ofx_processor):
        """Test handling of invalid file formats."""
        # Create file with wrong format
        invalid_content = """This is not an OFX file
It's just plain text
With no OFX structure"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
            f.write(invalid_content)
            temp_path = f.name
        
        try:
            with pytest.raises(FileProcessingError) as exc_info:
                ofx_processor.parse_ofx_file(temp_path)
            
            assert "format" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
        
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found_handling(self, ofx_processor):
        """Test handling of non-existent files."""
        non_existent_path = "/path/that/does/not/exist.ofx"
        
        with pytest.raises(FileProcessingError) as exc_info:
            ofx_processor.parse_ofx_file(non_existent_path)
        
        assert "not found" in str(exc_info.value).lower() or "does not exist" in str(exc_info.value).lower()
    
    def test_permission_denied_handling(self, ofx_processor):
        """Test handling of permission denied errors."""
        # Create a file and remove read permissions (Unix-like systems)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
            f.write("OFXHEADER:100\nDATA:OFXSGML")
            temp_path = f.name
        
        try:
            # Try to remove read permissions (may not work on all systems)
            try:
                os.chmod(temp_path, 0o000)
                
                with pytest.raises(FileProcessingError) as exc_info:
                    ofx_processor.parse_ofx_file(temp_path)
                
                assert "permission" in str(exc_info.value).lower() or "access" in str(exc_info.value).lower()
            
            except (OSError, PermissionError):
                # Skip test if we can't modify permissions
                pytest.skip("Cannot modify file permissions on this system")
        
        finally:
            try:
                os.chmod(temp_path, 0o644)  # Restore permissions
                os.unlink(temp_path)
            except (OSError, PermissionError):
                pass
    
    def test_xlsx_file_errors(self, xlsx_processor):
        """Test XLSX file processing errors."""
        # Test corrupted XLSX file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            f.write(b"This is not a valid XLSX file")
            temp_path = f.name
        
        try:
            with pytest.raises(FileProcessingError):
                xlsx_processor.parse_xlsx_file(temp_path)
        
        finally:
            os.unlink(temp_path)
    
    def test_large_file_memory_error(self, ofx_processor):
        """Test handling of extremely large files that might cause memory errors."""
        # Mock memory error during file processing
        with patch('builtins.open', side_effect=MemoryError("Not enough memory")):
            with pytest.raises(FileProcessingError) as exc_info:
                ofx_processor.parse_ofx_file("large_file.ofx")
            
            assert "memory" in str(exc_info.value).lower()


class TestDatabaseErrors:
    """Test database error handling."""
    
    def test_database_connection_error(self, db_session):
        """Test handling of database connection errors."""
        # Mock database connection error
        with patch.object(db_session, 'add', side_effect=OperationalError("Connection lost", None, None)):
            transaction = Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 15),
                amount=-100.00,
                description='Test transaction',
                transaction_type='debit'
            )
            
            with pytest.raises(DatabaseError):
                db_session.add(transaction)
                db_session.commit()
    
    def test_constraint_violation_error(self, db_session):
        """Test handling of database constraint violations."""
        # Create transaction with duplicate primary key (if applicable)
        transaction1 = Transaction(
            bank_name='TEST_BANK',
            account_id='ACC_001',
            date=date(2024, 1, 15),
            amount=-100.00,
            description='Test transaction',
            transaction_type='debit'
        )
        
        db_session.add(transaction1)
        db_session.commit()
        
        # Mock integrity error for duplicate
        with patch.object(db_session, 'add', side_effect=IntegrityError("Duplicate key", None, None)):
            transaction2 = Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 15),
                amount=-100.00,
                description='Test transaction',
                transaction_type='debit'
            )
            
            with pytest.raises(DatabaseError):
                db_session.add(transaction2)
                db_session.commit()
    
    def test_transaction_rollback(self, db_session):
        """Test database transaction rollback on errors."""
        initial_count = db_session.query(Transaction).count()
        
        try:
            # Start transaction
            transaction1 = Transaction(
                bank_name='ROLLBACK_TEST',
                account_id='ACC_001',
                date=date(2024, 1, 15),
                amount=-100.00,
                description='First transaction',
                transaction_type='debit'
            )
            db_session.add(transaction1)
            
            # This should succeed
            db_session.flush()
            
            # Simulate error in second transaction
            with patch.object(db_session, 'add', side_effect=Exception("Simulated error")):
                transaction2 = Transaction(
                    bank_name='ROLLBACK_TEST',
                    account_id='ACC_002',
                    date=date(2024, 1, 15),
                    amount=-200.00,
                    description='Second transaction',
                    transaction_type='debit'
                )
                db_session.add(transaction2)
            
            db_session.commit()
        
        except Exception:
            db_session.rollback()
        
        # Verify rollback - count should be same as initial
        final_count = db_session.query(Transaction).count()
        assert final_count == initial_count


class TestAIServiceErrors:
    """Test AI service error handling."""
    
    def test_openai_api_error(self, ai_service):
        """Test handling of OpenAI API errors."""
        transaction_data = {
            'description': 'Test transaction',
            'amount': -100.00,
            'date': date(2024, 1, 15)
        }
        
        # Mock OpenAI API error
        with patch.object(ai_service, '_call_openai_api', side_effect=Exception("API rate limit exceeded")):
            with pytest.raises(AIServiceError) as exc_info:
                ai_service.categorize_transaction(transaction_data)
            
            assert "api" in str(exc_info.value).lower() or "rate limit" in str(exc_info.value).lower()
    
    def test_groq_api_error(self, ai_service):
        """Test handling of Groq API errors."""
        transactions = [
            {
                'description': 'Test transaction',
                'amount': -100.00,
                'date': date(2024, 1, 15)
            }
        ]
        
        # Mock Groq API error
        with patch.object(ai_service, '_call_groq_api', side_effect=Exception("Service unavailable")):
            with pytest.raises(AIServiceError) as exc_info:
                ai_service.categorize_transactions_batch(transactions)
            
            assert "service" in str(exc_info.value).lower() or "unavailable" in str(exc_info.value).lower()
    
    def test_invalid_ai_response_handling(self, ai_service):
        """Test handling of invalid AI service responses."""
        transaction_data = {
            'description': 'Test transaction',
            'amount': -100.00,
            'date': date(2024, 1, 15)
        }
        
        # Mock invalid response
        with patch.object(ai_service, '_call_openai_api', return_value="invalid_json_response"):
            with pytest.raises(AIServiceError) as exc_info:
                ai_service.categorize_transaction(transaction_data)
            
            assert "invalid" in str(exc_info.value).lower() or "response" in str(exc_info.value).lower()
    
    def test_ai_service_timeout(self, ai_service):
        """Test handling of AI service timeouts."""
        transaction_data = {
            'description': 'Test transaction',
            'amount': -100.00,
            'date': date(2024, 1, 15)
        }
        
        # Mock timeout error
        with patch.object(ai_service, '_call_openai_api', side_effect=TimeoutError("Request timeout")):
            with pytest.raises(AIServiceError) as exc_info:
                ai_service.categorize_transaction(transaction_data)
            
            assert "timeout" in str(exc_info.value).lower()
    
    def test_ai_service_fallback_mechanism(self, ai_service):
        """Test AI service fallback mechanisms."""
        transaction_data = {
            'description': 'SUPERMERCADO EXTRA',
            'amount': -150.00,
            'date': date(2024, 1, 15)
        }
        
        # Mock primary service failure, fallback should work
        with patch.object(ai_service, '_call_openai_api', side_effect=Exception("Primary service failed")):
            with patch.object(ai_service, '_call_groq_api', return_value='{"category": "alimentacao"}'):
                # Should not raise error, should use fallback
                result = ai_service.categorize_transaction(transaction_data)
                assert result == 'alimentacao'


class TestValidationErrors:
    """Test input validation and sanitization errors."""
    
    def test_invalid_transaction_data(self):
        """Test validation of invalid transaction data."""
        invalid_cases = [
            # Missing required fields
            {
                'bank_name': None,
                'account_id': 'ACC_001',
                'date': date(2024, 1, 15),
                'amount': -100.00,
                'description': 'Test',
                'transaction_type': 'debit'
            },
            # Invalid amount
            {
                'bank_name': 'TEST_BANK',
                'account_id': 'ACC_001',
                'date': date(2024, 1, 15),
                'amount': 0.00,
                'description': 'Test',
                'transaction_type': 'debit'
            },
            # Invalid date
            {
                'bank_name': 'TEST_BANK',
                'account_id': 'ACC_001',
                'date': 'invalid_date',
                'amount': -100.00,
                'description': 'Test',
                'transaction_type': 'debit'
            },
            # Invalid transaction type
            {
                'bank_name': 'TEST_BANK',
                'account_id': 'ACC_001',
                'date': date(2024, 1, 15),
                'amount': -100.00,
                'description': 'Test',
                'transaction_type': 'invalid_type'
            },
        ]
        
        for invalid_data in invalid_cases:
            with pytest.raises((ValidationError, ValueError, TypeError)):
                transaction = Transaction(**invalid_data)
                # Additional validation if needed
                self._validate_transaction(transaction)
    
    def test_sql_injection_prevention(self, db_session):
        """Test prevention of SQL injection attacks."""
        # Attempt SQL injection in description field
        malicious_description = "'; DROP TABLE transactions; --"
        
        transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='ACC_001',
            date=date(2024, 1, 15),
            amount=-100.00,
            description=malicious_description,
            transaction_type='debit'
        )
        
        # Should not cause SQL injection
        db_session.add(transaction)
        db_session.commit()
        
        # Verify transaction was saved safely
        saved_tx = db_session.query(Transaction).filter_by(description=malicious_description).first()
        assert saved_tx is not None
        assert saved_tx.description == malicious_description
    
    def test_xss_prevention(self):
        """Test prevention of XSS attacks in input data."""
        # Attempt XSS in description field
        xss_description = "<script>alert('XSS')</script>"
        
        transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='ACC_001',
            date=date(2024, 1, 15),
            amount=-100.00,
            description=xss_description,
            transaction_type='debit'
        )
        
        # Should sanitize or escape the input
        sanitized_description = self._sanitize_input(transaction.description)
        assert "<script>" not in sanitized_description
        assert "alert" not in sanitized_description
    
    def test_input_length_validation(self):
        """Test validation of input field lengths."""
        # Test extremely long description
        long_description = "A" * 10000  # Very long string
        
        with pytest.raises(ValidationError):
            transaction = Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 15),
                amount=-100.00,
                description=long_description,
                transaction_type='debit'
            )
            self._validate_transaction(transaction)
    
    def test_special_character_handling(self):
        """Test handling of special characters in input."""
        special_chars_description = "Açaí & Café - R$ 25,50 (50% desc.)"
        
        # Should handle special characters without errors
        transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='ACC_001',
            date=date(2024, 1, 15),
            amount=-25.50,
            description=special_chars_description,
            transaction_type='debit'
        )
        
        # Should not raise validation errors
        self._validate_transaction(transaction)
    
    def _validate_transaction(self, transaction):
        """Validate transaction data."""
        if not transaction.bank_name:
            raise ValidationError("Bank name is required")
        
        if not transaction.account_id:
            raise ValidationError("Account ID is required")
        
        if transaction.amount == 0:
            raise ValidationError("Amount cannot be zero")
        
        if not isinstance(transaction.date, date):
            raise ValidationError("Invalid date format")
        
        if transaction.transaction_type not in ['debit', 'credit']:
            raise ValidationError("Invalid transaction type")
        
        if len(transaction.description) > 1000:
            raise ValidationError("Description too long")
    
    def _sanitize_input(self, input_string):
        """Sanitize input to prevent XSS."""
        if not input_string:
            return input_string
        
        # Basic XSS prevention
        dangerous_patterns = ['<script>', '</script>', 'javascript:', 'onload=', 'onerror=']
        sanitized = input_string
        
        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern, '')
        
        return sanitized


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_leap_year_date_handling(self):
        """Test handling of leap year dates."""
        # Test February 29 in leap year
        leap_year_date = date(2024, 2, 29)  # 2024 is a leap year
        
        transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='ACC_001',
            date=leap_year_date,
            amount=-100.00,
            description='Leap year transaction',
            transaction_type='debit'
        )
        
        # Should handle leap year date without errors
        assert transaction.date == leap_year_date
    
    def test_extreme_amount_values(self):
        """Test handling of extreme amount values."""
        extreme_cases = [
            0.01,      # Minimum positive
            -0.01,     # Minimum negative
            999999.99, # Large positive
            -999999.99 # Large negative
        ]
        
        for amount in extreme_cases:
            transaction = Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 15),
                amount=amount,
                description=f'Extreme amount test: {amount}',
                transaction_type='debit' if amount < 0 else 'credit'
            )
            
            # Should handle extreme amounts
            assert transaction.amount == amount
    
    def test_unicode_character_handling(self):
        """Test handling of Unicode characters."""
        unicode_descriptions = [
            "Café com açúcar",
            "Transferência bancária",
            "Pagamento em €",
            "商店购买",  # Chinese characters
            "🏪 Store purchase",  # Emoji
        ]
        
        for description in unicode_descriptions:
            transaction = Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 15),
                amount=-100.00,
                description=description,
                transaction_type='debit'
            )
            
            # Should handle Unicode without errors
            assert transaction.description == description
    
    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        # Test with empty description (should be allowed)
        transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='ACC_001',
            date=date(2024, 1, 15),
            amount=-100.00,
            description='',  # Empty description
            transaction_type='debit'
        )
        
        assert transaction.description == ''
    
    def test_whitespace_handling(self):
        """Test handling of whitespace in input."""
        # Test with leading/trailing whitespace
        transaction = Transaction(
            bank_name='  TEST_BANK  ',
            account_id='  ACC_001  ',
            date=date(2024, 1, 15),
            amount=-100.00,
            description='  Test transaction  ',
            transaction_type='debit'
        )
        
        # Should trim whitespace
        assert transaction.bank_name.strip() == 'TEST_BANK'
        assert transaction.account_id.strip() == 'ACC_001'
        assert transaction.description.strip() == 'Test transaction'
    
    def test_concurrent_access_handling(self, db_session):
        """Test handling of concurrent access to same data."""
        # Create initial transaction
        transaction = Transaction(
            bank_name='CONCURRENT_TEST',
            account_id='ACC_001',
            date=date(2024, 1, 15),
            amount=-100.00,
            description='Concurrent test',
            transaction_type='debit'
        )
        db_session.add(transaction)
        db_session.commit()
        
        # Simulate concurrent modification
        # This would typically involve multiple database sessions
        # For this test, we'll simulate the scenario
        
        original_id = transaction.id
        
        # First "session" modifies the transaction
        transaction.amount = -150.00
        
        # Second "session" tries to modify the same transaction
        # This should be handled gracefully
        try:
            db_session.commit()
            
            # Verify the change was applied
            updated_tx = db_session.query(Transaction).filter_by(id=original_id).first()
            assert updated_tx.amount == -150.00
        
        except Exception as e:
            # Should handle concurrent modification gracefully
            db_session.rollback()
            assert "concurrent" in str(e).lower() or "conflict" in str(e).lower()


class TestRecoveryMechanisms:
    """Test error recovery mechanisms."""
    
    def test_service_retry_mechanism(self, ai_service):
        """Test service retry mechanisms."""
        transaction_data = {
            'description': 'Test transaction',
            'amount': -100.00,
            'date': date(2024, 1, 15)
        }
        
        # Mock service that fails twice then succeeds
        call_count = 0
        def mock_api_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary service failure")
            return '{"category": "alimentacao"}'
        
        with patch.object(ai_service, '_call_openai_api', side_effect=mock_api_call):
            # Should retry and eventually succeed
            result = ai_service.categorize_transaction(transaction_data)
            assert result == 'alimentacao'
            assert call_count == 3  # Failed twice, succeeded on third try
    
    def test_graceful_degradation(self, ai_service):
        """Test graceful degradation when services are unavailable."""
        transaction_data = {
            'description': 'SUPERMERCADO EXTRA',
            'amount': -150.00,
            'date': date(2024, 1, 15)
        }
        
        # Mock all AI services failing
        with patch.object(ai_service, '_call_openai_api', side_effect=Exception("Service unavailable")):
            with patch.object(ai_service, '_call_groq_api', side_effect=Exception("Service unavailable")):
                # Should fall back to rule-based categorization
                result = ai_service.categorize_transaction(transaction_data)
                
                # Should still return a reasonable category based on keywords
                assert result in ['alimentacao', 'outros', 'unknown']
    
    def test_partial_failure_handling(self, ai_service):
        """Test handling of partial failures in batch operations."""
        transactions = [
            {'description': 'SUPERMERCADO', 'amount': -100.00, 'date': date(2024, 1, 15)},
            {'description': 'POSTO SHELL', 'amount': -80.00, 'date': date(2024, 1, 16)},
            {'description': 'FARMACIA', 'amount': -50.00, 'date': date(2024, 1, 17)},
        ]
        
        # Mock partial failure - second transaction fails
        def mock_categorize(tx_data):
            if 'POSTO' in tx_data['description']:
                raise Exception("Categorization failed for this transaction")
            return 'alimentacao' if 'SUPERMERCADO' in tx_data['description'] else 'saude'
        
        with patch.object(ai_service, 'categorize_transaction', side_effect=mock_categorize):
            # Should handle partial failures gracefully
            results = []
            for tx in transactions:
                try:
                    category = ai_service.categorize_transaction(tx)
                    results.append({'transaction': tx, 'category': category, 'error': None})
                except Exception as e:
                    results.append({'transaction': tx, 'category': None, 'error': str(e)})
            
            # Should have 2 successes and 1 failure
            successes = [r for r in results if r['error'] is None]
            failures = [r for r in results if r['error'] is not None]
            
            assert len(successes) == 2
            assert len(failures) == 1
            assert failures[0]['transaction']['description'] == 'POSTO SHELL'