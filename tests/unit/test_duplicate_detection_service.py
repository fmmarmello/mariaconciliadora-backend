"""
Unit tests for DuplicateDetectionService.

Tests cover:
- File hash calculation
- File duplicate detection
- Transaction duplicate detection
- Financial entry duplicate detection
- Error handling and edge cases
- Performance with large datasets
"""

import pytest
import tempfile
import os
import hashlib
from datetime import date, datetime
from unittest.mock import Mock, patch, MagicMock
from src.services.duplicate_detection_service import DuplicateDetectionService
from src.models.transaction import Transaction, UploadHistory
from src.models.company_financial import CompanyFinancial


class TestDuplicateDetectionService:
    """Test suite for DuplicateDetectionService class."""
    
    def test_calculate_file_hash_success(self):
        """Test successful file hash calculation."""
        # Create a temporary file with known content
        test_content = b"This is test content for hash calculation"
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_path = temp_file.name
        
        try:
            result_hash = DuplicateDetectionService.calculate_file_hash(temp_path)
            
            # Calculate expected hash
            expected_hash = hashlib.sha256(test_content).hexdigest()
            
            assert isinstance(result_hash, str)
            assert len(result_hash) == 64  # SHA-256 produces 64-character hex string
            assert result_hash == expected_hash
        
        finally:
            os.unlink(temp_path)
    
    def test_calculate_file_hash_large_file(self):
        """Test file hash calculation with large file."""
        # Create a large file (simulate with repeated content)
        test_content = b"Large file content " * 10000  # ~200KB
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_path = temp_file.name
        
        try:
            result_hash = DuplicateDetectionService.calculate_file_hash(temp_path)
            
            # Calculate expected hash
            expected_hash = hashlib.sha256(test_content).hexdigest()
            
            assert isinstance(result_hash, str)
            assert len(result_hash) == 64
            assert result_hash == expected_hash
        
        finally:
            os.unlink(temp_path)
    
    def test_calculate_file_hash_empty_file(self):
        """Test file hash calculation with empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name  # Empty file
        
        try:
            result_hash = DuplicateDetectionService.calculate_file_hash(temp_path)
            
            # Hash of empty content
            expected_hash = hashlib.sha256(b"").hexdigest()
            
            assert isinstance(result_hash, str)
            assert len(result_hash) == 64
            assert result_hash == expected_hash
        
        finally:
            os.unlink(temp_path)
    
    def test_calculate_file_hash_nonexistent_file(self):
        """Test file hash calculation with non-existent file raises exception."""
        with pytest.raises(Exception) as exc_info:
            DuplicateDetectionService.calculate_file_hash('/non/existent/file.txt')
        
        assert "Error calculating file hash" in str(exc_info.value)
    
    def test_calculate_file_hash_permission_error(self):
        """Test file hash calculation with permission error."""
        # Create a file and then make it unreadable (Unix-like systems)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_path = temp_file.name
        
        try:
            # Try to make file unreadable (may not work on all systems)
            try:
                os.chmod(temp_path, 0o000)
                
                with pytest.raises(Exception) as exc_info:
                    DuplicateDetectionService.calculate_file_hash(temp_path)
                
                assert "Error calculating file hash" in str(exc_info.value)
            
            except (OSError, PermissionError):
                # If we can't change permissions, skip this test
                pytest.skip("Cannot change file permissions on this system")
        
        finally:
            # Restore permissions and cleanup
            try:
                os.chmod(temp_path, 0o644)
                os.unlink(temp_path)
            except (OSError, PermissionError):
                pass
    
    @patch('src.models.transaction.UploadHistory.query')
    def test_check_file_duplicate_found(self, mock_query):
        """Test file duplicate check when duplicate is found."""
        # Mock existing upload history record
        mock_record = Mock()
        mock_record.file_hash = 'abc123def456'
        mock_record.filename = 'existing_file.ofx'
        mock_record.upload_date = datetime(2024, 1, 1, 12, 0, 0)
        
        mock_query.filter_by.return_value.first.return_value = mock_record
        
        result = DuplicateDetectionService.check_file_duplicate('abc123def456')
        
        assert result == mock_record
        mock_query.filter_by.assert_called_once_with(file_hash='abc123def456')
    
    @patch('src.models.transaction.UploadHistory.query')
    def test_check_file_duplicate_not_found(self, mock_query):
        """Test file duplicate check when no duplicate is found."""
        mock_query.filter_by.return_value.first.return_value = None
        
        result = DuplicateDetectionService.check_file_duplicate('unique_hash_123')
        
        assert result is None
        mock_query.filter_by.assert_called_once_with(file_hash='unique_hash_123')
    
    @patch('src.models.transaction.UploadHistory.query')
    def test_check_file_duplicate_database_error(self, mock_query):
        """Test file duplicate check with database error."""
        mock_query.filter_by.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception) as exc_info:
            DuplicateDetectionService.check_file_duplicate('test_hash')
        
        assert "Error checking file duplicate" in str(exc_info.value)
    
    @patch('src.models.transaction.Transaction.query')
    def test_check_transaction_duplicate_found(self, mock_query):
        """Test transaction duplicate check when duplicate is found."""
        # Mock existing transaction
        mock_transaction = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_transaction
        
        result = DuplicateDetectionService.check_transaction_duplicate(
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST TRANSACTION'
        )
        
        assert result is True
        mock_query.filter_by.assert_called_once_with(
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST TRANSACTION'
        )
    
    @patch('src.models.transaction.Transaction.query')
    def test_check_transaction_duplicate_not_found(self, mock_query):
        """Test transaction duplicate check when no duplicate is found."""
        mock_query.filter_by.return_value.first.return_value = None
        
        result = DuplicateDetectionService.check_transaction_duplicate(
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='UNIQUE TRANSACTION'
        )
        
        assert result is False
        mock_query.filter_by.assert_called_once_with(
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='UNIQUE TRANSACTION'
        )
    
    @patch('src.models.transaction.Transaction.query')
    def test_check_transaction_duplicate_database_error(self, mock_query):
        """Test transaction duplicate check with database error."""
        mock_query.filter_by.side_effect = Exception("Database query failed")
        
        with pytest.raises(Exception) as exc_info:
            DuplicateDetectionService.check_transaction_duplicate(
                account_id='12345',
                date=date(2024, 1, 15),
                amount=-100.0,
                description='TEST'
            )
        
        assert "Error checking transaction duplicate" in str(exc_info.value)
    
    @patch('src.models.company_financial.CompanyFinancial.query')
    def test_check_financial_entry_duplicate_found(self, mock_query):
        """Test financial entry duplicate check when duplicate is found."""
        # Mock existing financial entry
        mock_entry = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_entry
        
        result = DuplicateDetectionService.check_financial_entry_duplicate(
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST ENTRY'
        )
        
        assert result is True
        mock_query.filter_by.assert_called_once_with(
            date=date(2024, 1, 15),
            amount=-100.0,
            description='TEST ENTRY'
        )
    
    @patch('src.models.company_financial.CompanyFinancial.query')
    def test_check_financial_entry_duplicate_not_found(self, mock_query):
        """Test financial entry duplicate check when no duplicate is found."""
        mock_query.filter_by.return_value.first.return_value = None
        
        result = DuplicateDetectionService.check_financial_entry_duplicate(
            date=date(2024, 1, 15),
            amount=-100.0,
            description='UNIQUE ENTRY'
        )
        
        assert result is False
        mock_query.filter_by.assert_called_once_with(
            date=date(2024, 1, 15),
            amount=-100.0,
            description='UNIQUE ENTRY'
        )
    
    @patch('src.models.company_financial.CompanyFinancial.query')
    def test_check_financial_entry_duplicate_database_error(self, mock_query):
        """Test financial entry duplicate check with database error."""
        mock_query.filter_by.side_effect = Exception("Database connection lost")
        
        with pytest.raises(Exception) as exc_info:
            DuplicateDetectionService.check_financial_entry_duplicate(
                date=date(2024, 1, 15),
                amount=-100.0,
                description='TEST'
            )
        
        assert "Error checking financial entry duplicate" in str(exc_info.value)
    
    @patch('src.models.transaction.Transaction.query')
    def test_get_duplicate_transactions(self, mock_query):
        """Test getting all duplicate transactions."""
        # Mock multiple duplicate transactions
        mock_transactions = [Mock(), Mock(), Mock()]
        mock_query.filter_by.return_value.all.return_value = mock_transactions
        
        result = DuplicateDetectionService.get_duplicate_transactions(
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='DUPLICATE TRANSACTION'
        )
        
        assert result == mock_transactions
        assert len(result) == 3
        mock_query.filter_by.assert_called_once_with(
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='DUPLICATE TRANSACTION'
        )
    
    @patch('src.models.transaction.Transaction.query')
    def test_get_duplicate_transactions_none_found(self, mock_query):
        """Test getting duplicate transactions when none are found."""
        mock_query.filter_by.return_value.all.return_value = []
        
        result = DuplicateDetectionService.get_duplicate_transactions(
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description='UNIQUE TRANSACTION'
        )
        
        assert result == []
        assert len(result) == 0
    
    @patch('src.models.transaction.Transaction.query')
    def test_get_duplicate_transactions_database_error(self, mock_query):
        """Test getting duplicate transactions with database error."""
        mock_query.filter_by.side_effect = Exception("Query execution failed")
        
        with pytest.raises(Exception) as exc_info:
            DuplicateDetectionService.get_duplicate_transactions(
                account_id='12345',
                date=date(2024, 1, 15),
                amount=-100.0,
                description='TEST'
            )
        
        assert "Error getting duplicate transactions" in str(exc_info.value)
    
    @patch('src.models.company_financial.CompanyFinancial.query')
    def test_get_duplicate_financial_entries(self, mock_query):
        """Test getting all duplicate financial entries."""
        # Mock multiple duplicate entries
        mock_entries = [Mock(), Mock()]
        mock_query.filter_by.return_value.all.return_value = mock_entries
        
        result = DuplicateDetectionService.get_duplicate_financial_entries(
            date=date(2024, 1, 15),
            amount=-100.0,
            description='DUPLICATE ENTRY'
        )
        
        assert result == mock_entries
        assert len(result) == 2
        mock_query.filter_by.assert_called_once_with(
            date=date(2024, 1, 15),
            amount=-100.0,
            description='DUPLICATE ENTRY'
        )
    
    @patch('src.models.company_financial.CompanyFinancial.query')
    def test_get_duplicate_financial_entries_none_found(self, mock_query):
        """Test getting duplicate financial entries when none are found."""
        mock_query.filter_by.return_value.all.return_value = []
        
        result = DuplicateDetectionService.get_duplicate_financial_entries(
            date=date(2024, 1, 15),
            amount=-100.0,
            description='UNIQUE ENTRY'
        )
        
        assert result == []
        assert len(result) == 0
    
    @patch('src.models.company_financial.CompanyFinancial.query')
    def test_get_duplicate_financial_entries_database_error(self, mock_query):
        """Test getting duplicate financial entries with database error."""
        mock_query.filter_by.side_effect = Exception("Database timeout")
        
        with pytest.raises(Exception) as exc_info:
            DuplicateDetectionService.get_duplicate_financial_entries(
                date=date(2024, 1, 15),
                amount=-100.0,
                description='TEST'
            )
        
        assert "Error getting duplicate financial entries" in str(exc_info.value)


class TestDuplicateDetectionServiceIntegration:
    """Integration tests for DuplicateDetectionService with real database operations."""
    
    def test_file_duplicate_workflow(self, db_session):
        """Test complete file duplicate detection workflow."""
        # Create test file
        test_content = b"Test file content for duplicate detection"
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_path = temp_file.name
        
        try:
            # Step 1: Calculate file hash
            file_hash = DuplicateDetectionService.calculate_file_hash(temp_path)
            assert isinstance(file_hash, str)
            assert len(file_hash) == 64
            
            # Step 2: Check for duplicate (should not exist initially)
            duplicate_check_1 = DuplicateDetectionService.check_file_duplicate(file_hash)
            assert duplicate_check_1 is None
            
            # Step 3: Create upload history record
            upload_record = UploadHistory(
                filename='test_file.ofx',
                bank_name='TEST_BANK',
                transactions_count=5,
                status='success',
                file_hash=file_hash
            )
            db_session.add(upload_record)
            db_session.commit()
            
            # Step 4: Check for duplicate again (should find it now)
            duplicate_check_2 = DuplicateDetectionService.check_file_duplicate(file_hash)
            assert duplicate_check_2 is not None
            assert duplicate_check_2.file_hash == file_hash
            assert duplicate_check_2.filename == 'test_file.ofx'
        
        finally:
            os.unlink(temp_path)
    
    def test_transaction_duplicate_workflow(self, db_session):
        """Test complete transaction duplicate detection workflow."""
        # Test data
        account_id = '12345-6'
        test_date = date(2024, 1, 15)
        test_amount = -100.0
        test_description = 'TEST TRANSACTION'
        
        # Step 1: Check for duplicate (should not exist initially)
        duplicate_check_1 = DuplicateDetectionService.check_transaction_duplicate(
            account_id, test_date, test_amount, test_description
        )
        assert duplicate_check_1 is False
        
        # Step 2: Create transaction
        transaction = Transaction(
            bank_name='TEST_BANK',
            account_id=account_id,
            date=test_date,
            amount=test_amount,
            description=test_description,
            transaction_type='debit'
        )
        db_session.add(transaction)
        db_session.commit()
        
        # Step 3: Check for duplicate again (should find it now)
        duplicate_check_2 = DuplicateDetectionService.check_transaction_duplicate(
            account_id, test_date, test_amount, test_description
        )
        assert duplicate_check_2 is True
        
        # Step 4: Get all duplicate transactions
        duplicates = DuplicateDetectionService.get_duplicate_transactions(
            account_id, test_date, test_amount, test_description
        )
        assert len(duplicates) == 1
        assert duplicates[0].account_id == account_id
        assert duplicates[0].amount == test_amount
    
    def test_financial_entry_duplicate_workflow(self, db_session):
        """Test complete financial entry duplicate detection workflow."""
        # Test data
        test_date = date(2024, 1, 15)
        test_amount = -100.0
        test_description = 'TEST FINANCIAL ENTRY'
        
        # Step 1: Check for duplicate (should not exist initially)
        duplicate_check_1 = DuplicateDetectionService.check_financial_entry_duplicate(
            test_date, test_amount, test_description
        )
        assert duplicate_check_1 is False
        
        # Step 2: Create financial entry
        entry = CompanyFinancial(
            date=test_date,
            amount=test_amount,
            description=test_description,
            transaction_type='expense'
        )
        db_session.add(entry)
        db_session.commit()
        
        # Step 3: Check for duplicate again (should find it now)
        duplicate_check_2 = DuplicateDetectionService.check_financial_entry_duplicate(
            test_date, test_amount, test_description
        )
        assert duplicate_check_2 is True
        
        # Step 4: Get all duplicate entries
        duplicates = DuplicateDetectionService.get_duplicate_financial_entries(
            test_date, test_amount, test_description
        )
        assert len(duplicates) == 1
        assert duplicates[0].date == test_date
        assert duplicates[0].amount == test_amount
    
    def test_multiple_duplicates_detection(self, db_session):
        """Test detection of multiple duplicate entries."""
        # Test data
        test_date = date(2024, 1, 15)
        test_amount = -100.0
        test_description = 'MULTIPLE DUPLICATE TEST'
        
        # Create multiple identical transactions
        for i in range(3):
            transaction = Transaction(
                bank_name='TEST_BANK',
                account_id='12345-6',
                date=test_date,
                amount=test_amount,
                description=test_description,
                transaction_type='debit'
            )
            db_session.add(transaction)
        
        db_session.commit()
        
        # Get all duplicates
        duplicates = DuplicateDetectionService.get_duplicate_transactions(
            '12345-6', test_date, test_amount, test_description
        )
        
        assert len(duplicates) == 3
        for duplicate in duplicates:
            assert duplicate.account_id == '12345-6'
            assert duplicate.date == test_date
            assert duplicate.amount == test_amount
            assert duplicate.description == test_description
    
    def test_case_sensitive_duplicate_detection(self, db_session):
        """Test that duplicate detection is case sensitive for descriptions."""
        # Test data with different cases
        base_data = {
            'date': date(2024, 1, 15),
            'amount': -100.0
        }
        
        descriptions = [
            'Test Transaction',
            'TEST TRANSACTION',
            'test transaction',
            'Test transaction'
        ]
        
        # Create entries with different case descriptions
        for desc in descriptions:
            entry = CompanyFinancial(
                description=desc,
                transaction_type='expense',
                **base_data
            )
            db_session.add(entry)
        
        db_session.commit()
        
        # Check each description separately
        for desc in descriptions:
            duplicates = DuplicateDetectionService.get_duplicate_financial_entries(
                base_data['date'], base_data['amount'], desc
            )
            # Should only find the exact match (case sensitive)
            assert len(duplicates) == 1
            assert duplicates[0].description == desc
    
    def test_date_precision_duplicate_detection(self, db_session):
        """Test duplicate detection with different date precisions."""
        # Test data
        base_data = {
            'amount': -100.0,
            'description': 'DATE PRECISION TEST'
        }
        
        # Create entries with same date but different times (should still be duplicates for date-only comparison)
        dates = [
            date(2024, 1, 15),
            date(2024, 1, 15),  # Same date
            date(2024, 1, 16)   # Different date
        ]
        
        for test_date in dates:
            entry = CompanyFinancial(
                date=test_date,
                transaction_type='expense',
                **base_data
            )
            db_session.add(entry)
        
        db_session.commit()
        
        # Check for duplicates on 2024-01-15
        duplicates_jan_15 = DuplicateDetectionService.get_duplicate_financial_entries(
            date(2024, 1, 15), base_data['amount'], base_data['description']
        )
        assert len(duplicates_jan_15) == 2  # Two entries with same date
        
        # Check for duplicates on 2024-01-16
        duplicates_jan_16 = DuplicateDetectionService.get_duplicate_financial_entries(
            date(2024, 1, 16), base_data['amount'], base_data['description']
        )
        assert len(duplicates_jan_16) == 1  # One entry with this date
    
    def test_amount_precision_duplicate_detection(self, db_session):
        """Test duplicate detection with different amount precisions."""
        # Test data
        base_data = {
            'date': date(2024, 1, 15),
            'description': 'AMOUNT PRECISION TEST'
        }
        
        # Create entries with slightly different amounts
        amounts = [-100.0, -100.00, -100.01, -99.99]
        
        for amount in amounts:
            entry = CompanyFinancial(
                amount=amount,
                transaction_type='expense',
                **base_data
            )
            db_session.add(entry)
        
        db_session.commit()
        
        # Check for exact matches
        for amount in amounts:
            duplicates = DuplicateDetectionService.get_duplicate_financial_entries(
                base_data['date'], amount, base_data['description']
            )
            # Should only find exact amount matches
            assert len(duplicates) >= 1
            for duplicate in duplicates:
                assert duplicate.amount == amount


class TestDuplicateDetectionServicePerformance:
    """Performance tests for DuplicateDetectionService."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_file_hash_calculation(self):
        """Test hash calculation performance with large files."""
        # Create a large file (10MB)
        large_content = b"Large file content for performance testing " * 250000  # ~10MB
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(large_content)
            temp_path = temp_file.name
        
        try:
            import time
            start_time = time.time()
            
            file_hash = DuplicateDetectionService.calculate_file_hash(temp_path)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process within reasonable time
            assert processing_time < 5  # 5 seconds max for 10MB file
            assert isinstance(file_hash, str)
            assert len(file_hash) == 64
        
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.performance
    def test_multiple_duplicate_checks_performance(self, db_session):
        """Test performance of multiple duplicate checks."""
        # Create a base set of transactions
        base_transactions = []
        for i in range(100):
            transaction = Transaction(
                bank_name='TEST_BANK',
                account_id=f'ACC_{i:03d}',
                date=date(2024, 1, i % 28 + 1),
                amount=-100.0 - i,
                description=f'TRANSACTION {i}',
                transaction_type='debit'
            )
            base_transactions.append(transaction)
            db_session.add(transaction)
        
        db_session.commit()
        
        # Perform multiple duplicate checks
        import time
        start_time = time.time()
        
        duplicate_count = 0
        for i in range(100):
            is_duplicate = DuplicateDetectionService.check_transaction_duplicate(
                f'ACC_{i:03d}',
                date(2024, 1, i % 28 + 1),
                -100.0 - i,
                f'TRANSACTION {i}'
            )
            if is_duplicate:
                duplicate_count += 1
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process within reasonable time
        assert processing_time < 2  # 2 seconds max for 100 checks
        assert duplicate_count == 100  # All should be duplicates
    
    @pytest.mark.performance
    def test_hash_calculation_consistency(self):
        """Test that hash calculation is consistent across multiple runs."""
        test_content = b"Consistency test content for hash calculation"
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_path = temp_file.name
        
        try:
            # Calculate hash multiple times
            hashes = []
            for _ in range(10):
                file_hash = DuplicateDetectionService.calculate_file_hash(temp_path)
                hashes.append(file_hash)
            
            # All hashes should be identical
            assert len(set(hashes)) == 1  # All hashes are the same
            assert all(len(h) == 64 for h in hashes)  # All are valid SHA-256 hashes
        
        finally:
            os.unlink(temp_path)


class TestDuplicateDetectionServiceEdgeCases:
    """Test edge cases for DuplicateDetectionService."""
    
    def test_hash_calculation_binary_file(self):
        """Test hash calculation with binary file content."""
        # Create binary content with various byte values
        binary_content = bytes(range(256))  # All possible byte values
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(binary_content)
            temp_path = temp_file.name
        
        try:
            file_hash = DuplicateDetectionService.calculate_file_hash(temp_path)
            
            # Calculate expected hash
            expected_hash = hashlib.sha256(binary_content).hexdigest()
            
            assert file_hash == expected_hash
            assert len(file_hash) == 64
        
        finally:
            os.unlink(temp_path)
    
    def test_duplicate_detection_with_none_values(self, db_session):
        """Test duplicate detection with None values in database."""
        # Create transaction with None description
        transaction = Transaction(
            bank_name='TEST_BANK',
            account_id='12345',
            date=date(2024, 1, 15),
            amount=-100.0,
            description=None,  # None description
            transaction_type='debit'
        )
        db_session.add(transaction)
        db_session.commit()
        
        # Check for duplicate with None description
        is_duplicate = DuplicateDetectionService.check_transaction_duplicate(
            '12345',
            date(2024, 1, 15),
            -100.0,
            None
        )
        
        assert is_duplicate is True
    
    def test_duplicate_detection_with_special_characters(self, db_session):
        """Test duplicate detection with special characters in descriptions."""
        special_descriptions = [
            'Transaction with émojis 🏦💰',
            'Special chars: !@#$%^&*()',
            'Unicode: αβγδε',
            'Newlines\nand\ttabs',
            'Quotes "single" and \'double\'',
            'SQL injection attempt\'; DROP TABLE transactions; --'
        ]
        
        for i, desc in enumerate(special_descriptions):
            entry = CompanyFinancial(
                date=date(2024, 1, 15),
                amount=-100.0 - i,
                description=desc,
                transaction_type='expense'
            )
            db_session.add(entry)
        
        db_session.commit()
        
        # Check each special description
        for i, desc in enumerate(special_descriptions):
            is_duplicate = DuplicateDetectionService.check_financial_entry_duplicate(
                date(2024, 1, 15),
                -100.0 - i,
                desc
            )
            assert is_duplicate is True
    
    def test_duplicate_detection_with_extreme_amounts(self, db_session):
        """Test duplicate detection with extreme amount values."""
        extreme_amounts = [
            0.0,           # Zero
            0.01,          # Very small positive
            -0.01,         # Very small negative
            999999999.99,  # Very large positive
            -999999999.99, # Very large negative
            float('inf'),  # Infinity (if supported)
            float('-inf')  # Negative infinity (if supported)
        ]
        
        for i, amount in enumerate(extreme_amounts):
            try:
                entry = CompanyFinancial(
                    date=date(2024, 1, 15),
                    amount=amount,
                    description=f'EXTREME AMOUNT TEST {i}',
                    transaction_type='expense' if amount < 0 else 'income'
                )
                db_session.add(entry)
                db_session.commit()
                
                # Check for duplicate
                is_duplicate = DuplicateDetectionService.check_financial_entry_duplicate(
                    date(2024, 1, 15),
                    amount,
                    f'EXTREME AMOUNT TEST {i}'
                )
                assert is_duplicate is True
                
            except (ValueError, OverflowError):
                # Skip if the database can't handle extreme values
                pytest.skip(f"Database cannot handle extreme amount: {amount}")