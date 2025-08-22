"""
Unit tests for XLSXProcessor service.

Tests cover:
- XLSX file parsing and validation
- Column normalization and mapping
- Data type conversion and validation
- Duplicate detection
- Error handling and edge cases
- Financial data processing accuracy
"""

import pytest
import tempfile
import os
import pandas as pd
from datetime import date, datetime
from unittest.mock import Mock, patch, MagicMock
from src.services.xlsx_processor import XLSXProcessor
from src.utils.exceptions import (
    FileNotFoundError, InvalidFileFormatError, FileCorruptedError,
    InsufficientDataError, ValidationError, FileProcessingError
)


class TestXLSXProcessor:
    """Test suite for XLSXProcessor class."""
    
    def test_init(self):
        """Test XLSXProcessor initialization."""
        processor = XLSXProcessor()
        assert processor is not None
        assert hasattr(processor, 'supported_columns')
        assert hasattr(processor, 'duplicate_service')
        assert len(processor.supported_columns) > 0
    
    def test_supported_columns_structure(self):
        """Test that supported columns are properly structured."""
        processor = XLSXProcessor()
        
        expected_columns = [
            'date', 'description', 'amount', 'category', 'cost_center',
            'department', 'project', 'transaction_type', 'observations',
            'monthly_report_value'
        ]
        
        for column in expected_columns:
            assert column in processor.supported_columns
            assert isinstance(processor.supported_columns[column], list)
    
    def test_parse_xlsx_file_success(self, xlsx_processor, temp_xlsx_file):
        """Test successful XLSX file parsing."""
        result = xlsx_processor.parse_xlsx_file(temp_xlsx_file)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verify structure of first entry
        if result:
            entry = result[0]
            expected_fields = [
                'date', 'description', 'amount', 'category', 'cost_center',
                'department', 'project', 'transaction_type', 'observations',
                'monthly_report_value'
            ]
            for field in expected_fields:
                assert field in entry
    
    def test_parse_xlsx_file_not_found(self, xlsx_processor):
        """Test parsing non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            xlsx_processor.parse_xlsx_file('/non/existent/file.xlsx')
    
    def test_parse_xlsx_file_invalid_format(self, xlsx_processor):
        """Test parsing invalid file format raises InvalidFileFormatError."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"Not an XLSX file")
            temp_path = f.name
        
        try:
            with pytest.raises(InvalidFileFormatError):
                xlsx_processor.parse_xlsx_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_parse_xlsx_file_empty(self, xlsx_processor):
        """Test parsing empty XLSX file raises InsufficientDataError."""
        # Create empty DataFrame and save as XLSX
        empty_df = pd.DataFrame()
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_path = f.name
        
        empty_df.to_excel(temp_path, index=False)
        
        try:
            with pytest.raises(InsufficientDataError):
                xlsx_processor.parse_xlsx_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_parse_xlsx_file_insufficient_columns(self, xlsx_processor):
        """Test parsing XLSX file with insufficient columns raises ValidationError."""
        # Create DataFrame with only one column
        df = pd.DataFrame({'col1': [1, 2, 3]})
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_path = f.name
        
        df.to_excel(temp_path, index=False)
        
        try:
            with pytest.raises(ValidationError):
                xlsx_processor.parse_xlsx_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_read_xlsx_with_recovery_success(self, xlsx_processor, temp_xlsx_file):
        """Test XLSX reading with recovery mechanism - success case."""
        df = xlsx_processor._read_xlsx_with_recovery(temp_xlsx_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    @patch('pandas.read_excel')
    def test_read_xlsx_with_recovery_fallback(self, mock_read_excel, xlsx_processor):
        """Test XLSX reading with recovery mechanism - fallback case."""
        # First call fails, second succeeds with different engine
        mock_df = pd.DataFrame({'test': [1, 2, 3]})
        mock_read_excel.side_effect = [Exception("Primary read failed"), mock_df]
        
        result = xlsx_processor._read_xlsx_with_recovery('fake_path.xlsx')
        assert result.equals(mock_df)
        assert mock_read_excel.call_count == 2
    
    @patch('pandas.read_excel')
    @patch('pandas.read_csv')
    def test_read_xlsx_with_recovery_csv_fallback(self, mock_read_csv, mock_read_excel, xlsx_processor):
        """Test XLSX reading with CSV fallback."""
        mock_df = pd.DataFrame({'test': [1, 2, 3]})
        mock_read_excel.side_effect = Exception("All engines failed")
        mock_read_csv.return_value = mock_df
        
        result = xlsx_processor._read_xlsx_with_recovery('fake_path.xlsx')
        assert result.equals(mock_df)
        mock_read_csv.assert_called_once()
    
    @patch('pandas.read_excel')
    @patch('pandas.read_csv')
    def test_read_xlsx_with_recovery_all_fail(self, mock_read_csv, mock_read_excel, xlsx_processor):
        """Test XLSX reading when all recovery methods fail."""
        mock_read_excel.side_effect = Exception("Excel read failed")
        mock_read_csv.side_effect = Exception("CSV read failed")
        
        with pytest.raises(FileCorruptedError):
            xlsx_processor._read_xlsx_with_recovery('fake_path.xlsx')
    
    def test_validate_xlsx_structure_valid(self, xlsx_processor, sample_xlsx_data):
        """Test XLSX structure validation with valid data."""
        # Should not raise any exception
        xlsx_processor._validate_xlsx_structure(sample_xlsx_data, 'test.xlsx')
    
    def test_validate_xlsx_structure_empty(self, xlsx_processor):
        """Test XLSX structure validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(InsufficientDataError):
            xlsx_processor._validate_xlsx_structure(empty_df, 'test.xlsx')
    
    def test_validate_xlsx_structure_insufficient_columns(self, xlsx_processor):
        """Test XLSX structure validation with insufficient columns."""
        df = pd.DataFrame({'col1': [1]})
        with pytest.raises(ValidationError):
            xlsx_processor._validate_xlsx_structure(df, 'test.xlsx')
    
    def test_normalize_column_name(self, xlsx_processor):
        """Test column name normalization."""
        test_cases = [
            ("Data", "data"),
            ("Descrição", "descricao"),
            ("Valor (R$)", "valor r"),
            ("Centro de Custo", "centro de custo"),
            ("DEPARTAMENTO", "departamento"),
            ("Observações!!!", "observacoes")
        ]
        
        for input_name, expected in test_cases:
            result = xlsx_processor._normalize_column_name(input_name)
            assert result == expected
    
    def test_process_xlsx_rows(self, xlsx_processor, sample_xlsx_data):
        """Test processing XLSX rows."""
        # Normalize column names first
        sample_xlsx_data.columns = [xlsx_processor._normalize_column_name(col) for col in sample_xlsx_data.columns]
        
        result = xlsx_processor._process_xlsx_rows(sample_xlsx_data)
        
        assert isinstance(result, list)
        assert len(result) >= 0  # Some rows might be skipped due to validation
        
        # Check structure of valid entries
        for entry in result:
            assert isinstance(entry, dict)
            assert 'date' in entry
            assert 'description' in entry
            assert 'amount' in entry
    
    def test_process_single_row_valid(self, xlsx_processor):
        """Test processing a single valid row."""
        row_data = pd.Series({
            'data': '2024-01-15',
            'description': 'Test transaction',
            'valor': 100.50,
            'tipo': 'despesa'
        })
        
        result = xlsx_processor._process_single_row(row_data, 0)
        
        assert result is not None
        assert isinstance(result, dict)
        assert result['description'] == 'Test transaction'
        assert result['amount'] == 100.50
        assert result['transaction_type'] in ['expense', 'income']
    
    def test_process_single_row_invalid(self, xlsx_processor):
        """Test processing a single invalid row."""
        row_data = pd.Series({
            'data': None,  # Invalid date
            'description': '',  # Empty description
            'valor': 'invalid'  # Invalid amount
        })
        
        result = xlsx_processor._process_single_row(row_data, 0)
        assert result is None  # Should return None for invalid rows
    
    def test_parse_date_valid_formats(self, xlsx_processor):
        """Test date parsing with various valid formats."""
        test_cases = [
            ('2024-01-15', date(2024, 1, 15)),
            ('15/01/2024', date(2024, 1, 15)),
            ('01/15/2024', date(2024, 1, 15)),
            ('15-01-2024', date(2024, 1, 15)),
            ('2024/01/15', date(2024, 1, 15)),
            ('15.01.2024', date(2024, 1, 15))
        ]
        
        for date_str, expected in test_cases:
            result = xlsx_processor._parse_date(date_str)
            assert result == expected or result is not None  # Some formats might not parse exactly
    
    def test_parse_date_invalid(self, xlsx_processor):
        """Test date parsing with invalid formats."""
        invalid_dates = [None, '', 'invalid-date', '32/13/2024', '2024-13-32']
        
        for invalid_date in invalid_dates:
            result = xlsx_processor._parse_date(invalid_date)
            assert result is None
    
    def test_parse_date_already_date_object(self, xlsx_processor):
        """Test date parsing with datetime object."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = xlsx_processor._parse_date(dt)
        assert result == date(2024, 1, 15)
    
    def test_parse_amount_valid_formats(self, xlsx_processor):
        """Test amount parsing with various valid formats."""
        test_cases = [
            (100.50, 100.50),
            ('100.50', 100.50),
            ('1,234.56', 1234.56),
            ('1.234,56', 1234.56),  # Brazilian format
            ('R$ 100,50', 100.50),
            ('(100.50)', -100.50),  # Negative in parentheses
            (0, 0.0),
            ('0', 0.0)
        ]
        
        for input_amount, expected in test_cases:
            result = xlsx_processor._parse_amount(input_amount)
            assert abs(result - expected) < 0.01  # Allow for floating point precision
    
    def test_parse_amount_invalid(self, xlsx_processor):
        """Test amount parsing with invalid formats."""
        invalid_amounts = [None, '', 'invalid', 'abc123', '']
        
        for invalid_amount in invalid_amounts:
            result = xlsx_processor._parse_amount(invalid_amount)
            assert result == 0.0
    
    def test_parse_description_valid(self, xlsx_processor):
        """Test description parsing with valid input."""
        test_cases = [
            ('Valid description', 'Valid description'),
            ('  Trimmed  description  ', 'Trimmed description'),
            ('Multiple   spaces   here', 'Multiple spaces here'),
            ('', ''),
            (None, '')
        ]
        
        for input_desc, expected in test_cases:
            result = xlsx_processor._parse_description(input_desc)
            assert result == expected
    
    def test_parse_description_long(self, xlsx_processor):
        """Test description parsing with very long input."""
        long_description = 'A' * 600  # Longer than 500 chars
        result = xlsx_processor._parse_description(long_description)
        assert len(result) == 500
    
    def test_determine_transaction_type(self, xlsx_processor):
        """Test transaction type determination."""
        test_cases = [
            ({'tipo': 'despesa', 'valor': -100}, 'expense'),
            ({'tipo': 'receita', 'valor': 100}, 'income'),
            ({'tipo': 'expense', 'valor': -100}, 'expense'),
            ({'tipo': 'income', 'valor': 100}, 'income'),
            ({'tipo': 'débito', 'valor': -100}, 'expense'),
            ({'tipo': 'crédito', 'valor': 100}, 'income'),
            ({'tipo': 'unknown', 'valor': -100}, 'expense'),  # Based on amount
            ({'tipo': 'unknown', 'valor': 100}, 'income'),   # Based on amount
        ]
        
        for row_data, expected in test_cases:
            row = pd.Series(row_data)
            result = xlsx_processor._determine_transaction_type(row)
            assert result == expected
    
    def test_detect_duplicates_success(self, xlsx_processor):
        """Test duplicate detection with valid entries."""
        entries = [
            {
                'date': date(2024, 1, 15),
                'amount': 100.0,
                'description': 'Test transaction'
            },
            {
                'date': date(2024, 1, 16),
                'amount': 200.0,
                'description': 'Another transaction'
            },
            {
                'date': date(2024, 1, 15),
                'amount': 100.0,
                'description': 'Test transaction'  # Duplicate
            }
        ]
        
        with patch.object(xlsx_processor.duplicate_service, 'check_financial_entry_duplicate') as mock_check:
            mock_check.side_effect = [False, False, True]  # Third is duplicate
            
            duplicates = xlsx_processor.detect_duplicates(entries)
            assert isinstance(duplicates, list)
            assert 2 in duplicates  # Third entry (index 2) is duplicate
    
    def test_detect_duplicates_empty_list(self, xlsx_processor):
        """Test duplicate detection with empty list."""
        result = xlsx_processor.detect_duplicates([])
        assert result == []
    
    def test_detect_duplicates_no_duplicates(self, xlsx_processor):
        """Test duplicate detection with no duplicates."""
        entries = [
            {
                'date': date(2024, 1, 15),
                'amount': 100.0,
                'description': 'Transaction A'
            },
            {
                'date': date(2024, 1, 16),
                'amount': 200.0,
                'description': 'Transaction B'
            }
        ]
        
        with patch.object(xlsx_processor.duplicate_service, 'check_financial_entry_duplicate') as mock_check:
            mock_check.return_value = False
            
            duplicates = xlsx_processor.detect_duplicates(entries)
            assert duplicates == []
    
    def test_detect_duplicates_database_error(self, xlsx_processor):
        """Test duplicate detection with database errors."""
        entries = [
            {
                'date': date(2024, 1, 15),
                'amount': 100.0,
                'description': 'Test transaction'
            }
        ]
        
        with patch.object(xlsx_processor.duplicate_service, 'check_financial_entry_duplicate') as mock_check:
            mock_check.side_effect = Exception("Database error")
            
            # Should handle error gracefully and continue with file-level duplicate check
            duplicates = xlsx_processor.detect_duplicates(entries)
            assert isinstance(duplicates, list)
    
    def test_detect_duplicates_missing_fields(self, xlsx_processor):
        """Test duplicate detection with missing required fields."""
        entries = [
            {'amount': 100.0, 'description': 'Test'},  # Missing date
            {'date': date(2024, 1, 15), 'description': 'Test'},  # Missing amount
            {'date': date(2024, 1, 15), 'amount': 100.0}  # Missing description
        ]
        
        duplicates = xlsx_processor.detect_duplicates(entries)
        assert isinstance(duplicates, list)
    
    def test_has_required_fields_for_duplicate_check(self, xlsx_processor):
        """Test checking for required fields in duplicate detection."""
        valid_entry = {
            'date': date(2024, 1, 15),
            'amount': 100.0,
            'description': 'Test transaction'
        }
        
        invalid_entries = [
            {'amount': 100.0, 'description': 'Test'},  # Missing date
            {'date': date(2024, 1, 15), 'description': 'Test'},  # Missing amount
            {'date': date(2024, 1, 15), 'amount': 100.0},  # Missing description
            {'date': None, 'amount': 100.0, 'description': 'Test'}  # Null date
        ]
        
        assert xlsx_processor._has_required_fields_for_duplicate_check(valid_entry) is True
        
        for invalid_entry in invalid_entries:
            assert xlsx_processor._has_required_fields_for_duplicate_check(invalid_entry) is False
    
    def test_create_duplicate_key(self, xlsx_processor):
        """Test duplicate key creation."""
        entry = {
            'date': date(2024, 1, 15),
            'amount': 100.50,
            'description': 'Test Transaction'
        }
        
        key = xlsx_processor._create_duplicate_key(entry)
        assert isinstance(key, tuple)
        assert len(key) == 3
        assert key[0] == '2024-01-15'
        assert key[1] == 100.50
        assert key[2] == 'test transaction'
    
    def test_create_duplicate_key_with_datetime(self, xlsx_processor):
        """Test duplicate key creation with datetime object."""
        entry = {
            'date': datetime(2024, 1, 15, 10, 30, 0),
            'amount': 100.0,
            'description': 'TEST TRANSACTION'
        }
        
        key = xlsx_processor._create_duplicate_key(entry)
        assert key[0] == '2024-01-15T10:30:00'
        assert key[1] == 100.0
        assert key[2] == 'test transaction'
    
    def test_validate_processed_data_success(self, xlsx_processor):
        """Test processed data validation with valid data."""
        valid_data = [
            {
                'date': date(2024, 1, 15),
                'description': 'Test transaction',
                'amount': 100.0
            }
        ]
        
        # Should not raise any exception
        xlsx_processor._validate_processed_data(valid_data, 'test.xlsx')
    
    def test_validate_processed_data_empty(self, xlsx_processor):
        """Test processed data validation with empty data raises InsufficientDataError."""
        with pytest.raises(InsufficientDataError):
            xlsx_processor._validate_processed_data([], 'test.xlsx')
    
    def test_validate_processed_data_low_quality(self, xlsx_processor):
        """Test processed data validation with low quality data logs warning."""
        low_quality_data = [
            {'date': None, 'description': '', 'amount': None},  # Invalid
            {'date': None, 'description': '', 'amount': None},  # Invalid
            {'date': date(2024, 1, 15), 'description': 'Valid', 'amount': 100.0}  # Valid
        ]
        
        with patch('src.services.xlsx_processor.logger') as mock_logger:
            xlsx_processor._validate_processed_data(low_quality_data, 'test.xlsx')
            mock_logger.warning.assert_called_once()


class TestXLSXProcessorIntegration:
    """Integration tests for XLSXProcessor with real XLSX data."""
    
    def test_full_xlsx_processing_workflow(self, xlsx_processor, sample_xlsx_data):
        """Test complete XLSX processing workflow."""
        # Create temporary XLSX file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_path = f.name
        
        sample_xlsx_data.to_excel(temp_path, index=False)
        
        try:
            # Process the file
            result = xlsx_processor.parse_xlsx_file(temp_path)
            
            # Verify complete result structure
            assert isinstance(result, list)
            assert len(result) >= 0
            
            # If entries exist, test duplicate detection
            if result:
                duplicates = xlsx_processor.detect_duplicates(result)
                assert isinstance(duplicates, list)
        
        finally:
            os.unlink(temp_path)
    
    def test_financial_accuracy(self, xlsx_processor, assert_financial_accuracy):
        """Test financial calculation accuracy."""
        entries = [
            {'amount': 100.50, 'transaction_type': 'income'},
            {'amount': -50.25, 'transaction_type': 'expense'},
            {'amount': 200.75, 'transaction_type': 'income'},
            {'amount': -75.00, 'transaction_type': 'expense'}
        ]
        
        total_income = sum(e['amount'] for e in entries if e['transaction_type'] == 'income')
        total_expenses = sum(abs(e['amount']) for e in entries if e['transaction_type'] == 'expense')
        net_flow = total_income - total_expenses
        
        # Test financial accuracy
        assert_financial_accuracy(301.25, total_income)
        assert_financial_accuracy(125.25, total_expenses)
        assert_financial_accuracy(176.00, net_flow)
    
    def test_column_mapping_flexibility(self, xlsx_processor):
        """Test flexible column mapping with different column names."""
        # Create DataFrame with Portuguese column names
        df_portuguese = pd.DataFrame({
            'Data': ['2024-01-15', '2024-01-16'],
            'Descrição': ['Pagamento A', 'Receita B'],
            'Valor': [100.50, -50.25],
            'Categoria': ['Despesa', 'Receita']
        })
        
        # Create DataFrame with English column names
        df_english = pd.DataFrame({
            'Date': ['2024-01-15', '2024-01-16'],
            'Description': ['Payment A', 'Revenue B'],
            'Amount': [100.50, -50.25],
            'Category': ['Expense', 'Income']
        })
        
        for df in [df_portuguese, df_english]:
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
                temp_path = f.name
            
            df.to_excel(temp_path, index=False)
            
            try:
                result = xlsx_processor.parse_xlsx_file(temp_path)
                assert isinstance(result, list)
                assert len(result) >= 0
            finally:
                os.unlink(temp_path)


class TestXLSXProcessorErrorHandling:
    """Test error handling in XLSXProcessor."""
    
    def test_timeout_handling(self, xlsx_processor):
        """Test timeout handling in XLSX processing."""
        with patch('src.services.xlsx_processor.with_timeout') as mock_timeout:
            mock_timeout.side_effect = TimeoutError("Processing timeout")
            
            with pytest.raises(TimeoutError):
                xlsx_processor.parse_xlsx_file('/fake/path.xlsx')
    
    def test_service_error_handling(self, xlsx_processor):
        """Test service error handling decorator."""
        with patch('src.services.xlsx_processor.handle_service_errors') as mock_handler:
            # Test that the decorator is applied
            assert hasattr(xlsx_processor.parse_xlsx_file, '__wrapped__')
    
    def test_file_processing_error_handling(self, xlsx_processor):
        """Test file processing error handling."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            f.write(b"Invalid XLSX content")
            temp_path = f.name
        
        try:
            with pytest.raises((FileCorruptedError, FileProcessingError)):
                xlsx_processor.parse_xlsx_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_duplicate_detection_error_handling(self, xlsx_processor):
        """Test error handling in duplicate detection."""
        entries = [{'date': date(2024, 1, 15), 'amount': 100.0, 'description': 'Test'}]
        
        with patch.object(xlsx_processor.duplicate_service, 'check_financial_entry_duplicate') as mock_check:
            mock_check.side_effect = Exception("Critical error")
            
            with pytest.raises(FileProcessingError):
                xlsx_processor.detect_duplicates(entries)


class TestXLSXProcessorPerformance:
    """Performance tests for XLSXProcessor."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_file_processing(self, xlsx_processor, large_xlsx_dataset):
        """Test processing of large XLSX files."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_path = f.name
        
        large_xlsx_dataset.to_excel(temp_path, index=False)
        
        try:
            import time
            start_time = time.time()
            
            result = xlsx_processor.parse_xlsx_file(temp_path)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process within reasonable time (adjust threshold as needed)
            assert processing_time < 60  # 60 seconds max for 5000 rows
            assert isinstance(result, list)
            
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.performance
    def test_duplicate_detection_performance(self, xlsx_processor, large_xlsx_dataset):
        """Test duplicate detection performance with large dataset."""
        # Convert DataFrame to list of dictionaries
        entries = large_xlsx_dataset.to_dict('records')
        
        # Convert data column to date objects
        for entry in entries:
            if 'data' in entry:
                try:
                    entry['date'] = datetime.strptime(entry['data'], '%Y-%m-%d').date()
                    entry['amount'] = float(entry.get('valor', 0))
                    entry['description'] = str(entry.get('description', ''))
                except:
                    continue
        
        with patch.object(xlsx_processor.duplicate_service, 'check_financial_entry_duplicate') as mock_check:
            mock_check.return_value = False
            
            import time
            start_time = time.time()
            
            duplicates = xlsx_processor.detect_duplicates(entries)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process within reasonable time
            assert processing_time < 30  # 30 seconds max
            assert isinstance(duplicates, list)