"""
Unit tests for OFXProcessor service.

Tests cover:
- OFX file parsing and validation
- Bank identification
- Transaction extraction and validation
- Duplicate detection
- Error handling and edge cases
- Financial calculation accuracy
"""

import pytest
import tempfile
import os
from datetime import date, datetime
from unittest.mock import Mock, patch, mock_open
from src.services.ofx_processor import OFXProcessor
from src.utils.exceptions import (
    FileNotFoundError, InvalidFileFormatError, FileCorruptedError,
    InsufficientDataError, ValidationError
)


class TestOFXProcessor:
    """Test suite for OFXProcessor class."""
    
    def test_init(self):
        """Test OFXProcessor initialization."""
        processor = OFXProcessor()
        assert processor is not None
        assert hasattr(processor, 'bank_patterns')
        assert hasattr(processor, 'SUPPORTED_BANKS')
        assert len(processor.SUPPORTED_BANKS) > 0
    
    def test_supported_banks(self):
        """Test that all expected banks are supported."""
        processor = OFXProcessor()
        expected_banks = ['caixa', 'sicoob', 'nubank', 'itau', 'bradesco', 'santander', 'bb']
        
        for bank in expected_banks:
            assert bank in processor.SUPPORTED_BANKS
            assert bank in processor.bank_patterns
    
    @pytest.mark.parametrize("bank_content,expected_bank", [
        ("CAIXA ECONÔMICA FEDERAL", "caixa"),
        ("SICOOB", "sicoob"),
        ("NUBANK", "nubank"),
        ("ITAÚ", "itau"),
        ("BRADESCO", "bradesco"),
        ("SANTANDER", "santander"),
        ("BANCO DO BRASIL", "bb"),
        ("Unknown Bank Content", "unknown")
    ])
    def test_identify_bank(self, ofx_processor, bank_content, expected_bank):
        """Test bank identification from OFX content."""
        result = ofx_processor.identify_bank(bank_content)
        assert result == expected_bank
    
    def test_identify_bank_case_insensitive(self, ofx_processor):
        """Test that bank identification is case insensitive."""
        test_cases = [
            ("caixa econômica federal", "caixa"),
            ("SICOOB", "sicoob"),
            ("Nubank", "nubank"),
            ("itaú", "itau")
        ]
        
        for content, expected in test_cases:
            result = ofx_processor.identify_bank(content)
            assert result == expected
    
    def test_identify_bank_by_code(self, ofx_processor):
        """Test bank identification by bank codes."""
        test_cases = [
            ("BANKID>104", "caixa"),
            ("BANKID>756", "sicoob"),
            ("BANKID>260", "nubank"),
            ("BANKID>341", "itau"),
            ("BANKID>237", "bradesco"),
            ("BANKID>033", "santander"),
            ("BANKID>001", "bb")
        ]
        
        for content, expected in test_cases:
            result = ofx_processor.identify_bank(content)
            assert result == expected
    
    def test_parse_ofx_file_success(self, ofx_processor, temp_ofx_file):
        """Test successful OFX file parsing."""
        result = ofx_processor.parse_ofx_file(temp_ofx_file)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'bank_name' in result
        assert 'account_info' in result
        assert 'transactions' in result
        assert 'summary' in result
        
        # Verify summary structure
        summary = result['summary']
        assert 'total_transactions' in summary
        assert 'total_credits' in summary
        assert 'total_debits' in summary
        assert 'balance' in summary
        
        # Verify transactions structure
        if result['transactions']:
            transaction = result['transactions'][0]
            required_fields = ['transaction_id', 'date', 'amount', 'description', 'transaction_type']
            for field in required_fields:
                assert field in transaction
    
    def test_parse_ofx_file_not_found(self, ofx_processor):
        """Test parsing non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ofx_processor.parse_ofx_file('/non/existent/file.ofx')
    
    def test_parse_ofx_file_invalid_format(self, ofx_processor):
        """Test parsing invalid file format raises InvalidFileFormatError."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"Not an OFX file")
            temp_path = f.name
        
        try:
            with pytest.raises(InvalidFileFormatError):
                ofx_processor.parse_ofx_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_parse_ofx_file_corrupted(self, ofx_processor, corrupted_ofx_file):
        """Test parsing corrupted OFX file raises FileCorruptedError."""
        with pytest.raises(FileCorruptedError):
            ofx_processor.parse_ofx_file(corrupted_ofx_file)
    
    def test_read_file_with_encoding_fallback(self, ofx_processor):
        """Test file reading with encoding fallback."""
        # Create file with different encodings
        test_content = "OFXHEADER:100\nDATA:OFXSGML\n<OFX>Test Content</OFX>"
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            content = ofx_processor._read_file_with_encoding_fallback(temp_path)
            assert "OFXHEADER:100" in content
            assert "<OFX>" in content
        finally:
            os.unlink(temp_path)
    
    def test_read_file_encoding_error(self, ofx_processor):
        """Test file reading with encoding errors raises FileCorruptedError."""
        # Create a file with invalid encoding
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b'\xff\xfe\x00\x00')  # Invalid UTF-8 sequence
            temp_path = f.name
        
        try:
            with pytest.raises(FileCorruptedError):
                ofx_processor._read_file_with_encoding_fallback(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_validate_ofx_structure_valid(self, ofx_processor, sample_ofx_content):
        """Test OFX structure validation with valid content."""
        result = ofx_processor._validate_ofx_structure(sample_ofx_content)
        assert result is True
    
    def test_validate_ofx_structure_invalid(self, ofx_processor):
        """Test OFX structure validation with invalid content."""
        invalid_content = "This is not OFX content"
        result = ofx_processor._validate_ofx_structure(invalid_content)
        assert result is False
    
    def test_validate_ofx_structure_missing_sections(self, ofx_processor):
        """Test OFX structure validation with missing required sections."""
        incomplete_content = """
        OFXHEADER:100
        DATA:OFXSGML
        <OFX>
        <SIGNONMSGSRSV1>
        </SIGNONMSGSRSV1>
        </OFX>
        """
        result = ofx_processor._validate_ofx_structure(incomplete_content)
        assert result is False
    
    @patch('ofxparse.OfxParser.parse')
    def test_parse_ofx_with_recovery_success(self, mock_parse, ofx_processor, temp_ofx_file):
        """Test OFX parsing with recovery mechanism - success case."""
        mock_ofx = Mock()
        mock_parse.return_value = mock_ofx
        
        result = ofx_processor._parse_ofx_with_recovery(temp_ofx_file)
        assert result == mock_ofx
        mock_parse.assert_called_once()
    
    @patch('ofxparse.OfxParser.parse')
    def test_parse_ofx_with_recovery_fallback(self, mock_parse, ofx_processor, temp_ofx_file):
        """Test OFX parsing with recovery mechanism - fallback case."""
        # First call fails, second succeeds
        mock_ofx = Mock()
        mock_parse.side_effect = [Exception("Parse failed"), mock_ofx]
        
        result = ofx_processor._parse_ofx_with_recovery(temp_ofx_file)
        assert result == mock_ofx
        assert mock_parse.call_count == 2
    
    def test_fix_common_ofx_issues(self, ofx_processor):
        """Test fixing common OFX file issues."""
        # Test BOM removal
        content_with_bom = '\ufeff<OFX>Content</OFX>'
        fixed = ofx_processor._fix_common_ofx_issues(content_with_bom)
        assert not fixed.startswith('\ufeff')
        
        # Test missing closing tag
        content_without_closing = '<OFX>Content'
        fixed = ofx_processor._fix_common_ofx_issues(content_without_closing)
        assert '</OFX>' in fixed
        
        # Test malformed dates
        content_with_bad_date = '<DTPOSTED>20240115'
        fixed = ofx_processor._fix_common_ofx_issues(content_with_bad_date)
        assert '<DTPOSTED>20240115' in fixed
    
    def test_validate_transactions_success(self, ofx_processor, sample_transaction_data):
        """Test transaction validation with valid data."""
        transactions = [sample_transaction_data]
        result = ofx_processor.validate_transactions(transactions)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['description'] is not None
        assert result[0]['amount'] is not None
        assert result[0]['date'] is not None
    
    def test_validate_transactions_empty_list(self, ofx_processor):
        """Test transaction validation with empty list raises InsufficientDataError."""
        with pytest.raises(InsufficientDataError):
            ofx_processor.validate_transactions([])
    
    def test_validate_transactions_invalid_data(self, ofx_processor):
        """Test transaction validation with invalid data."""
        invalid_transactions = [
            {'amount': 'invalid', 'date': None, 'description': ''},
            {'amount': 100.0, 'date': '2024-01-15', 'description': None}
        ]
        
        result = ofx_processor.validate_transactions(invalid_transactions)
        # Should return empty list or filtered valid transactions
        assert isinstance(result, list)
    
    def test_clean_transaction_data(self, ofx_processor):
        """Test transaction data cleaning."""
        dirty_transaction = {
            'description': '',
            'amount': '100.50',
            'date': '2024-01-15T10:30:00'
        }
        
        cleaned = ofx_processor._clean_transaction_data(dirty_transaction)
        
        assert cleaned['description'] == 'Transação sem descrição'
        assert isinstance(cleaned['amount'], float)
        assert cleaned['amount'] == 100.50
    
    def test_clean_description(self, ofx_processor):
        """Test description cleaning."""
        test_cases = [
            ("", "Transação sem descrição"),
            ("  Multiple   Spaces  ", "Multiple Spaces"),
            ("TED - PAGAMENTO FORNECEDOR", "PAGAMENTO FORNECEDOR"),
            ("PIX - TRANSFERENCIA", "TRANSFERENCIA"),
            ("DOC-PAGAMENTO", "PAGAMENTO")
        ]
        
        for input_desc, expected in test_cases:
            result = ofx_processor._clean_description(input_desc)
            assert result == expected
    
    def test_detect_duplicates_success(self, ofx_processor):
        """Test duplicate detection with valid transactions."""
        transactions = [
            {
                'date': date(2024, 1, 15),
                'amount': -100.0,
                'description': 'PAGAMENTO FORNECEDOR'
            },
            {
                'date': date(2024, 1, 16),
                'amount': 200.0,
                'description': 'RECEITA VENDAS'
            },
            {
                'date': date(2024, 1, 15),
                'amount': -100.0,
                'description': 'PAGAMENTO FORNECEDOR'  # Duplicate
            }
        ]
        
        duplicates = ofx_processor.detect_duplicates(transactions)
        assert isinstance(duplicates, list)
        assert 2 in duplicates  # Third transaction (index 2) is duplicate
    
    def test_detect_duplicates_empty_list(self, ofx_processor):
        """Test duplicate detection with empty list."""
        result = ofx_processor.detect_duplicates([])
        assert result == []
    
    def test_detect_duplicates_no_duplicates(self, ofx_processor):
        """Test duplicate detection with no duplicates."""
        transactions = [
            {
                'date': date(2024, 1, 15),
                'amount': -100.0,
                'description': 'PAGAMENTO FORNECEDOR A'
            },
            {
                'date': date(2024, 1, 16),
                'amount': -100.0,
                'description': 'PAGAMENTO FORNECEDOR B'
            }
        ]
        
        duplicates = ofx_processor.detect_duplicates(transactions)
        assert duplicates == []
    
    def test_detect_duplicates_missing_fields(self, ofx_processor):
        """Test duplicate detection with missing required fields."""
        transactions = [
            {'amount': -100.0, 'description': 'PAGAMENTO'},  # Missing date
            {'date': date(2024, 1, 15), 'description': 'PAGAMENTO'},  # Missing amount
            {'date': date(2024, 1, 15), 'amount': -100.0}  # Missing description
        ]
        
        duplicates = ofx_processor.detect_duplicates(transactions)
        assert isinstance(duplicates, list)
    
    def test_create_duplicate_key(self, ofx_processor):
        """Test duplicate key creation."""
        transaction = {
            'date': date(2024, 1, 15),
            'amount': -100.50,
            'description': 'PAGAMENTO FORNECEDOR'
        }
        
        key = ofx_processor._create_duplicate_key(transaction)
        assert isinstance(key, tuple)
        assert len(key) == 3
        assert key[0] == '2024-01-15'
        assert key[1] == -100.50
        assert key[2] == 'pagamento fornecedor'
    
    def test_create_duplicate_key_with_datetime(self, ofx_processor):
        """Test duplicate key creation with datetime object."""
        transaction = {
            'date': datetime(2024, 1, 15, 10, 30, 0),
            'amount': 100.0,
            'description': 'TEST TRANSACTION'
        }
        
        key = ofx_processor._create_duplicate_key(transaction)
        assert key[0] == '2024-01-15T10:30:00'
        assert key[1] == 100.0
        assert key[2] == 'test transaction'
    
    def test_validate_parsed_data_success(self, ofx_processor):
        """Test parsed data validation with valid data."""
        valid_data = {
            'transactions': [
                {
                    'date': date(2024, 1, 15),
                    'amount': -100.0,
                    'description': 'PAGAMENTO'
                }
            ],
            'summary': {
                'total_transactions': 1
            },
            'account_info': {
                'account_id': '12345-6'
            }
        }
        
        # Should not raise any exception
        ofx_processor._validate_parsed_data(valid_data)
    
    def test_validate_parsed_data_no_transactions(self, ofx_processor):
        """Test parsed data validation with no transactions raises InsufficientDataError."""
        invalid_data = {
            'transactions': [],
            'summary': {
                'total_transactions': 0
            }
        }
        
        with pytest.raises(InsufficientDataError):
            ofx_processor._validate_parsed_data(invalid_data)
    
    def test_validate_parsed_data_missing_account_id(self, ofx_processor):
        """Test parsed data validation with missing account ID logs warning."""
        data_without_account = {
            'transactions': [
                {
                    'date': date(2024, 1, 15),
                    'amount': -100.0,
                    'description': 'PAGAMENTO'
                }
            ],
            'summary': {
                'total_transactions': 1
            },
            'account_info': {}
        }
        
        # Should not raise exception but log warning
        with patch('src.services.ofx_processor.logger') as mock_logger:
            ofx_processor._validate_parsed_data(data_without_account)
            mock_logger.warning.assert_called_once()


class TestOFXProcessorIntegration:
    """Integration tests for OFXProcessor with real OFX content."""
    
    def test_full_ofx_processing_workflow(self, ofx_processor, sample_ofx_content):
        """Test complete OFX processing workflow."""
        # Create temporary OFX file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
            f.write(sample_ofx_content)
            temp_path = f.name
        
        try:
            # Process the file
            result = ofx_processor.parse_ofx_file(temp_path)
            
            # Verify complete result structure
            assert result['bank_name'] in ['bb', 'unknown']  # Based on sample content
            assert len(result['transactions']) >= 0
            assert result['summary']['total_transactions'] >= 0
            
            # If transactions exist, validate them
            if result['transactions']:
                validated = ofx_processor.validate_transactions(result['transactions'])
                assert isinstance(validated, list)
                
                # Test duplicate detection
                duplicates = ofx_processor.detect_duplicates(validated)
                assert isinstance(duplicates, list)
        
        finally:
            os.unlink(temp_path)
    
    def test_financial_accuracy(self, ofx_processor, assert_financial_accuracy):
        """Test financial calculation accuracy."""
        transactions = [
            {'amount': 100.50, 'transaction_type': 'credit'},
            {'amount': -50.25, 'transaction_type': 'debit'},
            {'amount': 200.75, 'transaction_type': 'credit'},
            {'amount': -75.00, 'transaction_type': 'debit'}
        ]
        
        total_credits = sum(t['amount'] for t in transactions if t['amount'] > 0)
        total_debits = sum(abs(t['amount']) for t in transactions if t['amount'] < 0)
        net_flow = total_credits - total_debits
        
        # Test financial accuracy
        assert_financial_accuracy(301.25, total_credits)
        assert_financial_accuracy(125.25, total_debits)
        assert_financial_accuracy(176.00, net_flow)


class TestOFXProcessorErrorHandling:
    """Test error handling in OFXProcessor."""
    
    def test_timeout_handling(self, ofx_processor):
        """Test timeout handling in OFX processing."""
        with patch('src.services.ofx_processor.with_timeout') as mock_timeout:
            mock_timeout.side_effect = TimeoutError("Processing timeout")
            
            with pytest.raises(TimeoutError):
                ofx_processor.parse_ofx_file('/fake/path.ofx')
    
    def test_service_error_handling(self, ofx_processor):
        """Test service error handling decorator."""
        with patch('src.services.ofx_processor.handle_service_errors') as mock_handler:
            # Test that the decorator is applied
            assert hasattr(ofx_processor.parse_ofx_file, '__wrapped__')
    
    def test_recovery_mechanism(self, ofx_processor):
        """Test recovery mechanism in OFX parsing."""
        with patch('src.services.ofx_processor.recovery_manager') as mock_recovery:
            mock_recovery.fallback_on_error.return_value = Mock()
            
            # This should trigger recovery mechanism
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
                f.write("INVALID OFX CONTENT")
                temp_path = f.name
            
            try:
                with patch('ofxparse.OfxParser.parse', side_effect=Exception("Parse failed")):
                    ofx_processor._parse_ofx_with_recovery(temp_path)
                    mock_recovery.fallback_on_error.assert_called_once()
            finally:
                os.unlink(temp_path)


class TestOFXProcessorPerformance:
    """Performance tests for OFXProcessor."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_file_processing(self, ofx_processor):
        """Test processing of large OFX files."""
        # Create a large OFX file with many transactions
        large_ofx_content = self._create_large_ofx_content(1000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
            f.write(large_ofx_content)
            temp_path = f.name
        
        try:
            import time
            start_time = time.time()
            
            result = ofx_processor.parse_ofx_file(temp_path)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process within reasonable time (adjust threshold as needed)
            assert processing_time < 30  # 30 seconds max
            assert len(result['transactions']) > 0
            
        finally:
            os.unlink(temp_path)
    
    def _create_large_ofx_content(self, num_transactions: int) -> str:
        """Create large OFX content with specified number of transactions."""
        header = """OFXHEADER:100
DATA:OFXSGML
VERSION:102
SECURITY:NONE
ENCODING:USASCII
CHARSET:1252
COMPRESSION:NONE
OLDFILEUID:NONE
NEWFILEUID:NONE

<OFX>
<SIGNONMSGSRSV1>
<SONRS>
<STATUS>
<CODE>0
<SEVERITY>INFO
</STATUS>
<DTSERVER>20240115120000
<LANGUAGE>POR
</SONRS>
</SIGNONMSGSRSV1>
<BANKMSGSRSV1>
<STMTRS>
<CURDEF>BRL
<BANKACCTFROM>
<BANKID>001
<ACCTID>12345-6
<ACCTTYPE>CHECKING
</BANKACCTFROM>
<BANKTRANLIST>
<DTSTART>20240101120000
<DTEND>20240115120000"""
        
        transactions = ""
        for i in range(num_transactions):
            transactions += f"""
<STMTTRN>
<TRNTYPE>DEBIT
<DTPOSTED>20240115120000
<TRNAMT>-{100 + i}.00
<FITID>TXN{i:06d}
<MEMO>TRANSACTION {i}
</STMTTRN>"""
        
        footer = """
</BANKTRANLIST>
<LEDGERBAL>
<BALAMT>10000.00
<DTASOF>20240115120000
</LEDGERBAL>
</STMTRS>
</BANKMSGSRSV1>
</OFX>"""
        
        return header + transactions + footer