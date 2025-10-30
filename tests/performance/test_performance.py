"""
Performance tests for Maria Conciliadora application.

Tests cover:
- Large file processing performance
- Bulk database operations
- Memory usage optimization
- API response times
- Concurrent request handling
- Database query performance
"""

import pytest
import time
import tempfile
import os
import pandas as pd
import psutil
import threading
from datetime import date, datetime
from unittest.mock import patch
from src.services.ofx_processor import OFXProcessor
from src.services.xlsx_processor import XLSXProcessor
from src.services.ai_service import AIService
from src.services.reconciliation_service import ReconciliationService
from src.services.duplicate_detection_service import DuplicateDetectionService
from src.models.transaction import Transaction
from src.models.company_financial import CompanyFinancial


class TestFileProcessingPerformance:
    """Test file processing performance."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_ofx_file_processing(self, ofx_processor):
        """Test processing of large OFX files."""
        # Create large OFX content
        large_ofx_content = self._create_large_ofx_content(5000)  # 5000 transactions
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
            f.write(large_ofx_content)
            temp_path = f.name
        
        try:
            # Measure processing time
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            result = ofx_processor.parse_ofx_file(temp_path)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            processing_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            # Performance assertions
            assert processing_time < 60  # Should process within 60 seconds
            assert memory_used < 500  # Should use less than 500MB additional memory
            assert len(result['transactions']) == 5000
            
            # Calculate throughput
            throughput = len(result['transactions']) / processing_time
            assert throughput > 50  # Should process at least 50 transactions per second
            
            print(f"OFX Performance: {processing_time:.2f}s, {memory_used:.2f}MB, {throughput:.2f} tx/s")
        
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_xlsx_file_processing(self, xlsx_processor):
        """Test processing of large XLSX files."""
        # Create large XLSX dataset
        large_data = []
        for i in range(10000):  # 10,000 entries
            large_data.append({
                'data': f'2024-01-{(i % 28) + 1:02d}',
                'description': f'Large dataset entry {i}',
                'valor': -100.0 - (i % 1000),
                'tipo': 'despesa' if i % 2 == 0 else 'receita',
                'categoria': ['alimentacao', 'transporte', 'servicos', 'multa', 'saude'][i % 5]
            })
        
        df = pd.DataFrame(large_data)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_path = f.name
        
        df.to_excel(temp_path, index=False)
        
        try:
            # Measure processing time
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            result = xlsx_processor.parse_xlsx_file(temp_path)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            processing_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            # Performance assertions
            assert processing_time < 120  # Should process within 2 minutes
            assert memory_used < 1000  # Should use less than 1GB additional memory
            assert len(result) > 0
            
            # Calculate throughput
            throughput = len(result) / processing_time
            assert throughput > 50  # Should process at least 50 entries per second
            
            print(f"XLSX Performance: {processing_time:.2f}s, {memory_used:.2f}MB, {throughput:.2f} entries/s")
        
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.performance
    def test_duplicate_detection_performance(self, duplicate_detection_service, db_session):
        """Test duplicate detection performance with large datasets."""
        # Create large number of transactions in database
        transactions = []
        for i in range(1000):
            transaction = Transaction(
                bank_name='PERF_TEST_BANK',
                account_id=f'ACC_{i % 10}',
                date=date(2024, 1, (i % 28) + 1),
                amount=-100.0 - i,
                description=f'Performance test transaction {i}',
                transaction_type='debit'
            )
            transactions.append(transaction)
            db_session.add(transaction)
        
        db_session.commit()
        
        # Test duplicate detection performance
        start_time = time.time()
        
        duplicate_count = 0
        for i in range(1000):
            is_duplicate = DuplicateDetectionService.check_transaction_duplicate(
                f'ACC_{i % 10}',
                date(2024, 1, (i % 28) + 1),
                -100.0 - i,
                f'Performance test transaction {i}'
            )
            if is_duplicate:
                duplicate_count += 1
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processing_time < 10  # Should complete within 10 seconds
        assert duplicate_count == 1000  # All should be duplicates
        
        # Calculate throughput
        throughput = 1000 / processing_time
        assert throughput > 50  # Should check at least 50 duplicates per second
        
        print(f"Duplicate Detection Performance: {processing_time:.2f}s, {throughput:.2f} checks/s")
    
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
<DTEND>20240131120000"""
        
        transactions = ""
        for i in range(num_transactions):
            day = (i % 28) + 1
            transactions += f"""
<STMTTRN>
<TRNTYPE>DEBIT
<DTPOSTED>202401{day:02d}120000
<TRNAMT>-{100 + (i % 1000)}.00
<FITID>PERF{i:08d}
<MEMO>PERFORMANCE TEST TRANSACTION {i}
</STMTTRN>"""
        
        footer = """
</BANKTRANLIST>
<LEDGERBAL>
<BALAMT>50000.00
<DTASOF>20240131120000
</LEDGERBAL>
</STMTRS>
</BANKMSGSRSV1>
</OFX>"""
        
        return header + transactions + footer


class TestDatabasePerformance:
    """Test database operation performance."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_bulk_transaction_insert(self, db_session):
        """Test bulk transaction insertion performance."""
        # Create large number of transactions
        transactions = []
        for i in range(5000):
            transaction = Transaction(
                bank_name='BULK_TEST_BANK',
                account_id=f'BULK_ACC_{i % 100}',
                date=date(2024, 1, (i % 28) + 1),
                amount=-100.0 - (i % 1000),
                description=f'Bulk test transaction {i}',
                transaction_type='debit',
                category=['alimentacao', 'transporte', 'servicos', 'multa', 'saude'][i % 5]
            )
            transactions.append(transaction)
        
        # Measure insertion time
        start_time = time.time()
        
        # Bulk insert
        db_session.bulk_save_objects(transactions)
        db_session.commit()
        
        end_time = time.time()
        insertion_time = end_time - start_time
        
        # Performance assertions
        assert insertion_time < 30  # Should insert within 30 seconds
        
        # Calculate throughput
        throughput = 5000 / insertion_time
        assert throughput > 100  # Should insert at least 100 transactions per second
        
        print(f"Bulk Insert Performance: {insertion_time:.2f}s, {throughput:.2f} tx/s")
    
    @pytest.mark.performance
    def test_complex_query_performance(self, db_session, create_test_transactions):
        """Test complex database query performance."""
        # Create test data
        transactions = create_test_transactions(2000, db_session)
        
        # Test complex query performance
        start_time = time.time()
        
        # Complex query with joins, filters, and aggregations
        result = db_session.query(Transaction).filter(
            Transaction.amount < -50,
            Transaction.date >= date(2024, 1, 1),
            Transaction.category.in_(['alimentacao', 'transporte'])
        ).order_by(Transaction.date.desc()).limit(100).all()
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Performance assertions
        assert query_time < 2  # Should complete within 2 seconds
        assert len(result) <= 100
        
        print(f"Complex Query Performance: {query_time:.2f}s")
    
    @pytest.mark.performance
    def test_reconciliation_performance(self, reconciliation_service, db_session):
        """Test reconciliation performance with large datasets."""
        # Create large datasets
        bank_transactions = []
        company_entries = []
        
        for i in range(1000):
            # Bank transaction
            bank_tx = Transaction(
                bank_name='RECON_TEST_BANK',
                account_id='RECON_ACC',
                date=date(2024, 1, (i % 28) + 1),
                amount=-100.0 - (i % 500),
                description=f'Reconciliation test {i}',
                transaction_type='debit'
            )
            bank_transactions.append(bank_tx)
            db_session.add(bank_tx)
            
            # Company entry (50% match rate)
            if i % 2 == 0:
                company_entry = CompanyFinancial(
                    date=date(2024, 1, (i % 28) + 1),
                    amount=-100.0 - (i % 500),
                    description=f'Reconciliation test {i}',
                    transaction_type='expense'
                )
                company_entries.append(company_entry)
                db_session.add(company_entry)
        
        db_session.commit()
        
        # Test reconciliation performance
        start_time = time.time()
        
        matches = reconciliation_service.find_matches(bank_transactions, company_entries)
        
        end_time = time.time()
        reconciliation_time = end_time - start_time
        
        # Performance assertions
        assert reconciliation_time < 60  # Should complete within 60 seconds
        assert len(matches) > 0
        
        # Calculate throughput
        total_comparisons = len(bank_transactions) * len(company_entries)
        throughput = total_comparisons / reconciliation_time
        
        print(f"Reconciliation Performance: {reconciliation_time:.2f}s, {len(matches)} matches, {throughput:.0f} comparisons/s")


class TestAIPerformance:
    """Test AI service performance."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_batch_categorization_performance(self, ai_service):
        """Test batch transaction categorization performance."""
        # Create large batch of transactions
        transactions = []
        for i in range(2000):
            transactions.append({
                'description': f'Performance test transaction {i} - {"MERCADO" if i % 4 == 0 else "POSTO" if i % 4 == 1 else "FARMACIA" if i % 4 == 2 else "ESCOLA"}',
                'amount': -100.0 - (i % 500),
                'date': date(2024, 1, (i % 28) + 1)
            })
        
        # Test categorization performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        categorized = ai_service.categorize_transactions_batch(transactions)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        processing_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Performance assertions
        assert processing_time < 30  # Should complete within 30 seconds
        assert memory_used < 200  # Should use less than 200MB additional memory
        assert len(categorized) == 2000
        
        # Calculate throughput
        throughput = 2000 / processing_time
        assert throughput > 50  # Should categorize at least 50 transactions per second
        
        print(f"AI Categorization Performance: {processing_time:.2f}s, {memory_used:.2f}MB, {throughput:.2f} tx/s")
    
    @pytest.mark.performance
    def test_anomaly_detection_performance(self, ai_service):
        """Test anomaly detection performance."""
        # Create large dataset with some anomalies
        transactions = []
        for i in range(1000):
            amount = -100.0 - (i % 100)  # Regular pattern
            if i % 100 == 0:  # Add anomalies
                amount = -10000.0  # Anomalous amount
            
            transactions.append({
                'description': f'Performance test transaction {i}',
                'amount': amount,
                'date': date(2024, 1, (i % 28) + 1)
            })
        
        # Test anomaly detection performance
        start_time = time.time()
        
        result = ai_service.detect_anomalies(transactions)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processing_time < 20  # Should complete within 20 seconds
        assert len(result) == 1000
        
        # Verify anomalies were detected
        anomaly_count = sum(1 for t in result if t.get('is_anomaly', False))
        assert anomaly_count > 0
        
        # Calculate throughput
        throughput = 1000 / processing_time
        assert throughput > 30  # Should process at least 30 transactions per second
        
        print(f"Anomaly Detection Performance: {processing_time:.2f}s, {anomaly_count} anomalies, {throughput:.2f} tx/s")


class TestAPIPerformance:
    """Test API endpoint performance."""
    
    @pytest.mark.performance
    def test_transaction_listing_performance(self, client, db_session, create_test_transactions):
        """Test transaction listing API performance."""
        # Create large dataset
        transactions = create_test_transactions(5000, db_session)
        
        # Test API performance
        start_time = time.time()
        
        response = client.get('/api/transactions?per_page=100')
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Performance assertions
        assert response.status_code == 200
        assert response_time < 5  # Should respond within 5 seconds
        
        print(f"Transaction Listing Performance: {response_time:.2f}s")
    
    @pytest.mark.performance
    def test_summary_generation_performance(self, client, db_session, create_test_transactions):
        """Test summary generation API performance."""
        # Create test data
        transactions = create_test_transactions(2000, db_session)
        
        # Test summary performance
        start_time = time.time()
        
        response = client.get('/api/summary')
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Performance assertions
        assert response.status_code == 200
        assert response_time < 10  # Should respond within 10 seconds
        
        print(f"Summary Generation Performance: {response_time:.2f}s")
    
    @pytest.mark.performance
    def test_concurrent_api_requests(self, client, db_session, create_test_transactions):
        """Test concurrent API request handling."""
        # Create test data
        transactions = create_test_transactions(1000, db_session)
        
        results = []
        errors = []
        
        def make_request(endpoint):
            try:
                start_time = time.time()
                response = client.get(endpoint)
                end_time = time.time()
                
                results.append({
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'response_time': end_time - start_time
                })
            except Exception as e:
                errors.append(str(e))
        
        # Create concurrent requests
        threads = []
        endpoints = [
            '/api/transactions',
            '/api/summary',
            '/api/upload-history',
            '/api/company-financial',
            '/api/company-financial/summary'
        ]
        
        # Create multiple threads for each endpoint
        for _ in range(3):  # 3 requests per endpoint
            for endpoint in endpoints:
                thread = threading.Thread(target=make_request, args=(endpoint,))
                threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert len(errors) == 0  # No errors should occur
        assert len(results) == 15  # 3 requests Ã— 5 endpoints
        assert all(r['status_code'] == 200 for r in results)
        assert total_time < 30  # Should complete within 30 seconds
        
        # Check individual response times
        avg_response_time = sum(r['response_time'] for r in results) / len(results)
        assert avg_response_time < 5  # Average response time should be under 5 seconds
        
        print(f"Concurrent API Performance: {total_time:.2f}s total, {avg_response_time:.2f}s average")


class TestMemoryPerformance:
    """Test memory usage and optimization."""
    
    @pytest.mark.performance
    def test_memory_usage_large_dataset(self, db_session):
        """Test memory usage with large datasets."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        transactions = []
        for i in range(10000):
            transaction = Transaction(
                bank_name='MEMORY_TEST_BANK',
                account_id=f'MEM_ACC_{i % 100}',
                date=date(2024, 1, (i % 28) + 1),
                amount=-100.0 - (i % 1000),
                description=f'Memory test transaction {i}',
                transaction_type='debit'
            )
            transactions.append(transaction)
        
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Bulk insert
        db_session.bulk_save_objects(transactions)
        db_session.commit()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Clear references
        del transactions
        
        # Memory assertions
        memory_increase = peak_memory - initial_memory
        assert memory_increase < 1000  # Should use less than 1GB for 10k transactions
        
        print(f"Memory Usage: Initial={initial_memory:.2f}MB, Peak={peak_memory:.2f}MB, Final={final_memory:.2f}MB")
    
    @pytest.mark.performance
    def test_memory_cleanup_after_processing(self, ofx_processor):
        """Test memory cleanup after file processing."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple files
        for i in range(5):
            large_ofx_content = self._create_medium_ofx_content(1000)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
                f.write(large_ofx_content)
                temp_path = f.name
            
            try:
                result = ofx_processor.parse_ofx_file(temp_path)
                assert len(result['transactions']) == 1000
            finally:
                os.unlink(temp_path)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase significantly after processing multiple files
        assert memory_increase < 200  # Should use less than 200MB additional memory
        
        print(f"Memory Cleanup: Initial={initial_memory:.2f}MB, Final={final_memory:.2f}MB, Increase={memory_increase:.2f}MB")
    
    def _create_medium_ofx_content(self, num_transactions: int) -> str:
        """Create medium-sized OFX content for memory testing."""
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
<DTEND>20240131120000"""
        
        transactions = ""
        for i in range(num_transactions):
            day = (i % 28) + 1
            transactions += f"""
<STMTTRN>
<TRNTYPE>DEBIT
<DTPOSTED>202401{day:02d}120000
<TRNAMT>-{100 + (i % 500)}.00
<FITID>MEM{i:06d}
<MEMO>MEMORY TEST TRANSACTION {i}
</STMTTRN>"""
        
        footer = """
</BANKTRANLIST>
<LEDGERBAL>
<BALAMT>25000.00
<DTASOF>20240131120000
</LEDGERBAL>
</STMTRS>
</BANKMSGSRSV1>
</OFX>"""
        
        return header + transactions + footer


class TestScalabilityPerformance:
    """Test application scalability."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_scalability_with_increasing_load(self, client, db_session):
        """Test application performance with increasing data load."""
        results = []
        
        # Test with increasing dataset sizes
        dataset_sizes = [100, 500, 1000, 2000, 5000]
        
        for size in dataset_sizes:
            # Create dataset
            transactions = []
            for i in range(size):
                transaction = Transaction(
                    bank_name='SCALE_TEST_BANK',
                    account_id=f'SCALE_ACC_{i % 10}',
                    date=date(2024, 1, (i % 28) + 1),
                    amount=-100.0 - (i % 100),
                    description=f'Scalability test transaction {i}',
                    transaction_type='debit'
                )
                transactions.append(transaction)
            
            db_session.bulk_save_objects(transactions)
            db_session.commit()
            
            # Test API performance
            start_time = time.time()
            response = client.get('/api/transactions?per_page=100')
            end_time = time.time()
            
            response_time = end_time - start_time
            
            results.append({
                'dataset_size': size,
                'response_time': response_time
            })
            
            assert response.status_code == 200
            assert response_time < 10  # Should respond within 10 seconds
            
            # Clean up for next iteration
            db_session.query(Transaction).filter(
                Transaction.bank_name == 'SCALE_TEST_BANK'
            ).delete()
            db_session.commit()
        
        # Analyze scalability
        print("Scalability Results:")
        for result in results:
            print(f"  Dataset: {result['dataset_size']}, Response Time: {result['response_time']:.2f}s")
        
        # Response time should not increase dramatically with dataset size
        # (due to pagination and proper indexing)
        max_response_time = max(r['response_time'] for r in results)
        min_response_time = min(r['response_time'] for r in results)
        
        # Response time should not increase more than 5x
        assert max_response_time / min_response_time < 5
