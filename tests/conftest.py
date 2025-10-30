"""
Pytest configuration and fixtures for Maria Conciliadora tests.

This module provides:
- Application fixtures for testing
- Database fixtures with cleanup
- Mock data factories
- Test utilities and helpers
- Common test configurations
"""

import os
import sys
import tempfile
import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Generator
import pandas as pd
from faker import Faker

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set test environment variables
os.environ['FLASK_ENV'] = 'testing'
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['CONSOLE_LOGGING'] = 'false'
os.environ['FILE_LOGGING'] = 'false'
os.environ['AUDIT_LOGGING'] = 'false'

# Import application modules after setting environment
from src.main import app
from src.models.user import db
from src.models.transaction import Transaction, UploadHistory, ReconciliationRecord
from src.models.company_financial import CompanyFinancial
from src.services.ofx_processor import OFXProcessor
from src.services.xlsx_processor import XLSXProcessor
from src.services.ai_service import AIService
from src.services.reconciliation_service import ReconciliationService
from src.services.duplicate_detection_service import DuplicateDetectionService

# Initialize Faker for generating test data
fake = Faker('pt_BR')  # Brazilian Portuguese locale


# Application Fixtures
@pytest.fixture(scope='session')
def flask_app():
    """Create and configure a test Flask application."""
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'SECRET_KEY': 'test-secret-key',
        'WTF_CSRF_ENABLED': False,
    })
    
    with app.app_context():
        yield app


@pytest.fixture(scope='function')
def client(flask_app):
    """Create a test client for the Flask application."""
    return flask_app.test_client()


@pytest.fixture(scope='function')
def app_context(flask_app):
    """Create an application context for tests."""
    with flask_app.app_context():
        yield flask_app


# Database Fixtures
@pytest.fixture(scope='function')
def database(app_context):
    """Create and configure test database."""
    db.create_all()
    yield db
    db.session.remove()
    db.drop_all()


@pytest.fixture(scope='function')
def db_session(database):
    """Create a database session for tests."""
    connection = database.engine.connect()
    transaction = connection.begin()
    
    # Configure session to use the connection
    database.session.configure(bind=connection)
    
    yield database.session
    
    # Rollback and cleanup
    transaction.rollback()
    connection.close()
    database.session.remove()


# Service Fixtures
@pytest.fixture
def ofx_processor():
    """Create an OFXProcessor instance for testing."""
    return OFXProcessor()


@pytest.fixture
def xlsx_processor():
    """Create an XLSXProcessor instance for testing."""
    return XLSXProcessor()


@pytest.fixture
def ai_service():
    """Create an AIService instance for testing."""
    return AIService()


@pytest.fixture
def reconciliation_service():
    """Create a ReconciliationService instance for testing."""
    return ReconciliationService()


@pytest.fixture
def duplicate_detection_service():
    """Create a DuplicateDetectionService instance for testing."""
    return DuplicateDetectionService()


# Mock Data Factories
@pytest.fixture
def sample_transaction_data():
    """Generate sample transaction data for testing."""
    return {
        'transaction_id': fake.uuid4(),
        'date': fake.date_between(start_date='-1y', end_date='today'),
        'amount': round(fake.pyfloat(left_digits=4, right_digits=2, positive=False), 2),
        'description': fake.sentence(nb_words=4),
        'transaction_type': 'debit',
        'balance': round(fake.pyfloat(left_digits=5, right_digits=2), 2)
    }


@pytest.fixture
def sample_company_financial_data():
    """Generate sample company financial data for testing."""
    return {
        'date': fake.date_between(start_date='-1y', end_date='today'),
        'description': fake.sentence(nb_words=4),
        'amount': round(fake.pyfloat(left_digits=4, right_digits=2), 2),
        'category': fake.random_element(['alimentacao', 'transporte', 'servicos', 'multa', 'saude']),
        'cost_center': fake.random_element(['TI', 'RH', 'Vendas', 'Marketing']),
        'department': fake.random_element(['Administrativo', 'Operacional', 'Comercial']),
        'project': fake.random_element(['Projeto A', 'Projeto B', 'Projeto C']),
        'transaction_type': fake.random_element(['expense', 'income'])
    }


@pytest.fixture
def sample_ofx_content():
    """Generate sample OFX file content for testing."""
    return """OFXHEADER:100
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
<DTEND>20240115120000
<STMTTRN>
<TRNTYPE>DEBIT
<DTPOSTED>20240115120000
<TRNAMT>-150.00
<FITID>202401151
<MEMO>PAGAMENTO FORNECEDOR A
</STMTTRN>
<STMTTRN>
<TRNTYPE>CREDIT
<DTPOSTED>20240114120000
<TRNAMT>2500.00
<FITID>202401142
<MEMO>RECEITA VENDAS
</STMTTRN>
</BANKTRANLIST>
<LEDGERBAL>
<BALAMT>2350.00
<DTASOF>20240115120000
</LEDGERBAL>
</STMTRS>
</BANKMSGSRSV1>
</OFX>"""


@pytest.fixture
def sample_xlsx_data():
    """Generate sample XLSX data as pandas DataFrame."""
    data = []
    for _ in range(10):
        data.append({
            'data': fake.date_between(start_date='-1y', end_date='today').strftime('%Y-%m-%d'),
            'description': fake.sentence(nb_words=4),
            'valor': round(fake.pyfloat(left_digits=4, right_digits=2), 2),
            'tipo': fake.random_element(['despesa', 'receita']),
            'categoria': fake.random_element(['alimentacao', 'transporte', 'servicos']),
            'cost_center': fake.random_element(['TI', 'RH', 'Vendas']),
            'department': fake.random_element(['Admin', 'Ops', 'Sales']),
            'project': fake.random_element(['Projeto A', 'Projeto B'])
        })
    return pd.DataFrame(data)


# File Fixtures
@pytest.fixture
def temp_ofx_file(sample_ofx_content):
    """Create a temporary OFX file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
        f.write(sample_ofx_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_xlsx_file(sample_xlsx_data):
    """Create a temporary XLSX file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        temp_path = f.name
    
    # Write DataFrame to Excel file
    sample_xlsx_data.to_excel(temp_path, index=False)
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def corrupted_ofx_file():
    """Create a corrupted OFX file for testing error handling."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
        f.write("INVALID OFX CONTENT WITHOUT PROPER STRUCTURE")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# Database Model Fixtures
@pytest.fixture
def sample_transaction(db_session, sample_transaction_data):
    """Create a sample Transaction record in the database."""
    transaction = Transaction(
        bank_name='TEST_BANK',
        account_id='12345-6',
        **sample_transaction_data
    )
    db_session.add(transaction)
    db_session.commit()
    return transaction


@pytest.fixture
def sample_company_financial(db_session, sample_company_financial_data):
    """Create a sample CompanyFinancial record in the database."""
    entry = CompanyFinancial(**sample_company_financial_data)
    db_session.add(entry)
    db_session.commit()
    return entry


@pytest.fixture
def sample_upload_history(db_session):
    """Create a sample UploadHistory record in the database."""
    upload = UploadHistory(
        filename='test_file.ofx',
        bank_name='TEST_BANK',
        transactions_count=5,
        status='success',
        file_hash='abc123def456',
        duplicate_files_count=0,
        duplicate_entries_count=1,
        total_entries_processed=5
    )
    db_session.add(upload)
    db_session.commit()
    return upload


@pytest.fixture
def sample_reconciliation_record(db_session, sample_transaction, sample_company_financial):
    """Create a sample ReconciliationRecord in the database."""
    record = ReconciliationRecord(
        bank_transaction_id=sample_transaction.id,
        company_entry_id=sample_company_financial.id,
        match_score=0.85,
        status='pending'
    )
    db_session.add(record)
    db_session.commit()
    return record


# Mock Fixtures
@pytest.fixture
def mock_ai_service():
    """Create a mock AI service for testing."""
    mock = Mock(spec=AIService)
    mock.categorize_transaction.return_value = 'alimentacao'
    mock.categorize_transactions_batch.return_value = []
    mock.detect_anomalies.return_value = []
    mock.generate_insights.return_value = {'summary': {}, 'categories': {}}
    mock.generate_ai_insights.return_value = "Test AI insights"
    mock.train_custom_model.return_value = {'success': True, 'accuracy': 0.85}
    mock.predict_financial_trends.return_value = {'success': True, 'data': {}}
    return mock


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    mock = Mock()
    mock.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Test AI response"))]
    )
    return mock


@pytest.fixture
def mock_groq_client():
    """Create a mock Groq client for testing."""
    mock = Mock()
    mock.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Test Groq response"))]
    )
    return mock


# Performance Testing Fixtures
@pytest.fixture
def large_transaction_dataset():
    """Generate a large dataset for performance testing."""
    transactions = []
    for i in range(1000):
        transactions.append({
            'transaction_id': f'TXN_{i:06d}',
            'date': fake.date_between(start_date='-2y', end_date='today'),
            'amount': round(fake.pyfloat(left_digits=4, right_digits=2), 2),
            'description': fake.sentence(nb_words=6),
            'transaction_type': fake.random_element(['credit', 'debit']),
            'category': fake.random_element(['alimentacao', 'transporte', 'servicos', 'multa', 'saude', 'lazer'])
        })
    return transactions


@pytest.fixture
def large_xlsx_dataset():
    """Generate a large XLSX dataset for performance testing."""
    data = []
    for i in range(5000):
        data.append({
            'data': fake.date_between(start_date='-2y', end_date='today').strftime('%Y-%m-%d'),
            'description': fake.sentence(nb_words=5),
            'valor': round(fake.pyfloat(left_digits=4, right_digits=2), 2),
            'tipo': fake.random_element(['despesa', 'receita']),
            'categoria': fake.random_element(['alimentacao', 'transporte', 'servicos', 'saude'])
        })
    return pd.DataFrame(data)


# Test Utilities
@pytest.fixture
def assert_financial_accuracy():
    """Utility fixture for asserting financial calculation accuracy."""
    def _assert_accuracy(expected: float, actual: float, tolerance: float = 0.01):
        """Assert that financial calculations are within acceptable tolerance."""
        assert abs(expected - actual) <= tolerance, f"Expected {expected}, got {actual}, tolerance {tolerance}"
    return _assert_accuracy


@pytest.fixture
def create_test_transactions():
    """Utility fixture for creating multiple test transactions."""
    def _create_transactions(count: int, db_session) -> List[Transaction]:
        transactions = []
        for i in range(count):
            transaction = Transaction(
                bank_name='TEST_BANK',
                account_id=f'ACC_{i:03d}',
                transaction_id=f'TXN_{i:06d}',
                date=fake.date_between(start_date='-1y', end_date='today'),
                amount=round(fake.pyfloat(left_digits=4, right_digits=2), 2),
                description=fake.sentence(nb_words=4),
                transaction_type=fake.random_element(['credit', 'debit']),
                category=fake.random_element(['alimentacao', 'transporte', 'servicos'])
            )
            transactions.append(transaction)
            db_session.add(transaction)
        db_session.commit()
        return transactions
    return _create_transactions


# Error Testing Fixtures
@pytest.fixture
def mock_database_error():
    """Mock database errors for testing error handling."""
    def _mock_error(error_type='connection'):
        if error_type == 'connection':
            return patch('src.models.user.db.session.commit', side_effect=Exception("Database connection failed"))
        elif error_type == 'constraint':
            return patch('src.models.user.db.session.commit', side_effect=Exception("Constraint violation"))
        elif error_type == 'timeout':
            return patch('src.models.user.db.session.commit', side_effect=Exception("Database timeout"))
        else:
            return patch('src.models.user.db.session.commit', side_effect=Exception("Unknown database error"))
    return _mock_error


# Cleanup Fixtures
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    temp_files = []
    
    def register_temp_file(filepath):
        temp_files.append(filepath)
    
    yield register_temp_file
    
    # Cleanup all registered temp files
    for filepath in temp_files:
        if os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except OSError:
                pass  # File might already be deleted


# Pytest Hooks
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )
    config.addinivalue_line(
        "markers", "database: Tests that require database access"
    )
    config.addinivalue_line(
        "markers", "ai: Tests that use AI services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and paths."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Add markers based on test names
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)
        if "database" in item.name or "db" in item.name:
            item.add_marker(pytest.mark.database)
        if "ai" in item.name:
            item.add_marker(pytest.mark.ai)

