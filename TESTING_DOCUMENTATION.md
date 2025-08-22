# Maria Conciliadora - Testing Documentation

## Overview

This document provides comprehensive information about the automated test suite for the Maria Conciliadora application. The test suite ensures financial accuracy, data integrity, robust error handling, performance under load, security validation, and regression prevention.

## Table of Contents

1. [Test Architecture](#test-architecture)
2. [Test Categories](#test-categories)
3. [Running Tests](#running-tests)
4. [Test Coverage](#test-coverage)
5. [Writing Tests](#writing-tests)
6. [Continuous Integration](#continuous-integration)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Test Architecture

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── coverage_config.py          # Coverage reporting utilities
├── unit/                       # Unit tests
│   ├── __init__.py
│   ├── test_ofx_processor.py
│   ├── test_xlsx_processor.py
│   ├── test_ai_service.py
│   ├── test_reconciliation_service.py
│   ├── test_duplicate_detection_service.py
│   ├── test_business_logic.py
│   └── test_error_handling.py
├── integration/                # Integration tests
│   ├── __init__.py
│   └── test_api_endpoints.py
├── performance/                # Performance tests
│   ├── __init__.py
│   └── test_performance.py
└── reports/                    # Test reports and coverage
    ├── coverage_html/
    ├── coverage.xml
    ├── coverage.json
    └── coverage_report.html
```

### Test Framework

- **pytest**: Primary testing framework
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **pytest-xdist**: Parallel test execution
- **Faker**: Test data generation

## Test Categories

### 1. Unit Tests

**Purpose**: Test individual components in isolation

**Coverage**:
- **OFX Processor** (`test_ofx_processor.py`): 568 lines
  - File parsing and validation
  - Bank identification
  - Transaction extraction
  - Error handling
  
- **XLSX Processor** (`test_xlsx_processor.py`): 598 lines
  - Excel file parsing
  - Column normalization
  - Data validation
  - Duplicate detection
  
- **AI Service** (`test_ai_service.py`): 750 lines
  - Transaction categorization
  - Anomaly detection
  - Model training
  - API integration
  
- **Reconciliation Service** (`test_reconciliation_service.py`): 600+ lines
  - Matching algorithms
  - Scoring mechanisms
  - Report generation
  
- **Duplicate Detection** (`test_duplicate_detection_service.py`): 800+ lines
  - File-level duplicates
  - Entry-level duplicates
  - Hash-based detection

### 2. Integration Tests

**Purpose**: Test component interactions and API endpoints

**Coverage** (`test_api_endpoints.py`): 1000+ lines
- File upload endpoints
- Transaction management
- Reconciliation workflows
- AI service integration
- Security validation
- Performance benchmarks

### 3. Performance Tests

**Purpose**: Ensure application performance under load

**Coverage** (`test_performance.py`): 650 lines
- Large file processing
- Database operations
- Memory usage
- API response times
- Concurrent requests
- Scalability testing

### 4. Business Logic Tests

**Purpose**: Validate critical business operations

**Coverage** (`test_business_logic.py`): 650 lines
- Financial calculations
- Data integrity
- Business rules
- Categorization logic
- Reconciliation accuracy
- Anomaly detection

### 5. Error Handling Tests

**Purpose**: Test error scenarios and edge cases

**Coverage** (`test_error_handling.py`): 650 lines
- File processing errors
- Database failures
- AI service errors
- Input validation
- Recovery mechanisms
- Edge cases

## Running Tests

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Setup**:
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Configure test database
   export DATABASE_URL="sqlite:///test.db"
   export TESTING=true
   ```

### Basic Test Execution

#### Run All Tests
```bash
# Run complete test suite
pytest

# Run with verbose output
pytest -v

# Run with detailed output
pytest -vv
```

#### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Performance tests only
pytest tests/performance/

# Specific test file
pytest tests/unit/test_ofx_processor.py

# Specific test function
pytest tests/unit/test_ofx_processor.py::TestOFXProcessor::test_parse_valid_ofx_file
```

#### Run Tests with Markers
```bash
# Run only fast tests
pytest -m "not slow"

# Run only performance tests
pytest -m performance

# Run only critical tests
pytest -m critical

# Run integration tests
pytest -m integration
```

### Parallel Test Execution

```bash
# Run tests in parallel (4 workers)
pytest -n 4

# Auto-detect number of CPUs
pytest -n auto
```

### Test Coverage

#### Generate Coverage Report
```bash
# Run tests with coverage
pytest --cov=src --cov-report=html --cov-report=term-missing

# Generate all coverage formats
pytest --cov=src \
       --cov-report=html:tests/coverage_html \
       --cov-report=xml:tests/coverage.xml \
       --cov-report=json:tests/coverage.json \
       --cov-report=term-missing
```

#### Coverage Analysis
```bash
# Run comprehensive coverage analysis
python tests/coverage_config.py
```

#### Coverage Thresholds
- **Overall Coverage**: Minimum 85%
- **Critical Modules**: Minimum 95%
- **Business Logic**: Minimum 95%
- **Error Handling**: Minimum 90%

### Advanced Test Options

#### Database Testing
```bash
# Run with test database cleanup
pytest --db-cleanup

# Run with database transactions
pytest --db-transactions
```

#### Debugging Tests
```bash
# Run with Python debugger
pytest --pdb

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Capture output (disable for debugging)
pytest -s
```

#### Test Selection
```bash
# Run tests matching pattern
pytest -k "test_ofx"

# Run failed tests from last run
pytest --lf

# Run failed tests first
pytest --ff
```

## Test Coverage

### Coverage Requirements

| Component | Minimum Coverage | Current Status |
|-----------|------------------|----------------|
| Overall | 85% | ✅ Target |
| OFX Processor | 95% | ✅ Critical |
| XLSX Processor | 95% | ✅ Critical |
| AI Service | 95% | ✅ Critical |
| Reconciliation | 95% | ✅ Critical |
| Duplicate Detection | 95% | ✅ Critical |
| Business Logic | 95% | ✅ Critical |
| Error Handling | 90% | ✅ Important |
| API Endpoints | 90% | ✅ Important |

### Coverage Reports

#### HTML Report
```bash
# Generate and view HTML coverage report
pytest --cov=src --cov-report=html
open tests/coverage_html/index.html
```

#### Terminal Report
```bash
# Show coverage in terminal
pytest --cov=src --cov-report=term-missing
```

#### Coverage Badge
The coverage badge is automatically generated and can be used in README:

```markdown
![Coverage](https://img.shields.io/badge/coverage-XX.X%25-brightgreen)
```

## Writing Tests

### Test Structure

```python
"""
Test module docstring explaining what is being tested.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.services.example_service import ExampleService


class TestExampleService:
    """Test class for ExampleService."""
    
    def test_basic_functionality(self, example_fixture):
        """Test basic functionality with descriptive name."""
        # Arrange
        service = ExampleService()
        test_data = {"key": "value"}
        
        # Act
        result = service.process(test_data)
        
        # Assert
        assert result is not None
        assert result["status"] == "success"
    
    @pytest.mark.parametrize("input_data,expected", [
        ({"amount": 100}, "valid"),
        ({"amount": 0}, "invalid"),
        ({"amount": -100}, "valid"),
    ])
    def test_validation_scenarios(self, input_data, expected):
        """Test multiple validation scenarios."""
        service = ExampleService()
        result = service.validate(input_data)
        assert result == expected
    
    def test_error_handling(self):
        """Test error handling scenarios."""
        service = ExampleService()
        
        with pytest.raises(ValueError) as exc_info:
            service.process(None)
        
        assert "invalid input" in str(exc_info.value)
    
    @patch('src.services.example_service.external_api')
    def test_external_dependency(self, mock_api):
        """Test with mocked external dependencies."""
        mock_api.return_value = {"status": "success"}
        
        service = ExampleService()
        result = service.call_external_api()
        
        assert result["status"] == "success"
        mock_api.assert_called_once()
```

### Test Fixtures

Common fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def db_session():
    """Database session fixture."""
    # Setup and teardown logic
    pass

@pytest.fixture
def sample_transaction():
    """Sample transaction fixture."""
    return Transaction(
        bank_name='TEST_BANK',
        account_id='ACC_001',
        date=date(2024, 1, 15),
        amount=-100.00,
        description='Test transaction',
        transaction_type='debit'
    )
```

### Mocking Guidelines

```python
# Mock external services
@patch('src.services.ai_service.openai_client')
def test_ai_categorization(mock_openai):
    mock_openai.return_value = {"category": "alimentacao"}
    # Test logic

# Mock database operations
@patch('src.models.transaction.db.session')
def test_database_operation(mock_session):
    mock_session.add.return_value = None
    # Test logic

# Mock file operations
@patch('builtins.open', mock_open(read_data="file content"))
def test_file_processing():
    # Test logic
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Errors
```bash
# Solution: Reset test database
rm test.db
pytest --db-setup
```

#### 2. Import Errors
```bash
# Solution: Install in development mode
pip install -e .
```

#### 3. Fixture Not Found
```bash
# Solution: Check conftest.py location
# Ensure conftest.py is in the correct directory
```

#### 4. Slow Tests
```bash
# Solution: Run only fast tests
pytest -m "not slow"

# Or run in parallel
pytest -n auto
```

#### 5. Memory Issues
```bash
# Solution: Run tests with memory profiling
pytest --memprof

# Or run smaller test subsets
pytest tests/unit/test_specific.py
```

### Debug Mode

```bash
# Run single test with debugging
pytest -s -vv tests/unit/test_example.py::test_specific_function

# Drop into debugger on failure
pytest --pdb

# Show full traceback
pytest --tb=long
```

## Best Practices

### 1. Test Naming
- Use descriptive test names: `test_should_categorize_grocery_transaction_as_food`
- Follow pattern: `test_[condition]_[expected_result]`
- Group related tests in classes

### 2. Test Organization
- One test file per source file
- Group tests by functionality
- Use clear class and method names

### 3. Test Data
- Use factories for complex objects
- Prefer fixtures over hardcoded data
- Keep test data minimal and focused

### 4. Assertions
- Use specific assertions: `assert result.status == 'success'`
- Include meaningful error messages
- Test both positive and negative cases

### 5. Mocking
- Mock external dependencies
- Don't mock the system under test
- Verify mock interactions when relevant

### 6. Performance
- Mark slow tests with `@pytest.mark.slow`
- Use parametrized tests for multiple scenarios
- Run tests in parallel when possible

### 7. Maintenance
- Keep tests simple and readable
- Update tests when code changes
- Remove obsolete tests promptly

## Test Execution Examples

### Development Workflow
```bash
# Quick test run during development
pytest tests/unit/test_current_feature.py -v

# Run tests related to changes
pytest -k "test_ofx or test_xlsx" --tb=short

# Full test suite before commit
pytest --cov=src --cov-report=term-missing
```

### CI/CD Pipeline
```bash
# Complete test suite with coverage
pytest --cov=src \
       --cov-report=html \
       --cov-report=xml \
       --cov-report=json \
       --junitxml=tests/junit.xml \
       -n auto

# Generate coverage reports
python tests/coverage_config.py
```

### Performance Testing
```bash
# Run performance tests only
pytest tests/performance/ -v

# Run with performance profiling
pytest tests/performance/ --profile

# Memory usage testing
pytest tests/performance/ --memprof
```

## Conclusion

This comprehensive test suite ensures the Maria Conciliadora application maintains high quality, reliability, and performance. The tests cover all critical functionality including financial calculations, data integrity, error handling, and performance requirements.

For questions or issues with the test suite, please refer to this documentation or contact the development team.

---

**Last Updated**: January 2024  
**Version**: 1.0  
**Maintainer**: Development Team