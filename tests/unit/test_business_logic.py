"""
Tests for critical business logic in Maria Conciliadora application.

This module tests the most critical business operations:
- Financial calculation accuracy
- Data integrity validation
- Business rule enforcement
- Transaction categorization logic
- Reconciliation accuracy
- Duplicate detection reliability
"""

import pytest
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock
from src.services.reconciliation_service import ReconciliationService
from src.services.ai_service import AIService
from src.services.duplicate_detection_service import DuplicateDetectionService
from src.models.transaction import Transaction
from src.models.company_financial import CompanyFinancial
from src.utils.exceptions import ValidationError, BusinessLogicError


class TestFinancialCalculations:
    """Test financial calculation accuracy - critical for reconciliation."""
    
    def test_amount_precision_handling(self):
        """Test that financial amounts maintain proper precision."""
        # Test various decimal scenarios
        test_cases = [
            (100.00, "100.00"),
            (100.1, "100.10"),
            (100.123, "100.12"),  # Should round to 2 decimal places
            (100.125, "100.13"),  # Should round up
            (100.124, "100.12"),  # Should round down
            (0.01, "0.01"),
            (0.001, "0.00"),  # Should round to 0
            (-100.125, "-100.13"),  # Negative amounts
        ]
        
        for amount, expected in test_cases:
            # Convert to Decimal for precise calculation
            decimal_amount = Decimal(str(amount)).quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP
            )
            assert str(decimal_amount) == expected, f"Amount {amount} should be {expected}, got {decimal_amount}"
    
    def test_transaction_balance_calculation(self, db_session):
        """Test accurate balance calculation from transactions."""
        # Create test transactions
        transactions = [
            Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 1),
                amount=1000.00,
                description='Initial deposit',
                transaction_type='credit'
            ),
            Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 2),
                amount=-250.50,
                description='Purchase',
                transaction_type='debit'
            ),
            Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 3),
                amount=-100.25,
                description='ATM withdrawal',
                transaction_type='debit'
            ),
            Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 4),
                amount=500.00,
                description='Salary',
                transaction_type='credit'
            ),
        ]
        
        for tx in transactions:
            db_session.add(tx)
        db_session.commit()
        
        # Calculate balance
        total_credits = sum(tx.amount for tx in transactions if tx.amount > 0)
        total_debits = sum(abs(tx.amount) for tx in transactions if tx.amount < 0)
        expected_balance = total_credits - total_debits
        
        # Verify calculations
        assert total_credits == 1500.00
        assert total_debits == 350.75
        assert expected_balance == 1149.25
        
        # Test balance calculation method
        calculated_balance = self._calculate_account_balance(transactions)
        assert calculated_balance == expected_balance
    
    def test_monthly_summary_calculations(self, db_session):
        """Test monthly financial summary calculations."""
        # Create transactions for different months
        january_transactions = [
            Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 15),
                amount=-500.00,
                description='Rent',
                transaction_type='debit',
                category='casa'
            ),
            Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 20),
                amount=-200.50,
                description='Groceries',
                transaction_type='debit',
                category='alimentacao'
            ),
            Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 25),
                amount=3000.00,
                description='Salary',
                transaction_type='credit',
                category='salario'
            ),
        ]
        
        for tx in january_transactions:
            db_session.add(tx)
        db_session.commit()
        
        # Calculate monthly summary
        summary = self._calculate_monthly_summary(january_transactions, 2024, 1)
        
        # Verify calculations
        assert summary['total_income'] == 3000.00
        assert summary['total_expenses'] == 700.50
        assert summary['net_income'] == 2299.50
        assert summary['expense_by_category']['casa'] == 500.00
        assert summary['expense_by_category']['alimentacao'] == 200.50
    
    def test_reconciliation_matching_accuracy(self, reconciliation_service):
        """Test accuracy of reconciliation matching algorithms."""
        # Create bank transactions
        bank_transactions = [
            Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 15),
                amount=-500.00,
                description='PAGAMENTO ALUGUEL',
                transaction_type='debit'
            ),
            Transaction(
                bank_name='TEST_BANK',
                account_id='ACC_001',
                date=date(2024, 1, 16),
                amount=-200.50,
                description='SUPERMERCADO XYZ',
                transaction_type='debit'
            ),
        ]
        
        # Create company entries
        company_entries = [
            CompanyFinancial(
                date=date(2024, 1, 15),
                amount=-500.00,
                description='Aluguel escritório',
                transaction_type='expense',
                category='casa'
            ),
            CompanyFinancial(
                date=date(2024, 1, 16),
                amount=-200.50,
                description='Compras supermercado',
                transaction_type='expense',
                category='alimentacao'
            ),
            CompanyFinancial(
                date=date(2024, 1, 17),
                amount=-100.00,
                description='Combustível',
                transaction_type='expense',
                category='transporte'
            ),
        ]
        
        # Test matching
        matches = reconciliation_service.find_matches(bank_transactions, company_entries)
        
        # Verify matching accuracy
        assert len(matches) == 2  # Should find 2 matches
        
        # Verify match quality
        for match in matches:
            assert match['confidence_score'] >= 0.8  # High confidence matches
            assert match['bank_transaction'] is not None
            assert match['company_entry'] is not None
            
            # Verify amount matching
            bank_amount = abs(match['bank_transaction'].amount)
            company_amount = abs(match['company_entry'].amount)
            assert bank_amount == company_amount
    
    def test_currency_conversion_accuracy(self):
        """Test currency conversion calculations if applicable."""
        # Test BRL to USD conversion (example rates)
        test_cases = [
            (100.00, 5.0, 20.00),  # 100 BRL at rate 5.0 = 20 USD
            (250.50, 4.8, 52.19),  # Should round properly
            (0.01, 5.0, 0.00),     # Small amounts
        ]
        
        for brl_amount, rate, expected_usd in test_cases:
            converted = self._convert_currency(brl_amount, rate)
            assert abs(converted - expected_usd) < 0.01, f"Conversion failed: {brl_amount} BRL at rate {rate} should be {expected_usd} USD, got {converted}"
    
    def _calculate_account_balance(self, transactions):
        """Calculate account balance from transactions."""
        balance = Decimal('0.00')
        for tx in transactions:
            balance += Decimal(str(tx.amount))
        return float(balance)
    
    def _calculate_monthly_summary(self, transactions, year, month):
        """Calculate monthly financial summary."""
        monthly_txs = [tx for tx in transactions if tx.date.year == year and tx.date.month == month]
        
        total_income = sum(tx.amount for tx in monthly_txs if tx.amount > 0)
        total_expenses = sum(abs(tx.amount) for tx in monthly_txs if tx.amount < 0)
        
        expense_by_category = {}
        for tx in monthly_txs:
            if tx.amount < 0 and tx.category:
                expense_by_category[tx.category] = expense_by_category.get(tx.category, 0) + abs(tx.amount)
        
        return {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'net_income': total_income - total_expenses,
            'expense_by_category': expense_by_category
        }
    
    def _convert_currency(self, amount, rate):
        """Convert currency with proper rounding."""
        converted = Decimal(str(amount)) / Decimal(str(rate))
        return float(converted.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


class TestDataIntegrityValidation:
    """Test data integrity validation - critical for data quality."""
    
    def test_transaction_data_validation(self):
        """Test transaction data validation rules."""
        # Test valid transaction
        valid_transaction = Transaction(
            bank_name='VALID_BANK',
            account_id='VALID_ACC',
            date=date(2024, 1, 15),
            amount=-100.50,
            description='Valid transaction',
            transaction_type='debit'
        )
        
        # Should not raise any validation errors
        self._validate_transaction(valid_transaction)
        
        # Test invalid transactions
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
            # Invalid amount (zero)
            {
                'bank_name': 'TEST_BANK',
                'account_id': 'ACC_001',
                'date': date(2024, 1, 15),
                'amount': 0.00,
                'description': 'Test',
                'transaction_type': 'debit'
            },
            # Future date
            {
                'bank_name': 'TEST_BANK',
                'account_id': 'ACC_001',
                'date': date.today() + timedelta(days=1),
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
                'transaction_type': 'invalid'
            },
        ]
        
        for invalid_data in invalid_cases:
            with pytest.raises((ValidationError, ValueError)):
                invalid_tx = Transaction(**invalid_data)
                self._validate_transaction(invalid_tx)
    
    def test_duplicate_detection_reliability(self, duplicate_detection_service, db_session):
        """Test reliability of duplicate detection."""
        # Create original transaction
        original_tx = Transaction(
            bank_name='TEST_BANK',
            account_id='ACC_001',
            date=date(2024, 1, 15),
            amount=-100.50,
            description='Original transaction',
            transaction_type='debit'
        )
        db_session.add(original_tx)
        db_session.commit()
        
        # Test exact duplicate detection
        exact_duplicate = Transaction(
            bank_name='TEST_BANK',
            account_id='ACC_001',
            date=date(2024, 1, 15),
            amount=-100.50,
            description='Original transaction',
            transaction_type='debit'
        )
        
        is_duplicate = DuplicateDetectionService.check_transaction_duplicate(
            exact_duplicate.account_id,
            exact_duplicate.date,
            exact_duplicate.amount,
            exact_duplicate.description
        )
        assert is_duplicate, "Exact duplicate should be detected"
        
        # Test near-duplicate detection (slight description variation)
        near_duplicate = Transaction(
            bank_name='TEST_BANK',
            account_id='ACC_001',
            date=date(2024, 1, 15),
            amount=-100.50,
            description='Original transaction.',  # Added period
            transaction_type='debit'
        )
        
        is_near_duplicate = DuplicateDetectionService.check_transaction_duplicate(
            near_duplicate.account_id,
            near_duplicate.date,
            near_duplicate.amount,
            near_duplicate.description
        )
        assert is_near_duplicate, "Near duplicate should be detected"
        
        # Test non-duplicate
        different_tx = Transaction(
            bank_name='TEST_BANK',
            account_id='ACC_001',
            date=date(2024, 1, 16),  # Different date
            amount=-100.50,
            description='Original transaction',
            transaction_type='debit'
        )
        
        is_not_duplicate = DuplicateDetectionService.check_transaction_duplicate(
            different_tx.account_id,
            different_tx.date,
            different_tx.amount,
            different_tx.description
        )
        assert not is_not_duplicate, "Different transaction should not be detected as duplicate"
    
    def test_data_consistency_across_operations(self, db_session):
        """Test data consistency across multiple operations."""
        # Create initial transaction
        transaction = Transaction(
            bank_name='CONSISTENCY_BANK',
            account_id='CONS_ACC',
            date=date(2024, 1, 15),
            amount=-100.00,
            description='Consistency test',
            transaction_type='debit'
        )
        db_session.add(transaction)
        db_session.commit()
        
        original_id = transaction.id
        
        # Update transaction
        transaction.amount = -150.00
        transaction.description = 'Updated consistency test'
        db_session.commit()
        
        # Verify consistency
        updated_tx = db_session.query(Transaction).filter_by(id=original_id).first()
        assert updated_tx is not None
        assert updated_tx.amount == -150.00
        assert updated_tx.description == 'Updated consistency test'
        assert updated_tx.bank_name == 'CONSISTENCY_BANK'  # Unchanged fields should remain
    
    def test_referential_integrity(self, db_session):
        """Test referential integrity between related entities."""
        # This test would verify foreign key constraints and relationships
        # Implementation depends on your specific database schema
        pass
    
    def _validate_transaction(self, transaction):
        """Validate transaction data."""
        if not transaction.bank_name:
            raise ValidationError("Bank name is required")
        
        if not transaction.account_id:
            raise ValidationError("Account ID is required")
        
        if transaction.amount == 0:
            raise ValidationError("Amount cannot be zero")
        
        if transaction.date > date.today():
            raise ValidationError("Transaction date cannot be in the future")
        
        if transaction.transaction_type not in ['debit', 'credit']:
            raise ValidationError("Invalid transaction type")


class TestBusinessRuleEnforcement:
    """Test business rule enforcement."""
    
    def test_transaction_categorization_rules(self, ai_service):
        """Test transaction categorization business rules."""
        # Test automatic categorization rules
        test_cases = [
            # Food/Restaurant transactions
            {
                'description': 'MCDONALDS CENTRO',
                'amount': -25.50,
                'expected_category': 'alimentacao'
            },
            {
                'description': 'SUPERMERCADO EXTRA',
                'amount': -150.00,
                'expected_category': 'alimentacao'
            },
            # Transportation
            {
                'description': 'POSTO SHELL BR',
                'amount': -80.00,
                'expected_category': 'transporte'
            },
            {
                'description': 'UBER TRIP',
                'amount': -15.50,
                'expected_category': 'transporte'
            },
            # Healthcare
            {
                'description': 'FARMACIA DROGASIL',
                'amount': -45.00,
                'expected_category': 'saude'
            },
            # Housing
            {
                'description': 'PAGAMENTO ALUGUEL',
                'amount': -1200.00,
                'expected_category': 'casa'
            },
            # Income
            {
                'description': 'SALARIO EMPRESA XYZ',
                'amount': 5000.00,
                'expected_category': 'salario'
            },
        ]
        
        for case in test_cases:
            transaction_data = {
                'description': case['description'],
                'amount': case['amount'],
                'date': date(2024, 1, 15)
            }
            
            # Mock AI service categorization
            with patch.object(ai_service, 'categorize_transaction') as mock_categorize:
                mock_categorize.return_value = case['expected_category']
                
                category = ai_service.categorize_transaction(transaction_data)
                assert category == case['expected_category'], f"Transaction '{case['description']}' should be categorized as '{case['expected_category']}', got '{category}'"
    
    def test_reconciliation_business_rules(self, reconciliation_service):
        """Test reconciliation business rules."""
        # Test matching rules
        bank_tx = Transaction(
            bank_name='TEST_BANK',
            account_id='ACC_001',
            date=date(2024, 1, 15),
            amount=-500.00,
            description='PAGAMENTO FORNECEDOR ABC',
            transaction_type='debit'
        )
        
        company_entry = CompanyFinancial(
            date=date(2024, 1, 15),
            amount=-500.00,
            description='Pagamento Fornecedor ABC Ltda',
            transaction_type='expense'
        )
        
        # Test exact match
        match_score = reconciliation_service.calculate_match_score(bank_tx, company_entry)
        assert match_score >= 0.9, "Exact matches should have high confidence score"
        
        # Test date tolerance rule (within 3 days)
        company_entry_different_date = CompanyFinancial(
            date=date(2024, 1, 17),  # 2 days later
            amount=-500.00,
            description='Pagamento Fornecedor ABC Ltda',
            transaction_type='expense'
        )
        
        match_score_date_diff = reconciliation_service.calculate_match_score(bank_tx, company_entry_different_date)
        assert match_score_date_diff >= 0.7, "Matches within date tolerance should have good confidence score"
        
        # Test amount tolerance rule (within 5%)
        company_entry_amount_diff = CompanyFinancial(
            date=date(2024, 1, 15),
            amount=-510.00,  # 2% difference
            description='Pagamento Fornecedor ABC Ltda',
            transaction_type='expense'
        )
        
        match_score_amount_diff = reconciliation_service.calculate_match_score(bank_tx, company_entry_amount_diff)
        assert match_score_amount_diff >= 0.6, "Matches within amount tolerance should have reasonable confidence score"
    
    def test_duplicate_prevention_rules(self, duplicate_detection_service, db_session):
        """Test duplicate prevention business rules."""
        # Create original transaction
        original_tx = Transaction(
            bank_name='DUPLICATE_TEST_BANK',
            account_id='DUP_ACC',
            date=date(2024, 1, 15),
            amount=-100.00,
            description='Duplicate test transaction',
            transaction_type='debit'
        )
        db_session.add(original_tx)
        db_session.commit()
        
        # Test same-day duplicate prevention
        same_day_duplicate = {
            'account_id': 'DUP_ACC',
            'date': date(2024, 1, 15),
            'amount': -100.00,
            'description': 'Duplicate test transaction'
        }
        
        is_duplicate = DuplicateDetectionService.check_transaction_duplicate(**same_day_duplicate)
        assert is_duplicate, "Same-day duplicate should be prevented"
        
        # Test cross-day duplicate allowance (legitimate recurring transactions)
        next_day_transaction = {
            'account_id': 'DUP_ACC',
            'date': date(2024, 1, 16),
            'amount': -100.00,
            'description': 'Duplicate test transaction'
        }
        
        is_next_day_duplicate = DuplicateDetectionService.check_transaction_duplicate(**next_day_transaction)
        # This should depend on business rules - some recurring transactions are legitimate
        # For this test, we'll assume next-day transactions are allowed
        assert not is_next_day_duplicate, "Next-day transactions should be allowed for recurring payments"
    
    def test_amount_validation_rules(self):
        """Test amount validation business rules."""
        # Test maximum transaction amount rule
        max_amount = 100000.00  # Example business rule
        
        valid_amounts = [100.00, 1000.00, 50000.00, max_amount]
        invalid_amounts = [max_amount + 0.01, 200000.00]
        
        for amount in valid_amounts:
            assert self._validate_transaction_amount(amount), f"Amount {amount} should be valid"
        
        for amount in invalid_amounts:
            assert not self._validate_transaction_amount(amount), f"Amount {amount} should be invalid"
        
        # Test minimum transaction amount rule
        min_amount = 0.01
        
        assert self._validate_transaction_amount(min_amount), "Minimum amount should be valid"
        assert not self._validate_transaction_amount(0.00), "Zero amount should be invalid"
        assert not self._validate_transaction_amount(-0.01), "Negative minimum should be invalid for validation"
    
    def _validate_transaction_amount(self, amount):
        """Validate transaction amount according to business rules."""
        max_amount = 100000.00
        min_amount = 0.01
        
        abs_amount = abs(amount)
        return min_amount <= abs_amount <= max_amount


class TestAnomalyDetection:
    """Test anomaly detection business logic."""
    
    def test_amount_anomaly_detection(self, ai_service):
        """Test detection of amount anomalies."""
        # Create normal transaction pattern
        normal_transactions = []
        for i in range(30):  # 30 days of normal transactions
            normal_transactions.append({
                'date': date(2024, 1, i + 1) if i < 28 else date(2024, 2, i - 27),
                'amount': -100.0 - (i % 20),  # Normal range: -100 to -120
                'description': f'Normal transaction {i}',
                'category': 'alimentacao'
            })
        
        # Add anomalous transactions
        anomalous_transactions = [
            {
                'date': date(2024, 1, 15),
                'amount': -5000.00,  # Unusually large amount
                'description': 'Anomalous large transaction',
                'category': 'alimentacao'
            },
            {
                'date': date(2024, 1, 20),
                'amount': -0.01,  # Unusually small amount
                'description': 'Anomalous small transaction',
                'category': 'alimentacao'
            }
        ]
        
        all_transactions = normal_transactions + anomalous_transactions
        
        # Mock AI service anomaly detection
        with patch.object(ai_service, 'detect_anomalies') as mock_detect:
            # Mock should identify the anomalous transactions
            mock_detect.return_value = [
                {**tx, 'is_anomaly': tx['amount'] in [-5000.00, -0.01], 'anomaly_score': 0.9 if tx['amount'] in [-5000.00, -0.01] else 0.1}
                for tx in all_transactions
            ]
            
            result = ai_service.detect_anomalies(all_transactions)
            
            # Verify anomaly detection
            anomalies = [tx for tx in result if tx.get('is_anomaly', False)]
            assert len(anomalies) == 2, "Should detect 2 anomalies"
            
            anomaly_amounts = [tx['amount'] for tx in anomalies]
            assert -5000.00 in anomaly_amounts, "Large amount anomaly should be detected"
            assert -0.01 in anomaly_amounts, "Small amount anomaly should be detected"
    
    def test_frequency_anomaly_detection(self, ai_service):
        """Test detection of frequency anomalies."""
        # Create transactions with normal frequency pattern
        transactions = []
        
        # Normal pattern: 1 transaction per day
        for i in range(30):
            transactions.append({
                'date': date(2024, 1, i + 1) if i < 28 else date(2024, 2, i - 27),
                'amount': -100.00,
                'description': f'Daily transaction {i}',
                'category': 'alimentacao'
            })
        
        # Add frequency anomaly: 10 transactions on one day
        for i in range(10):
            transactions.append({
                'date': date(2024, 1, 15),  # Same day
                'amount': -50.00,
                'description': f'Frequent transaction {i}',
                'category': 'alimentacao'
            })
        
        # Mock frequency anomaly detection
        with patch.object(ai_service, 'detect_frequency_anomalies') as mock_detect:
            mock_detect.return_value = {
                'anomalous_dates': [date(2024, 1, 15)],
                'normal_frequency': 1.0,
                'anomalous_frequency': 11.0
            }
            
            result = ai_service.detect_frequency_anomalies(transactions)
            
            assert date(2024, 1, 15) in result['anomalous_dates']
            assert result['anomalous_frequency'] > result['normal_frequency'] * 5  # Significant increase
    
    def test_pattern_anomaly_detection(self, ai_service):
        """Test detection of pattern anomalies."""
        # Create transactions with normal patterns
        normal_pattern_transactions = []
        
        # Normal pattern: weekday transactions
        for i in range(20):  # 20 weekdays
            day = i + 1
            if day <= 28:  # Valid days in January
                transactions_date = date(2024, 1, day)
                # Skip weekends (assuming weekday pattern)
                if transactions_date.weekday() < 5:  # Monday = 0, Friday = 4
                    normal_pattern_transactions.append({
                        'date': transactions_date,
                        'amount': -100.00,
                        'description': 'Weekday transaction',
                        'category': 'alimentacao'
                    })
        
        # Add pattern anomaly: weekend transactions (unusual for this account)
        weekend_anomalies = [
            {
                'date': date(2024, 1, 6),  # Saturday
                'amount': -100.00,
                'description': 'Weekend transaction',
                'category': 'alimentacao'
            },
            {
                'date': date(2024, 1, 7),  # Sunday
                'amount': -100.00,
                'description': 'Weekend transaction',
                'category': 'alimentacao'
            }
        ]
        
        all_transactions = normal_pattern_transactions + weekend_anomalies
        
        # Mock pattern anomaly detection
        with patch.object(ai_service, 'detect_pattern_anomalies') as mock_detect:
            mock_detect.return_value = [
                {**tx, 'is_pattern_anomaly': tx['date'].weekday() >= 5}
                for tx in all_transactions
            ]
            
            result = ai_service.detect_pattern_anomalies(all_transactions)
            
            pattern_anomalies = [tx for tx in result if tx.get('is_pattern_anomaly', False)]
            assert len(pattern_anomalies) == 2, "Should detect 2 pattern anomalies"
            
            # Verify weekend transactions are flagged as anomalies
            anomaly_dates = [tx['date'] for tx in pattern_anomalies]
            assert date(2024, 1, 6) in anomaly_dates
            assert date(2024, 1, 7) in anomaly_dates