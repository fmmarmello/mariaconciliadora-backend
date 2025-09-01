import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime

from src.services.financial_category_balancer import FinancialCategoryBalancer
from src.utils.exceptions import ValidationError


class TestFinancialCategoryBalancer:
    """Test cases for FinancialCategoryBalancer class"""

    @pytest.fixture
    def balancer(self):
        """Create financial balancer instance"""
        config = {
            'smote_config': {'random_state': 42},
            'synthetic_config': {'methods': ['vae'], 'sample_size_ratio': 0.3}
        }
        return FinancialCategoryBalancer(config)

    @pytest.fixture
    def sample_financial_data(self):
        """Create sample financial transaction data"""
        np.random.seed(42)

        n_samples = 1000

        # Create imbalanced financial data
        data = []

        # Majority category: despesa (60%)
        for i in range(600):
            data.append({
                'amount': np.random.uniform(10, 500),
                'description': f'Compra {i}',
                'category': 'despesa',
                'date': (datetime.now() - pd.Timedelta(days=np.random.randint(1, 365))).isoformat()
            })

        # Medium category: receita (30%)
        for i in range(300):
            data.append({
                'amount': np.random.uniform(1000, 5000),
                'description': f'Salario {i}',
                'category': 'receita',
                'date': (datetime.now() - pd.Timedelta(days=np.random.randint(1, 365))).isoformat()
            })

        # Minority category: investimento (10%)
        for i in range(100):
            data.append({
                'amount': np.random.uniform(100, 10000),
                'description': f'Investimento {i}',
                'category': 'investimento',
                'date': (datetime.now() - pd.Timedelta(days=np.random.randint(1, 365))).isoformat()
            })

        return data

    def test_initialization(self, balancer):
        """Test balancer initialization"""
        assert balancer.config is not None
        assert balancer.smote_handler is not None
        assert balancer.synthetic_generator is not None
        assert 'category_patterns' in dir(balancer)
        assert 'business_rules' in dir(balancer)

    def test_balance_financial_categories_auto(self, balancer, sample_financial_data):
        """Test automatic financial category balancing"""
        balanced_data = balancer.balance_financial_categories(
            sample_financial_data, method='auto'
        )

        assert len(balanced_data) >= len(sample_financial_data)

        # Check that synthetic transactions were added
        synthetic_count = sum(1 for item in balanced_data if item.get('is_synthetic', False))
        assert synthetic_count > 0

    def test_balance_specific_category(self, balancer, sample_financial_data):
        """Test balancing specific category"""
        # Balance only 'investimento' category
        balanced_data = balancer.balance_financial_categories(
            sample_financial_data, target_category='investimento', method='pattern_based'
        )

        assert len(balanced_data) >= len(sample_financial_data)

        # Check that investimento transactions increased
        original_investimento = sum(1 for item in sample_financial_data
                                   if item['category'] == 'investimento')
        balanced_investimento = sum(1 for item in balanced_data
                                   if item['category'] == 'investimento')

        assert balanced_investimento > original_investimento

    def test_analyze_category_distribution(self, balancer, sample_financial_data):
        """Test category distribution analysis"""
        df = pd.DataFrame(sample_financial_data)
        distribution = balancer._analyze_category_distribution(df)

        assert 'distribution' in distribution
        assert 'minority_categories' in distribution
        assert 'majority_category' in distribution
        assert 'total_transactions' in distribution
        assert 'imbalance_ratio' in distribution

        # Check that investimento is identified as minority
        assert 'investimento' in distribution['minority_categories']
        assert distribution['majority_category'] == 'despesa'

    def test_generate_pattern_based_synthetic(self, balancer, sample_financial_data):
        """Test pattern-based synthetic generation"""
        df = pd.DataFrame(sample_financial_data)
        investimento_data = df[df['category'] == 'investimento']

        synthetic_data = balancer._generate_pattern_based_synthetic(
            investimento_data, 'investimento', balancer.category_patterns.get('investimento', {})
        )

        assert isinstance(synthetic_data, list)

        if len(synthetic_data) > 0:
            # Check synthetic transaction structure
            transaction = synthetic_data[0]
            assert 'amount' in transaction
            assert 'description' in transaction
            assert 'category' in transaction
            assert 'date' in transaction
            assert transaction.get('is_synthetic', False) == True

    def test_generate_realistic_amount(self, balancer):
        """Test realistic amount generation"""
        category_pattern = balancer.category_patterns['receita']

        amount = balancer._generate_realistic_amount(category_pattern)
        assert amount > 0

        # Check amount is within expected range
        min_amount, max_amount = category_pattern['typical_amount_range']
        assert min_amount <= amount <= max_amount

    def test_generate_realistic_description(self, balancer):
        """Test realistic description generation"""
        description = balancer._generate_realistic_description('receita')
        assert isinstance(description, str)
        assert len(description) > 0

    def test_generate_realistic_date(self, balancer):
        """Test realistic date generation"""
        date = balancer._generate_realistic_date()
        assert isinstance(date, datetime)

        # Check date is in reasonable range (past year)
        now = datetime.now()
        one_year_ago = now.replace(year=now.year - 1)

        assert one_year_ago <= date <= now

    def test_validate_synthetic_transactions(self, balancer):
        """Test synthetic transaction validation"""
        # Create valid synthetic transaction
        valid_transaction = {
            'amount': 100.50,
            'description': 'Valid transaction description',
            'category': 'receita',
            'date': datetime.now().isoformat(),
            'is_synthetic': True
        }

        # Create invalid synthetic transaction
        invalid_transaction = {
            'amount': -100.50,  # Negative amount
            'description': 'X',  # Too short
            'category': 'receita',
            'date': (datetime.now().replace(year=datetime.now().year + 1)).isoformat(),  # Future date
            'is_synthetic': True
        }

        synthetic_data = [valid_transaction, invalid_transaction]
        validated_data = balancer._validate_synthetic_transactions(synthetic_data, 'receita')

        # Should only contain valid transaction
        assert len(validated_data) == 1
        assert validated_data[0]['amount'] == 100.50

    def test_business_rules_validation(self, balancer):
        """Test business rules validation"""
        # Valid transaction
        valid_transaction = {
            'amount': 100.50,
            'description': 'Valid transaction',
            'category': 'receita',
            'date': datetime.now().isoformat()
        }

        assert balancer._validate_single_transaction(valid_transaction, balancer.business_rules) == True

        # Invalid amount
        invalid_amount = valid_transaction.copy()
        invalid_amount['amount'] = -100
        assert balancer._validate_single_transaction(invalid_amount, balancer.business_rules) == False

        # Invalid description length
        invalid_desc = valid_transaction.copy()
        invalid_desc['description'] = 'X'
        assert balancer._validate_single_transaction(invalid_desc, balancer.business_rules) == False

    def test_get_category_statistics(self, balancer):
        """Test category statistics retrieval"""
        stats = balancer.get_category_statistics()

        assert 'category_patterns' in stats
        assert 'business_rules' in stats
        assert 'category_stats' in stats

        # Check that all expected categories are present
        expected_categories = ['receita', 'despesa', 'transferencia', 'investimento', 'outros']
        for category in expected_categories:
            assert category in stats['category_patterns']

    def test_update_category_patterns(self, balancer):
        """Test category patterns update"""
        new_patterns = {
            'nova_categoria': {
                'typical_amount_range': (50, 200),
                'common_descriptions': ['nova transacao'],
                'temporal_patterns': ['monthly']
            }
        }

        balancer.update_category_patterns(new_patterns)

        assert 'nova_categoria' in balancer.category_patterns

    def test_empty_data_handling(self, balancer):
        """Test handling of empty data"""
        empty_data = []
        result = balancer.balance_financial_categories(empty_data)

        # Should handle gracefully
        assert isinstance(result, list)
        assert len(result) == 0

    def test_invalid_category_handling(self, balancer, sample_financial_data):
        """Test handling of invalid category"""
        # Try to balance non-existent category
        result = balancer.balance_financial_categories(
            sample_financial_data, target_category='non_existent_category'
        )

        # Should return original data
        assert len(result) == len(sample_financial_data)

    def test_smote_based_generation(self, balancer, sample_financial_data):
        """Test SMOTE-based synthetic generation"""
        df = pd.DataFrame(sample_financial_data)
        investimento_data = df[df['category'] == 'investimento']

        if len(investimento_data) >= 6:  # SMOTE needs at least 6 samples
            synthetic_data = balancer._generate_smote_based_synthetic(
                investimento_data, 'investimento'
            )

            assert isinstance(synthetic_data, list)
        else:
            # Should fallback to pattern-based
            synthetic_data = balancer._generate_pattern_based_synthetic(
                investimento_data, 'investimento', balancer.category_patterns.get('investimento', {})
            )
            assert isinstance(synthetic_data, list)

    def test_extract_description_patterns(self, balancer, sample_financial_data):
        """Test description pattern extraction"""
        df = pd.DataFrame(sample_financial_data)
        investimento_data = df[df['category'] == 'investimento']

        patterns = balancer._extract_description_patterns(investimento_data)

        assert isinstance(patterns, list)
        if len(investimento_data) > 0:
            assert len(patterns) > 0

    def test_save_load_balancer_state(self, balancer, tmp_path):
        """Test saving and loading balancer state"""
        # Save state
        filepath = tmp_path / "balancer_state.pkl"
        balancer.save_balancer_state(str(filepath))

        # Create new balancer and load state
        new_balancer = FinancialCategoryBalancer(balancer.config)
        new_balancer.load_balancer_state(str(filepath))

        # Check that patterns were loaded
        assert new_balancer.category_patterns == balancer.category_patterns
        assert new_balancer.business_rules == balancer.business_rules

    def test_method_selection(self, balancer):
        """Test automatic method selection"""
        # Small dataset
        small_data = pd.DataFrame([{'amount': 100, 'description': 'test', 'category': 'receita'}] * 10)
        method = balancer._select_financial_method(small_data, 'receita')
        assert method == 'pattern_based'

        # Medium dataset
        medium_data = pd.DataFrame([{'amount': 100, 'description': 'test', 'category': 'receita'}] * 200)
        method = balancer._select_financial_method(medium_data, 'receita')
        assert method in ['smote', 'synthetic']

    def test_generate_conditional_synthetic(self, balancer, sample_financial_data):
        """Test conditional synthetic generation"""
        df = pd.DataFrame(sample_financial_data)
        receita_data = df[df['category'] == 'receita']

        if len(receita_data) > 0:
            synthetic_data = balancer._generate_category_specific_synthetic(
                receita_data, 'receita', 'pattern_based'
            )

            assert isinstance(synthetic_data, list)

            if len(synthetic_data) > 0:
                # Check that synthetic data has correct category
                for transaction in synthetic_data:
                    assert transaction['category'] == 'receita'

    def test_temporal_pattern_validation(self, balancer):
        """Test temporal pattern validation"""
        # Valid date
        valid_transaction = {
            'amount': 100,
            'description': 'Valid transaction',
            'category': 'receita',
            'date': datetime.now().isoformat()
        }

        assert balancer._validate_single_transaction(valid_transaction, balancer.business_rules) == True

        # Future date (invalid)
        future_date = datetime.now().replace(year=datetime.now().year + 1)
        invalid_transaction = valid_transaction.copy()
        invalid_transaction['date'] = future_date.isoformat()

        assert balancer._validate_single_transaction(invalid_transaction, balancer.business_rules) == False

    def test_amount_range_validation(self, balancer):
        """Test amount range validation"""
        rules = balancer.business_rules

        # Valid amounts
        valid_amounts = [0.01, 100, 5000, 99999]
        for amount in valid_amounts:
            transaction = {
                'amount': amount,
                'description': 'Valid transaction',
                'category': 'receita',
                'date': datetime.now().isoformat()
            }
            assert balancer._validate_single_transaction(transaction, rules) == True

        # Invalid amounts
        invalid_amounts = [-100, 0, 1000000]
        for amount in invalid_amounts:
            transaction = {
                'amount': amount,
                'description': 'Invalid transaction',
                'category': 'receita',
                'date': datetime.now().isoformat()
            }
            assert balancer._validate_single_transaction(transaction, rules) == False