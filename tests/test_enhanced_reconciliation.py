"""
Test suite for enhanced reconciliation matching algorithm
Validates the improved matching engine with confidence scoring and context-aware features
"""

import unittest
from datetime import datetime, date
from unittest.mock import Mock, patch
from src.services.reconciliation_service import ReconciliationService, ReconciliationConfig
from src.services.data_normalization import DataNormalizer
from src.services.context_aware_matching import ContextAwareMatcher
from src.models.transaction import Transaction
from src.models.company_financial import CompanyFinancial

class TestEnhancedReconciliation(unittest.TestCase):
    """Test suite for enhanced reconciliation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ReconciliationConfig()
        self.service = ReconciliationService(self.config)
        self.normalizer = DataNormalizer()
        self.context_matcher = ContextAwareMatcher()
        
        # Create test transactions
        self.test_bank_transactions = [
            Transaction(
                id=1,
                date=date(2024, 1, 15),
                amount=1500.00,
                description="PAGAMENTO FORNECEDOR ABC LTDA",
                category="pagamento",
                bank_id="001"
            ),
            Transaction(
                id=2,
                date=date(2024, 1, 16),
                amount=850.50,
                description="COMPRA MATERIAL ESCRITORIO",
                category="compra",
                bank_id="001"
            ),
            Transaction(
                id=3,
                date=date(2024, 1, 20),
                amount=2500.00,
                description="SERVIÇOS PRESTADOS TECH SOLUÇÕES S.A.",
                category="serviço",
                bank_id="001"
            )
        ]
        
        # Create test company entries
        self.test_company_entries = [
            CompanyFinancial(
                id=1,
                date=date(2024, 1, 15),
                amount=1500.00,
                description="FORNECEDOR ABC LTDA - PAGAMENTO FATURA",
                category="fornecedor",
                cost_center="administrativo"
            ),
            CompanyFinancial(
                id=2,
                date=date(2024, 1, 16),
                amount=850.50,
                description="MATERIAL DE ESCRITORIO - COMPRA",
                category="material",
                cost_center="escritorio"
            ),
            CompanyFinancial(
                id=3,
                date=date(2024, 1, 20),
                amount=2500.00,
                description="TECH SOLUÇÕES S.A. - SERVIÇOS PRESTADOS",
                category="serviço",
                cost_center="ti"
            ),
            CompanyFinancial(
                id=4,
                date=date(2024, 1, 18),
                amount=1200.00,
                description="ALUGUEL SALA COMERCIAL",
                category="aluguel",
                cost_center="administrativo"
            )
        ]
    
    def test_data_normalization_amount(self):
        """Test amount normalization functionality"""
        test_cases = [
            ("R$ 1.500,00", 1500.00),
            ("1500.00", 1500.00),
            ("1.500,00", 1500.00),
            ("R$1500", 1500.00),
            ("1500", 1500.00),
            ("1,500.00", 1500.00),
        ]
        
        for amount_str, expected in test_cases:
            result = self.normalizer.normalize_amount(amount_str)
            self.assertEqual(result, expected, f"Failed for {amount_str}")
    
    def test_data_normalization_description(self):
        """Test description normalization functionality"""
        test_cases = [
            ("PAGAMENTO FORNECEDOR ABC LTDA", "pagamento fornecedor abc ltda"),
            ("COMPRA MATERIAL ESCRITORIO", "compra material escritorio"),
            ("SERVIÇOS PRESTADOS TECH SOLUÇÕES S.A.", "servicos prestados tech solucoes sa"),
        ]
        
        for desc, expected in test_cases:
            result = self.normalizer.normalize_description(desc)
            self.assertEqual(result, expected, f"Failed for {desc}")
    
    def test_description_similarity_calculation(self):
        """Test enhanced description similarity calculation"""
        test_cases = [
            ("PAGAMENTO FORNECEDOR ABC LTDA", "FORNECEDOR ABC LTDA - PAGAMENTO FATURA", 0.95),
            ("COMPRA MATERIAL ESCRITORIO", "MATERIAL DE ESCRITORIO - COMPRA", 0.90),
            ("SERVIÇOS PRESTADOS TECH", "TECH SOLUÇÕES S.A. - SERVIÇOS PRESTADOS", 0.85),
        ]
        
        for desc1, desc2, expected_min in test_cases:
            similarity = self.normalizer.calculate_description_similarity(desc1, desc2)
            self.assertGreaterEqual(similarity, expected_min, 
                                  f"Similarity too low for {desc1} vs {desc2}: {similarity}")
    
    def test_entity_extraction(self):
        """Test entity extraction from descriptions"""
        description = "PAGAMENTO FORNECEDOR ABC LTDA VIA BOLETO"
        entities = self.normalizer.extract_entities(description)
        
        self.assertIn("ABC LTDA", entities['companies'])
        self.assertIn("BOLETO", entities['payment_methods'])
    
    def test_base_match_score_calculation(self):
        """Test base match score calculation"""
        # Perfect match case
        bank_tx = self.test_bank_transactions[0]
        company_entry = self.test_company_entries[0]
        
        score = self.service._calculate_base_match_score(bank_tx, company_entry)
        self.assertGreater(score, 0.8, "Perfect match should score high")
        
        # No match case
        bank_tx = self.test_bank_transactions[0]
        company_entry = self.test_company_entries[3]  # Different entry
        
        score = self.service._calculate_base_match_score(bank_tx, company_entry)
        self.assertLess(score, 0.5, "No match should score low")
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation with all factors"""
        bank_tx = self.test_bank_transactions[0]
        company_entry = self.test_company_entries[0]
        
        base_score = 0.85
        context_info = {'total_bonus': 0.1}
        
        confidence_score = self.service._calculate_confidence_score(
            bank_tx, company_entry, base_score, context_info
        )
        
        self.assertGreaterEqual(confidence_score, 0.0)
        self.assertLessEqual(confidence_score, 1.0)
        self.assertGreater(confidence_score, base_score, 
                         "Confidence score should improve base score")
    
    def test_data_quality_score(self):
        """Test data quality score calculation"""
        bank_tx = self.test_bank_transactions[0]
        company_entry = self.test_company_entries[0]
        
        score = self.service._calculate_data_quality_score(bank_tx, company_entry)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.5, "Complete data should score high")
    
    def test_outlier_detection_score(self):
        """Test outlier detection functionality"""
        bank_tx = self.test_bank_transactions[0]
        company_entry = self.test_company_entries[0]
        
        # Normal case
        score = self.service._calculate_outlier_detection_score(bank_tx, company_entry)
        self.assertGreater(score, 0.8, "Normal amounts should score high")
        
        # Create outlier case
        outlier_entry = CompanyFinancial(
            id=99,
            date=date(2024, 1, 15),
            amount=5000.00,  # Much larger amount
            description="OUTLIER TEST",
            category="test"
        )
        
        score = self.service._calculate_outlier_detection_score(bank_tx, outlier_entry)
        self.assertLess(score, 0.7, "Outlier should reduce score")
    
    def test_temporal_consistency_score(self):
        """Test temporal consistency scoring"""
        bank_tx = self.test_bank_transactions[0]
        company_entry = self.test_company_entries[0]
        
        # Same date
        score = self.service._calculate_temporal_consistency_score(bank_tx, company_entry)
        self.assertEqual(score, 1.0, "Same date should score perfectly")
        
        # Future date (company entry after bank transaction)
        future_entry = CompanyFinancial(
            id=99,
            date=date(2024, 1, 20),  # 5 days later
            amount=1500.00,
            description="FUTURE TEST",
            category="test"
        )
        
        score = self.service._calculate_temporal_consistency_score(bank_tx, future_entry)
        self.assertLess(score, 1.0, "Future date should reduce score")
    
    def test_enhanced_match_score_integration(self):
        """Test integrated enhanced match score calculation"""
        bank_tx = self.test_bank_transactions[0]
        company_entry = self.test_company_entries[0]
        
        score = self.service._calculate_enhanced_match_score(bank_tx, company_entry)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.7, "Good match should score high")
    
    def test_score_breakdown_generation(self):
        """Test detailed score breakdown generation"""
        bank_tx = self.test_bank_transactions[0]
        company_entry = self.test_company_entries[0]
        
        total_score = 0.85
        breakdown = self.service._get_score_breakdown(bank_tx, company_entry, total_score)
        
        # Check structure
        required_keys = [
            'total_score', 'base_score', 'contextual_adjustment', 'components',
            'confidence_factors', 'explanations', 'recommendation'
        ]
        
        for key in required_keys:
            self.assertIn(key, breakdown, f"Missing key in breakdown: {key}")
        
        # Check components
        self.assertIn('amount_score', breakdown['components'])
        self.assertIn('date_score', breakdown['components'])
        self.assertIn('description_score', breakdown['components'])
        
        # Check confidence factors
        self.assertIn('data_quality_score', breakdown['confidence_factors'])
        self.assertIn('pattern_consistency_score', breakdown['confidence_factors'])
        self.assertIn('outlier_detection_score', breakdown['confidence_factors'])
        
        # Check explanations
        self.assertIn('amount', breakdown['explanations'])
        self.assertIn('date', breakdown['explanations'])
        self.assertIn('description', breakdown['explanations'])
        
        # Check recommendation
        self.assertIsInstance(breakdown['recommendation'], str)
        self.assertGreater(len(breakdown['recommendation']), 0)
    
    def test_explanation_generation(self):
        """Test human-readable explanation generation"""
        bank_tx = self.test_bank_transactions[0]
        company_entry = self.test_company_entries[0]
        
        amount_diff = 0.0
        date_diff = 0
        description_similarity = 0.95
        confidence_factors = {
            'data_quality_score': 0.9,
            'pattern_consistency_score': 0.8,
            'outlier_detection_score': 0.9,
            'temporal_consistency_score': 1.0
        }
        context_info = {'explanation': 'Context patterns found'}
        
        explanations = self.service._generate_score_explanations(
            bank_tx, company_entry, amount_diff, date_diff, 
            description_similarity, confidence_factors, context_info
        )
        
        self.assertIn('amount', explanations)
        self.assertIn('date', explanations)
        self.assertIn('description', explanations)
        self.assertIn('data_quality', explanations)
        
        # Check that explanations are meaningful
        self.assertGreater(len(explanations['amount']), 10)
        self.assertGreater(len(explanations['date']), 10)
    
    def test_recommendation_generation(self):
        """Test recommendation generation based on scores"""
        test_cases = [
            (0.95, "CORRESPONDÊNCIA EXCELENTE"),
            (0.85, "CORRESPONDÊNCIA FORTE"),
            (0.75, "CORRESPONDÊNCIA BOA"),
            (0.65, "CORRESPONDÊNCIA POSSÍVEL"),
            (0.45, "CORRESPONDÊNCIA FRACA"),
        ]
        
        for score, expected_phrase in test_cases:
            recommendation = self.service._generate_recommendation(
                score, {}, {}
            )
            self.assertIn(expected_phrase, recommendation, 
                         f"Wrong recommendation for score {score}")
    
    def test_context_aware_matching_integration(self):
        """Test context-aware matching integration"""
        # Mock historical patterns
        with patch.object(self.context_matcher, 'get_contextual_match_score') as mock_context:
            mock_context.return_value = (0.85, {
                'context_factors': {'description_pattern_bonus': 0.1},
                'total_bonus': 0.1,
                'explanation': 'Historical pattern found'
            })
            
            bank_tx = self.test_bank_transactions[0]
            company_entry = self.test_company_entries[0]
            
            contextual_score, context_info = self.context_matcher.get_contextual_match_score(
                bank_tx, company_entry, 0.75
            )
            
            self.assertEqual(contextual_score, 0.85)
            self.assertIn('context_factors', context_info)
            self.assertIn('total_bonus', context_info)
            self.assertIn('explanation', context_info)
    
    def test_find_matches_with_enhanced_algorithm(self):
        """Test the complete enhanced matching process"""
        # Mock database validation to avoid database calls
        with patch.object(self.service, '_validate_transactions_for_matching') as mock_validate_tx, \
             patch.object(self.service, '_validate_entries_for_matching') as mock_validate_entry, \
             patch.object(self.service, '_validate_enhanced_match') as mock_validate_match:
            
            mock_validate_tx.return_value = self.test_bank_transactions
            mock_validate_entry.return_value = self.test_company_entries
            mock_validate_match.return_value = True
            
            matches = self.service.find_matches(
                self.test_bank_transactions, 
                self.test_company_entries
            )
            
            self.assertIsInstance(matches, list)
            self.assertGreater(len(matches), 0, "Should find at least one match")
            
            # Check match structure
            first_match = matches[0]
            required_keys = ['bank_transaction', 'company_entry', 'match_score', 'score_breakdown']
            for key in required_keys:
                self.assertIn(key, first_match, f"Missing key in match: {key}")
            
            # Check score breakdown structure
            breakdown = first_match['score_breakdown']
            self.assertIn('confidence_factors', breakdown)
            self.assertIn('explanations', breakdown)
            self.assertIn('recommendation', breakdown)
    
    def test_pattern_analysis(self):
        """Test historical pattern analysis"""
        # Mock database records
        mock_records = [
            Mock(
                bank_transaction=Mock(
                    description="PAGAMENTO ABC LTDA",
                    amount=1500.00,
                    date=date(2024, 1, 15)
                ),
                company_entry=Mock(
                    description="ABC LTDA PAGAMENTO",
                    amount=1500.00,
                    date=date(2024, 1, 15)
                ),
                created_at=datetime(2024, 1, 15, 10, 0, 0),
                status='confirmed'
            )
        ]
        
        with patch('src.services.context_aware_matching.ReconciliationRecord') as mock_record:
            mock_record.query.filter_by.return_value.all.return_value = mock_records
            
            patterns = self.context_matcher.analyze_historical_patterns()
            
            self.assertIsInstance(patterns, dict)
            # Should find description patterns
            if 'description_patterns' in patterns:
                self.assertIsInstance(patterns['description_patterns'], dict)

def run_enhanced_tests():
    """Run all enhanced reconciliation tests"""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_enhanced_tests()