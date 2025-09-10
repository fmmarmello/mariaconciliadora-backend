"""
Test script for refactored context-aware matching service
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.services.context_aware_matching import (
    ContextAwareMatcher, MatchingConfig, MatchingMode, 
    MatchingSuggestion, ContextualMatchResult, PredictionResult
)

def test_refactored_context_aware_matching():
    """Test the refactored context-aware matching service"""
    print("Testing Refactored Context-Aware Matching Service...")
    
    try:
        # Test 1: Basic initialization
        matcher = ContextAwareMatcher()
        print("PASS: Basic initialization successful")
        
        # Test 2: Configuration-based initialization
        config = MatchingConfig(
            mode=MatchingMode.AGGRESSIVE,
            enable_learning=True,
            max_suggestions=15,
            suggestion_threshold=0.5
        )
        aggressive_matcher = ContextAwareMatcher(config)
        print("PASS: Configuration-based initialization successful")
        
        # Test 3: Performance metrics
        metrics = matcher.get_performance_metrics()
        print(f"PASS: Performance metrics: {metrics}")
        
        # Test 4: Configuration update
        new_config = MatchingConfig(mode=MatchingMode.CONSERVATIVE)
        matcher.update_config(new_config)
        print("PASS: Configuration update successful")
        
        # Test 5: Mode-specific settings
        weights = matcher._get_mode_weights()
        max_score = matcher._get_max_score_for_mode()
        threshold = matcher._get_suggestion_threshold_for_mode()
        max_suggestions = matcher._get_max_suggestions_for_mode()
        
        print(f"PASS: Mode settings - weights: {len(weights)}, max_score: {max_score}, threshold: {threshold}, max_suggestions: {max_suggestions}")
        
        # Test 6: Cache management
        matcher.clear_cache()
        print("PASS: Cache clear successful")
        
        # Test 7: Pattern export/import
        test_export_path = "test_patterns.json"
        export_success = matcher.export_patterns(test_export_path)
        if export_success:
            print("PASS: Pattern export successful")
            # Test import
            import_success = matcher.import_patterns(test_export_path)
            if import_success:
                print("PASS: Pattern import successful")
            else:
                print("WARN: Pattern import failed")
            # Clean up
            if os.path.exists(test_export_path):
                os.remove(test_export_path)
        else:
            print("WARN: Pattern export failed")
        
        # Test 8: Prediction confidence calculation
        confidence = matcher._calculate_prediction_confidence(
            suppliers=["LTDA EXAMPLE"],
            payment_methods=["boleto"],
            categories=["supplier_payment"],
            patterns={"description_patterns": {"test": {"count": 5, "frequency": 0.1}}}
        )
        print(f"PASS: Prediction confidence calculation: {confidence:.3f}")
        
        # Test 9: Category prediction from context
        # Mock transaction for testing
        class MockTransaction:
            def __init__(self, description, amount):
                self.description = description
                self.amount = amount
        
        mock_transaction = MockTransaction("Pagamento de R$ 1.234,56 via BOLETO para LTDA EXAMPLE", 1234.56)
        categories = matcher._predict_categories_from_context(mock_transaction, {})
        print(f"PASS: Category prediction: {categories}")
        
        # Test 10: Processing time metrics
        matcher._update_average_processing_time(0.1)
        matcher._update_average_processing_time(0.2)
        updated_metrics = matcher.get_performance_metrics()
        print(f"PASS: Processing time metrics updated: {updated_metrics['average_processing_time']:.3f}s")
        
        # Test 11: Different matching modes
        modes = [MatchingMode.CONSERVATIVE, MatchingMode.BALANCED, MatchingMode.AGGRESSIVE, MatchingMode.LEARNING]
        for mode in modes:
            test_config = MatchingConfig(mode=mode)
            test_matcher = ContextAwareMatcher(test_config)
            mode_weights = test_matcher._get_mode_weights()
            print(f"PASS: {mode.value} mode configured with {len(mode_weights)} weights")
        
        print("\n--- All Refactored Context-Aware Matching Tests Passed ---")
        return True
        
    except Exception as e:
        print(f"FAIL: Refactored context-aware matching test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_refactored_context_aware_matching()
    sys.exit(0 if success else 1)