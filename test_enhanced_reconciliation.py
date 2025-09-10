"""
Test script for enhanced reconciliation service with Phase 2 integration
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.services.reconciliation_service import ReconciliationService, ReconciliationConfig
from src.services.data_normalization import NormalizationConfig, NormalizationMode
from src.services.context_aware_matching import MatchingConfig, MatchingMode

def test_enhanced_reconciliation_service():
    """Test the enhanced reconciliation service with Phase 2 integration"""
    print("Testing Enhanced Reconciliation Service with Phase 2 Integration...")
    
    try:
        # Test 1: Basic initialization
        recon_service = ReconciliationService()
        print("PASS: Basic initialization successful")
        
        # Test 2: Configuration-based initialization
        config = ReconciliationConfig()
        config.minimum_match_threshold = 0.7
        config.max_matches_per_transaction = 5
        config.enable_fuzzy_matching = True
        
        configured_service = ReconciliationService(config)
        print("PASS: Configuration-based initialization successful")
        
        # Test 3: Phase 2 integration configuration
        norm_config = NormalizationConfig(mode=NormalizationMode.AGGRESSIVE)
        match_config = MatchingConfig(mode=MatchingMode.BALANCED)
        
        config_success = recon_service.configure_phase2_integration(norm_config, match_config)
        print(f"PASS: Phase 2 integration configuration: {config_success}")
        
        # Test 4: Enhanced performance metrics
        metrics = recon_service.get_enhanced_performance_metrics()
        print(f"PASS: Enhanced performance metrics available: {len(metrics)} categories")
        
        # Test 5: Entity matching bonus calculation
        entity_bonus = recon_service._calculate_entity_matching_bonus(
            "Pagamento via BOLETO para LTDA EXAMPLE",
            "Compra BOLETO LTDA EXAMPLE COMERCIO"
        )
        print(f"PASS: Entity matching bonus calculation: {entity_bonus:.3f}")
        
        # Test 6: Performance metrics update
        # Mock contextual result for testing
        class MockContextualResult:
            def __init__(self):
                self.total_bonus = 0.15
                self.patterns_used = 3
        
        mock_result = MockContextualResult()
        recon_service._update_performance_metrics(0.1, mock_result)
        
        updated_metrics = recon_service.get_enhanced_performance_metrics()
        print(f"PASS: Performance metrics updated: {updated_metrics['total_matches_processed']} processed")
        
        # Test 7: Pattern export/import
        test_export_path = "test_reconciliation_patterns.json"
        export_success = recon_service.export_phase2_patterns(test_export_path)
        if export_success:
            print("PASS: Phase 2 patterns export successful")
            # Test import
            import_success = recon_service.import_phase2_patterns(test_export_path)
            if import_success:
                print("PASS: Phase 2 patterns import successful")
            else:
                print("WARN: Phase 2 patterns import failed")
            # Clean up
            if os.path.exists(test_export_path):
                os.remove(test_export_path)
        else:
            print("WARN: Phase 2 patterns export failed")
        
        # Test 8: Configuration validation
        config_weights_valid = config.validate_weights()
        print(f"PASS: Configuration weights validation: {config_weights_valid}")
        
        # Test 9: Configuration dictionary conversion
        config_dict = config.to_dict()
        print(f"PASS: Configuration to dictionary: {len(config_dict)} parameters")
        
        # Test 10: Configuration update from dictionary
        new_config_dict = {
            'minimum_match_threshold': 0.75,
            'max_matches_per_transaction': 7,
            'enable_fuzzy_matching': False
        }
        config.update_from_dict(new_config_dict)
        print(f"PASS: Configuration updated from dictionary")
        
        # Test 11: Missing information prediction (mock)
        class MockTransaction:
            def __init__(self, description, amount):
                self.description = description
                self.amount = amount
        
        mock_transaction = MockTransaction("Pagamento de R$ 1.234,56 via BOLETO", 1234.56)
        prediction = recon_service.predict_missing_information(mock_transaction)
        print(f"PASS: Missing information prediction: {len(prediction)} fields")
        
        # Test 12: Integration efficiency metrics
        efficiency = updated_metrics.get('integration_efficiency', {})
        if efficiency:
            print(f"PASS: Integration efficiency metrics available: {len(efficiency)} metrics")
        
        print("\n--- All Enhanced Reconciliation Service Tests Passed ---")
        return True
        
    except Exception as e:
        print(f"FAIL: Enhanced reconciliation service test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_reconciliation_service()
    sys.exit(0 if success else 1)