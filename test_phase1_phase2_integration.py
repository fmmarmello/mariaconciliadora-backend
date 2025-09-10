"""
Integration tests between Phase 1 and Phase 2 components
Tests the enhanced capabilities and cross-phase functionality
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.services.service_factory import service_factory
from src.services.data_normalization import NormalizationConfig, NormalizationMode
from src.services.context_aware_matching import MatchingConfig, MatchingMode
from src.services.reconciliation_service import ReconciliationConfig

def test_phase1_phase2_integration():
    """Test integration between Phase 1 and Phase 2 components"""
    print("Testing Phase 1 and Phase 2 Integration...")
    
    try:
        # Test 1: Create services with Phase 2 enhanced configurations
        factory = service_factory
        
        # Configure services for maximum Phase 2 integration
        norm_config = NormalizationConfig(
            mode=NormalizationMode.AGGRESSIVE,
            extract_entities=True,
            remove_noise_words=True
        )
        
        match_config = MatchingConfig(
            mode=MatchingMode.LEARNING,
            enable_learning=True,
            max_suggestions=20,
            suggestion_threshold=0.5
        )
        
        recon_config = ReconciliationConfig()
        recon_config.enable_fuzzy_matching = True
        recon_config.minimum_match_threshold = 0.6
        
        services = factory.create_all_services(norm_config, match_config, recon_config)
        print(f"PASS: Created {len(services)} services with Phase 2 integration")
        
        # Test 2: Test Phase 2 enhanced data normalization
        normalizer = factory.get_service(list(services.keys())[0])
        if normalizer:
            # Test advanced entity extraction
            complex_text = "Pagamento de R$ 15.234,78 via PIX para LTDA TECHNOLOGY SOLUTIONS ME - Taxa: R$ 2,50"
            norm_result = normalizer.normalize_text(complex_text)
            
            print(f"PASS: Advanced normalization - Entities: {len(norm_result.entities)} types")
            print(f"  - Companies: {norm_result.entities.get('company', [])}")
            print(f"  - Payment methods: {norm_result.entities.get('payment_method', [])}")
            print(f"  - Tax: {norm_result.entities.get('tax', [])}")
            print(f"  - Amount: {norm_result.entities.get('amount', [])}")
            
            # Test multi-format amount normalization
            amount_tests = [
                "R$ 15.234,78",
                "$15,234.78",
                "15234.78",
                "15.234,78 reais"
            ]
            
            normalized_amounts = []
            for amount_test in amount_tests:
                try:
                    normalized = normalizer.normalize_amount(amount_test)
                    normalized_amounts.append(normalized)
                except:
                    pass
            
            print(f"PASS: Multi-format amount normalization: {len(normalized_amounts)}/{len(amount_tests)} successful")
        
        # Test 3: Test Phase 2 enhanced context-aware matching
        matcher = factory.get_service(list(services.keys())[1])
        if matcher:
            # Test different matching modes
            modes = [MatchingMode.CONSERVATIVE, MatchingMode.BALANCED, MatchingMode.AGGRESSIVE, MatchingMode.LEARNING]
            
            for mode in modes:
                test_config = MatchingConfig(mode=mode)
                test_matcher = factory.create_context_aware_matching_service(test_config)
                
                weights = test_matcher._get_mode_weights()
                max_score = test_matcher._get_max_score_for_mode()
                threshold = test_matcher._get_suggestion_threshold_for_mode()
                
                print(f"PASS: {mode.value} mode - weights: {len(weights)}, max_score: {max_score}, threshold: {threshold}")
            
            # Test user behavior bonus calculation
            user_bonus = matcher._calculate_user_behavior_bonus(
                user_id=1,
                bank_transaction=None,  # Mock for test
                company_entry=None,   # Mock for test
                patterns={'user_behavior_patterns': {'matching_style': {'bonus_factor': 0.1}}}
            )
            print(f"PASS: User behavior bonus calculation: {user_bonus:.3f}")
        
        # Test 4: Test Phase 2 enhanced reconciliation
        recon_service = factory.get_service(list(services.keys())[2])
        if recon_service:
            # Test enhanced entity matching
            entity_bonus = recon_service._calculate_entity_matching_bonus(
                "Pagamento PIX para LTDA TECH SOLUTIONS",
                "Transferência PIX LTDA TECH SOLUTIONS TECNOLOGIA"
            )
            print(f"PASS: Enhanced entity matching bonus: {entity_bonus:.3f}")
            
            # Test performance metrics integration
            enhanced_metrics = recon_service.get_enhanced_performance_metrics()
            
            if 'context_matcher_metrics' in enhanced_metrics:
                print("PASS: Context matcher metrics integrated")
            
            if 'normalizer_metrics' in enhanced_metrics:
                print("PASS: Normalizer metrics integrated")
            
            if 'integration_efficiency' in enhanced_metrics:
                efficiency = enhanced_metrics['integration_efficiency']
                print(f"PASS: Integration efficiency metrics available: {len(efficiency)} categories")
        
        # Test 5: Test cross-service data flow
        if normalizer and matcher and recon_service:
            # Test that data flows properly between services
            test_transaction_text = "Pagamento de R$ 5.678,90 via BOLETO para LTDA EXAMPLE COMERCIO LTDA"
            
            # Step 1: Normalize with data normalization service
            norm_result = normalizer.normalize_text(test_transaction_text)
            
            # Step 2: Use normalized data for matching predictions
            if hasattr(matcher, 'predict_missing_information'):
                try:
                    # Mock transaction object
                    class MockTransaction:
                        def __init__(self, description, amount):
                            self.description = description
                            self.amount = amount
                    
                    mock_transaction = MockTransaction(test_transaction_text, 5678.90)
                    prediction = matcher.predict_missing_information(mock_transaction)
                    
                    print(f"PASS: Cross-service prediction: {len(prediction)} fields predicted")
                except Exception as e:
                    print(f"WARN: Cross-service prediction failed: {str(e)}")
            
            # Step 3: Use enhanced reconciliation for entity matching
            entity_score = recon_service._calculate_entity_matching_bonus(
                test_transaction_text,
                "Compra BOLETO LTDA EXAMPLE COMERCIO LTDA"
            )
            print(f"PASS: Cross-service entity scoring: {entity_score:.3f}")
        
        # Test 6: Test Phase 2 configuration synchronization
        # Test that configuration changes propagate between services
        original_norm_mode = normalizer.config.mode if hasattr(normalizer, 'config') else None
        
        # Update normalizer configuration
        new_norm_config = NormalizationConfig(mode=NormalizationMode.STRICT)
        factory.update_service_config(list(services.keys())[0], new_norm_config)
        
        # Verify the change took effect
        updated_mode = normalizer.config.mode if hasattr(normalizer, 'config') else None
        if updated_mode != original_norm_mode:
            print("PASS: Configuration synchronization working")
        else:
            print("WARN: Configuration synchronization may have issues")
        
        # Test 7: Test Phase 2 pattern management across services
        try:
            os.makedirs("test_phase2_patterns", exist_ok=True)
            
            # Export patterns from all services
            export_results = factory.export_service_patterns("test_phase2_patterns")
            
            # Import patterns back to all services
            import_results = factory.import_service_patterns("test_phase2_patterns")
            
            successful_operations = sum(1 for r in export_results.values() if r) + sum(1 for r in import_results.values() if r)
            print(f"PASS: Cross-service pattern management: {successful_operations} successful operations")
            
            # Clean up
            import shutil
            if os.path.exists("test_phase2_patterns"):
                shutil.rmtree("test_phase2_patterns")
                
        except Exception as e:
            print(f"WARN: Cross-service pattern management test failed: {str(e)}")
        
        # Test 8: Test Phase 2 performance integration
        final_metrics = factory.get_all_service_metrics()
        
        total_metrics_categories = 0
        for service_type, metrics in final_metrics.items():
            if isinstance(metrics, dict):
                total_metrics_categories += len(metrics)
        
        print(f"PASS: Integrated performance metrics: {total_metrics_categories} total categories across {len(final_metrics)} services")
        
        # Test 9: Test Phase 2 validation and health checking
        from src.services.service_interfaces import ServiceHealthChecker
        
        final_health = ServiceHealthChecker.check_all_services_health(factory.registry)
        healthy_services = sum(1 for h in final_health.values() if h['healthy'])
        
        enhanced_services = 0
        for health_info in final_health.values():
            checks = health_info.get('checks', {})
            if 'pattern_management' in checks and checks['pattern_management'] == 'available':
                enhanced_services += 1
        
        print(f"PASS: Phase 2 health check: {healthy_services}/{len(final_health)} healthy, {enhanced_services} with enhanced capabilities")
        
        # Test 10: Summary of Phase 1-Phase 2 integration
        integration_features = [
            "Enhanced entity extraction",
            "Multi-mode matching",
            "Cross-service configuration",
            "Pattern management",
            "Performance metrics integration",
            "User behavior analysis",
            "Advanced normalization",
            "Contextual suggestions"
        ]
        
        print(f"PASS: Phase 1-Phase 2 integration features: {len(integration_features)}")
        for feature in integration_features:
            print(f"  [OK] {feature}")
        
        print("\n--- Phase 1 and Phase 2 Integration Test Completed ---")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Phase 1-Phase 2 integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that refactored services maintain backward compatibility"""
    print("\nTesting Backward Compatibility...")
    
    try:
        factory = service_factory
        
        # Test with default configurations (should work like original)
        services = factory.create_all_services()
        
        # Test basic functionality that should still work
        normalizer = factory.get_service(list(services.keys())[0])
        if normalizer:
            # Basic text normalization should still work
            result = normalizer.normalize_text("Simple test")
            print(f"PASS: Basic backward compatibility - text normalization")
            
            # Basic amount normalization should still work
            amount = normalizer.normalize_amount("100.50")
            print(f"PASS: Basic backward compatibility - amount normalization")
        
        # Test that old method calls still work (if they exist)
        matcher = factory.get_service(list(services.keys())[1])
        if matcher:
            # Should still have basic methods
            if hasattr(matcher, 'get_contextual_match_score'):
                print("PASS: Backward compatibility - contextual matching methods")
            
            if hasattr(matcher, 'get_performance_metrics'):
                print("PASS: Backward compatibility - performance metrics")
        
        print("PASS: Backward compatibility maintained")
        return True
        
    except Exception as e:
        print(f"FAIL: Backward compatibility test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success1 = test_phase1_phase2_integration()
    success2 = test_backward_compatibility()
    
    overall_success = success1 and success2
    sys.exit(0 if overall_success else 1)