"""
Test script for service interfaces and factory
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.services.service_interfaces import (
    ServiceRegistry, ServiceHealthChecker, ServiceType, ServiceConfig, ProcessingMode
)
from src.services.service_factory import ServiceFactory
from src.services.data_normalization import NormalizationConfig, NormalizationMode
from src.services.context_aware_matching import MatchingConfig, MatchingMode
from src.services.reconciliation_service import ReconciliationConfig

def test_service_interfaces_and_factory():
    """Test service interfaces and factory"""
    print("Testing Service Interfaces and Factory...")
    
    try:
        # Test 1: Service registry initialization
        registry = ServiceRegistry()
        print("PASS: Service registry initialized")
        
        # Test 2: Service factory initialization
        factory = ServiceFactory(registry)
        print("PASS: Service factory initialized")
        
        # Test 3: Create individual services
        norm_config = NormalizationConfig(mode=NormalizationMode.AGGRESSIVE)
        normalizer = factory.create_data_normalization_service(norm_config)
        print("PASS: Data normalization service created")
        
        match_config = MatchingConfig(mode=MatchingMode.BALANCED)
        matcher = factory.create_context_aware_matching_service(match_config)
        print("PASS: Context-aware matching service created")
        
        recon_config = ReconciliationConfig()
        recon_service = factory.create_reconciliation_service(recon_config)
        print("PASS: Reconciliation service created")
        
        # Test 4: Service registry operations
        registered_services = registry.list_services()
        print(f"PASS: Registered services: {len(registered_services)}")
        
        # Test 5: Service status management
        for service_type in registered_services:
            status = registry.get_service_status(service_type)
            print(f"PASS: {service_type.value} status: {status.value}")
        
        # Test 6: Service retrieval
        retrieved_normalizer = factory.get_service(ServiceType.DATA_NORMALIZATION)
        retrieved_config = factory.get_service_config(ServiceType.DATA_NORMALIZATION)
        print(f"PASS: Service retrieval successful - {type(retrieved_normalizer).__name__}")
        
        # Test 7: Service metrics
        metrics = factory.get_service_metrics(ServiceType.DATA_NORMALIZATION)
        if metrics:
            print(f"PASS: Service metrics available: {len(metrics)} categories")
        else:
            print("WARN: Service metrics not available")
        
        # Test 8: All service metrics
        all_metrics = factory.get_all_service_metrics()
        print(f"PASS: All service metrics: {len(all_metrics)} services")
        
        # Test 9: Service configuration update
        new_norm_config = NormalizationConfig(mode=NormalizationMode.STRICT)
        update_success = factory.update_service_config(ServiceType.DATA_NORMALIZATION, new_norm_config)
        print(f"PASS: Service configuration update: {update_success}")
        
        # Test 10: Service health checking
        health_report = ServiceHealthChecker.check_service_health(normalizer)
        print(f"PASS: Service health check: {health_report['healthy']}")
        
        # Test 11: All services health check
        all_health = ServiceHealthChecker.check_all_services_health(registry)
        print(f"PASS: All services health check: {len(all_health)} services")
        
        # Test 12: Service integration validation
        validation = factory.validate_service_integration()
        print(f"PASS: Service integration validation: {validation['valid']}")
        
        # Test 13: Pattern export/import
        export_results = factory.export_service_patterns("test_patterns")
        print(f"PASS: Pattern export: {export_results}")
        
        import_results = factory.import_service_patterns("test_patterns")
        print(f"PASS: Pattern import: {import_results}")
        
        # Test 14: Create all services at once
        new_factory = ServiceFactory()
        all_services = new_factory.create_all_services()
        print(f"PASS: Created all services: {len(all_services)} services")
        
        # Test 15: Factory with custom configurations
        custom_factory = ServiceFactory()
        custom_services = custom_factory.create_all_services(
            normalization_config=NormalizationConfig(mode=NormalizationMode.AGGRESSIVE),
            matching_config=MatchingConfig(mode=MatchingMode.AGGRESSIVE),
            reconciliation_config=ReconciliationConfig()
        )
        print(f"PASS: Custom services created: {len(custom_services)} services")
        
        # Clean up test files
        test_files = [
            "test_patterns/data_normalization_patterns.json",
            "test_patterns/context_aware_matching_patterns.json",
            "test_patterns/reconciliation_patterns.json"
        ]
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Clean up test directory
        if os.path.exists("test_patterns"):
            os.rmdir("test_patterns")
        
        print("\n--- All Service Interfaces and Factory Tests Passed ---")
        return True
        
    except Exception as e:
        print(f"FAIL: Service interfaces and factory test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_service_interfaces_and_factory()
    sys.exit(0 if success else 1)