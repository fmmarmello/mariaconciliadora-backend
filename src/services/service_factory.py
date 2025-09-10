"""
Service factory for creating and configuring Phase 1 refactored services
Provides centralized service creation with dependency injection
"""

from typing import Dict, Any, Optional, Union
from src.services.service_interfaces import (
    ServiceRegistry, ServiceConfig, ProcessingMode, ServiceType,
    IDataNormalizationService, IContextAwareMatchingService, IReconciliationService
)
from src.services.data_normalization import BrazilianDataNormalizer, NormalizationConfig, NormalizationMode
from src.services.context_aware_matching import ContextAwareMatcher, MatchingConfig, MatchingMode
from src.services.reconciliation_service import ReconciliationService, ReconciliationConfig
import logging

logger = logging.getLogger(__name__)

class ServiceFactory:
    """Factory for creating and configuring services"""
    
    def __init__(self, registry: ServiceRegistry = None):
        self.registry = registry or ServiceRegistry()
        self._default_configs = self._create_default_configs()
    
    def _create_default_configs(self) -> Dict[ServiceType, Any]:
        """Create default configurations for all services"""
        return {
            ServiceType.DATA_NORMALIZATION: NormalizationConfig(
                mode=NormalizationMode.STANDARD,
                extract_entities=True,
                remove_noise_words=True
            ),
            ServiceType.CONTEXT_AWARE_MATCHING: MatchingConfig(
                mode=MatchingMode.BALANCED,
                enable_learning=True,
                max_suggestions=10,
                suggestion_threshold=0.6
            ),
            ServiceType.RECONCILIATION: ReconciliationConfig()
        }
    
    def create_data_normalization_service(self, 
                                        config: NormalizationConfig = None) -> BrazilianDataNormalizer:
        """Create and configure data normalization service"""
        try:
            service_config = config or self._default_configs[ServiceType.DATA_NORMALIZATION]
            service = BrazilianDataNormalizer(service_config)
            
            # Register service
            self.registry.register_service(ServiceType.DATA_NORMALIZATION, service, service_config)
            
            logger.info(f"Created data normalization service with {service_config.mode.value} mode")
            return service
            
        except Exception as e:
            logger.error(f"Error creating data normalization service: {str(e)}")
            raise
    
    def create_context_aware_matching_service(self, 
                                            config: MatchingConfig = None) -> ContextAwareMatcher:
        """Create and configure context-aware matching service"""
        try:
            service_config = config or self._default_configs[ServiceType.CONTEXT_AWARE_MATCHING]
            service = ContextAwareMatcher(service_config)
            
            # Register service
            self.registry.register_service(ServiceType.CONTEXT_AWARE_MATCHING, service, service_config)
            
            logger.info(f"Created context-aware matching service with {service_config.mode.value} mode")
            return service
            
        except Exception as e:
            logger.error(f"Error creating context-aware matching service: {str(e)}")
            raise
    
    def create_reconciliation_service(self, 
                                   config: ReconciliationConfig = None,
                                   use_phase2_integration: bool = True) -> ReconciliationService:
        """Create and configure reconciliation service"""
        try:
            service_config = config or self._default_configs[ServiceType.RECONCILIATION]
            service = ReconciliationService(service_config)
            
            # Enable Phase 2 integration if requested
            if use_phase2_integration:
                self._setup_phase2_integration(service)
            
            # Register service
            self.registry.register_service(ServiceType.RECONCILIATION, service, service_config)
            
            logger.info("Created reconciliation service with Phase 2 integration")
            return service
            
        except Exception as e:
            logger.error(f"Error creating reconciliation service: {str(e)}")
            raise
    
    def _setup_phase2_integration(self, reconciliation_service: ReconciliationService):
        """Setup Phase 2 integration for reconciliation service"""
        try:
            # Get existing services or create them
            normalizer = self.registry.get_service(ServiceType.DATA_NORMALIZATION)
            matcher = self.registry.get_service(ServiceType.CONTEXT_AWARE_MATCHING)
            
            if not normalizer:
                normalizer = self.create_data_normalization_service()
            
            if not matcher:
                matcher = self.create_context_aware_matching_service()
            
            # Configure integration
            norm_config = self.registry.get_service_config(ServiceType.DATA_NORMALIZATION)
            match_config = self.registry.get_service_config(ServiceType.CONTEXT_AWARE_MATCHING)
            
            reconciliation_service.configure_phase2_integration(norm_config, match_config)
            
            logger.info("Phase 2 integration configured for reconciliation service")
            
        except Exception as e:
            logger.error(f"Error setting up Phase 2 integration: {str(e)}")
            raise
    
    def create_all_services(self, 
                           normalization_config: NormalizationConfig = None,
                           matching_config: MatchingConfig = None,
                           reconciliation_config: ReconciliationConfig = None) -> Dict[ServiceType, Any]:
        """Create all services with proper integration"""
        services = {}
        
        try:
            # Create data normalization service
            services[ServiceType.DATA_NORMALIZATION] = self.create_data_normalization_service(normalization_config)
            
            # Create context-aware matching service
            services[ServiceType.CONTEXT_AWARE_MATCHING] = self.create_context_aware_matching_service(matching_config)
            
            # Create reconciliation service with Phase 2 integration
            services[ServiceType.RECONCILIATION] = self.create_reconciliation_service(reconciliation_config)
            
            logger.info("All services created successfully with Phase 2 integration")
            return services
            
        except Exception as e:
            logger.error(f"Error creating services: {str(e)}")
            raise
    
    def get_service(self, service_type: ServiceType) -> Any:
        """Get service from registry"""
        return self.registry.get_service(service_type)
    
    def get_service_config(self, service_type: ServiceType) -> Any:
        """Get service configuration from registry"""
        return self.registry.get_service_config(service_type)
    
    def update_service_config(self, service_type: ServiceType, config: Any) -> bool:
        """Update service configuration"""
        try:
            service = self.registry.get_service(service_type)
            if service and hasattr(service, 'update_config'):
                service.update_config(config)
                self.registry._service_configs[service_type] = config
                logger.info(f"Updated {service_type.value} service configuration")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating {service_type.value} service configuration: {str(e)}")
            return False
    
    def get_service_metrics(self, service_type: ServiceType) -> Optional[Dict[str, Any]]:
        """Get service metrics"""
        service = self.registry.get_service(service_type)
        if service and hasattr(service, 'get_performance_metrics'):
            return service.get_performance_metrics()
        return None
    
    def get_all_service_metrics(self) -> Dict[ServiceType, Dict[str, Any]]:
        """Get metrics for all services"""
        return self.registry.get_all_metrics()
    
    def validate_service_integration(self) -> Dict[str, Any]:
        """Validate integration between services"""
        validation_result = {
            'valid': True,
            'services': {},
            'integration': {},
            'errors': []
        }
        
        try:
            # Check individual services
            for service_type in ServiceType:
                service = self.registry.get_service(service_type)
                if service:
                    validation_result['services'][service_type.value] = {
                        'available': True,
                        'type': type(service).__name__
                    }
                else:
                    validation_result['services'][service_type.value] = {
                        'available': False,
                        'type': None
                    }
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"{service_type.value} service not available")
            
            # Check integration between services
            recon_service = self.registry.get_service(ServiceType.RECONCILIATION)
            if recon_service:
                # Check if Phase 2 integration is working
                try:
                    metrics = recon_service.get_enhanced_performance_metrics()
                    if 'context_matcher_metrics' in metrics:
                        validation_result['integration']['phase2'] = 'active'
                    else:
                        validation_result['integration']['phase2'] = 'limited'
                except Exception as e:
                    validation_result['integration']['phase2'] = f'error: {str(e)}'
                    validation_result['errors'].append(f"Phase 2 integration error: {str(e)}")
            else:
                validation_result['integration']['phase2'] = 'not_available'
            
            logger.info(f"Service integration validation: {validation_result['valid']}")
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Error validating service integration: {str(e)}")
        
        return validation_result
    
    def export_service_patterns(self, base_path: str) -> Dict[str, bool]:
        """Export patterns from all services that support it"""
        export_results = {}
        
        for service_type in [ServiceType.CONTEXT_AWARE_MATCHING, ServiceType.RECONCILIATION]:
            service = self.registry.get_service(service_type)
            if service and hasattr(service, 'export_patterns'):
                file_path = f"{base_path}/{service_type.value}_patterns.json"
                try:
                    success = service.export_patterns(file_path)
                    export_results[service_type.value] = success
                    logger.info(f"Exported patterns for {service_type.value} to {file_path}")
                except Exception as e:
                    export_results[service_type.value] = False
                    logger.error(f"Error exporting patterns for {service_type.value}: {str(e)}")
        
        return export_results
    
    def import_service_patterns(self, base_path: str) -> Dict[str, bool]:
        """Import patterns to all services that support it"""
        import_results = {}
        
        for service_type in [ServiceType.CONTEXT_AWARE_MATCHING, ServiceType.RECONCILIATION]:
            service = self.registry.get_service(service_type)
            if service and hasattr(service, 'import_patterns'):
                file_path = f"{base_path}/{service_type.value}_patterns.json"
                try:
                    success = service.import_patterns(file_path)
                    import_results[service_type.value] = success
                    logger.info(f"Imported patterns for {service_type.value} from {file_path}")
                except Exception as e:
                    import_results[service_type.value] = False
                    logger.error(f"Error importing patterns for {service_type.value}: {str(e)}")
        
        return import_results

# Global service factory instance
service_factory = ServiceFactory()