"""
Service interfaces and contracts for Phase 1 refactored services
Defines standardized interfaces for data normalization, context-aware matching, and reconciliation
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class ServiceType(Enum):
    """Types of services in the system"""
    DATA_NORMALIZATION = "data_normalization"
    CONTEXT_AWARE_MATCHING = "context_aware_matching"
    RECONCILIATION = "reconciliation"

class ProcessingMode(Enum):
    """Processing modes for services"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"

class ServiceStatus(Enum):
    """Service status indicators"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class ServiceMetrics:
    """Standardized service metrics"""
    total_processed: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_processing_time: float = 0.0
    last_operation_time: Optional[datetime] = None
    cache_hit_rate: float = 0.0
    memory_usage: float = 0.0
    uptime: float = 0.0

@dataclass
class ServiceConfig:
    """Base configuration for all services"""
    processing_mode: ProcessingMode = ProcessingMode.SYNCHRONOUS
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_metrics: bool = True
    debug_mode: bool = False
    max_concurrent_operations: int = 10
    timeout_seconds: int = 300

@dataclass
class ProcessingResult:
    """Standardized processing result"""
    success: bool
    result_data: Any = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None

class IDataNormalizationService(ABC):
    """Interface for data normalization services"""
    
    @abstractmethod
    def normalize_text(self, text: str) -> Any:
        """Normalize text content"""
        pass
    
    @abstractmethod
    def normalize_amount(self, amount_text: str) -> float:
        """Normalize amount text to float"""
        pass
    
    @abstractmethod
    def normalize_date(self, date_text: str) -> Optional[str]:
        """Normalize date text to standard format"""
        pass
    
    @abstractmethod
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> ServiceMetrics:
        """Get service performance metrics"""
        pass
    
    @abstractmethod
    def update_config(self, config: Any) -> bool:
        """Update service configuration"""
        pass

class IContextAwareMatchingService(ABC):
    """Interface for context-aware matching services"""
    
    @abstractmethod
    def analyze_historical_patterns(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Analyze historical patterns"""
        pass
    
    @abstractmethod
    def get_contextual_match_score(self, bank_transaction: Any, company_entry: Any, 
                                 base_score: float, user_id: Optional[int] = None) -> Any:
        """Calculate contextual match score"""
        pass
    
    @abstractmethod
    def predict_missing_information(self, transaction: Any, entry_type: str = 'company') -> Any:
        """Predict missing information"""
        pass
    
    @abstractmethod
    def get_matching_suggestions(self, bank_transaction: Any, available_entries: List[Any],
                               user_id: Optional[int] = None) -> List[Any]:
        """Get matching suggestions"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> ServiceMetrics:
        """Get service performance metrics"""
        pass
    
    @abstractmethod
    def update_config(self, config: Any) -> bool:
        """Update service configuration"""
        pass
    
    @abstractmethod
    def export_patterns(self, file_path: str) -> bool:
        """Export learned patterns"""
        pass
    
    @abstractmethod
    def import_patterns(self, file_path: str) -> bool:
        """Import learned patterns"""
        pass

class IReconciliationService(ABC):
    """Interface for reconciliation services"""
    
    @abstractmethod
    def find_matches(self, bank_transactions: List[Any], company_entries: List[Any]) -> List[Dict[str, Any]]:
        """Find matches between bank transactions and company entries"""
        pass
    
    @abstractmethod
    def get_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics"""
        pass
    
    @abstractmethod
    def configure_phase2_integration(self, normalization_config: Any = None,
                                   matching_config: Any = None) -> bool:
        """Configure Phase 2 integration"""
        pass
    
    @abstractmethod
    def get_contextual_suggestions(self, bank_transaction: Any, 
                                 available_entries: List[Any]) -> List[Dict[str, Any]]:
        """Get contextual matching suggestions"""
        pass
    
    @abstractmethod
    def predict_missing_information(self, transaction: Any) -> Dict[str, Any]:
        """Predict missing information"""
        pass

class ServiceRegistry:
    """Registry for managing service instances"""
    
    def __init__(self):
        self._services: Dict[ServiceType, Any] = {}
        self._service_configs: Dict[ServiceType, Any] = {}
        self._service_status: Dict[ServiceType, ServiceStatus] = {}
    
    def register_service(self, service_type: ServiceType, service: Any, config: Any = None):
        """Register a service instance"""
        self._services[service_type] = service
        self._service_configs[service_type] = config
        self._service_status[service_type] = ServiceStatus.IDLE
    
    def get_service(self, service_type: ServiceType) -> Any:
        """Get a service instance"""
        return self._services.get(service_type)
    
    def get_service_config(self, service_type: ServiceType) -> Any:
        """Get service configuration"""
        return self._service_configs.get(service_type)
    
    def get_service_status(self, service_type: ServiceType) -> ServiceStatus:
        """Get service status"""
        return self._service_status.get(service_type, ServiceStatus.IDLE)
    
    def set_service_status(self, service_type: ServiceType, status: ServiceStatus):
        """Set service status"""
        self._service_status[service_type] = status
    
    def list_services(self) -> List[ServiceType]:
        """List all registered services"""
        return list(self._services.keys())
    
    def get_all_metrics(self) -> Dict[ServiceType, ServiceMetrics]:
        """Get metrics for all services"""
        metrics = {}
        for service_type, service in self._services.items():
            if hasattr(service, 'get_performance_metrics'):
                metrics[service_type] = service.get_performance_metrics()
        return metrics

class ServiceHealthChecker:
    """Health checker for services"""
    
    @staticmethod
    def check_service_health(service: Any) -> Dict[str, Any]:
        """Check health of a service"""
        health_info = {
            'healthy': True,
            'status': 'ok',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Basic functionality check
            if hasattr(service, 'get_performance_metrics'):
                metrics = service.get_performance_metrics()
                health_info['checks']['metrics'] = 'available'
                health_info['metrics'] = metrics
            else:
                health_info['checks']['metrics'] = 'unavailable'
            
            # Configuration check
            if hasattr(service, 'config'):
                health_info['checks']['config'] = 'available'
            else:
                health_info['checks']['config'] = 'unavailable'
            
            # Pattern management check (for matching services)
            if hasattr(service, 'export_patterns') and hasattr(service, 'import_patterns'):
                health_info['checks']['pattern_management'] = 'available'
            else:
                health_info['checks']['pattern_management'] = 'unavailable'
            
        except Exception as e:
            health_info['healthy'] = False
            health_info['status'] = f'error: {str(e)}'
        
        return health_info
    
    @staticmethod
    def check_all_services_health(registry: ServiceRegistry) -> Dict[ServiceType, Dict[str, Any]]:
        """Check health of all registered services"""
        health_report = {}
        
        for service_type in registry.list_services():
            service = registry.get_service(service_type)
            if service:
                health_report[service_type] = ServiceHealthChecker.check_service_health(service)
            else:
                health_report[service_type] = {
                    'healthy': False,
                    'status': 'service_not_found',
                    'timestamp': datetime.now().isoformat()
                }
        
        return health_report

# Global service registry instance
service_registry = ServiceRegistry()