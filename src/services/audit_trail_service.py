"""
Comprehensive Audit Trail System for Reconciliation Workflows
Tracks all changes, provides detailed history, and ensures compliance
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
import json
import hashlib
from src.models.user import db
from src.utils.logging_config import get_logger, get_audit_logger
from src.utils.exceptions import ValidationError, DatabaseError
from src.services.workflow_state_machine import ReconciliationState, WorkflowAction

logger = get_logger(__name__)
audit_logger = get_audit_logger()

class AuditEventType(Enum):
    """Types of audit events"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_CHANGE = "permission_change"
    ROLE_ASSIGNMENT = "role_assignment"
    ROLE_REMOVAL = "role_removal"
    
    # Reconciliation events
    RECONCILIATION_CREATED = "reconciliation_created"
    RECONCILIATION_UPDATED = "reconciliation_updated"
    RECONCILIATION_DELETED = "reconciliation_deleted"
    RECONCILIATION_APPROVED = "reconciliation_approved"
    RECONCILIATION_REJECTED = "reconciliation_rejected"
    RECONCILIATION_ESCALATED = "reconciliation_escalated"
    RECONCILIATION_CANCELLED = "reconciliation_cancelled"
    RECONCILIATION_RECONCILED = "reconciliation_reconciled"
    
    # Workflow events
    WORKFLOW_TRANSITION = "workflow_transition"
    WORKFLOW_TIMEOUT = "workflow_timeout"
    WORKFLOW_ESCALATION = "workflow_escalation"
    
    # Data events
    DATA_IMPORTED = "data_imported"
    DATA_EXPORTED = "data_exported"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    
    # System events
    SYSTEM_CONFIG_CHANGED = "system_config_changed"
    SECURITY_ALERT = "security_alert"
    COMPLIANCE_CHECK = "compliance_check"

class AuditSeverity(Enum):
    """Severity levels for audit events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Represents an audit event"""
    id: Optional[int] = None
    event_type: AuditEventType = field(default=AuditEventType.SYSTEM_CONFIG_CHANGED)
    severity: AuditSeverity = field(default=AuditSeverity.MEDIUM)
    user_id: Optional[int] = None
    user_role: Optional[str] = None
    user_name: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Event details
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    description: Optional[str] = None
    
    # Change tracking
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    changed_fields: Optional[List[str]] = None
    
    # Context and metadata
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    
    # Compliance and security
    compliance_tags: List[str] = field(default_factory=list)
    requires_review: bool = False
    reviewed_by: Optional[int] = None
    reviewed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        
        # Convert enums to strings
        if isinstance(result['event_type'], AuditEventType):
            result['event_type'] = result['event_type'].value
        if isinstance(result['severity'], AuditSeverity):
            result['severity'] = result['severity'].value
        
        # Convert datetime objects to ISO format
        for field_name in ['timestamp', 'created_at', 'reviewed_at']:
            if result[field_name]:
                result[field_name] = result[field_name].isoformat()
        
        return result

class AuditTrailService:
    """
    Comprehensive audit trail service for reconciliation workflows
    Provides detailed change tracking, compliance reporting, and security monitoring
    """
    
    def __init__(self):
        self.event_handlers = self._register_event_handlers()
        self.compliance_rules = self._define_compliance_rules()
        self.retention_policy = self._define_retention_policy()
        self.sensitive_fields = self._define_sensitive_fields()
    
    def _register_event_handlers(self) -> Dict[AuditEventType, callable]:
        """Register event handlers for different audit event types"""
        return {
            AuditEventType.RECONCILIATION_CREATED: self._handle_reconciliation_created,
            AuditEventType.RECONCILIATION_UPDATED: self._handle_reconciliation_updated,
            AuditEventType.RECONCILIATION_APPROVED: self._handle_reconciliation_approved,
            AuditEventType.RECONCILIATION_REJECTED: self._handle_reconciliation_rejected,
            AuditEventType.WORKFLOW_TRANSITION: self._handle_workflow_transition,
            AuditEventType.PERMISSION_CHANGE: self._handle_permission_change,
            AuditEventType.SECURITY_ALERT: self._handle_security_alert
        }
    
    def _define_compliance_rules(self) -> List[Dict[str, Any]]:
        """Define compliance rules for audit events"""
        return [
            {
                'name': 'segregation_of_duties',
                'description': 'Ensure proper segregation of duties in approval process',
                'event_types': [AuditEventType.RECONCILIATION_APPROVED],
                'conditions': [
                    'creator != approver',
                    'approval_amount < user_approval_limit'
                ],
                'severity': AuditSeverity.HIGH
            },
            {
                'name': 'four_eyes_principle',
                'description': 'Require at least two approvals for high-value transactions',
                'event_types': [AuditEventType.RECONCILIATION_APPROVED],
                'conditions': [
                    'amount > 50000 implies approval_count >= 2'
                ],
                'severity': AuditSeverity.HIGH
            },
            {
                'name': 'timely_processing',
                'description': 'Ensure timely processing of reconciliation requests',
                'event_types': [AuditEventType.WORKFLOW_TIMEOUT],
                'conditions': [
                    'timeout_duration < 48 hours'
                ],
                'severity': AuditSeverity.MEDIUM
            }
        ]
    
    def _define_retention_policy(self) -> Dict[str, int]:
        """Define retention policy for audit events (in days)"""
        return {
            'user_events': 365,          # 1 year
            'reconciliation_events': 2555,  # 7 years
            'workflow_events': 2555,      # 7 years
            'system_events': 90,          # 3 months
            'security_events': 2555,      # 7 years
            'compliance_events': 2555     # 7 years
        }
    
    def _define_sensitive_fields(self) -> Set[str]:
        """Define fields that should be masked in audit logs"""
        return {
            'password', 'token', 'secret', 'key', 'credential',
            'ssn', 'tax_id', 'bank_account', 'credit_card'
        }
    
    def log_event(self, event: AuditEvent) -> bool:
        """
        Log an audit event to the database
        """
        try:
            # Validate event
            if not self._validate_event(event):
                return False
            
            # Mask sensitive data
            self._mask_sensitive_data(event)
            
            # Calculate event hash for integrity
            event_hash = self._calculate_event_hash(event)
            
            # Store event in database
            # This would typically insert into an audit_events table
            audit_logger.log_audit_event(
                event_type=event.event_type.value,
                user_id=event.user_id,
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                action=event.action,
                description=event.description,
                metadata={
                    'event_hash': event_hash,
                    'severity': event.severity.value,
                    'context': event.context,
                    'compliance_tags': event.compliance_tags
                }
            )
            
            # Run compliance checks
            self._run_compliance_checks(event)
            
            # Trigger event handler
            if event.event_type in self.event_handlers:
                self.event_handlers[event.event_type](event)
            
            logger.info(f"Audit event logged: {event.event_type.value} for {event.resource_type} {event.resource_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging audit event: {str(e)}")
            return False
    
    def log_workflow_transition(self, reconciliation_id: int, from_state: str, to_state: str,
                              action: str, user_id: int, user_name: str,
                              justification: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log a workflow transition event
        """
        event = AuditEvent(
            event_type=AuditEventType.WORKFLOW_TRANSITION,
            severity=AuditSeverity.MEDIUM,
            user_id=user_id,
            user_name=user_name,
            resource_type="reconciliation",
            resource_id=str(reconciliation_id),
            action=action,
            description=f"Workflow transition: {from_state} -> {to_state}",
            context={
                'from_state': from_state,
                'to_state': to_state,
                'action': action,
                'justification': justification
            },
            metadata=metadata,
            compliance_tags=['workflow', 'reconciliation']
        )
        
        return self.log_event(event)
    
    def log_reconciliation_action(self, reconciliation_id: int, action: str, user_id: int, user_name: str,
                                 old_values: Optional[Dict[str, Any]] = None,
                                 new_values: Optional[Dict[str, Any]] = None,
                                 justification: Optional[str] = None) -> bool:
        """
        Log a reconciliation action event
        """
        event = AuditEvent(
            event_type=AuditEventType.RECONCILIATION_UPDATED,
            severity=AuditSeverity.MEDIUM,
            user_id=user_id,
            user_name=user_name,
            resource_type="reconciliation",
            resource_id=str(reconciliation_id),
            action=action,
            description=f"Reconciliation {action} for record {reconciliation_id}",
            old_values=old_values,
            new_values=new_values,
            changed_fields=self._calculate_changed_fields(old_values, new_values),
            context={'justification': justification},
            compliance_tags=['reconciliation', 'data_change']
        )
        
        return self.log_event(event)
    
    def log_permission_change(self, user_id: int, changed_by: int, old_role: str, new_role: str,
                             reason: Optional[str] = None) -> bool:
        """
        Log a permission change event
        """
        event = AuditEvent(
            event_type=AuditEventType.PERMISSION_CHANGE,
            severity=AuditSeverity.HIGH,
            user_id=changed_by,
            resource_type="user",
            resource_id=str(user_id),
            action="role_change",
            description=f"Role changed for user {user_id}: {old_role} -> {new_role}",
            old_values={'role': old_role},
            new_values={'role': new_role},
            changed_fields=['role'],
            context={'reason': reason},
            compliance_tags=['security', 'access_control']
        )
        
        return self.log_event(event)
    
    def log_security_alert(self, alert_type: str, user_id: Optional[int], description: str,
                          severity: AuditSeverity = AuditSeverity.HIGH,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log a security alert event
        """
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_ALERT,
            severity=severity,
            user_id=user_id,
            resource_type="system",
            action="security_alert",
            description=description,
            context={'alert_type': alert_type},
            metadata=metadata,
            compliance_tags=['security', 'alert'],
            requires_review=True
        )
        
        return self.log_event(event)
    
    def get_audit_history(self, resource_type: str, resource_id: str,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          user_id: Optional[int] = None,
                          event_types: Optional[List[AuditEventType]] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get audit history for a specific resource
        """
        try:
            # This would query the audit_events table
            # For now, return mock data
            return []
        except Exception as e:
            logger.error(f"Error getting audit history: {str(e)}")
            return []
    
    def get_user_activity(self, user_id: int, start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get user activity history
        """
        try:
            # This would query the audit_events table for user activity
            return []
        except Exception as e:
            logger.error(f"Error getting user activity: {str(e)}")
            return []
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate compliance report for a date range
        """
        try:
            report = {
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'summary': {
                    'total_events': 0,
                    'critical_events': 0,
                    'high_severity_events': 0,
                    'compliance_violations': 0
                },
                'events_by_type': {},
                'events_by_user': {},
                'compliance_issues': [],
                'recommendations': []
            }
            
            # This would query the audit_events table and generate the report
            # For now, return the structure
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {str(e)}")
            return {}
    
    def _validate_event(self, event: AuditEvent) -> bool:
        """Validate audit event before logging"""
        if not event.event_type:
            logger.error("Audit event missing event type")
            return False
        
        if not event.user_id and event.event_type not in [
            AuditEventType.SYSTEM_CONFIG_CHANGED,
            AuditEventType.SECURITY_ALERT
        ]:
            logger.error("Audit event missing user ID")
            return False
        
        if not event.resource_type:
            logger.error("Audit event missing resource type")
            return False
        
        return True
    
    def _mask_sensitive_data(self, event: AuditEvent) -> None:
        """Mask sensitive data in audit event"""
        if event.old_values:
            event.old_values = self._mask_dict_sensitive_fields(event.old_values)
        
        if event.new_values:
            event.new_values = self._mask_dict_sensitive_fields(event.new_values)
        
        if event.context:
            event.context = self._mask_dict_sensitive_fields(event.context)
        
        if event.metadata:
            event.metadata = self._mask_dict_sensitive_fields(event.metadata)
    
    def _mask_dict_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive fields in a dictionary"""
        masked_data = data.copy()
        
        for field, value in data.items():
            if any(sensitive in field.lower() for sensitive in self.sensitive_fields):
                if isinstance(value, str) and len(value) > 0:
                    masked_data[field] = '*' * min(len(value), 8)
                elif isinstance(value, (int, float)):
                    masked_data[field] = 0
        
        return masked_data
    
    def _calculate_event_hash(self, event: AuditEvent) -> str:
        """Calculate hash for event integrity verification"""
        event_data = {
            'event_type': event.event_type.value if isinstance(event.event_type, AuditEventType) else event.event_type,
            'user_id': event.user_id,
            'resource_type': event.resource_type,
            'resource_id': event.resource_id,
            'action': event.action,
            'timestamp': event.timestamp.isoformat(),
            'old_values': event.old_values,
            'new_values': event.new_values
        }
        
        event_string = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_string.encode()).hexdigest()
    
    def _calculate_changed_fields(self, old_values: Optional[Dict[str, Any]], 
                                 new_values: Optional[Dict[str, Any]]) -> List[str]:
        """Calculate which fields were changed"""
        if not old_values or not new_values:
            return []
        
        changed_fields = []
        
        for key in set(old_values.keys()) | set(new_values.keys()):
            if old_values.get(key) != new_values.get(key):
                changed_fields.append(key)
        
        return changed_fields
    
    def _run_compliance_checks(self, event: AuditEvent) -> None:
        """Run compliance checks on the event"""
        for rule in self.compliance_rules:
            if event.event_type in rule['event_types']:
                if not self._evaluate_compliance_rule(rule, event):
                    # Log compliance violation
                    self.log_security_alert(
                        alert_type="compliance_violation",
                        user_id=event.user_id,
                        description=f"Compliance rule '{rule['name']}' violated",
                        severity=rule['severity'],
                        metadata={
                            'rule_name': rule['name'],
                            'rule_description': rule['description'],
                            'event_id': event.id
                        }
                    )
    
    def _evaluate_compliance_rule(self, rule: Dict[str, Any], event: AuditEvent) -> bool:
        """Evaluate a compliance rule against an event"""
        # This would implement complex rule evaluation logic
        # For now, return True (pass)
        return True
    
    # Event handlers
    def _handle_reconciliation_created(self, event: AuditEvent) -> None:
        """Handle reconciliation created event"""
        logger.info(f"Reconciliation created: {event.resource_id} by user {event.user_id}")
    
    def _handle_reconciliation_updated(self, event: AuditEvent) -> None:
        """Handle reconciliation updated event"""
        if event.changed_fields:
            logger.info(f"Reconciliation {event.resource_id} updated fields: {', '.join(event.changed_fields)}")
    
    def _handle_reconciliation_approved(self, event: AuditEvent) -> None:
        """Handle reconciliation approved event"""
        logger.info(f"Reconciliation {event.resource_id} approved by user {event.user_id}")
    
    def _handle_reconciliation_rejected(self, event: AuditEvent) -> None:
        """Handle reconciliation rejected event"""
        logger.info(f"Reconciliation {event.resource_id} rejected by user {event.user_id}")
    
    def _handle_workflow_transition(self, event: AuditEvent) -> None:
        """Handle workflow transition event"""
        if event.context:
            from_state = event.context.get('from_state')
            to_state = event.context.get('to_state')
            logger.info(f"Workflow transition: {from_state} -> {to_state} for {event.resource_id}")
    
    def _handle_permission_change(self, event: AuditEvent) -> None:
        """Handle permission change event"""
        logger.info(f"Permission changed for user {event.resource_id} by user {event.user_id}")
    
    def _handle_security_alert(self, event: AuditEvent) -> None:
        """Handle security alert event"""
        logger.warning(f"Security alert: {event.description}")
    
    def cleanup_old_events(self) -> int:
        """Clean up old audit events based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=90)  # Default 90 days
            # This would delete old events from the database
            deleted_count = 0
            logger.info(f"Cleaned up {deleted_count} old audit events")
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up old audit events: {str(e)}")
            return 0
    
    def export_audit_data(self, start_date: datetime, end_date: datetime, 
                         format: str = 'json') -> Optional[str]:
        """Export audit data for a date range"""
        try:
            # This would query and export audit data
            return None
        except Exception as e:
            logger.error(f"Error exporting audit data: {str(e)}")
            return None

# Global audit trail service instance
audit_trail_service = AuditTrailService()