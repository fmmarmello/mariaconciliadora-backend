"""
Escalation Workflow Service

Handles automatic escalation of reconciliation items based on:
- Time-based SLA violations
- Approval thresholds
- Compliance requirements
- Risk-based prioritization
- Multi-level escalation chains
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from sqlalchemy import and_, or_
from src.models.transaction import db, ReconciliationRecord, Transaction
from src.models.user import User
from src.services.workflow_state_machine import AdvancedReconciliationStateMachine, ReconciliationState
from src.services.rbac_service import RBACService, UserRole, Permission
from src.services.audit_trail_service import AuditTrailService

logger = logging.getLogger(__name__)

class EscalationLevel(Enum):
    """Escalation levels for reconciliation items"""
    LEVEL_1 = "level_1"  # Reviewer
    LEVEL_2 = "level_2"  # Approver
    LEVEL_3 = "level_3"  # Admin
    LEVEL_4 = "level_4"  # Executive
    URGENT = "urgent"    # Immediate attention required

class EscalationTrigger(Enum):
    """Reasons for escalation"""
    TIMEOUT = "timeout"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    COMPLIANCE_VIOLATION = "compliance_violation"
    RISK_THRESHOLD = "risk_threshold"
    MANUAL_ESCALATION = "manual_escalation"
    REJECTION_THRESHOLD = "rejection_threshold"
    QUALITY_ISSUE = "quality_issue"

@dataclass
class EscalationRule:
    """Defines an escalation rule"""
    id: str
    name: str
    trigger: EscalationTrigger
    conditions: Dict[str, Any]
    escalation_level: EscalationLevel
    target_role: UserRole
    time_threshold: Optional[timedelta] = None
    amount_threshold: Optional[float] = None
    risk_threshold: Optional[float] = None
    is_active: bool = True

@dataclass
class EscalationEvent:
    """Represents an escalation event"""
    id: int
    reconciliation_id: int
    rule_id: str
    rule_name: str
    trigger: EscalationTrigger
    from_level: Optional[EscalationLevel]
    to_level: EscalationLevel
    target_user_id: int
    target_user_name: str
    reason: str
    severity: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None

class EscalationService:
    """Service for managing escalation workflows"""
    
    def __init__(self):
        self.rbac_service = RBACService()
        self.workflow_machine = AdvancedReconciliationStateMachine()
        self.audit_service = AuditTrailService()
        
        # Define default escalation rules
        self.rules = self._initialize_rules()
        
        # SLA thresholds (in hours)
        self.sla_thresholds = {
            ReconciliationState.PENDING_REVIEW: 24,      # 24 hours for review
            ReconciliationState.PENDING_APPROVAL: 48,     # 48 hours for approval
            ReconciliationState.ESCALATED: 8,            # 8 hours for escalated items
            ReconciliationState.ON_HOLD: 72,             # 72 hours for on-hold items
        }
    
    def _initialize_rules(self) -> List[EscalationRule]:
        """Initialize default escalation rules"""
        return [
            # Time-based escalation rules
            EscalationRule(
                id="timeout_review",
                name="Review Timeout",
                trigger=EscalationTrigger.TIMEOUT,
                conditions={"state": "pending_review", "hours_pending": 24},
                escalation_level=EscalationLevel.LEVEL_1,
                target_role=UserRole.REVIEWER,
                time_threshold=timedelta(hours=24)
            ),
            EscalationRule(
                id="timeout_approval",
                name="Approval Timeout",
                trigger=EscalationTrigger.TIMEOUT,
                conditions={"state": "pending_approval", "hours_pending": 48},
                escalation_level=EscalationLevel.LEVEL_2,
                target_role=UserRole.APPROVER,
                time_threshold=timedelta(hours=48)
            ),
            EscalationRule(
                id="timeout_escalated",
                name="Escalated Item Timeout",
                trigger=EscalationTrigger.TIMEOUT,
                conditions={"state": "escalated", "hours_pending": 8},
                escalation_level=EscalationLevel.LEVEL_3,
                target_role=UserRole.ADMIN,
                time_threshold=timedelta(hours=8)
            ),
            
            # Amount-based escalation rules
            EscalationRule(
                id="high_value_transaction",
                name="High Value Transaction",
                trigger=EscalationTrigger.THRESHOLD_EXCEEDED,
                conditions={"min_amount": 100000},
                escalation_level=EscalationLevel.LEVEL_2,
                target_role=UserRole.APPROVER,
                amount_threshold=100000.0
            ),
            EscalationRule(
                id="critical_value_transaction",
                name="Critical Value Transaction",
                trigger=EscalationTrigger.RISK_THRESHOLD,
                conditions={"min_amount": 500000},
                escalation_level=EscalationLevel.LEVEL_3,
                target_role=UserRole.ADMIN,
                amount_threshold=500000.0
            ),
            
            # Risk-based escalation rules
            EscalationRule(
                id="high_risk_match",
                name="High Risk Match",
                trigger=EscalationTrigger.RISK_THRESHOLD,
                conditions={"max_match_score": 0.3},
                escalation_level=EscalationLevel.LEVEL_2,
                target_role=UserRole.APPROVER,
                risk_threshold=0.3
            ),
            EscalationRule(
                id="very_high_risk_match",
                name="Very High Risk Match",
                trigger=EscalationTrigger.RISK_THRESHOLD,
                conditions={"max_match_score": 0.1},
                escalation_level=EscalationLevel.LEVEL_3,
                target_role=UserRole.ADMIN,
                risk_threshold=0.1
            ),
            
            # Compliance escalation rules
            EscalationRule(
                id="compliance_violation",
                name="Compliance Violation",
                trigger=EscalationTrigger.COMPLIANCE_VIOLATION,
                conditions={"compliance_issue": True},
                escalation_level=EscalationLevel.LEVEL_3,
                target_role=UserRole.ADMIN
            ),
            
            # Rejection threshold rules
            EscalationRule(
                id="multiple_rejections",
                name="Multiple Rejections",
                trigger=EscalationTrigger.REJECTION_THRESHOLD,
                conditions={"rejection_count": 3},
                escalation_level=EscalationLevel.LEVEL_2,
                target_role=UserRole.APPROVER
            ),
        ]
    
    def check_escalations(self) -> List[EscalationEvent]:
        """Check for items that need escalation and create escalation events"""
        escalation_events = []
        
        # Get all pending reconciliation items
        pending_items = ReconciliationRecord.query.filter(
            ReconciliationRecord.status.in_([
                'pending_review', 'pending_approval', 'escalated', 'on_hold'
            ])
        ).all()
        
        for item in pending_items:
            # Check each rule against the item
            for rule in self.rules:
                if not rule.is_active:
                    continue
                    
                if self._should_escalate(item, rule):
                    escalation_event = self._create_escalation_event(item, rule)
                    if escalation_event:
                        escalation_events.append(escalation_event)
                        self._process_escalation(escalation_event)
        
        return escalation_events
    
    def _should_escalate(self, item: ReconciliationRecord, rule: EscalationRule) -> bool:
        """Check if an item should be escalated based on a rule"""
        try:
            # Time-based checks
            if rule.trigger == EscalationTrigger.TIMEOUT:
                if rule.time_threshold:
                    time_pending = datetime.utcnow() - item.created_at
                    return time_pending >= rule.time_threshold
            
            # Amount-based checks
            elif rule.trigger == EscalationTrigger.THRESHOLD_EXCEEDED:
                if rule.amount_threshold and item.bank_transaction:
                    return abs(item.bank_transaction.amount) >= rule.amount_threshold
            
            # Risk-based checks
            elif rule.trigger == EscalationTrigger.RISK_THRESHOLD:
                if rule.risk_threshold:
                    return item.match_score <= rule.risk_threshold
            
            # State-based checks
            if "state" in rule.conditions:
                return item.status == rule.conditions["state"]
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking escalation rule {rule.id}: {str(e)}")
            return False
    
    def _create_escalation_event(self, item: ReconciliationRecord, rule: EscalationRule) -> Optional[EscalationEvent]:
        """Create an escalation event for a matching rule"""
        try:
            # Find target user with appropriate role
            target_user = self._find_escalation_target(rule.target_role)
            if not target_user:
                logger.warning(f"No target user found for role {rule.target_role}")
                return None
            
            # Determine current escalation level
            current_level = self._get_current_escalation_level(item)
            
            # Create escalation event
            escalation_event = EscalationEvent(
                id=0,  # Will be set when saved to database
                reconciliation_id=item.id,
                rule_id=rule.id,
                rule_name=rule.name,
                trigger=rule.trigger,
                from_level=current_level,
                to_level=rule.escalation_level,
                target_user_id=target_user.id,
                target_user_name=target_user.username,
                reason=self._generate_escalation_reason(item, rule),
                severity=self._calculate_severity(rule),
                created_at=datetime.utcnow()
            )
            
            return escalation_event
            
        except Exception as e:
            logger.error(f"Error creating escalation event: {str(e)}")
            return None
    
    def _find_escalation_target(self, target_role: UserRole) -> Optional[User]:
        """Find a user with the target role for escalation"""
        try:
            # For now, find any user with the required role
            # In production, this would consider workload, availability, etc.
            target_users = User.query.filter(User.role == target_role.value).all()
            
            if not target_users:
                logger.warning(f"No users found with role {target_role}")
                return None
            
            # Simple round-robin selection (could be enhanced with proper load balancing)
            return target_users[0]
            
        except Exception as e:
            logger.error(f"Error finding escalation target: {str(e)}")
            return None
    
    def _get_current_escalation_level(self, item: ReconciliationRecord) -> Optional[EscalationLevel]:
        """Determine current escalation level of an item"""
        if item.status == 'escalated':
            return EscalationLevel.LEVEL_1  # Default to level 1
        return None
    
    def _generate_escalation_reason(self, item: ReconciliationRecord, rule: EscalationRule) -> str:
        """Generate a human-readable reason for escalation"""
        reasons = {
            EscalationTrigger.TIMEOUT: f"Item has been pending for {(datetime.utcnow() - item.created_at).total_seconds() / 3600:.1f} hours",
            EscalationTrigger.THRESHOLD_EXCEEDED: f"Transaction amount {abs(item.bank_transaction.amount):.2f} exceeds threshold",
            EscalationTrigger.RISK_THRESHOLD: f"Low match score {item.match_score:.2f} indicates high risk",
            EscalationTrigger.COMPLIANCE_VIOLATION: "Compliance violation detected",
            EscalationTrigger.REJECTION_THRESHOLD: "Multiple rejections detected",
            EscalationTrigger.MANUAL_ESCALATION: "Manual escalation requested",
            EscalationTrigger.QUALITY_ISSUE: "Quality issues detected in reconciliation",
        }
        
        return reasons.get(rule.trigger, f"Escalation triggered by {rule.trigger.value}")
    
    def _calculate_severity(self, rule: EscalationRule) -> str:
        """Calculate severity level for escalation"""
        severity_map = {
            EscalationLevel.LEVEL_1: "low",
            EscalationLevel.LEVEL_2: "medium",
            EscalationLevel.LEVEL_3: "high",
            EscalationLevel.LEVEL_4: "critical",
            EscalationLevel.URGENT: "urgent",
        }
        return severity_map.get(rule.escalation_level, "medium")
    
    def _process_escalation(self, escalation_event: EscalationEvent) -> bool:
        """Process an escalation event (update state, notify, etc.)"""
        try:
            # Update reconciliation record state
            reconciliation = ReconciliationRecord.query.get(escalation_event.reconciliation_id)
            if reconciliation:
                old_state = reconciliation.status
                new_state = "escalated"
                
                reconciliation.status = new_state
                db.session.commit()
                
                # Log workflow transition
                self.audit_service.log_workflow_transition(
                    reconciliation_id=reconciliation.id,
                    from_state=old_state,
                    to_state=new_state,
                    action="escalate",
                    user_id=0,  # System user
                    user_name="System",
                    justification=escalation_event.reason
                )
                
                # Send notification (in production, this would integrate with email/Slack)
                self._send_escalation_notification(escalation_event)
                
                logger.info(f"Escalated reconciliation {reconciliation.id} to {escalation_event.target_user_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing escalation: {str(e)}")
            return False
    
    def _send_escalation_notification(self, escalation_event: EscalationEvent) -> None:
        """Send notification about escalation (placeholder for actual notification system)"""
        # In production, this would send emails, Slack messages, etc.
        logger.info(f"NOTIFICATION: Escalation for reconciliation {escalation_event.reconciliation_id}")
        logger.info(f"Target: {escalation_event.target_user_name}")
        logger.info(f"Reason: {escalation_event.reason}")
        logger.info(f"Severity: {escalation_event.severity}")
    
    def get_pending_escalations(self, user_id: int) -> List[EscalationEvent]:
        """Get pending escalations for a specific user"""
        # This would query the database for pending escalations
        # For now, return empty list as placeholder
        return []
    
    def resolve_escalation(self, escalation_id: int, resolution_notes: str, user_id: int) -> bool:
        """Resolve an escalation event"""
        try:
            # This would update the escalation event in the database
            # For now, just log the resolution
            logger.info(f"Escalation {escalation_id} resolved by user {user_id}")
            logger.info(f"Resolution notes: {resolution_notes}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving escalation: {str(e)}")
            return False
    
    def get_escalation_statistics(self) -> Dict[str, Any]:
        """Get escalation statistics and metrics"""
        try:
            # This would query the database for actual statistics
            # For now, return placeholder data
            return {
                "total_escalations": 0,
                "pending_escalations": 0,
                "resolved_escalations": 0,
                "average_resolution_time": 0,
                "escalations_by_level": {
                    "level_1": 0,
                    "level_2": 0,
                    "level_3": 0,
                    "level_4": 0,
                    "urgent": 0
                },
                "escalations_by_trigger": {
                    "timeout": 0,
                    "threshold_exceeded": 0,
                    "compliance_violation": 0,
                    "risk_threshold": 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting escalation statistics: {str(e)}")
            return {}
    
    def add_custom_rule(self, rule: EscalationRule) -> bool:
        """Add a custom escalation rule"""
        try:
            self.rules.append(rule)
            logger.info(f"Added custom escalation rule: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom rule: {str(e)}")
            return False
    
    def deactivate_rule(self, rule_id: str) -> bool:
        """Deactivate an escalation rule"""
        try:
            for rule in self.rules:
                if rule.id == rule_id:
                    rule.is_active = False
                    logger.info(f"Deactivated escalation rule: {rule.name}")
                    return True
            
            logger.warning(f"Rule not found: {rule_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error deactivating rule: {str(e)}")
            return False
    
    def get_active_rules(self) -> List[EscalationRule]:
        """Get all active escalation rules"""
        return [rule for rule in self.rules if rule.is_active]