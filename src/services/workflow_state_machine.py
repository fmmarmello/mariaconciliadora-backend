"""
Advanced Reconciliation State Machine
Implements multi-stage approval workflows, role-based access control, and audit trails
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from src.models.transaction import ReconciliationRecord
from src.models.user import db
from src.utils.logging_config import get_logger, get_audit_logger
from src.utils.exceptions import (
    ValidationError, BusinessLogicError, AuthorizationError,
    ReconciliationError
)

logger = get_logger(__name__)
audit_logger = get_audit_logger()

class ReconciliationState(Enum):
    """Advanced reconciliation workflow states"""
    DRAFT = "draft"                    # Initial state, not yet ready for review
    PENDING_REVIEW = "pending_review"  # Ready for reviewer approval
    PENDING_APPROVAL = "pending_approval"  # Ready for manager approval
    APPROVED = "approved"              # Approved, ready for reconciliation
    RECONCILED = "reconciled"          # Successfully reconciled
    REJECTED = "rejected"              # Rejected, requires resubmission
    ESCALATED = "escalated"            # Escalated to higher authority
    CANCELLED = "cancelled"            # Cancelled by user
    ON_HOLD = "on_hold"                # Temporarily on hold
    COMPLETED = "completed"            # Fully completed and archived

class UserRole(Enum):
    """User roles for workflow permissions"""
    OPERATOR = "operator"              # Basic data entry and draft creation
    REVIEWER = "reviewer"              # Review and approve/reject drafts
    APPROVER = "approver"              # Manager-level approval
    ADMIN = "admin"                    # Full system access
    AUDITOR = "auditor"                # Read-only access for audits

class WorkflowAction(Enum):
    """Available workflow actions"""
    CREATE = "create"
    SUBMIT_FOR_REVIEW = "submit_for_review"
    REQUEST_CHANGES = "request_changes"
    APPROVE_REVIEW = "approve_review"
    REJECT_REVIEW = "reject_review"
    SUBMIT_FOR_APPROVAL = "submit_for_approval"
    APPROVE_FINAL = "approve_final"
    REJECT_FINAL = "reject_final"
    ESCALATE = "escalate"
    CANCEL = "cancel"
    PUT_ON_HOLD = "put_on_hold"
    RESUME = "resume"
    RECONCILE = "reconcile"
    COMPLETE = "complete"

@dataclass
class WorkflowTransition:
    """Defines a workflow transition between states"""
    from_state: ReconciliationState
    to_state: ReconciliationState
    action: WorkflowAction
    required_roles: Set[UserRole] = field(default_factory=set)
    conditions: List[str] = field(default_factory=list)
    auto_transition: bool = False
    timeout_minutes: Optional[int] = None

@dataclass
class WorkflowContext:
    """Context information for workflow operations"""
    user_id: int
    user_role: UserRole
    user_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    justification: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowResult:
    """Result of workflow operation"""
    success: bool
    new_state: Optional[ReconciliationState] = None
    message: str = ""
    audit_trail_entry: Optional[Dict[str, Any]] = None
    notifications: List[Dict[str, Any]] = field(default_factory=list)

class AdvancedReconciliationStateMachine:
    """
    Advanced state machine for reconciliation workflows
    Handles complex approval processes, role-based access, and audit trails
    """
    
    def __init__(self):
        self.transitions = self._define_workflow_transitions()
        self.state_permissions = self._define_state_permissions()
        self.timeout_config = self._define_timeout_config()
        self.escalation_rules = self._define_escalation_rules()
    
    def _define_workflow_transitions(self) -> List[WorkflowTransition]:
        """Define all possible workflow transitions"""
        return [
            # Draft to Review
            WorkflowTransition(
                from_state=ReconciliationState.DRAFT,
                to_state=ReconciliationState.PENDING_REVIEW,
                action=WorkflowAction.SUBMIT_FOR_REVIEW,
                required_roles={UserRole.OPERATOR, UserRole.REVIEWER}
            ),
            
            # Review transitions
            WorkflowTransition(
                from_state=ReconciliationState.PENDING_REVIEW,
                to_state=ReconciliationState.PENDING_APPROVAL,
                action=WorkflowAction.APPROVE_REVIEW,
                required_roles={UserRole.REVIEWER, UserRole.APPROVER}
            ),
            WorkflowTransition(
                from_state=ReconciliationState.PENDING_REVIEW,
                to_state=ReconciliationState.DRAFT,
                action=WorkflowAction.REQUEST_CHANGES,
                required_roles={UserRole.REVIEWER}
            ),
            WorkflowTransition(
                from_state=ReconciliationState.PENDING_REVIEW,
                to_state=ReconciliationState.REJECTED,
                action=WorkflowAction.REJECT_REVIEW,
                required_roles={UserRole.REVIEWER, UserRole.APPROVER}
            ),
            
            # Approval transitions
            WorkflowTransition(
                from_state=ReconciliationState.PENDING_APPROVAL,
                to_state=ReconciliationState.APPROVED,
                action=WorkflowAction.APPROVE_FINAL,
                required_roles={UserRole.APPROVER}
            ),
            WorkflowTransition(
                from_state=ReconciliationState.PENDING_APPROVAL,
                to_state=ReconciliationState.ESCALATED,
                action=WorkflowAction.ESCALATE,
                required_roles={UserRole.APPROVER}
            ),
            WorkflowTransition(
                from_state=ReconciliationState.PENDING_APPROVAL,
                to_state=ReconciliationState.REJECTED,
                action=WorkflowAction.REJECT_FINAL,
                required_roles={UserRole.APPROVER}
            ),
            
            # Final transitions
            WorkflowTransition(
                from_state=ReconciliationState.APPROVED,
                to_state=ReconciliationState.RECONCILED,
                action=WorkflowAction.RECONCILE,
                required_roles={UserRole.OPERATOR, UserRole.REVIEWER}
            ),
            WorkflowTransition(
                from_state=ReconciliationState.RECONCILED,
                to_state=ReconciliationState.COMPLETED,
                action=WorkflowAction.COMPLETE,
                required_roles={UserRole.ADMIN}
            ),
            
            # Hold/Resume transitions
            WorkflowTransition(
                from_state=ReconciliationState.PENDING_REVIEW,
                to_state=ReconciliationState.ON_HOLD,
                action=WorkflowAction.PUT_ON_HOLD,
                required_roles={UserRole.REVIEWER, UserRole.APPROVER}
            ),
            WorkflowTransition(
                from_state=ReconciliationState.PENDING_APPROVAL,
                to_state=ReconciliationState.ON_HOLD,
                action=WorkflowAction.PUT_ON_HOLD,
                required_roles={UserRole.APPROVER}
            ),
            WorkflowTransition(
                from_state=ReconciliationState.ON_HOLD,
                to_state=ReconciliationState.PENDING_REVIEW,
                action=WorkflowAction.RESUME,
                required_roles={UserRole.REVIEWER, UserRole.APPROVER}
            ),
            
            # Cancellation transitions
            WorkflowTransition(
                from_state=ReconciliationState.DRAFT,
                to_state=ReconciliationState.CANCELLED,
                action=WorkflowAction.CANCEL,
                required_roles={UserRole.OPERATOR, UserRole.REVIEWER}
            ),
            WorkflowTransition(
                from_state=ReconciliationState.PENDING_REVIEW,
                to_state=ReconciliationState.CANCELLED,
                action=WorkflowAction.CANCEL,
                required_roles={UserRole.REVIEWER, UserRole.APPROVER}
            ),
            
            # Auto-transitions (time-based)
            WorkflowTransition(
                from_state=ReconciliationState.PENDING_REVIEW,
                to_state=ReconciliationState.ESCALATED,
                action=WorkflowAction.ESCALATE,
                auto_transition=True,
                timeout_minutes=1440  # 24 hours
            ),
            WorkflowTransition(
                from_state=ReconciliationState.PENDING_APPROVAL,
                to_state=ReconciliationState.ESCALATED,
                action=WorkflowAction.ESCALATE,
                auto_transition=True,
                timeout_minutes=2880  # 48 hours
            )
        ]
    
    def _define_state_permissions(self) -> Dict[ReconciliationState, Set[UserRole]]:
        """Define permissions for each state"""
        return {
            ReconciliationState.DRAFT: {UserRole.OPERATOR, UserRole.REVIEWER, UserRole.APPROVER, UserRole.ADMIN},
            ReconciliationState.PENDING_REVIEW: {UserRole.REVIEWER, UserRole.APPROVER, UserRole.ADMIN},
            ReconciliationState.PENDING_APPROVAL: {UserRole.APPROVER, UserRole.ADMIN},
            ReconciliationState.APPROVED: {UserRole.OPERATOR, UserRole.REVIEWER, UserRole.APPROVER, UserRole.ADMIN},
            ReconciliationState.RECONCILED: {UserRole.OPERATOR, UserRole.REVIEWER, UserRole.APPROVER, UserRole.ADMIN, UserRole.AUDITOR},
            ReconciliationState.REJECTED: {UserRole.OPERATOR, UserRole.REVIEWER, UserRole.APPROVER, UserRole.ADMIN},
            ReconciliationState.ESCALATED: {UserRole.APPROVER, UserRole.ADMIN},
            ReconciliationState.CANCELLED: {UserRole.OPERATOR, UserRole.REVIEWER, UserRole.APPROVER, UserRole.ADMIN, UserRole.AUDITOR},
            ReconciliationState.ON_HOLD: {UserRole.REVIEWER, UserRole.APPROVER, UserRole.ADMIN},
            ReconciliationState.COMPLETED: {UserRole.OPERATOR, UserRole.REVIEWER, UserRole.APPROVER, UserRole.ADMIN, UserRole.AUDITOR}
        }
    
    def _define_timeout_config(self) -> Dict[ReconciliationState, int]:
        """Define timeout configuration for automatic transitions"""
        return {
            ReconciliationState.PENDING_REVIEW: 1440,    # 24 hours
            ReconciliationState.PENDING_APPROVAL: 2880,  # 48 hours
            ReconciliationState.ESCALATED: 7200,         # 5 days
            ReconciliationState.ON_HOLD: 10080          # 7 days
        }
    
    def _define_escalation_rules(self) -> Dict[ReconciliationState, Dict[str, Any]]:
        """Define escalation rules for different states"""
        return {
            ReconciliationState.PENDING_REVIEW: {
                'timeout_hours': 24,
                'escalate_to': UserRole.APPROVER,
                'notification_template': 'review_pending_escalation'
            },
            ReconciliationState.PENDING_APPROVAL: {
                'timeout_hours': 48,
                'escalate_to': UserRole.ADMIN,
                'notification_template': 'approval_pending_escalation'
            },
            ReconciliationState.ESCALATED: {
                'timeout_hours': 120,  # 5 days
                'escalate_to': UserRole.ADMIN,
                'notification_template': 'escalation_timeout'
            }
        }
    
    def execute_workflow_action(self, 
                               reconciliation_id: int, 
                               action: WorkflowAction, 
                               context: WorkflowContext) -> WorkflowResult:
        """
        Execute a workflow action on a reconciliation record
        """
        try:
            # Get current record
            record = ReconciliationRecord.query.get(reconciliation_id)
            if not record:
                raise ValidationError(f"Reconciliation record {reconciliation_id} not found")
            
            current_state = self._get_current_state(record)
            
            # Find matching transition
            transition = self._find_transition(current_state, action, context.user_role)
            if not transition:
                return WorkflowResult(
                    success=False,
                    message=f"Invalid action '{action.value}' from state '{current_state.value}' for role '{context.user_role.value}'"
                )
            
            # Validate transition conditions
            if not self._validate_transition_conditions(transition, record, context):
                return WorkflowResult(
                    success=False,
                    message=f"Transition conditions not met for action '{action.value}'"
                )
            
            # Execute transition
            new_record = self._execute_transition(record, transition, context)
            
            # Create audit trail entry
            audit_entry = self._create_audit_trail_entry(
                reconciliation_id, current_state, transition.to_state, 
                action, context, new_record
            )
            
            # Generate notifications
            notifications = self._generate_notifications(
                reconciliation_id, transition, context, new_record
            )
            
            # Check for auto-transitions
            self._schedule_auto_transitions(reconciliation_id, transition.to_state)
            
            logger.info(f"Workflow action '{action.value}' executed successfully "
                       f"for reconciliation {reconciliation_id}: {current_state.value} -> {transition.to_state.value}")
            
            return WorkflowResult(
                success=True,
                new_state=transition.to_state,
                message=f"Successfully transitioned to {transition.to_state.value}",
                audit_trail_entry=audit_entry,
                notifications=notifications
            )
            
        except Exception as e:
            logger.error(f"Error executing workflow action: {str(e)}")
            return WorkflowResult(
                success=False,
                message=f"Workflow execution failed: {str(e)}"
            )
    
    def _get_current_state(self, record: ReconciliationRecord) -> ReconciliationState:
        """Get current state from record"""
        # Map legacy status to new state
        status_mapping = {
            'pending': ReconciliationState.PENDING_REVIEW,
            'confirmed': ReconciliationState.RECONCILED,
            'rejected': ReconciliationState.REJECTED
        }
        
        # Check if record has new state field
        if hasattr(record, 'workflow_state') and record.workflow_state:
            try:
                return ReconciliationState(record.workflow_state)
            except ValueError:
                pass
        
        # Fallback to legacy status
        legacy_status = getattr(record, 'status', 'pending')
        return status_mapping.get(legacy_status, ReconciliationState.DRAFT)
    
    def _find_transition(self, 
                        current_state: ReconciliationState, 
                        action: WorkflowAction, 
                        user_role: UserRole) -> Optional[WorkflowTransition]:
        """Find matching transition for current state and action"""
        for transition in self.transitions:
            if (transition.from_state == current_state and 
                transition.action == action and
                user_role in transition.required_roles):
                return transition
        return None
    
    def _validate_transition_conditions(self, 
                                       transition: WorkflowTransition, 
                                       record: ReconciliationRecord, 
                                       context: WorkflowContext) -> bool:
        """Validate transition conditions"""
        for condition in transition.conditions:
            if condition == 'has_justification':
                if not context.justification:
                    return False
            elif condition == 'high_value':
                # Check if reconciliation amount is high
                if hasattr(record, 'bank_transaction') and record.bank_transaction:
                    if abs(record.bank_transaction.amount) > 10000:  # R$ 10,000 threshold
                        return False
            elif condition == 'requires_attachment':
                # Check if required attachments are present
                if not hasattr(record, 'attachments') or not record.attachments:
                    return False
        
        return True
    
    def _execute_transition(self, 
                           record: ReconciliationRecord, 
                           transition: WorkflowTransition, 
                           context: WorkflowContext) -> ReconciliationRecord:
        """Execute the transition and update record"""
        # Update workflow state
        if hasattr(record, 'workflow_state'):
            record.workflow_state = transition.to_state.value
        else:
            # Legacy fallback - update status field
            status_mapping = {
                ReconciliationState.PENDING_REVIEW: 'pending',
                ReconciliationState.RECONCILED: 'confirmed',
                ReconciliationState.REJECTED: 'rejected'
            }
            record.status = status_mapping.get(transition.to_state, 'pending')
        
        # Update timestamps
        now = datetime.now()
        if transition.to_state == ReconciliationState.PENDING_REVIEW:
            record.submitted_at = now
            record.submitted_by = context.user_id
        elif transition.to_state == ReconciliationState.APPROVED:
            record.approved_at = now
            record.approved_by = context.user_id
        elif transition.to_state == ReconciliationState.RECONCILED:
            record.reconciled_at = now
            record.reconciled_by = context.user_id
        
        # Store justification if provided
        if context.justification:
            if not hasattr(record, 'workflow_justifications'):
                record.workflow_justifications = []
            
            record.workflow_justifications.append({
                'user_id': context.user_id,
                'user_name': context.user_name,
                'action': transition.action.value,
                'justification': context.justification,
                'timestamp': now.isoformat()
            })
        
        # Update last modified
        record.last_modified_at = now
        record.last_modified_by = context.user_id
        
        db.session.commit()
        return record
    
    def _create_audit_trail_entry(self, 
                                 reconciliation_id: int,
                                 from_state: ReconciliationState,
                                 to_state: ReconciliationState,
                                 action: WorkflowAction,
                                 context: WorkflowContext,
                                 record: ReconciliationRecord) -> Dict[str, Any]:
        """Create audit trail entry for the transition"""
        audit_entry = {
            'reconciliation_id': reconciliation_id,
            'from_state': from_state.value,
            'to_state': to_state.value,
            'action': action.value,
            'user_id': context.user_id,
            'user_role': context.user_role.value,
            'user_name': context.user_name,
            'timestamp': context.timestamp.isoformat(),
            'justification': context.justification,
            'metadata': context.metadata
        }
        
        # Log audit entry
        audit_logger.log_workflow_transition(
            reconciliation_id, from_state.value, to_state.value,
            action.value, context.user_id, context.user_name
        )
        
        return audit_entry
    
    def _generate_notifications(self, 
                               reconciliation_id: int,
                               transition: WorkflowTransition,
                               context: WorkflowContext,
                               record: ReconciliationRecord) -> List[Dict[str, Any]]:
        """Generate notifications for workflow transitions"""
        notifications = []
        
        # Define notification rules
        notification_rules = {
            ReconciliationState.PENDING_REVIEW: {
                'recipients': [UserRole.REVIEWER],
                'template': 'pending_review_notification'
            },
            ReconciliationState.PENDING_APPROVAL: {
                'recipients': [UserRole.APPROVER],
                'template': 'pending_approval_notification'
            },
            ReconciliationState.ESCALATED: {
                'recipients': [UserRole.ADMIN],
                'template': 'escalation_notification'
            },
            ReconciliationState.REJECTED: {
                'recipients': [UserRole.OPERATOR],
                'template': 'rejection_notification'
            },
            ReconciliationState.APPROVED: {
                'recipients': [UserRole.OPERATOR],
                'template': 'approval_notification'
            }
        }
        
        # Generate notifications for new state
        if transition.to_state in notification_rules:
            rule = notification_rules[transition.to_state]
            
            for recipient_role in rule['recipients']:
                notification = {
                    'recipient_role': recipient_role.value,
                    'template': rule['template'],
                    'reconciliation_id': reconciliation_id,
                    'action_performed': transition.action.value,
                    'performed_by': context.user_name,
                    'timestamp': context.timestamp.isoformat(),
                    'priority': 'high' if transition.to_state in [ReconciliationState.ESCALATED, ReconciliationState.REJECTED] else 'normal'
                }
                notifications.append(notification)
        
        return notifications
    
    def _schedule_auto_transitions(self, reconciliation_id: int, new_state: ReconciliationState):
        """Schedule automatic transitions based on timeouts"""
        if new_state in self.timeout_config:
            timeout_minutes = self.timeout_config[new_state]
            # This would typically integrate with a job scheduler like Celery
            logger.info(f"Scheduling auto-transition for reconciliation {reconciliation_id} "
                       f"from {new_state.value} in {timeout_minutes} minutes")
    
    def get_available_actions(self, reconciliation_id: int, user_role: UserRole) -> List[WorkflowAction]:
        """Get available actions for a user on a reconciliation record"""
        try:
            record = ReconciliationRecord.query.get(reconciliation_id)
            if not record:
                return []
            
            current_state = self._get_current_state(record)
            available_actions = []
            
            for transition in self.transitions:
                if (transition.from_state == current_state and 
                    user_role in transition.required_roles):
                    available_actions.append(transition.action)
            
            return available_actions
            
        except Exception as e:
            logger.error(f"Error getting available actions: {str(e)}")
            return []
    
    def get_workflow_history(self, reconciliation_id: int) -> List[Dict[str, Any]]:
        """Get complete workflow history for a reconciliation record"""
        try:
            record = ReconciliationRecord.query.get(reconciliation_id)
            if not record:
                return []
            
            history = []
            
            # Get workflow justifications if available
            if hasattr(record, 'workflow_justifications') and record.workflow_justifications:
                for justification in record.workflow_justifications:
                    history.append({
                        'type': 'justification',
                        'user_id': justification['user_id'],
                        'user_name': justification['user_name'],
                        'action': justification['action'],
                        'justification': justification['justification'],
                        'timestamp': justification['timestamp']
                    })
            
            # Add state change history (this would typically be stored in a separate audit table)
            current_state = self._get_current_state(record)
            history.append({
                'type': 'state_change',
                'current_state': current_state.value,
                'timestamp': getattr(record, 'last_modified_at', record.created_at).isoformat() if hasattr(record, 'last_modified_at') else record.created_at.isoformat()
            })
            
            return sorted(history, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting workflow history: {str(e)}")
            return []
    
    def check_timeout_transitions(self) -> List[int]:
        """Check for and execute timeout-based transitions"""
        timeout_records = []
        now = datetime.now()
        
        try:
            # Find records that have timed out
            for state, timeout_minutes in self.timeout_config.items():
                timeout_threshold = now - timedelta(minutes=timeout_minutes)
                
                # Query records in this state that haven't been updated recently
                query = ReconciliationRecord.query.filter(
                    ReconciliationRecord.status == self._state_to_status(state),
                    ReconciliationRecord.last_modified_at < timeout_threshold
                )
                
                if hasattr(ReconciliationRecord, 'workflow_state'):
                    query = query.filter(ReconciliationRecord.workflow_state == state.value)
                
                timed_out_records = query.all()
                
                for record in timed_out_records:
                    timeout_records.append(record.id)
                    
                    # Execute auto-transition
                    context = WorkflowContext(
                        user_id=0,  # System user
                        user_role=UserRole.ADMIN,
                        user_name="System",
                        justification="Automatic timeout escalation"
                    )
                    
                    self.execute_workflow_action(
                        record.id, WorkflowAction.ESCALATE, context
                    )
            
            return timeout_records
            
        except Exception as e:
            logger.error(f"Error checking timeout transitions: {str(e)}")
            return []
    
    def _state_to_status(self, state: ReconciliationState) -> str:
        """Convert state to legacy status"""
        status_mapping = {
            ReconciliationState.PENDING_REVIEW: 'pending',
            ReconciliationState.RECONCILED: 'confirmed',
            ReconciliationState.REJECTED: 'rejected'
        }
        return status_mapping.get(state, 'pending')