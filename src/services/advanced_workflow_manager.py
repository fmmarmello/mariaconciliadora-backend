"""
Advanced Workflow Management Service

Integrates all workflow components into a unified management system:
- State machine orchestration
- RBAC permission management
- Audit trail tracking
- Escalation workflows
- Real-time processing
- Analytics and reporting
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from sqlalchemy import and_, or_, func
from src.models.transaction import db, ReconciliationRecord, Transaction, UploadHistory
from src.models.user import User
from src.services.workflow_state_machine import AdvancedReconciliationStateMachine, ReconciliationState, WorkflowAction, UserRole
from src.services.rbac_service import RBACService, Permission
from src.services.audit_trail_service import AuditTrailService
from src.services.escalation_service import EscalationService, EscalationLevel, EscalationTrigger
from src.services.realtime_processing_service import RealtimeProcessingService, ProcessingPriority, EventType
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class WorkflowPriority(Enum):
    """Workflow priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class WorkflowStatus(Enum):
    """Overall workflow status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"

@dataclass
class WorkflowInstance:
    """Represents a workflow instance"""
    id: str
    name: str
    description: str
    priority: WorkflowPriority
    status: WorkflowStatus
    created_at: datetime
    created_by: int
    updated_at: datetime
    updated_by: int
    reconciliation_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None

@dataclass
class WorkflowStep:
    """Represents a workflow step"""
    id: str
    workflow_id: str
    name: str
    description: str
    step_type: str
    order: int
    required: bool = True
    timeout_minutes: int = 60
    assignee_role: Optional[UserRole] = None
    assignee_user: Optional[int] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    completed_at: Optional[datetime] = None
    completed_by: Optional[int] = None
    notes: Optional[str] = None

@dataclass
class WorkflowAnalytics:
    """Workflow analytics data"""
    total_workflows: int
    active_workflows: int
    completed_workflows: int
    failed_workflows: int
    average_completion_time: float
    escalation_rate: float
    rejection_rate: float
    user_performance: Dict[int, Dict[str, Any]]
    workflow_type_stats: Dict[str, Dict[str, Any]]

class AdvancedWorkflowManager:
    """Advanced workflow management system"""
    
    def __init__(self):
        # Initialize all services
        self.state_machine = AdvancedReconciliationStateMachine()
        self.rbac_service = RBACService()
        self.audit_service = AuditTrailService()
        self.escalation_service = EscalationService()
        self.realtime_service = RealtimeProcessingService()
        
        # Workflow storage
        self.workflows: Dict[str, WorkflowInstance] = {}
        self.workflow_steps: Dict[str, List[WorkflowStep]] = {}
        
        # Configuration
        self.default_workflow_template = self._create_default_workflow_template()
        
        # Analytics
        self.analytics_cache = {}
        self.last_analytics_update = None
        
        logger.info("Advanced workflow manager initialized")
    
    def _create_default_workflow_template(self) -> Dict[str, Any]:
        """Create default workflow template"""
        return {
            "name": "Standard Reconciliation",
            "description": "Standard reconciliation workflow",
            "steps": [
                {
                    "name": "Data Upload",
                    "description": "Upload bank and company data",
                    "step_type": "upload",
                    "order": 1,
                    "required": True,
                    "timeout_minutes": 30,
                    "assignee_role": UserRole.OPERATOR,
                    "actions": ["upload_ofx", "upload_xlsx"]
                },
                {
                    "name": "Initial Matching",
                    "description": "AI-powered transaction matching",
                    "step_type": "processing",
                    "order": 2,
                    "required": True,
                    "timeout_minutes": 60,
                    "assignee_role": None,  # System step
                    "actions": ["auto_match", "categorize"]
                },
                {
                    "name": "Review",
                    "description": "Review matches and discrepancies",
                    "step_type": "review",
                    "order": 3,
                    "required": True,
                    "timeout_minutes": 1440,  # 24 hours
                    "assignee_role": UserRole.REVIEWER,
                    "actions": ["approve", "reject", "request_changes"]
                },
                {
                    "name": "Approval",
                    "description": "Manager approval for high-value items",
                    "step_type": "approval",
                    "order": 4,
                    "required": False,
                    "timeout_minutes": 2880,  # 48 hours
                    "assignee_role": UserRole.APPROVER,
                    "conditions": {"min_amount": 100000},
                    "actions": ["approve", "reject"]
                },
                {
                    "name": "Reconciliation",
                    "description": "Final reconciliation and reporting",
                    "step_type": "reconciliation",
                    "order": 5,
                    "required": True,
                    "timeout_minutes": 120,
                    "assignee_role": UserRole.OPERATOR,
                    "actions": ["reconcile", "generate_report"]
                }
            ]
        }
    
    def create_workflow(self, name: str, description: str, priority: WorkflowPriority,
                        created_by: int, reconciliation_ids: List[int] = None,
                        template: Dict[str, Any] = None) -> str:
        """Create a new workflow instance"""
        workflow_id = f"wf_{int(datetime.utcnow().timestamp())}_{hash(name) % 10000}"
        
        if template is None:
            template = self.default_workflow_template
        
        workflow = WorkflowInstance(
            id=workflow_id,
            name=name,
            description=description,
            priority=priority,
            status=WorkflowStatus.ACTIVE,
            created_at=datetime.utcnow(),
            created_by=created_by,
            updated_at=datetime.utcnow(),
            updated_by=created_by,
            reconciliation_ids=reconciliation_ids or [],
            metadata={
                "template_name": template.get("name", "Custom"),
                "total_steps": len(template.get("steps", []))
            }
        )
        
        # Create workflow steps
        steps = []
        for step_data in template.get("steps", []):
            step = WorkflowStep(
                id=f"step_{workflow_id}_{len(steps) + 1}",
                workflow_id=workflow_id,
                name=step_data["name"],
                description=step_data["description"],
                step_type=step_data["step_type"],
                order=step_data["order"],
                required=step_data.get("required", True),
                timeout_minutes=step_data.get("timeout_minutes", 60),
                assignee_role=step_data.get("assignee_role"),
                conditions=step_data.get("conditions", {}),
                actions=step_data.get("actions", [])
            )
            steps.append(step)
        
        # Store workflow
        self.workflows[workflow_id] = workflow
        self.workflow_steps[workflow_id] = steps
        
        # Log workflow creation
        self.audit_service.log_workflow_transition(
            reconciliation_id=0,  # System workflow
            from_state="none",
            to_state="created",
            action="create_workflow",
            user_id=created_by,
            user_name=f"User {created_by}",
            justification=f"Created workflow: {name}"
        )
        
        # Emit workflow created event
        self.realtime_service.emit_event(
            EventType.SYSTEM_ALERT,
            {
                "type": "workflow_created",
                "workflow_id": workflow_id,
                "workflow_name": name,
                "priority": priority.value,
                "created_by": created_by
            }
        )
        
        logger.info(f"Workflow created: {name} ({workflow_id})")
        return workflow_id
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowInstance]:
        """Get workflow by ID"""
        return self.workflows.get(workflow_id)
    
    def get_workflow_steps(self, workflow_id: str) -> List[WorkflowStep]:
        """Get workflow steps"""
        return self.workflow_steps.get(workflow_id, [])
    
    def update_workflow_status(self, workflow_id: str, status: WorkflowStatus, 
                              updated_by: int, notes: str = None) -> bool:
        """Update workflow status"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return False
        
        old_status = workflow.status
        workflow.status = status
        workflow.updated_at = datetime.utcnow()
        workflow.updated_by = updated_by
        
        # Log status change
        self.audit_service.log_workflow_transition(
            reconciliation_id=0,
            from_state=old_status.value,
            to_state=status.value,
            action="update_workflow_status",
            user_id=updated_by,
            user_name=f"User {updated_by}",
            justification=notes or f"Status changed to {status.value}"
        )
        
        # Emit status change event
        self.realtime_service.emit_event(
            EventType.SYSTEM_ALERT,
            {
                "type": "workflow_status_changed",
                "workflow_id": workflow_id,
                "old_status": old_status.value,
                "new_status": status.value,
                "updated_by": updated_by,
                "notes": notes
            }
        )
        
        logger.info(f"Workflow {workflow_id} status updated: {old_status.value} -> {status.value}")
        return True
    
    def complete_workflow_step(self, workflow_id: str, step_id: str, 
                              completed_by: int, notes: str = None) -> bool:
        """Complete a workflow step"""
        steps = self.workflow_steps.get(workflow_id, [])
        step = next((s for s in steps if s.id == step_id), None)
        
        if not step:
            return False
        
        step.completed_at = datetime.utcnow()
        step.completed_by = completed_by
        step.notes = notes
        
        # Log step completion
        self.audit_service.log_workflow_transition(
            reconciliation_id=0,
            from_state="in_progress",
            to_state="completed",
            action="complete_step",
            user_id=completed_by,
            user_name=f"User {completed_by}",
            justification=f"Completed step: {step.name}"
        )
        
        # Check if workflow is complete
        if self._is_workflow_complete(workflow_id):
            self.update_workflow_status(workflow_id, WorkflowStatus.COMPLETED, completed_by)
        
        # Emit step completion event
        self.realtime_service.emit_event(
            EventType.SYSTEM_ALERT,
            {
                "type": "workflow_step_completed",
                "workflow_id": workflow_id,
                "step_id": step_id,
                "step_name": step.name,
                "completed_by": completed_by
            }
        )
        
        logger.info(f"Workflow step completed: {step.name} ({step_id})")
        return True
    
    def _is_workflow_complete(self, workflow_id: str) -> bool:
        """Check if all required steps are completed"""
        steps = self.workflow_steps.get(workflow_id, [])
        required_steps = [s for s in steps if s.required]
        return all(step.completed_at is not None for step in required_steps)
    
    def assign_workflow_step(self, workflow_id: str, step_id: str, 
                            assignee_user: int, assigned_by: int) -> bool:
        """Assign a workflow step to a user"""
        steps = self.workflow_steps.get(workflow_id, [])
        step = next((s for s in steps if s.id == step_id), None)
        
        if not step:
            return False
        
        step.assignee_user = assignee_user
        
        # Log assignment
        self.audit_service.log_workflow_transition(
            reconciliation_id=0,
            from_state="unassigned",
            to_state="assigned",
            action="assign_step",
            user_id=assigned_by,
            user_name=f"User {assigned_by}",
            justification=f"Assigned step {step.name} to user {assignee_user}"
        )
        
        # Emit assignment event
        self.realtime_service.emit_event(
            EventType.SYSTEM_ALERT,
            {
                "type": "workflow_step_assigned",
                "workflow_id": workflow_id,
                "step_id": step_id,
                "step_name": step.name,
                "assignee_user": assignee_user,
                "assigned_by": assigned_by
            }
        )
        
        logger.info(f"Workflow step assigned: {step.name} to user {assignee_user}")
        return True
    
    def get_user_workflows(self, user_id: int, status: WorkflowStatus = None) -> List[WorkflowInstance]:
        """Get workflows for a specific user"""
        user_workflows = []
        
        for workflow in self.workflows.values():
            if status and workflow.status != status:
                continue
            
            # Check if user has any steps assigned
            steps = self.workflow_steps.get(workflow.id, [])
            user_steps = [s for s in steps if s.assignee_user == user_id]
            
            if user_steps:
                user_workflows.append(workflow)
        
        return user_workflows
    
    def get_workflow_analytics(self, force_refresh: bool = False) -> WorkflowAnalytics:
        """Get comprehensive workflow analytics"""
        current_time = datetime.utcnow()
        
        # Check if we need to refresh analytics
        if (not force_refresh and 
            self.last_analytics_update and 
            (current_time - self.last_analytics_update).total_seconds() < 300):  # 5 minutes
            return self.analytics_cache
        
        # Calculate analytics
        total_workflows = len(self.workflows)
        active_workflows = len([w for w in self.workflows.values() if w.status == WorkflowStatus.ACTIVE])
        completed_workflows = len([w for w in self.workflows.values() if w.status == WorkflowStatus.COMPLETED])
        failed_workflows = len([w for w in self.workflows.values() if w.status == WorkflowStatus.ERROR])
        
        # Calculate average completion time
        completed_wf = [w for w in self.workflows.values() if w.status == WorkflowStatus.COMPLETED]
        if completed_wf:
            completion_times = [(w.updated_at - w.created_at).total_seconds() for w in completed_wf]
            average_completion_time = sum(completion_times) / len(completion_times)
        else:
            average_completion_time = 0.0
        
        # Get escalation and rejection rates
        escalation_rate = self._calculate_escalation_rate()
        rejection_rate = self._calculate_rejection_rate()
        
        # Get user performance metrics
        user_performance = self._calculate_user_performance()
        
        # Get workflow type statistics
        workflow_type_stats = self._calculate_workflow_type_stats()
        
        analytics = WorkflowAnalytics(
            total_workflows=total_workflows,
            active_workflows=active_workflows,
            completed_workflows=completed_workflows,
            failed_workflows=failed_workflows,
            average_completion_time=average_completion_time,
            escalation_rate=escalation_rate,
            rejection_rate=rejection_rate,
            user_performance=user_performance,
            workflow_type_stats=workflow_type_stats
        )
        
        # Cache analytics
        self.analytics_cache = analytics
        self.last_analytics_update = current_time
        
        return analytics
    
    def _calculate_escalation_rate(self) -> float:
        """Calculate escalation rate"""
        # This would integrate with the escalation service
        # For now, return a placeholder
        return 0.05  # 5% escalation rate
    
    def _calculate_rejection_rate(self) -> float:
        """Calculate rejection rate"""
        # This would analyze reconciliation records
        # For now, return a placeholder
        return 0.02  # 2% rejection rate
    
    def _calculate_user_performance(self) -> Dict[int, Dict[str, Any]]:
        """Calculate user performance metrics"""
        user_performance = {}
        
        for workflow in self.workflows.values():
            steps = self.workflow_steps.get(workflow.id, [])
            
            for step in steps:
                if step.completed_by:
                    user_id = step.completed_by
                    
                    if user_id not in user_performance:
                        user_performance[user_id] = {
                            "completed_steps": 0,
                            "average_completion_time": 0,
                            "on_time_completion_rate": 0
                        }
                    
                    user_performance[user_id]["completed_steps"] += 1
        
        return user_performance
    
    def _calculate_workflow_type_stats(self) -> Dict[str, Dict[str, Any]]:
        """Calculate workflow type statistics"""
        workflow_type_stats = {}
        
        for workflow in self.workflows.values():
            template_name = workflow.metadata.get("template_name", "Unknown")
            
            if template_name not in workflow_type_stats:
                workflow_type_stats[template_name] = {
                    "total": 0,
                    "active": 0,
                    "completed": 0,
                    "failed": 0,
                    "average_completion_time": 0
                }
            
            workflow_type_stats[template_name]["total"] += 1
            
            if workflow.status == WorkflowStatus.ACTIVE:
                workflow_type_stats[template_name]["active"] += 1
            elif workflow.status == WorkflowStatus.COMPLETED:
                workflow_type_stats[template_name]["completed"] += 1
            elif workflow.status == WorkflowStatus.ERROR:
                workflow_type_stats[template_name]["failed"] += 1
        
        return workflow_type_stats
    
    def get_pending_tasks(self, user_id: int) -> List[Dict[str, Any]]:
        """Get pending tasks for a user"""
        pending_tasks = []
        
        for workflow in self.workflows.values():
            if workflow.status != WorkflowStatus.ACTIVE:
                continue
            
            steps = self.workflow_steps.get(workflow.id, [])
            
            for step in steps:
                if (step.completed_at is None and 
                    (step.assignee_user == user_id or 
                     (step.assignee_role and self._user_has_role(user_id, step.assignee_role)))):
                    
                    pending_tasks.append({
                        "workflow_id": workflow.id,
                        "workflow_name": workflow.name,
                        "step_id": step.id,
                        "step_name": step.name,
                        "step_type": step.step_type,
                        "priority": workflow.priority.value,
                        "assigned_at": workflow.created_at,
                        "timeout_minutes": step.timeout_minutes
                    })
        
        return pending_tasks
    
    def _user_has_role(self, user_id: int, role: UserRole) -> bool:
        """Check if user has a specific role"""
        # This would integrate with the RBAC service
        # For now, return a placeholder
        return True
    
    def monitor_workflow_timeouts(self) -> List[Dict[str, Any]]:
        """Monitor and report workflow timeouts"""
        current_time = datetime.utcnow()
        timeouts = []
        
        for workflow in self.workflows.values():
            if workflow.status != WorkflowStatus.ACTIVE:
                continue
            
            steps = self.workflow_steps.get(workflow.id, [])
            
            for step in steps:
                if step.completed_at is None:
                    step_age = (current_time - workflow.created_at).total_seconds() / 60
                    
                    if step_age > step.timeout_minutes:
                        timeouts.append({
                            "workflow_id": workflow.id,
                            "workflow_name": workflow.name,
                            "step_id": step.id,
                            "step_name": step.name,
                            "timeout_minutes": step.timeout_minutes,
                            "overdue_minutes": step_age - step.timeout_minutes,
                            "assignee": step.assignee_user or step.assignee_role
                        })
        
        return timeouts
    
    def generate_workflow_report(self, workflow_id: str) -> Dict[str, Any]:
        """Generate comprehensive workflow report"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return {}
        
        steps = self.workflow_steps.get(workflow_id, [])
        
        report = {
            "workflow_id": workflow_id,
            "workflow_name": workflow.name,
            "description": workflow.description,
            "status": workflow.status.value,
            "priority": workflow.priority.value,
            "created_at": workflow.created_at.isoformat(),
            "created_by": workflow.created_by,
            "updated_at": workflow.updated_at.isoformat(),
            "updated_by": workflow.updated_by,
            "reconciliation_count": len(workflow.reconciliation_ids),
            "steps": []
        }
        
        for step in steps:
            step_report = {
                "step_id": step.id,
                "name": step.name,
                "description": step.description,
                "type": step.step_type,
                "order": step.order,
                "required": step.required,
                "status": "completed" if step.completed_at else "pending",
                "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                "completed_by": step.completed_by,
                "assignee": step.assignee_user or (step.assignee_role.value if step.assignee_role else None),
                "timeout_minutes": step.timeout_minutes,
                "notes": step.notes
            }
            report["steps"].append(step_report)
        
        return report
    
    def cleanup_old_workflows(self, days_old: int = 30) -> int:
        """Clean up old completed workflows"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        workflows_to_remove = []
        
        for workflow_id, workflow in self.workflows.items():
            if (workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.CANCELLED] and
                workflow.updated_at < cutoff_date):
                workflows_to_remove.append(workflow_id)
        
        # Remove old workflows
        for workflow_id in workflows_to_remove:
            del self.workflows[workflow_id]
            if workflow_id in self.workflow_steps:
                del self.workflow_steps[workflow_id]
        
        logger.info(f"Cleaned up {len(workflows_to_remove)} old workflows")
        return len(workflows_to_remove)