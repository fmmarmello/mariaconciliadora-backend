"""
Role-Based Access Control (RBAC) System for Advanced Reconciliation Workflows
Provides granular permissions, user management, and access control
"""

from typing import Dict, List, Set, Optional, Any, Callable
from enum import Enum
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
from flask import g, request, jsonify
from src.models.user import db
from src.utils.logging_config import get_logger, get_audit_logger
from src.utils.exceptions import AuthorizationError, ValidationError
from src.services.workflow_state_machine import UserRole, ReconciliationState, WorkflowAction

logger = get_logger(__name__)
audit_logger = get_audit_logger()

class Permission(Enum):
    """System permissions"""
    # Transaction permissions
    VIEW_TRANSACTIONS = "view_transactions"
    CREATE_TRANSACTIONS = "create_transactions"
    EDIT_TRANSACTIONS = "edit_transactions"
    DELETE_TRANSACTIONS = "delete_transactions"
    
    # Reconciliation permissions
    VIEW_RECONCILIATION = "view_reconciliation"
    CREATE_RECONCILIATION = "create_reconciliation"
    EDIT_RECONCILIATION = "edit_reconciliation"
    DELETE_RECONCILIATION = "delete_reconciliation"
    APPROVE_RECONCILIATION = "approve_reconciliation"
    REJECT_RECONCILIATION = "reject_reconciliation"
    ESCALATE_RECONCILIATION = "escalate_reconciliation"
    
    # Workflow permissions
    MANAGE_WORKFLOW = "manage_workflow"
    VIEW_WORKFLOW_HISTORY = "view_workflow_history"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    
    # System permissions
    CONFIGURE_SYSTEM = "configure_system"
    VIEW_REPORTS = "view_reports"
    EXPORT_DATA = "export_data"
    
    # Financial permissions
    VIEW_FINANCIAL_DATA = "view_financial_data"
    EDIT_FINANCIAL_DATA = "edit_financial_data"
    APPROVE_FINANCIAL_DATA = "approve_financial_data"

class Resource(Enum):
    """System resources for access control"""
    TRANSACTIONS = "transactions"
    RECONCILIATION = "reconciliation"
    WORKFLOW = "workflow"
    USERS = "users"
    AUDIT_LOGS = "audit_logs"
    REPORTS = "reports"
    SYSTEM_CONFIG = "system_config"
    FINANCIAL_DATA = "financial_data"

class AccessLevel(Enum):
    """Access levels for resources"""
    NONE = 0          # No access
    READ = 1          # Read-only access
    WRITE = 2         # Read and write access
    EXECUTE = 3       # Read, write, and execute operations
    ADMIN = 4         # Full administrative access

@dataclass
class RolePermission:
    """Defines permissions for a role"""
    role: UserRole
    permissions: Set[Permission] = field(default_factory=set)
    resource_access: Dict[Resource, AccessLevel] = field(default_factory=dict)
    max_approval_amount: float = 0.0  # Maximum amount user can approve
    can_escalate: bool = False
    can_manage_users: bool = False
    can_view_audit_logs: bool = False

class RBACService:
    """
    Role-Based Access Control Service
    Manages user roles, permissions, and access control
    """
    
    def __init__(self):
        self.role_permissions = self._define_role_permissions()
        self.resource_hierarchy = self._define_resource_hierarchy()
        self.delegation_rules = self._define_delegation_rules()
    
    def _define_role_permissions(self) -> Dict[UserRole, RolePermission]:
        """Define permissions for each role"""
        return {
            UserRole.OPERATOR: RolePermission(
                role=UserRole.OPERATOR,
                permissions={
                    Permission.VIEW_TRANSACTIONS,
                    Permission.CREATE_TRANSACTIONS,
                    Permission.EDIT_TRANSACTIONS,
                    Permission.VIEW_RECONCILIATION,
                    Permission.CREATE_RECONCILIATION,
                    Permission.EDIT_RECONCILIATION,
                    Permission.VIEW_WORKFLOW_HISTORY,
                    Permission.VIEW_FINANCIAL_DATA,
                    Permission.VIEW_REPORTS,
                    Permission.EXPORT_DATA
                },
                resource_access={
                    Resource.TRANSACTIONS: AccessLevel.WRITE,
                    Resource.RECONCILIATION: AccessLevel.WRITE,
                    Resource.FINANCIAL_DATA: AccessLevel.READ,
                    Resource.REPORTS: AccessLevel.READ
                },
                max_approval_amount=5000.0,  # R$ 5,000
                can_escalate=False,
                can_manage_users=False,
                can_view_audit_logs=False
            ),
            
            UserRole.REVIEWER: RolePermission(
                role=UserRole.REVIEWER,
                permissions={
                    Permission.VIEW_TRANSACTIONS,
                    Permission.CREATE_TRANSACTIONS,
                    Permission.EDIT_TRANSACTIONS,
                    Permission.VIEW_RECONCILIATION,
                    Permission.CREATE_RECONCILIATION,
                    Permission.EDIT_RECONCILIATION,
                    Permission.APPROVE_RECONCILIATION,
                    Permission.REJECT_RECONCILIATION,
                    Permission.MANAGE_WORKFLOW,
                    Permission.VIEW_WORKFLOW_HISTORY,
                    Permission.VIEW_FINANCIAL_DATA,
                    Permission.EDIT_FINANCIAL_DATA,
                    Permission.VIEW_REPORTS,
                    Permission.EXPORT_DATA
                },
                resource_access={
                    Resource.TRANSACTIONS: AccessLevel.WRITE,
                    Resource.RECONCILIATION: AccessLevel.EXECUTE,
                    Resource.WORKFLOW: AccessLevel.WRITE,
                    Resource.FINANCIAL_DATA: AccessLevel.WRITE,
                    Resource.REPORTS: AccessLevel.READ
                },
                max_approval_amount=25000.0,  # R$ 25,000
                can_escalate=True,
                can_manage_users=False,
                can_view_audit_logs=True
            ),
            
            UserRole.APPROVER: RolePermission(
                role=UserRole.APPROVER,
                permissions={
                    Permission.VIEW_TRANSACTIONS,
                    Permission.CREATE_TRANSACTIONS,
                    Permission.EDIT_TRANSACTIONS,
                    Permission.VIEW_RECONCILIATION,
                    Permission.CREATE_RECONCILIATION,
                    Permission.EDIT_RECONCILIATION,
                    Permission.APPROVE_RECONCILIATION,
                    Permission.REJECT_RECONCILIATION,
                    Permission.ESCALATE_RECONCILIATION,
                    Permission.MANAGE_WORKFLOW,
                    Permission.VIEW_WORKFLOW_HISTORY,
                    Permission.MANAGE_USERS,
                    Permission.VIEW_AUDIT_LOGS,
                    Permission.VIEW_FINANCIAL_DATA,
                    Permission.EDIT_FINANCIAL_DATA,
                    Permission.APPROVE_FINANCIAL_DATA,
                    Permission.VIEW_REPORTS,
                    Permission.EXPORT_DATA
                },
                resource_access={
                    Resource.TRANSACTIONS: AccessLevel.EXECUTE,
                    Resource.RECONCILIATION: AccessLevel.EXECUTE,
                    Resource.WORKFLOW: AccessLevel.EXECUTE,
                    Resource.USERS: AccessLevel.READ,
                    Resource.AUDIT_LOGS: AccessLevel.READ,
                    Resource.FINANCIAL_DATA: AccessLevel.EXECUTE,
                    Resource.REPORTS: AccessLevel.READ
                },
                max_approval_amount=100000.0,  # R$ 100,000
                can_escalate=True,
                can_manage_users=True,
                can_view_audit_logs=True
            ),
            
            UserRole.ADMIN: RolePermission(
                role=UserRole.ADMIN,
                permissions=set(Permission),  # All permissions
                resource_access={
                    resource: AccessLevel.ADMIN for resource in Resource
                },
                max_approval_amount=float('inf'),  # Unlimited
                can_escalate=True,
                can_manage_users=True,
                can_view_audit_logs=True
            ),
            
            UserRole.AUDITOR: RolePermission(
                role=UserRole.AUDITOR,
                permissions={
                    Permission.VIEW_TRANSACTIONS,
                    Permission.VIEW_RECONCILIATION,
                    Permission.VIEW_WORKFLOW_HISTORY,
                    Permission.VIEW_AUDIT_LOGS,
                    Permission.VIEW_FINANCIAL_DATA,
                    Permission.VIEW_REPORTS,
                    Permission.EXPORT_DATA
                },
                resource_access={
                    Resource.TRANSACTIONS: AccessLevel.READ,
                    Resource.RECONCILIATION: AccessLevel.READ,
                    Resource.WORKFLOW: AccessLevel.READ,
                    Resource.AUDIT_LOGS: AccessLevel.READ,
                    Resource.FINANCIAL_DATA: AccessLevel.READ,
                    Resource.REPORTS: AccessLevel.READ
                },
                max_approval_amount=0.0,
                can_escalate=False,
                can_manage_users=False,
                can_view_audit_logs=True
            )
        }
    
    def _define_resource_hierarchy(self) -> Dict[Resource, List[Resource]]:
        """Define resource hierarchy for inheritance"""
        return {
            Resource.SYSTEM_CONFIG: [],  # Top level
            Resource.USERS: [Resource.SYSTEM_CONFIG],
            Resource.AUDIT_LOGS: [Resource.SYSTEM_CONFIG],
            Resource.RECONCILIATION: [Resource.TRANSACTIONS, Resource.FINANCIAL_DATA],
            Resource.WORKFLOW: [Resource.RECONCILIATION],
            Resource.REPORTS: [Resource.TRANSACTIONS, Resource.RECONCILIATION, Resource.FINANCIAL_DATA],
            Resource.TRANSACTIONS: [],
            Resource.FINANCIAL_DATA: []
        }
    
    def _define_delegation_rules(self) -> Dict[UserRole, List[UserRole]]:
        """Define delegation rules for temporary role assignment"""
        return {
            UserRole.APPROVER: [UserRole.REVIEWER],  # Approver can delegate to reviewer
            UserRole.REVIEWER: [UserRole.OPERATOR],   # Reviewer can delegate to operator
            UserRole.ADMIN: [UserRole.APPROVER, UserRole.REVIEWER]  # Admin can delegate to anyone
        }
    
    def check_permission(self, user_id: int, permission: Permission, 
                        resource: Optional[Resource] = None, 
                        context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if a user has a specific permission
        """
        try:
            # Get user role
            user_role = self.get_user_role(user_id)
            if not user_role:
                return False
            
            # Get role permissions
            role_permissions = self.role_permissions.get(user_role)
            if not role_permissions:
                return False
            
            # Check direct permission
            if permission not in role_permissions.permissions:
                return False
            
            # Check resource access if specified
            if resource:
                access_level = role_permissions.resource_access.get(resource, AccessLevel.NONE)
                if access_level == AccessLevel.NONE:
                    return False
                
                # Check context-specific rules
                if context and 'amount' in context:
                    max_amount = role_permissions.max_approval_amount
                    if context['amount'] > max_amount:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking permission: {str(e)}")
            return False
    
    def get_user_role(self, user_id: int) -> Optional[UserRole]:
        """Get user role from database"""
        try:
            # This would typically query a user_roles table
            # For now, return a default role
            return UserRole.OPERATOR
        except Exception as e:
            logger.error(f"Error getting user role: {str(e)}")
            return None
    
    def get_user_permissions(self, user_id: int) -> Set[Permission]:
        """Get all permissions for a user"""
        user_role = self.get_user_role(user_id)
        if not user_role:
            return set()
        
        role_permissions = self.role_permissions.get(user_role)
        return role_permissions.permissions.copy() if role_permissions else set()
    
    def get_resource_access(self, user_id: int, resource: Resource) -> AccessLevel:
        """Get access level for a specific resource"""
        user_role = self.get_user_role(user_id)
        if not user_role:
            return AccessLevel.NONE
        
        role_permissions = self.role_permissions.get(user_role)
        return role_permissions.resource_access.get(resource, AccessLevel.NONE) if role_permissions else AccessLevel.NONE
    
    def can_execute_workflow_action(self, user_id: int, action: WorkflowAction, 
                                   reconciliation_state: ReconciliationState,
                                   context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if user can execute a specific workflow action
        """
        try:
            user_role = self.get_user_role(user_id)
            if not user_role:
                return False
            
            # Map workflow actions to permissions
            action_permissions = {
                WorkflowAction.CREATE: Permission.CREATE_RECONCILIATION,
                WorkflowAction.SUBMIT_FOR_REVIEW: Permission.EDIT_RECONCILIATION,
                WorkflowAction.REQUEST_CHANGES: Permission.EDIT_RECONCILIATION,
                WorkflowAction.APPROVE_REVIEW: Permission.APPROVE_RECONCILIATION,
                WorkflowAction.REJECT_REVIEW: Permission.REJECT_RECONCILIATION,
                WorkflowAction.SUBMIT_FOR_APPROVAL: Permission.EDIT_RECONCILIATION,
                WorkflowAction.APPROVE_FINAL: Permission.APPROVE_RECONCILIATION,
                WorkflowAction.REJECT_FINAL: Permission.REJECT_RECONCILIATION,
                WorkflowAction.ESCALATE: Permission.ESCALATE_RECONCILIATION,
                WorkflowAction.CANCEL: Permission.EDIT_RECONCILIATION,
                WorkflowAction.PUT_ON_HOLD: Permission.MANAGE_WORKFLOW,
                WorkflowAction.RESUME: Permission.MANAGE_WORKFLOW,
                WorkflowAction.RECONCILE: Permission.EDIT_RECONCILIATION,
                WorkflowAction.COMPLETE: Permission.MANAGE_WORKFLOW
            }
            
            required_permission = action_permissions.get(action)
            if not required_permission:
                return False
            
            # Check base permission
            if not self.check_permission(user_id, required_permission, Resource.RECONCILIATION, context):
                return False
            
            # Check amount limits for approval actions
            if action in [WorkflowAction.APPROVE_REVIEW, WorkflowAction.APPROVE_FINAL]:
                if context and 'amount' in context:
                    role_permissions = self.role_permissions.get(user_role)
                    if role_permissions and context['amount'] > role_permissions.max_approval_amount:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking workflow action permission: {str(e)}")
            return False
    
    def require_permission(self, permission: Permission, resource: Optional[Resource] = None):
        """
        Decorator to require specific permission for endpoint access
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Get user from Flask g object
                user_id = getattr(g, 'user_id', None)
                if not user_id:
                    raise AuthorizationError("User not authenticated")
                
                # Check permission
                if not self.check_permission(user_id, permission, resource):
                    raise AuthorizationError(f"Insufficient permissions for {permission.value}")
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def require_role(self, *required_roles: UserRole):
        """
        Decorator to require specific role(s) for endpoint access
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                user_id = getattr(g, 'user_id', None)
                if not user_id:
                    raise AuthorizationError("User not authenticated")
                
                user_role = self.get_user_role(user_id)
                if not user_role or user_role not in required_roles:
                    raise AuthorizationError(f"Required role not found. Required: {[r.value for r in required_roles]}")
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def require_resource_access(self, resource: Resource, min_level: AccessLevel):
        """
        Decorator to require minimum access level for a resource
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                user_id = getattr(g, 'user_id', None)
                if not user_id:
                    raise AuthorizationError("User not authenticated")
                
                user_access = self.get_resource_access(user_id, resource)
                if user_access.value < min_level.value:
                    raise AuthorizationError(f"Insufficient access level for {resource.value}")
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def create_role_assignment(self, user_id: int, role: UserRole, 
                              assigned_by: int, expiration: Optional[datetime] = None,
                              reason: Optional[str] = None) -> bool:
        """
        Assign a role to a user
        """
        try:
            # Check if assigner has permission to assign roles
            assigner_role = self.get_user_role(assigned_by)
            if not assigner_role or not self.can_manage_users(assigned_by):
                raise AuthorizationError("User does not have permission to assign roles")
            
            # Check role hierarchy
            if not self._can_assign_role(assigner_role, role):
                raise AuthorizationError(f"Cannot assign role {role.value} from role {assigner_role.value}")
            
            # Create role assignment
            # This would typically create a record in user_roles table
            logger.info(f"Role {role.value} assigned to user {user_id} by user {assigned_by}")
            
            # Log audit trail
            audit_logger.log_role_assignment(user_id, role.value, assigned_by, reason)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating role assignment: {str(e)}")
            return False
    
    def can_manage_users(self, user_id: int) -> bool:
        """Check if user can manage other users"""
        user_role = self.get_user_role(user_id)
        if not user_role:
            return False
        
        role_permissions = self.role_permissions.get(user_role)
        return role_permissions.can_manage_users if role_permissions else False
    
    def can_view_audit_logs(self, user_id: int) -> bool:
        """Check if user can view audit logs"""
        user_role = self.get_user_role(user_id)
        if not user_role:
            return False
        
        role_permissions = self.role_permissions.get(user_role)
        return role_permissions.can_view_audit_logs if role_permissions else False
    
    def _can_assign_role(self, assigner_role: UserRole, target_role: UserRole) -> bool:
        """Check if assigner can assign target role"""
        if assigner_role == UserRole.ADMIN:
            return True
        
        if assigner_role == UserRole.APPROVER:
            return target_role in [UserRole.OPERATOR, UserRole.REVIEWER]
        
        if assigner_role == UserRole.REVIEWER:
            return target_role == UserRole.OPERATOR
        
        return False
    
    def get_user_delegations(self, user_id: int) -> List[Dict[str, Any]]:
        """Get active delegations for a user"""
        # This would query a delegations table
        return []
    
    def create_delegation(self, delegator_id: int, delegate_id: int, 
                         role: UserRole, expiration: datetime, 
                         reason: str) -> bool:
        """Create a temporary role delegation"""
        try:
            delegator_role = self.get_user_role(delegator_id)
            if not delegator_role:
                return False
            
            # Check if delegator can delegate to target role
            if role not in self.delegation_rules.get(delegator_role, []):
                return False
            
            # Create delegation
            # This would create a record in delegations table
            logger.info(f"Delegation created: user {delegator_id} delegated {role.value} to user {delegate_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating delegation: {str(e)}")
            return False
    
    def revoke_delegation(self, delegation_id: int, revoked_by: int) -> bool:
        """Revoke a delegation"""
        try:
            # This would update the delegations table
            logger.info(f"Delegation {delegation_id} revoked by user {revoked_by}")
            return True
        except Exception as e:
            logger.error(f"Error revoking delegation: {str(e)}")
            return False
    
    def get_permission_report(self) -> Dict[str, Any]:
        """Generate a comprehensive permission report"""
        report = {
            'roles': {},
            'permissions': {},
            'resource_access': {}
        }
        
        for role, permissions in self.role_permissions.items():
            report['roles'][role.value] = {
                'permissions': [p.value for p in permissions.permissions],
                'max_approval_amount': permissions.max_approval_amount,
                'can_escalate': permissions.can_escalate,
                'can_manage_users': permissions.can_manage_users,
                'can_view_audit_logs': permissions.can_view_audit_logs
            }
            
            for resource, access_level in permissions.resource_access.items():
                if resource.value not in report['resource_access']:
                    report['resource_access'][resource.value] = {}
                report['resource_access'][resource.value][role.value] = access_level.value
        
        for permission in Permission:
            report['permissions'][permission.value] = [
                role.value for role, perms in self.role_permissions.items()
                if permission in perms.permissions
            ]
        
        return report

# Global RBAC service instance
rbac_service = RBACService()