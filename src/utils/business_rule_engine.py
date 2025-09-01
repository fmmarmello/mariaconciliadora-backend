"""
Business Rule Engine for Maria Conciliadora system.

This module provides:
- Configurable business rules for financial transaction validation
- Amount range validation by category
- Temporal consistency rules
- Cross-field dependency validation
- Industry-specific validation rules
"""

import re
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from decimal import Decimal, InvalidOperation
from enum import Enum

from .validation_result import ValidationResult, ValidationSeverity, FieldValidationResult
from .logging_config import get_logger

logger = get_logger(__name__)


class RuleType(Enum):
    """Types of business rules."""
    AMOUNT_RANGE = "AMOUNT_RANGE"
    TEMPORAL_CONSISTENCY = "TEMPORAL_CONSISTENCY"
    CROSS_FIELD_DEPENDENCY = "CROSS_FIELD_DEPENDENCY"
    PATTERN_VALIDATION = "PATTERN_VALIDATION"
    CUSTOM_VALIDATION = "CUSTOM_VALIDATION"


class BusinessRule:
    """Represents a single business rule."""

    def __init__(self, rule_id: str, name: str, rule_type: RuleType,
                 description: str, severity: ValidationSeverity = ValidationSeverity.MEDIUM,
                 enabled: bool = True, metadata: Optional[Dict[str, Any]] = None):
        self.rule_id = rule_id
        self.name = name
        self.rule_type = rule_type
        self.description = description
        self.severity = severity
        self.enabled = enabled
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.last_modified = datetime.now()

    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data against this rule. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement validate method")

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'rule_type': self.rule_type.value,
            'description': self.description,
            'severity': self.severity.value,
            'enabled': self.enabled,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat()
        }


class AmountRangeRule(BusinessRule):
    """Rule for validating amount ranges by category."""

    def __init__(self, rule_id: str, name: str, category_field: str,
                 amount_field: str, min_amount: Optional[Decimal] = None,
                 max_amount: Optional[Decimal] = None, category_values: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(rule_id, name, RuleType.AMOUNT_RANGE,
                        f"Amount range validation for {category_field}", **kwargs)
        self.category_field = category_field
        self.amount_field = amount_field
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.category_values = category_values or []

    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult()

        # Check if this rule applies to the data
        if self.category_values:
            category_value = data.get(self.category_field, '').lower()
            if category_value not in [cat.lower() for cat in self.category_values]:
                return result  # Rule doesn't apply

        # Get amount value
        amount_value = data.get(self.amount_field)
        if amount_value is None:
            return result

        try:
            amount = Decimal(str(amount_value))

            # Check minimum amount
            if self.min_amount is not None and amount < self.min_amount:
                error_msg = f"Amount {amount} is below minimum allowed {self.min_amount}"
                result.add_error(error_msg, self.amount_field, self.severity)

            # Check maximum amount
            if self.max_amount is not None and amount > self.max_amount:
                error_msg = f"Amount {amount} exceeds maximum allowed {self.max_amount}"
                result.add_error(error_msg, self.amount_field, self.severity)

        except (ValueError, InvalidOperation):
            result.add_error(f"Invalid amount format: {amount_value}", self.amount_field)

        return result


class TemporalConsistencyRule(BusinessRule):
    """Rule for validating temporal consistency."""

    def __init__(self, rule_id: str, name: str, date_field: str,
                 max_future_days: int = 30, min_past_days: int = 365*10,
                 business_days_only: bool = False, **kwargs):
        super().__init__(rule_id, name, RuleType.TEMPORAL_CONSISTENCY,
                        f"Temporal consistency validation for {date_field}", **kwargs)
        self.date_field = date_field
        self.max_future_days = max_future_days
        self.min_past_days = min_past_days
        self.business_days_only = business_days_only

    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult()

        date_value = data.get(self.date_field)
        if date_value is None:
            return result

        try:
            if isinstance(date_value, str):
                parsed_date = datetime.fromisoformat(date_value).date()
            elif isinstance(date_value, datetime):
                parsed_date = date_value.date()
            elif isinstance(date_value, date):
                parsed_date = date_value
            else:
                result.add_error(f"Invalid date format for {self.date_field}", self.date_field)
                return result

            today = date.today()

            # Check future dates
            max_future_date = today + timedelta(days=self.max_future_days)
            if parsed_date > max_future_date:
                error_msg = f"Date {parsed_date} is too far in the future (max {self.max_future_days} days)"
                result.add_error(error_msg, self.date_field, self.severity)

            # Check past dates
            min_past_date = today - timedelta(days=self.min_past_days)
            if parsed_date < min_past_date:
                error_msg = f"Date {parsed_date} is too far in the past (max {self.min_past_days} days)"
                result.add_error(error_msg, self.date_field, self.severity)

            # Check business days if required
            if self.business_days_only and parsed_date <= today:
                if parsed_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    result.add_warning(f"Date {parsed_date} falls on a weekend", self.date_field)

        except (ValueError, TypeError):
            result.add_error(f"Invalid date format: {date_value}", self.date_field)

        return result


class CrossFieldDependencyRule(BusinessRule):
    """Rule for validating cross-field dependencies."""

    def __init__(self, rule_id: str, name: str, primary_field: str,
                 dependent_field: str, dependency_rules: List[Dict[str, Any]],
                 **kwargs):
        super().__init__(rule_id, name, RuleType.CROSS_FIELD_DEPENDENCY,
                        f"Cross-field dependency validation between {primary_field} and {dependent_field}", **kwargs)
        self.primary_field = primary_field
        self.dependent_field = dependent_field
        self.dependency_rules = dependency_rules

    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult()

        primary_value = data.get(self.primary_field)
        dependent_value = data.get(self.dependent_field)

        if primary_value is None:
            return result

        for rule in self.dependency_rules:
            if self._matches_rule(primary_value, rule):
                if not self._validate_dependent_field(dependent_value, rule):
                    error_msg = rule.get('error_message',
                                       f"Invalid {self.dependent_field} value for {self.primary_field} = {primary_value}")
                    result.add_error(error_msg, self.dependent_field, self.severity)
                break

        return result

    def _matches_rule(self, value: Any, rule: Dict[str, Any]) -> bool:
        """Check if the primary value matches the rule condition."""
        condition = rule.get('condition', {})
        operator = condition.get('operator', 'equals')
        expected_value = condition.get('value')

        if operator == 'equals':
            return str(value).lower() == str(expected_value).lower()
        elif operator == 'not_equals':
            return str(value).lower() != str(expected_value).lower()
        elif operator == 'contains':
            return str(expected_value).lower() in str(value).lower()
        elif operator == 'in':
            return value in expected_value if isinstance(expected_value, list) else False

        return False

    def _validate_dependent_field(self, value: Any, rule: Dict[str, Any]) -> bool:
        """Validate the dependent field value."""
        validation = rule.get('dependent_validation', {})
        validation_type = validation.get('type', 'required')

        if validation_type == 'required':
            return value is not None and str(value).strip() != ''
        elif validation_type == 'equals':
            return str(value).lower() == str(validation.get('value', '')).lower()
        elif validation_type == 'not_equals':
            return str(value).lower() != str(validation.get('value', '')).lower()
        elif validation_type == 'in':
            allowed_values = validation.get('values', [])
            return value in allowed_values
        elif validation_type == 'range':
            try:
                num_value = float(value)
                min_val = validation.get('min')
                max_val = validation.get('max')
                if min_val is not None and num_value < min_val:
                    return False
                if max_val is not None and num_value > max_val:
                    return False
                return True
            except (ValueError, TypeError):
                return False

        return True


class PatternValidationRule(BusinessRule):
    """Rule for validating patterns in field values."""

    def __init__(self, rule_id: str, name: str, field: str,
                 pattern: str, error_message: str = None, **kwargs):
        super().__init__(rule_id, name, RuleType.PATTERN_VALIDATION,
                        f"Pattern validation for {field}", **kwargs)
        self.field = field
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.error_message = error_message or f"Value does not match required pattern"

    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult()

        value = data.get(self.field)
        if value is None:
            return result

        if not self.pattern.match(str(value)):
            result.add_error(self.error_message, self.field, self.severity)

        return result


class CustomValidationRule(BusinessRule):
    """Rule for custom validation logic."""

    def __init__(self, rule_id: str, name: str, validation_function: Callable,
                 field: str = None, **kwargs):
        super().__init__(rule_id, name, RuleType.CUSTOM_VALIDATION,
                        f"Custom validation for {field or 'data'}", **kwargs)
        self.validation_function = validation_function
        self.field = field

    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        try:
            return self.validation_function(data, context or {})
        except Exception as e:
            result = ValidationResult()
            result.add_error(f"Custom validation error: {str(e)}", self.field)
            return result


class BusinessRuleEngine:
    """Engine for managing and executing business rules."""

    def __init__(self):
        self.rules: Dict[str, BusinessRule] = {}
        self.rule_groups: Dict[str, List[str]] = {}
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default business rules for financial data validation."""

        # Amount range rules for different categories
        self.add_rule(AmountRangeRule(
            rule_id="expense_amount_range",
            name="Expense Amount Range",
            category_field="transaction_type",
            amount_field="amount",
            min_amount=Decimal("0.01"),
            max_amount=Decimal("1000000.00"),
            category_values=["expense", "débito"]
        ))

        self.add_rule(AmountRangeRule(
            rule_id="income_amount_range",
            name="Income Amount Range",
            category_field="transaction_type",
            amount_field="amount",
            min_amount=Decimal("0.01"),
            max_amount=Decimal("5000000.00"),
            category_values=["income", "crédito"]
        ))

        # Temporal consistency rules
        self.add_rule(TemporalConsistencyRule(
            rule_id="transaction_date_consistency",
            name="Transaction Date Consistency",
            date_field="date",
            max_future_days=30,
            min_past_days=365*5  # 5 years
        ))

        # Cross-field dependency rules
        self.add_rule(CrossFieldDependencyRule(
            rule_id="expense_negative_amount",
            name="Expense Negative Amount",
            primary_field="transaction_type",
            dependent_field="amount",
            dependency_rules=[{
                'condition': {'operator': 'in', 'value': ['expense', 'débito']},
                'dependent_validation': {'type': 'range', 'max': 0},
                'error_message': 'Expense amounts should be negative'
            }]
        ))

        self.add_rule(CrossFieldDependencyRule(
            rule_id="income_positive_amount",
            name="Income Positive Amount",
            primary_field="transaction_type",
            dependent_field="amount",
            dependency_rules=[{
                'condition': {'operator': 'in', 'value': ['income', 'crédito']},
                'dependent_validation': {'type': 'range', 'min': 0},
                'error_message': 'Income amounts should be positive'
            }]
        ))

        # Pattern validation rules
        self.add_rule(PatternValidationRule(
            rule_id="bank_name_pattern",
            name="Bank Name Pattern",
            field="bank_name",
            pattern=r'^(itau|bradesco|santander|nubank|sicoob|caixa|bb|banco do brasil)$',
            error_message="Unknown or invalid bank name"
        ))

        # Group rules
        self.rule_groups['financial_transaction'] = [
            'expense_amount_range', 'income_amount_range',
            'transaction_date_consistency', 'expense_negative_amount',
            'income_positive_amount'
        ]

        self.rule_groups['company_financial'] = [
            'expense_amount_range', 'income_amount_range',
            'transaction_date_consistency'
        ]

    def add_rule(self, rule: BusinessRule):
        """Add a business rule to the engine."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added business rule: {rule.name} ({rule.rule_id})")

    def remove_rule(self, rule_id: str):
        """Remove a business rule from the engine."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed business rule: {rule_id}")

    def enable_rule(self, rule_id: str):
        """Enable a business rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            logger.info(f"Enabled business rule: {rule_id}")

    def disable_rule(self, rule_id: str):
        """Disable a business rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            logger.info(f"Disabled business rule: {rule_id}")

    def validate(self, data: Dict[str, Any], rule_group: str = None,
                context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data against business rules."""
        result = ValidationResult()
        result.set_validator_info("BusinessRuleEngine", "1.0")

        # Determine which rules to apply
        if rule_group and rule_group in self.rule_groups:
            rule_ids = self.rule_groups[rule_group]
        else:
            rule_ids = list(self.rules.keys())

        # Apply each rule
        for rule_id in rule_ids:
            if rule_id in self.rules:
                rule = self.rules[rule_id]
                if rule.enabled:
                    try:
                        rule_result = rule.validate(data, context)
                        result.merge(rule_result)
                    except Exception as e:
                        logger.error(f"Error executing rule {rule_id}: {str(e)}")
                        result.add_error(f"Rule execution error: {rule_id}")

        return result

    def get_rule(self, rule_id: str) -> Optional[BusinessRule]:
        """Get a specific rule by ID."""
        return self.rules.get(rule_id)

    def list_rules(self, rule_group: str = None) -> List[Dict[str, Any]]:
        """List all rules or rules in a specific group."""
        if rule_group and rule_group in self.rule_groups:
            rule_ids = self.rule_groups[rule_group]
        else:
            rule_ids = list(self.rules.keys())

        return [self.rules[rule_id].to_dict() for rule_id in rule_ids if rule_id in self.rules]

    def create_rule_group(self, group_name: str, rule_ids: List[str]):
        """Create a new rule group."""
        self.rule_groups[group_name] = rule_ids
        logger.info(f"Created rule group: {group_name} with {len(rule_ids)} rules")

    def add_rule_to_group(self, group_name: str, rule_id: str):
        """Add a rule to an existing group."""
        if group_name not in self.rule_groups:
            self.rule_groups[group_name] = []
        if rule_id not in self.rule_groups[group_name]:
            self.rule_groups[group_name].append(rule_id)
            logger.info(f"Added rule {rule_id} to group {group_name}")


# Global business rule engine instance
business_rule_engine = BusinessRuleEngine()