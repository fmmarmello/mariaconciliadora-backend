"""
Cross-Field Validation Engine for Maria Conciliadora system.

This module provides advanced cross-field validation with:
- Dependency validation between related fields
- Business rule enforcement across multiple fields
- Temporal consistency validation
- Referential integrity checks
- Conditional validation based on field values
"""

import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Set
from decimal import Decimal, InvalidOperation
from enum import Enum
import re

from .validation_result import ValidationResult, ValidationSeverity
from .logging_config import get_logger

logger = get_logger(__name__)


class DependencyType(Enum):
    """Types of field dependencies."""
    REQUIRED_IF = "REQUIRED_IF"
    EXCLUDED_IF = "EXCLUDED_IF"
    VALUE_DEPENDENT = "VALUE_DEPENDENT"
    RANGE_DEPENDENT = "RANGE_DEPENDENT"
    TEMPORAL_DEPENDENT = "TEMPORAL_DEPENDENT"
    REFERENTIAL_DEPENDENT = "REFERENTIAL_DEPENDENT"


class ValidationCondition:
    """Represents a validation condition for cross-field validation."""

    def __init__(self, field: str, operator: str, value: Any = None,
                 case_sensitive: bool = False):
        self.field = field
        self.operator = operator
        self.value = value
        self.case_sensitive = case_sensitive

    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate the condition against the data."""
        field_value = data.get(self.field)

        # Handle different operators
        if self.operator == 'equals':
            if field_value is None:
                return self.value is None
            if not self.case_sensitive and isinstance(field_value, str) and isinstance(self.value, str):
                return field_value.lower() == self.value.lower()
            return field_value == self.value

        elif self.operator == 'not_equals':
            if field_value is None:
                return self.value is not None
            if not self.case_sensitive and isinstance(field_value, str) and isinstance(self.value, str):
                return field_value.lower() != self.value.lower()
            return field_value != self.value

        elif self.operator == 'contains':
            if field_value is None:
                return False
            if isinstance(field_value, str) and isinstance(self.value, str):
                return self.value.lower() in field_value.lower() if not self.case_sensitive else self.value in field_value
            return False

        elif self.operator == 'in':
            if field_value is None:
                return False
            if isinstance(self.value, list):
                if not self.case_sensitive and isinstance(field_value, str):
                    return field_value.lower() in [v.lower() if isinstance(v, str) else v for v in self.value]
                return field_value in self.value
            return False

        elif self.operator == 'greater_than':
            if field_value is None:
                return False
            try:
                return float(field_value) > float(self.value)
            except (ValueError, TypeError):
                return False

        elif self.operator == 'less_than':
            if field_value is None:
                return False
            try:
                return float(field_value) < float(self.value)
            except (ValueError, TypeError):
                return False

        elif self.operator == 'is_null':
            return field_value is None

        elif self.operator == 'not_null':
            return field_value is not None

        elif self.operator == 'not_exists':
            return field_value is None

        return False


class CrossFieldRule:
    """Represents a cross-field validation rule."""

    def __init__(self, rule_id: str, name: str, dependency_type: DependencyType,
                 primary_field: str, dependent_fields: List[str],
                 conditions: List[ValidationCondition],
                 error_message: str, severity: ValidationSeverity = ValidationSeverity.MEDIUM,
                 enabled: bool = True):
        self.rule_id = rule_id
        self.name = name
        self.dependency_type = dependency_type
        self.primary_field = primary_field
        self.dependent_fields = dependent_fields
        self.conditions = conditions
        self.error_message = error_message
        self.severity = severity
        self.enabled = enabled

    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data against this cross-field rule."""
        result = ValidationResult()

        if not self.enabled:
            return result

        # Check if all conditions are met
        conditions_met = all(condition.evaluate(data) for condition in self.conditions)

        if not conditions_met:
            return result  # Rule doesn't apply

        # Apply validation based on dependency type
        if self.dependency_type == DependencyType.REQUIRED_IF:
            self._validate_required_if(data, result)
        elif self.dependency_type == DependencyType.EXCLUDED_IF:
            self._validate_excluded_if(data, result)
        elif self.dependency_type == DependencyType.VALUE_DEPENDENT:
            self._validate_value_dependent(data, result)
        elif self.dependency_type == DependencyType.RANGE_DEPENDENT:
            self._validate_range_dependent(data, result)
        elif self.dependency_type == DependencyType.TEMPORAL_DEPENDENT:
            self._validate_temporal_dependent(data, result)
        elif self.dependency_type == DependencyType.REFERENTIAL_DEPENDENT:
            self._validate_referential_dependent(data, result, context)

        return result

    def _validate_required_if(self, data: Dict[str, Any], result: ValidationResult):
        """Validate required fields when conditions are met."""
        primary_value = data.get(self.primary_field)
        for field in self.dependent_fields:
            value = data.get(field)
            if value is None or (isinstance(value, str) and value.strip() == ''):
                result.add_error(
                    self.error_message.format(field=field, primary=primary_value),
                    field,
                    self.severity
                )

    def _validate_excluded_if(self, data: Dict[str, Any], result: ValidationResult):
        """Validate excluded fields when conditions are met."""
        for field in self.dependent_fields:
            value = data.get(field)
            if value is not None and not (isinstance(value, str) and value.strip() == ''):
                result.add_error(
                    self.error_message.format(field=field, primary=self.primary_field),
                    field,
                    self.severity
                )

    def _validate_value_dependent(self, data: Dict[str, Any], result: ValidationResult):
        """Validate value dependencies between fields."""
        primary_value = data.get(self.primary_field)
        if primary_value is None:
            return

        for field in self.dependent_fields:
            dependent_value = data.get(field)
            if dependent_value is None:
                continue

            # Check for logical inconsistencies
            if self._has_value_inconsistency(primary_value, dependent_value, field):
                result.add_error(
                    self.error_message.format(
                        primary=self.primary_field,
                        dependent=field,
                        primary_value=primary_value,
                        dependent_value=dependent_value
                    ),
                    field,
                    self.severity
                )

    def _validate_range_dependent(self, data: Dict[str, Any], result: ValidationResult):
        """Validate range dependencies between numeric fields."""
        primary_value = data.get(self.primary_field)
        if primary_value is None:
            return

        try:
            primary_num = float(primary_value)

            for field in self.dependent_fields:
                dependent_value = data.get(field)
                if dependent_value is None:
                    continue

                try:
                    dependent_num = float(dependent_value)

                    # Check range relationships
                    if not self._validate_range_relationship(primary_num, dependent_num, field):
                        result.add_error(
                            self.error_message.format(
                                primary=self.primary_field,
                                dependent=field,
                                primary_value=primary_num,
                                dependent_value=dependent_num
                            ),
                            field,
                            self.severity
                        )

                except (ValueError, TypeError):
                    continue

        except (ValueError, TypeError):
            pass

    def _validate_temporal_dependent(self, data: Dict[str, Any], result: ValidationResult):
        """Validate temporal dependencies between date fields."""
        primary_date = self._parse_date(data.get(self.primary_field))
        if primary_date is None:
            return

        for field in self.dependent_fields:
            dependent_date = self._parse_date(data.get(field))
            if dependent_date is None:
                continue

            # Check temporal relationships
            if not self._validate_temporal_relationship(primary_date, dependent_date, field):
                result.add_error(
                    self.error_message.format(
                        primary=self.primary_field,
                        dependent=field,
                        primary_date=primary_date,
                        dependent_date=dependent_date
                    ),
                    field,
                    self.severity
                )

    def _validate_referential_dependent(self, data: Dict[str, Any], result: ValidationResult,
                                      context: Optional[Dict[str, Any]]):
        """Validate referential dependencies using context data."""
        if not context or 'reference_data' not in context:
            return

        reference_data = context['reference_data']
        primary_value = data.get(self.primary_field)

        if primary_value is None:
            return

        for field in self.dependent_fields:
            dependent_value = data.get(field)
            if dependent_value is None:
                continue

            # Check referential integrity
            if not self._validate_referential_integrity(
                primary_value, dependent_value, field, reference_data
            ):
                result.add_error(
                    self.error_message.format(
                        primary=self.primary_field,
                        dependent=field,
                        primary_value=primary_value,
                        dependent_value=dependent_value
                    ),
                    field,
                    self.severity
                )

    def _has_value_inconsistency(self, primary_value: Any, dependent_value: Any, field: str) -> bool:
        """Check for logical value inconsistencies."""
        # Transaction type and amount sign consistency
        if self.primary_field == 'transaction_type' and field == 'amount':
            try:
                amount = float(dependent_value)
                tx_type = str(primary_value).lower()

                if tx_type in ['debit', 'expense', 'saída'] and amount > 0:
                    return True
                elif tx_type in ['credit', 'income', 'entrada'] and amount < 0:
                    return True
            except (ValueError, TypeError):
                pass

        # Category and description consistency
        elif self.primary_field == 'category' and field == 'description':
            category = str(primary_value).lower()
            description = str(dependent_value).lower()

            # Check for obvious mismatches
            if 'salary' in category and 'expense' in description:
                return True
            elif 'rent' in category and 'salary' in description:
                return True

        return False

    def _validate_range_relationship(self, primary_num: float, dependent_num: float, field: str) -> bool:
        """Validate range relationships between numeric values."""
        # Amount and balance consistency
        if self.primary_field == 'amount' and field == 'balance':
            # Balance should be affected by amount
            return True  # This would need more context about previous balance

        # Tax amount should be proportional to base amount
        elif 'tax' in field.lower() and 'amount' in self.primary_field:
            tax_rate = abs(dependent_num / primary_num) if primary_num != 0 else 0
            return 0 <= tax_rate <= 0.5  # Reasonable tax rate range

        return True

    def _validate_temporal_relationship(self, primary_date: date, dependent_date: date, field: str) -> bool:
        """Validate temporal relationships between dates."""
        # Created date should be before or equal to updated date
        if self.primary_field == 'created_at' and field == 'updated_at':
            return primary_date <= dependent_date

        # Transaction date should be before reconciliation date
        elif self.primary_field == 'date' and 'reconciliation' in field:
            return primary_date <= dependent_date

        # Due date should be after transaction date
        elif self.primary_field == 'date' and 'due' in field:
            return primary_date <= dependent_date

        return True

    def _validate_referential_integrity(self, primary_value: Any, dependent_value: Any,
                                      field: str, reference_data: Dict[str, Any]) -> bool:
        """Validate referential integrity against reference data."""
        # Check if primary value exists in reference data
        if str(primary_value) not in reference_data:
            return False

        # Check if dependent value is valid for the primary value
        primary_refs = reference_data.get(str(primary_value), {})
        if field not in primary_refs:
            return True  # No specific validation defined

        valid_values = primary_refs[field]
        if isinstance(valid_values, list):
            return dependent_value in valid_values
        else:
            return dependent_value == valid_values

    def _parse_date(self, date_value: Any) -> Optional[date]:
        """Parse date from various formats."""
        if date_value is None:
            return None

        try:
            if isinstance(date_value, str):
                return datetime.fromisoformat(date_value).date()
            elif isinstance(date_value, datetime):
                return date_value.date()
            elif isinstance(date_value, date):
                return date_value
        except (ValueError, TypeError):
            pass

        return None


class CrossFieldValidationEngine:
    """
    Advanced cross-field validation engine for comprehensive field dependency validation.
    """

    def __init__(self):
        self.rules: Dict[str, CrossFieldRule] = {}
        self.rule_groups: Dict[str, List[str]] = {}
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default cross-field validation rules."""

        # Transaction type and amount consistency
        self.add_rule(CrossFieldRule(
            rule_id="transaction_type_amount_consistency",
            name="Transaction Type Amount Consistency",
            dependency_type=DependencyType.VALUE_DEPENDENT,
            primary_field="transaction_type",
            dependent_fields=["amount"],
            conditions=[
                ValidationCondition("transaction_type", "not_null"),
                ValidationCondition("amount", "not_null")
            ],
            error_message="Amount sign inconsistent with transaction type {primary_value}"
        ))

        # Date and balance consistency
        self.add_rule(CrossFieldRule(
            rule_id="date_balance_temporal_consistency",
            name="Date Balance Temporal Consistency",
            dependency_type=DependencyType.TEMPORAL_DEPENDENT,
            primary_field="date",
            dependent_fields=["balance"],
            conditions=[
                ValidationCondition("date", "not_null"),
                ValidationCondition("balance", "not_null")
            ],
            error_message="Balance date {dependent_date} inconsistent with transaction date {primary_date}"
        ))

        # Category and department consistency
        self.add_rule(CrossFieldRule(
            rule_id="category_department_consistency",
            name="Category Department Consistency",
            dependency_type=DependencyType.VALUE_DEPENDENT,
            primary_field="category",
            dependent_fields=["department"],
            conditions=[
                ValidationCondition("category", "not_null"),
                ValidationCondition("department", "not_null")
            ],
            error_message="Category '{primary_value}' unusual for department '{dependent_value}'"
        ))

        # Amount and tax consistency
        self.add_rule(CrossFieldRule(
            rule_id="amount_tax_range_consistency",
            name="Amount Tax Range Consistency",
            dependency_type=DependencyType.RANGE_DEPENDENT,
            primary_field="amount",
            dependent_fields=["tax_amount"],
            conditions=[
                ValidationCondition("amount", "not_null"),
                ValidationCondition("tax_amount", "not_null")
            ],
            error_message="Tax amount {dependent_value} unreasonable for base amount {primary_value}"
        ))

        # Required fields for specific transaction types
        self.add_rule(CrossFieldRule(
            rule_id="invoice_required_fields",
            name="Invoice Required Fields",
            dependency_type=DependencyType.REQUIRED_IF,
            primary_field="transaction_type",
            dependent_fields=["invoice_number", "supplier_name"],
            conditions=[
                ValidationCondition("transaction_type", "equals", "invoice")
            ],
            error_message="Field {field} is required for invoice transactions"
        ))

        # Excluded fields for specific transaction types
        self.add_rule(CrossFieldRule(
            rule_id="cash_excluded_fields",
            name="Cash Transaction Excluded Fields",
            dependency_type=DependencyType.EXCLUDED_IF,
            primary_field="transaction_type",
            dependent_fields=["check_number", "card_number"],
            conditions=[
                ValidationCondition("transaction_type", "equals", "cash")
            ],
            error_message="Field {field} should not be present for cash transactions"
        ))

        # Referential integrity for bank accounts
        self.add_rule(CrossFieldRule(
            rule_id="bank_account_referential_integrity",
            name="Bank Account Referential Integrity",
            dependency_type=DependencyType.REFERENTIAL_DEPENDENT,
            primary_field="bank_name",
            dependent_fields=["account_number", "branch_code"],
            conditions=[
                ValidationCondition("bank_name", "not_null"),
                ValidationCondition("account_number", "not_null")
            ],
            error_message="Account {dependent_value} not valid for bank {primary_value}"
        ))

        # Group rules
        self.rule_groups['financial_transaction'] = [
            'transaction_type_amount_consistency',
            'date_balance_temporal_consistency',
            'category_department_consistency',
            'amount_tax_range_consistency'
        ]

        self.rule_groups['invoice_validation'] = [
            'invoice_required_fields',
            'cash_excluded_fields'
        ]

        self.rule_groups['banking_validation'] = [
            'bank_account_referential_integrity'
        ]

    def add_rule(self, rule: CrossFieldRule):
        """Add a cross-field validation rule."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added cross-field validation rule: {rule.name} ({rule.rule_id})")

    def remove_rule(self, rule_id: str):
        """Remove a cross-field validation rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed cross-field validation rule: {rule_id}")

    def validate(self, data: Dict[str, Any], rule_group: str = None,
                context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate data against cross-field rules.

        Args:
            data: Data to validate
            rule_group: Specific rule group to use
            context: Additional context for validation

        Returns:
            Validation result
        """
        result = ValidationResult()
        result.set_validator_info("CrossFieldValidationEngine", "1.0")

        start_time = time.time()

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
                        logger.error(f"Error executing cross-field rule {rule_id}: {str(e)}")
                        result.add_error(f"Cross-field rule execution error: {rule_id}")

        # Set validation duration
        duration = time.time() - start_time
        result.set_validation_duration(duration * 1000)

        logger.info(f"Cross-field validation completed in {duration:.3f}s with {len(result.errors)} errors, {len(result.warnings)} warnings")

        return result

    def get_rule(self, rule_id: str) -> Optional[CrossFieldRule]:
        """Get a specific rule by ID."""
        return self.rules.get(rule_id)

    def list_rules(self, rule_group: str = None) -> List[Dict[str, Any]]:
        """List all rules or rules in a specific group."""
        if rule_group and rule_group in self.rule_groups:
            rule_ids = self.rule_groups[rule_group]
        else:
            rule_ids = list(self.rules.keys())

        rules_list = []
        for rule_id in rule_ids:
            if rule_id in self.rules:
                rule = self.rules[rule_id]
                rules_list.append({
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'dependency_type': rule.dependency_type.value,
                    'primary_field': rule.primary_field,
                    'dependent_fields': rule.dependent_fields,
                    'severity': rule.severity.value,
                    'enabled': rule.enabled
                })

        return rules_list

    def create_rule_group(self, group_name: str, rule_ids: List[str]):
        """Create a new rule group."""
        self.rule_groups[group_name] = rule_ids
        logger.info(f"Created cross-field rule group: {group_name} with {len(rule_ids)} rules")

    def enable_rule(self, rule_id: str):
        """Enable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            logger.info(f"Enabled cross-field validation rule: {rule_id}")

    def disable_rule(self, rule_id: str):
        """Disable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            logger.info(f"Disabled cross-field validation rule: {rule_id}")


# Global cross-field validation engine instance
cross_field_validation_engine = CrossFieldValidationEngine()