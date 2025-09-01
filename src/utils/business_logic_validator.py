"""
Business Logic Validator for Maria Conciliadora system.

This module provides domain-specific business rules validation with:
- Financial transaction business rules
- Amount range validation by transaction type
- Date consistency and business day validation
- Account balance and reconciliation rules
- Tax-related validation rules
"""

import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from decimal import Decimal, InvalidOperation
from enum import Enum
import re

from .validation_result import ValidationResult, ValidationSeverity
from .logging_config import get_logger

logger = get_logger(__name__)


class BusinessRuleType(Enum):
    """Types of business logic rules."""
    FINANCIAL_TRANSACTION = "FINANCIAL_TRANSACTION"
    AMOUNT_RANGE = "AMOUNT_RANGE"
    DATE_CONSISTENCY = "DATE_CONSISTENCY"
    ACCOUNT_BALANCE = "ACCOUNT_BALANCE"
    TAX_VALIDATION = "TAX_VALIDATION"
    RECONCILIATION = "RECONCILIATION"
    COMPLIANCE = "COMPLIANCE"


class BusinessLogicRule:
    """Represents a business logic validation rule."""

    def __init__(self, rule_id: str, name: str, rule_type: BusinessRuleType,
                 validation_function: Callable, description: str,
                 severity: ValidationSeverity = ValidationSeverity.MEDIUM,
                 enabled: bool = True, metadata: Optional[Dict[str, Any]] = None):
        self.rule_id = rule_id
        self.name = name
        self.rule_type = rule_type
        self.validation_function = validation_function
        self.description = description
        self.severity = severity
        self.enabled = enabled
        self.metadata = metadata or {}

    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data against this business logic rule."""
        if not self.enabled:
            return ValidationResult()

        try:
            return self.validation_function(data, context or {})
        except Exception as e:
            logger.error(f"Error executing business logic rule {self.rule_id}: {str(e)}")
            result = ValidationResult()
            result.add_error(f"Business logic rule execution error: {str(e)}")
            return result


class BusinessLogicValidator:
    """
    Domain-specific business logic validator for financial data validation.
    """

    def __init__(self):
        self.rules: Dict[str, BusinessLogicRule] = {}
        self.rule_groups: Dict[str, List[str]] = {}
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default business logic validation rules."""

        # Financial transaction rules
        self.add_rule(BusinessLogicRule(
            rule_id="transaction_amount_sign_consistency",
            name="Transaction Amount Sign Consistency",
            rule_type=BusinessRuleType.FINANCIAL_TRANSACTION,
            validation_function=self._validate_transaction_amount_sign,
            description="Validate that transaction amounts have correct signs based on type"
        ))

        self.add_rule(BusinessLogicRule(
            rule_id="transaction_amount_reasonable_range",
            name="Transaction Amount Reasonable Range",
            rule_type=BusinessRuleType.AMOUNT_RANGE,
            validation_function=self._validate_transaction_amount_range,
            description="Validate transaction amounts are within reasonable ranges"
        ))

        self.add_rule(BusinessLogicRule(
            rule_id="duplicate_transaction_detection",
            name="Duplicate Transaction Detection",
            rule_type=BusinessRuleType.FINANCIAL_TRANSACTION,
            validation_function=self._validate_duplicate_transaction,
            description="Detect potential duplicate transactions"
        ))

        # Date consistency rules
        self.add_rule(BusinessLogicRule(
            rule_id="business_day_validation",
            name="Business Day Validation",
            rule_type=BusinessRuleType.DATE_CONSISTENCY,
            validation_function=self._validate_business_day,
            description="Validate transactions occur on business days"
        ))

        self.add_rule(BusinessLogicRule(
            rule_id="transaction_date_sequence",
            name="Transaction Date Sequence",
            rule_type=BusinessRuleType.DATE_CONSISTENCY,
            validation_function=self._validate_transaction_date_sequence,
            description="Validate transaction dates are in logical sequence"
        ))

        # Account balance rules
        self.add_rule(BusinessLogicRule(
            rule_id="account_balance_consistency",
            name="Account Balance Consistency",
            rule_type=BusinessRuleType.ACCOUNT_BALANCE,
            validation_function=self._validate_account_balance_consistency,
            description="Validate account balance changes are consistent with transactions"
        ))

        self.add_rule(BusinessLogicRule(
            rule_id="balance_reconciliation_check",
            name="Balance Reconciliation Check",
            rule_type=BusinessRuleType.RECONCILIATION,
            validation_function=self._validate_balance_reconciliation,
            description="Check balance reconciliation accuracy"
        ))

        # Tax validation rules
        self.add_rule(BusinessLogicRule(
            rule_id="tax_amount_calculation",
            name="Tax Amount Calculation",
            rule_type=BusinessRuleType.TAX_VALIDATION,
            validation_function=self._validate_tax_amount_calculation,
            description="Validate tax amounts are correctly calculated"
        ))

        self.add_rule(BusinessLogicRule(
            rule_id="tax_exemption_validation",
            name="Tax Exemption Validation",
            rule_type=BusinessRuleType.TAX_VALIDATION,
            validation_function=self._validate_tax_exemption,
            description="Validate tax exemption eligibility and application"
        ))

        # Compliance rules
        self.add_rule(BusinessLogicRule(
            rule_id="suspicious_transaction_detection",
            name="Suspicious Transaction Detection",
            rule_type=BusinessRuleType.COMPLIANCE,
            validation_function=self._validate_suspicious_transaction,
            description="Detect potentially suspicious transactions"
        ))

        self.add_rule(BusinessLogicRule(
            rule_id="regulatory_reporting_threshold",
            name="Regulatory Reporting Threshold",
            rule_type=BusinessRuleType.COMPLIANCE,
            validation_function=self._validate_regulatory_threshold,
            description="Check transactions against regulatory reporting thresholds"
        ))

        # Group rules
        self.rule_groups['transaction_validation'] = [
            'transaction_amount_sign_consistency',
            'transaction_amount_reasonable_range',
            'duplicate_transaction_detection',
            'business_day_validation'
        ]

        self.rule_groups['reconciliation_validation'] = [
            'account_balance_consistency',
            'balance_reconciliation_check',
            'transaction_date_sequence'
        ]

        self.rule_groups['tax_compliance'] = [
            'tax_amount_calculation',
            'tax_exemption_validation'
        ]

        self.rule_groups['regulatory_compliance'] = [
            'suspicious_transaction_detection',
            'regulatory_reporting_threshold'
        ]

    def _validate_transaction_amount_sign(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate transaction amount signs based on transaction type."""
        result = ValidationResult()

        amount = data.get('amount')
        transaction_type = data.get('transaction_type', '').lower()

        if amount is None or not transaction_type:
            return result

        try:
            amount_val = float(amount)

            # Define expected signs for different transaction types
            debit_types = ['debit', 'expense', 'saída', 'withdrawal', 'payment']
            credit_types = ['credit', 'income', 'entrada', 'deposit', 'receipt']

            if transaction_type in debit_types and amount_val > 0:
                result.add_error(
                    f"Debit transaction should have negative amount, got {amount_val}",
                    'amount',
                    ValidationSeverity.HIGH
                )
            elif transaction_type in credit_types and amount_val < 0:
                result.add_error(
                    f"Credit transaction should have positive amount, got {amount_val}",
                    'amount',
                    ValidationSeverity.HIGH
                )

        except (ValueError, TypeError):
            result.add_error("Invalid amount format", 'amount')

        return result

    def _validate_transaction_amount_range(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate transaction amounts are within reasonable ranges."""
        result = ValidationResult()

        amount = data.get('amount')
        transaction_type = data.get('transaction_type', '').lower()
        currency = data.get('currency', 'BRL').upper()

        if amount is None:
            return result

        try:
            amount_val = abs(float(amount))

            # Define amount ranges by transaction type and currency
            ranges = {
                'BRL': {
                    'salary': (1000, 50000),
                    'rent': (500, 10000),
                    'utilities': (50, 2000),
                    'groceries': (20, 1000),
                    'transport': (10, 500),
                    'entertainment': (5, 1000),
                    'large_transaction': (10000, 1000000)
                },
                'USD': {
                    'salary': (500, 10000),
                    'rent': (200, 5000),
                    'utilities': (20, 1000),
                    'groceries': (10, 500),
                    'transport': (5, 200),
                    'entertainment': (2, 500),
                    'large_transaction': (5000, 500000)
                }
            }

            currency_ranges = ranges.get(currency, ranges['BRL'])

            # Check for unusually small amounts
            if amount_val < 0.01:
                result.add_warning("Very small transaction amount", 'amount')

            # Check for unusually large amounts
            if amount_val > 1000000:  # 1 million
                result.add_warning("Very large transaction amount - please verify", 'amount')

            # Type-specific validation
            if transaction_type in ['salary', 'rent', 'utilities'] and transaction_type in currency_ranges:
                min_amount, max_amount = currency_ranges[transaction_type]
                if amount_val < min_amount:
                    result.add_warning(f"Amount seems low for {transaction_type}", 'amount')
                elif amount_val > max_amount:
                    result.add_warning(f"Amount seems high for {transaction_type}", 'amount')

        except (ValueError, TypeError):
            result.add_error("Invalid amount format", 'amount')

        return result

    def _validate_duplicate_transaction(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Detect potential duplicate transactions."""
        result = ValidationResult()

        # This would typically check against a database of recent transactions
        # For now, we'll implement basic duplicate detection logic

        if not context or 'recent_transactions' not in context:
            return result

        recent_transactions = context['recent_transactions']
        current_amount = data.get('amount')
        current_date = data.get('date')
        current_description = data.get('description', '').lower()

        if not all([current_amount, current_date, current_description]):
            return result

        try:
            current_amount_val = float(current_amount)
            current_date_parsed = datetime.fromisoformat(current_date) if isinstance(current_date, str) else current_date

            for transaction in recent_transactions:
                # Check for exact duplicates
                if (abs(float(transaction.get('amount', 0)) - current_amount_val) < 0.01 and
                    transaction.get('description', '').lower() == current_description):

                    # Check if dates are very close (within 1 day)
                    trans_date = transaction.get('date')
                    if trans_date:
                        trans_date_parsed = (datetime.fromisoformat(trans_date) if isinstance(trans_date, str)
                                           else trans_date)

                        if abs((current_date_parsed - trans_date_parsed).days) <= 1:
                            result.add_warning(
                                "Potential duplicate transaction detected",
                                'general',
                                ValidationSeverity.MEDIUM
                            )
                            break

        except (ValueError, TypeError, AttributeError):
            pass

        return result

    def _validate_business_day(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate transactions occur on business days."""
        result = ValidationResult()

        transaction_date = data.get('date')
        if not transaction_date:
            return result

        try:
            if isinstance(transaction_date, str):
                parsed_date = datetime.fromisoformat(transaction_date).date()
            elif isinstance(transaction_date, datetime):
                parsed_date = transaction_date.date()
            else:
                parsed_date = transaction_date

            # Check if it's a weekend
            if parsed_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                result.add_warning(
                    f"Transaction date {parsed_date} falls on a weekend",
                    'date',
                    ValidationSeverity.LOW
                )

            # Check for holidays (simplified - would need a proper holiday calendar)
            brazilian_holidays = [
                (1, 1),   # New Year
                (4, 21),  # Tiradentes
                (5, 1),   # Labor Day
                (9, 7),   # Independence
                (10, 12), # Our Lady of Aparecida
                (11, 2),  # All Souls
                (11, 15), # Republic Proclamation
                (12, 25), # Christmas
            ]

            for month, day in brazilian_holidays:
                if parsed_date.month == month and parsed_date.day == day:
                    result.add_warning(
                        f"Transaction date {parsed_date} falls on a holiday",
                        'date',
                        ValidationSeverity.LOW
                    )
                    break

        except (ValueError, TypeError):
            result.add_error("Invalid date format", 'date')

        return result

    def _validate_transaction_date_sequence(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate transaction dates are in logical sequence."""
        result = ValidationResult()

        if not context or 'previous_transaction' not in context:
            return result

        previous_transaction = context['previous_transaction']
        current_date = data.get('date')
        previous_date = previous_transaction.get('date')

        if not current_date or not previous_date:
            return result

        try:
            current_parsed = (datetime.fromisoformat(current_date) if isinstance(current_date, str)
                            else current_date)
            previous_parsed = (datetime.fromisoformat(previous_date) if isinstance(previous_date, str)
                             else previous_date)

            # Check if current transaction is before previous transaction
            if current_parsed < previous_parsed:
                result.add_error(
                    "Transaction date is earlier than previous transaction",
                    'date',
                    ValidationSeverity.HIGH
                )

            # Check for large gaps (more than 90 days)
            days_diff = (current_parsed - previous_parsed).days
            if days_diff > 90:
                result.add_warning(
                    f"Large gap of {days_diff} days since last transaction",
                    'date',
                    ValidationSeverity.MEDIUM
                )

        except (ValueError, TypeError, AttributeError):
            pass

        return result

    def _validate_account_balance_consistency(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate account balance changes are consistent with transactions."""
        result = ValidationResult()

        amount = data.get('amount')
        balance = data.get('balance')
        previous_balance = context.get('previous_balance') if context else None

        if amount is None or balance is None:
            return result

        try:
            amount_val = float(amount)
            balance_val = float(balance)

            if previous_balance is not None:
                previous_balance_val = float(previous_balance)
                expected_balance = previous_balance_val + amount_val

                # Allow for small rounding differences
                if abs(expected_balance - balance_val) > 0.01:
                    result.add_error(
                        f"Balance inconsistency: expected {expected_balance}, got {balance_val}",
                        'balance',
                        ValidationSeverity.HIGH
                    )

        except (ValueError, TypeError):
            result.add_error("Invalid numeric format for balance calculation", 'balance')

        return result

    def _validate_balance_reconciliation(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Check balance reconciliation accuracy."""
        result = ValidationResult()

        if not context or 'reconciliation_data' not in context:
            return result

        reconciliation_data = context['reconciliation_data']
        book_balance = data.get('book_balance')
        bank_balance = reconciliation_data.get('bank_balance')
        outstanding_checks = reconciliation_data.get('outstanding_checks', 0)
        deposits_in_transit = reconciliation_data.get('deposits_in_transit', 0)

        if book_balance is None or bank_balance is None:
            return result

        try:
            book_val = float(book_balance)
            bank_val = float(bank_balance)
            outstanding_val = float(outstanding_checks)
            deposits_val = float(deposits_in_transit)

            # Calculate adjusted book balance
            adjusted_balance = book_val - outstanding_val + deposits_val

            # Check reconciliation
            difference = abs(adjusted_balance - bank_val)
            if difference > 0.01:  # Allow for small differences
                result.add_error(
                    f"Balance reconciliation difference: {difference}",
                    'book_balance',
                    ValidationSeverity.HIGH
                )

        except (ValueError, TypeError):
            result.add_error("Invalid numeric format for reconciliation", 'book_balance')

        return result

    def _validate_tax_amount_calculation(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate tax amounts are correctly calculated."""
        result = ValidationResult()

        amount = data.get('amount')
        tax_amount = data.get('tax_amount')
        tax_rate = data.get('tax_rate')

        if amount is None or tax_amount is None:
            return result

        try:
            amount_val = float(amount)
            tax_val = float(tax_amount)

            if tax_rate:
                # If tax rate is provided, verify calculation
                expected_tax = amount_val * float(tax_rate)
                if abs(expected_tax - tax_val) > 0.01:
                    result.add_error(
                        f"Tax amount mismatch: expected {expected_tax}, got {tax_val}",
                        'tax_amount',
                        ValidationSeverity.HIGH
                    )
            else:
                # Estimate reasonable tax rate
                if amount_val != 0:
                    calculated_rate = abs(tax_val / amount_val)
                    if calculated_rate > 0.5:  # More than 50% tax
                        result.add_warning(
                            f"Unusually high tax rate: {calculated_rate:.2%}",
                            'tax_amount',
                            ValidationSeverity.MEDIUM
                        )

        except (ValueError, TypeError):
            result.add_error("Invalid numeric format for tax calculation", 'tax_amount')

        return result

    def _validate_tax_exemption(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate tax exemption eligibility and application."""
        result = ValidationResult()

        transaction_type = data.get('transaction_type', '').lower()
        tax_exempt = data.get('tax_exempt', False)
        tax_amount = data.get('tax_amount', 0)

        # Certain transaction types might be tax-exempt
        exempt_types = ['salary', 'pension', 'social_security', 'charitable_donation']

        if transaction_type in exempt_types and not tax_exempt:
            result.add_warning(
                f"Transaction type '{transaction_type}' may be tax-exempt",
                'tax_exempt',
                ValidationSeverity.LOW
            )

        # If marked as tax-exempt, tax amount should be zero
        if tax_exempt and float(tax_amount or 0) > 0:
            result.add_error(
                "Tax-exempt transaction should have zero tax amount",
                'tax_amount',
                ValidationSeverity.HIGH
            )

        return result

    def _validate_suspicious_transaction(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Detect potentially suspicious transactions."""
        result = ValidationResult()

        amount = data.get('amount')
        description = data.get('description', '').lower()
        recipient = data.get('recipient', '').lower()

        if amount is None:
            return result

        try:
            amount_val = abs(float(amount))

            # Check for large round numbers
            if amount_val > 10000 and amount_val == int(amount_val):
                result.add_warning(
                    "Large round number transaction - please verify",
                    'amount',
                    ValidationSeverity.MEDIUM
                )

            # Check for suspicious descriptions
            suspicious_keywords = ['urgent', 'confidential', 'secret', 'cash', 'wire transfer']
            for keyword in suspicious_keywords:
                if keyword in description:
                    result.add_warning(
                        f"Suspicious keyword '{keyword}' in transaction description",
                        'description',
                        ValidationSeverity.LOW
                    )

            # Check for unusual timing (if context available)
            if context and 'transaction_hour' in context:
                hour = context['transaction_hour']
                if hour < 6 or hour > 22:  # Outside normal business hours
                    result.add_warning(
                        "Transaction outside normal business hours",
                        'date',
                        ValidationSeverity.LOW
                    )

        except (ValueError, TypeError):
            pass

        return result

    def _validate_regulatory_threshold(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Check transactions against regulatory reporting thresholds."""
        result = ValidationResult()

        amount = data.get('amount')
        if amount is None:
            return result

        try:
            amount_val = abs(float(amount))

            # Brazilian regulatory thresholds (simplified)
            thresholds = {
                'bacen_reporting': 10000,  # Banco Central reporting threshold
                'receita_federal': 30000,  # Federal Revenue Service
                'coaf_suspicious': 50000,  # COAF suspicious activity
            }

            for threshold_name, threshold_value in thresholds.items():
                if amount_val >= threshold_value:
                    result.add_warning(
                        f"Transaction exceeds {threshold_name} threshold ({threshold_value})",
                        'amount',
                        ValidationSeverity.MEDIUM
                    )

        except (ValueError, TypeError):
            pass

        return result

    def add_rule(self, rule: BusinessLogicRule):
        """Add a business logic validation rule."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added business logic rule: {rule.name} ({rule.rule_id})")

    def remove_rule(self, rule_id: str):
        """Remove a business logic validation rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed business logic rule: {rule_id}")

    def validate(self, data: Dict[str, Any], rule_group: str = None,
                context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate data against business logic rules.

        Args:
            data: Data to validate
            rule_group: Specific rule group to use
            context: Additional context for validation

        Returns:
            Validation result
        """
        result = ValidationResult()
        result.set_validator_info("BusinessLogicValidator", "1.0")

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
                        logger.error(f"Error executing business logic rule {rule_id}: {str(e)}")
                        result.add_error(f"Business logic rule execution error: {rule_id}")

        # Set validation duration
        duration = time.time() - start_time
        result.set_validation_duration(duration * 1000)

        logger.info(f"Business logic validation completed in {duration:.3f}s with {len(result.errors)} errors, {len(result.warnings)} warnings")

        return result

    def get_rule(self, rule_id: str) -> Optional[BusinessLogicRule]:
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
                    'rule_type': rule.rule_type.value,
                    'description': rule.description,
                    'severity': rule.severity.value,
                    'enabled': rule.enabled
                })

        return rules_list

    def create_rule_group(self, group_name: str, rule_ids: List[str]):
        """Create a new rule group."""
        self.rule_groups[group_name] = rule_ids
        logger.info(f"Created business logic rule group: {group_name} with {len(rule_ids)} rules")

    def enable_rule(self, rule_id: str):
        """Enable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            logger.info(f"Enabled business logic rule: {rule_id}")

    def disable_rule(self, rule_id: str):
        """Disable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            logger.info(f"Disabled business logic rule: {rule_id}")


# Global business logic validator instance
business_logic_validator = BusinessLogicValidator()