"""
Financial Business Rules for Maria Conciliadora system.

This module provides comprehensive financial validation with:
- Transaction type and amount consistency
- Bank-specific validation rules
- Currency and exchange rate validation
- Regulatory compliance validation
- Industry-specific business rules
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


class FinancialRuleType(Enum):
    """Types of financial business rules."""
    TRANSACTION_CONSISTENCY = "TRANSACTION_CONSISTENCY"
    BANK_SPECIFIC = "BANK_SPECIFIC"
    CURRENCY_EXCHANGE = "CURRENCY_EXCHANGE"
    REGULATORY_COMPLIANCE = "REGULATORY_COMPLIANCE"
    INDUSTRY_SPECIFIC = "INDUSTRY_SPECIFIC"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"


class FinancialBusinessRule:
    """Represents a financial business rule."""

    def __init__(self, rule_id: str, name: str, rule_type: FinancialRuleType,
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
        """Validate data against this financial business rule."""
        if not self.enabled:
            return ValidationResult()

        try:
            return self.validation_function(data, context or {})
        except Exception as e:
            logger.error(f"Error executing financial rule {self.rule_id}: {str(e)}")
            result = ValidationResult()
            result.add_error(f"Financial rule execution error: {str(e)}")
            return result


class FinancialBusinessRules:
    """
    Comprehensive financial business rules validator for transaction processing.
    """

    def __init__(self):
        self.rules: Dict[str, FinancialBusinessRule] = {}
        self.rule_groups: Dict[str, List[str]] = {}
        self.bank_rules: Dict[str, Dict[str, Any]] = {}
        self.currency_rates: Dict[str, Dict[str, float]] = {}
        self._load_default_rules()
        self._load_bank_specific_rules()
        self._load_currency_rates()

    def _load_default_rules(self):
        """Load default financial business rules."""

        # Transaction consistency rules
        self.add_rule(FinancialBusinessRule(
            rule_id="transaction_type_amount_consistency",
            name="Transaction Type Amount Consistency",
            rule_type=FinancialRuleType.TRANSACTION_CONSISTENCY,
            validation_function=self._validate_transaction_type_amount,
            description="Validate transaction type and amount consistency"
        ))

        self.add_rule(FinancialBusinessRule(
            rule_id="transaction_category_validation",
            name="Transaction Category Validation",
            rule_type=FinancialRuleType.TRANSACTION_CONSISTENCY,
            validation_function=self._validate_transaction_category,
            description="Validate transaction categories and subcategories"
        ))

        # Bank-specific rules
        self.add_rule(FinancialBusinessRule(
            rule_id="bank_specific_format_validation",
            name="Bank Specific Format Validation",
            rule_type=FinancialRuleType.BANK_SPECIFIC,
            validation_function=self._validate_bank_specific_format,
            description="Validate bank-specific transaction formats"
        ))

        self.add_rule(FinancialBusinessRule(
            rule_id="bank_transaction_limits",
            name="Bank Transaction Limits",
            rule_type=FinancialRuleType.BANK_SPECIFIC,
            validation_function=self._validate_bank_transaction_limits,
            description="Validate transaction limits for specific banks"
        ))

        # Currency and exchange rate rules
        self.add_rule(FinancialBusinessRule(
            rule_id="currency_exchange_validation",
            name="Currency Exchange Validation",
            rule_type=FinancialRuleType.CURRENCY_EXCHANGE,
            validation_function=self._validate_currency_exchange,
            description="Validate currency exchange rates and conversions"
        ))

        self.add_rule(FinancialBusinessRule(
            rule_id="multi_currency_transaction",
            name="Multi Currency Transaction",
            rule_type=FinancialRuleType.CURRENCY_EXCHANGE,
            validation_function=self._validate_multi_currency_transaction,
            description="Validate multi-currency transaction consistency"
        ))

        # Regulatory compliance rules
        self.add_rule(FinancialBusinessRule(
            rule_id="aml_compliance_check",
            name="AML Compliance Check",
            rule_type=FinancialRuleType.REGULATORY_COMPLIANCE,
            validation_function=self._validate_aml_compliance,
            description="Anti-Money Laundering compliance validation"
        ))

        self.add_rule(FinancialBusinessRule(
            rule_id="tax_reporting_threshold",
            name="Tax Reporting Threshold",
            rule_type=FinancialRuleType.REGULATORY_COMPLIANCE,
            validation_function=self._validate_tax_reporting_threshold,
            description="Validate tax reporting requirements"
        ))

        # Industry-specific rules
        self.add_rule(FinancialBusinessRule(
            rule_id="retail_transaction_patterns",
            name="Retail Transaction Patterns",
            rule_type=FinancialRuleType.INDUSTRY_SPECIFIC,
            validation_function=self._validate_retail_transaction_patterns,
            description="Validate retail industry transaction patterns"
        ))

        self.add_rule(FinancialBusinessRule(
            rule_id="corporate_payment_validation",
            name="Corporate Payment Validation",
            rule_type=FinancialRuleType.INDUSTRY_SPECIFIC,
            validation_function=self._validate_corporate_payment,
            description="Validate corporate payment transaction rules"
        ))

        # Risk assessment rules
        self.add_rule(FinancialBusinessRule(
            rule_id="transaction_risk_scoring",
            name="Transaction Risk Scoring",
            rule_type=FinancialRuleType.RISK_ASSESSMENT,
            validation_function=self._validate_transaction_risk_scoring,
            description="Score transactions for risk assessment"
        ))

        self.add_rule(FinancialBusinessRule(
            rule_id="velocity_checks",
            name="Velocity Checks",
            rule_type=FinancialRuleType.RISK_ASSESSMENT,
            validation_function=self._validate_velocity_checks,
            description="Validate transaction velocity patterns"
        ))

        # Group rules
        self.rule_groups['transaction_processing'] = [
            'transaction_type_amount_consistency',
            'transaction_category_validation',
            'bank_specific_format_validation'
        ]

        self.rule_groups['compliance_validation'] = [
            'aml_compliance_check',
            'tax_reporting_threshold',
            'bank_transaction_limits'
        ]

        self.rule_groups['currency_validation'] = [
            'currency_exchange_validation',
            'multi_currency_transaction'
        ]

        self.rule_groups['risk_assessment'] = [
            'transaction_risk_scoring',
            'velocity_checks'
        ]

    def _load_bank_specific_rules(self):
        """Load bank-specific validation rules."""

        # Brazilian banks
        self.bank_rules = {
            'itau': {
                'account_format': r'^\d{4}-\d{1}$',
                'transaction_codes': ['DEB', 'CRE', 'PIX', 'TED', 'DOC'],
                'daily_limit': 50000,
                'max_transaction': 10000,
                'business_hours': (6, 22)
            },
            'bradesco': {
                'account_format': r'^\d{7}-\d{1}$',
                'transaction_codes': ['DB', 'CR', 'TRANSF', 'PAG'],
                'daily_limit': 100000,
                'max_transaction': 25000,
                'business_hours': (6, 22)
            },
            'santander': {
                'account_format': r'^\d{8}-\d{1}$',
                'transaction_codes': ['DEBITO', 'CREDITO', 'TRANSFERENCIA'],
                'daily_limit': 75000,
                'max_transaction': 15000,
                'business_hours': (7, 21)
            },
            'nubank': {
                'account_format': r'^\d{10}$',
                'transaction_codes': ['DEBIT', 'CREDIT', 'PIX'],
                'daily_limit': 25000,
                'max_transaction': 5000,
                'business_hours': (0, 24)  # 24/7
            }
        }

    def _load_currency_rates(self):
        """Load currency exchange rates."""
        # Simplified exchange rates (in production, these would be fetched from an API)
        self.currency_rates = {
            'USD': {'BRL': 5.20, 'EUR': 0.85, 'GBP': 0.73},
            'EUR': {'BRL': 6.10, 'USD': 1.18, 'GBP': 0.86},
            'GBP': {'BRL': 7.10, 'USD': 1.37, 'EUR': 1.16},
            'BRL': {'USD': 0.19, 'EUR': 0.16, 'GBP': 0.14}
        }

    def _validate_transaction_type_amount(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate transaction type and amount consistency."""
        result = ValidationResult()

        amount = data.get('amount')
        transaction_type = data.get('transaction_type', '').lower()
        operation_type = data.get('operation_type', '').lower()

        if amount is None:
            return result

        try:
            amount_val = float(amount)

            # Enhanced transaction type validation
            type_amount_rules = {
                'debit': {'expected_sign': -1, 'max_amount': 100000},
                'credit': {'expected_sign': 1, 'max_amount': 500000},
                'transfer': {'expected_sign': -1, 'max_amount': 100000},
                'payment': {'expected_sign': -1, 'max_amount': 50000},
                'deposit': {'expected_sign': 1, 'max_amount': 100000},
                'withdrawal': {'expected_sign': -1, 'max_amount': 10000},
                'fee': {'expected_sign': -1, 'max_amount': 1000},
                'interest': {'expected_sign': 1, 'max_amount': 10000},
                'chargeback': {'expected_sign': 1, 'max_amount': 50000},
                'refund': {'expected_sign': 1, 'max_amount': 100000}
            }

            if transaction_type in type_amount_rules:
                rules = type_amount_rules[transaction_type]

                # Check sign consistency
                expected_sign = rules['expected_sign']
                actual_sign = 1 if amount_val > 0 else -1

                if expected_sign != actual_sign:
                    result.add_error(
                        f"{transaction_type.title()} transactions should have {'positive' if expected_sign > 0 else 'negative'} amounts",
                        'amount',
                        ValidationSeverity.HIGH
                    )

                # Check amount limits
                max_amount = rules['max_amount']
                if abs(amount_val) > max_amount:
                    result.add_warning(
                        f"Amount exceeds typical limit for {transaction_type} ({max_amount})",
                        'amount',
                        ValidationSeverity.MEDIUM
                    )

            # Cross-validate with operation type
            if operation_type and operation_type != transaction_type:
                if operation_type in ['pix', 'ted', 'doc'] and transaction_type not in ['transfer', 'payment']:
                    result.add_warning(
                        f"Operation type '{operation_type}' typically uses transaction type 'transfer' or 'payment'",
                        'transaction_type',
                        ValidationSeverity.LOW
                    )

        except (ValueError, TypeError):
            result.add_error("Invalid amount format", 'amount')

        return result

    def _validate_transaction_category(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate transaction categories and subcategories."""
        result = ValidationResult()

        category = data.get('category', '').lower()
        subcategory = data.get('subcategory', '').lower()
        amount = data.get('amount')

        if not category:
            return result

        # Category validation rules
        category_rules = {
            'food': {
                'subcategories': ['restaurant', 'grocery', 'delivery', 'snack'],
                'amount_range': (5, 500)
            },
            'transport': {
                'subcategories': ['taxi', 'bus', 'gas', 'parking', 'uber'],
                'amount_range': (2, 200)
            },
            'shopping': {
                'subcategories': ['clothing', 'electronics', 'home', 'books'],
                'amount_range': (10, 5000)
            },
            'entertainment': {
                'subcategories': ['movie', 'concert', 'game', 'sport'],
                'amount_range': (5, 1000)
            },
            'health': {
                'subcategories': ['doctor', 'pharmacy', 'insurance', 'dental'],
                'amount_range': (10, 2000)
            },
            'utilities': {
                'subcategories': ['electricity', 'water', 'internet', 'phone'],
                'amount_range': (20, 500)
            }
        }

        if category in category_rules:
            rules = category_rules[category]

            # Validate subcategory
            if subcategory and subcategory not in rules['subcategories']:
                result.add_warning(
                    f"Subcategory '{subcategory}' is unusual for category '{category}'",
                    'subcategory',
                    ValidationSeverity.LOW
                )

            # Validate amount range
            if amount is not None:
                try:
                    amount_val = abs(float(amount))
                    min_amount, max_amount = rules['amount_range']

                    if amount_val < min_amount:
                        result.add_warning(
                            f"Amount seems low for {category} category",
                            'amount',
                            ValidationSeverity.LOW
                        )
                    elif amount_val > max_amount:
                        result.add_warning(
                            f"Amount seems high for {category} category",
                            'amount',
                            ValidationSeverity.MEDIUM
                        )

                except (ValueError, TypeError):
                    pass

        return result

    def _validate_bank_specific_format(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate bank-specific transaction formats."""
        result = ValidationResult()

        bank_name = data.get('bank_name', '').lower()
        account_number = data.get('account_number', '')
        transaction_code = data.get('transaction_code', '')

        if not bank_name or bank_name not in self.bank_rules:
            return result

        bank_config = self.bank_rules[bank_name]

        # Validate account number format
        if account_number:
            account_pattern = bank_config.get('account_format')
            if account_pattern and not re.match(account_pattern, str(account_number)):
                result.add_error(
                    f"Invalid account number format for {bank_name.title()}",
                    'account_number',
                    ValidationSeverity.HIGH
                )

        # Validate transaction codes
        if transaction_code:
            valid_codes = bank_config.get('transaction_codes', [])
            if valid_codes and transaction_code.upper() not in valid_codes:
                result.add_warning(
                    f"Unknown transaction code '{transaction_code}' for {bank_name.title()}",
                    'transaction_code',
                    ValidationSeverity.MEDIUM
                )

        return result

    def _validate_bank_transaction_limits(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate transaction limits for specific banks."""
        result = ValidationResult()

        bank_name = data.get('bank_name', '').lower()
        amount = data.get('amount')
        transaction_time = data.get('transaction_time')

        if not bank_name or bank_name not in self.bank_rules or amount is None:
            return result

        bank_config = self.bank_rules[bank_name]

        try:
            amount_val = abs(float(amount))

            # Check transaction limit
            max_transaction = bank_config.get('max_transaction', 0)
            if max_transaction > 0 and amount_val > max_transaction:
                result.add_error(
                    f"Amount exceeds maximum transaction limit for {bank_name.title()} ({max_transaction})",
                    'amount',
                    ValidationSeverity.HIGH
                )

            # Check business hours
            if transaction_time:
                try:
                    if isinstance(transaction_time, str):
                        hour = datetime.fromisoformat(transaction_time).hour
                    else:
                        hour = transaction_time.hour

                    start_hour, end_hour = bank_config.get('business_hours', (0, 24))
                    if not (start_hour <= hour < end_hour):
                        result.add_warning(
                            f"Transaction outside business hours for {bank_name.title()}",
                            'transaction_time',
                            ValidationSeverity.LOW
                        )

                except (ValueError, TypeError, AttributeError):
                    pass

        except (ValueError, TypeError):
            result.add_error("Invalid amount format", 'amount')

        return result

    def _validate_currency_exchange(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate currency exchange rates and conversions."""
        result = ValidationResult()

        amount = data.get('amount')
        currency = data.get('currency', 'BRL').upper()
        exchange_rate = data.get('exchange_rate')
        original_amount = data.get('original_amount')
        original_currency = data.get('original_currency', '').upper()

        if not exchange_rate or not original_amount:
            return result

        try:
            amount_val = float(amount) if amount else 0
            original_val = float(original_amount)
            rate_val = float(exchange_rate)

            # Check if currencies are different
            if currency == original_currency:
                result.add_warning(
                    "Exchange rate provided but currencies are the same",
                    'exchange_rate',
                    ValidationSeverity.LOW
                )
                return result

            # Validate exchange rate reasonableness
            if original_currency in self.currency_rates and currency in self.currency_rates.get(original_currency, {}):
                expected_rate = self.currency_rates[original_currency][currency]
                rate_diff = abs(rate_val - expected_rate) / expected_rate

                if rate_diff > 0.1:  # More than 10% difference
                    result.add_warning(
                        f"Exchange rate differs significantly from market rate (expected: {expected_rate})",
                        'exchange_rate',
                        ValidationSeverity.MEDIUM
                    )

            # Validate conversion calculation
            expected_amount = original_val * rate_val
            if amount and abs(expected_amount - amount_val) > 0.01:
                result.add_error(
                    f"Amount mismatch in currency conversion (expected: {expected_amount})",
                    'amount',
                    ValidationSeverity.HIGH
                )

        except (ValueError, TypeError):
            result.add_error("Invalid numeric format for currency conversion", 'amount')

        return result

    def _validate_multi_currency_transaction(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate multi-currency transaction consistency."""
        result = ValidationResult()

        currencies = []
        amounts = []

        # Collect all currency and amount fields
        for key, value in data.items():
            if key.endswith('_currency'):
                if value:
                    currencies.append(value.upper())
            elif key.endswith('_amount'):
                if value:
                    try:
                        amounts.append(float(value))
                    except (ValueError, TypeError):
                        pass

        # Check for multiple currencies
        if len(set(currencies)) > 1:
            result.add_metadata('multi_currency_transaction', True)

            # Validate that amounts are reasonable for their currencies
            for i, (currency, amount) in enumerate(zip(currencies, amounts)):
                if currency in ['USD', 'EUR', 'GBP'] and amount > 10000:
                    result.add_warning(
                        f"Large amount in {currency} currency",
                        f'amount_{i}',
                        ValidationSeverity.LOW
                    )

        return result

    def _validate_aml_compliance(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Anti-Money Laundering compliance validation."""
        result = ValidationResult()

        amount = data.get('amount')
        recipient = data.get('recipient', '').lower()
        description = data.get('description', '').lower()

        if amount is None:
            return result

        try:
            amount_val = abs(float(amount))

            # Check for suspicious patterns
            suspicious_patterns = [
                r'cash',
                r'wire.*transfer',
                r'bulk.*cash',
                r'money.*order',
                r'gambling',
                r'casino'
            ]

            for pattern in suspicious_patterns:
                if re.search(pattern, description):
                    result.add_warning(
                        f"Suspicious transaction pattern detected: {pattern}",
                        'description',
                        ValidationSeverity.MEDIUM
                    )

            # Check for large cash transactions
            if 'cash' in description and amount_val > 10000:
                result.add_warning(
                    "Large cash transaction - AML review recommended",
                    'amount',
                    ValidationSeverity.HIGH
                )

            # Check for round number transactions
            if amount_val > 1000 and amount_val == int(amount_val):
                result.add_warning(
                    "Large round number transaction - please verify",
                    'amount',
                    ValidationSeverity.LOW
                )

        except (ValueError, TypeError):
            pass

        return result

    def _validate_tax_reporting_threshold(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate tax reporting requirements."""
        result = ValidationResult()

        amount = data.get('amount')
        transaction_type = data.get('transaction_type', '').lower()

        if amount is None:
            return result

        try:
            amount_val = abs(float(amount))

            # Brazilian tax reporting thresholds
            thresholds = {
                'bacen_foreign': 30000,  # Foreign exchange transactions
                'receita_income': 30000,  # Income tax reporting
                'coaf_suspicious': 50000,  # Suspicious activity reporting
                'large_transaction': 100000  # Large transaction reporting
            }

            for threshold_name, threshold_value in thresholds.items():
                if amount_val >= threshold_value:
                    result.add_warning(
                        f"Transaction exceeds {threshold_name} reporting threshold ({threshold_value})",
                        'amount',
                        ValidationSeverity.MEDIUM
                    )

                    # Add metadata for reporting
                    result.add_metadata(f'reporting_required_{threshold_name}', True)
                    result.add_metadata('reporting_threshold_exceeded', threshold_value)

        except (ValueError, TypeError):
            pass

        return result

    def _validate_retail_transaction_patterns(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate retail industry transaction patterns."""
        result = ValidationResult()

        transaction_type = data.get('transaction_type', '').lower()
        amount = data.get('amount')
        time_of_day = data.get('time_of_day')

        if transaction_type not in ['sale', 'purchase', 'refund']:
            return result

        try:
            if amount:
                amount_val = abs(float(amount))

                # Retail transaction patterns
                if transaction_type == 'sale':
                    if amount_val > 5000:  # Unusually large retail sale
                        result.add_warning(
                            "Large retail sale amount",
                            'amount',
                            ValidationSeverity.LOW
                        )

                elif transaction_type == 'refund':
                    if amount_val > 1000:  # Large refund
                        result.add_warning(
                            "Large refund amount - please verify",
                            'amount',
                            ValidationSeverity.MEDIUM
                        )

            # Check transaction timing for retail
            if time_of_day:
                try:
                    hour = int(time_of_day)
                    if transaction_type == 'sale' and (hour < 8 or hour > 20):
                        result.add_warning(
                            "Retail sale outside typical business hours",
                            'time_of_day',
                            ValidationSeverity.LOW
                        )
                except (ValueError, TypeError):
                    pass

        except (ValueError, TypeError):
            pass

        return result

    def _validate_corporate_payment(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate corporate payment transaction rules."""
        result = ValidationResult()

        transaction_type = data.get('transaction_type', '').lower()
        amount = data.get('amount')
        vendor_id = data.get('vendor_id')
        invoice_number = data.get('invoice_number')

        if transaction_type not in ['payment', 'bill_pay', 'vendor_payment']:
            return result

        try:
            if amount:
                amount_val = abs(float(amount))

                # Corporate payment validation
                if amount_val > 50000:  # Large corporate payment
                    if not invoice_number:
                        result.add_error(
                            "Large corporate payment requires invoice number",
                            'invoice_number',
                            ValidationSeverity.HIGH
                        )

                    if not vendor_id:
                        result.add_warning(
                            "Large payment should have vendor ID for tracking",
                            'vendor_id',
                            ValidationSeverity.MEDIUM
                        )

                # Check for duplicate invoice numbers (if context available)
                if invoice_number and context and 'processed_invoices' in context:
                    if invoice_number in context['processed_invoices']:
                        result.add_error(
                            "Duplicate invoice number detected",
                            'invoice_number',
                            ValidationSeverity.HIGH
                        )

        except (ValueError, TypeError):
            pass

        return result

    def _validate_transaction_risk_scoring(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Score transactions for risk assessment."""
        result = ValidationResult()

        amount = data.get('amount')
        transaction_type = data.get('transaction_type', '').lower()
        recipient = data.get('recipient', '').lower()

        if amount is None:
            return result

        try:
            amount_val = abs(float(amount))
            risk_score = 0

            # Risk factors
            if amount_val > 10000:
                risk_score += 2
            elif amount_val > 50000:
                risk_score += 3

            if transaction_type in ['wire_transfer', 'international']:
                risk_score += 2

            if 'cash' in recipient or 'anonymous' in recipient:
                risk_score += 3

            # Add risk metadata
            result.add_metadata('risk_score', risk_score)
            result.add_metadata('risk_level', 'LOW' if risk_score < 3 else 'MEDIUM' if risk_score < 6 else 'HIGH')

            if risk_score >= 5:
                result.add_warning(
                    f"High risk transaction detected (score: {risk_score})",
                    'amount',
                    ValidationSeverity.MEDIUM
                )

        except (ValueError, TypeError):
            pass

        return result

    def _validate_velocity_checks(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate transaction velocity patterns."""
        result = ValidationResult()

        if not context or 'recent_transactions' not in context:
            return result

        recent_transactions = context['recent_transactions']
        current_amount = data.get('amount')
        current_time = data.get('timestamp')

        if not current_amount or not current_time:
            return result

        try:
            current_amount_val = float(current_amount)
            current_time_parsed = datetime.fromisoformat(current_time) if isinstance(current_time, str) else current_time

            # Analyze transaction velocity
            time_window_hours = 24
            amount_threshold = 50000
            frequency_threshold = 10

            recent_amounts = []
            recent_times = []

            for transaction in recent_transactions:
                trans_time = transaction.get('timestamp')
                trans_amount = transaction.get('amount')

                if trans_time and trans_amount:
                    try:
                        trans_time_parsed = (datetime.fromisoformat(trans_time) if isinstance(trans_time, str)
                                           else trans_time)
                        time_diff = (current_time_parsed - trans_time_parsed).total_seconds() / 3600

                        if time_diff <= time_window_hours:
                            recent_amounts.append(float(trans_amount))
                            recent_times.append(trans_time_parsed)
                    except (ValueError, TypeError):
                        continue

            # Check velocity thresholds
            if len(recent_amounts) >= frequency_threshold:
                result.add_warning(
                    f"High transaction frequency detected ({len(recent_amounts)} transactions in {time_window_hours} hours)",
                    'timestamp',
                    ValidationSeverity.MEDIUM
                )

            total_recent_amount = sum(abs(amount) for amount in recent_amounts)
            if total_recent_amount + abs(current_amount_val) > amount_threshold:
                result.add_warning(
                    f"High transaction volume detected (total: {total_recent_amount + abs(current_amount_val)})",
                    'amount',
                    ValidationSeverity.MEDIUM
                )

        except (ValueError, TypeError):
            pass

        return result

    def add_rule(self, rule: FinancialBusinessRule):
        """Add a financial business rule."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added financial business rule: {rule.name} ({rule.rule_id})")

    def remove_rule(self, rule_id: str):
        """Remove a financial business rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed financial business rule: {rule_id}")

    def validate(self, data: Dict[str, Any], rule_group: str = None,
                context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate data against financial business rules.

        Args:
            data: Data to validate
            rule_group: Specific rule group to use
            context: Additional context for validation

        Returns:
            Validation result
        """
        result = ValidationResult()
        result.set_validator_info("FinancialBusinessRules", "1.0")

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
                        logger.error(f"Error executing financial rule {rule_id}: {str(e)}")
                        result.add_error(f"Financial rule execution error: {rule_id}")

        # Set validation duration
        duration = time.time() - start_time
        result.set_validation_duration(duration * 1000)

        logger.info(f"Financial business rules validation completed in {duration:.3f}s with {len(result.errors)} errors, {len(result.warnings)} warnings")

        return result

    def get_rule(self, rule_id: str) -> Optional[FinancialBusinessRule]:
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
        logger.info(f"Created financial rule group: {group_name} with {len(rule_ids)} rules")

    def enable_rule(self, rule_id: str):
        """Enable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            logger.info(f"Enabled financial business rule: {rule_id}")

    def disable_rule(self, rule_id: str):
        """Disable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            logger.info(f"Disabled financial business rule: {rule_id}")

    def update_currency_rates(self, rates: Dict[str, Dict[str, float]]):
        """Update currency exchange rates."""
        self.currency_rates.update(rates)
        logger.info("Updated currency exchange rates")

    def add_bank_rules(self, bank_name: str, rules: Dict[str, Any]):
        """Add bank-specific validation rules."""
        self.bank_rules[bank_name.lower()] = rules
        logger.info(f"Added bank-specific rules for: {bank_name}")


# Global financial business rules instance
financial_business_rules = FinancialBusinessRules()