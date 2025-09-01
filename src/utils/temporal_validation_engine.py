"""
Temporal Validation Engine for Maria Conciliadora system.

This module provides comprehensive time-based validation with:
- Date range and sequence validation
- Business day and holiday validation
- Temporal consistency across related transactions
- Future date and backdating validation
- Seasonal and periodic pattern validation
"""

import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Set
from enum import Enum
import calendar
import re

from .validation_result import ValidationResult, ValidationSeverity
from .logging_config import get_logger

logger = get_logger(__name__)


class TemporalRuleType(Enum):
    """Types of temporal validation rules."""
    DATE_RANGE = "DATE_RANGE"
    BUSINESS_DAY = "BUSINESS_DAY"
    SEQUENCE_VALIDATION = "SEQUENCE_VALIDATION"
    FUTURE_DATING = "FUTURE_DATING"
    BACKDATING = "BACKDATING"
    SEASONAL_PATTERN = "SEASONAL_PATTERN"
    PERIODIC_VALIDATION = "PERIODIC_VALIDATION"


class TemporalValidationRule:
    """Represents a temporal validation rule."""

    def __init__(self, rule_id: str, name: str, rule_type: TemporalRuleType,
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
        """Validate data against this temporal rule."""
        if not self.enabled:
            return ValidationResult()

        try:
            return self.validation_function(data, context or {})
        except Exception as e:
            logger.error(f"Error executing temporal rule {self.rule_id}: {str(e)}")
            result = ValidationResult()
            result.add_error(f"Temporal rule execution error: {str(e)}")
            return result


class TemporalValidationEngine:
    """
    Comprehensive temporal validation engine for time-based data validation.
    """

    def __init__(self):
        self.rules: Dict[str, TemporalValidationRule] = {}
        self.rule_groups: Dict[str, List[str]] = {}
        self.holidays: Set[date] = set()
        self.business_hours: Dict[str, Dict[str, Any]] = {}
        self._load_default_rules()
        self._load_holidays()
        self._load_business_hours()

    def _load_default_rules(self):
        """Load default temporal validation rules."""

        # Date range and sequence rules
        self.add_rule(TemporalValidationRule(
            rule_id="transaction_date_range_validation",
            name="Transaction Date Range Validation",
            rule_type=TemporalRuleType.DATE_RANGE,
            validation_function=self._validate_transaction_date_range,
            description="Validate transaction dates are within acceptable ranges"
        ))

        self.add_rule(TemporalValidationRule(
            rule_id="date_sequence_consistency",
            name="Date Sequence Consistency",
            rule_type=TemporalRuleType.SEQUENCE_VALIDATION,
            validation_function=self._validate_date_sequence_consistency,
            description="Validate date sequences in related transactions"
        ))

        # Business day validation
        self.add_rule(TemporalValidationRule(
            rule_id="business_day_validation",
            name="Business Day Validation",
            rule_type=TemporalRuleType.BUSINESS_DAY,
            validation_function=self._validate_business_day,
            description="Validate transactions occur on business days"
        ))

        self.add_rule(TemporalValidationRule(
            rule_id="holiday_validation",
            name="Holiday Validation",
            rule_type=TemporalRuleType.BUSINESS_DAY,
            validation_function=self._validate_holiday_validation,
            description="Check for transactions on holidays"
        ))

        # Future dating and backdating
        self.add_rule(TemporalValidationRule(
            rule_id="future_date_validation",
            name="Future Date Validation",
            rule_type=TemporalRuleType.FUTURE_DATING,
            validation_function=self._validate_future_date,
            description="Validate future-dated transactions"
        ))

        self.add_rule(TemporalValidationRule(
            rule_id="backdating_validation",
            name="Backdating Validation",
            rule_type=TemporalRuleType.BACKDATING,
            validation_function=self._validate_backdating,
            description="Validate backdated transactions"
        ))

        # Seasonal and periodic patterns
        self.add_rule(TemporalValidationRule(
            rule_id="seasonal_pattern_validation",
            name="Seasonal Pattern Validation",
            rule_type=TemporalRuleType.SEASONAL_PATTERN,
            validation_function=self._validate_seasonal_pattern,
            description="Validate seasonal transaction patterns"
        ))

        self.add_rule(TemporalValidationRule(
            rule_id="periodic_transaction_validation",
            name="Periodic Transaction Validation",
            rule_type=TemporalRuleType.PERIODIC_VALIDATION,
            validation_function=self._validate_periodic_transaction,
            description="Validate periodic transaction patterns"
        ))

        # Group rules
        self.rule_groups['date_validation'] = [
            'transaction_date_range_validation',
            'business_day_validation',
            'holiday_validation'
        ]

        self.rule_groups['temporal_consistency'] = [
            'date_sequence_consistency',
            'future_date_validation',
            'backdating_validation'
        ]

        self.rule_groups['pattern_analysis'] = [
            'seasonal_pattern_validation',
            'periodic_transaction_validation'
        ]

    def _load_holidays(self):
        """Load Brazilian holidays for validation."""
        current_year = datetime.now().year

        # Fixed date holidays
        fixed_holidays = [
            (1, 1),   # New Year
            (4, 21),  # Tiradentes
            (5, 1),   # Labor Day
            (9, 7),   # Independence
            (10, 12), # Our Lady of Aparecida
            (11, 2),  # All Souls
            (11, 15), # Republic Proclamation
            (12, 25), # Christmas
        ]

        for year in range(current_year - 1, current_year + 2):  # Last year, current, next year
            for month, day in fixed_holidays:
                self.holidays.add(date(year, month, day))

            # Easter-related holidays (simplified)
            # Good Friday and Easter Monday would need more complex calculation
            # For now, we'll add Carnival (simplified)
            carnival_dates = self._calculate_carnival(year)
            for carnival_date in carnival_dates:
                self.holidays.add(carnival_date)

    def _calculate_carnival(self, year: int) -> List[date]:
        """Calculate Carnival dates (simplified)."""
        # This is a simplified calculation - in production, use a proper Easter calculation
        return [
            date(year, 2, 20),  # Simplified Carnival Monday
            date(year, 2, 21),  # Simplified Carnival Tuesday
        ]

    def _load_business_hours(self):
        """Load business hours for different transaction types."""
        self.business_hours = {
            'banking': {
                'weekdays': (9, 17),  # 9 AM to 5 PM
                'weekends': None,     # No weekend banking
                'timezone': 'America/Sao_Paulo'
            },
            'retail': {
                'weekdays': (8, 20),  # 8 AM to 8 PM
                'weekends': (9, 18),  # 9 AM to 6 PM
                'timezone': 'America/Sao_Paulo'
            },
            'online': {
                'weekdays': (0, 24),  # 24/7
                'weekends': (0, 24),  # 24/7
                'timezone': 'UTC'
            }
        }

    def _validate_transaction_date_range(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate transaction dates are within acceptable ranges."""
        result = ValidationResult()

        date_fields = ['date', 'transaction_date', 'created_at', 'updated_at', 'effective_date']
        current_date = datetime.now().date()

        for field in date_fields:
            date_value = data.get(field)
            if date_value is None:
                continue

            try:
                parsed_date = self._parse_date(date_value)

                # Check for dates too far in the past
                days_old = (current_date - parsed_date).days
                if days_old > 365 * 10:  # 10 years
                    result.add_warning(
                        f"Transaction date is very old ({days_old} days)",
                        field,
                        ValidationSeverity.MEDIUM
                    )

                # Check for dates too far in the future
                days_future = (parsed_date - current_date).days
                if days_future > 90:  # 90 days
                    result.add_warning(
                        f"Transaction date is far in the future ({days_future} days)",
                        field,
                        ValidationSeverity.MEDIUM
                    )

                # Check for dates before 1990 (unlikely for modern transactions)
                if parsed_date.year < 1990:
                    result.add_error(
                        f"Transaction date is unrealistically old (before 1990)",
                        field,
                        ValidationSeverity.HIGH
                    )

            except (ValueError, TypeError):
                result.add_error(f"Invalid date format in {field}", field)

        return result

    def _validate_date_sequence_consistency(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate date sequences in related transactions."""
        result = ValidationResult()

        # Check created_at vs updated_at
        created_at = data.get('created_at')
        updated_at = data.get('updated_at')

        if created_at and updated_at:
            try:
                created_date = self._parse_datetime(created_at)
                updated_date = self._parse_datetime(updated_at)

                if created_date > updated_date:
                    result.add_error(
                        "Created date cannot be after updated date",
                        'created_at',
                        ValidationSeverity.HIGH
                    )

                # Check for suspiciously quick updates
                time_diff = (updated_date - created_date).total_seconds()
                if time_diff < 1:  # Less than 1 second
                    result.add_warning(
                        "Updated date is suspiciously close to created date",
                        'updated_at',
                        ValidationSeverity.LOW
                    )

            except (ValueError, TypeError):
                pass

        # Check transaction date vs reconciliation date
        transaction_date = data.get('date')
        reconciliation_date = data.get('reconciliation_date')

        if transaction_date and reconciliation_date:
            try:
                trans_date = self._parse_date(transaction_date)
                recon_date = self._parse_date(reconciliation_date)

                if trans_date > recon_date:
                    result.add_error(
                        "Transaction date cannot be after reconciliation date",
                        'date',
                        ValidationSeverity.HIGH
                    )

                # Check for delayed reconciliation
                days_diff = (recon_date - trans_date).days
                if days_diff > 30:  # More than 30 days
                    result.add_warning(
                        f"Reconciliation delayed by {days_diff} days",
                        'reconciliation_date',
                        ValidationSeverity.MEDIUM
                    )

            except (ValueError, TypeError):
                pass

        return result

    def _validate_business_day(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate transactions occur on business days."""
        result = ValidationResult()

        transaction_date = data.get('date')
        transaction_type = data.get('transaction_type', '').lower()
        channel = data.get('channel', 'banking').lower()

        if not transaction_date:
            return result

        try:
            parsed_date = self._parse_date(transaction_date)

            # Check if it's a weekend
            if parsed_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                # Some channels might operate on weekends
                if channel in self.business_hours and self.business_hours[channel].get('weekends') is None:
                    result.add_warning(
                        f"Transaction on weekend for {channel} channel",
                        'date',
                        ValidationSeverity.MEDIUM
                    )

            # Check business hours
            if channel in self.business_hours:
                hours_config = self.business_hours[channel]
                is_weekend = parsed_date.weekday() >= 5

                if is_weekend and hours_config.get('weekends') is None:
                    result.add_error(
                        f"{channel.title()} transactions not allowed on weekends",
                        'date',
                        ValidationSeverity.HIGH
                    )

        except (ValueError, TypeError):
            result.add_error("Invalid date format", 'date')

        return result

    def _validate_holiday_validation(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Check for transactions on holidays."""
        result = ValidationResult()

        transaction_date = data.get('date')
        transaction_type = data.get('transaction_type', '').lower()

        if not transaction_date:
            return result

        try:
            parsed_date = self._parse_date(transaction_date)

            if parsed_date in self.holidays:
                # Some transaction types might be allowed on holidays
                if transaction_type not in ['online', 'atm', 'mobile']:
                    result.add_warning(
                        f"Transaction on holiday: {parsed_date}",
                        'date',
                        ValidationSeverity.LOW
                    )

        except (ValueError, TypeError):
            pass

        return result

    def _validate_future_date(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate future-dated transactions."""
        result = ValidationResult()

        transaction_date = data.get('date')
        transaction_type = data.get('transaction_type', '').lower()
        current_date = datetime.now().date()

        if not transaction_date:
            return result

        try:
            parsed_date = self._parse_date(transaction_date)

            if parsed_date > current_date:
                days_future = (parsed_date - current_date).days

                # Allow some future dating for certain transaction types
                allowed_future_days = {
                    'scheduled_payment': 365,  # 1 year
                    'recurring': 31,          # 1 month
                    'investment': 1,          # 1 day
                    'default': 7              # 1 week
                }

                max_allowed = allowed_future_days.get(transaction_type, allowed_future_days['default'])

                if days_future > max_allowed:
                    result.add_error(
                        f"Future date exceeds maximum allowed ({max_allowed} days)",
                        'date',
                        ValidationSeverity.HIGH
                    )
                elif days_future > 30:  # More than 30 days
                    result.add_warning(
                        f"Transaction dated {days_future} days in the future",
                        'date',
                        ValidationSeverity.MEDIUM
                    )

                # Add metadata about future dating
                result.add_metadata('future_dated', True)
                result.add_metadata('days_in_future', days_future)

        except (ValueError, TypeError):
            pass

        return result

    def _validate_backdating(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate backdated transactions."""
        result = ValidationResult()

        transaction_date = data.get('date')
        created_at = data.get('created_at')
        transaction_type = data.get('transaction_type', '').lower()
        current_date = datetime.now().date()

        if not transaction_date:
            return result

        try:
            parsed_date = self._parse_date(transaction_date)

            if parsed_date < current_date:
                days_past = (current_date - parsed_date).days

                # Check against creation date if available
                if created_at:
                    try:
                        created_date = self._parse_date(created_at)
                        if parsed_date < created_date:
                            result.add_error(
                                "Transaction date cannot be before creation date",
                                'date',
                                ValidationSeverity.HIGH
                            )
                    except (ValueError, TypeError):
                        pass

                # Allow limited backdating for corrections
                max_backdate_days = 90  # 90 days for corrections
                if days_past > max_backdate_days:
                    result.add_warning(
                        f"Transaction backdated by {days_past} days (max allowed: {max_backdate_days})",
                        'date',
                        ValidationSeverity.MEDIUM
                    )

                # Add metadata about backdating
                result.add_metadata('backdated', True)
                result.add_metadata('days_backdated', days_past)

        except (ValueError, TypeError):
            pass

        return result

    def _validate_seasonal_pattern(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate seasonal transaction patterns."""
        result = ValidationResult()

        transaction_date = data.get('date')
        amount = data.get('amount')
        category = data.get('category', '').lower()

        if not transaction_date or not amount:
            return result

        try:
            parsed_date = self._parse_date(transaction_date)
            amount_val = float(amount)

            # Seasonal patterns for different categories
            seasonal_patterns = {
                'bonus': [12],  # December bonuses
                'tax': [3, 4, 12],  # Q1 and Q4 tax payments
                'insurance': [1, 7],  # January and July renewals
                'vacation': [1, 7, 12],  # Holiday seasons
                'education': [2, 8],  # February and August
            }

            if category in seasonal_patterns:
                expected_months = seasonal_patterns[category]
                if parsed_date.month not in expected_months:
                    result.add_warning(
                        f"{category.title()} transaction unusual for month {parsed_date.month}",
                        'date',
                        ValidationSeverity.LOW
                    )

            # Year-end patterns
            if parsed_date.month == 12 and amount_val > 10000:
                result.add_metadata('year_end_transaction', True)
                result.add_metadata('potential_bonus_or_adjustment', True)

            # Tax season patterns
            if parsed_date.month in [3, 4] and 'tax' in category:
                result.add_metadata('tax_season_transaction', True)

        except (ValueError, TypeError):
            pass

        return result

    def _validate_periodic_transaction(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate periodic transaction patterns."""
        result = ValidationResult()

        if not context or 'transaction_history' not in context:
            return result

        transaction_date = data.get('date')
        amount = data.get('amount')
        description = data.get('description', '').lower()

        if not transaction_date or not amount:
            return result

        try:
            current_date = self._parse_date(transaction_date)
            current_amount = float(amount)
            transaction_history = context['transaction_history']

            # Look for periodic patterns
            periodic_transactions = []
            for trans in transaction_history:
                trans_date = trans.get('date')
                trans_amount = trans.get('amount')
                trans_desc = trans.get('description', '').lower()

                if trans_date and trans_amount and trans_desc == description:
                    try:
                        trans_date_parsed = self._parse_date(trans_date)
                        trans_amount_val = float(trans_amount)

                        # Check if amounts are similar (within 10%)
                        amount_diff = abs(current_amount - trans_amount_val) / max(current_amount, trans_amount_val)
                        if amount_diff <= 0.1:
                            periodic_transactions.append(trans_date_parsed)
                    except (ValueError, TypeError):
                        continue

            if len(periodic_transactions) >= 2:
                # Calculate intervals
                intervals = []
                sorted_dates = sorted(periodic_transactions + [current_date])

                for i in range(1, len(sorted_dates)):
                    interval = (sorted_dates[i] - sorted_dates[i-1]).days
                    intervals.append(interval)

                if intervals:
                    avg_interval = sum(intervals) / len(intervals)

                    # Check for common periodic patterns
                    if 25 <= avg_interval <= 35:  # Monthly
                        result.add_metadata('periodic_pattern', 'monthly')
                        result.add_metadata('average_interval_days', avg_interval)
                    elif 85 <= avg_interval <= 95:  # Quarterly
                        result.add_metadata('periodic_pattern', 'quarterly')
                        result.add_metadata('average_interval_days', avg_interval)
                    elif 360 <= avg_interval <= 370:  # Annual
                        result.add_metadata('periodic_pattern', 'annual')
                        result.add_metadata('average_interval_days', avg_interval)

                    # Check for unusual timing
                    if current_date not in sorted_dates[:-1]:  # Not already in history
                        expected_next_date = sorted_dates[-2] + timedelta(days=int(avg_interval))
                        days_diff = abs((current_date - expected_next_date).days)

                        if days_diff > 7:  # More than 1 week off schedule
                            result.add_warning(
                                f"Periodic transaction {days_diff} days off schedule",
                                'date',
                                ValidationSeverity.LOW
                            )

        except (ValueError, TypeError):
            pass

        return result

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

    def _parse_datetime(self, datetime_value: Any) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if datetime_value is None:
            return None

        try:
            if isinstance(datetime_value, str):
                return datetime.fromisoformat(datetime_value)
            elif isinstance(datetime_value, datetime):
                return datetime_value
        except (ValueError, TypeError):
            pass

        return None

    def add_rule(self, rule: TemporalValidationRule):
        """Add a temporal validation rule."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added temporal validation rule: {rule.name} ({rule.rule_id})")

    def remove_rule(self, rule_id: str):
        """Remove a temporal validation rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed temporal validation rule: {rule_id}")

    def validate(self, data: Dict[str, Any], rule_group: str = None,
                context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate data against temporal validation rules.

        Args:
            data: Data to validate
            rule_group: Specific rule group to use
            context: Additional context for validation

        Returns:
            Validation result
        """
        result = ValidationResult()
        result.set_validator_info("TemporalValidationEngine", "1.0")

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
                        logger.error(f"Error executing temporal rule {rule_id}: {str(e)}")
                        result.add_error(f"Temporal rule execution error: {rule_id}")

        # Set validation duration
        duration = time.time() - start_time
        result.set_validation_duration(duration * 1000)

        logger.info(f"Temporal validation completed in {duration:.3f}s with {len(result.errors)} errors, {len(result.warnings)} warnings")

        return result

    def get_rule(self, rule_id: str) -> Optional[TemporalValidationRule]:
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
        logger.info(f"Created temporal rule group: {group_name} with {len(rule_ids)} rules")

    def enable_rule(self, rule_id: str):
        """Enable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            logger.info(f"Enabled temporal validation rule: {rule_id}")

    def disable_rule(self, rule_id: str):
        """Disable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            logger.info(f"Disabled temporal validation rule: {rule_id}")

    def add_holiday(self, holiday_date: date):
        """Add a holiday date."""
        self.holidays.add(holiday_date)
        logger.info(f"Added holiday: {holiday_date}")

    def remove_holiday(self, holiday_date: date):
        """Remove a holiday date."""
        self.holidays.discard(holiday_date)
        logger.info(f"Removed holiday: {holiday_date}")

    def update_business_hours(self, channel: str, hours_config: Dict[str, Any]):
        """Update business hours for a channel."""
        self.business_hours[channel] = hours_config
        logger.info(f"Updated business hours for channel: {channel}")


# Global temporal validation engine instance
temporal_validation_engine = TemporalValidationEngine()