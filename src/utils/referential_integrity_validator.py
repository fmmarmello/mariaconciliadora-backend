"""
Referential Integrity Validator for Maria Conciliadora system.

This module provides data relationship validation with:
- Foreign key relationship validation
- Cross-table consistency checks
- Master data reference validation
- Hierarchical relationship validation
- Data lineage and dependency tracking
"""

import time
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Callable, Union, Set
from enum import Enum
import re

from .validation_result import ValidationResult, ValidationSeverity
from .logging_config import get_logger

logger = get_logger(__name__)


class ReferentialRuleType(Enum):
    """Types of referential integrity rules."""
    FOREIGN_KEY = "FOREIGN_KEY"
    CROSS_TABLE_CONSISTENCY = "CROSS_TABLE_CONSISTENCY"
    MASTER_DATA_REFERENCE = "MASTER_DATA_REFERENCE"
    HIERARCHICAL_RELATIONSHIP = "HIERARCHICAL_RELATIONSHIP"
    DATA_LINEAGE = "DATA_LINEAGE"


class ReferentialIntegrityRule:
    """Represents a referential integrity validation rule."""

    def __init__(self, rule_id: str, name: str, rule_type: ReferentialRuleType,
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
        """Validate data against this referential integrity rule."""
        if not self.enabled:
            return ValidationResult()

        try:
            return self.validation_function(data, context or {})
        except Exception as e:
            logger.error(f"Error executing referential rule {self.rule_id}: {str(e)}")
            result = ValidationResult()
            result.add_error(f"Referential rule execution error: {str(e)}")
            return result


class ReferentialIntegrityValidator:
    """
    Comprehensive referential integrity validator for data relationship validation.
    """

    def __init__(self):
        self.rules: Dict[str, ReferentialIntegrityRule] = {}
        self.rule_groups: Dict[str, List[str]] = {}
        self.master_data: Dict[str, Dict[str, Any]] = {}
        self.foreign_key_mappings: Dict[str, Dict[str, str]] = {}
        self.hierarchical_relationships: Dict[str, Dict[str, Any]] = {}
        self._load_default_rules()
        self._load_master_data()
        self._load_relationship_mappings()

    def _load_default_rules(self):
        """Load default referential integrity validation rules."""

        # Foreign key validation rules
        self.add_rule(ReferentialIntegrityRule(
            rule_id="account_foreign_key_validation",
            name="Account Foreign Key Validation",
            rule_type=ReferentialRuleType.FOREIGN_KEY,
            validation_function=self._validate_account_foreign_key,
            description="Validate account references exist in master data"
        ))

        self.add_rule(ReferentialIntegrityRule(
            rule_id="customer_foreign_key_validation",
            name="Customer Foreign Key Validation",
            rule_type=ReferentialRuleType.FOREIGN_KEY,
            validation_function=self._validate_customer_foreign_key,
            description="Validate customer references exist in master data"
        ))

        # Cross-table consistency rules
        self.add_rule(ReferentialIntegrityRule(
            rule_id="transaction_account_consistency",
            name="Transaction Account Consistency",
            rule_type=ReferentialRuleType.CROSS_TABLE_CONSISTENCY,
            validation_function=self._validate_transaction_account_consistency,
            description="Validate transaction data consistency across related tables"
        ))

        self.add_rule(ReferentialIntegrityRule(
            rule_id="balance_reconciliation_consistency",
            name="Balance Reconciliation Consistency",
            rule_type=ReferentialRuleType.CROSS_TABLE_CONSISTENCY,
            validation_function=self._validate_balance_reconciliation_consistency,
            description="Validate balance reconciliation across multiple data sources"
        ))

        # Master data reference rules
        self.add_rule(ReferentialIntegrityRule(
            rule_id="bank_master_data_validation",
            name="Bank Master Data Validation",
            rule_type=ReferentialRuleType.MASTER_DATA_REFERENCE,
            validation_function=self._validate_bank_master_data,
            description="Validate bank references against master data"
        ))

        self.add_rule(ReferentialIntegrityRule(
            rule_id="currency_master_data_validation",
            name="Currency Master Data Validation",
            rule_type=ReferentialRuleType.MASTER_DATA_REFERENCE,
            validation_function=self._validate_currency_master_data,
            description="Validate currency codes against master data"
        ))

        # Hierarchical relationship rules
        self.add_rule(ReferentialIntegrityRule(
            rule_id="organizational_hierarchy_validation",
            name="Organizational Hierarchy Validation",
            rule_type=ReferentialRuleType.HIERARCHICAL_RELATIONSHIP,
            validation_function=self._validate_organizational_hierarchy,
            description="Validate organizational hierarchy relationships"
        ))

        self.add_rule(ReferentialIntegrityRule(
            rule_id="category_hierarchy_validation",
            name="Category Hierarchy Validation",
            rule_type=ReferentialRuleType.HIERARCHICAL_RELATIONSHIP,
            validation_function=self._validate_category_hierarchy,
            description="Validate category hierarchy relationships"
        ))

        # Data lineage rules
        self.add_rule(ReferentialIntegrityRule(
            rule_id="data_lineage_tracking",
            name="Data Lineage Tracking",
            rule_type=ReferentialRuleType.DATA_LINEAGE,
            validation_function=self._validate_data_lineage,
            description="Track and validate data lineage dependencies"
        ))

        self.add_rule(ReferentialIntegrityRule(
            rule_id="audit_trail_integrity",
            name="Audit Trail Integrity",
            rule_type=ReferentialRuleType.DATA_LINEAGE,
            validation_function=self._validate_audit_trail_integrity,
            description="Validate audit trail data integrity"
        ))

        # Group rules
        self.rule_groups['foreign_key_validation'] = [
            'account_foreign_key_validation',
            'customer_foreign_key_validation'
        ]

        self.rule_groups['cross_table_validation'] = [
            'transaction_account_consistency',
            'balance_reconciliation_consistency'
        ]

        self.rule_groups['master_data_validation'] = [
            'bank_master_data_validation',
            'currency_master_data_validation'
        ]

        self.rule_groups['hierarchy_validation'] = [
            'organizational_hierarchy_validation',
            'category_hierarchy_validation'
        ]

        self.rule_groups['data_integrity'] = [
            'data_lineage_tracking',
            'audit_trail_integrity'
        ]

    def _load_master_data(self):
        """Load master data for referential validation."""

        # Bank master data
        self.master_data['banks'] = {
            'itau': {'name': 'Itaú Unibanco', 'country': 'BR', 'active': True},
            'bradesco': {'name': 'Bradesco', 'country': 'BR', 'active': True},
            'santander': {'name': 'Santander', 'country': 'BR', 'active': True},
            'nubank': {'name': 'Nubank', 'country': 'BR', 'active': True},
            'caixa': {'name': 'Caixa Econômica Federal', 'country': 'BR', 'active': True},
            'bb': {'name': 'Banco do Brasil', 'country': 'BR', 'active': True}
        }

        # Currency master data
        self.master_data['currencies'] = {
            'BRL': {'name': 'Brazilian Real', 'symbol': 'R$', 'active': True},
            'USD': {'name': 'US Dollar', 'symbol': '$', 'active': True},
            'EUR': {'name': 'Euro', 'symbol': '€', 'active': True},
            'GBP': {'name': 'British Pound', 'symbol': '£', 'active': True}
        }

        # Transaction type master data
        self.master_data['transaction_types'] = {
            'debit': {'description': 'Debit transaction', 'category': 'expense'},
            'credit': {'description': 'Credit transaction', 'category': 'income'},
            'transfer': {'description': 'Transfer between accounts', 'category': 'transfer'},
            'fee': {'description': 'Bank fee', 'category': 'expense'},
            'interest': {'description': 'Interest payment', 'category': 'income'},
            'adjustment': {'description': 'Account adjustment', 'category': 'adjustment'}
        }

        # Category hierarchy
        self.master_data['categories'] = {
            'food': {'parent': None, 'level': 1},
            'groceries': {'parent': 'food', 'level': 2},
            'restaurant': {'parent': 'food', 'level': 2},
            'transport': {'parent': None, 'level': 1},
            'fuel': {'parent': 'transport', 'level': 2},
            'public_transport': {'parent': 'transport', 'level': 2}
        }

    def _load_relationship_mappings(self):
        """Load foreign key and relationship mappings."""

        # Foreign key mappings
        self.foreign_key_mappings = {
            'transactions': {
                'account_id': 'accounts.id',
                'customer_id': 'customers.id',
                'bank_id': 'banks.id'
            },
            'accounts': {
                'customer_id': 'customers.id',
                'bank_id': 'banks.id'
            }
        }

        # Hierarchical relationships
        self.hierarchical_relationships = {
            'organizational': {
                'levels': ['company', 'department', 'team', 'employee'],
                'relationships': {
                    'department': 'company',
                    'team': 'department',
                    'employee': 'team'
                }
            },
            'product': {
                'levels': ['category', 'subcategory', 'product'],
                'relationships': {
                    'subcategory': 'category',
                    'product': 'subcategory'
                }
            }
        }

    def _validate_account_foreign_key(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate account foreign key references."""
        result = ValidationResult()

        account_id = data.get('account_id')
        bank_id = data.get('bank_id')

        if not account_id:
            return result

        # Check if we have account master data in context
        if context and 'accounts_master' in context:
            accounts_master = context['accounts_master']
            if account_id not in accounts_master:
                result.add_error(
                    f"Account ID {account_id} not found in master data",
                    'account_id',
                    ValidationSeverity.HIGH
                )
            else:
                account_data = accounts_master[account_id]

                # Validate bank consistency
                if bank_id and account_data.get('bank_id') != bank_id:
                    result.add_error(
                        f"Account {account_id} bank mismatch: expected {account_data.get('bank_id')}, got {bank_id}",
                        'bank_id',
                        ValidationSeverity.HIGH
                    )

                # Check account status
                if not account_data.get('active', True):
                    result.add_warning(
                        f"Account {account_id} is inactive",
                        'account_id',
                        ValidationSeverity.MEDIUM
                    )

        return result

    def _validate_customer_foreign_key(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate customer foreign key references."""
        result = ValidationResult()

        customer_id = data.get('customer_id')

        if not customer_id:
            return result

        # Check if we have customer master data in context
        if context and 'customers_master' in context:
            customers_master = context['customers_master']
            if customer_id not in customers_master:
                result.add_error(
                    f"Customer ID {customer_id} not found in master data",
                    'customer_id',
                    ValidationSeverity.HIGH
                )
            else:
                customer_data = customers_master[customer_id]

                # Check customer status
                if not customer_data.get('active', True):
                    result.add_warning(
                        f"Customer {customer_id} is inactive",
                        'customer_id',
                        ValidationSeverity.MEDIUM
                    )

                # Check for sanctions or compliance issues
                if customer_data.get('sanctions_list', False):
                    result.add_error(
                        f"Customer {customer_id} is on sanctions list",
                        'customer_id',
                        ValidationSeverity.CRITICAL
                    )

        return result

    def _validate_transaction_account_consistency(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate transaction data consistency across related tables."""
        result = ValidationResult()

        if not context or 'related_tables' not in context:
            return result

        related_tables = context['related_tables']
        transaction_id = data.get('id')
        account_id = data.get('account_id')

        if not transaction_id or not account_id:
            return result

        # Check transaction exists in accounts table
        if 'accounts' in related_tables:
            accounts_table = related_tables['accounts']
            account_transactions = accounts_table.get(account_id, [])

            if transaction_id not in account_transactions:
                result.add_error(
                    f"Transaction {transaction_id} not found in account {account_id} records",
                    'account_id',
                    ValidationSeverity.HIGH
                )

        # Check balance consistency
        if 'balances' in related_tables:
            balances_table = related_tables['balances']
            account_balance = balances_table.get(account_id)

            if account_balance is not None:
                transaction_amount = data.get('amount')
                transaction_type = data.get('transaction_type', '').lower()

                if transaction_amount is not None:
                    try:
                        amount_val = float(transaction_amount)
                        balance_val = float(account_balance)

                        # Simple balance validation (would need more complex logic in production)
                        if transaction_type in ['debit', 'expense'] and balance_val < amount_val:
                            result.add_warning(
                                f"Insufficient balance for transaction: balance {balance_val}, amount {amount_val}",
                                'amount',
                                ValidationSeverity.MEDIUM
                            )

                    except (ValueError, TypeError):
                        pass

        return result

    def _validate_balance_reconciliation_consistency(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate balance reconciliation across multiple data sources."""
        result = ValidationResult()

        if not context or 'reconciliation_data' not in context:
            return result

        reconciliation_data = context['reconciliation_data']
        account_id = data.get('account_id')
        book_balance = data.get('book_balance')

        if not account_id or book_balance is None:
            return result

        try:
            book_val = float(book_balance)

            # Check against bank statement balance
            bank_balance = reconciliation_data.get('bank_balance')
            if bank_balance is not None:
                bank_val = float(bank_balance)
                difference = abs(book_val - bank_val)

                if difference > 0.01:  # Allow for small differences
                    result.add_error(
                        f"Balance reconciliation difference: book {book_val}, bank {bank_val}",
                        'book_balance',
                        ValidationSeverity.HIGH
                    )

            # Check outstanding items
            outstanding_checks = reconciliation_data.get('outstanding_checks', 0)
            deposits_in_transit = reconciliation_data.get('deposits_in_transit', 0)

            adjusted_balance = book_val - float(outstanding_checks) + float(deposits_in_transit)

            if bank_balance is not None:
                adjusted_diff = abs(adjusted_balance - bank_val)
                if adjusted_diff > 0.01:
                    result.add_warning(
                        f"Adjusted balance still differs: {adjusted_diff}",
                        'book_balance',
                        ValidationSeverity.MEDIUM
                    )

        except (ValueError, TypeError):
            result.add_error("Invalid numeric format for balance reconciliation", 'book_balance')

        return result

    def _validate_bank_master_data(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate bank references against master data."""
        result = ValidationResult()

        bank_name = data.get('bank_name', '').lower()
        bank_code = data.get('bank_code')

        if not bank_name and not bank_code:
            return result

        # Check bank name
        if bank_name and bank_name not in self.master_data.get('banks', {}):
            result.add_error(
                f"Unknown bank: {bank_name}",
                'bank_name',
                ValidationSeverity.HIGH
            )
        elif bank_name:
            bank_data = self.master_data['banks'][bank_name]

            # Check if bank is active
            if not bank_data.get('active', True):
                result.add_warning(
                    f"Bank {bank_name} is inactive",
                    'bank_name',
                    ValidationSeverity.MEDIUM
                )

            # Validate country consistency
            transaction_country = data.get('country', 'BR').upper()
            bank_country = bank_data.get('country', 'BR').upper()

            if transaction_country != bank_country:
                result.add_warning(
                    f"Bank country mismatch: transaction {transaction_country}, bank {bank_country}",
                    'bank_name',
                    ValidationSeverity.LOW
                )

        # Check bank code if provided
        if bank_code:
            # This would typically validate against a bank code master table
            result.add_metadata('bank_code_provided', bank_code)

        return result

    def _validate_currency_master_data(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate currency codes against master data."""
        result = ValidationResult()

        currency = data.get('currency', 'BRL').upper()

        if currency not in self.master_data.get('currencies', {}):
            result.add_error(
                f"Unknown currency code: {currency}",
                'currency',
                ValidationSeverity.HIGH
            )
        else:
            currency_data = self.master_data['currencies'][currency]

            # Check if currency is active
            if not currency_data.get('active', True):
                result.add_warning(
                    f"Currency {currency} is inactive",
                    'currency',
                    ValidationSeverity.MEDIUM
                )

            # Add currency metadata
            result.add_metadata('currency_name', currency_data.get('name'))
            result.add_metadata('currency_symbol', currency_data.get('symbol'))

        return result

    def _validate_organizational_hierarchy(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate organizational hierarchy relationships."""
        result = ValidationResult()

        org_hierarchy = self.hierarchical_relationships.get('organizational', {})
        levels = org_hierarchy.get('levels', [])
        relationships = org_hierarchy.get('relationships', {})

        # Check each level in the hierarchy
        for i, level in enumerate(levels):
            level_id = data.get(f'{level}_id')
            if level_id is None:
                continue

            # Check parent relationship
            if level in relationships:
                parent_level = relationships[level]
                parent_id = data.get(f'{parent_level}_id')

                if parent_id is None:
                    result.add_error(
                        f"Missing parent {parent_level} for {level} {level_id}",
                        f'{parent_level}_id',
                        ValidationSeverity.HIGH
                    )
                elif context and 'hierarchy_data' in context:
                    hierarchy_data = context['hierarchy_data']
                    if not self._validate_hierarchy_relationship(
                        hierarchy_data, parent_level, parent_id, level, level_id
                    ):
                        result.add_error(
                            f"Invalid hierarchy relationship: {parent_level} {parent_id} -> {level} {level_id}",
                            f'{level}_id',
                            ValidationSeverity.HIGH
                        )

        return result

    def _validate_category_hierarchy(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate category hierarchy relationships."""
        result = ValidationResult()

        category = data.get('category', '').lower()
        subcategory = data.get('subcategory', '').lower()

        if not category:
            return result

        categories_master = self.master_data.get('categories', {})

        if category not in categories_master:
            result.add_error(
                f"Unknown category: {category}",
                'category',
                ValidationSeverity.HIGH
            )
        else:
            category_data = categories_master[category]

            # Check subcategory relationship
            if subcategory:
                expected_parent = category_data.get('parent')
                if expected_parent and expected_parent != category:
                    result.add_error(
                        f"Invalid subcategory relationship: {subcategory} should belong to {expected_parent}",
                        'subcategory',
                        ValidationSeverity.HIGH
                    )

                # Check subcategory exists in master data
                if subcategory not in categories_master:
                    result.add_warning(
                        f"Unknown subcategory: {subcategory}",
                        'subcategory',
                        ValidationSeverity.MEDIUM
                    )

        return result

    def _validate_data_lineage(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Track and validate data lineage dependencies."""
        result = ValidationResult()

        if not context or 'data_lineage' not in context:
            return result

        data_lineage = context['data_lineage']
        record_id = data.get('id')
        source_system = data.get('source_system')

        if not record_id:
            return result

        # Check data lineage
        if record_id in data_lineage:
            lineage_info = data_lineage[record_id]

            # Validate source system consistency
            if source_system and lineage_info.get('source_system') != source_system:
                result.add_warning(
                    f"Source system mismatch in data lineage for record {record_id}",
                    'source_system',
                    ValidationSeverity.MEDIUM
                )

            # Check for data transformation issues
            transformations = lineage_info.get('transformations', [])
            if transformations:
                result.add_metadata('data_transformations_applied', len(transformations))
                result.add_metadata('transformation_chain', transformations)

            # Validate timestamp consistency
            created_at = data.get('created_at')
            lineage_created = lineage_info.get('created_at')

            if created_at and lineage_created:
                try:
                    created_dt = datetime.fromisoformat(created_at) if isinstance(created_at, str) else created_at
                    lineage_dt = datetime.fromisoformat(lineage_created) if isinstance(lineage_created, str) else lineage_created

                    if abs((created_dt - lineage_dt).total_seconds()) > 300:  # 5 minutes tolerance
                        result.add_warning(
                            "Data lineage timestamp inconsistency",
                            'created_at',
                            ValidationSeverity.LOW
                        )

                except (ValueError, TypeError):
                    pass

        else:
            result.add_warning(
                f"No data lineage information found for record {record_id}",
                'id',
                ValidationSeverity.LOW
            )

        return result

    def _validate_audit_trail_integrity(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate audit trail data integrity."""
        result = ValidationResult()

        if not context or 'audit_trail' not in context:
            return result

        audit_trail = context['audit_trail']
        record_id = data.get('id')

        if not record_id or record_id not in audit_trail:
            return result

        record_audit = audit_trail[record_id]

        # Validate audit trail completeness
        required_audit_fields = ['created_by', 'created_at', 'last_modified_by', 'last_modified_at']

        for field in required_audit_fields:
            if field not in record_audit:
                result.add_error(
                    f"Missing audit trail field: {field}",
                    field,
                    ValidationSeverity.HIGH
                )

        # Validate audit trail consistency
        if 'created_at' in record_audit and 'last_modified_at' in record_audit:
            try:
                created_at = record_audit['created_at']
                modified_at = record_audit['last_modified_at']

                created_dt = datetime.fromisoformat(created_at) if isinstance(created_at, str) else created_at
                modified_dt = datetime.fromisoformat(modified_at) if isinstance(modified_at, str) else modified_at

                if created_dt > modified_dt:
                    result.add_error(
                        "Audit trail inconsistency: created date after modified date",
                        'last_modified_at',
                        ValidationSeverity.HIGH
                    )

            except (ValueError, TypeError):
                result.add_error("Invalid audit trail date format", 'created_at')

        # Check for suspicious audit patterns
        modifications = record_audit.get('modification_count', 0)
        if modifications > 10:  # Arbitrary threshold
            result.add_warning(
                f"High number of modifications: {modifications}",
                'id',
                ValidationSeverity.MEDIUM
            )

        return result

    def _validate_hierarchy_relationship(self, hierarchy_data: Dict[str, Any],
                                       parent_level: str, parent_id: str,
                                       child_level: str, child_id: str) -> bool:
        """Validate hierarchical relationship in master data."""
        if parent_level not in hierarchy_data or parent_id not in hierarchy_data[parent_level]:
            return False

        parent_record = hierarchy_data[parent_level][parent_id]
        child_records = parent_record.get('children', {}).get(child_level, [])

        return child_id in child_records

    def add_rule(self, rule: ReferentialIntegrityRule):
        """Add a referential integrity validation rule."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added referential integrity rule: {rule.name} ({rule.rule_id})")

    def remove_rule(self, rule_id: str):
        """Remove a referential integrity validation rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed referential integrity rule: {rule_id}")

    def validate(self, data: Dict[str, Any], rule_group: str = None,
                context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate data against referential integrity rules.

        Args:
            data: Data to validate
            rule_group: Specific rule group to use
            context: Additional context for validation

        Returns:
            Validation result
        """
        result = ValidationResult()
        result.set_validator_info("ReferentialIntegrityValidator", "1.0")

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
                        logger.error(f"Error executing referential rule {rule_id}: {str(e)}")
                        result.add_error(f"Referential rule execution error: {rule_id}")

        # Set validation duration
        duration = time.time() - start_time
        result.set_validation_duration(duration * 1000)

        logger.info(f"Referential integrity validation completed in {duration:.3f}s with {len(result.errors)} errors, {len(result.warnings)} warnings")

        return result

    def get_rule(self, rule_id: str) -> Optional[ReferentialIntegrityRule]:
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
        logger.info(f"Created referential rule group: {group_name} with {len(rule_ids)} rules")

    def enable_rule(self, rule_id: str):
        """Enable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            logger.info(f"Enabled referential integrity rule: {rule_id}")

    def disable_rule(self, rule_id: str):
        """Disable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            logger.info(f"Disabled referential integrity rule: {rule_id}")

    def update_master_data(self, data_type: str, data: Dict[str, Any]):
        """Update master data for referential validation."""
        self.master_data[data_type] = data
        logger.info(f"Updated master data for: {data_type}")

    def add_foreign_key_mapping(self, table: str, mappings: Dict[str, str]):
        """Add foreign key mappings for a table."""
        self.foreign_key_mappings[table] = mappings
        logger.info(f"Added foreign key mappings for table: {table}")


# Global referential integrity validator instance
referential_integrity_validator = ReferentialIntegrityValidator()