"""
Advanced Validation Engine for Maria Conciliadora system.

This module provides:
- Multi-layer validation framework with schema, business rules, and cross-field validation
- Temporal consistency validation
- Data type and format validation
- Extensible validation pipeline
- Comprehensive validation results
"""

import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .validation_result import ValidationResult, ValidationSeverity
from .business_rule_engine import business_rule_engine
from .schema_validator import schema_validator
from .validators import SecurityValidator, InputSanitizer
from .logging_config import get_logger

# Import data quality services
try:
    from src.services.data_completeness_analyzer import DataCompletenessAnalyzer
    COMPLETENESS_ANALYSIS_AVAILABLE = True
except ImportError:
    COMPLETENESS_ANALYSIS_AVAILABLE = False
    DataCompletenessAnalyzer = None

# Import outlier detection services
try:
    from src.services.ai_service import AIService
    OUTLIER_DETECTION_AVAILABLE = True
except ImportError:
    OUTLIER_DETECTION_AVAILABLE = False
    AIService = None

# Import text preprocessing services
try:
    from src.services.portuguese_preprocessor import PortugueseTextPreprocessor
    TEXT_PREPROCESSING_AVAILABLE = True
except ImportError:
    TEXT_PREPROCESSING_AVAILABLE = False
    PortugueseTextPreprocessor = None

logger = get_logger(__name__)


class ValidationLayer:
    """Represents a validation layer in the pipeline."""

    def __init__(self, name: str, validator: Callable, enabled: bool = True,
                 severity: ValidationSeverity = ValidationSeverity.MEDIUM):
        self.name = name
        self.validator = validator
        self.enabled = enabled
        self.severity = severity

    def execute(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Execute this validation layer."""
        if not self.enabled:
            return ValidationResult()

        try:
            start_time = time.time()
            result = self.validator(data, context)
            duration = time.time() - start_time

            result.set_validation_duration(duration * 1000)  # Convert to milliseconds
            return result

        except Exception as e:
            logger.error(f"Error in validation layer {self.name}: {str(e)}")
            result = ValidationResult()
            result.add_error(f"Validation layer error: {str(e)}")
            return result


class AdvancedValidationEngine:
    """
    Advanced validation engine that orchestrates multiple validation layers
    for comprehensive data validation.
    """

    def __init__(self):
        self.validation_layers: Dict[str, ValidationLayer] = {}
        self.validation_profiles: Dict[str, List[str]] = {}
        self._initialize_default_layers()
        self._initialize_default_profiles()

    def _initialize_default_layers(self):
        """Initialize default validation layers."""

        # Schema validation layer
        self.add_validation_layer(
            ValidationLayer(
                name="schema_validation",
                validator=self._schema_validation,
                severity=ValidationSeverity.HIGH
            )
        )

        # Business rules validation layer
        self.add_validation_layer(
            ValidationLayer(
                name="business_rules",
                validator=self._business_rules_validation,
                severity=ValidationSeverity.MEDIUM
            )
        )

        # Security validation layer
        self.add_validation_layer(
            ValidationLayer(
                name="security_validation",
                validator=self._security_validation,
                severity=ValidationSeverity.CRITICAL
            )
        )

        # Cross-field validation layer
        self.add_validation_layer(
            ValidationLayer(
                name="cross_field_validation",
                validator=self._cross_field_validation,
                severity=ValidationSeverity.MEDIUM
            )
        )

        # Temporal consistency validation layer
        self.add_validation_layer(
            ValidationLayer(
                name="temporal_consistency",
                validator=self._temporal_consistency_validation,
                severity=ValidationSeverity.MEDIUM
            )
        )

        # Data quality validation layer
        self.add_validation_layer(
            ValidationLayer(
                name="data_quality",
                validator=self._data_quality_validation,
                severity=ValidationSeverity.LOW
            )
        )

        # Text preprocessing validation layer (only if available)
        if TEXT_PREPROCESSING_AVAILABLE:
            self.add_validation_layer(
                ValidationLayer(
                    name="text_preprocessing",
                    validator=self._text_preprocessing_validation,
                    severity=ValidationSeverity.MEDIUM,
                    enabled=True
                )
            )

        # Outlier detection validation layer (only if available)
        if OUTLIER_DETECTION_AVAILABLE:
            self.add_validation_layer(
                ValidationLayer(
                    name="outlier_detection",
                    validator=self._outlier_detection_validation,
                    severity=ValidationSeverity.MEDIUM,
                    enabled=False  # Disabled by default for performance
                )
            )

    def _initialize_default_profiles(self):
        """Initialize default validation profiles."""

        # Transaction validation profile
        self.validation_profiles['transaction'] = [
            'security_validation',
            'schema_validation',
            'business_rules',
            'cross_field_validation',
            'temporal_consistency',
            'text_preprocessing',
            'data_quality'
        ]

        # Transaction validation with outlier detection profile
        self.validation_profiles['transaction_with_outliers'] = [
            'security_validation',
            'schema_validation',
            'business_rules',
            'cross_field_validation',
            'temporal_consistency',
            'data_quality',
            'outlier_detection'
        ]

        # Company financial validation profile
        self.validation_profiles['company_financial'] = [
            'security_validation',
            'schema_validation',
            'business_rules',
            'cross_field_validation',
            'temporal_consistency',
            'data_quality'
        ]

        # Bank statement validation profile
        self.validation_profiles['bank_statement'] = [
            'security_validation',
            'schema_validation',
            'business_rules',
            'cross_field_validation',
            'temporal_consistency',
            'data_quality'
        ]

        # File upload validation profile
        self.validation_profiles['file_upload'] = [
            'security_validation',
            'schema_validation'
        ]

        # API request validation profile
        self.validation_profiles['api_request'] = [
            'security_validation',
            'schema_validation'
        ]

    def add_validation_layer(self, layer: ValidationLayer):
        """Add a validation layer to the engine."""
        self.validation_layers[layer.name] = layer
        logger.info(f"Added validation layer: {layer.name}")

    def remove_validation_layer(self, layer_name: str):
        """Remove a validation layer from the engine."""
        if layer_name in self.validation_layers:
            del self.validation_layers[layer_name]
            logger.info(f"Removed validation layer: {layer_name}")

    def enable_layer(self, layer_name: str):
        """Enable a validation layer."""
        if layer_name in self.validation_layers:
            self.validation_layers[layer_name].enabled = True
            logger.info(f"Enabled validation layer: {layer_name}")

    def disable_layer(self, layer_name: str):
        """Disable a validation layer."""
        if layer_name in self.validation_layers:
            self.validation_layers[layer_name].enabled = False
            logger.info(f"Disabled validation layer: {layer_name}")

    def create_profile(self, profile_name: str, layer_names: List[str]):
        """Create a validation profile with specific layers."""
        # Validate that all layer names exist
        invalid_layers = [name for name in layer_names if name not in self.validation_layers]
        if invalid_layers:
            raise ValueError(f"Invalid layer names: {invalid_layers}")

        self.validation_profiles[profile_name] = layer_names
        logger.info(f"Created validation profile: {profile_name} with layers: {layer_names}")

    def validate(self, data: Any, profile: str = None, layer_names: List[str] = None,
                context: Optional[Dict[str, Any]] = None,
                parallel: bool = False) -> ValidationResult:
        """
        Validate data using specified profile or layers.

        Args:
            data: Data to validate
            profile: Validation profile name
            layer_names: Specific layer names to use
            context: Additional context for validation
            parallel: Whether to run validation layers in parallel

        Returns:
            Comprehensive validation result
        """
        result = ValidationResult()
        result.set_validator_info("AdvancedValidationEngine", "1.0")

        start_time = time.time()

        # Determine which layers to execute
        if layer_names:
            layers_to_execute = [name for name in layer_names if name in self.validation_layers]
        elif profile and profile in self.validation_profiles:
            layers_to_execute = self.validation_profiles[profile]
        else:
            # Default to all enabled layers
            layers_to_execute = [name for name, layer in self.validation_layers.items() if layer.enabled]

        if not layers_to_execute:
            result.add_warning("No validation layers configured or enabled")
            return result

        logger.debug(f"Executing validation layers: {layers_to_execute}")

        # Execute validation layers
        if parallel and len(layers_to_execute) > 1:
            layer_results = self._execute_parallel(layers_to_execute, data, context)
        else:
            layer_results = self._execute_sequential(layers_to_execute, data, context)

        # Merge all layer results
        for layer_result in layer_results:
            result.merge(layer_result)

        # Set overall validation duration
        total_duration = time.time() - start_time
        result.set_validation_duration(total_duration * 1000)

        # Add validation summary
        self._add_validation_summary(result, layers_to_execute)

        logger.info(f"Validation completed in {total_duration:.3f}s with {len(result.errors)} errors, {len(result.warnings)} warnings")

        return result

    def _execute_sequential(self, layer_names: List[str], data: Any,
                           context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """Execute validation layers sequentially."""
        results = []

        for layer_name in layer_names:
            layer = self.validation_layers[layer_name]
            layer_result = layer.execute(data, context)
            results.append(layer_result)

        return results

    def _execute_parallel(self, layer_names: List[str], data: Any,
                         context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """Execute validation layers in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=min(len(layer_names), 4)) as executor:
            # Submit all validation tasks
            future_to_layer = {
                executor.submit(self.validation_layers[layer_name].execute, data, context): layer_name
                for layer_name in layer_names
            }

            # Collect results as they complete
            for future in as_completed(future_to_layer):
                layer_name = future_to_layer[future]
                try:
                    layer_result = future.result()
                    results.append(layer_result)
                except Exception as e:
                    logger.error(f"Error executing layer {layer_name}: {str(e)}")
                    error_result = ValidationResult()
                    error_result.add_error(f"Layer execution error: {layer_name}")
                    results.append(error_result)

        return results

    def _schema_validation(self, data: Any, context: Optional[Dict[str, Any]]) -> ValidationResult:
        """Perform schema validation."""
        schema_name = context.get('schema_name', 'transaction') if context else 'transaction'
        return schema_validator.validate(data, schema_name, context)

    def _business_rules_validation(self, data: Any, context: Optional[Dict[str, Any]]) -> ValidationResult:
        """Perform business rules validation."""
        rule_group = context.get('rule_group', 'financial_transaction') if context else 'financial_transaction'
        return business_rule_engine.validate(data, rule_group, context)

    def _security_validation(self, data: Any, context: Optional[Dict[str, Any]]) -> ValidationResult:
        """Perform security validation."""
        result = ValidationResult()

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    security_result = SecurityValidator.validate_input_security(value, key)
                    result.merge(security_result)

        return result

    def _cross_field_validation(self, data: Any, context: Optional[Dict[str, Any]]) -> ValidationResult:
        """Perform cross-field validation."""
        result = ValidationResult()

        if not isinstance(data, dict):
            return result

        # Amount and transaction type consistency
        amount = data.get('amount')
        transaction_type = data.get('transaction_type', '').lower()

        if amount is not None and transaction_type:
            try:
                amount_val = float(amount)

                if transaction_type in ['debit', 'expense'] and amount_val > 0:
                    result.add_warning("Debit/expense transaction should have negative amount", 'amount')
                elif transaction_type in ['credit', 'income'] and amount_val < 0:
                    result.add_warning("Credit/income transaction should have positive amount", 'amount')

            except (ValueError, TypeError):
                pass

        # Date and balance consistency (for bank statements)
        date_val = data.get('date')
        balance = data.get('balance')

        if date_val and balance is not None:
            try:
                # Check if balance is reasonable for the date
                transaction_date = datetime.fromisoformat(date_val) if isinstance(date_val, str) else date_val
                today = datetime.now()

                if isinstance(transaction_date, datetime):
                    days_diff = (today - transaction_date).days

                    # Very old transactions shouldn't have very high balances (inflation adjustment)
                    if days_diff > 365*5 and abs(float(balance)) > 1000000:  # 5 years, 1M balance
                        result.add_warning("Unusually high balance for transaction age", 'balance')

            except (ValueError, TypeError):
                pass

        # Category and department consistency
        category = data.get('category', '').lower()
        department = data.get('department', '').lower()

        if category and department:
            # Define expected department-category relationships
            dept_category_map = {
                'it': ['software', 'hardware', 'consulting'],
                'finance': ['accounting', 'taxes', 'audit'],
                'sales': ['marketing', 'travel', 'commission'],
                'hr': ['salaries', 'training', 'benefits']
            }

            for dept, categories in dept_category_map.items():
                if dept in department and category not in categories:
                    result.add_warning(f"Category '{category}' is unusual for {dept} department", 'category')
                    break

        return result

    def _temporal_consistency_validation(self, data: Any, context: Optional[Dict[str, Any]]) -> ValidationResult:
        """Perform temporal consistency validation."""
        result = ValidationResult()

        if not isinstance(data, dict):
            return result

        # Validate date fields
        date_fields = ['date', 'transaction_date', 'created_at', 'updated_at']

        dates = {}
        for field in date_fields:
            date_val = data.get(field)
            if date_val:
                try:
                    if isinstance(date_val, str):
                        parsed_date = datetime.fromisoformat(date_val)
                    elif isinstance(date_val, datetime):
                        parsed_date = date_val
                    else:
                        continue

                    dates[field] = parsed_date
                except (ValueError, TypeError):
                    result.add_error(f"Invalid date format in {field}", field)

        # Check temporal consistency between dates
        if len(dates) > 1:
            # Created date should be before updated date
            if 'created_at' in dates and 'updated_at' in dates:
                if dates['created_at'] > dates['updated_at']:
                    result.add_error("Created date cannot be after updated date", 'created_at')

            # Transaction date should be reasonable
            if 'date' in dates:
                transaction_date = dates['date']
                today = datetime.now()

                # Not too far in the future
                if transaction_date > today:
                    days_future = (transaction_date - today).days
                    if days_future > 90:  # 90 days
                        result.add_error("Transaction date is too far in the future", 'date')

                # Not too far in the past
                if transaction_date < today:
                    days_past = (today - transaction_date).days
                    if days_past > 365*10:  # 10 years
                        result.add_warning("Transaction date is very old", 'date')

        # Validate amount changes over time (if historical data available)
        if context and 'historical_data' in context:
            self._validate_amount_trends(data, context['historical_data'], result)

        return result

    def _validate_amount_trends(self, current_data: Dict[str, Any],
                               historical_data: List[Dict[str, Any]],
                               result: ValidationResult):
        """Validate amount trends against historical data."""
        if not historical_data:
            return

        current_amount = current_data.get('amount')
        if current_amount is None:
            return

        try:
            current_amount_val = float(current_amount)

            # Calculate average of recent transactions
            recent_amounts = []
            for hist_item in historical_data[-10:]:  # Last 10 transactions
                hist_amount = hist_item.get('amount')
                if hist_amount is not None:
                    try:
                        recent_amounts.append(float(hist_amount))
                    except (ValueError, TypeError):
                        pass

            if recent_amounts:
                avg_recent = sum(recent_amounts) / len(recent_amounts)

                # Check for unusual deviation
                deviation = abs(current_amount_val - avg_recent)
                if avg_recent != 0:
                    deviation_percent = (deviation / abs(avg_recent)) * 100

                    if deviation_percent > 500:  # 500% deviation
                        result.add_warning("Amount deviates significantly from recent transactions", 'amount')

        except (ValueError, TypeError):
            pass

    def _outlier_detection_validation(self, data: Any, context: Optional[Dict[str, Any]]) -> ValidationResult:
        """Perform outlier detection validation."""
        result = ValidationResult()

        if not OUTLIER_DETECTION_AVAILABLE:
            result.add_warning("Outlier detection service not available")
            return result

        if not isinstance(data, dict):
            return result

        # Extract transaction data
        amount = data.get('amount')
        if amount is None:
            return result

        try:
            # Convert amount to float
            amount_val = float(amount)

            # Get outlier detection method from context
            method = context.get('outlier_method', 'iqr') if context else 'iqr'
            include_contextual = context.get('include_contextual', False) if context else False

            # Create AI service instance
            ai_service = AIService()

            # Create transaction data for outlier detection
            transaction_data = [data]

            # Perform outlier detection
            analyzed_transactions = ai_service.detect_anomalies(
                transaction_data,
                method=method,
                include_contextual=include_contextual
            )

            if analyzed_transactions and len(analyzed_transactions) > 0:
                transaction_result = analyzed_transactions[0]

                # Check if transaction is flagged as anomaly
                is_anomaly = transaction_result.get('is_anomaly', False)
                anomaly_score = transaction_result.get('anomaly_score', 0.0)

                if is_anomaly:
                    severity = "HIGH" if anomaly_score > 0.8 else "MEDIUM" if anomaly_score > 0.6 else "LOW"
                    result.add_warning(
                        f"Transaction flagged as outlier (severity: {severity}, score: {anomaly_score:.3f})",
                        'amount'
                    )

                    # Add metadata about outlier detection
                    result.add_metadata('outlier_detected', True)
                    result.add_metadata('outlier_score', anomaly_score)
                    result.add_metadata('outlier_method', method)
                    result.add_metadata('outlier_severity', severity)

                    # Add recommendation
                    result.add_recommendation(
                        f"Review transaction for potential data quality issues. "
                        f"Outlier score: {anomaly_score:.3f} using {method} method."
                    )
                else:
                    # Add metadata for non-outliers too
                    result.add_metadata('outlier_detected', False)
                    result.add_metadata('outlier_score', anomaly_score)
                    result.add_metadata('outlier_method', method)

        except Exception as e:
            logger.warning(f"Error in outlier detection validation: {str(e)}")
            result.add_warning(f"Outlier detection validation failed: {str(e)}")

        return result

    def _text_preprocessing_validation(self, data: Any, context: Optional[Dict[str, Any]]) -> ValidationResult:
        """Perform text preprocessing validation and enhancement."""
        result = ValidationResult()

        if not TEXT_PREPROCESSING_AVAILABLE:
            result.add_warning("Text preprocessing service not available")
            return result

        if not isinstance(data, dict):
            return result

        try:
            # Initialize text preprocessor if not already done
            if not hasattr(self, '_text_preprocessor'):
                self._text_preprocessor = PortugueseTextPreprocessor(use_advanced_pipeline=True)

            # Process text fields
            text_fields = ['description', 'memo', 'comment', 'notes', 'transaction_description']

            for field in text_fields:
                text_value = data.get(field)
                if text_value and isinstance(text_value, str) and text_value.strip():
                    # Get advanced preprocessing results
                    preprocessing_result = self._text_preprocessor.preprocess_with_advanced_features(text_value)

                    # Validate preprocessing quality
                    quality_score = preprocessing_result.get('quality_score', 0.0)

                    # Check for quality issues
                    if quality_score < 0.5:
                        result.add_warning(
                            f"Low text quality score ({quality_score:.2f}) for field '{field}'",
                            field,
                            ValidationSeverity.MEDIUM
                        )

                    # Check for potential data quality issues
                    original_length = len(text_value)
                    processed_length = len(preprocessing_result.get('processed_text', ''))

                    if processed_length < original_length * 0.3:
                        result.add_warning(
                            f"Significant text reduction in preprocessing ({processed_length}/{original_length} chars)",
                            field,
                            ValidationSeverity.MEDIUM
                        )

                    # Check for suspicious patterns
                    if self._has_suspicious_text_patterns(text_value):
                        result.add_warning(
                            f"Suspicious text patterns detected in field '{field}'",
                            field,
                            ValidationSeverity.LOW
                        )

                    # Add preprocessing metadata
                    result.add_metadata(f'{field}_preprocessing_quality', quality_score)
                    result.add_metadata(f'{field}_original_length', original_length)
                    result.add_metadata(f'{field}_processed_length', processed_length)

                    # Extract and validate entities
                    entities = preprocessing_result.get('entities', {})
                    if entities:
                        # Validate monetary amounts
                        monetary_values = entities.get('monetary_values', [])
                        if len(monetary_values) > 1:
                            result.add_warning(
                                f"Multiple monetary values detected in '{field}' - please verify",
                                field,
                                ValidationSeverity.LOW
                            )

                        # Validate dates
                        dates = entities.get('dates', [])
                        if len(dates) > 1:
                            result.add_warning(
                                f"Multiple dates detected in '{field}' - please verify",
                                field,
                                ValidationSeverity.LOW
                            )

                        result.add_metadata(f'{field}_extracted_entities', entities)

            # Overall text quality assessment
            if text_fields:
                processed_fields = [f for f in text_fields if data.get(f)]
                if processed_fields:
                    result.add_metadata('text_preprocessing_performed', True)
                    result.add_metadata('processed_text_fields', processed_fields)

                    # Add recommendation for text quality improvement
                    result.add_recommendation(
                        "Text preprocessing completed. Review any warnings for potential data quality improvements."
                    )

        except Exception as e:
            logger.warning(f"Error in text preprocessing validation: {str(e)}")
            result.add_warning(f"Text preprocessing validation failed: {str(e)}")

        return result

    def _has_suspicious_text_patterns(self, text: str) -> bool:
        """Check for suspicious text patterns that might indicate data quality issues."""
        try:
            # Check for excessive repetition
            if len(text) > 10:
                # Check for repeated characters
                for char in set(text.lower()):
                    if text.lower().count(char) > len(text) * 0.8:  # 80% same character
                        return True

                # Check for repeated words
                words = text.lower().split()
                if len(words) > 3:
                    word_counts = {}
                    for word in words:
                        word_counts[word] = word_counts.get(word, 0) + 1

                    # If any word appears more than 50% of the time
                    max_count = max(word_counts.values())
                    if max_count > len(words) * 0.5:
                        return True

            # Check for gibberish patterns (excessive consonants without vowels)
            consonants = 'bcdfghjklmnpqrstvwxyz'
            vowels = 'aeiou'

            text_lower = text.lower()
            consonant_count = sum(1 for c in text_lower if c in consonants)
            vowel_count = sum(1 for c in text_lower if c in vowels)

            if len(text_lower) > 10 and consonant_count > 0:
                vowel_ratio = vowel_count / consonant_count
                if vowel_ratio < 0.1:  # Very few vowels
                    return True

            return False

        except Exception:
            return False

    def _data_quality_validation(self, data: Any, context: Optional[Dict[str, Any]]) -> ValidationResult:
        """Perform comprehensive data quality validation including completeness analysis."""
        result = ValidationResult()

        if not isinstance(data, dict):
            return result

        # Perform completeness analysis if available
        if COMPLETENESS_ANALYSIS_AVAILABLE and context:
            completeness_result = self._perform_completeness_validation(data, context, result)
            result.merge(completeness_result)

        # Check for suspicious patterns
        for field, value in data.items():
            if isinstance(value, str):
                # Check for repeated characters
                if len(value) > 3 and len(set(value)) == 1:
                    result.add_warning(f"Field contains only repeated characters", field)

                # Check for very long words without spaces
                words = value.split()
                for word in words:
                    if len(word) > 50 and '_' not in word and '-' not in word:
                        result.add_warning(f"Very long word without separators: {word[:20]}...", field)

                # Check for excessive special characters
                special_chars = sum(1 for c in value if not c.isalnum() and not c.isspace())
                if len(value) > 0 and (special_chars / len(value)) > 0.5:
                    result.add_warning("Excessive special characters", field)

        # Check numeric field quality
        numeric_fields = ['amount', 'balance', 'value', 'price', 'cost']
        for field in numeric_fields:
            value = data.get(field)
            if value is not None:
                try:
                    num_val = float(value)

                    # Check for suspicious round numbers
                    if num_val > 1000 and num_val == int(num_val):
                        # Check if it's a multiple of 1000
                        if num_val % 1000 == 0:
                            result.add_warning("Large round number - please verify", field)

                    # Check for extremely small amounts
                    if 0 < abs(num_val) < 0.01:
                        result.add_warning("Very small amount - please verify", field)

                except (ValueError, TypeError):
                    pass

        return result

    def _perform_completeness_validation(self, data: Dict[str, Any],
                                       context: Dict[str, Any],
                                       result: ValidationResult) -> ValidationResult:
        """Perform completeness validation using the DataCompletenessAnalyzer."""
        completeness_result = ValidationResult()

        try:
            # Initialize completeness analyzer if not already done
            if not hasattr(self, '_completeness_analyzer'):
                self._completeness_analyzer = DataCompletenessAnalyzer()

            # Convert single record to list for analysis
            data_list = [data]

            # Perform field-level completeness analysis
            for field, value in data.items():
                field_completeness = 1.0 if value is not None else 0.0

                # Check against critical fields
                critical_fields = context.get('critical_fields', ['date', 'amount', 'description'])
                if field in critical_fields and field_completeness < 1.0:
                    completeness_result.add_error(
                        f"Critical field '{field}' is missing or null",
                        field,
                        ValidationSeverity.HIGH
                    )

                # Check for low completeness in important fields
                important_fields = ['date', 'amount', 'description', 'transaction_type', 'balance']
                if field in important_fields and field_completeness < 1.0:
                    completeness_result.add_warning(
                        f"Important field '{field}' is missing",
                        field,
                        ValidationSeverity.MEDIUM
                    )

            # Check for data consistency issues
            self._check_data_consistency(data, completeness_result)

            # Add completeness metadata
            completeness_result.add_metadata('completeness_analysis_performed', True)
            completeness_result.add_metadata('missing_fields',
                [field for field, value in data.items() if value is None or (isinstance(value, str) and value.strip() == '')])

        except Exception as e:
            logger.warning(f"Error in completeness validation: {str(e)}")
            completeness_result.add_warning(f"Completeness analysis failed: {str(e)}")

        return completeness_result

    def _check_data_consistency(self, data: Dict[str, Any], result: ValidationResult):
        """Check for data consistency issues that might indicate completeness problems."""
        try:
            # Check if amount is present but description is missing
            amount = data.get('amount')
            description = data.get('description', '').strip() if data.get('description') else ''

            if amount is not None and not description:
                result.add_warning(
                    "Transaction has amount but missing description",
                    'description',
                    ValidationSeverity.MEDIUM
                )

            # Check if date is present but amount is missing
            date = data.get('date')
            if date is not None and amount is None:
                result.add_warning(
                    "Transaction has date but missing amount",
                    'amount',
                    ValidationSeverity.MEDIUM
                )

            # Check for incomplete transaction records
            required_fields = ['date', 'amount', 'description']
            missing_required = [field for field in required_fields if not data.get(field)]

            if len(missing_required) > 1:
                result.add_error(
                    f"Multiple required fields missing: {', '.join(missing_required)}",
                    'general',
                    ValidationSeverity.HIGH
                )

        except Exception as e:
            logger.warning(f"Error in data consistency check: {str(e)}")

    def _add_validation_summary(self, result: ValidationResult, executed_layers: List[str]):
        """Add validation summary and recommendations."""
        # Add executed layers info
        result.add_metadata('executed_layers', executed_layers)
        result.add_metadata('total_layers', len(executed_layers))

        # Add recommendations based on validation results
        if result.errors:
            result.add_recommendation("Review and correct validation errors before proceeding")

        if len(result.warnings) > 5:
            result.add_recommendation("Consider reviewing the warnings for data quality improvements")

        if not result.errors and not result.warnings:
            result.add_recommendation("Data validation passed successfully")

    def validate_bulk(self, data_list: List[Any], profile: str = None,
                     layer_names: List[str] = None, context: Optional[Dict[str, Any]] = None,
                     parallel: bool = True, max_workers: int = 4) -> ValidationResult:
        """Validate a list of data items in bulk."""
        result = ValidationResult()
        result.set_validator_info("AdvancedValidationEngine", "1.0")

        if not data_list:
            result.add_warning("Empty data list provided for validation")
            return result

        start_time = time.time()

        # Validate items (parallel or sequential)
        if parallel and len(data_list) > 1:
            item_results = self._validate_bulk_parallel(data_list, profile, layer_names, context, max_workers)
        else:
            item_results = []
            for item in data_list:
                item_result = self.validate(item, profile, layer_names, context, parallel=False)
                item_results.append(item_result)

        # Merge all results
        for i, item_result in enumerate(item_results):
            # Prefix field names with array index
            field_names_to_update = list(item_result.field_results.keys())
            for field_name in field_names_to_update:
                prefixed_name = f"[{i}].{field_name}"
                field_result = item_result.field_results.pop(field_name)
                field_result.field_name = prefixed_name
                item_result.field_results[prefixed_name] = field_result

            result.merge(item_result)

        # Set bulk validation metadata
        total_duration = time.time() - start_time
        result.set_validation_duration(total_duration * 1000)
        result.add_metadata('bulk_validation', True)
        result.add_metadata('total_items', len(data_list))
        result.add_metadata('items_with_errors', len([r for r in item_results if not r.is_valid]))

        logger.info(f"Bulk validation completed for {len(data_list)} items in {total_duration:.3f}s")

        return result

    def _validate_bulk_parallel(self, data_list: List[Any], profile: str,
                              layer_names: List[str], context: Optional[Dict[str, Any]],
                              max_workers: int) -> List[ValidationResult]:
        """Validate data items in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit validation tasks
            future_to_index = {
                executor.submit(self.validate, item, profile, layer_names, context, False): i
                for i, item in enumerate(data_list)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    item_result = future.result()
                    results.append((index, item_result))
                except Exception as e:
                    logger.error(f"Error validating item {index}: {str(e)}")
                    error_result = ValidationResult()
                    error_result.add_error(f"Validation error for item {index}: {str(e)}")
                    results.append((index, error_result))

        # Sort results by original index
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation engine statistics."""
        return {
            'total_layers': len(self.validation_layers),
            'enabled_layers': len([l for l in self.validation_layers.values() if l.enabled]),
            'total_profiles': len(self.validation_profiles),
            'layers': list(self.validation_layers.keys()),
            'profiles': list(self.validation_profiles.keys())
        }


# Global advanced validation engine instance
advanced_validation_engine = AdvancedValidationEngine()