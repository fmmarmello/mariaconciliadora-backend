"""
Enhanced Validation Result for Maria Conciliadora system.

This module provides:
- Structured validation results with detailed error information
- Validation status tracking (PASS/FAIL/WARNING)
- Field-level validation results
- Severity levels and recommendations
- Validation metadata and timestamps
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import json


class ValidationStatus(Enum):
    """Validation status enumeration."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


class ValidationSeverity(Enum):
    """Validation severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class FieldValidationResult:
    """Result of validation for a specific field."""

    def __init__(self, field_name: str, is_valid: bool = True,
                 errors: Optional[List[str]] = None,
                 warnings: Optional[List[str]] = None,
                 severity: ValidationSeverity = ValidationSeverity.MEDIUM):
        self.field_name = field_name
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.severity = severity
        self.validated_at = datetime.now()
        self.metadata: Dict[str, Any] = {}

    def add_error(self, error: str, severity: ValidationSeverity = ValidationSeverity.MEDIUM):
        """Add an error to the field validation result."""
        self.errors.append(error)
        self.is_valid = False
        if severity.value > self.severity.value:
            self.severity = severity

    def add_warning(self, warning: str):
        """Add a warning to the field validation result."""
        self.warnings.append(warning)

    def add_metadata(self, key: str, value: Any):
        """Add metadata to the field validation result."""
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'field_name': self.field_name,
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'severity': self.severity.value,
            'validated_at': self.validated_at.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldValidationResult':
        """Create from dictionary representation."""
        result = cls(
            field_name=data['field_name'],
            is_valid=data['is_valid'],
            errors=data['errors'],
            warnings=data['warnings'],
            severity=ValidationSeverity(data['severity'])
        )
        result.validated_at = datetime.fromisoformat(data['validated_at'])
        result.metadata = data['metadata']
        return result


class ValidationResult:
    """
    Comprehensive validation result with detailed error information,
    field-level results, and validation metadata.
    """

    def __init__(self, is_valid: bool = True,
                 status: ValidationStatus = ValidationStatus.PASS,
                 errors: Optional[List[str]] = None,
                 warnings: Optional[List[str]] = None,
                 recommendations: Optional[List[str]] = None):
        self.is_valid = is_valid
        self.status = status
        self.errors = errors or []
        self.warnings = warnings or []
        self.recommendations = recommendations or []
        self.field_results: Dict[str, FieldValidationResult] = {}
        self.metadata: Dict[str, Any] = {}
        self.validation_timestamp = datetime.now()
        self.validation_duration_ms: Optional[float] = None
        self.validator_name: Optional[str] = None
        self.validator_version: Optional[str] = None

    def add_error(self, error: str, field_name: Optional[str] = None,
                  severity: ValidationSeverity = ValidationSeverity.MEDIUM):
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False
        self.status = ValidationStatus.FAIL

        if field_name:
            if field_name not in self.field_results:
                self.field_results[field_name] = FieldValidationResult(field_name)
            self.field_results[field_name].add_error(error, severity)

    def add_warning(self, warning: str, field_name: Optional[str] = None):
        """Add a warning to the validation result."""
        self.warnings.append(warning)
        if self.status == ValidationStatus.PASS:
            self.status = ValidationStatus.WARNING

        if field_name:
            if field_name not in self.field_results:
                self.field_results[field_name] = FieldValidationResult(field_name)
            self.field_results[field_name].add_warning(warning)

    def add_recommendation(self, recommendation: str):
        """Add a recommendation to the validation result."""
        self.recommendations.append(recommendation)

    def add_field_result(self, field_result: FieldValidationResult):
        """Add a field validation result."""
        self.field_results[field_result.field_name] = field_result

        # Update overall result based on field result
        if not field_result.is_valid:
            self.is_valid = False
            self.status = ValidationStatus.FAIL
        elif field_result.warnings and self.status == ValidationStatus.PASS:
            self.status = ValidationStatus.WARNING

    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one."""
        # Merge basic properties
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.recommendations.extend(other.recommendations)

        # Update validity and status
        if not other.is_valid:
            self.is_valid = False
        if other.status == ValidationStatus.FAIL:
            self.status = ValidationStatus.FAIL
        elif other.status == ValidationStatus.WARNING and self.status == ValidationStatus.PASS:
            self.status = ValidationStatus.WARNING

        # Merge field results
        for field_name, field_result in other.field_results.items():
            if field_name in self.field_results:
                # Merge into existing field result
                existing = self.field_results[field_name]
                existing.errors.extend(field_result.errors)
                existing.warnings.extend(field_result.warnings)
                if not field_result.is_valid:
                    existing.is_valid = False
                if field_result.severity.value > existing.severity.value:
                    existing.severity = field_result.severity
                existing.metadata.update(field_result.metadata)
            else:
                # Add new field result
                self.field_results[field_name] = field_result

        # Merge metadata
        self.metadata.update(other.metadata)

    def add_metadata(self, key: str, value: Any):
        """Add metadata to the validation result."""
        self.metadata[key] = value

    def set_validation_duration(self, duration_ms: float):
        """Set the validation duration in milliseconds."""
        self.validation_duration_ms = duration_ms

    def set_validator_info(self, name: str, version: str = None):
        """Set validator information."""
        self.validator_name = name
        self.validator_version = version

    def get_field_errors(self, field_name: str) -> List[str]:
        """Get errors for a specific field."""
        if field_name in self.field_results:
            return self.field_results[field_name].errors
        return []

    def get_field_warnings(self, field_name: str) -> List[str]:
        """Get warnings for a specific field."""
        if field_name in self.field_results:
            return self.field_results[field_name].warnings
        return []

    def get_critical_errors(self) -> List[str]:
        """Get all critical errors."""
        critical_errors = []
        for error in self.errors:
            critical_errors.append(error)

        for field_result in self.field_results.values():
            if field_result.severity == ValidationSeverity.CRITICAL:
                critical_errors.extend(field_result.errors)

        return critical_errors

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the validation result."""
        return {
            'is_valid': self.is_valid,
            'status': self.status.value,
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'total_recommendations': len(self.recommendations),
            'fields_validated': len(self.field_results),
            'fields_with_errors': len([f for f in self.field_results.values() if not f.is_valid]),
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'validator_name': self.validator_name,
            'validator_version': self.validator_version
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'is_valid': self.is_valid,
            'status': self.status.value,
            'errors': self.errors,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'field_results': {name: result.to_dict() for name, result in self.field_results.items()},
            'metadata': self.metadata,
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'validation_duration_ms': self.validation_duration_ms,
            'validator_name': self.validator_name,
            'validator_version': self.validator_version
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create from dictionary representation."""
        result = cls(
            is_valid=data['is_valid'],
            status=ValidationStatus(data['status']),
            errors=data['errors'],
            warnings=data['warnings'],
            recommendations=data['recommendations']
        )

        # Restore field results
        for name, field_data in data['field_results'].items():
            result.field_results[name] = FieldValidationResult.from_dict(field_data)

        # Restore other attributes
        result.metadata = data['metadata']
        result.validation_timestamp = datetime.fromisoformat(data['validation_timestamp'])
        result.validation_duration_ms = data['validation_duration_ms']
        result.validator_name = data['validator_name']
        result.validator_version = data['validator_version']

        return result

    @classmethod
    def from_json(cls, json_str: str) -> 'ValidationResult':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation of the validation result."""
        summary = self.get_summary()
        return (f"ValidationResult(status={summary['status']}, "
                f"errors={summary['total_errors']}, "
                f"warnings={summary['total_warnings']}, "
                f"fields_validated={summary['fields_validated']})")

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()