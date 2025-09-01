"""
Validation API endpoints for Maria Conciliadora system.

This module provides REST API endpoints for:
- Cross-field validation
- Business logic validation
- Financial business rules validation
- Temporal validation
- Referential integrity validation
- Validation rule management
- Validation reporting
"""

from flask import Blueprint, request, jsonify, g
from datetime import datetime
from typing import Dict, Any, Optional

from src.utils.logging_config import get_logger
from src.utils.error_handler import handle_errors
from src.utils.exceptions import ValidationError
from src.utils.validation_middleware import rate_limit, require_content_type, validate_input_fields

# Import validation engines
from src.utils.cross_field_validation_engine import cross_field_validation_engine
from src.utils.business_logic_validator import business_logic_validator
from src.utils.financial_business_rules import financial_business_rules
from src.utils.temporal_validation_engine import temporal_validation_engine
from src.utils.referential_integrity_validator import referential_integrity_validator

# Import existing validation engines for integration
from src.utils.advanced_validation_engine import advanced_validation_engine
from src.utils.business_rule_engine import business_rule_engine

# Import validation reporting service
from src.services.validation_reporting_service import validation_reporting_service

logger = get_logger(__name__)

validation_bp = Blueprint('validation', __name__)


@validation_bp.route('/validate/transaction', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@require_content_type('application/json')
@validate_input_fields('data')
def validate_transaction():
    """
    Endpoint for comprehensive transaction validation using all validation engines.
    """
    try:
        request_data = request.get_json()
        if not request_data or 'data' not in request_data:
            raise ValidationError("Transaction data is required")

        transaction_data = request_data['data']
        validation_profile = request_data.get('profile', 'transaction')
        context = request_data.get('context', {})

        logger.info(f"Starting comprehensive transaction validation for profile: {validation_profile}")

        # Initialize combined validation result
        combined_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metadata': {},
            'validation_duration_ms': 0,
            'engines_used': []
        }

        start_time = datetime.now()

        # 1. Advanced Validation Engine (existing)
        try:
            advanced_result = advanced_validation_engine.validate(
                transaction_data, profile=validation_profile, context=context
            )
            combined_result['engines_used'].append('advanced_validation_engine')
            combined_result['errors'].extend([f"Advanced: {err}" for err in advanced_result.errors])
            combined_result['warnings'].extend([f"Advanced: {warn}" for warn in advanced_result.warnings])
            if advanced_result.metadata:
                combined_result['metadata']['advanced_validation'] = advanced_result.metadata
        except Exception as e:
            logger.warning(f"Advanced validation engine error: {str(e)}")
            combined_result['warnings'].append(f"Advanced validation engine failed: {str(e)}")

        # 2. Cross-Field Validation Engine
        try:
            cross_field_result = cross_field_validation_engine.validate(
                transaction_data, rule_group='financial_transaction', context=context
            )
            combined_result['engines_used'].append('cross_field_validation_engine')
            combined_result['errors'].extend([f"Cross-field: {err}" for err in cross_field_result.errors])
            combined_result['warnings'].extend([f"Cross-field: {warn}" for warn in cross_field_result.warnings])
            if cross_field_result.metadata:
                combined_result['metadata']['cross_field_validation'] = cross_field_result.metadata
        except Exception as e:
            logger.warning(f"Cross-field validation engine error: {str(e)}")
            combined_result['warnings'].append(f"Cross-field validation engine failed: {str(e)}")

        # 3. Business Logic Validator
        try:
            business_logic_result = business_logic_validator.validate(
                transaction_data, rule_group='transaction_validation', context=context
            )
            combined_result['engines_used'].append('business_logic_validator')
            combined_result['errors'].extend([f"Business Logic: {err}" for err in business_logic_result.errors])
            combined_result['warnings'].extend([f"Business Logic: {warn}" for warn in business_logic_result.warnings])
            if business_logic_result.metadata:
                combined_result['metadata']['business_logic_validation'] = business_logic_result.metadata
        except Exception as e:
            logger.warning(f"Business logic validator error: {str(e)}")
            combined_result['warnings'].append(f"Business logic validator failed: {str(e)}")

        # 4. Financial Business Rules
        try:
            financial_result = financial_business_rules.validate(
                transaction_data, rule_group='transaction_processing', context=context
            )
            combined_result['engines_used'].append('financial_business_rules')
            combined_result['errors'].extend([f"Financial: {err}" for err in financial_result.errors])
            combined_result['warnings'].extend([f"Financial: {warn}" for warn in financial_result.warnings])
            if financial_result.metadata:
                combined_result['metadata']['financial_business_rules'] = financial_result.metadata
        except Exception as e:
            logger.warning(f"Financial business rules error: {str(e)}")
            combined_result['warnings'].append(f"Financial business rules failed: {str(e)}")

        # 5. Temporal Validation Engine
        try:
            temporal_result = temporal_validation_engine.validate(
                transaction_data, rule_group='date_validation', context=context
            )
            combined_result['engines_used'].append('temporal_validation_engine')
            combined_result['errors'].extend([f"Temporal: {err}" for err in temporal_result.errors])
            combined_result['warnings'].extend([f"Temporal: {warn}" for warn in temporal_result.warnings])
            if temporal_result.metadata:
                combined_result['metadata']['temporal_validation'] = temporal_result.metadata
        except Exception as e:
            logger.warning(f"Temporal validation engine error: {str(e)}")
            combined_result['warnings'].append(f"Temporal validation engine failed: {str(e)}")

        # 6. Referential Integrity Validator
        try:
            referential_result = referential_integrity_validator.validate(
                transaction_data, rule_group='foreign_key_validation', context=context
            )
            combined_result['engines_used'].append('referential_integrity_validator')
            combined_result['errors'].extend([f"Referential: {err}" for err in referential_result.errors])
            combined_result['warnings'].extend([f"Referential: {warn}" for warn in referential_result.warnings])
            if referential_result.metadata:
                combined_result['metadata']['referential_integrity'] = referential_result.metadata
        except Exception as e:
            logger.warning(f"Referential integrity validator error: {str(e)}")
            combined_result['warnings'].append(f"Referential integrity validator failed: {str(e)}")

        # Calculate overall validity and duration
        combined_result['is_valid'] = len(combined_result['errors']) == 0
        duration = (datetime.now() - start_time).total_seconds() * 1000
        combined_result['validation_duration_ms'] = duration

        logger.info(f"Comprehensive transaction validation completed in {duration:.2f}ms with {len(combined_result['errors'])} errors, {len(combined_result['warnings'])} warnings")

        return jsonify({
            'success': True,
            'data': combined_result
        })

    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'is_valid': False,
                'errors': [str(e)],
                'warnings': [],
                'engines_used': []
            }
        }), 400

    except Exception as e:
        logger.error(f"Unexpected error in transaction validation: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Validation failed: {str(e)}',
            'data': {
                'is_valid': False,
                'errors': [f'Unexpected error: {str(e)}'],
                'warnings': [],
                'engines_used': []
            }
        }), 500


@validation_bp.route('/validate/cross-field', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@require_content_type('application/json')
@validate_input_fields('data')
def validate_cross_field():
    """
    Endpoint for cross-field validation.
    """
    try:
        request_data = request.get_json()
        if not request_data or 'data' not in request_data:
            raise ValidationError("Data is required for cross-field validation")

        data = request_data['data']
        rule_group = request_data.get('rule_group', 'financial_transaction')
        context = request_data.get('context', {})

        logger.info(f"Cross-field validation request for rule group: {rule_group}")

        result = cross_field_validation_engine.validate(data, rule_group=rule_group, context=context)

        return jsonify({
            'success': True,
            'data': {
                'is_valid': result.is_valid,
                'errors': result.errors,
                'warnings': result.warnings,
                'metadata': result.metadata,
                'validation_duration_ms': result.validation_duration
            }
        })

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@validation_bp.route('/validate/business-logic', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@require_content_type('application/json')
@validate_input_fields('data')
def validate_business_logic():
    """
    Endpoint for business logic validation.
    """
    try:
        request_data = request.get_json()
        if not request_data or 'data' not in request_data:
            raise ValidationError("Data is required for business logic validation")

        data = request_data['data']
        rule_group = request_data.get('rule_group', 'transaction_validation')
        context = request_data.get('context', {})

        logger.info(f"Business logic validation request for rule group: {rule_group}")

        result = business_logic_validator.validate(data, rule_group=rule_group, context=context)

        return jsonify({
            'success': True,
            'data': {
                'is_valid': result.is_valid,
                'errors': result.errors,
                'warnings': result.warnings,
                'metadata': result.metadata,
                'validation_duration_ms': result.validation_duration
            }
        })

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@validation_bp.route('/validate/financial-rules', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@require_content_type('application/json')
@validate_input_fields('data')
def validate_financial_rules():
    """
    Endpoint for financial business rules validation.
    """
    try:
        request_data = request.get_json()
        if not request_data or 'data' not in request_data:
            raise ValidationError("Data is required for financial rules validation")

        data = request_data['data']
        rule_group = request_data.get('rule_group', 'transaction_processing')
        context = request_data.get('context', {})

        logger.info(f"Financial rules validation request for rule group: {rule_group}")

        result = financial_business_rules.validate(data, rule_group=rule_group, context=context)

        return jsonify({
            'success': True,
            'data': {
                'is_valid': result.is_valid,
                'errors': result.errors,
                'warnings': result.warnings,
                'metadata': result.metadata,
                'validation_duration_ms': result.validation_duration
            }
        })

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@validation_bp.route('/validate/temporal', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@require_content_type('application/json')
@validate_input_fields('data')
def validate_temporal():
    """
    Endpoint for temporal validation.
    """
    try:
        request_data = request.get_json()
        if not request_data or 'data' not in request_data:
            raise ValidationError("Data is required for temporal validation")

        data = request_data['data']
        rule_group = request_data.get('rule_group', 'date_validation')
        context = request_data.get('context', {})

        logger.info(f"Temporal validation request for rule group: {rule_group}")

        result = temporal_validation_engine.validate(data, rule_group=rule_group, context=context)

        return jsonify({
            'success': True,
            'data': {
                'is_valid': result.is_valid,
                'errors': result.errors,
                'warnings': result.warnings,
                'metadata': result.metadata,
                'validation_duration_ms': result.validation_duration
            }
        })

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@validation_bp.route('/validate/referential-integrity', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@require_content_type('application/json')
@validate_input_fields('data')
def validate_referential_integrity():
    """
    Endpoint for referential integrity validation.
    """
    try:
        request_data = request.get_json()
        if not request_data or 'data' not in request_data:
            raise ValidationError("Data is required for referential integrity validation")

        data = request_data['data']
        rule_group = request_data.get('rule_group', 'foreign_key_validation')
        context = request_data.get('context', {})

        logger.info(f"Referential integrity validation request for rule group: {rule_group}")

        result = referential_integrity_validator.validate(data, rule_group=rule_group, context=context)

        return jsonify({
            'success': True,
            'data': {
                'is_valid': result.is_valid,
                'errors': result.errors,
                'warnings': result.warnings,
                'metadata': result.metadata,
                'validation_duration_ms': result.validation_duration
            }
        })

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@validation_bp.route('/rules/cross-field', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_cross_field_rules():
    """
    Endpoint to get cross-field validation rules.
    """
    try:
        rule_group = request.args.get('rule_group')
        rules = cross_field_validation_engine.list_rules(rule_group=rule_group)

        return jsonify({
            'success': True,
            'data': {
                'rules': rules,
                'total_count': len(rules)
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve cross-field rules: {str(e)}'
        }), 500


@validation_bp.route('/rules/business-logic', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_business_logic_rules():
    """
    Endpoint to get business logic validation rules.
    """
    try:
        rule_group = request.args.get('rule_group')
        rules = business_logic_validator.list_rules(rule_group=rule_group)

        return jsonify({
            'success': True,
            'data': {
                'rules': rules,
                'total_count': len(rules)
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve business logic rules: {str(e)}'
        }), 500


@validation_bp.route('/rules/financial', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_financial_rules():
    """
    Endpoint to get financial business rules.
    """
    try:
        rule_group = request.args.get('rule_group')
        rules = financial_business_rules.list_rules(rule_group=rule_group)

        return jsonify({
            'success': True,
            'data': {
                'rules': rules,
                'total_count': len(rules)
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve financial rules: {str(e)}'
        }), 500


@validation_bp.route('/rules/temporal', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_temporal_rules():
    """
    Endpoint to get temporal validation rules.
    """
    try:
        rule_group = request.args.get('rule_group')
        rules = temporal_validation_engine.list_rules(rule_group=rule_group)

        return jsonify({
            'success': True,
            'data': {
                'rules': rules,
                'total_count': len(rules)
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve temporal rules: {str(e)}'
        }), 500


@validation_bp.route('/rules/referential-integrity', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_referential_integrity_rules():
    """
    Endpoint to get referential integrity validation rules.
    """
    try:
        rule_group = request.args.get('rule_group')
        rules = referential_integrity_validator.list_rules(rule_group=rule_group)

        return jsonify({
            'success': True,
            'data': {
                'rules': rules,
                'total_count': len(rules)
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve referential integrity rules: {str(e)}'
        }), 500


@validation_bp.route('/rules/groups', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_rule_groups():
    """
    Endpoint to get all validation rule groups.
    """
    try:
        rule_groups = {
            'cross_field': list(cross_field_validation_engine.rule_groups.keys()),
            'business_logic': list(business_logic_validator.rule_groups.keys()),
            'financial': list(financial_business_rules.rule_groups.keys()),
            'temporal': list(temporal_validation_engine.rule_groups.keys()),
            'referential_integrity': list(referential_integrity_validator.rule_groups.keys())
        }

        return jsonify({
            'success': True,
            'data': {
                'rule_groups': rule_groups
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve rule groups: {str(e)}'
        }), 500


@validation_bp.route('/validate/bulk', methods=['POST'])
@handle_errors
@rate_limit(max_requests=20, window_minutes=60)  # Lower limit for bulk operations
@require_content_type('application/json')
@validate_input_fields('data_list')
def validate_bulk():
    """
    Endpoint for bulk validation of multiple data items.
    """
    try:
        request_data = request.get_json()
        if not request_data or 'data_list' not in request_data:
            raise ValidationError("Data list is required for bulk validation")

        data_list = request_data['data_list']
        if not isinstance(data_list, list):
            raise ValidationError("data_list must be an array")

        if len(data_list) > 100:  # Limit bulk validation to 100 items
            raise ValidationError("Bulk validation limited to 100 items per request")

        validation_profile = request_data.get('profile', 'transaction')
        context = request_data.get('context', {})

        logger.info(f"Starting bulk validation for {len(data_list)} items with profile: {validation_profile}")

        # Use advanced validation engine for bulk validation
        result = advanced_validation_engine.validate_bulk(
            data_list, profile=validation_profile, context=context
        )

        return jsonify({
            'success': True,
            'data': {
                'total_items': len(data_list),
                'is_valid': result.is_valid,
                'errors': result.errors,
                'warnings': result.warnings,
                'metadata': result.metadata,
                'validation_duration_ms': result.validation_duration
            }
        })

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@validation_bp.route('/report/validation-stats', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
def get_validation_stats():
    """
    Endpoint to get validation system statistics.
    """
    try:
        stats = {
            'advanced_validation_engine': advanced_validation_engine.get_validation_stats(),
            'cross_field_validation_engine': {
                'total_rules': len(cross_field_validation_engine.rules),
                'enabled_rules': len([r for r in cross_field_validation_engine.rules.values() if r.enabled]),
                'total_groups': len(cross_field_validation_engine.rule_groups)
            },
            'business_logic_validator': {
                'total_rules': len(business_logic_validator.rules),
                'enabled_rules': len([r for r in business_logic_validator.rules.values() if r.enabled]),
                'total_groups': len(business_logic_validator.rule_groups)
            },
            'financial_business_rules': {
                'total_rules': len(financial_business_rules.rules),
                'enabled_rules': len([r for r in financial_business_rules.rules.values() if r.enabled]),
                'total_groups': len(financial_business_rules.rule_groups)
            },
            'temporal_validation_engine': {
                'total_rules': len(temporal_validation_engine.rules),
                'enabled_rules': len([r for r in temporal_validation_engine.rules.values() if r.enabled]),
                'total_groups': len(temporal_validation_engine.rule_groups)
            },
            'referential_integrity_validator': {
                'total_rules': len(referential_integrity_validator.rules),
                'enabled_rules': len([r for r in referential_integrity_validator.rules.values() if r.enabled]),
                'total_groups': len(referential_integrity_validator.rule_groups)
            }
        }

        return jsonify({
            'success': True,
            'data': {
                'validation_stats': stats,
                'timestamp': datetime.now().isoformat()
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve validation stats: {str(e)}'
        }), 500


@validation_bp.route('/health/validation-engines', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def health_check():
    """
    Endpoint for validation engines health check.
    """
    try:
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'engines': {}
        }

        # Test each validation engine
        engines_to_test = [
            ('advanced_validation_engine', advanced_validation_engine),
            ('cross_field_validation_engine', cross_field_validation_engine),
            ('business_logic_validator', business_logic_validator),
            ('financial_business_rules', financial_business_rules),
            ('temporal_validation_engine', temporal_validation_engine),
            ('referential_integrity_validator', referential_integrity_validator)
        ]

        for engine_name, engine in engines_to_test:
            try:
                # Simple health check - try to get stats
                if hasattr(engine, 'get_validation_stats'):
                    stats = engine.get_validation_stats()
                elif hasattr(engine, 'rules'):
                    stats = {'total_rules': len(engine.rules)}
                else:
                    stats = {'status': 'unknown'}

                health_status['engines'][engine_name] = {
                    'status': 'healthy',
                    'stats': stats
                }

            except Exception as e:
                health_status['engines'][engine_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }

        # Overall health status
        unhealthy_engines = [name for name, status in health_status['engines'].items()
                           if status['status'] != 'healthy']

        overall_status = 'unhealthy' if unhealthy_engines else 'healthy'

        return jsonify({
            'success': True,
            'data': {
                'overall_status': overall_status,
                'unhealthy_engines': unhealthy_engines,
                'health_details': health_status
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Health check failed: {str(e)}'
        }), 500


@validation_bp.route('/reports/generate', methods=['POST'])
@handle_errors
@rate_limit(max_requests=20, window_minutes=60)  # Lower limit for report generation
@require_content_type('application/json')
def generate_report():
    """
    Endpoint to generate validation reports.
    """
    try:
        request_data = request.get_json() or {}
        report_type = request_data.get('report_type', 'daily')
        start_date_str = request_data.get('start_date')
        end_date_str = request_data.get('end_date')
        format_type = request_data.get('format', 'json')

        # Parse dates if provided
        start_date = None
        end_date = None

        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str)
            except ValueError:
                raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")

        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str)
            except ValueError:
                raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")

        logger.info(f"Generating {report_type} validation report")

        # Generate report
        report = validation_reporting_service.generate_report(report_type, start_date, end_date)

        # Export in requested format
        if format_type == 'json':
            report_data = report.to_dict()
        else:
            report_data = validation_reporting_service.export_report(report, format_type)

        return jsonify({
            'success': True,
            'data': {
                'report': report_data,
                'format': format_type
            }
        })

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@validation_bp.route('/reports/dashboard', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_dashboard_data():
    """
    Endpoint to get validation dashboard data.
    """
    try:
        logger.info("Fetching validation dashboard data")

        dashboard_data = validation_reporting_service.get_dashboard_data()

        return jsonify({
            'success': True,
            'data': dashboard_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve dashboard data: {str(e)}'
        }), 500


@validation_bp.route('/reports/health', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_health_status():
    """
    Endpoint to get validation system health status.
    """
    try:
        logger.info("Checking validation system health")

        health_status = validation_reporting_service.get_health_status()

        return jsonify({
            'success': True,
            'data': health_status
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve health status: {str(e)}'
        }), 500


@validation_bp.route('/reports/export/<report_id>', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
def export_report(report_id: str):
    """
    Endpoint to export a specific report.
    """
    try:
        format_type = request.args.get('format', 'json')

        # For now, generate a new report based on report_id
        # In production, this would retrieve cached reports
        if report_id == 'daily':
            report = validation_reporting_service.generate_report('daily')
        elif report_id == 'weekly':
            report = validation_reporting_service.generate_report('weekly')
        elif report_id == 'monthly':
            report = validation_reporting_service.generate_report('monthly')
        else:
            raise ValidationError(f"Unknown report type: {report_id}")

        logger.info(f"Exporting {report_id} report in {format_type} format")

        # Export report
        exported_data = validation_reporting_service.export_report(report, format_type)

        return jsonify({
            'success': True,
            'data': {
                'report_id': report_id,
                'format': format_type,
                'content': exported_data
            }
        })

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@validation_bp.route('/analytics/trends', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
def get_validation_trends():
    """
    Endpoint to get validation trends and analytics.
    """
    try:
        days = request.args.get('days', 30, type=int)
        if days > 90:  # Limit to 90 days
            days = 90

        logger.info(f"Fetching validation trends for last {days} days")

        # Get trends data
        error_trends = validation_reporting_service._calculate_daily_trends('error_rate', days)
        validation_trends = validation_reporting_service._calculate_daily_trends('validations', days)

        return jsonify({
            'success': True,
            'data': {
                'period_days': days,
                'error_rate_trends': error_trends,
                'validation_count_trends': validation_trends,
                'generated_at': datetime.now().isoformat()
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve trends: {str(e)}'
        }), 500


@validation_bp.route('/analytics/issues', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
def get_common_issues():
    """
    Endpoint to get most common validation issues.
    """
    try:
        limit = request.args.get('limit', 10, type=int)
        if limit > 50:  # Limit results
            limit = 50

        logger.info(f"Fetching top {limit} common validation issues")

        # Get common issues
        common_errors = validation_reporting_service._get_most_common_issues('errors', limit)
        common_warnings = validation_reporting_service._get_most_common_issues('warnings', limit)
        problematic_fields = validation_reporting_service._get_problematic_fields(limit)

        return jsonify({
            'success': True,
            'data': {
                'common_errors': common_errors,
                'common_warnings': common_warnings,
                'problematic_fields': problematic_fields,
                'limit': limit,
                'generated_at': datetime.now().isoformat()
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve issues: {str(e)}'
        }), 500


@validation_bp.route('/analytics/performance', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
def get_performance_analytics():
    """
    Endpoint to get validation performance analytics.
    """
    try:
        logger.info("Fetching validation performance analytics")

        # Get performance data
        slowest_rules = validation_reporting_service._get_slowest_rules(10)
        engine_performance = validation_reporting_service._calculate_engine_performance()

        return jsonify({
            'success': True,
            'data': {
                'slowest_rules': slowest_rules,
                'engine_performance': engine_performance,
                'generated_at': datetime.now().isoformat()
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve performance analytics: {str(e)}'
        }), 500


@validation_bp.route('/record-result', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@require_content_type('application/json')
@validate_input_fields('result')
def record_validation_result():
    """
    Endpoint to record validation results for analytics.
    """
    try:
        request_data = request.get_json()
        if not request_data or 'result' not in request_data:
            raise ValidationError("Validation result data is required")

        result_data = request_data['result']
        engine_name = request_data.get('engine_name')
        context = request_data.get('context', {})

        # Reconstruct ValidationResult from data
        from src.utils.validation_result import ValidationResult
        validation_result = ValidationResult()
        validation_result.errors = result_data.get('errors', [])
        validation_result.warnings = result_data.get('warnings', [])
        validation_result.validation_duration_ms = result_data.get('validation_duration_ms', 0)

        # Record the result
        validation_reporting_service.record_validation_result(
            validation_result, engine_name, context
        )

        logger.info(f"Recorded validation result: {len(validation_result.errors)} errors, {len(validation_result.warnings)} warnings")

        return jsonify({
            'success': True,
            'message': 'Validation result recorded successfully'
        })

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400