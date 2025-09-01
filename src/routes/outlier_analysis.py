"""
Outlier Analysis API Endpoints for Maria Conciliadora

This module provides REST API endpoints for advanced statistical outlier detection
and analysis of financial transaction data.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
from src.models.transaction import Transaction, db
from src.services.ai_service import AIService
from src.utils.logging_config import get_logger, get_audit_logger
from src.utils.error_handler import handle_errors
from src.utils.exceptions import ValidationError, InsufficientDataError
from src.utils.validation_middleware import rate_limit, validate_input_fields

# Initialize loggers
logger = get_logger(__name__)
audit_logger = get_audit_logger()

outlier_bp = Blueprint('outlier_analysis', __name__)


@outlier_bp.route('/detect', methods=['POST'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
@validate_input_fields('method', 'include_contextual')
def detect_outliers():
    """
    Advanced outlier detection endpoint

    POST /api/outlier-analysis/detect
    Body: {
        "method": "ensemble|iqr|zscore|lof|isolation_forest|mahalanobis",
        "include_contextual": true|false,
        "bank_name": "optional_bank_filter",
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD",
        "limit": 1000
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        method = data.get('method', 'ensemble')
        include_contextual = data.get('include_contextual', True)
        bank_name = data.get('bank_name')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        limit = min(data.get('limit', 1000), 5000)  # Cap at 5000 for performance

        # Validate method
        valid_methods = ['ensemble', 'iqr', 'zscore', 'lof', 'isolation_forest', 'mahalanobis', 'ocsvm']
        if method not in valid_methods:
            raise ValidationError(f"Invalid method. Must be one of: {', '.join(valid_methods)}")

        # Validate dates
        start_date_obj = None
        end_date_obj = None

        if start_date:
            try:
                start_date_obj = datetime.fromisoformat(start_date)
            except ValueError:
                raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")

        if end_date:
            try:
                end_date_obj = datetime.fromisoformat(end_date)
            except ValueError:
                raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")

        if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
            raise ValidationError("start_date must be before end_date")

        # Build query
        query = Transaction.query

        if bank_name:
            query = query.filter(Transaction.bank_name == bank_name)

        if start_date_obj:
            query = query.filter(Transaction.date >= start_date_obj.date())

        if end_date_obj:
            query = query.filter(Transaction.date <= end_date_obj.date())

        # Get transactions
        transactions = query.order_by(Transaction.date.desc()).limit(limit).all()

        if not transactions:
            raise InsufficientDataError('outlier detection', 10, 0)

        # Convert to dictionaries
        transactions_data = [t.to_dict() for t in transactions]

        logger.info(f"Starting outlier detection with {method} method on {len(transactions_data)} transactions")

        # Perform outlier detection
        ai_service = AIService()
        analyzed_transactions = ai_service.detect_anomalies(
            transactions_data,
            method=method,
            include_contextual=include_contextual
        )

        # Count outliers
        outlier_count = sum(1 for t in analyzed_transactions if t.get('is_anomaly', False))

        audit_logger.log_ai_operation('advanced_outlier_detection', len(transactions_data), True, {
            'method': method,
            'outlier_count': outlier_count,
            'include_contextual': include_contextual
        })

        return jsonify({
            'success': True,
            'message': f'Outlier detection completed using {method} method',
            'data': {
                'method': method,
                'include_contextual': include_contextual,
                'total_transactions': len(analyzed_transactions),
                'outlier_count': outlier_count,
                'outlier_percentage': round(outlier_count / len(analyzed_transactions) * 100, 2),
                'transactions': analyzed_transactions,
                'filters': {
                    'bank_name': bank_name,
                    'start_date': start_date,
                    'end_date': end_date,
                    'limit': limit
                }
            }
        })

    except Exception as e:
        logger.error(f"Error in outlier detection endpoint: {str(e)}")
        audit_logger.log_ai_operation('outlier_detection', 0, False, error=str(e))
        return jsonify({'error': f'Outlier detection failed: {str(e)}'}), 500


@outlier_bp.route('/comprehensive', methods=['POST'])
@handle_errors
@rate_limit(max_requests=20, window_minutes=60)  # Lower limit for intensive operations
@validate_input_fields('export_path')
def comprehensive_analysis():
    """
    Comprehensive outlier analysis endpoint

    POST /api/outlier-analysis/comprehensive
    Body: {
        "ground_truth": [true, false, ...],  # optional
        "export_path": "/path/to/export.json",  # optional
        "bank_name": "optional_bank_filter",
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD"
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        ground_truth = data.get('ground_truth')
        export_path = data.get('export_path')
        bank_name = data.get('bank_name')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        # Validate dates
        start_date_obj = None
        end_date_obj = None

        if start_date:
            try:
                start_date_obj = datetime.fromisoformat(start_date)
            except ValueError:
                raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")

        if end_date:
            try:
                end_date_obj = datetime.fromisoformat(end_date)
            except ValueError:
                raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")

        if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
            raise ValidationError("start_date must be before end_date")

        # Build query
        query = Transaction.query

        if bank_name:
            query = query.filter(Transaction.bank_name == bank_name)

        if start_date_obj:
            query = query.filter(Transaction.date >= start_date_obj.date())

        if end_date_obj:
            query = query.filter(Transaction.date <= end_date_obj.date())

        # Get transactions
        transactions = query.order_by(Transaction.date.desc()).all()

        if not transactions:
            raise InsufficientDataError('comprehensive outlier analysis', 10, 0)

        # Convert to dictionaries
        transactions_data = [t.to_dict() for t in transactions]

        logger.info(f"Starting comprehensive outlier analysis on {len(transactions_data)} transactions")

        # Perform comprehensive analysis
        ai_service = AIService()
        analysis_result = ai_service.perform_comprehensive_outlier_analysis(
            transactions_data,
            ground_truth=ground_truth,
            export_path=export_path
        )

        if 'error' in analysis_result:
            return jsonify({'error': analysis_result['error']}), 500

        audit_logger.log_ai_operation('comprehensive_outlier_analysis', len(transactions_data), True, {
            'export_success': analysis_result.get('export_success', False),
            'export_path': export_path
        })

        return jsonify({
            'success': True,
            'message': 'Comprehensive outlier analysis completed successfully',
            'data': analysis_result
        })

    except Exception as e:
        logger.error(f"Error in comprehensive analysis endpoint: {str(e)}")
        audit_logger.log_ai_operation('comprehensive_outlier_analysis', 0, False, error=str(e))
        return jsonify({'error': f'Comprehensive analysis failed: {str(e)}'}), 500


@outlier_bp.route('/contextual', methods=['POST'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
@validate_input_fields('analysis_type')
def contextual_analysis():
    """
    Contextual outlier analysis endpoint

    POST /api/outlier-analysis/contextual
    Body: {
        "analysis_type": "category|temporal|frequency|merchant|balance|all",
        "bank_name": "optional_bank_filter",
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD"
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        analysis_type = data.get('analysis_type', 'all')
        bank_name = data.get('bank_name')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        # Validate analysis type
        valid_types = ['category', 'temporal', 'frequency', 'merchant', 'balance', 'all']
        if analysis_type not in valid_types:
            raise ValidationError(f"Invalid analysis_type. Must be one of: {', '.join(valid_types)}")

        # Validate dates
        start_date_obj = None
        end_date_obj = None

        if start_date:
            try:
                start_date_obj = datetime.fromisoformat(start_date)
            except ValueError:
                raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")

        if end_date:
            try:
                end_date_obj = datetime.fromisoformat(end_date)
            except ValueError:
                raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")

        if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
            raise ValidationError("start_date must be before end_date")

        # Build query
        query = Transaction.query

        if bank_name:
            query = query.filter(Transaction.bank_name == bank_name)

        if start_date_obj:
            query = query.filter(Transaction.date >= start_date_obj.date())

        if end_date_obj:
            query = query.filter(Transaction.date <= end_date_obj.date())

        # Get transactions
        transactions = query.order_by(Transaction.date.desc()).all()

        if not transactions:
            raise InsufficientDataError('contextual outlier analysis', 10, 0)

        # Convert to dictionaries
        transactions_data = [t.to_dict() for t in transactions]

        logger.info(f"Starting contextual outlier analysis ({analysis_type}) on {len(transactions_data)} transactions")

        # Perform contextual analysis
        ai_service = AIService()
        analysis_result = ai_service.detect_contextual_outliers(
            transactions_data,
            analysis_type=analysis_type
        )

        if 'error' in analysis_result:
            return jsonify({'error': analysis_result['error']}), 500

        audit_logger.log_ai_operation('contextual_outlier_analysis', len(transactions_data), True, {
            'analysis_type': analysis_type
        })

        return jsonify({
            'success': True,
            'message': f'Contextual outlier analysis ({analysis_type}) completed successfully',
            'data': analysis_result
        })

    except Exception as e:
        logger.error(f"Error in contextual analysis endpoint: {str(e)}")
        audit_logger.log_ai_operation('contextual_outlier_analysis', 0, False, error=str(e))
        return jsonify({'error': f'Contextual analysis failed: {str(e)}'}), 500


@outlier_bp.route('/compare-methods', methods=['POST'])
@handle_errors
@rate_limit(max_requests=25, window_minutes=60)
def compare_methods():
    """
    Compare outlier detection methods endpoint

    POST /api/outlier-analysis/compare-methods
    Body: {
        "methods": ["iqr", "zscore", "lof", "isolation_forest"],
        "ground_truth": [true, false, ...],  # optional
        "bank_name": "optional_bank_filter",
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD"
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        methods = data.get('methods', ['iqr', 'zscore', 'lof', 'isolation_forest'])
        ground_truth = data.get('ground_truth')
        bank_name = data.get('bank_name')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        # Validate methods
        valid_methods = ['iqr', 'zscore', 'lof', 'isolation_forest', 'mahalanobis', 'ocsvm', 'ensemble']
        invalid_methods = [m for m in methods if m not in valid_methods]
        if invalid_methods:
            raise ValidationError(f"Invalid methods: {', '.join(invalid_methods)}")

        # Validate dates
        start_date_obj = None
        end_date_obj = None

        if start_date:
            try:
                start_date_obj = datetime.fromisoformat(start_date)
            except ValueError:
                raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")

        if end_date:
            try:
                end_date_obj = datetime.fromisoformat(end_date)
            except ValueError:
                raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")

        if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
            raise ValidationError("start_date must be before end_date")

        # Build query
        query = Transaction.query

        if bank_name:
            query = query.filter(Transaction.bank_name == bank_name)

        if start_date_obj:
            query = query.filter(Transaction.date >= start_date_obj.date())

        if end_date_obj:
            query = query.filter(Transaction.date <= end_date_obj.date())

        # Get transactions
        transactions = query.order_by(Transaction.date.desc()).all()

        if not transactions:
            raise InsufficientDataError('method comparison', 10, 0)

        # Convert to dictionaries
        transactions_data = [t.to_dict() for t in transactions]

        logger.info(f"Starting method comparison analysis on {len(transactions_data)} transactions")

        # Perform method comparison
        ai_service = AIService()
        comparison_result = ai_service.compare_outlier_detection_methods(
            transactions_data,
            methods=methods,
            ground_truth=ground_truth
        )

        if 'error' in comparison_result:
            return jsonify({'error': comparison_result['error']}), 500

        audit_logger.log_ai_operation('method_comparison', len(transactions_data), True, {
            'methods_compared': methods
        })

        return jsonify({
            'success': True,
            'message': f'Method comparison completed for {len(methods)} methods',
            'data': comparison_result
        })

    except Exception as e:
        logger.error(f"Error in method comparison endpoint: {str(e)}")
        audit_logger.log_ai_operation('method_comparison', 0, False, error=str(e))
        return jsonify({'error': f'Method comparison failed: {str(e)}'}), 500


@outlier_bp.route('/config', methods=['GET'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
def get_config():
    """
    Get outlier detection configuration endpoint

    GET /api/outlier-analysis/config
    """
    try:
        ai_service = AIService()
        config = ai_service.get_outlier_detection_config()

        if 'error' in config:
            return jsonify({'error': config['error']}), 500

        return jsonify({
            'success': True,
            'message': 'Outlier detection configuration retrieved successfully',
            'data': config
        })

    except Exception as e:
        logger.error(f"Error getting outlier detection config: {str(e)}")
        return jsonify({'error': f'Failed to get configuration: {str(e)}'}), 500


@outlier_bp.route('/config', methods=['PUT'])
@handle_errors
@rate_limit(max_requests=20, window_minutes=60)
def update_config():
    """
    Update outlier detection configuration endpoint

    PUT /api/outlier-analysis/config
    Body: {
        "advanced_detector": {...},
        "contextual_detector": {...},
        "statistical_analyzer": {...}
    }
    """
    try:
        # Parse request data
        config_updates = request.get_json() or {}

        if not config_updates:
            raise ValidationError("No configuration updates provided")

        ai_service = AIService()
        result = ai_service.update_outlier_detection_config(config_updates)

        if 'error' in result:
            return jsonify({'error': result['error']}), 500

        audit_logger.log_ai_operation('config_update', 0, True, {
            'updated_components': result.get('updated_components', [])
        })

        return jsonify({
            'success': True,
            'message': 'Outlier detection configuration updated successfully',
            'data': result
        })

    except Exception as e:
        logger.error(f"Error updating outlier detection config: {str(e)}")
        audit_logger.log_ai_operation('config_update', 0, False, error=str(e))
        return jsonify({'error': f'Failed to update configuration: {str(e)}'}), 500


@outlier_bp.route('/summary', methods=['GET'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
def get_summary():
    """
    Get outlier analysis summary endpoint

    GET /api/outlier-analysis/summary
    Query params: bank_name, start_date, end_date
    """
    try:
        # Parse query parameters
        bank_name = request.args.get('bank')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # Validate dates
        start_date_obj = None
        end_date_obj = None

        if start_date:
            try:
                start_date_obj = datetime.fromisoformat(start_date)
            except ValueError:
                raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")

        if end_date:
            try:
                end_date_obj = datetime.fromisoformat(end_date)
            except ValueError:
                raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")

        if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
            raise ValidationError("start_date must be before end_date")

        # Build query
        query = Transaction.query

        if bank_name:
            query = query.filter(Transaction.bank_name == bank_name)

        if start_date_obj:
            query = query.filter(Transaction.date >= start_date_obj.date())

        if end_date_obj:
            query = query.filter(Transaction.date <= end_date_obj.date())

        # Get summary statistics
        total_transactions = query.count()
        anomaly_count = query.filter(Transaction.is_anomaly == True).count()

        # Recent outliers
        recent_outliers = query.filter(Transaction.is_anomaly == True)\
                              .order_by(Transaction.date.desc())\
                              .limit(10)\
                              .all()

        # Outliers by category
        category_stats = db.session.query(
            Transaction.category,
            db.func.count(Transaction.id).label('total'),
            db.func.sum(db.case((Transaction.is_anomaly == True, 1), else_=0)).label('outliers')
        ).filter(Transaction.category.isnot(None))\
         .group_by(Transaction.category)\
         .all()

        return jsonify({
            'success': True,
            'message': 'Outlier analysis summary retrieved successfully',
            'data': {
                'overview': {
                    'total_transactions': total_transactions,
                    'anomaly_count': anomaly_count,
                    'anomaly_percentage': round(anomaly_count / total_transactions * 100, 2) if total_transactions > 0 else 0
                },
                'categories': [
                    {
                        'name': cat.category,
                        'total_transactions': cat.total,
                        'outlier_count': cat.outliers,
                        'outlier_percentage': round(cat.outliers / cat.total * 100, 2) if cat.total > 0 else 0
                    } for cat in category_stats
                ],
                'recent_outliers': [outlier.to_dict() for outlier in recent_outliers],
                'filters': {
                    'bank_name': bank_name,
                    'start_date': start_date,
                    'end_date': end_date
                }
            }
        })

    except Exception as e:
        logger.error(f"Error getting outlier analysis summary: {str(e)}")
        return jsonify({'error': f'Failed to get summary: {str(e)}'}), 500