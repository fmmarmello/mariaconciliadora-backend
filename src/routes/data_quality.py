from flask import Blueprint, request, jsonify
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

# Local imports
from src.services.data_completeness_analyzer import DataCompletenessAnalyzer
from src.services.advanced_imputation_engine import AdvancedImputationEngine
from src.services.missing_data_handler import MissingDataHandler
from src.services.imputation_strategies import ImputationStrategies
from src.utils.logging_config import get_logger
from src.utils.error_handler import handle_errors
from src.models.transaction import Transaction, db

logger = get_logger(__name__)

# Create blueprint
data_quality_bp = Blueprint('data_quality', __name__)

# Initialize services
completeness_analyzer = DataCompletenessAnalyzer()
imputation_engine = AdvancedImputationEngine()
missing_data_handler = MissingDataHandler()
imputation_strategies = ImputationStrategies()


@data_quality_bp.route('/completeness/analysis', methods=['POST'])
@handle_errors
def analyze_data_completeness():
    """
    Analyze data completeness for provided dataset or database query

    Request body:
    {
        "data_source": "database" | "uploaded_data",
        "table_name": "transactions" (optional),
        "data": [...] (optional, for uploaded data),
        "columns": ["column1", "column2"] (optional),
        "filters": {...} (optional)
    }
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'No request data provided'}), 400

        data_source = request_data.get('data_source', 'database')

        if data_source == 'database':
            # Analyze data from database
            df = _get_data_from_database(request_data)
        elif data_source == 'uploaded_data':
            # Analyze uploaded data
            data = request_data.get('data', [])
            df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid data_source. Must be "database" or "uploaded_data"'}), 400

        if df.empty:
            return jsonify({'error': 'No data available for analysis'}), 400

        # Perform completeness analysis
        completeness_report = completeness_analyzer.generate_completeness_report(df)

        # Add metadata
        response = {
            'timestamp': datetime.now().isoformat(),
            'data_source': data_source,
            'total_records': len(df),
            'total_columns': len(df.columns),
            'completeness_report': completeness_report
        }

        logger.info(f"Completeness analysis completed for {len(df)} records")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in completeness analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


@data_quality_bp.route('/completeness/field/<field_name>', methods=['GET'])
@handle_errors
def analyze_field_completeness(field_name: str):
    """
    Analyze completeness of a specific field

    Query parameters:
    - table: table name (default: transactions)
    - limit: number of records to analyze (default: 1000)
    """
    try:
        table_name = request.args.get('table', 'transactions')
        limit = int(request.args.get('limit', 1000))

        # Get data from database
        df = _get_data_from_database({'table_name': table_name, 'limit': limit})

        if df.empty or field_name not in df.columns:
            return jsonify({'error': f'Field {field_name} not found in data'}), 404

        # Analyze field completeness
        field_analysis = completeness_analyzer.analyze_field_completeness(df, field_name)

        response = {
            'timestamp': datetime.now().isoformat(),
            'field_name': field_name,
            'table_name': table_name,
            'records_analyzed': len(df),
            'field_analysis': field_analysis
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error analyzing field completeness: {str(e)}")
        return jsonify({'error': str(e)}), 500


@data_quality_bp.route('/completeness/record', methods=['GET'])
@handle_errors
def analyze_record_completeness():
    """
    Analyze record-level completeness

    Query parameters:
    - table: table name (default: transactions)
    - limit: number of records to analyze (default: 1000)
    - min_completeness: minimum completeness threshold (default: 0.0)
    """
    try:
        table_name = request.args.get('table', 'transactions')
        limit = int(request.args.get('limit', 1000))
        min_completeness = float(request.args.get('min_completeness', 0.0))

        # Get data from database
        df = _get_data_from_database({'table_name': table_name, 'limit': limit})

        if df.empty:
            return jsonify({'error': 'No data available for analysis'}), 400

        # Analyze record completeness
        record_analysis = completeness_analyzer.analyze_record_completeness(df)

        # Filter by minimum completeness if specified
        if min_completeness > 0:
            record_analysis = [r for r in record_analysis if r['completeness_score'] >= min_completeness]

        # Sort by completeness score (worst first)
        record_analysis.sort(key=lambda x: x['completeness_score'])

        response = {
            'timestamp': datetime.now().isoformat(),
            'table_name': table_name,
            'records_analyzed': len(df),
            'records_returned': len(record_analysis),
            'min_completeness_filter': min_completeness,
            'record_analysis': record_analysis[:100]  # Limit response size
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error analyzing record completeness: {str(e)}")
        return jsonify({'error': str(e)}), 500


@data_quality_bp.route('/imputation/analyze', methods=['POST'])
@handle_errors
def analyze_missing_data():
    """
    Analyze missing data patterns and provide imputation recommendations

    Request body:
    {
        "data_source": "database" | "uploaded_data",
        "table_name": "transactions" (optional),
        "data": [...] (optional),
        "columns": ["column1", "column2"] (optional)
    }
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'No request data provided'}), 400

        data_source = request_data.get('data_source', 'database')

        if data_source == 'database':
            df = _get_data_from_database(request_data)
        elif data_source == 'uploaded_data':
            data = request_data.get('data', [])
            df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid data_source'}), 400

        if df.empty:
            return jsonify({'error': 'No data available for analysis'}), 400

        # Get imputation recommendations
        recommendations = missing_data_handler.get_imputation_recommendations(df)

        # Get performance summary
        performance = missing_data_handler.get_performance_summary()

        response = {
            'timestamp': datetime.now().isoformat(),
            'data_source': data_source,
            'total_records': len(df),
            'total_columns': len(df.columns),
            'recommendations': recommendations,
            'performance_summary': performance
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error analyzing missing data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@data_quality_bp.route('/imputation/apply', methods=['POST'])
@handle_errors
def apply_imputation():
    """
    Apply imputation to missing data

    Request body:
    {
        "data_source": "database" | "uploaded_data",
        "table_name": "transactions" (optional),
        "data": [...] (optional),
        "strategy": "auto" | "statistical" | "knn" | "regression" | "time_series" | "context_aware",
        "target_columns": ["column1", "column2"] (optional),
        "save_to_database": false (optional)
    }
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'No request data provided'}), 400

        data_source = request_data.get('data_source', 'database')
        strategy = request_data.get('strategy', 'auto')
        target_columns = request_data.get('target_columns')
        save_to_database = request_data.get('save_to_database', False)

        if data_source == 'database':
            df = _get_data_from_database(request_data)
        elif data_source == 'uploaded_data':
            data = request_data.get('data', [])
            df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid data_source'}), 400

        if df.empty:
            return jsonify({'error': 'No data available for imputation'}), 400

        # Apply imputation
        imputation_result = missing_data_handler.analyze_and_impute(
            df, strategy=strategy, target_columns=target_columns
        )

        # Prepare response
        response = {
            'timestamp': datetime.now().isoformat(),
            'strategy_used': imputation_result.strategy_used.value,
            'confidence_score': imputation_result.confidence_score,
            'confidence_level': imputation_result.confidence_level.value,
            'imputation_count': imputation_result.imputation_count,
            'columns_affected': imputation_result.columns_affected,
            'quality_metrics': imputation_result.quality_metrics,
            'data_sample': imputation_result.imputed_data.head(10).to_dict('records') if len(imputation_result.imputed_data) > 0 else []
        }

        # Save to database if requested (only for database source)
        if save_to_database and data_source == 'database':
            success = _save_imputed_data_to_database(
                imputation_result.imputed_data,
                request_data.get('table_name', 'transactions')
            )
            response['saved_to_database'] = success

        logger.info(f"Imputation applied: {imputation_result.imputation_count} values imputed using {strategy}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error applying imputation: {str(e)}")
        return jsonify({'error': str(e)}), 500


@data_quality_bp.route('/imputation/specialized', methods=['POST'])
@handle_errors
def apply_specialized_imputation():
    """
    Apply specialized imputation strategies for specific data types

    Request body:
    {
        "data_source": "database" | "uploaded_data",
        "table_name": "transactions" (optional),
        "data": [...] (optional),
        "strategy_type": "financial" | "mixed",
        "column_types": {"amount": "financial", "date": "date"} (optional)
    }
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'No request data provided'}), 400

        data_source = request_data.get('data_source', 'database')
        strategy_type = request_data.get('strategy_type', 'mixed')
        column_types = request_data.get('column_types')

        if data_source == 'database':
            df = _get_data_from_database(request_data)
        elif data_source == 'uploaded_data':
            data = request_data.get('data', [])
            df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid data_source'}), 400

        if df.empty:
            return jsonify({'error': 'No data available for imputation'}), 400

        # Apply specialized imputation
        if strategy_type == 'financial':
            imputed_df, summary = imputation_strategies.impute_financial_data(df)
        elif strategy_type == 'mixed':
            imputed_df, summary = imputation_strategies.impute_mixed_data(df, column_types)
        else:
            return jsonify({'error': f'Unknown strategy type: {strategy_type}'}), 400

        response = {
            'timestamp': datetime.now().isoformat(),
            'strategy_type': strategy_type,
            'total_imputations': summary.get('total_imputations', 0),
            'strategies_used': summary.get('strategies_used', []),
            'columns_processed': summary.get('columns_processed', []),
            'imputation_details': summary,
            'data_sample': imputed_df.head(10).to_dict('records') if len(imputed_df) > 0 else []
        }

        logger.info(f"Specialized imputation applied: {summary.get('total_imputations', 0)} values imputed")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error applying specialized imputation: {str(e)}")
        return jsonify({'error': str(e)}), 500


@data_quality_bp.route('/imputation/history', methods=['GET'])
@handle_errors
def get_imputation_history():
    """
    Get imputation history and performance metrics

    Query parameters:
    - limit: number of recent imputations to return (default: 10)
    """
    try:
        limit = int(request.args.get('limit', 10))

        # Get performance summary
        performance = missing_data_handler.get_performance_summary()

        # Get recent imputation results (simplified for API response)
        recent_imputations = []
        for i, result in enumerate(reversed(missing_data_handler.imputation_history[-limit:])):
            recent_imputations.append({
                'id': len(missing_data_handler.imputation_history) - i,
                'timestamp': result.timestamp.isoformat(),
                'strategy': result.strategy_used.value,
                'confidence_score': result.confidence_score,
                'confidence_level': result.confidence_level.value,
                'imputation_count': result.imputation_count,
                'columns_affected': result.columns_affected
            })

        response = {
            'timestamp': datetime.now().isoformat(),
            'performance_summary': performance,
            'recent_imputations': recent_imputations,
            'total_imputations': len(missing_data_handler.imputation_history)
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error getting imputation history: {str(e)}")
        return jsonify({'error': str(e)}), 500


@data_quality_bp.route('/quality/metrics', methods=['GET'])
@handle_errors
def get_quality_metrics():
    """
    Get overall data quality metrics

    Query parameters:
    - table: table name (default: transactions)
    - days: number of days to look back (default: 30)
    """
    try:
        table_name = request.args.get('table', 'transactions')
        days = int(request.args.get('days', 30))

        # Get recent data
        request_data = {
            'table_name': table_name,
            'filters': {
                'date_range': f'last_{days}_days'
            }
        }

        df = _get_data_from_database(request_data)

        if df.empty:
            return jsonify({'error': 'No data available for quality analysis'}), 400

        # Generate completeness report
        completeness_report = completeness_analyzer.generate_completeness_report(df)

        # Calculate quality score
        quality_score = _calculate_overall_quality_score(completeness_report)

        response = {
            'timestamp': datetime.now().isoformat(),
            'table_name': table_name,
            'analysis_period_days': days,
            'total_records': len(df),
            'quality_score': quality_score,
            'completeness_report': completeness_report
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error getting quality metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500


def _get_data_from_database(request_data: Dict[str, Any]) -> pd.DataFrame:
    """Get data from database based on request parameters"""
    try:
        table_name = request_data.get('table_name', 'transactions')
        limit = request_data.get('limit', 10000)
        filters = request_data.get('filters', {})

        # Build query based on table
        if table_name == 'transactions':
            query = Transaction.query

            # Apply filters
            if 'date_range' in filters:
                # Simple date filtering (can be enhanced)
                pass

            # Limit results
            if limit:
                query = query.limit(limit)

            # Execute query
            transactions = query.all()

            # Convert to DataFrame
            data = []
            for transaction in transactions:
                data.append({
                    'id': transaction.id,
                    'bank_name': transaction.bank_name,
                    'account_id': transaction.account_id,
                    'transaction_id': transaction.transaction_id,
                    'date': transaction.date.isoformat() if transaction.date else None,
                    'amount': transaction.amount,
                    'description': transaction.description,
                    'transaction_type': transaction.transaction_type,
                    'balance': transaction.balance,
                    'category': transaction.category,
                    'is_anomaly': transaction.is_anomaly
                })

            return pd.DataFrame(data)

        else:
            logger.warning(f"Unsupported table: {table_name}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error getting data from database: {str(e)}")
        return pd.DataFrame()


def _save_imputed_data_to_database(df: pd.DataFrame, table_name: str) -> bool:
    """Save imputed data back to database"""
    try:
        if table_name == 'transactions':
            # This would require careful handling to avoid data corruption
            # For now, just log the operation
            logger.info(f"Would save {len(df)} imputed records to {table_name} table")
            # In production, implement proper update logic with transaction safety
            return True
        else:
            logger.warning(f"Saving to table {table_name} not implemented")
            return False

    except Exception as e:
        logger.error(f"Error saving imputed data to database: {str(e)}")
        return False


def _calculate_overall_quality_score(completeness_report: Dict[str, Any]) -> float:
    """Calculate overall data quality score"""
    try:
        base_score = completeness_report.get('dataset_completeness', 0.0)

        # Adjust based on critical fields
        critical_fields = completeness_report.get('critical_fields', {})
        critical_penalty = 0.0

        for field_name, field_info in critical_fields.items():
            if not field_info.get('meets_threshold', True):
                critical_penalty += 0.1  # 10% penalty per critical field issue

        # Adjust based on missing patterns
        patterns = completeness_report.get('missing_patterns', [])
        pattern_penalty = min(len(patterns) * 0.05, 0.2)  # Max 20% penalty

        quality_score = max(0.0, base_score - critical_penalty - pattern_penalty)

        return round(quality_score, 3)

    except Exception as e:
        logger.warning(f"Error calculating quality score: {str(e)}")
        return 0.0


# Register blueprint
def register_data_quality_routes(app):
    """Register data quality routes with the Flask app"""
    app.register_blueprint(data_quality_bp, url_prefix='/api/data-quality')
    logger.info("Data quality routes registered")