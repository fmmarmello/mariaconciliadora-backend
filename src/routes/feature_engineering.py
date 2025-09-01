from flask import Blueprint, request, jsonify
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd

from src.services.enhanced_feature_engineer import EnhancedFeatureEngineer
from src.services.advanced_text_feature_extractor import AdvancedTextFeatureExtractor
from src.services.temporal_feature_enhancer import TemporalFeatureEnhancer
from src.services.financial_feature_engineer import FinancialFeatureEngineer
from src.services.quality_assured_feature_pipeline import QualityAssuredFeaturePipeline
from src.services.smote_implementation import SMOTEImplementation
from src.services.data_augmentation_pipeline import DataAugmentationPipeline
from src.utils.logging_config import get_logger
from src.utils.error_handler import handle_errors
from src.utils.validation_middleware import rate_limit, validate_input_fields

logger = get_logger(__name__)

feature_engineering_bp = Blueprint('feature_engineering', __name__)

# Initialize feature engineering components
enhanced_engineer = EnhancedFeatureEngineer()
text_extractor = AdvancedTextFeatureExtractor()
temporal_enhancer = TemporalFeatureEnhancer()
financial_engineer = FinancialFeatureEngineer()
quality_pipeline = QualityAssuredFeaturePipeline()
smote_engine = SMOTEImplementation()
augmentation_pipeline = DataAugmentationPipeline()


@feature_engineering_bp.route('/enhanced/features', methods=['POST'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
@validate_input_fields('transactions')
def create_enhanced_features():
    """
    Create comprehensive enhanced features from transaction data

    Expected JSON payload:
    {
        "transactions": [
            {
                "description": "Compra no mercado",
                "amount": 150.50,
                "date": "2024-01-15",
                "category": "Alimentação"
            }
        ],
        "target_column": "category",
        "feature_types": ["text", "temporal", "financial"],
        "config": {
            "text_processing": {"use_advanced_portuguese": true},
            "data_augmentation": {"enabled": true}
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        transactions = data.get('transactions', [])
        target_column = data.get('target_column')
        feature_types = data.get('feature_types', ['text', 'temporal', 'financial'])
        config = data.get('config', {})

        if not transactions or not isinstance(transactions, list):
            return jsonify({'error': 'transactions must be a non-empty list'}), 400

        if len(transactions) > 10000:  # Limit dataset size
            return jsonify({'error': 'Dataset size cannot exceed 10,000 transactions'}), 400

        # Update configuration if provided
        if config:
            enhanced_engineer.config.update(config)

        # Create enhanced features
        features, feature_names, quality_report = enhanced_engineer.create_enhanced_features(
            transactions, target_column
        )

        # Convert numpy arrays to lists for JSON serialization
        features_list = features.tolist() if hasattr(features, 'tolist') else features

        logger.info(f"Enhanced feature engineering completed for {len(transactions)} transactions")

        return jsonify({
            'success': True,
            'data': {
                'features': features_list,
                'feature_names': feature_names,
                'feature_matrix_shape': list(features.shape) if hasattr(features, 'shape') else None,
                'quality_report': quality_report,
                'processing_timestamp': datetime.utcnow().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Error creating enhanced features: {str(e)}")
        return jsonify({'error': f'Enhanced feature engineering failed: {str(e)}'}), 500


@feature_engineering_bp.route('/text/features', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@validate_input_fields('texts')
def extract_text_features():
    """
    Extract advanced text features from a list of texts

    Expected JSON payload:
    {
        "texts": ["Texto em português 1", "Texto em português 2"],
        "feature_types": ["embeddings", "tfidf", "linguistic", "financial"],
        "config": {
            "portuguese_processing": {"use_advanced_preprocessor": true}
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        texts = data.get('texts', [])
        feature_types = data.get('feature_types', ['embeddings', 'linguistic'])
        config = data.get('config', {})

        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'texts must be a non-empty list'}), 400

        if len(texts) > 1000:  # Limit batch size
            return jsonify({'error': 'Batch size cannot exceed 1000 texts'}), 400

        # Update configuration if provided
        if config:
            text_extractor.config.update(config)

        # Extract text features
        features_dict = text_extractor.extract_text_features(texts, feature_types)

        # Convert numpy arrays to lists for JSON serialization
        serializable_features = {}
        for key, value in features_dict.items():
            if hasattr(value, 'tolist'):
                serializable_features[key] = value.tolist()
            elif hasattr(value, 'shape'):
                serializable_features[key] = {
                    'data': value.tolist(),
                    'shape': list(value.shape)
                }
            else:
                serializable_features[key] = value

        # Get quality assessment
        quality_report = text_extractor.assess_text_quality(texts)

        logger.info(f"Text feature extraction completed for {len(texts)} texts")

        return jsonify({
            'success': True,
            'data': {
                'features': serializable_features,
                'feature_types': list(features_dict.keys()),
                'quality_report': quality_report,
                'extraction_timestamp': datetime.utcnow().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Error extracting text features: {str(e)}")
        return jsonify({'error': f'Text feature extraction failed: {str(e)}'}), 500


@feature_engineering_bp.route('/temporal/features', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@validate_input_fields('dates')
def extract_temporal_features():
    """
    Extract advanced temporal features from dates

    Expected JSON payload:
    {
        "dates": ["2024-01-15", "2024-02-20", "2024-03-10"],
        "context_data": {
            "business_hours": {"start": 9, "end": 18}
        },
        "config": {
            "temporal_features": {"business_days": true, "holidays": true}
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        dates = data.get('dates', [])
        context_data = data.get('context_data', {})
        config = data.get('config', {})

        if not dates or not isinstance(dates, list):
            return jsonify({'error': 'dates must be a non-empty list'}), 400

        if len(dates) > 5000:  # Limit batch size
            return jsonify({'error': 'Batch size cannot exceed 5000 dates'}), 400

        # Update configuration if provided
        if config:
            temporal_enhancer.config.update(config)

        # Extract temporal features
        features, feature_names = temporal_enhancer.extract_temporal_features(dates, context_data)

        # Validate temporal consistency
        validation_report = temporal_enhancer.validate_temporal_consistency(dates)

        # Convert to serializable format
        features_list = features.tolist() if hasattr(features, 'tolist') else features

        logger.info(f"Temporal feature extraction completed for {len(dates)} dates")

        return jsonify({
            'success': True,
            'data': {
                'features': features_list,
                'feature_names': feature_names,
                'feature_matrix_shape': list(features.shape) if hasattr(features, 'shape') else None,
                'validation_report': validation_report,
                'processing_timestamp': datetime.utcnow().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Error extracting temporal features: {str(e)}")
        return jsonify({'error': f'Temporal feature extraction failed: {str(e)}'}), 500


@feature_engineering_bp.route('/financial/features', methods=['POST'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
@validate_input_fields('transactions')
def extract_financial_features():
    """
    Extract domain-specific financial features from transactions

    Expected JSON payload:
    {
        "transactions": [
            {
                "description": "Transferência PIX",
                "amount": 500.00,
                "date": "2024-01-15"
            }
        ],
        "config": {
            "amount_processing": {"round_amount_detection": true},
            "regulatory_compliance": {"aml_indicators": true}
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        transactions = data.get('transactions', [])
        config = data.get('config', {})

        if not transactions or not isinstance(transactions, list):
            return jsonify({'error': 'transactions must be a non-empty list'}), 400

        if len(transactions) > 5000:  # Limit dataset size
            return jsonify({'error': 'Dataset size cannot exceed 5000 transactions'}), 400

        # Update configuration if provided
        if config:
            financial_engineer.config.update(config)

        # Extract financial features
        features, feature_names = financial_engineer.extract_financial_features(transactions)

        # Validate financial data
        validation_report = financial_engineer.validate_financial_data(transactions)

        # Convert to serializable format
        features_list = features.tolist() if hasattr(features, 'tolist') else features

        logger.info(f"Financial feature extraction completed for {len(transactions)} transactions")

        return jsonify({
            'success': True,
            'data': {
                'features': features_list,
                'feature_names': feature_names,
                'feature_matrix_shape': list(features.shape) if hasattr(features, 'shape') else None,
                'validation_report': validation_report,
                'processing_timestamp': datetime.utcnow().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Error extracting financial features: {str(e)}")
        return jsonify({'error': f'Financial feature extraction failed: {str(e)}'}), 500


@feature_engineering_bp.route('/quality/pipeline', methods=['POST'])
@handle_errors
@rate_limit(max_requests=20, window_minutes=60)
@validate_input_fields('transactions')
def process_quality_pipeline():
    """
    Process dataset through the quality-assured feature engineering pipeline

    Expected JSON payload:
    {
        "transactions": [...],
        "target_column": "category",
        "validation_rules": {
            "business_rules": ["amount_must_be_positive"],
            "data_quality": {"completeness_threshold": 0.8}
        },
        "config": {
            "quality_control": {"quality_threshold": 0.8}
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        transactions = data.get('transactions', [])
        target_column = data.get('target_column')
        validation_rules = data.get('validation_rules', {})
        config = data.get('config', {})

        if not transactions or not isinstance(transactions, list):
            return jsonify({'error': 'transactions must be a non-empty list'}), 400

        if len(transactions) > 5000:  # Limit dataset size
            return jsonify({'error': 'Dataset size cannot exceed 5000 transactions'}), 400

        # Update configuration if provided
        if config:
            quality_pipeline.config.update(config)

        # Process through quality pipeline
        result = quality_pipeline.process_dataset(transactions, target_column, validation_rules)

        if not result.get('success', False):
            return jsonify({
                'success': False,
                'error': result.get('error', 'Pipeline processing failed')
            }), 400

        # Convert numpy arrays to lists for JSON serialization
        if 'features' in result and hasattr(result['features'], 'tolist'):
            result['features'] = result['features'].tolist()

        logger.info(f"Quality pipeline processing completed for {len(transactions)} transactions")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in quality pipeline processing: {str(e)}")
        return jsonify({'error': f'Quality pipeline processing failed: {str(e)}'}), 500


@feature_engineering_bp.route('/smote/balance', methods=['POST'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
@validate_input_fields(['features', 'target'])
def apply_smote_balancing():
    """
    Apply SMOTE for dataset balancing

    Expected JSON payload:
    {
        "features": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "target": [0, 1, 0, 1],
        "method": "auto",
        "config": {
            "sampling_strategy": "auto",
            "k_neighbors": 5
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        features = data.get('features', [])
        target = data.get('target', [])
        method = data.get('method', 'auto')
        config = data.get('config', {})

        if not features or not target:
            return jsonify({'error': 'features and target are required'}), 400

        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(target)

        if len(X) > 10000 or len(y) > 10000:
            return jsonify({'error': 'Dataset size cannot exceed 10,000 samples'}), 400

        # Apply SMOTE
        X_resampled, y_resampled = smote_engine.apply_smote(
            X, y, method=method, **config
        )

        # Get imbalance information
        imbalance_info = smote_engine.detect_imbalance(X, y)
        final_imbalance = smote_engine.detect_imbalance(X_resampled, y_resampled)

        logger.info(f"SMOTE balancing completed: {len(X)} -> {len(X_resampled)} samples")

        return jsonify({
            'success': True,
            'data': {
                'original_samples': len(X),
                'resampled_samples': len(X_resampled),
                'original_imbalance': imbalance_info,
                'final_imbalance': final_imbalance,
                'features_resampled': X_resampled.tolist(),
                'target_resampled': y_resampled.tolist(),
                'processing_timestamp': datetime.utcnow().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Error applying SMOTE balancing: {str(e)}")
        return jsonify({'error': f'SMOTE balancing failed: {str(e)}'}), 500


@feature_engineering_bp.route('/augmentation/generate', methods=['POST'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
@validate_input_fields('data')
def generate_data_augmentation():
    """
    Generate synthetic data using augmentation techniques

    Expected JSON payload:
    {
        "data": [...],  # List of dictionaries or DataFrame-like data
        "data_type": "transaction",
        "config": {
            "augmentation_ratio": 2.0,
            "text_augmentation": {"enabled": true}
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        input_data = data.get('data', [])
        data_type = data.get('data_type', 'mixed')
        config = data.get('config', {})

        if not input_data:
            return jsonify({'error': 'data cannot be empty'}), 400

        if len(input_data) > 2000:  # Limit input size
            return jsonify({'error': 'Input data size cannot exceed 2000 records'}), 400

        # Update configuration if provided
        if config:
            augmentation_pipeline.config.update(config)

        # Generate augmented data
        augmented_data, augmentation_report = augmentation_pipeline.augment_dataset(
            input_data, data_type=data_type
        )

        logger.info(f"Data augmentation completed: {len(input_data)} -> {len(augmented_data)} records")

        return jsonify({
            'success': True,
            'data': {
                'original_size': len(input_data),
                'augmented_size': len(augmented_data),
                'augmentation_report': augmentation_report,
                'augmented_data': augmented_data.to_dict('records') if hasattr(augmented_data, 'to_dict') else augmented_data,
                'processing_timestamp': datetime.utcnow().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Error generating data augmentation: {str(e)}")
        return jsonify({'error': f'Data augmentation failed: {str(e)}'}), 500


@feature_engineering_bp.route('/analytics/performance', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_performance_analytics():
    """
    Get feature engineering performance analytics

    Query parameters:
    - include_history: Include historical performance data (default: true)
    - time_range: Time range for analytics ('1h', '24h', '7d', '30d') (default: '24h')
    """
    try:
        include_history = request.args.get('include_history', 'true').lower() == 'true'
        time_range = request.args.get('time_range', '24h')

        # Get analytics from quality pipeline
        analytics = quality_pipeline.get_quality_metrics()

        # Add component-specific analytics
        component_analytics = {
            'enhanced_engineer': {
                'total_extractions': len(enhanced_engineer.performance_history),
                'avg_quality_score': np.mean([p.get('quality_score', 0) for p in enhanced_engineer.performance_history[-20:]]) if enhanced_engineer.performance_history else 0
            },
            'text_extractor': {
                'extraction_stats': text_extractor.get_extraction_stats(),
                'quality_metrics': text_extractor.get_quality_metrics()
            },
            'temporal_enhancer': {
                'temporal_stats': temporal_enhancer.get_temporal_stats()
            },
            'financial_engineer': {
                'feature_stats': financial_engineer.get_feature_stats()
            }
        }

        analytics.update({
            'component_analytics': component_analytics,
            'timestamp': datetime.utcnow().isoformat(),
            'time_range': time_range
        })

        return jsonify({
            'success': True,
            'data': analytics
        })

    except Exception as e:
        logger.error(f"Error getting performance analytics: {str(e)}")
        return jsonify({'error': f'Failed to get analytics: {str(e)}'}), 500


@feature_engineering_bp.route('/config', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
def get_feature_engineering_config():
    """
    Get current feature engineering configuration
    """
    try:
        config_info = {
            'enhanced_engineer': enhanced_engineer.config,
            'text_extractor': text_extractor.config,
            'temporal_enhancer': temporal_enhancer.config,
            'financial_engineer': financial_engineer.config,
            'quality_pipeline': quality_pipeline.config,
            'smote_engine': smote_engine.config if hasattr(smote_engine, 'config') else {},
            'augmentation_pipeline': augmentation_pipeline.config
        }

        return jsonify({
            'success': True,
            'data': config_info
        })

    except Exception as e:
        logger.error(f"Error getting feature engineering config: {str(e)}")
        return jsonify({'error': f'Failed to get config: {str(e)}'}), 500


@feature_engineering_bp.route('/config', methods=['POST'])
@handle_errors
@rate_limit(max_requests=10, window_minutes=60)
def update_feature_engineering_config():
    """
    Update feature engineering configuration

    Expected JSON payload:
    {
        "enhanced_engineer": {...},
        "text_extractor": {...},
        "quality_pipeline": {...}
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Update configurations for each component
        if 'enhanced_engineer' in data:
            enhanced_engineer.config.update(data['enhanced_engineer'])

        if 'text_extractor' in data:
            text_extractor.config.update(data['text_extractor'])

        if 'temporal_enhancer' in data:
            temporal_enhancer.config.update(data['temporal_enhancer'])

        if 'financial_engineer' in data:
            financial_engineer.config.update(data['financial_engineer'])

        if 'quality_pipeline' in data:
            quality_pipeline.config.update(data['quality_pipeline'])

        logger.info("Feature engineering configuration updated")

        return jsonify({
            'success': True,
            'message': 'Feature engineering configuration updated successfully'
        })

    except Exception as e:
        logger.error(f"Error updating feature engineering config: {str(e)}")
        return jsonify({'error': f'Configuration update failed: {str(e)}'}), 500


@feature_engineering_bp.route('/cache/clear', methods=['POST'])
@handle_errors
@rate_limit(max_requests=10, window_minutes=60)
def clear_feature_engineering_cache():
    """
    Clear all feature engineering caches
    """
    try:
        enhanced_engineer.clear_cache()
        text_extractor.clear_cache()
        temporal_enhancer.clear_cache()
        financial_engineer.clear_cache()
        quality_pipeline.clear_cache()

        logger.info("Feature engineering caches cleared")

        return jsonify({
            'success': True,
            'message': 'Feature engineering caches cleared successfully'
        })

    except Exception as e:
        logger.error(f"Error clearing feature engineering cache: {str(e)}")
        return jsonify({'error': f'Cache clear failed: {str(e)}'}), 500


@feature_engineering_bp.route('/health', methods=['GET'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
def feature_engineering_health_check():
    """
    Health check for feature engineering services
    """
    try:
        # Test basic functionality
        test_transactions = [
            {
                'description': 'Teste de transação',
                'amount': 100.50,
                'date': '2024-01-15'
            }
        ]

        # Test enhanced feature engineering
        features, names, report = enhanced_engineer.create_enhanced_features(test_transactions)

        # Get component status
        component_status = {
            'enhanced_engineer': 'operational' if features.size > 0 else 'degraded',
            'text_extractor': 'operational',
            'temporal_enhancer': 'operational',
            'financial_engineer': 'operational',
            'quality_pipeline': 'operational',
            'smote_engine': 'operational',
            'augmentation_pipeline': 'operational'
        }

        # Get quality metrics
        quality_metrics = quality_pipeline.get_quality_metrics()

        health_status = {
            'status': 'healthy' if all(status == 'operational' for status in component_status.values()) else 'degraded',
            'services': component_status,
            'metrics': {
                'total_runs': quality_metrics.get('total_runs', 0),
                'average_quality_score': quality_metrics.get('average_final_quality', 0.0),
                'cache_status': 'operational'
            },
            'timestamp': datetime.utcnow().isoformat()
        }

        return jsonify(health_status)

    except Exception as e:
        logger.error(f"Feature engineering health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@feature_engineering_bp.route('/save/state', methods=['POST'])
@handle_errors
@rate_limit(max_requests=5, window_minutes=60)
def save_feature_engineering_state():
    """
    Save the current state of feature engineering components

    Expected JSON payload:
    {
        "filepath": "path/to/save/state.pkl",
        "components": ["enhanced_engineer", "quality_pipeline"]
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        filepath = data.get('filepath', 'feature_engineering_state.pkl')
        components = data.get('components', ['enhanced_engineer', 'quality_pipeline'])

        saved_components = []

        if 'enhanced_engineer' in components:
            enhanced_engineer.save_enhanced_engineer(f"{filepath}_enhanced.pkl")
            saved_components.append('enhanced_engineer')

        if 'quality_pipeline' in components:
            quality_pipeline.save_pipeline(f"{filepath}_pipeline.pkl")
            saved_components.append('quality_pipeline')

        if 'text_extractor' in components:
            text_extractor.save_extractor(f"{filepath}_text.pkl")
            saved_components.append('text_extractor')

        logger.info(f"Feature engineering state saved for components: {saved_components}")

        return jsonify({
            'success': True,
            'message': f'State saved successfully for components: {saved_components}',
            'filepath': filepath
        })

    except Exception as e:
        logger.error(f"Error saving feature engineering state: {str(e)}")
        return jsonify({'error': f'State save failed: {str(e)}'}), 500


@feature_engineering_bp.route('/load/state', methods=['POST'])
@handle_errors
@rate_limit(max_requests=5, window_minutes=60)
def load_feature_engineering_state():
    """
    Load saved state of feature engineering components

    Expected JSON payload:
    {
        "filepath": "path/to/load/state.pkl",
        "components": ["enhanced_engineer", "quality_pipeline"]
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        filepath = data.get('filepath', 'feature_engineering_state.pkl')
        components = data.get('components', ['enhanced_engineer', 'quality_pipeline'])

        loaded_components = []

        if 'enhanced_engineer' in components:
            enhanced_engineer.load_enhanced_engineer(f"{filepath}_enhanced.pkl")
            loaded_components.append('enhanced_engineer')

        if 'quality_pipeline' in components:
            quality_pipeline.load_pipeline(f"{filepath}_pipeline.pkl")
            loaded_components.append('quality_pipeline')

        if 'text_extractor' in components:
            text_extractor.load_extractor(f"{filepath}_text.pkl")
            loaded_components.append('text_extractor')

        logger.info(f"Feature engineering state loaded for components: {loaded_components}")

        return jsonify({
            'success': True,
            'message': f'State loaded successfully for components: {loaded_components}',
            'filepath': filepath
        })

    except Exception as e:
        logger.error(f"Error loading feature engineering state: {str(e)}")
        return jsonify({'error': f'State load failed: {str(e)}'}), 500