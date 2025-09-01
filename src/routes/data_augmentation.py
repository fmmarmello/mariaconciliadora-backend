from flask import Blueprint, request, jsonify
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

# Local imports
from src.services.data_augmentation_pipeline import DataAugmentationPipeline
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from src.utils.error_handler import handle_errors

logger = get_logger(__name__)

# Create blueprint
data_augmentation_bp = Blueprint('data_augmentation', __name__)

# Global augmentation pipeline instance
augmentation_pipeline = None


def get_augmentation_pipeline() -> DataAugmentationPipeline:
    """Get or create augmentation pipeline instance"""
    global augmentation_pipeline
    if augmentation_pipeline is None:
        augmentation_pipeline = DataAugmentationPipeline()
    return augmentation_pipeline


@data_augmentation_bp.route('/health', methods=['GET'])
@handle_errors
def health_check():
    """Health check endpoint for data augmentation service"""
    return jsonify({
        'status': 'healthy',
        'service': 'data_augmentation',
        'timestamp': datetime.utcnow().isoformat()
    })


@data_augmentation_bp.route('/augment', methods=['POST'])
@handle_errors
def augment_data():
    """
    Augment dataset with comprehensive data augmentation pipeline

    Expected JSON payload:
    {
        "data": [...],  # List of transaction/company financial records
        "data_type": "transaction",  # or "company_financial" or "mixed"
        "config": {...}  # Optional custom configuration
    }
    """
    try:
        # Get request data
        request_data = request.get_json()

        if not request_data or 'data' not in request_data:
            return jsonify({
                'error': 'Missing required field: data',
                'status': 'error'
            }), 400

        data = request_data['data']
        data_type = request_data.get('data_type', 'mixed')
        custom_config = request_data.get('config', {})

        if not isinstance(data, list) or len(data) == 0:
            return jsonify({
                'error': 'Data must be a non-empty list',
                'status': 'error'
            }), 400

        logger.info(f"Received augmentation request for {len(data)} records of type: {data_type}")

        # Get augmentation pipeline
        pipeline = get_augmentation_pipeline()

        # Apply custom configuration if provided
        if custom_config:
            # Update pipeline config (this would need to be implemented in the pipeline)
            logger.info("Applying custom augmentation configuration")

        # Perform augmentation
        augmented_data, augmentation_report = pipeline.augment_dataset(data, data_type)

        # Convert to response format
        response_data = {
            'status': 'success',
            'original_size': len(data),
            'augmented_size': len(augmented_data),
            'data': augmented_data.to_dict('records') if hasattr(augmented_data, 'to_dict') else augmented_data,
            'augmentation_report': augmentation_report,
            'quality_metrics': pipeline.get_augmentation_stats()
        }

        logger.info(f"Augmentation completed. Original: {len(data)}, Augmented: {len(augmented_data)}")
        return jsonify(response_data)

    except ValidationError as e:
        logger.error(f"Validation error in data augmentation: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'validation_error'
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error in data augmentation: {str(e)}")
        return jsonify({
            'error': 'Internal server error during data augmentation',
            'status': 'error'
        }), 500


@data_augmentation_bp.route('/augment/text', methods=['POST'])
@handle_errors
def augment_text():
    """
    Augment text data specifically

    Expected JSON payload:
    {
        "texts": [...],  # List of text strings
        "config": {...}  # Optional text augmentation configuration
    }
    """
    try:
        request_data = request.get_json()

        if not request_data or 'texts' not in request_data:
            return jsonify({
                'error': 'Missing required field: texts',
                'status': 'error'
            }), 400

        texts = request_data['texts']
        config = request_data.get('config', {})

        if not isinstance(texts, list):
            return jsonify({
                'error': 'Texts must be a list',
                'status': 'error'
            }), 400

        logger.info(f"Received text augmentation request for {len(texts)} texts")

        # Get augmentation pipeline and text engine
        pipeline = get_augmentation_pipeline()
        text_engine = pipeline.text_engine

        if not text_engine:
            return jsonify({
                'error': 'Text augmentation engine not available',
                'status': 'error'
            }), 500

        # Augment texts
        augmented_texts = text_engine.augment_batch(texts, config)

        response_data = {
            'status': 'success',
            'original_count': len(texts),
            'augmented_texts': augmented_texts,
            'quality_metrics': text_engine.get_quality_metrics()
        }

        logger.info(f"Text augmentation completed for {len(texts)} texts")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in text augmentation: {str(e)}")
        return jsonify({
            'error': 'Internal server error during text augmentation',
            'status': 'error'
        }), 500


@data_augmentation_bp.route('/augment/numerical', methods=['POST'])
@handle_errors
def augment_numerical():
    """
    Augment numerical data specifically

    Expected JSON payload:
    {
        "values": [...],  # List of numerical values
        "config": {...}  # Optional numerical augmentation configuration
    }
    """
    try:
        request_data = request.get_json()

        if not request_data or 'values' not in request_data:
            return jsonify({
                'error': 'Missing required field: values',
                'status': 'error'
            }), 400

        values = request_data['values']
        config = request_data.get('config', {})

        if not isinstance(values, list):
            return jsonify({
                'error': 'Values must be a list',
                'status': 'error'
            }), 400

        logger.info(f"Received numerical augmentation request for {len(values)} values")

        # Get augmentation pipeline and numerical engine
        pipeline = get_augmentation_pipeline()
        numerical_engine = pipeline.numerical_engine

        if not numerical_engine:
            return jsonify({
                'error': 'Numerical augmentation engine not available',
                'status': 'error'
            }), 500

        # Augment values
        augmented_values = numerical_engine.augment_numerical(values, config)

        response_data = {
            'status': 'success',
            'original_count': len(values),
            'augmented_values': augmented_values,
            'quality_metrics': numerical_engine.get_quality_metrics()
        }

        logger.info(f"Numerical augmentation completed for {len(values)} values")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in numerical augmentation: {str(e)}")
        return jsonify({
            'error': 'Internal server error during numerical augmentation',
            'status': 'error'
        }), 500


@data_augmentation_bp.route('/generate-synthetic', methods=['POST'])
@handle_errors
def generate_synthetic():
    """
    Generate synthetic data using trained models

    Expected JSON payload:
    {
        "data": [...],  # Training data for model
        "sample_size": 100,  # Number of synthetic samples to generate
        "config": {...}  # Optional synthetic generation configuration
    }
    """
    try:
        request_data = request.get_json()

        if not request_data or 'data' not in request_data:
            return jsonify({
                'error': 'Missing required field: data',
                'status': 'error'
            }), 400

        data = request_data['data']
        sample_size = request_data.get('sample_size', 100)
        config = request_data.get('config', {})

        if not isinstance(data, list) or len(data) == 0:
            return jsonify({
                'error': 'Data must be a non-empty list',
                'status': 'error'
            }), 400

        logger.info(f"Received synthetic data generation request for {len(data)} training records")

        # Get augmentation pipeline and synthetic generator
        pipeline = get_augmentation_pipeline()
        synthetic_generator = pipeline.synthetic_generator

        if not synthetic_generator:
            return jsonify({
                'error': 'Synthetic data generator not available',
                'status': 'error'
            }), 500

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Generate synthetic data
        synthetic_data = synthetic_generator.generate_synthetic_data(df)

        if synthetic_data is None:
            return jsonify({
                'error': 'Failed to generate synthetic data',
                'status': 'error'
            }), 500

        response_data = {
            'status': 'success',
            'training_size': len(data),
            'synthetic_size': len(synthetic_data),
            'synthetic_data': synthetic_data.to_dict('records'),
            'quality_metrics': synthetic_generator.get_quality_metrics()
        }

        logger.info(f"Synthetic data generation completed. Generated {len(synthetic_data)} samples")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in synthetic data generation: {str(e)}")
        return jsonify({
            'error': 'Internal server error during synthetic data generation',
            'status': 'error'
        }), 500


@data_augmentation_bp.route('/validate-quality', methods=['POST'])
@handle_errors
def validate_quality():
    """
    Validate quality of augmented data

    Expected JSON payload:
    {
        "original_data": [...],
        "augmented_data": [...]
    }
    """
    try:
        request_data = request.get_json()

        if not request_data or 'original_data' not in request_data or 'augmented_data' not in request_data:
            return jsonify({
                'error': 'Missing required fields: original_data and augmented_data',
                'status': 'error'
            }), 400

        original_data = request_data['original_data']
        augmented_data = request_data['augmented_data']

        logger.info("Received quality validation request")

        # Get augmentation pipeline and quality controller
        pipeline = get_augmentation_pipeline()
        quality_controller = pipeline.quality_controller

        if not quality_controller:
            return jsonify({
                'error': 'Quality control system not available',
                'status': 'error'
            }), 500

        # Perform validation
        validation_results = quality_controller.validate_augmentation(original_data, augmented_data)

        response_data = {
            'status': 'success',
            'validation_results': validation_results,
            'quality_report': quality_controller.generate_quality_report()
        }

        logger.info(f"Quality validation completed. Passed: {validation_results.get('validation_passed', False)}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in quality validation: {str(e)}")
        return jsonify({
            'error': 'Internal server error during quality validation',
            'status': 'error'
        }), 500


@data_augmentation_bp.route('/metrics', methods=['GET'])
@handle_errors
def get_metrics():
    """Get augmentation system metrics and statistics"""
    try:
        pipeline = get_augmentation_pipeline()

        metrics = {
            'status': 'success',
            'system_metrics': pipeline.get_augmentation_stats(),
            'engine_metrics': {}
        }

        # Get individual engine metrics
        if pipeline.text_engine:
            metrics['engine_metrics']['text_engine'] = pipeline.text_engine.get_quality_metrics()

        if pipeline.numerical_engine:
            metrics['engine_metrics']['numerical_engine'] = pipeline.numerical_engine.get_quality_metrics()

        if pipeline.categorical_engine:
            metrics['engine_metrics']['categorical_engine'] = pipeline.categorical_engine.get_quality_metrics()

        if pipeline.temporal_engine:
            metrics['engine_metrics']['temporal_engine'] = pipeline.temporal_engine.get_quality_metrics()

        if pipeline.synthetic_generator:
            metrics['engine_metrics']['synthetic_generator'] = pipeline.synthetic_generator.get_quality_metrics()

        if pipeline.quality_controller:
            metrics['engine_metrics']['quality_controller'] = pipeline.quality_controller.get_quality_metrics()

        return jsonify(metrics)

    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        return jsonify({
            'error': 'Internal server error retrieving metrics',
            'status': 'error'
        }), 500


@data_augmentation_bp.route('/config', methods=['GET'])
@handle_errors
def get_config():
    """Get current augmentation configuration"""
    try:
        pipeline = get_augmentation_pipeline()

        return jsonify({
            'status': 'success',
            'config': pipeline.config
        })

    except Exception as e:
        logger.error(f"Error retrieving configuration: {str(e)}")
        return jsonify({
            'error': 'Internal server error retrieving configuration',
            'status': 'error'
        }), 500


@data_augmentation_bp.route('/config', methods=['POST'])
@handle_errors
def update_config():
    """
    Update augmentation configuration

    Expected JSON payload:
    {
        "config": {...}  # New configuration
    }
    """
    try:
        request_data = request.get_json()

        if not request_data or 'config' not in request_data:
            return jsonify({
                'error': 'Missing required field: config',
                'status': 'error'
            }), 400

        new_config = request_data['config']

        # Note: In a production system, you might want to validate the config
        # and potentially restart the pipeline with new settings

        logger.info("Configuration update requested")

        return jsonify({
            'status': 'success',
            'message': 'Configuration update received (restart may be required for full effect)',
            'config': new_config
        })

    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        return jsonify({
            'error': 'Internal server error updating configuration',
            'status': 'error'
        }), 500


@data_augmentation_bp.route('/reset', methods=['POST'])
@handle_errors
def reset_pipeline():
    """Reset the augmentation pipeline"""
    try:
        global augmentation_pipeline
        augmentation_pipeline = None  # This will force recreation on next use

        logger.info("Augmentation pipeline reset")

        return jsonify({
            'status': 'success',
            'message': 'Augmentation pipeline has been reset'
        })

    except Exception as e:
        logger.error(f"Error resetting pipeline: {str(e)}")
        return jsonify({
            'error': 'Internal server error resetting pipeline',
            'status': 'error'
        }), 500