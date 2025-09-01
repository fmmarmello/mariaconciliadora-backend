from flask import Blueprint, request, jsonify
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.services.preprocessing_pipeline import PreprocessingPipeline, ProcessingStep, PipelineConfig
from src.services.portuguese_preprocessor import PortugueseTextPreprocessor
from src.utils.logging_config import get_logger
from src.utils.error_handler import handle_errors
from src.utils.validation_middleware import rate_limit, validate_input_fields

logger = get_logger(__name__)

text_preprocessing_bp = Blueprint('text_preprocessing', __name__)

# Initialize preprocessing components
preprocessing_pipeline = PreprocessingPipeline()
portuguese_preprocessor = PortugueseTextPreprocessor(use_advanced_pipeline=True)


@text_preprocessing_bp.route('/preprocess/text', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@validate_input_fields('text')
def preprocess_text():
    """
    Preprocess a single text using the advanced Portuguese preprocessing pipeline

    Expected JSON payload:
    {
        "text": "Texto em português para processar",
        "config": {
            "expand_abbreviations": true,
            "lowercase": true,
            "remove_accents": true,
            "remove_punctuation": true,
            "stopwords": true
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        text = data.get('text', '').strip()
        config = data.get('config', {})

        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        # Use advanced preprocessing
        result = portuguese_preprocessor.preprocess_with_advanced_features(text)

        # Apply any custom config
        if config:
            # Re-preprocess with custom config if needed
            processed_text = portuguese_preprocessor.preprocess(text, config)
            result['processed_text'] = processed_text

        logger.info(f"Text preprocessing completed for text of length {len(text)}")

        return jsonify({
            'success': True,
            'data': result
        })

    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        return jsonify({'error': f'Text preprocessing failed: {str(e)}'}), 500


@text_preprocessing_bp.route('/preprocess/batch', methods=['POST'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
@validate_input_fields('texts')
def preprocess_batch():
    """
    Preprocess a batch of texts using the advanced preprocessing pipeline

    Expected JSON payload:
    {
        "texts": ["Texto 1", "Texto 2", "Texto 3"],
        "config": {
            "expand_abbreviations": true,
            "lowercase": true,
            "remove_accents": true,
            "remove_punctuation": true,
            "stopwords": true
        },
        "batch_size": 32
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        texts = data.get('texts', [])
        config = data.get('config', {})
        batch_size = data.get('batch_size', 32)

        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'texts must be a non-empty list'}), 400

        if len(texts) > 1000:  # Limit batch size
            return jsonify({'error': 'Batch size cannot exceed 1000 texts'}), 400

        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            return jsonify({'error': 'No valid texts provided'}), 400

        # Process batch
        start_time = datetime.now()

        if len(valid_texts) == 1:
            # Single text processing
            result = portuguese_preprocessor.preprocess_with_advanced_features(valid_texts[0])
            results = [result]
        else:
            # Batch processing
            results = []
            for text in valid_texts:
                result = portuguese_preprocessor.preprocess_with_advanced_features(text)
                results.append(result)

        processing_time = (datetime.now() - start_time).total_seconds()

        # Get pipeline metrics
        metrics = preprocessing_pipeline.get_pipeline_metrics()

        logger.info(f"Batch preprocessing completed for {len(valid_texts)} texts in {processing_time:.2f}s")

        return jsonify({
            'success': True,
            'data': {
                'results': results,
                'batch_size': len(valid_texts),
                'processing_time': processing_time,
                'pipeline_metrics': metrics
            }
        })

    except Exception as e:
        logger.error(f"Error in batch preprocessing: {str(e)}")
        return jsonify({'error': f'Batch preprocessing failed: {str(e)}'}), 500


@text_preprocessing_bp.route('/preprocess/advanced', methods=['POST'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
@validate_input_fields('text')
def advanced_preprocessing():
    """
    Advanced text preprocessing with full pipeline and context awareness

    Expected JSON payload:
    {
        "text": "Texto complexo para análise avançada",
        "context_history": ["Texto anterior 1", "Texto anterior 2"],
        "pipeline_config": {
            "steps": ["advanced_portuguese", "financial_processing", "context_aware", "quality_assessment"],
            "batch_size": 16,
            "max_workers": 2
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        text = data.get('text', '').strip()
        context_history = data.get('context_history', [])
        pipeline_config_data = data.get('pipeline_config', {})

        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        # Create custom pipeline configuration
        pipeline_config = PipelineConfig(
            steps=[ProcessingStep(step) for step in pipeline_config_data.get('steps', [
                ProcessingStep.ADVANCED_PORTUGUESE,
                ProcessingStep.FINANCIAL_PROCESSING,
                ProcessingStep.CONTEXT_AWARE,
                ProcessingStep.QUALITY_ASSESSMENT
            ])],
            batch_size=pipeline_config_data.get('batch_size', 16),
            max_workers=pipeline_config_data.get('max_workers', 2),
            enable_parallel_processing=pipeline_config_data.get('parallel', True),
            error_handling=pipeline_config_data.get('error_handling', 'continue'),
            quality_threshold=pipeline_config_data.get('quality_threshold', 0.6),
            cache_enabled=pipeline_config_data.get('cache_enabled', True),
            performance_monitoring=True
        )

        # Create custom pipeline instance
        custom_pipeline = PreprocessingPipeline(pipeline_config)

        # Process with context
        result = custom_pipeline.process_text(text)

        # Convert result to dict format
        response_data = {
            'original_text': result.original_text,
            'processed_text': result.processed_text,
            'success': result.success,
            'processing_time': result.processing_time,
            'quality_metrics': result.quality_metrics,
            'intermediate_results': {}
        }

        # Convert intermediate results
        for step_name, step_result in result.intermediate_results.items():
            if hasattr(step_result, '__dict__'):
                response_data['intermediate_results'][step_name] = step_result.__dict__
            else:
                response_data['intermediate_results'][step_name] = step_result

        if not result.success and result.error_message:
            response_data['error'] = result.error_message

        # Get pipeline metrics
        pipeline_metrics = custom_pipeline.get_pipeline_metrics()

        logger.info(f"Advanced preprocessing completed for text of length {len(text)}")

        return jsonify({
            'success': True,
            'data': response_data,
            'pipeline_metrics': pipeline_metrics
        })

    except Exception as e:
        logger.error(f"Error in advanced preprocessing: {str(e)}")
        return jsonify({'error': f'Advanced preprocessing failed: {str(e)}'}), 500


@text_preprocessing_bp.route('/preprocess/optimize', methods=['POST'])
@handle_errors
@rate_limit(max_requests=10, window_minutes=60)
@validate_input_fields('sample_texts')
def optimize_preprocessing():
    """
    Optimize preprocessing pipeline configuration based on sample data

    Expected JSON payload:
    {
        "sample_texts": ["Texto 1", "Texto 2", "Texto 3"],
        "current_config": {
            "batch_size": 32,
            "max_workers": 4
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        sample_texts = data.get('sample_texts', [])
        current_config = data.get('current_config', {})

        if not sample_texts or not isinstance(sample_texts, list):
            return jsonify({'error': 'sample_texts must be a non-empty list'}), 400

        if len(sample_texts) < 5:
            return jsonify({'error': 'At least 5 sample texts required for optimization'}), 400

        # Run optimization
        optimization_result = preprocessing_pipeline.optimize_pipeline(sample_texts)

        if 'error' in optimization_result:
            return jsonify({
                'success': False,
                'error': optimization_result['error']
            }), 500

        logger.info("Preprocessing pipeline optimization completed")

        return jsonify({
            'success': True,
            'data': optimization_result
        })

    except Exception as e:
        logger.error(f"Error optimizing preprocessing: {str(e)}")
        return jsonify({'error': f'Optimization failed: {str(e)}'}), 500


@text_preprocessing_bp.route('/preprocess/metrics', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_preprocessing_metrics():
    """
    Get preprocessing pipeline metrics and statistics

    Query parameters:
    - include_history: Include historical metrics (default: false)
    - reset: Reset metrics after retrieval (default: false)
    """
    try:
        include_history = request.args.get('include_history', 'false').lower() == 'true'
        reset = request.args.get('reset', 'false').lower() == 'true'

        # Get current metrics
        metrics = preprocessing_pipeline.get_pipeline_metrics()

        # Get Portuguese preprocessor metrics
        preprocessor_metrics = portuguese_preprocessor.get_advanced_metrics()

        # Combine metrics
        combined_metrics = {
            'pipeline_metrics': metrics,
            'preprocessor_metrics': preprocessor_metrics,
            'timestamp': datetime.now().isoformat()
        }

        if include_history:
            # Add historical data if available
            combined_metrics['history'] = {
                'cache_hit_rate': metrics.get('cache_hit_rate', 0.0),
                'average_processing_time': metrics.get('average_processing_time', 0.0),
                'total_processed': metrics.get('total_processed', 0)
            }

        if reset:
            # Reset metrics
            preprocessing_pipeline.reset_metrics()
            portuguese_preprocessor.clear_advanced_cache()
            combined_metrics['metrics_reset'] = True

        return jsonify({
            'success': True,
            'data': combined_metrics
        })

    except Exception as e:
        logger.error(f"Error getting preprocessing metrics: {str(e)}")
        return jsonify({'error': f'Failed to get metrics: {str(e)}'}), 500


@text_preprocessing_bp.route('/preprocess/config', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
def get_preprocessing_config():
    """
    Get current preprocessing configuration
    """
    try:
        pipeline_config = preprocessing_pipeline.config
        preprocessor_config = portuguese_preprocessor.preprocessing_config

        config_info = {
            'pipeline_config': {
                'steps': [step.value for step in pipeline_config.steps],
                'batch_size': pipeline_config.batch_size,
                'max_workers': pipeline_config.max_workers,
                'parallel_processing': pipeline_config.enable_parallel_processing,
                'error_handling': pipeline_config.error_handling,
                'quality_threshold': pipeline_config.quality_threshold,
                'cache_enabled': pipeline_config.cache_enabled
            },
            'preprocessor_config': preprocessor_config,
            'advanced_mode_enabled': portuguese_preprocessor.use_advanced_pipeline
        }

        return jsonify({
            'success': True,
            'data': config_info
        })

    except Exception as e:
        logger.error(f"Error getting preprocessing config: {str(e)}")
        return jsonify({'error': f'Failed to get config: {str(e)}'}), 500


@text_preprocessing_bp.route('/preprocess/config', methods=['POST'])
@handle_errors
@rate_limit(max_requests=10, window_minutes=60)
def update_preprocessing_config():
    """
    Update preprocessing configuration

    Expected JSON payload:
    {
        "pipeline_config": {
            "batch_size": 64,
            "max_workers": 8,
            "parallel_processing": true
        },
        "preprocessor_config": {
            "expand_abbreviations": false,
            "remove_numbers": true
        },
        "enable_advanced_mode": true
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Update pipeline configuration
        pipeline_config_data = data.get('pipeline_config', {})
        if pipeline_config_data:
            # Create new pipeline with updated config
            new_pipeline_config = PipelineConfig(
                steps=preprocessing_pipeline.config.steps,  # Keep existing steps
                batch_size=pipeline_config_data.get('batch_size', preprocessing_pipeline.config.batch_size),
                max_workers=pipeline_config_data.get('max_workers', preprocessing_pipeline.config.max_workers),
                enable_parallel_processing=pipeline_config_data.get('parallel_processing',
                    preprocessing_pipeline.config.enable_parallel_processing),
                error_handling=pipeline_config_data.get('error_handling', preprocessing_pipeline.config.error_handling),
                quality_threshold=pipeline_config_data.get('quality_threshold', preprocessing_pipeline.config.quality_threshold),
                cache_enabled=pipeline_config_data.get('cache_enabled', preprocessing_pipeline.config.cache_enabled),
                performance_monitoring=True
            )

            # Replace pipeline instance
            global preprocessing_pipeline
            preprocessing_pipeline = PreprocessingPipeline(new_pipeline_config)

        # Update preprocessor configuration
        preprocessor_config_data = data.get('preprocessor_config', {})
        if preprocessor_config_data:
            portuguese_preprocessor.preprocessing_config.update(preprocessor_config_data)

        # Update advanced mode
        if 'enable_advanced_mode' in data:
            if data['enable_advanced_mode']:
                portuguese_preprocessor.enable_advanced_mode()
            else:
                portuguese_preprocessor.disable_advanced_mode()

        logger.info("Preprocessing configuration updated")

        return jsonify({
            'success': True,
            'message': 'Preprocessing configuration updated successfully'
        })

    except Exception as e:
        logger.error(f"Error updating preprocessing config: {str(e)}")
        return jsonify({'error': f'Configuration update failed: {str(e)}'}), 500


@text_preprocessing_bp.route('/preprocess/cache/clear', methods=['POST'])
@handle_errors
@rate_limit(max_requests=10, window_minutes=60)
def clear_preprocessing_cache():
    """
    Clear preprocessing cache
    """
    try:
        preprocessing_pipeline.clear_cache()
        portuguese_preprocessor.clear_advanced_cache()

        logger.info("Preprocessing cache cleared")

        return jsonify({
            'success': True,
            'message': 'Preprocessing cache cleared successfully'
        })

    except Exception as e:
        logger.error(f"Error clearing preprocessing cache: {str(e)}")
        return jsonify({'error': f'Cache clear failed: {str(e)}'}), 500


@text_preprocessing_bp.route('/preprocess/health', methods=['GET'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
def preprocessing_health_check():
    """
    Health check for preprocessing services
    """
    try:
        # Test basic preprocessing
        test_text = "Teste de processamento de texto em português"
        result = portuguese_preprocessor.preprocess(test_text, {})

        # Get metrics
        metrics = preprocessing_pipeline.get_pipeline_metrics()

        health_status = {
            'status': 'healthy',
            'services': {
                'portuguese_preprocessor': 'operational' if result else 'degraded',
                'preprocessing_pipeline': 'operational',
                'advanced_features': portuguese_preprocessor.use_advanced_pipeline
            },
            'metrics': {
                'total_processed': metrics.get('total_processed', 0),
                'average_quality_score': metrics.get('average_quality_score', 0.0),
                'cache_size': metrics.get('cache_size', 0)
            },
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(health_status)

    except Exception as e:
        logger.error(f"Preprocessing health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500