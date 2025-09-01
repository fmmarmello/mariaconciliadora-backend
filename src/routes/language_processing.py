from flask import Blueprint, request, jsonify
from typing import Dict, Any, List, Optional
import logging

from src.utils.logging_config import get_logger
from src.services.language_detector import LanguageDetector, DetectionResult
from src.services.multi_language_preprocessor import MultiLanguagePreprocessor, PreprocessingResult
from src.services.language_aware_processor import LanguageAwareProcessor, ProcessingResult as LanguageProcessingResult
from src.services.international_financial_processor import InternationalFinancialProcessor, FinancialProcessingResult
from src.services.preprocessing_pipeline import PreprocessingPipeline, ProcessingResult

logger = get_logger(__name__)

# Create blueprint
language_bp = Blueprint('language_processing', __name__)

# Initialize services
_language_detector = None
_multi_lang_preprocessor = None
_language_aware_processor = None
_international_financial_processor = None
_preprocessing_pipeline = None


def get_language_detector() -> LanguageDetector:
    """Get or create language detector instance"""
    global _language_detector
    if _language_detector is None:
        _language_detector = LanguageDetector()
    return _language_detector


def get_multi_lang_preprocessor() -> MultiLanguagePreprocessor:
    """Get or create multi-language preprocessor instance"""
    global _multi_lang_preprocessor
    if _multi_lang_preprocessor is None:
        _multi_lang_preprocessor = MultiLanguagePreprocessor()
    return _multi_lang_preprocessor


def get_language_aware_processor() -> LanguageAwareProcessor:
    """Get or create language-aware processor instance"""
    global _language_aware_processor
    if _language_aware_processor is None:
        _language_aware_processor = LanguageAwareProcessor()
    return _language_aware_processor


def get_international_financial_processor() -> InternationalFinancialProcessor:
    """Get or create international financial processor instance"""
    global _international_financial_processor
    if _international_financial_processor is None:
        _international_financial_processor = InternationalFinancialProcessor()
    return _international_financial_processor


def get_preprocessing_pipeline() -> PreprocessingPipeline:
    """Get or create preprocessing pipeline instance"""
    global _preprocessing_pipeline
    if _preprocessing_pipeline is None:
        _preprocessing_pipeline = PreprocessingPipeline()
    return _preprocessing_pipeline


@language_bp.route('/detect-language', methods=['POST'])
def detect_language():
    """
    Detect language of input text

    Request body:
    {
        "text": "Input text to analyze",
        "options": {
            "include_alternatives": true,
            "confidence_threshold": 0.7
        }
    }

    Response:
    {
        "success": true,
        "language": "pt",
        "confidence": 0.95,
        "alternatives": [...],
        "processing_time": 0.123
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: text'
            }), 400

        text = data['text']
        options = data.get('options', {})

        detector = get_language_detector()
        result = detector.detect_language(text)

        response = {
            'success': True,
            'language': result.consensus_language,
            'confidence': result.consensus_confidence,
            'processing_time': result.primary_detection.processing_time,
            'method_used': result.primary_detection.method
        }

        if options.get('include_alternatives', False):
            response['alternatives'] = [
                {
                    'language': alt[0],
                    'confidence': alt[1]
                } for alt in result.primary_detection.alternatives
            ]

        if options.get('include_details', False):
            response['detection_details'] = {
                'all_detections': [
                    {
                        'language': d.language,
                        'confidence': d.confidence,
                        'method': d.method
                    } for d in result.all_detections
                ],
                'detection_methods': result.detection_methods,
                'fallback_used': result.fallback_used
            }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in language detection: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@language_bp.route('/preprocess-multi-language', methods=['POST'])
def preprocess_multi_language():
    """
    Preprocess text with multi-language support

    Request body:
    {
        "text": "Input text to preprocess",
        "detected_language": "pt",  // optional
        "options": {
            "normalize_currency": true,
            "normalize_dates": true,
            "extract_entities": true
        }
    }

    Response:
    {
        "success": true,
        "original_text": "...",
        "processed_text": "...",
        "detected_language": "pt",
        "language_confidence": 0.95,
        "extracted_entities": {...},
        "quality_metrics": {...},
        "processing_time": 0.234
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: text'
            }), 400

        text = data['text']
        detected_language = data.get('detected_language')
        options = data.get('options', {})

        preprocessor = get_multi_lang_preprocessor()
        result = preprocessor.preprocess_text(text, detected_language)

        response = {
            'success': result.success,
            'original_text': result.original_text,
            'processed_text': result.processed_text,
            'detected_language': result.detected_language,
            'language_confidence': result.language_confidence,
            'processing_time': result.processing_time,
            'normalization_steps': result.normalization_steps,
            'quality_metrics': result.quality_metrics
        }

        if result.extracted_entities:
            response['extracted_entities'] = result.extracted_entities

        if not result.success and result.error_message:
            response['error'] = result.error_message

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in multi-language preprocessing: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@language_bp.route('/process-language-aware', methods=['POST'])
def process_language_aware():
    """
    Process text with language-aware analysis

    Request body:
    {
        "text": "Input text to analyze",
        "detected_language": "pt",  // optional
        "options": {
            "use_stemming": true,
            "use_lemmatization": true,
            "extract_financial_terms": true,
            "enable_translation": false
        }
    }

    Response:
    {
        "success": true,
        "original_text": "...",
        "processed_text": "...",
        "detected_language": "pt",
        "tokens": [...],
        "lemmas": [...],
        "financial_terms": [...],
        "entities": [...],
        "quality_metrics": {...},
        "processing_time": 0.345
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: text'
            }), 400

        text = data['text']
        detected_language = data.get('detected_language')
        options = data.get('options', {})

        processor = get_language_aware_processor()
        result = processor.process_text(text, detected_language)

        response = {
            'success': result.success,
            'original_text': result.original_text,
            'processed_text': result.processed_text,
            'detected_language': result.detected_language,
            'language_confidence': result.language_confidence,
            'processing_time': result.processing_time,
            'quality_metrics': result.quality_metrics
        }

        if result.tokens:
            response['tokens'] = result.tokens[:100]  # Limit for API response
        if result.lemmas:
            response['lemmas'] = result.lemmas[:100]
        if result.stems:
            response['stems'] = result.stems[:100]
        if result.entities:
            response['entities'] = result.entities
        if result.financial_terms:
            response['financial_terms'] = result.financial_terms
        if result.translations:
            response['translations'] = result.translations

        if not result.success and result.error_message:
            response['error'] = result.error_message

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in language-aware processing: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@language_bp.route('/process-financial-international', methods=['POST'])
def process_financial_international():
    """
    Process financial text with international compliance and risk analysis

    Request body:
    {
        "text": "Financial transaction text",
        "detected_language": "pt",  // optional
        "options": {
            "enable_compliance_checking": true,
            "enable_risk_assessment": true,
            "include_jurisdictions": true
        }
    }

    Response:
    {
        "success": true,
        "original_text": "...",
        "processed_text": "...",
        "detected_language": "pt",
        "currencies": [...],
        "banks": [...],
        "compliance_flags": [...],
        "risk_assessment": {...},
        "jurisdictions": [...],
        "processing_time": 0.456
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: text'
            }), 400

        text = data['text']
        detected_language = data.get('detected_language')
        options = data.get('options', {})

        processor = get_international_financial_processor()
        result = processor.process_financial_text(text, detected_language)

        response = {
            'success': result.success,
            'original_text': result.original_text,
            'processed_text': result.processed_text,
            'detected_language': result.detected_language,
            'processing_time': result.processing_time
        }

        if result.currencies:
            response['currencies'] = result.currencies
        if result.banks:
            response['banks'] = result.banks
        if result.transaction_patterns:
            response['transaction_patterns'] = result.transaction_patterns
        if result.compliance_flags:
            response['compliance_flags'] = result.compliance_flags
        if result.tax_indicators:
            response['tax_indicators'] = result.tax_indicators
        if result.risk_assessment:
            response['risk_assessment'] = result.risk_assessment
        if result.jurisdictions:
            response['jurisdictions'] = result.jurisdictions

        if not result.success and result.error_message:
            response['error'] = result.error_message

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in international financial processing: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@language_bp.route('/process-full-pipeline', methods=['POST'])
def process_full_pipeline():
    """
    Process text through the complete multi-language pipeline

    Request body:
    {
        "text": "Input text for full processing",
        "options": {
            "enable_multi_language": true,
            "enable_compliance_checking": true,
            "include_intermediate_results": false
        }
    }

    Response:
    {
        "success": true,
        "original_text": "...",
        "processed_text": "...",
        "detected_language": "pt",
        "language_confidence": 0.95,
        "intermediate_results": {...},
        "quality_metrics": {...},
        "processing_time": 0.567
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: text'
            }), 400

        text = data['text']
        options = data.get('options', {})

        pipeline = get_preprocessing_pipeline()
        result = pipeline.process_text(text)

        response = {
            'success': result.success,
            'original_text': result.original_text,
            'processed_text': result.processed_text,
            'processing_time': result.processing_time,
            'quality_metrics': result.quality_metrics
        }

        # Add language detection results if available
        if result.detected_language:
            response['detected_language'] = result.detected_language
            response['language_confidence'] = result.language_confidence

        # Include intermediate results if requested
        if options.get('include_intermediate_results', False):
            response['intermediate_results'] = result.intermediate_results

        if not result.success and result.error_message:
            response['error'] = result.error_message

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in full pipeline processing: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@language_bp.route('/batch-detect-language', methods=['POST'])
def batch_detect_language():
    """
    Detect languages for a batch of texts

    Request body:
    {
        "texts": ["Text 1", "Text 2", "Text 3"],
        "options": {
            "include_alternatives": false,
            "confidence_threshold": 0.7
        }
    }

    Response:
    {
        "success": true,
        "results": [
            {
                "text": "Text 1",
                "language": "pt",
                "confidence": 0.95,
                "processing_time": 0.123
            },
            ...
        ],
        "summary": {
            "total_processed": 3,
            "language_distribution": {"pt": 2, "en": 1}
        }
    }
    """
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: texts'
            }), 400

        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({
                'success': False,
                'error': 'texts must be a list'
            }), 400

        options = data.get('options', {})

        detector = get_language_detector()
        results = detector.detect_batch(texts)

        # Prepare response
        response_results = []
        language_distribution = {}

        for i, result in enumerate(results):
            result_data = {
                'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                'language': result.consensus_language,
                'confidence': result.consensus_confidence,
                'processing_time': result.primary_detection.processing_time,
                'success': result.consensus_language != 'unknown'
            }

            if options.get('include_alternatives', False):
                result_data['alternatives'] = [
                    {'language': alt[0], 'confidence': alt[1]}
                    for alt in result.primary_detection.alternatives
                ]

            response_results.append(result_data)

            # Update language distribution
            lang = result.consensus_language
            language_distribution[lang] = language_distribution.get(lang, 0) + 1

        response = {
            'success': True,
            'results': response_results,
            'summary': {
                'total_processed': len(texts),
                'language_distribution': language_distribution
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in batch language detection: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@language_bp.route('/language-stats', methods=['GET'])
def get_language_stats():
    """
    Get language processing statistics

    Response:
    {
        "success": true,
        "language_detector": {...},
        "multi_lang_preprocessor": {...},
        "language_aware_processor": {...},
        "international_financial_processor": {...},
        "preprocessing_pipeline": {...}
    }
    """
    try:
        stats = {
            'success': True
        }

        # Get stats from each service
        if _language_detector:
            stats['language_detector'] = _language_detector.get_performance_stats()

        if _multi_lang_preprocessor:
            stats['multi_lang_preprocessor'] = _multi_lang_preprocessor.get_performance_stats()

        if _language_aware_processor:
            stats['language_aware_processor'] = _language_aware_processor.get_performance_stats()

        if _international_financial_processor:
            stats['international_financial_processor'] = _international_financial_processor.get_performance_stats()

        if _preprocessing_pipeline:
            stats['preprocessing_pipeline'] = _preprocessing_pipeline.get_pipeline_metrics()

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error getting language stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@language_bp.route('/supported-languages', methods=['GET'])
def get_supported_languages():
    """
    Get list of supported languages and their capabilities

    Response:
    {
        "success": true,
        "languages": {
            "pt": {
                "name": "Portuguese",
                "stemming": true,
                "lemmatization": true,
                "ner": true,
                "financial_terms": true
            },
            ...
        },
        "features": [...]
    }
    """
    try:
        languages = {
            'pt': {
                'name': 'Portuguese',
                'stemming': True,
                'lemmatization': True,
                'ner': True,
                'financial_terms': True,
                'currency_support': True,
                'date_formats': True
            },
            'en': {
                'name': 'English',
                'stemming': True,
                'lemmatization': True,
                'ner': True,
                'financial_terms': True,
                'currency_support': True,
                'date_formats': True
            },
            'es': {
                'name': 'Spanish',
                'stemming': True,
                'lemmatization': True,
                'ner': True,
                'financial_terms': True,
                'currency_support': True,
                'date_formats': True
            },
            'fr': {
                'name': 'French',
                'stemming': True,
                'lemmatization': True,
                'ner': True,
                'financial_terms': True,
                'currency_support': True,
                'date_formats': True
            },
            'de': {
                'name': 'German',
                'stemming': True,
                'lemmatization': True,
                'ner': True,
                'financial_terms': True,
                'currency_support': True,
                'date_formats': True
            },
            'it': {
                'name': 'Italian',
                'stemming': True,
                'lemmatization': True,
                'ner': True,
                'financial_terms': True,
                'currency_support': True,
                'date_formats': True
            }
        }

        features = [
            'language_detection',
            'text_preprocessing',
            'entity_extraction',
            'currency_normalization',
            'date_normalization',
            'financial_term_identification',
            'stemming_lemmatization',
            'named_entity_recognition',
            'translation_support',
            'compliance_checking',
            'risk_assessment'
        ]

        return jsonify({
            'success': True,
            'languages': languages,
            'features': features
        })

    except Exception as e:
        logger.error(f"Error getting supported languages: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500