from flask import Blueprint, request, jsonify
from datetime import datetime
from src.services.model_manager import ModelManager
from src.services.imbalanced_data_handler import ImbalancedDataHandler
from src.services.financial_category_balancer import FinancialCategoryBalancer
from src.services.quality_assessment_engine import QualityAssessmentEngine
from src.models.transaction import Transaction, db
from src.models.company_financial import CompanyFinancial
from src.utils.logging_config import get_logger
from src.utils.error_handler import handle_errors
from src.utils.exceptions import ValidationError, InsufficientDataError
from src.utils.validation_middleware import rate_limit, validate_input_fields

logger = get_logger(__name__)

model_manager_bp = Blueprint('model_manager', __name__)

# Initialize services
model_manager = ModelManager()

# Initialize imbalance handling services
imbalance_config = {
    'smote_config': {'random_state': 42},
    'synthetic_config': {'methods': ['vae', 'gan'], 'sample_size_ratio': 0.5}
}
imbalanced_data_handler = ImbalancedDataHandler(imbalance_config)

financial_balancer_config = {
    'smote_config': {'random_state': 42},
    'synthetic_config': {'methods': ['vae'], 'sample_size_ratio': 0.3}
}
financial_category_balancer = FinancialCategoryBalancer(financial_balancer_config)

quality_config = {}
quality_assessment_engine = QualityAssessmentEngine(quality_config)


@model_manager_bp.route('/models/train-with-augmentation', methods=['POST'])
@handle_errors
@rate_limit(max_requests=5, window_minutes=60)  # Lower limit for augmentation training
@validate_input_fields('model_type', 'data_source')
def train_model_with_augmentation():
    """
    Train a machine learning model with integrated data augmentation

    Expected JSON payload:
    {
        "model_type": "auto|kmeans|random_forest|xgboost|lightgbm|bert",
        "data_source": "transactions|company_financial",
        "use_augmentation": true,
        "augmentation_config": {
            "general": {
                "augmentation_ratio": 2.0
            },
            "text_augmentation": {
                "enabled": true,
                "strategies": ["synonym_replacement", "paraphrasing"]
            },
            "numerical_augmentation": {
                "enabled": true,
                "strategies": ["gaussian_noise", "scaling"]
            }
        },
        "optimize": false,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "category_filter": "optional_category"
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        model_type = data.get('model_type', 'auto')
        data_source = data.get('data_source', 'transactions')
        use_augmentation = data.get('use_augmentation', True)
        augmentation_config = data.get('augmentation_config')
        optimize = data.get('optimize', False)
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        category_filter = data.get('category_filter')

        # Validate model type
        valid_models = ['auto', 'kmeans', 'random_forest', 'xgboost', 'lightgbm', 'bert']
        if model_type not in valid_models:
            raise ValidationError(f"Invalid model type. Must be one of: {', '.join(valid_models)}")

        # Validate data source
        valid_sources = ['transactions', 'company_financial']
        if data_source not in valid_sources:
            raise ValidationError(f"Invalid data source. Must be one of: {', '.join(valid_sources)}")

        # Fetch training data
        if data_source == 'transactions':
            query = Transaction.query
        else:  # company_financial
            query = CompanyFinancial.query

        # Apply filters
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date() if data_source == 'transactions'
                                else CompanyFinancial.date >= start_date_obj.date())

        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date() if data_source == 'transactions'
                                else CompanyFinancial.date <= end_date_obj.date())

        if category_filter:
            query = query.filter(Transaction.category == category_filter if data_source == 'transactions'
                                else CompanyFinancial.category == category_filter)

        # Get data
        raw_data = query.all()

        if len(raw_data) < 10:
            raise InsufficientDataError('model training', 10, len(raw_data))

        # Convert to dictionaries
        training_data = [item.to_dict() for item in raw_data]

        # Train model with augmentation
        result = model_manager.train_model_with_augmentation(
            model_type=model_type,
            transactions=training_data,
            target_column='category',
            use_augmentation=use_augmentation,
            augmentation_config=augmentation_config,
            optimize=optimize
        )

        if result['success']:
            logger.info(f"Model {model_type} trained with augmentation successfully on {len(training_data)} samples")

            return jsonify({
                'success': True,
                'message': f'Model {model_type} trained with augmentation successfully',
                'data': {
                    'model_type': model_type,
                    'data_source': data_source,
                    'training_samples': len(training_data),
                    'augmentation_used': use_augmentation,
                    'training_result': result
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Training failed')
            }), 500

    except Exception as e:
        logger.error(f"Error training model with augmentation: {str(e)}")
        return jsonify({'error': f'Training with augmentation failed: {str(e)}'}), 500


@model_manager_bp.route('/models/train', methods=['POST'])
@handle_errors
@rate_limit(max_requests=10, window_minutes=60)  # Lower limit for training operations
@validate_input_fields('model_type', 'data_source')
def train_model():
    """
    Train a machine learning model with specified configuration

    Expected JSON payload:
    {
        "model_type": "auto|kmeans|random_forest|xgboost|lightgbm|bert",
        "data_source": "transactions|company_financial",
        "optimize": true|false,
        "n_trials": 50,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "category_filter": "optional_category"
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        model_type = data.get('model_type', 'auto')
        data_source = data.get('data_source', 'transactions')
        optimize = data.get('optimize', False)
        n_trials = data.get('n_trials', 50)
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        category_filter = data.get('category_filter')

        # Validate model type
        valid_models = ['auto', 'kmeans', 'random_forest', 'xgboost', 'lightgbm', 'bert']
        if model_type not in valid_models:
            raise ValidationError(f"Invalid model type. Must be one of: {', '.join(valid_models)}")

        # Validate data source
        valid_sources = ['transactions', 'company_financial']
        if data_source not in valid_sources:
            raise ValidationError(f"Invalid data source. Must be one of: {', '.join(valid_sources)}")

        # Fetch training data
        if data_source == 'transactions':
            query = Transaction.query
        else:  # company_financial
            query = CompanyFinancial.query

        # Apply filters
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date >= start_date_obj.date())

        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date <= end_date_obj.date())

        if category_filter:
            query = query.filter(Transaction.category == category_filter if data_source == 'transactions'
                               else CompanyFinancial.category == category_filter)

        # Get data
        raw_data = query.all()

        if len(raw_data) < 10:
            raise InsufficientDataError('model training', 10, len(raw_data))

        # Convert to dictionaries
        training_data = [item.to_dict() for item in raw_data]

        # Process data through ModelManager
        X, y, feature_names = model_manager.process_data(training_data)

        if len(X) == 0:
            raise InsufficientDataError('processed training data', 1, 0)

        # Extract texts for BERT if needed
        texts = None
        if model_type == 'bert' or (model_type == 'auto' and any('description' in item for item in training_data)):
            texts = [item.get('description', '') for item in training_data]

        # Train the model
        kwargs = {}
        if texts:
            kwargs['texts'] = texts

        result = model_manager.train_model(model_type, X, y, optimize=optimize, n_trials=n_trials, **kwargs)

        if result['success']:
            logger.info(f"Model {model_type} trained successfully on {len(training_data)} samples")

            return jsonify({
                'success': True,
                'message': f'Model {model_type} trained successfully',
                'data': {
                    'model_type': model_type,
                    'data_source': data_source,
                    'training_samples': len(training_data),
                    'feature_count': len(feature_names),
                    'training_result': result
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Training failed')
            }), 500

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return jsonify({'error': f'Training failed: {str(e)}'}), 500


@model_manager_bp.route('/models/predict', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@validate_input_fields('model_type', 'data')
def predict():
    """
    Make predictions using a trained model

    Expected JSON payload:
    {
        "model_type": "kmeans|random_forest|xgboost|lightgbm|bert",
        "data": {
            "description": "transaction description",
            "amount": 100.0,
            "date": "2024-01-01",
            "category": "optional_existing_category"
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        model_type = data.get('model_type')
        prediction_data = data.get('data')

        if not model_type or not prediction_data:
            raise ValidationError("model_type and data are required")

        # Validate model type
        valid_models = ['kmeans', 'random_forest', 'xgboost', 'lightgbm', 'bert']
        if model_type not in valid_models:
            raise ValidationError(f"Invalid model type. Must be one of: {', '.join(valid_models)}")

        # Process prediction data
        processed_data = [prediction_data]
        X, _, _ = model_manager.process_data(processed_data)

        if len(X) == 0:
            raise ValidationError("Failed to process prediction data")

        # Extract texts for BERT
        texts = None
        if model_type == 'bert':
            texts = [prediction_data.get('description', '')]
            # For BERT, we need to create a special input format
            X = type('MockData', (), {'texts': texts})()

        # Make prediction
        try:
            predictions = model_manager.predict(model_type, X, texts=texts if texts else None)

            # Get prediction probabilities if available
            try:
                probabilities = model_manager.predict_proba(model_type, X, texts=texts if texts else None)
                probabilities_list = probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
            except:
                probabilities_list = None

            return jsonify({
                'success': True,
                'data': {
                    'model_type': model_type,
                    'prediction': predictions[0] if len(predictions) > 0 else None,
                    'probabilities': probabilities_list[0] if probabilities_list and len(probabilities_list) > 0 else None,
                    'confidence': max(probabilities_list[0]) if probabilities_list and len(probabilities_list[0]) > 0 else None
                }
            })

        except Exception as pred_error:
            logger.warning(f"Prediction failed, trying fallback: {str(pred_error)}")

            # Try fallback prediction
            try:
                fallback_predictions = model_manager._fallback_predict(X)
                return jsonify({
                    'success': True,
                    'data': {
                        'model_type': model_type,
                        'prediction': fallback_predictions[0] if len(fallback_predictions) > 0 else 'outros',
                        'fallback_used': True,
                        'original_error': str(pred_error)
                    }
                })
            except Exception as fallback_error:
                return jsonify({
                    'success': False,
                    'error': f'Prediction failed: {str(pred_error)}, Fallback failed: {str(fallback_error)}'
                }), 500

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@model_manager_bp.route('/models/compare-with-augmentation', methods=['POST'])
@handle_errors
@rate_limit(max_requests=10, window_minutes=60)
@validate_input_fields('data_source')
def compare_models_with_augmentation():
    """
    Compare multiple models with integrated data augmentation

    Expected JSON payload:
    {
        "data_source": "transactions|company_financial",
        "models_to_compare": ["random_forest", "xgboost", "lightgbm", "bert"],
        "use_augmentation": true,
        "augmentation_config": {
            "general": {
                "augmentation_ratio": 1.5
            },
            "text_augmentation": {
                "enabled": true,
                "strategies": ["synonym_replacement"]
            }
        },
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        data_source = data.get('data_source', 'transactions')
        models_to_compare = data.get('models_to_compare', ['random_forest', 'xgboost', 'lightgbm'])
        use_augmentation = data.get('use_augmentation', True)
        augmentation_config = data.get('augmentation_config')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        # Validate data source
        valid_sources = ['transactions', 'company_financial']
        if data_source not in valid_sources:
            raise ValidationError(f"Invalid data source. Must be one of: {', '.join(valid_sources)}")

        # Fetch comparison data
        if data_source == 'transactions':
            query = Transaction.query
        else:
            query = CompanyFinancial.query

        # Apply date filters
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date() if data_source == 'transactions'
                                else CompanyFinancial.date >= start_date_obj.date())

        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date() if data_source == 'transactions'
                                else CompanyFinancial.date <= end_date_obj.date())

        # Get data
        raw_data = query.all()

        if len(raw_data) < 20:  # Need more data for meaningful comparison
            raise InsufficientDataError('model comparison', 20, len(raw_data))

        # Convert to dictionaries
        comparison_data = [item.to_dict() for item in raw_data]

        # Compare models with augmentation
        comparison_result = model_manager.compare_models_with_augmentation(
            transactions=comparison_data,
            target_column='category',
            models_to_compare=models_to_compare,
            use_augmentation=use_augmentation,
            augmentation_config=augmentation_config
        )

        if 'error' not in comparison_result:
            logger.info(f"Model comparison with augmentation completed. Best model: {comparison_result.get('best_model')}")

            return jsonify({
                'success': True,
                'message': 'Model comparison with augmentation completed successfully',
                'data': comparison_result
            })
        else:
            return jsonify({
                'success': False,
                'error': comparison_result['error']
            }), 500

    except Exception as e:
        logger.error(f"Error comparing models with augmentation: {str(e)}")
        return jsonify({'error': f'Model comparison with augmentation failed: {str(e)}'}), 500


@model_manager_bp.route('/models/compare', methods=['POST'])
@handle_errors
@rate_limit(max_requests=20, window_minutes=60)
@validate_input_fields('data_source')
def compare_models():
    """
    Compare performance of multiple models

    Expected JSON payload:
    {
        "data_source": "transactions|company_financial",
        "models_to_compare": ["random_forest", "xgboost", "lightgbm", "bert"],
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        data_source = data.get('data_source', 'transactions')
        models_to_compare = data.get('models_to_compare', ['random_forest', 'xgboost', 'lightgbm'])
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        # Validate data source
        valid_sources = ['transactions', 'company_financial']
        if data_source not in valid_sources:
            raise ValidationError(f"Invalid data source. Must be one of: {', '.join(valid_sources)}")

        # Fetch comparison data
        if data_source == 'transactions':
            query = Transaction.query
        else:
            query = CompanyFinancial.query

        # Apply date filters
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date >= start_date_obj.date())

        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date <= end_date_obj.date())

        # Get data
        raw_data = query.all()

        if len(raw_data) < 20:  # Need more data for meaningful comparison
            raise InsufficientDataError('model comparison', 20, len(raw_data))

        # Convert to dictionaries
        comparison_data = [item.to_dict() for item in raw_data]

        # Process data
        X, y, feature_names = model_manager.process_data(comparison_data)

        if len(X) == 0:
            raise InsufficientDataError('processed comparison data', 1, 0)

        # Extract texts for BERT
        texts = None
        if 'bert' in models_to_compare:
            texts = [item.get('description', '') for item in comparison_data]

        # Compare models
        kwargs = {}
        if texts:
            kwargs['texts'] = texts

        comparison_result = model_manager.compare_models(X, y, models_to_compare, **kwargs)

        if 'error' not in comparison_result:
            logger.info(f"Model comparison completed. Best model: {comparison_result.get('best_model')}")

            # Add performance report if detailed comparison is available
            if 'detailed_comparison' in comparison_result:
                # Generate comprehensive report
                try:
                    report = model_manager.generate_performance_report(
                        comparison_result['detailed_comparison'],
                        {}  # data_characteristics will be extracted from results
                    )
                    comparison_result['performance_report'] = report
                except Exception as report_error:
                    logger.warning(f"Could not generate performance report: {str(report_error)}")

            return jsonify({
                'success': True,
                'message': 'Model comparison completed successfully',
                'data': comparison_result
            })
        else:
            return jsonify({
                'success': False,
                'error': comparison_result['error']
            }), 500

    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        return jsonify({'error': f'Model comparison failed: {str(e)}'}), 500


@model_manager_bp.route('/models/optimize', methods=['POST'])
@handle_errors
@rate_limit(max_requests=5, window_minutes=60)  # Very low limit for optimization
@validate_input_fields('model_type', 'data_source')
def optimize_model():
    """
    Optimize hyperparameters for a specific model

    Expected JSON payload:
    {
        "model_type": "random_forest|xgboost|lightgbm",
        "data_source": "transactions|company_financial",
        "n_trials": 50,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        model_type = data.get('model_type')
        data_source = data.get('data_source', 'transactions')
        n_trials = data.get('n_trials', 50)
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        # Validate model type (only supervised models can be optimized)
        valid_models = ['random_forest', 'xgboost', 'lightgbm']
        if model_type not in valid_models:
            raise ValidationError(f"Invalid model type for optimization. Must be one of: {', '.join(valid_models)}")

        # Fetch optimization data
        if data_source == 'transactions':
            query = Transaction.query
        else:
            query = CompanyFinancial.query

        # Apply date filters
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date >= start_date_obj.date())

        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date <= end_date_obj.date())

        # Get data
        raw_data = query.all()

        if len(raw_data) < 30:  # Need sufficient data for optimization
            raise InsufficientDataError('model optimization', 30, len(raw_data))

        # Convert to dictionaries
        optimization_data = [item.to_dict() for item in raw_data]

        # Process data
        X, y, feature_names = model_manager.process_data(optimization_data)

        if len(X) == 0:
            raise InsufficientDataError('processed optimization data', 1, 0)

        # Run optimization
        optimization_result = model_manager.optimize_hyperparameters(model_type, X, y, n_trials=n_trials)

        if optimization_result['success']:
            logger.info(f"Model {model_type} optimization completed. Best score: {optimization_result['best_score']:.4f}")

            return jsonify({
                'success': True,
                'message': f'Model {model_type} optimization completed successfully',
                'data': optimization_result
            })
        else:
            return jsonify({
                'success': False,
                'error': optimization_result.get('error', 'Optimization failed')
            }), 500

    except Exception as e:
        logger.error(f"Error optimizing model: {str(e)}")
        return jsonify({'error': f'Optimization failed: {str(e)}'}), 500


@model_manager_bp.route('/models/info', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_models_info():
    """
    Get information about all available models
    """
    try:
        models_info = model_manager.get_model_info()

        return jsonify({
            'success': True,
            'data': {
                'models': models_info,
                'feature_importance': model_manager.get_feature_importance()
            }
        })

    except Exception as e:
        logger.error(f"Error getting models info: {str(e)}")
        return jsonify({'error': f'Failed to get models info: {str(e)}'}), 500


@model_manager_bp.route('/models/<model_type>/info', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_model_info(model_type):
    """
    Get detailed information about a specific model
    """
    try:
        # Validate model type
        valid_models = ['kmeans', 'random_forest', 'xgboost', 'lightgbm', 'bert']
        if model_type not in valid_models:
            raise ValidationError(f"Invalid model type. Must be one of: {', '.join(valid_models)}")

        model_info = model_manager.get_model_info(model_type)

        if not model_info:
            return jsonify({
                'success': False,
                'error': f'Model {model_type} not found'
            }), 404

        return jsonify({
            'success': True,
            'data': model_info
        })

    except Exception as e:
        logger.error(f"Error getting model info for {model_type}: {str(e)}")
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500


@model_manager_bp.route('/models/select', methods=['POST'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
@validate_input_fields('data_source')
def select_best_model():
    """
    Automatically select the best model for the given data

    Expected JSON payload:
    {
        "data_source": "transactions|company_financial",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "candidate_models": ["random_forest", "xgboost", "lightgbm", "bert"]
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        data_source = data.get('data_source', 'transactions')
        candidate_models = data.get('candidate_models')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        # Fetch selection data
        if data_source == 'transactions':
            query = Transaction.query
        else:
            query = CompanyFinancial.query

        # Apply date filters
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date >= start_date_obj.date())

        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date <= end_date_obj.date())

        # Get data
        raw_data = query.all()

        if len(raw_data) < 20:
            raise InsufficientDataError('model selection', 20, len(raw_data))

        # Convert to dictionaries
        selection_data = [item.to_dict() for item in raw_data]

        # Process data
        X, y, feature_names = model_manager.process_data(selection_data)

        if len(X) == 0:
            raise InsufficientDataError('processed selection data', 1, 0)

        # Extract texts for BERT consideration
        texts = None
        if not candidate_models or 'bert' in candidate_models:
            texts = [item.get('description', '') for item in selection_data]
            if texts and any(text.strip() for text in texts):
                # Add BERT to candidates if we have text data
                if not candidate_models:
                    candidate_models = ['random_forest', 'xgboost', 'lightgbm', 'bert']
                elif 'bert' not in candidate_models:
                    candidate_models.append('bert')

        # Select best model using advanced selection
        selection_result = model_manager.select_best_model(X, y, candidate_models)

        logger.info(f"Best model selected: {selection_result.get('best_model')}")

        return jsonify({
            'success': True,
            'message': f'Best model selected: {selection_result.get("best_model")}',
            'data': {
                'best_model': selection_result.get('best_model'),
                'recommendation': selection_result.get('recommendation'),
                'candidate_models': candidate_models,
                'data_samples': len(selection_data),
                'features_count': len(feature_names),
                'selection_details': selection_result
            }
        })

    except Exception as e:
        logger.error(f"Error selecting best model: {str(e)}")
        return jsonify({'error': f'Model selection failed: {str(e)}'}), 500


@model_manager_bp.route('/models/<model_type>/evaluate', methods=['POST'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
@validate_input_fields('data_source')
def evaluate_model(model_type):
    """
    Evaluate a specific model's performance

    Expected JSON payload:
    {
        "data_source": "transactions|company_financial",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "test_size": 0.2
    }
    """
    try:
        # Validate model type
        valid_models = ['kmeans', 'random_forest', 'xgboost', 'lightgbm', 'bert']
        if model_type not in valid_models:
            raise ValidationError(f"Invalid model type. Must be one of: {', '.join(valid_models)}")

        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        data_source = data.get('data_source', 'transactions')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        test_size = data.get('test_size', 0.2)

        # Fetch evaluation data
        if data_source == 'transactions':
            query = Transaction.query
        else:
            query = CompanyFinancial.query

        # Apply date filters
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date >= start_date_obj.date())

        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date <= end_date_obj.date())

        # Get data
        raw_data = query.all()

        if len(raw_data) < 10:
            raise InsufficientDataError('model evaluation', 10, len(raw_data))

        # Convert to dictionaries
        evaluation_data = [item.to_dict() for item in raw_data]

        # Process data
        X, y, feature_names = model_manager.process_data(evaluation_data)

        if len(X) == 0:
            raise InsufficientDataError('processed evaluation data', 1, 0)

        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(set(y)) > 1 else None
        )

        # Extract texts for BERT
        texts_train = texts_test = None
        if model_type == 'bert':
            texts = [item.get('description', '') for item in evaluation_data]
            texts_train = texts[:len(X_train)]
            texts_test = texts[len(X_train):]

        # Evaluate model
        kwargs = {}
        if texts_test:
            kwargs['texts'] = texts_test

        evaluation_result = model_manager.evaluate_model(model_type, X_train, y_train, X_test, y_test, **kwargs)

        if 'error' not in evaluation_result:
            logger.info(f"Model {model_type} evaluation completed")

            return jsonify({
                'success': True,
                'message': f'Model {model_type} evaluation completed successfully',
                'data': evaluation_result
            })
        else:
            return jsonify({
                'success': False,
                'error': evaluation_result['error']
            }), 500

    except Exception as e:
        logger.error(f"Error evaluating model {model_type}: {str(e)}")
        return jsonify({'error': f'Model evaluation failed: {str(e)}'}), 500


@model_manager_bp.route('/models/batch-predict', methods=['POST'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
@validate_input_fields('model_type', 'data')
def batch_predict():
    """
    Make batch predictions using a trained model

    Expected JSON payload:
    {
        "model_type": "kmeans|random_forest|xgboost|lightgbm|bert",
        "data": [
            {
                "description": "transaction description 1",
                "amount": 100.0,
                "date": "2024-01-01"
            },
            {
                "description": "transaction description 2",
                "amount": 200.0,
                "date": "2024-01-02"
            }
        ]
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        model_type = data.get('model_type')
        batch_data = data.get('data', [])

        if not model_type or not batch_data:
            raise ValidationError("model_type and data are required")

        if not isinstance(batch_data, list):
            raise ValidationError("data must be a list of prediction items")

        if len(batch_data) == 0:
            raise ValidationError("data list cannot be empty")

        if len(batch_data) > 100:  # Limit batch size
            raise ValidationError("Batch size cannot exceed 100 items")

        # Validate model type
        valid_models = ['kmeans', 'random_forest', 'xgboost', 'lightgbm', 'bert']
        if model_type not in valid_models:
            raise ValidationError(f"Invalid model type. Must be one of: {', '.join(valid_models)}")

        # Process batch data
        X, _, _ = model_manager.process_data(batch_data)

        if len(X) == 0:
            raise ValidationError("Failed to process batch data")

        # Extract texts for BERT
        texts = None
        if model_type == 'bert':
            texts = [item.get('description', '') for item in batch_data]
            # For BERT, create special input format
            X = type('MockData', (), {'texts': texts})()

        # Make batch predictions
        try:
            predictions = model_manager.predict(model_type, X, texts=texts if texts else None)

            # Get prediction probabilities if available
            try:
                probabilities = model_manager.predict_proba(model_type, X, texts=texts if texts else None)
                probabilities_list = probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
            except:
                probabilities_list = None

            # Format results
            results = []
            for i, prediction in enumerate(predictions):
                result = {
                    'index': i,
                    'prediction': prediction,
                    'input_data': batch_data[i]
                }

                if probabilities_list and i < len(probabilities_list):
                    result['probabilities'] = probabilities_list[i]
                    if probabilities_list[i]:
                        result['confidence'] = max(probabilities_list[i])

                results.append(result)

            return jsonify({
                'success': True,
                'data': {
                    'model_type': model_type,
                    'batch_size': len(batch_data),
                    'results': results
                }
            })

        except Exception as pred_error:
            logger.warning(f"Batch prediction failed, trying fallback: {str(pred_error)}")

            # Try fallback prediction
            try:
                fallback_predictions = model_manager._fallback_predict(X)
                results = []
                for i, prediction in enumerate(fallback_predictions):
                    results.append({
                        'index': i,
                        'prediction': prediction,
                        'input_data': batch_data[i],
                        'fallback_used': True,
                        'original_error': str(pred_error)
                    })

                return jsonify({
                    'success': True,
                    'data': {
                        'model_type': model_type,
                        'batch_size': len(batch_data),
                        'results': results,
                        'fallback_used': True
                    }
                })
            except Exception as fallback_error:
                return jsonify({
                    'success': False,
                    'error': f'Batch prediction failed: {str(pred_error)}, Fallback failed: {str(fallback_error)}'
                }), 500

    except Exception as e:
        logger.error(f"Error making batch prediction: {str(e)}")
@model_manager_bp.route('/models/advanced-select', methods=['POST'])
@handle_errors
@rate_limit(max_requests=20, window_minutes=60)
@validate_input_fields('data_source')
def advanced_model_selection():
    """
    Advanced model selection with comprehensive analysis and recommendations

    Expected JSON payload:
    {
        "data_source": "transactions|company_financial",
        "candidate_models": ["random_forest", "xgboost", "lightgbm", "bert"],
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "include_data_analysis": true,
        "generate_report": true
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        data_source = data.get('data_source', 'transactions')
        candidate_models = data.get('candidate_models')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        include_data_analysis = data.get('include_data_analysis', True)
        generate_report = data.get('generate_report', True)

        # Validate data source
        valid_sources = ['transactions', 'company_financial']
        if data_source not in valid_sources:
            raise ValidationError(f"Invalid data source. Must be one of: {', '.join(valid_sources)}")

        # Fetch data
        if data_source == 'transactions':
            query = Transaction.query
        else:
            query = CompanyFinancial.query

        # Apply date filters
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date() if data_source == 'transactions'
                                else CompanyFinancial.date >= start_date_obj.date())

        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date() if data_source == 'transactions'
                                else CompanyFinancial.date <= end_date_obj.date())

        # Get data
        raw_data = query.all()

        if len(raw_data) < 50:  # Need sufficient data for meaningful analysis
            raise InsufficientDataError('advanced model selection', 50, len(raw_data))

        # Convert to dictionaries
        selection_data = [item.to_dict() for item in raw_data]

        # Process data
        X, y, feature_names = model_manager.process_data(selection_data)

        if len(X) == 0:
            raise InsufficientDataError('processed selection data', 1, 0)

        # Extract texts for BERT
        texts = None
        if candidate_models and 'bert' in candidate_models:
            texts = [item.get('description', '') for item in selection_data]

        # Perform advanced model selection
        kwargs = {}
        if texts:
            kwargs['texts'] = texts

        selection_result = model_manager.select_best_model(X, y, candidate_models, **kwargs)

        # Add data analysis if requested
        if include_data_analysis:
            data_analysis = model_manager.analyze_data_characteristics(X, y, feature_names)
            selection_result['data_analysis'] = data_analysis

        # Generate performance report if requested
        if generate_report and 'comparison_results' in selection_result:
            # Convert comparison results to expected format
            comparison_results = []
            for result_dict in selection_result['comparison_results']:
                # Create a mock object with the expected attributes
                class MockResult:
                    def __init__(self, data):
                        self.model_name = data.get('model_name', '')
                        self.performance_metrics = type('MockMetrics', (), {
                            'f1_score': data.get('performance_metrics', {}).get('f1_score', 0),
                            'accuracy': data.get('performance_metrics', {}).get('accuracy', 0),
                            'precision': data.get('performance_metrics', {}).get('precision', 0),
                            'recall': data.get('performance_metrics', {}).get('recall', 0),
                            'cross_val_mean': data.get('performance_metrics', {}).get('cross_val_mean'),
                            'cross_val_std': data.get('performance_metrics', {}).get('cross_val_std'),
                            'training_time': data.get('performance_metrics', {}).get('training_time'),
                            'to_dict': lambda: data.get('performance_metrics', {})
                        })()
                        self.rank = data.get('rank', 0)
                        self.recommendation_score = data.get('recommendation_score', 0)
                        self.data_characteristics = data.get('data_characteristics', {})
                        self.timestamp = data.get('timestamp', '')

                    def to_dict(self):
                        return {
                            'model_name': self.model_name,
                            'performance_metrics': self.performance_metrics.to_dict(),
                            'rank': self.rank,
                            'recommendation_score': self.recommendation_score,
                            'data_characteristics': self.data_characteristics,
                            'timestamp': self.timestamp
                        }

                comparison_results.append(MockResult(result_dict))

            report = model_manager.generate_performance_report(
                comparison_results,
                selection_result.get('data_analysis', selection_result.get('data_characteristics', {}))
            )
            selection_result['performance_report'] = report

        logger.info(f"Advanced model selection completed. Best model: {selection_result.get('best_model')}")

        return jsonify({
            'success': True,
            'message': 'Advanced model selection completed successfully',
            'data': selection_result
        })

    except Exception as e:
        logger.error(f"Error in advanced model selection: {str(e)}")
        return jsonify({'error': f'Advanced model selection failed: {str(e)}'}), 500


@model_manager_bp.route('/models/ab-test', methods=['POST'])
@handle_errors
@rate_limit(max_requests=10, window_minutes=60)  # Lower limit for A/B testing
@validate_input_fields('model_a', 'model_b', 'data_source')
def perform_ab_test():
    """
    Perform A/B testing between two models

    Expected JSON payload:
    {
        "model_a": "random_forest",
        "model_b": "xgboost",
        "data_source": "transactions|company_financial",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "test_duration_hours": 24
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        model_a = data.get('model_a')
        model_b = data.get('model_b')
        data_source = data.get('data_source', 'transactions')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        test_duration_hours = data.get('test_duration_hours', 24)

        # Validate models
        valid_models = ['random_forest', 'xgboost', 'lightgbm', 'bert']
        if model_a not in valid_models or model_b not in valid_models:
            raise ValidationError(f"Invalid model names. Must be one of: {', '.join(valid_models)}")

        if model_a == model_b:
            raise ValidationError("model_a and model_b must be different models")

        # Validate data source
        valid_sources = ['transactions', 'company_financial']
        if data_source not in valid_sources:
            raise ValidationError(f"Invalid data source. Must be one of: {', '.join(valid_sources)}")

        # Fetch data
        if data_source == 'transactions':
            query = Transaction.query
        else:
            query = CompanyFinancial.query

        # Apply date filters
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date() if data_source == 'transactions'
                                else CompanyFinancial.date >= start_date_obj.date())

        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date() if data_source == 'transactions'
                                else CompanyFinancial.date <= end_date_obj.date())

        # Get data
        raw_data = query.all()

        if len(raw_data) < 100:  # Need sufficient data for A/B testing
            raise InsufficientDataError('A/B testing', 100, len(raw_data))

        # Convert to dictionaries
        test_data = [item.to_dict() for item in raw_data]

        # Process data
        X, y, feature_names = model_manager.process_data(test_data)

        if len(X) == 0:
            raise InsufficientDataError('processed A/B test data', 1, 0)

        # Extract texts for BERT
        texts = None
        if model_a == 'bert' or model_b == 'bert':
            texts = [item.get('description', '') for item in test_data]

        # Perform A/B test
        kwargs = {}
        if texts:
            kwargs['texts'] = texts

        ab_result = model_manager.perform_ab_test(model_a, model_b, X, y, test_duration_hours, **kwargs)

        if 'error' not in ab_result:
            logger.info(f"A/B test completed: {model_a} vs {model_b}. Winner: {ab_result.get('winner')}")

            return jsonify({
                'success': True,
                'message': f'A/B test completed successfully',
                'data': ab_result
            })
        else:
            return jsonify({
                'success': False,
                'error': ab_result['error']
            }), 500

    except Exception as e:
        logger.error(f"Error performing A/B test: {str(e)}")
        return jsonify({'error': f'A/B test failed: {str(e)}'}), 500


@model_manager_bp.route('/models/performance-history', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
def get_performance_history():
    """
    Get historical performance data

    Query parameters:
    - model_name: Specific model name (optional)
    - days: Number of days to look back (default: 30)
    """
    try:
        model_name = request.args.get('model_name')
        days = int(request.args.get('days', 30))

        if days < 1 or days > 365:
            raise ValidationError("days must be between 1 and 365")

        history = model_manager.get_performance_history(model_name, days)

        return jsonify({
            'success': True,
            'data': history
        })

    except Exception as e:
        logger.error(f"Error getting performance history: {str(e)}")
        return jsonify({'error': f'Failed to get performance history: {str(e)}'}), 500


@model_manager_bp.route('/models/ab-tests', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
def get_ab_test_results():
    """
    Get A/B test results

    Query parameters:
    - test_id: Specific test ID (optional)
    """
    try:
        test_id = request.args.get('test_id')

        results = model_manager.get_ab_test_results(test_id)

        if test_id and 'error' in results:
            return jsonify({
                'success': False,
                'error': results['error']
            }), 404

        return jsonify({
            'success': True,
            'data': results
        })

    except Exception as e:
        logger.error(f"Error getting A/B test results: {str(e)}")
        return jsonify({'error': f'Failed to get A/B test results: {str(e)}'}), 500


@model_manager_bp.route('/models/data-analysis', methods=['POST'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
@validate_input_fields('data_source')
def analyze_data():
    """
    Analyze data characteristics for model selection guidance

    Expected JSON payload:
    {
        "data_source": "transactions|company_financial",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        data_source = data.get('data_source', 'transactions')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        # Validate data source
        valid_sources = ['transactions', 'company_financial']
        if data_source not in valid_sources:
            raise ValidationError(f"Invalid data source. Must be one of: {', '.join(valid_sources)}")

        # Fetch data
        if data_source == 'transactions':
            query = Transaction.query
        else:
            query = CompanyFinancial.query

        # Apply date filters
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date() if data_source == 'transactions'
                                else CompanyFinancial.date >= start_date_obj.date())

        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date() if data_source == 'transactions'
                                else CompanyFinancial.date <= end_date_obj.date())

        # Get data
        raw_data = query.all()

        if len(raw_data) < 10:
            raise InsufficientDataError('data analysis', 10, len(raw_data))

        # Convert to dictionaries
        analysis_data = [item.to_dict() for item in raw_data]

        # Process data
        X, y, feature_names = model_manager.process_data(analysis_data)

        if len(X) == 0:
            raise InsufficientDataError('processed analysis data', 1, 0)

        # Analyze data characteristics
        analysis = model_manager.analyze_data_characteristics(X, y, feature_names)

        # Generate recommendations based on analysis
        recommendations = []

        n_samples = analysis.get('n_samples', 0)
        n_features = analysis.get('n_features', 0)
        class_imbalance = analysis.get('data_complexity', {}).get('class_imbalance_ratio', 1.0)

        if n_samples < 1000:
            recommendations.append("Consider using simpler models due to small dataset size")
        if class_imbalance > 3.0:
            recommendations.append("Data shows class imbalance - consider techniques like SMOTE or class weights")
        if n_features > 50:
            recommendations.append("High-dimensional data - LightGBM or Random Forest may perform well")

        analysis['recommendations'] = recommendations

        return jsonify({
            'success': True,
            'message': 'Data analysis completed successfully',
            'data': analysis
        })

    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        return jsonify({'error': f'Data analysis failed: {str(e)}'}), 500


@model_manager_bp.route('/models/export-comparison', methods=['POST'])
@handle_errors
@rate_limit(max_requests=10, window_minutes=60)
@validate_input_fields('comparison_results', 'filepath')
def export_comparison_report():
    """
    Export model comparison results to file

    Expected JSON payload:
    {
        "comparison_results": [...],
        "filepath": "models/comparison_report.json"
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        comparison_results = data.get('comparison_results', [])
        filepath = data.get('filepath', f'models/comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

        if not comparison_results:
            raise ValidationError("comparison_results cannot be empty")

        # Convert to expected format if needed
        converted_results = []
        for result in comparison_results:
            if isinstance(result, dict):
                # Create mock object
                class MockResult:
                    def __init__(self, data):
                        self.model_name = data.get('model_name', '')
                        self.performance_metrics = type('MockMetrics', (), {
                            'f1_score': data.get('performance_metrics', {}).get('f1_score', 0),
                            'accuracy': data.get('performance_metrics', {}).get('accuracy', 0),
                            'precision': data.get('performance_metrics', {}).get('precision', 0),
                            'recall': data.get('performance_metrics', {}).get('recall', 0),
                            'cross_val_mean': data.get('performance_metrics', {}).get('cross_val_mean'),
                            'cross_val_std': data.get('performance_metrics', {}).get('cross_val_std'),
                            'training_time': data.get('performance_metrics', {}).get('training_time'),
                            'to_dict': lambda: data.get('performance_metrics', {})
                        })()
                        self.rank = data.get('rank', 0)
                        self.recommendation_score = data.get('recommendation_score', 0)
                        self.data_characteristics = data.get('data_characteristics', {})
                        self.timestamp = data.get('timestamp', '')

                    def to_dict(self):
                        return result

                converted_results.append(MockResult(result))

        # Export report
        success = model_manager.export_comparison_report(converted_results, filepath)

        if success:
            return jsonify({
                'success': True,
                'message': f'Comparison report exported successfully to {filepath}',
                'data': {
                    'filepath': filepath,
                    'export_timestamp': datetime.now().isoformat()
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to export comparison report'
            }), 500

    except Exception as e:
        logger.error(f"Error exporting comparison report: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


@model_manager_bp.route('/augmentation/stats', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
def get_augmentation_stats():
    """Get data augmentation statistics and metrics"""
    try:
        stats = model_manager.get_augmentation_stats()

        return jsonify({
            'success': True,
            'data': stats
        })

    except Exception as e:
        logger.error(f"Error getting augmentation stats: {str(e)}")
        return jsonify({'error': f'Failed to get augmentation stats: {str(e)}'}), 500


@model_manager_bp.route('/augmentation/augment-data', methods=['POST'])
@handle_errors
@rate_limit(max_requests=20, window_minutes=60)
@validate_input_fields('data')
def augment_training_data():
    """
    Augment training data using the data augmentation pipeline

    Expected JSON payload:
    {
        "data": [...],
        "data_type": "transaction",
        "augmentation_config": {
            "general": {
                "augmentation_ratio": 2.0
            }
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        training_data = data.get('data', [])
        data_type = data.get('data_type', 'transaction')
        augmentation_config = data.get('augmentation_config')

        if not training_data:
            raise ValidationError("Training data cannot be empty")

        # Augment data
        augmented_data, augmentation_report = model_manager.augment_training_data(
            transactions=training_data,
            data_type=data_type,
            augmentation_config=augmentation_config
        )

        return jsonify({
            'success': True,
            'message': f'Data augmentation completed. Generated {len(augmented_data)} samples',
            'data': {
                'original_size': len(training_data),
                'augmented_size': len(augmented_data),
                'augmented_data': augmented_data,
                'augmentation_report': augmentation_report
            }
        })

    except Exception as e:
        logger.error(f"Error augmenting training data: {str(e)}")
        return jsonify({'error': f'Data augmentation failed: {str(e)}'}), 500


@model_manager_bp.route('/augmentation/quality-report', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
def get_augmentation_quality_report():
    """Get comprehensive augmentation quality report"""
    try:
        # Get quality report from the augmentation pipeline
        quality_report = model_manager.data_augmentation_pipeline.quality_controller.generate_quality_report()

        return jsonify({
            'success': True,
            'data': quality_report
        })

    except Exception as e:
        logger.error(f"Error getting quality report: {str(e)}")
        return jsonify({'error': f'Failed to get quality report: {str(e)}'}), 500


@model_manager_bp.route('/models/train-with-balancing', methods=['POST'])
@handle_errors
@rate_limit(max_requests=5, window_minutes=60)
@validate_input_fields('model_type', 'data_source')
def train_model_with_balancing():
    """
    Train a machine learning model with integrated data balancing

    Expected JSON payload:
    {
        "model_type": "auto|kmeans|random_forest|xgboost|lightgbm|bert",
        "data_source": "transactions|company_financial",
        "balancing_method": "auto|smote|synthetic|hybrid",
        "target_balance_ratio": 1.0,
        "use_augmentation": true,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "category_filter": "optional_category"
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        model_type = data.get('model_type', 'auto')
        data_source = data.get('data_source', 'transactions')
        balancing_method = data.get('balancing_method', 'auto')
        target_balance_ratio = data.get('target_balance_ratio', 1.0)
        use_augmentation = data.get('use_augmentation', True)
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        category_filter = data.get('category_filter')

        # Validate inputs
        valid_models = ['auto', 'kmeans', 'random_forest', 'xgboost', 'lightgbm', 'bert']
        if model_type not in valid_models:
            raise ValidationError(f"Invalid model type. Must be one of: {', '.join(valid_models)}")

        valid_sources = ['transactions', 'company_financial']
        if data_source not in valid_sources:
            raise ValidationError(f"Invalid data source. Must be one of: {', '.join(valid_sources)}")

        # Fetch training data
        if data_source == 'transactions':
            query = Transaction.query
        else:
            query = CompanyFinancial.query

        # Apply filters
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date >= start_date_obj.date())

        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date <= end_date_obj.date())

        if category_filter:
            query = query.filter(Transaction.category == category_filter if data_source == 'transactions'
                               else CompanyFinancial.category == category_filter)

        # Get data
        raw_data = query.all()

        if len(raw_data) < 10:
            raise InsufficientDataError('model training', 10, len(raw_data))

        # Convert to dictionaries
        training_data = [item.to_dict() for item in raw_data]

        # Process data
        X, y, feature_names = model_manager.process_data(training_data)

        if len(X) == 0:
            raise InsufficientDataError('processed training data', 1, 0)

        # Apply balancing
        logger.info(f"Applying {balancing_method} balancing to {len(X)} samples")
        X_balanced, y_balanced = imbalanced_data_handler.handle_imbalanced_data(
            X, y, strategy=balancing_method, target_ratio=target_balance_ratio
        )

        # Extract texts for BERT if needed
        texts = None
        if model_type == 'bert' or (model_type == 'auto' and any('description' in item for item in training_data)):
            texts = [item.get('description', '') for item in training_data]

        # Train model on balanced data
        kwargs = {}
        if texts:
            kwargs['texts'] = texts

        result = model_manager.train_model(model_type, X_balanced, y_balanced, **kwargs)

        if result['success']:
            logger.info(f"Model {model_type} trained with balancing successfully on {len(X_balanced)} samples")

            return jsonify({
                'success': True,
                'message': f'Model {model_type} trained with balancing successfully',
                'data': {
                    'model_type': model_type,
                    'data_source': data_source,
                    'original_samples': len(X),
                    'balanced_samples': len(X_balanced),
                    'balancing_method': balancing_method,
                    'target_balance_ratio': target_balance_ratio,
                    'training_result': result
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Training failed')
            }), 500

    except Exception as e:
        logger.error(f"Error training model with balancing: {str(e)}")
        return jsonify({'error': f'Training with balancing failed: {str(e)}'}), 500


@model_manager_bp.route('/balancing/analyze-imbalance', methods=['POST'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
@validate_input_fields('data_source')
def analyze_imbalance():
    """
    Analyze imbalance in the dataset

    Expected JSON payload:
    {
        "data_source": "transactions|company_financial",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "category_filter": "optional_category"
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        data_source = data.get('data_source', 'transactions')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        category_filter = data.get('category_filter')

        # Fetch data
        if data_source == 'transactions':
            query = Transaction.query
        else:
            query = CompanyFinancial.query

        # Apply filters
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date >= start_date_obj.date())

        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date <= end_date_obj.date())

        if category_filter:
            query = query.filter(Transaction.category == category_filter if data_source == 'transactions'
                               else CompanyFinancial.category == category_filter)

        # Get data
        raw_data = query.all()

        if len(raw_data) < 10:
            raise InsufficientDataError('imbalance analysis', 10, len(raw_data))

        # Convert to features
        training_data = [item.to_dict() for item in raw_data]
        X, y, _ = model_manager.process_data(training_data)

        # Analyze imbalance
        imbalance_info = imbalanced_data_handler.smote_handler.detect_imbalance(X, y)

        # Get balancing recommendations
        recommendations = imbalanced_data_handler.get_balancing_recommendations(X, y)

        return jsonify({
            'success': True,
            'message': 'Imbalance analysis completed successfully',
            'data': {
                'data_source': data_source,
                'total_samples': len(raw_data),
                'processed_samples': len(X),
                'imbalance_analysis': imbalance_info,
                'recommendations': recommendations
            }
        })

    except Exception as e:
        logger.error(f"Error analyzing imbalance: {str(e)}")
        return jsonify({'error': f'Imbalance analysis failed: {str(e)}'}), 500


@model_manager_bp.route('/balancing/financial-balance', methods=['POST'])
@handle_errors
@rate_limit(max_requests=20, window_minutes=60)
@validate_input_fields('data_source')
def balance_financial_categories():
    """
    Balance financial categories with domain-specific methods

    Expected JSON payload:
    {
        "data_source": "transactions|company_financial",
        "target_category": "optional_specific_category",
        "method": "auto|smote|synthetic|pattern_based",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        data_source = data.get('data_source', 'transactions')
        target_category = data.get('target_category')
        method = data.get('method', 'auto')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        # Fetch data
        if data_source == 'transactions':
            query = Transaction.query
        else:
            query = CompanyFinancial.query

        # Apply filters
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(Transaction.date >= start_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date >= start_date_obj.date())

        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(Transaction.date <= end_date_obj.date() if data_source == 'transactions'
                               else CompanyFinancial.date <= end_date_obj.date())

        # Get data
        raw_data = query.all()

        if len(raw_data) < 10:
            raise InsufficientDataError('financial balancing', 10, len(raw_data))

        # Convert to dictionaries
        transaction_data = [item.to_dict() for item in raw_data]

        # Apply financial category balancing
        balanced_data = financial_category_balancer.balance_financial_categories(
            transaction_data, target_category=target_category, method=method
        )

        return jsonify({
            'success': True,
            'message': 'Financial category balancing completed successfully',
            'data': {
                'data_source': data_source,
                'original_transactions': len(transaction_data),
                'balanced_transactions': len(balanced_data),
                'target_category': target_category,
                'method': method,
                'synthetic_transactions': len(balanced_data) - len(transaction_data),
                'balanced_data': balanced_data
            }
        })

    except Exception as e:
        logger.error(f"Error balancing financial categories: {str(e)}")
        return jsonify({'error': f'Financial balancing failed: {str(e)}'}), 500


@model_manager_bp.route('/balancing/quality-assessment', methods=['POST'])
@handle_errors
@rate_limit(max_requests=20, window_minutes=60)
@validate_input_fields('original_data', 'synthetic_data')
def assess_synthetic_quality():
    """
    Assess quality of synthetic data

    Expected JSON payload:
    {
        "original_data": [...],
        "synthetic_data": [...],
        "metadata": {
            "generation_method": "smote|gan|vae",
            "target_variable": "category"
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No data provided")

        original_data = data.get('original_data', [])
        synthetic_data = data.get('synthetic_data', [])
        metadata = data.get('metadata', {})

        if not original_data or not synthetic_data:
            raise ValidationError("Both original_data and synthetic_data are required")

        # Assess quality
        quality_report = quality_assessment_engine.assess_synthetic_quality(
            original_data, synthetic_data, metadata
        )

        return jsonify({
            'success': True,
            'message': 'Synthetic data quality assessment completed',
            'data': quality_report
        })

    except Exception as e:
        logger.error(f"Error assessing synthetic quality: {str(e)}")
        return jsonify({'error': f'Quality assessment failed: {str(e)}'}), 500


@model_manager_bp.route('/balancing/history', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
def get_balancing_history():
    """Get history of balancing operations"""
    try:
        history = imbalanced_data_handler.get_balancing_history()

        return jsonify({
            'success': True,
            'data': {
                'total_operations': len(history),
                'history': history,
                'performance_metrics': imbalanced_data_handler.get_performance_metrics()
            }
        })

    except Exception as e:
        logger.error(f"Error getting balancing history: {str(e)}")
        return jsonify({'error': f'Failed to get balancing history: {str(e)}'}), 500


@model_manager_bp.route('/balancing/quality-history', methods=['GET'])
@handle_errors
@rate_limit(max_requests=30, window_minutes=60)
def get_quality_history():
    """Get history of quality assessments"""
    try:
        history = quality_assessment_engine.get_quality_history()
        summary = quality_assessment_engine.get_quality_summary()

        return jsonify({
            'success': True,
            'data': {
                'total_assessments': len(history),
                'summary': summary,
                'recent_assessments': history[-10:]  # Last 10 assessments
            }
        })

    except Exception as e:
        logger.error(f"Error getting quality history: {str(e)}")
        return jsonify({'error': f'Failed to get quality history: {str(e)}'}), 500


@model_manager_bp.route('/balancing/methods-info', methods=['GET'])
@handle_errors
@rate_limit(max_requests=50, window_minutes=60)
def get_balancing_methods_info():
    """Get information about available balancing methods"""
    try:
        smote_info = imbalanced_data_handler.smote_handler.get_method_info()
        financial_info = financial_category_balancer.get_category_statistics()

        return jsonify({
            'success': True,
            'data': {
                'smote_methods': smote_info,
                'financial_categories': financial_info,
                'available_strategies': ['auto', 'smote', 'synthetic', 'hybrid', 'undersample'],
                'financial_methods': ['auto', 'smote', 'synthetic', 'pattern_based']
            }
        })

    except Exception as e:
        logger.error(f"Error getting balancing methods info: {str(e)}")
        return jsonify({'error': f'Failed to get balancing methods info: {str(e)}'}), 500