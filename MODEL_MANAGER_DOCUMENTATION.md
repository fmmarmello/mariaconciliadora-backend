# ModelManager Documentation

## Overview

The ModelManager is a comprehensive machine learning orchestration system that provides unified training, prediction, and management capabilities for multiple ML algorithms including KMeans, Random Forest, XGBoost, LightGBM, and BERT. The system now includes an advanced ModelSelector component that provides intelligent model selection, comparison, benchmarking, and recommendation capabilities.

## Architecture

### Core Components

1. **BaseModel**: Abstract base class defining the interface for all ML models
2. **ModelManager**: Main orchestration class managing model lifecycle
3. **ModelSelector**: Advanced model selection and comparison system
4. **Specific Model Implementations**: Concrete implementations for each algorithm
5. **FeatureEngineer**: Advanced feature engineering pipeline
6. **Model Persistence**: Versioned model storage and retrieval

### Supported Algorithms

- **KMeans**: Unsupervised clustering
- **Random Forest**: Ensemble classification
- **XGBoost**: Gradient boosting classification
- **LightGBM**: Light gradient boosting classification
- **BERT**: Transformer-based text classification

### Advanced Features

- **Intelligent Model Selection**: Automatic model selection based on data characteristics
- **Comprehensive Model Comparison**: Detailed benchmarking with cross-validation
- **Performance Tracking**: Historical performance analysis and trend identification
- **A/B Testing**: Statistical comparison between models
- **Data Analysis**: Automatic data characteristics analysis for model recommendations
- **Performance Reporting**: Comprehensive performance reports and visualizations

## API Endpoints

### Training Endpoints

#### POST `/api/models/train`
Train a machine learning model with specified configuration.

**Request Body:**
```json
{
  "model_type": "auto|kmeans|random_forest|xgboost|lightgbm|bert",
  "data_source": "transactions|company_financial",
  "optimize": true,
  "n_trials": 50,
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "category_filter": "optional_category"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Model random_forest trained successfully",
  "data": {
    "model_type": "random_forest",
    "data_source": "transactions",
    "training_samples": 1000,
    "feature_count": 422,
    "training_result": {
      "success": true,
      "n_estimators": 100,
      "feature_importances": [0.1, 0.2, ...]
    }
  }
}
```

### Prediction Endpoints

#### POST `/api/models/predict`
Make predictions using a trained model.

**Request Body:**
```json
{
  "model_type": "random_forest",
  "data": {
    "description": "Compra no mercado",
    "amount": 150.50,
    "date": "2024-01-15",
    "category": "alimentacao"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "model_type": "random_forest",
    "prediction": "alimentacao",
    "probabilities": [0.1, 0.8, 0.1],
    "confidence": 0.8
  }
}
```

#### POST `/api/models/batch-predict`
Make batch predictions for multiple items.

**Request Body:**
```json
{
  "model_type": "random_forest",
  "data": [
    {
      "description": "Compra no mercado",
      "amount": 150.50,
      "date": "2024-01-15"
    },
    {
      "description": "Pagamento de conta",
      "amount": -200.00,
      "date": "2024-01-16"
    }
  ]
}
```

### Model Management Endpoints

#### POST `/api/models/compare`
Compare performance of multiple models.

**Request Body:**
```json
{
  "data_source": "transactions",
  "models_to_compare": ["random_forest", "xgboost", "lightgbm"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

#### POST `/api/models/optimize`
Optimize hyperparameters for a specific model.

**Request Body:**
```json
{
  "model_type": "random_forest",
  "data_source": "transactions",
  "n_trials": 50,
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

#### POST `/api/models/select`
Automatically select the best model for given data.

**Request Body:**
```json
{
  "data_source": "transactions",
  "candidate_models": ["random_forest", "xgboost", "lightgbm", "bert"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

### Advanced Model Selection and Comparison Endpoints

#### POST `/api/models/advanced-select`
Advanced model selection with comprehensive analysis and recommendations.

**Request Body:**
```json
{
  "data_source": "transactions|company_financial",
  "candidate_models": ["random_forest", "xgboost", "lightgbm", "bert"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "include_data_analysis": true,
  "generate_report": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Advanced model selection completed successfully",
  "data": {
    "best_model": "xgboost",
    "recommendation": {
      "recommended_model": "xgboost",
      "confidence_level": "high",
      "performance_score": 0.87,
      "key_advantages": ["High overall performance", "Stable cross-validation"],
      "considerations": [],
      "alternative_models": [...]
    },
    "comparison_results": [...],
    "data_analysis": {...},
    "performance_report": {...}
  }
}
```

#### POST `/api/models/ab-test`
Perform A/B testing between two models.

**Request Body:**
```json
{
  "model_a": "random_forest",
  "model_b": "xgboost",
  "data_source": "transactions",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "test_duration_hours": 24
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "test_id": "ab_test_random_forest_vs_xgboost_20240101_120000",
    "winner": "xgboost",
    "confidence_level": 0.85,
    "statistical_significance": true,
    "performance_difference": {
      "f1_diff": 0.05,
      "accuracy_diff": 0.03
    },
    "model_a_results": {...},
    "model_b_results": {...}
  }
}
```

#### GET `/api/models/performance-history`
Get historical performance data.

**Query Parameters:**
- `model_name`: Specific model name (optional)
- `days`: Number of days to look back (default: 30)

#### GET `/api/models/ab-tests`
Get A/B test results.

**Query Parameters:**
- `test_id`: Specific test ID (optional)

#### POST `/api/models/data-analysis`
Analyze data characteristics for model selection guidance.

**Request Body:**
```json
{
  "data_source": "transactions",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

#### POST `/api/models/export-comparison`
Export model comparison results to file.

**Request Body:**
```json
{
  "comparison_results": [...],
  "filepath": "models/comparison_report.json"
}
```

#### GET `/api/models/info`
Get information about all available models.

#### GET `/api/models/{model_type}/info`
Get detailed information about a specific model.

#### POST `/api/models/{model_type}/evaluate`
Evaluate a specific model's performance.

## Configuration

### Default Configuration

```python
{
  'model_dir': 'models/',
  'cache_dir': 'cache/',
  'fallback_models': ['random_forest', 'xgboost', 'lightgbm'],
  'model_configs': {
    'random_forest': {
      'n_estimators': 100,
      'max_depth': 10,
      'min_samples_split': 5,
      'min_samples_leaf': 2
    },
    'xgboost': {
      'n_estimators': 100,
      'max_depth': 6,
      'learning_rate': 0.1,
      'subsample': 0.8,
      'colsample_bytree': 0.8
    },
    'lightgbm': {
      'n_estimators': 100,
      'max_depth': 6,
      'learning_rate': 0.1,
      'subsample': 0.8,
      'colsample_bytree': 0.8
    },
    'kmeans': {
      'n_clusters': 10
    },
    'bert': {
      'model_name': 'neuralmind/bert-base-portuguese-cased',
      'max_length': 128,
      'batch_size': 16,
      'learning_rate': 2e-5,
      'num_epochs': 5
    }
  }
}
```

## Error Handling

### Comprehensive Error Types

1. **ValidationError**: Invalid input parameters
2. **InsufficientDataError**: Not enough data for operation
3. **ModelNotFoundError**: Requested model not available
4. **TrainingError**: Model training failures
5. **PredictionError**: Prediction failures
6. **PersistenceError**: Model save/load failures

### Fallback Mechanisms

1. **Model Fallback Chain**: Automatic fallback to alternative models
2. **Graceful Degradation**: Continue operation with reduced functionality
3. **Default Predictions**: Provide sensible defaults when models fail
4. **Error Recovery**: Automatic retry with exponential backoff

### Logging

All operations are comprehensively logged with:
- Operation timestamps
- Input parameters (sanitized)
- Performance metrics
- Error details with stack traces
- Audit trails for compliance

## Usage Examples

### Python API Usage

```python
from src.services.model_manager import ModelManager

# Initialize ModelManager
model_manager = ModelManager()

# Process data
transactions = [
    {
        'description': 'Compra no mercado',
        'amount': 150.50,
        'date': '2024-01-15',
        'category': 'alimentacao'
    }
]

X, y, feature_names = model_manager.process_data(transactions)

# Train model
result = model_manager.train_model('random_forest', X, y)
print(f"Training result: {result}")

# Make predictions
predictions = model_manager.predict('random_forest', X)
print(f"Predictions: {predictions}")

# Compare models
comparison = model_manager.compare_models(X, y, ['random_forest', 'xgboost'])
print(f"Best model: {comparison['best_model']}")
```

### Automatic Model Selection

```python
# Let the system choose the best model
best_model = model_manager.select_best_model(X, y)
print(f"Automatically selected: {best_model}")

# Train with optimization
result = model_manager.train_model(
    best_model, X, y,
    optimize=True,
    n_trials=30
)
```

### Model Persistence

```python
# Save trained model
model_manager.save_model('random_forest')

# Load model later
model_manager.load_model('random_forest')

# Get model information
info = model_manager.get_model_info('random_forest')
print(f"Model info: {info}")
```

## ModelSelector: Advanced Model Selection and Comparison

The ModelSelector provides intelligent model selection, comprehensive comparison, benchmarking, and recommendation capabilities. It analyzes data characteristics, performs cross-validation, and provides detailed performance analysis.

### Key Features

- **Data Characteristics Analysis**: Automatically analyzes dataset properties to inform model selection
- **Intelligent Recommendations**: Recommends models based on data size, complexity, and characteristics
- **Comprehensive Benchmarking**: Detailed performance comparison with cross-validation
- **A/B Testing**: Statistical comparison between models with confidence intervals
- **Historical Tracking**: Performance history and trend analysis
- **Performance Reporting**: Comprehensive reports with recommendations

### Usage Examples

#### Advanced Model Selection

```python
from src.services.model_selector import ModelSelector

# Initialize ModelSelector
model_selector = ModelSelector(model_manager)

# Analyze data characteristics
data_analysis = model_selector.analyze_data_characteristics(X, y, feature_names)
print(f"Data analysis: {data_analysis}")

# Advanced model selection with recommendations
selection_result = model_selector.select_best_model(
    X, y,
    candidate_models=['random_forest', 'xgboost', 'lightgbm', 'bert'],
    data_characteristics=data_analysis
)

print(f"Best model: {selection_result['best_model']}")
print(f"Recommendation: {selection_result['recommendation']}")
```

#### Comprehensive Model Comparison

```python
# Compare multiple models with detailed benchmarking
comparison_results = model_selector.compare_models_comprehensive(
    X, y,
    models_to_compare=['random_forest', 'xgboost', 'lightgbm'],
    data_characteristics=data_analysis
)

# Generate performance report
report = model_selector.generate_performance_report(
    comparison_results, data_analysis
)

print(f"Best performer: {report['summary']['best_model']}")
print(f"Performance analysis: {report['performance_analysis']}")
```

#### A/B Testing

```python
# Perform A/B testing between models
ab_result = model_selector.perform_ab_test(
    'random_forest', 'xgboost', X, y, test_duration_hours=24
)

print(f"A/B Test Winner: {ab_result['winner']}")
print(f"Confidence Level: {ab_result['confidence_level']}")
print(f"Statistical Significance: {ab_result['statistical_significance']}")
```

#### Performance History and Trends

```python
# Get performance history
history = model_selector.get_performance_history(days=30)
print(f"Total comparisons: {history['total_entries']}")

# Get history for specific model
rf_history = model_selector.get_performance_history('random_forest', days=30)
print(f"Random Forest performance: {rf_history['average_performance']}")

# Analyze model distribution trends
trends = history.get('performance_trends', {})
print(f"Model preferences over time: {trends.get('model_preference_trends', [])}")
```

#### Data Analysis for Model Guidance

```python
# Analyze data to get model recommendations
data_analysis = model_selector.analyze_data_characteristics(X, y)

# Get model recommendations based on data
recommendations = []
if data_analysis['n_samples'] < 1000:
    recommendations.append("Consider simpler models for small dataset")
if data_analysis['data_complexity']['class_imbalance_ratio'] > 3.0:
    recommendations.append("Consider models that handle imbalanced data well")

print(f"Data-based recommendations: {recommendations}")
```

### Model Selection Criteria

The ModelSelector uses a weighted scoring system for model selection:

```python
selection_criteria = {
    'primary_metric': 'f1_score',
    'secondary_metrics': ['accuracy', 'precision', 'recall'],
    'weights': {
        'f1_score': 0.4,
        'accuracy': 0.3,
        'precision': 0.15,
        'recall': 0.15
    }
}
```

### Data Characteristics Analysis

The system analyzes various data properties:

- **Dataset Size**: Number of samples and features
- **Class Distribution**: Balance/imbalance analysis
- **Feature Complexity**: Dimensionality and sparsity
- **Data Complexity**: Relationships and patterns

### Performance Metrics Tracked

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Cross-Validation**: Mean scores and standard deviation
- **Training Performance**: Time and resource usage
- **Prediction Performance**: Latency and throughput
- **Stability Analysis**: Cross-validation consistency

### A/B Testing Framework

The A/B testing provides:

- **Statistical Significance**: Confidence intervals and p-values
- **Performance Differences**: Detailed metric comparisons
- **Sample Size Analysis**: Statistical power calculations
- **Test Duration Tracking**: Time-based performance analysis

### Historical Performance Tracking

The system maintains:

- **Performance History**: All model comparisons and selections
- **Trend Analysis**: Performance changes over time
- **Model Distribution**: Usage patterns and preferences
- **Benchmarking Data**: Comparative performance data

### Export and Reporting

```python
# Export comparison results
success = model_selector.export_comparison_report(
    comparison_results, 'models/detailed_comparison.json'
)

# Generate comprehensive performance report
report = model_selector.generate_performance_report(
    comparison_results, data_characteristics
)

# Access report sections
print(f"Summary: {report['summary']}")
print(f"Rankings: {report['model_rankings']}")
print(f"Analysis: {report['performance_analysis']}")
print(f"Recommendations: {report['recommendations']}")
```

## Performance Optimization

### Hyperparameter Optimization

The system supports automatic hyperparameter optimization using Optuna:

```python
# Optimize model hyperparameters
optimization_result = model_manager.optimize_hyperparameters(
    'random_forest', X, y, n_trials=50
)
print(f"Best parameters: {optimization_result['best_params']}")
print(f"Best score: {optimization_result['best_score']}")
```

### Cross-Validation

All models support cross-validation for robust performance evaluation:

```python
# Evaluate with cross-validation
evaluation = model_manager.evaluate_model(
    'random_forest', X, y, cv_folds=5
)
print(f"Cross-validation F1: {evaluation['f1_score']}")
```

## Feature Engineering

### Comprehensive Feature Pipeline

The system includes advanced feature engineering:

1. **Text Embeddings**: Sentence transformers for text data
2. **Temporal Features**: Date/time-based features
3. **Transaction Patterns**: Rolling statistics and patterns
4. **Categorical Encoding**: Target encoding and one-hot encoding
5. **Feature Scaling**: Standardization and normalization
6. **Feature Selection**: Mutual information and model-based selection
7. **Dimensionality Reduction**: PCA, t-SNE, UMAP

### Custom Feature Engineering

```python
from src.services.feature_engineer import FeatureEngineer

# Initialize feature engineer
feature_engineer = FeatureEngineer()

# Create comprehensive features
X, feature_names = feature_engineer.create_comprehensive_features(
    transactions, target_column='category'
)
```

## Monitoring and Metrics

### Performance Tracking

- Training time and resource usage
- Model accuracy, precision, recall, F1-score
- Cross-validation performance
- Feature importance rankings
- Prediction latency and throughput

### Health Checks

```python
# Get system health
health = model_manager.get_system_health()
print(f"System health: {health}")

# Get model performance history
history = model_manager.get_performance_history()
print(f"Performance history: {history}")
```

## Security Considerations

### Input Validation

- All inputs are validated and sanitized
- File uploads are scanned for security
- Rate limiting prevents abuse
- SQL injection protection

### Model Security

- Models are validated before deployment
- Input data is sanitized before processing
- Secure model storage with access controls
- Audit logging for all operations

## Troubleshooting

### Common Issues

1. **NaN Values in Features**
   - Solution: Enable NaN handling in feature engineering
   - Check data quality and preprocessing

2. **Model Training Failures**
   - Solution: Check data size and quality
   - Verify model configuration
   - Enable fallback mechanisms

3. **Memory Issues**
   - Solution: Reduce batch sizes
   - Use feature selection to reduce dimensionality
   - Enable model compression

4. **Slow Predictions**
   - Solution: Use model optimization
   - Enable caching for repeated predictions
   - Consider model quantization

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger('src.services.model_manager').setLevel(logging.DEBUG)
```

## Integration with Existing Systems

### Database Integration

The ModelManager seamlessly integrates with the existing database:

- Transaction data from `Transaction` model
- Company financial data from `CompanyFinancial` model
- Upload history tracking
- Audit logging

### API Integration

New endpoints are registered alongside existing ones:

- `/api/models/*` - Model management endpoints
- Compatible with existing authentication and validation
- Consistent error response format

### Frontend Integration

The API endpoints are designed for easy frontend integration:

```javascript
// Train a model
const response = await fetch('/api/models/train', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model_type: 'auto',
    data_source: 'transactions'
  })
});

const result = await response.json();
```

## Future Enhancements

### Planned Features

1. **Model Ensembling**: Combine multiple models for better performance
2. **Online Learning**: Support for streaming data and model updates
3. **Model Explainability**: SHAP values and feature importance visualization
4. **AutoML**: Fully automated model selection and hyperparameter tuning
5. **Model Compression**: Quantization and pruning for deployment
6. **Multi-language Support**: Extend beyond Portuguese text processing

### Extensibility

The modular architecture allows easy addition of new models:

```python
class NewModel(BaseModel):
    def train(self, X, y, **kwargs):
        # Implement training logic
        pass

    def predict(self, X):
        # Implement prediction logic
        pass

# Register new model
model_manager.register_model('new_model', NewModel)
```

## Conclusion

The ModelManager and ModelSelector provide a comprehensive, production-ready machine learning orchestration system with:

### Core Features
- ✅ Unified interface for multiple algorithms (KMeans, Random Forest, XGBoost, LightGBM, BERT)
- ✅ Automatic model selection and optimization
- ✅ Robust error handling and fallback mechanisms
- ✅ Comprehensive logging and monitoring
- ✅ Secure and scalable architecture
- ✅ Easy integration with existing systems

### Advanced ModelSelector Capabilities
- ✅ **Intelligent Model Selection**: Data-driven model recommendations based on characteristics
- ✅ **Comprehensive Benchmarking**: Detailed performance comparison with cross-validation
- ✅ **A/B Testing Framework**: Statistical model comparison with confidence intervals
- ✅ **Historical Performance Tracking**: Trend analysis and performance history
- ✅ **Data Analysis**: Automatic dataset characteristics analysis
- ✅ **Performance Reporting**: Comprehensive reports with actionable recommendations
- ✅ **Export Functionality**: Save comparison results and reports to files

### Key Benefits
- **Improved Model Performance**: Intelligent selection leads to better-performing models
- **Reduced Time-to-Production**: Automated selection and comparison workflows
- **Data-Driven Decisions**: Evidence-based model selection and recommendations
- **Scalable Architecture**: Handles multiple models and large datasets efficiently
- **Production-Ready**: Comprehensive error handling and monitoring
- **Extensible Design**: Easy to add new models and evaluation metrics

This enhanced system significantly improves the ML capabilities of the Maria Conciliadora application, providing state-of-the-art model selection, comparison, and recommendation services that adapt to different data characteristics and use cases.