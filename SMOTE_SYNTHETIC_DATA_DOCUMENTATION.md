# SMOTE and Synthetic Data Generation Documentation

## Overview

This document provides comprehensive documentation for the SMOTE (Synthetic Minority Oversampling Technique) and synthetic data generation system implemented in Maria Conciliadora. The system provides advanced capabilities for handling imbalanced datasets through multiple techniques including SMOTE variants, generative models, and domain-specific financial balancing.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [SMOTE Implementation](#smote-implementation)
4. [Synthetic Data Generation](#synthetic-data-generation)
5. [Imbalanced Data Handler](#imbalanced-data-handler)
6. [Financial Category Balancer](#financial-category-balancer)
7. [Quality Assessment Engine](#quality-assessment-engine)
8. [API Endpoints](#api-endpoints)
9. [Usage Examples](#usage-examples)
10. [Testing](#testing)
11. [Performance Considerations](#performance-considerations)
12. [Troubleshooting](#troubleshooting)

## Architecture Overview

The SMOTE and synthetic data generation system consists of several interconnected components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Layer     │    │  Business Logic  │    │  Data Access    │
│                 │    │                  │    │                 │
│ • REST Endpoints│    │ • ImbalancedData │    │ • Transactions  │
│ • Request/Resp  │    │   Handler        │    │ • Company Data  │
│ • Validation    │    │ • Quality Engine │    │ • File Uploads  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │   Core Services     │
                    │                     │
                    │ • SMOTEImplementation│
                    │ • SyntheticDataGen  │
                    │ • FinancialBalancer │
                    │ • QualityAssessment │
                    └─────────────────────┘
```

## Core Components

### 1. SMOTEImplementation (`src/services/smote_implementation.py`)

Advanced SMOTE implementation with multiple variants for handling imbalanced datasets.

**Features:**
- Classic SMOTE for oversampling minority classes
- BorderlineSMOTE for samples near decision boundaries
- ADASYN for adaptive synthetic sampling
- SVMSMOTE for support vector machine-based sampling
- KMeansSMOTE for cluster-based sampling
- Hybrid approaches combining SMOTE with undersampling

**Key Methods:**
```python
# Initialize SMOTE handler
smote_handler = SMOTEImplementation(config)

# Apply SMOTE variant
X_balanced, y_balanced = smote_handler.apply_smote(X, y, method='borderline')

# Compare different methods
results = smote_handler.compare_smote_methods(X, y, methods=['classic', 'adasyn'])

# Get method information
info = smote_handler.get_method_info()
```

### 2. AdvancedSyntheticDataGenerator (`src/services/synthetic_data_generator.py`)

Enhanced synthetic data generator with multiple generative models.

**Features:**
- GAN-based generation for complex distributions
- Conditional GAN (CGAN) for class-specific generation
- Variational Autoencoder (VAE) for feature learning
- Diffusion models for high-quality synthetic data
- Quality assessment and validation

**Key Methods:**
```python
# Initialize generator
generator = AdvancedSyntheticDataGenerator(config)

# Train models
generator.train_models(data, epochs=100)

# Generate synthetic data
synthetic_data = generator.generate_synthetic_data(original_data)

# Assess quality
quality_report = generator.get_quality_metrics()
```

### 3. ImbalancedDataHandler (`src/services/imbalanced_data_handler.py`)

Intelligent orchestrator for handling imbalanced datasets.

**Features:**
- Automatic imbalance detection and severity assessment
- Strategy selection based on dataset characteristics
- Hybrid approaches combining SMOTE with undersampling
- Performance monitoring and adjustment
- Integration with existing training pipelines

**Key Methods:**
```python
# Initialize handler
handler = ImbalancedDataHandler(config)

# Handle imbalanced data
X_balanced, y_balanced = handler.handle_imbalanced_data(X, y, strategy='auto')

# Get balancing recommendations
recommendations = handler.get_balancing_recommendations(X, y)

# Get performance history
history = handler.get_balancing_history()
```

### 4. FinancialCategoryBalancer (`src/services/financial_category_balancer.py`)

Domain-specific balancer for financial transaction categories.

**Features:**
- Category-specific SMOTE parameters
- Financial transaction pattern preservation
- Amount distribution maintenance
- Temporal pattern preservation
- Business rule compliance in synthetic data

**Key Methods:**
```python
# Initialize balancer
balancer = FinancialCategoryBalancer(config)

# Balance financial categories
balanced_data = balancer.balance_financial_categories(transactions, method='auto')

# Balance specific category
balanced_data = balancer.balance_financial_categories(
    transactions, target_category='investimento', method='pattern_based'
)

# Get category statistics
stats = balancer.get_category_statistics()
```

### 5. QualityAssessmentEngine (`src/services/quality_assessment_engine.py`)

Comprehensive quality validation for synthetic data.

**Features:**
- Statistical similarity checks between original and synthetic data
- Distribution preservation validation
- Business rule compliance verification
- Model performance impact assessment
- Synthetic data quality scoring

**Key Methods:**
```python
# Initialize quality engine
quality_engine = QualityAssessmentEngine(config)

# Assess synthetic quality
report = quality_engine.assess_synthetic_quality(original_data, synthetic_data)

# Get quality history
history = quality_engine.get_quality_history()

# Generate quality summary
summary = quality_engine.get_quality_summary()
```

## API Endpoints

### Training with Balancing

**Endpoint:** `POST /api/model_manager/models/train-with-balancing`

**Request:**
```json
{
    "model_type": "random_forest",
    "data_source": "transactions",
    "balancing_method": "auto",
    "target_balance_ratio": 1.0,
    "use_augmentation": true,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
}
```

**Response:**
```json
{
    "success": true,
    "message": "Model trained with balancing successfully",
    "data": {
        "model_type": "random_forest",
        "original_samples": 1000,
        "balanced_samples": 1200,
        "balancing_method": "smote",
        "training_result": {...}
    }
}
```

### Imbalance Analysis

**Endpoint:** `POST /api/model_manager/balancing/analyze-imbalance`

**Request:**
```json
{
    "data_source": "transactions",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "imbalance_analysis": {
            "imbalance_ratio": 3.5,
            "severity": "moderate",
            "requires_balancing": true
        },
        "recommendations": [...]
    }
}
```

### Financial Category Balancing

**Endpoint:** `POST /api/model_manager/balancing/financial-balance`

**Request:**
```json
{
    "data_source": "transactions",
    "target_category": "investimento",
    "method": "pattern_based"
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "original_transactions": 1000,
        "balanced_transactions": 1100,
        "synthetic_transactions": 100,
        "balanced_data": [...]
    }
}
```

### Quality Assessment

**Endpoint:** `POST /api/model_manager/balancing/quality-assessment`

**Request:**
```json
{
    "original_data": [...],
    "synthetic_data": [...],
    "metadata": {
        "generation_method": "smote",
        "target_variable": "category"
    }
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "overall_quality_score": 0.85,
        "quality_grade": "B",
        "recommendations": [...],
        "statistical_similarity": {...},
        "distribution_preservation": {...}
    }
}
```

## Usage Examples

### Basic SMOTE Usage

```python
from src.services.smote_implementation import SMOTEImplementation

# Initialize
config = {'random_state': 42}
smote_handler = SMOTEImplementation(config)

# Load imbalanced data
X, y = load_your_data()

# Apply SMOTE
X_balanced, y_balanced = smote_handler.apply_smote(X, y, method='borderline')

# Check results
print(f"Original shape: {X.shape}, Balanced shape: {X_balanced.shape}")
```

### Advanced Synthetic Generation

```python
from src.services.synthetic_data_generator import AdvancedSyntheticDataGenerator

# Initialize
config = {'methods': ['vae', 'gan'], 'sample_size_ratio': 0.5}
generator = AdvancedSyntheticDataGenerator(config)

# Train on your data
generator.train_models(your_data, epochs=100)

# Generate synthetic data
synthetic_data = generator.generate_synthetic_data(your_data)

print(f"Generated {len(synthetic_data)} synthetic samples")
```

### Intelligent Imbalance Handling

```python
from src.services.imbalanced_data_handler import ImbalancedDataHandler

# Initialize
config = {
    'smote_config': {'random_state': 42},
    'synthetic_config': {'methods': ['vae']}
}
handler = ImbalancedDataHandler(config)

# Handle imbalanced data automatically
X_balanced, y_balanced = handler.handle_imbalanced_data(X, y, strategy='auto')

# Get recommendations for future use
recommendations = handler.get_balancing_recommendations(X, y)
```

### Financial Category Balancing

```python
from src.services.financial_category_balancer import FinancialCategoryBalancer

# Initialize
config = {'smote_config': {'random_state': 42}}
balancer = FinancialCategoryBalancer(config)

# Load transaction data
transactions = load_transaction_data()

# Balance categories
balanced_transactions = balancer.balance_financial_categories(
    transactions, method='auto'
)

print(f"Added {len(balanced_transactions) - len(transactions)} synthetic transactions")
```

### Quality Assessment

```python
from src.services.quality_assessment_engine import QualityAssessmentEngine

# Initialize
quality_engine = QualityAssessmentEngine({})

# Assess synthetic data quality
report = quality_engine.assess_synthetic_quality(original_data, synthetic_data)

print(f"Quality Score: {report['overall_quality_score']:.3f}")
print(f"Grade: {report['quality_grade']}")
print("Recommendations:")
for rec in report['recommendations']:
    print(f"- {rec}")
```

## Testing

### Running Unit Tests

```bash
# Run all SMOTE-related tests
pytest tests/unit/test_smote_implementation.py -v

# Run synthetic data generator tests
pytest tests/unit/test_synthetic_data_generator.py -v

# Run imbalance handler tests
pytest tests/unit/test_imbalanced_data_handler.py -v

# Run financial balancer tests
pytest tests/unit/test_financial_category_balancer.py -v

# Run quality assessment tests
pytest tests/unit/test_quality_assessment_engine.py -v
```

### Running Integration Tests

```bash
# Run API integration tests
pytest tests/integration/test_imbalance_api_endpoints.py -v

# Run with coverage
pytest tests/ --cov=src.services --cov-report=html
```

### Test Coverage

The test suite covers:
- Unit tests for all core components
- Integration tests for API endpoints
- Error handling and edge cases
- Performance and scalability tests
- Data validation and business rule compliance

## Performance Considerations

### Memory Usage

- SMOTE variants: O(n*m) where n is samples, m is features
- GAN training: High memory usage, consider GPU acceleration
- Diffusion models: Very high memory requirements
- Large datasets: Consider batch processing and sampling

### Computational Complexity

- SMOTE: O(k*n^2) for k nearest neighbors
- GAN training: O(epochs * batch_size * model_complexity)
- Quality assessment: O(n*m) for statistical tests

### Optimization Strategies

1. **Data Sampling**: Use representative samples for training
2. **Batch Processing**: Process large datasets in batches
3. **GPU Acceleration**: Use CUDA for deep learning models
4. **Caching**: Cache trained models for reuse
5. **Parallel Processing**: Use multiple cores for independent operations

## Troubleshooting

### Common Issues

#### 1. Memory Errors
```
Error: CUDA out of memory
Solution: Reduce batch size, use CPU, or sample smaller datasets
```

#### 2. Poor Synthetic Quality
```
Issue: Low quality scores from synthetic data
Solutions:
- Increase training epochs
- Adjust model hyperparameters
- Use different generation methods
- Validate input data quality
```

#### 3. Imbalance Not Resolved
```
Issue: Balancing doesn't improve performance
Solutions:
- Try different SMOTE variants
- Use hybrid approaches
- Check data quality and preprocessing
- Consider domain-specific methods
```

#### 4. API Timeouts
```
Issue: Long-running requests timeout
Solutions:
- Use asynchronous processing
- Implement progress tracking
- Break large operations into smaller tasks
- Optimize database queries
```

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug for specific components
logger = logging.getLogger('src.services.smote_implementation')
logger.setLevel(logging.DEBUG)
```

### Performance Monitoring

Monitor system performance:

```python
# Track operation timing
import time
start_time = time.time()

# Your operation here
result = handler.handle_imbalanced_data(X, y)

end_time = time.time()
print(f"Operation took {end_time - start_time:.2f} seconds")
```

## Configuration

### Default Configuration

```python
default_config = {
    'smote': {
        'random_state': 42,
        'k_neighbors': 5,
        'sampling_strategy': 'auto'
    },
    'synthetic': {
        'methods': ['vae', 'gan'],
        'sample_size_ratio': 0.5,
        'epochs': 100,
        'batch_size': 32
    },
    'quality': {
        'statistical_tests': ['ks_test', 'mann_whitney', 'js_divergence'],
        'quality_thresholds': {
            'statistical_similarity': 0.8,
            'distribution_preservation': 0.75,
            'business_rule_compliance': 0.9
        }
    }
}
```

### Environment Variables

```bash
# GPU settings
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Memory settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Logging
export LOG_LEVEL=INFO
export LOG_FILE=/var/log/maria_conciliadora.log
```

## Integration with Existing Systems

### Model Training Pipeline Integration

```python
# Integrate with existing training
def train_with_balancing(model_type, X, y, balancing_config=None):
    if balancing_config:
        handler = ImbalancedDataHandler(balancing_config)
        X, y = handler.handle_imbalanced_data(X, y, strategy='auto')

    # Continue with normal training
    model = train_model(model_type, X, y)
    return model
```

### Data Processing Pipeline Integration

```python
# Add balancing to data processing
def process_data_with_balancing(raw_data, config=None):
    # Preprocess data
    X, y = preprocess_data(raw_data)

    # Apply balancing if needed
    if config and config.get('use_balancing', False):
        balancer = ImbalancedDataHandler(config)
        X, y = balancer.handle_imbalanced_data(X, y)

    return X, y
```

## Future Enhancements

### Planned Features

1. **Advanced Generative Models**
   - StyleGAN for image-like tabular data
   - Transformer-based generative models
   - Multi-modal generation

2. **Automated Hyperparameter Tuning**
   - Bayesian optimization for SMOTE parameters
   - Neural architecture search for GANs
   - Automated quality threshold adjustment

3. **Real-time Balancing**
   - Streaming data balancing
   - Online learning adaptation
   - Incremental model updates

4. **Explainability Features**
   - SHAP value integration for synthetic data
   - Feature importance analysis
   - Bias detection and mitigation

5. **Scalability Improvements**
   - Distributed processing support
   - Cloud-native deployment options
   - Edge computing compatibility

## Support and Contributing

### Getting Help

1. Check the troubleshooting section above
2. Review the test cases for usage examples
3. Check the API documentation for endpoint details
4. Review the source code comments for implementation details

### Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure backward compatibility
5. Test performance impact of changes

### Version History

- **v1.0.0**: Initial implementation with basic SMOTE and synthetic generation
- **v1.1.0**: Added advanced SMOTE variants and quality assessment
- **v1.2.0**: Integrated financial domain-specific balancing
- **v1.3.0**: Added comprehensive API endpoints and testing
- **v2.0.0**: Major refactor with improved architecture and performance

---

For more information or support, please refer to the main Maria Conciliadora documentation or contact the development team.