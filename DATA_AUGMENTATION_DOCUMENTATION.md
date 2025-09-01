# Data Augmentation System Documentation

## Overview

The Maria Conciliadora data augmentation system provides comprehensive capabilities for expanding training datasets through multiple augmentation strategies. The system is designed to generate high-quality synthetic data while maintaining data integrity and providing detailed quality metrics.

## Architecture

### Core Components

1. **DataAugmentationPipeline** - Main orchestration framework
2. **TextAugmentationEngine** - Advanced text augmentation with multiple strategies
3. **NumericalAugmentationEngine** - Smart numerical data augmentation
4. **CategoricalAugmentationEngine** - Category augmentation with business rules
5. **TemporalAugmentationEngine** - Date/time augmentation with constraints
6. **SyntheticDataGenerator** - GAN/VAE-based synthetic data generation
7. **AugmentationQualityControl** - Quality assurance and validation

## Features

### Text Augmentation

- **Synonym Replacement**: Uses WordNet and domain-specific dictionaries
- **Back-translation**: Multi-language translation through English, Spanish, French
- **Paraphrasing**: Transformer-based paraphrasing using T5/PEGASUS models
- **Financial Terminology Preservation**: Maintains financial domain accuracy
- **Context-aware Augmentation**: Semantic meaning preservation

### Numerical Augmentation

- **Gaussian Noise Injection**: Configurable variance with distribution preservation
- **Scaling Operations**: Realistic value transformations
- **Financial Amount Augmentation**: Context-aware amount generation
- **Outlier Generation**: Realistic outlier creation
- **Statistical Distribution Preservation**: Maintains original data characteristics

### Categorical Augmentation

- **Label-preserving Transformations**: Maintains category integrity
- **Similar Category Mapping**: Intelligent category relationships
- **Rare Category Handling**: Automatic rare category detection and mapping
- **Business Rule Compliance**: Financial domain rule enforcement

### Temporal Augmentation

- **Date Shifting**: Realistic date modifications with constraints
- **Pattern Generation**: Business cycle and seasonal patterns
- **Holiday Awareness**: Brazilian holiday calendar integration
- **Business Day Compliance**: Weekday/weekend preservation

### Synthetic Data Generation

- **VAE (Variational Autoencoder)**: Feature learning and generation
- **GAN Support**: Generative Adversarial Networks (planned)
- **Conditional Generation**: Pattern-based synthetic data
- **Quality Assessment**: Synthetic data validation

### Quality Control

- **Statistical Similarity Checks**: Kolmogorov-Smirnov and other tests
- **Semantic Preservation Validation**: Text meaning integrity
- **Business Rule Verification**: Domain-specific constraint checking
- **Distribution Preservation**: Categorical and numerical distribution analysis

## API Endpoints

### Base URL: `/api/data-augmentation`

#### Health Check
```http
GET /api/data-augmentation/health
```

#### Comprehensive Data Augmentation
```http
POST /api/data-augmentation/augment
Content-Type: application/json

{
  "data": [
    {
      "description": "Pagamento de conta de luz",
      "amount": -150.50,
      "date": "2024-01-15",
      "transaction_type": "debit",
      "category": "serviços"
    }
  ],
  "data_type": "transaction",
  "config": {
    "general": {
      "augmentation_ratio": 2.0
    }
  }
}
```

#### Text Augmentation Only
```http
POST /api/data-augmentation/augment/text
Content-Type: application/json

{
  "texts": ["Pagamento de conta", "Recebimento de salário"],
  "config": {
    "strategies": ["synonym_replacement", "paraphrasing"]
  }
}
```

#### Numerical Augmentation Only
```http
POST /api/data-augmentation/augment/numerical
Content-Type: application/json

{
  "values": [100.50, 200.75, 300.25],
  "config": {
    "strategies": ["gaussian_noise", "scaling"]
  }
}
```

#### Synthetic Data Generation
```http
POST /api/data-augmentation/generate-synthetic
Content-Type: application/json

{
  "data": [...],
  "sample_size": 100,
  "config": {
    "method": "vae"
  }
}
```

#### Quality Validation
```http
POST /api/data-augmentation/validate-quality
Content-Type: application/json

{
  "original_data": [...],
  "augmented_data": [...]
}
```

#### System Metrics
```http
GET /api/data-augmentation/metrics
```

#### Configuration Management
```http
GET /api/data-augmentation/config
POST /api/data-augmentation/config
```

## Configuration

### Default Configuration

```python
{
  'text_augmentation': {
    'enabled': True,
    'strategies': ['synonym_replacement', 'back_translation', 'paraphrasing'],
    'synonym_config': {
      'replacement_rate': 0.3,
      'use_wordnet': True,
      'domain_specific_dict': True
    }
  },
  'numerical_augmentation': {
    'enabled': True,
    'strategies': ['gaussian_noise', 'scaling', 'outlier_generation'],
    'noise_config': {
      'std_multiplier': 0.1,
      'preserve_distribution': True
    }
  },
  'categorical_augmentation': {
    'enabled': True,
    'strategies': ['label_preservation', 'similar_category_mapping']
  },
  'temporal_augmentation': {
    'enabled': True,
    'strategies': ['date_shifting', 'pattern_generation']
  },
  'synthetic_generation': {
    'enabled': True,
    'method': 'vae',
    'sample_size_ratio': 0.5,
    'quality_threshold': 0.8
  },
  'quality_control': {
    'enabled': True,
    'statistical_similarity_threshold': 0.9,
    'semantic_preservation_threshold': 0.85,
    'business_rule_compliance': True
  },
  'general': {
    'augmentation_ratio': 2.0,
    'random_seed': 42,
    'batch_size': 100,
    'cache_enabled': True
  }
}
```

## Usage Examples

### Basic Data Augmentation

```python
from src.services.data_augmentation_pipeline import DataAugmentationPipeline

# Initialize pipeline
pipeline = DataAugmentationPipeline()

# Sample transaction data
data = [
    {
        'description': 'Pagamento de conta de luz',
        'amount': -150.50,
        'date': '2024-01-15',
        'category': 'serviços'
    }
]

# Augment data
augmented_data, report = pipeline.augment_dataset(data, 'transaction')

print(f"Original size: {report['original_size']}")
print(f"Augmented size: {report['final_size']}")
print(f"Quality score: {report['quality_score']}")
```

### Text Augmentation Only

```python
from src.services.text_augmentation_engine import TextAugmentationEngine

engine = TextAugmentationEngine({
    'strategies': ['synonym_replacement', 'paraphrasing']
})

texts = ["Pagamento de conta de luz", "Recebimento de salário"]
augmented_texts = engine.augment_batch(texts)

for i, variations in enumerate(augmented_texts):
    print(f"Original: {texts[i]}")
    print(f"Variations: {variations}")
```

### Quality Validation

```python
from src.services.augmentation_quality_control import AugmentationQualityControl

controller = AugmentationQualityControl({
    'statistical_similarity_threshold': 0.9,
    'semantic_preservation_threshold': 0.85
})

validation_results = controller.validate_augmentation(original_data, augmented_data)

print(f"Validation passed: {validation_results['validation_passed']}")
print(f"Quality score: {validation_results['quality_score']}")
```

## Integration with Training Pipelines

### Model Manager Integration

The data augmentation system integrates seamlessly with the existing model manager:

```python
from src.services.model_manager import ModelManager
from src.services.data_augmentation_pipeline import DataAugmentationPipeline

# Initialize components
model_manager = ModelManager()
augmentation_pipeline = DataAugmentationPipeline()

# Augment training data before model training
original_data = load_training_data()
augmented_data, _ = augmentation_pipeline.augment_dataset(original_data, 'transaction')

# Train model with augmented data
model_manager.train_model(augmented_data, target_column='is_anomaly')
```

### Feature Engineering Integration

Augmentation works with the existing feature engineering pipeline:

```python
from src.services.feature_engineer import FeatureEngineer
from src.services.data_augmentation_pipeline import DataAugmentationPipeline

# Augment data first
pipeline = DataAugmentationPipeline()
augmented_data, _ = pipeline.augment_dataset(original_data, 'transaction')

# Then apply feature engineering
feature_engineer = FeatureEngineer()
features, feature_names = feature_engineer.create_comprehensive_features(
    augmented_data.to_dict('records')
)
```

## Quality Metrics

### Statistical Metrics

- **KS Similarity Score**: Kolmogorov-Smirnov test for distribution similarity
- **Mean Similarity Score**: Relative difference in means
- **Std Similarity Score**: Relative difference in standard deviations
- **Statistical Similarity Score**: Overall statistical similarity (0-1)

### Semantic Metrics

- **Semantic Preservation Score**: Text meaning integrity (0-1)
- **Vocabulary Overlap**: Common words between original and augmented
- **Readability Preservation**: Text complexity maintenance

### Business Rule Metrics

- **Business Rule Compliance**: Boolean compliance status
- **Violation Count**: Number of rule violations detected
- **Compliance Score**: Percentage of compliant records

### Overall Quality Score

Weighted combination of all metrics:
- Statistical Similarity: 40%
- Semantic Preservation: 30%
- Distribution Preservation: 20%
- Business Rule Compliance: 10%

## Best Practices

### Data Preparation

1. **Clean Data First**: Ensure input data is properly cleaned and validated
2. **Balance Classes**: Use augmentation to balance underrepresented classes
3. **Domain Knowledge**: Incorporate financial domain expertise in configurations

### Configuration Tuning

1. **Start Conservative**: Begin with lower augmentation ratios
2. **Quality First**: Prioritize quality over quantity
3. **Iterative Refinement**: Monitor quality metrics and adjust configurations

### Performance Optimization

1. **Batch Processing**: Use batch processing for large datasets
2. **Caching**: Enable caching for repeated operations
3. **Selective Augmentation**: Apply different strategies to different data types

### Quality Assurance

1. **Regular Validation**: Continuously validate augmentation quality
2. **Threshold Monitoring**: Set appropriate quality thresholds
3. **Human Review**: Include human review for critical applications

## Troubleshooting

### Common Issues

1. **Low Quality Scores**
   - Check input data quality
   - Adjust augmentation parameters
   - Review business rules

2. **Performance Issues**
   - Reduce augmentation ratio
   - Enable caching
   - Use batch processing

3. **Memory Errors**
   - Process data in smaller batches
   - Disable memory-intensive features
   - Monitor system resources

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.getLogger('src.services.data_augmentation_pipeline').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **Advanced GAN Implementation**: Full GAN support with mode collapse prevention
2. **Multi-modal Augmentation**: Image and document augmentation
3. **Reinforcement Learning**: Quality-driven augmentation strategy optimization
4. **Federated Learning**: Privacy-preserving distributed augmentation
5. **Real-time Augmentation**: Streaming data augmentation capabilities

### Research Directions

1. **Adversarial Robustness**: Augmentation for adversarial attack resistance
2. **Domain Adaptation**: Cross-domain augmentation techniques
3. **Explainable AI**: Interpretable augmentation decision-making
4. **Automated Configuration**: ML-based hyperparameter optimization

## Support and Maintenance

### Monitoring

- Quality metrics dashboard
- Performance monitoring
- Error rate tracking
- Usage analytics

### Maintenance Tasks

- Regular model updates for text generation
- Quality threshold calibration
- Business rule updates
- Dependency updates

### Documentation Updates

- API documentation maintenance
- Configuration examples expansion
- Troubleshooting guide updates
- Performance benchmark updates

---

## Quick Start

1. **Install Dependencies**: Ensure all required packages are installed
2. **Initialize Pipeline**: Create DataAugmentationPipeline instance
3. **Configure Settings**: Adjust configuration for your use case
4. **Augment Data**: Call augment_dataset with your data
5. **Validate Quality**: Check quality metrics and adjust as needed
6. **Integrate**: Use augmented data in your training pipeline

For detailed API documentation, see the individual service modules.