# Multi-Language Support Documentation

## Overview

The Maria Conciliadora system now supports multi-language processing for financial documents, enabling effective analysis of financial transactions in multiple languages while maintaining performance and providing detailed language analysis.

## Architecture

### Core Components

#### 1. LanguageDetector (`src/services/language_detector.py`)
Advanced language detection service with multiple detection methods and financial domain specialization.

**Features:**
- FastText-based language identification (primary method)
- Multiple fallback detection methods (LangDetect, LangID, Polyglot)
- Ensemble voting for improved accuracy
- Financial domain language patterns for enhanced detection
- Confidence scoring and alternative language suggestions
- Caching for performance optimization

**Supported Languages:**
- Portuguese (pt) - Primary focus with financial domain specialization
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- 100+ additional languages via FastText

#### 2. MultiLanguagePreprocessor (`src/services/multi_language_preprocessor.py`)
Language-specific text preprocessing with normalization and entity extraction.

**Features:**
- Language-specific text normalization (accents, encoding)
- Currency symbol recognition and standardization
- Date format handling for different locales
- Number format parsing for international standards
- Character encoding detection and conversion
- Entity extraction (emails, phones, accounts, tax IDs)

#### 3. LanguageAwareProcessor (`src/services/language_aware_processor.py`)
Intelligent processing based on detected language with linguistic analysis.

**Features:**
- Language-specific stopword filtering
- Stemming and lemmatization for supported languages
- Named Entity Recognition (NER) integration
- Financial term identification and mapping
- Translation capabilities (optional)
- Cross-language financial terminology mapping

#### 4. InternationalFinancialProcessor (`src/services/international_financial_processor.py`)
Global financial processing with compliance and risk analysis.

**Features:**
- Multi-currency support and conversion
- International bank name recognition
- Cross-border transaction pattern recognition
- Regulatory compliance checking for different jurisdictions
- Tax-related term recognition across languages
- Risk assessment and scoring

#### 5. Enhanced PreprocessingPipeline (`src/services/preprocessing_pipeline.py`)
Updated pipeline with multi-language support.

**New Processing Steps:**
- `LANGUAGE_DETECTION` - Automatic language detection
- `MULTI_LANGUAGE_PREPROCESSING` - Language-specific preprocessing
- `LANGUAGE_AWARE_PROCESSING` - Advanced linguistic analysis
- `INTERNATIONAL_FINANCIAL` - Global financial processing

## API Endpoints

### Language Detection
```http
POST /api/language/detect-language
```

**Request:**
```json
{
  "text": "Olá, esta é uma transação bancária de R$ 1.000,00",
  "options": {
    "include_alternatives": true,
    "confidence_threshold": 0.7
  }
}
```

**Response:**
```json
{
  "success": true,
  "language": "pt",
  "confidence": 0.95,
  "alternatives": [
    {"language": "es", "confidence": 0.05}
  ],
  "processing_time": 0.123
}
```

### Multi-Language Preprocessing
```http
POST /api/language/preprocess-multi-language
```

**Request:**
```json
{
  "text": "Transferência de € 500,00 em 15/12/2023",
  "detected_language": "pt",
  "options": {
    "normalize_currency": true,
    "normalize_dates": true,
    "extract_entities": true
  }
}
```

### Language-Aware Processing
```http
POST /api/language/process-language-aware
```

**Request:**
```json
{
  "text": "Bank transfer of $1000 from account 12345",
  "detected_language": "en",
  "options": {
    "use_stemming": true,
    "extract_financial_terms": true,
    "enable_translation": false
  }
}
```

### International Financial Processing
```http
POST /api/language/process-financial-international
```

**Request:**
```json
{
  "text": "SWIFT transfer from Deutsche Bank to Santander",
  "detected_language": "en",
  "options": {
    "enable_compliance_checking": true,
    "enable_risk_assessment": true
  }
}
```

### Full Pipeline Processing
```http
POST /api/language/process-full-pipeline
```

**Request:**
```json
{
  "text": "Transação internacional PIX de R$ 2500,00",
  "options": {
    "enable_multi_language": true,
    "include_intermediate_results": false
  }
}
```

### Batch Processing
```http
POST /api/language/batch-detect-language
```

**Request:**
```json
{
  "texts": [
    "Olá, como você está?",
    "Hello, how are you?",
    "Hola, ¿cómo estás?"
  ],
  "options": {
    "include_alternatives": false
  }
}
```

### System Information
```http
GET /api/language/supported-languages
GET /api/language/language-stats
```

## Configuration

### Environment Variables
```bash
# Enable multi-language features
MULTI_LANGUAGE_ENABLED=true

# Language detection settings
LANGUAGE_DETECTION_THRESHOLD=0.7
ENABLE_TRANSLATION=false

# Compliance and risk settings
ENABLE_COMPLIANCE_CHECKING=true
ENABLE_RISK_ASSESSMENT=true
RISK_THRESHOLD=0.7
```

### Pipeline Configuration
```python
from src.services.preprocessing_pipeline import PreprocessingPipeline, PipelineConfig

config = PipelineConfig(
    enable_multi_language=True,
    language_detection_threshold=0.7,
    enable_translation=False,
    enable_compliance_checking=True,
    enable_risk_assessment=True,
    supported_languages=['pt', 'en', 'es', 'fr', 'de', 'it']
)

pipeline = PreprocessingPipeline(config)
```

## Supported Languages and Features

| Language | Code | Stemming | Lemmatization | NER | Financial Terms | Currency | Date Format |
|----------|------|----------|---------------|-----|-----------------|----------|-------------|
| Portuguese | pt | ✅ | ✅ | ✅ | ✅ | BRL (R$) | DD/MM/YYYY |
| English | en | ✅ | ✅ | ✅ | ✅ | USD ($) | MM/DD/YYYY |
| Spanish | es | ✅ | ✅ | ✅ | ✅ | EUR (€) | DD/MM/YYYY |
| French | fr | ✅ | ✅ | ✅ | ✅ | EUR (€) | DD/MM/YYYY |
| German | de | ✅ | ✅ | ✅ | ✅ | EUR (€) | DD.MM.YYYY |
| Italian | it | ✅ | ✅ | ✅ | ✅ | EUR (€) | DD/MM/YYYY |

## Financial Terminology Mapping

### Cross-Language Financial Terms
The system includes comprehensive mappings for financial terms across supported languages:

- **Bank/Account Terms:** banco/bank, conta/account, agência/agency
- **Transaction Types:** transferência/transfer, pagamento/payment, depósito/deposit
- **Financial Products:** crédito/credit, débito/debit, investimento/investment
- **Regulatory Terms:** compliance, AML, KYC, sanctions

### Currency Support
- Automatic currency symbol recognition
- Standardization to ISO codes (BRL, USD, EUR, etc.)
- Multi-format support (R$ 1.234,56, $1,234.56, €1.234,56)
- Exchange rate conversion capabilities

## Compliance and Risk Features

### Regulatory Compliance
- **Brazil (BACEN):** Foreign exchange regulations, sanctions screening
- **United States (OFAC):** SDN list checking, AML compliance
- **European Union:** EU sanctions, GDPR compliance
- **Multi-jurisdictional:** Automatic jurisdiction detection and reporting

### Risk Assessment
- Transaction pattern analysis
- Amount-based risk scoring
- Cross-border transaction monitoring
- Sanctions and PEP screening
- Regulatory reporting thresholds

### Risk Levels
- **Low:** Domestic transactions, standard amounts
- **Medium:** International transfers, large amounts
- **High:** Sanctions-related, unusual patterns
- **Critical:** Multiple red flags, high-risk jurisdictions

## Performance Considerations

### Caching
- Language detection results cached for improved performance
- Preprocessing results cached to avoid redundant processing
- Configurable cache sizes and TTL

### Batch Processing
- Parallel processing for multiple texts
- Optimized memory usage for large batches
- Progress tracking and error handling

### Optimization Features
- Lazy loading of language models
- Memory-efficient processing for large documents
- Configurable processing timeouts
- Performance monitoring and metrics

## Error Handling

### Graceful Degradation
- Fallback to simpler methods if advanced features fail
- Default to Portuguese processing for unknown languages
- Comprehensive error logging and reporting

### Validation
- Input text validation and sanitization
- Language confidence threshold checking
- Processing timeout handling

## Testing

### Unit Tests
```bash
# Run language detector tests
pytest tests/unit/test_language_detector.py -v

# Run multi-language preprocessor tests
pytest tests/unit/test_multi_language_preprocessor.py -v

# Run all multi-language tests
pytest tests/unit/ -k "language" -v
```

### Integration Tests
```bash
# Test API endpoints
pytest tests/integration/test_api_endpoints.py -k "language" -v

# Test full pipeline
pytest tests/integration/test_multi_language_pipeline.py -v
```

### Performance Tests
```bash
# Performance benchmarking
pytest tests/performance/test_multi_language_performance.py -v
```

## Usage Examples

### Basic Language Detection
```python
from src.services.language_detector import LanguageDetector

detector = LanguageDetector()
result = detector.detect_language("Olá, esta é uma transação de R$ 500,00")

print(f"Language: {result.consensus_language}")
print(f"Confidence: {result.consensus_confidence}")
```

### Multi-Language Preprocessing
```python
from src.services.multi_language_preprocessor import MultiLanguagePreprocessor

preprocessor = MultiLanguagePreprocessor()
result = preprocessor.preprocess_text("Transferência de € 1.000,00", "pt")

print(f"Processed: {result.processed_text}")
print(f"Entities: {result.extracted_entities}")
```

### Full Pipeline Processing
```python
from src.services.preprocessing_pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()
result = pipeline.process_text("International SWIFT transfer of $5000")

print(f"Language: {result.detected_language}")
print(f"Confidence: {result.language_confidence}")
print(f"Quality: {result.quality_metrics}")
```

## Future Enhancements

### Planned Features
- Additional language support (Chinese, Japanese, Arabic)
- Real-time translation integration
- Advanced NER models for financial entities
- Machine learning-based risk scoring
- Integration with external compliance databases

### Scalability Improvements
- Distributed processing capabilities
- GPU acceleration for language models
- Advanced caching strategies
- Real-time performance monitoring

## Troubleshooting

### Common Issues

1. **Language Detection Accuracy**
   - Ensure sufficient text length (minimum 10 characters)
   - Check for mixed-language content
   - Verify financial domain terms are recognized

2. **Performance Issues**
   - Enable caching for repeated texts
   - Use batch processing for multiple texts
   - Monitor memory usage with large documents

3. **Encoding Problems**
   - Ensure UTF-8 encoding for input texts
   - Check for special characters in financial symbols
   - Validate date and number formats

### Debug Information
```python
# Get detailed processing statistics
stats = detector.get_performance_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']}")
print(f"Total detections: {stats['total_detections']}")

# Enable debug logging
import logging
logging.getLogger('src.services.language_detector').setLevel(logging.DEBUG)
```

## Contributing

### Adding New Languages
1. Update language configurations in respective services
2. Add language-specific patterns and mappings
3. Create comprehensive test cases
4. Update documentation

### Extending Financial Terminology
1. Add new terms to financial mappings
2. Include language-specific variations
3. Update confidence scoring if needed
4. Test with real financial documents

---

## API Reference

### LanguageDetector
- `detect_language(text: str) -> DetectionResult`
- `detect_batch(texts: List[str]) -> List[DetectionResult]`
- `get_performance_stats() -> Dict[str, Any]`
- `clear_cache() -> None`

### MultiLanguagePreprocessor
- `preprocess_text(text: str, detected_language: Optional[str]) -> PreprocessingResult`
- `preprocess_batch(texts: List[str], languages: Optional[List[str]]) -> List[PreprocessingResult]`
- `get_performance_stats() -> Dict[str, Any]`

### LanguageAwareProcessor
- `process_text(text: str, detected_language: Optional[str]) -> ProcessingResult`
- `process_batch(texts: List[str], languages: Optional[List[str]]) -> List[ProcessingResult]`
- `get_performance_stats() -> Dict[str, Any]`

### InternationalFinancialProcessor
- `process_financial_text(text: str, detected_language: Optional[str]) -> FinancialProcessingResult`
- `process_batch(texts: List[str], languages: Optional[List[str]]) -> List[FinancialProcessingResult]`
- `get_performance_stats() -> Dict[str, Any]`

### PreprocessingPipeline
- `process_text(text: str) -> ProcessingResult`
- `process_batch(texts: List[str]) -> List[ProcessingResult]`
- `get_pipeline_metrics() -> Dict[str, Any]`