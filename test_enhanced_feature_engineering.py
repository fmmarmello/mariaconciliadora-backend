#!/usr/bin/env python3
"""
Test script for enhanced feature engineering components
Tests the new advanced feature engineering capabilities
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.enhanced_feature_engineer import EnhancedFeatureEngineer
from src.services.advanced_text_feature_extractor import AdvancedTextFeatureExtractor
from src.services.temporal_feature_enhancer import TemporalFeatureEnhancer
from src.services.financial_feature_engineer import FinancialFeatureEngineer
from src.services.quality_assured_feature_pipeline import QualityAssuredFeaturePipeline
from src.services.smote_implementation import SMOTEImplementation
from src.services.data_augmentation_pipeline import DataAugmentationPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_transaction_data():
    """Create sample transaction data for testing"""
    base_date = datetime(2024, 1, 1)

    transactions = [
        {
            'description': 'Compra no supermercado Extra com cartão de crédito',
            'amount': 150.50,
            'date': (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
            'category': 'Alimentação',
            'type': 'debit'
        }
        for i in range(10)
    ]

    # Add some variation
    transactions.extend([
        {
            'description': 'Transferência PIX recebida de João Silva',
            'amount': 500.00,
            'date': (base_date + timedelta(days=i+10)).strftime('%Y-%m-%d'),
            'category': 'Transferência',
            'type': 'credit'
        }
        for i in range(5)
    ])

    transactions.extend([
        {
            'description': 'Pagamento de conta de luz via boleto',
            'amount': 89.90,
            'date': (base_date + timedelta(days=i+15)).strftime('%Y-%m-%d'),
            'category': 'Utilidades',
            'type': 'debit'
        }
        for i in range(3)
    ])

    return transactions


def test_enhanced_feature_engineer():
    """Test the EnhancedFeatureEngineer"""
    logger.info("Testing EnhancedFeatureEngineer...")

    try:
        # Initialize the enhanced feature engineer
        config = {
            'text_processing': {'use_advanced_portuguese': True},
            'data_augmentation': {'enabled': False}  # Disable for basic test
        }
        engineer = EnhancedFeatureEngineer(config)

        # Create sample data
        transactions = create_sample_transaction_data()

        # Test enhanced feature creation
        features, feature_names, quality_report = engineer.create_enhanced_features(
            transactions, target_column='category'
        )

        logger.info(f"Enhanced features created: {features.shape}")
        logger.info(f"Feature names: {len(feature_names)}")
        logger.info(f"Quality score: {quality_report.get('overall_score', 'N/A')}")

        assert features.shape[0] == len(transactions), "Feature matrix rows should match input data"
        assert len(feature_names) > 0, "Should have feature names"
        assert isinstance(quality_report, dict), "Should return quality report"

        logger.info("✓ EnhancedFeatureEngineer test passed")
        return True

    except Exception as e:
        logger.error(f"✗ EnhancedFeatureEngineer test failed: {str(e)}")
        return False


def test_advanced_text_extractor():
    """Test the AdvancedTextFeatureExtractor"""
    logger.info("Testing AdvancedTextFeatureExtractor...")

    try:
        # Initialize the text extractor
        extractor = AdvancedTextFeatureExtractor()

        # Sample texts
        texts = [
            "Compra de supermercado no Extra",
            "Transferência PIX recebida",
            "Pagamento de conta de luz",
            "Saque no caixa eletrônico"
        ]

        # Test text feature extraction
        features_dict = extractor.extract_text_features(texts)

        # Test quality assessment
        quality_report = extractor.assess_text_quality(texts)

        logger.info(f"Text features extracted: {list(features_dict.keys())}")
        logger.info(f"Average quality score: {quality_report.get('average_score', 'N/A')}")

        assert len(features_dict) > 0, "Should extract some features"
        assert 'quality_scores' in quality_report, "Should have quality assessment"

        logger.info("✓ AdvancedTextFeatureExtractor test passed")
        return True

    except Exception as e:
        logger.error(f"✗ AdvancedTextFeatureExtractor test failed: {str(e)}")
        return False


def test_temporal_feature_enhancer():
    """Test the TemporalFeatureEnhancer"""
    logger.info("Testing TemporalFeatureEnhancer...")

    try:
        # Initialize the temporal enhancer
        enhancer = TemporalFeatureEnhancer()

        # Sample dates
        dates = [
            '2024-01-15',
            '2024-02-20',
            '2024-03-10',
            '2024-07-04',  # Holiday
            '2024-12-25'   # Christmas
        ]

        # Test temporal feature extraction
        features, feature_names = enhancer.extract_temporal_features(dates)

        # Test temporal validation
        validation_report = enhancer.validate_temporal_consistency(dates)

        logger.info(f"Temporal features extracted: {features.shape}")
        logger.info(f"Feature names: {len(feature_names)}")
        logger.info(f"Consistency score: {validation_report.get('consistency_score', 'N/A')}")

        assert features.shape[0] == len(dates), "Should have features for each date"
        assert len(feature_names) > 0, "Should have feature names"

        logger.info("✓ TemporalFeatureEnhancer test passed")
        return True

    except Exception as e:
        logger.error(f"✗ TemporalFeatureEnhancer test failed: {str(e)}")
        return False


def test_financial_feature_engineer():
    """Test the FinancialFeatureEngineer"""
    logger.info("Testing FinancialFeatureEngineer...")

    try:
        # Initialize the financial engineer
        engineer = FinancialFeatureEngineer()

        # Create sample financial data
        transactions = create_sample_transaction_data()

        # Test financial feature extraction
        features, feature_names = engineer.extract_financial_features(transactions)

        # Test financial data validation
        validation_report = engineer.validate_financial_data(transactions)

        logger.info(f"Financial features extracted: {features.shape}")
        logger.info(f"Feature names: {len(feature_names)}")
        logger.info(f"Data quality score: {validation_report.get('overall_quality_score', 'N/A')}")

        assert features.shape[0] == len(transactions), "Should have features for each transaction"
        assert len(feature_names) > 0, "Should have feature names"

        logger.info("✓ FinancialFeatureEngineer test passed")
        return True

    except Exception as e:
        logger.error(f"✗ FinancialFeatureEngineer test failed: {str(e)}")
        return False


def test_quality_assured_pipeline():
    """Test the QualityAssuredFeaturePipeline"""
    logger.info("Testing QualityAssuredFeaturePipeline...")

    try:
        # Initialize the quality pipeline
        config = {
            'quality_control': {'quality_threshold': 0.7},
            'data_augmentation': {'enabled': False}  # Disable for basic test
        }
        pipeline = QualityAssuredFeaturePipeline(config)

        # Create sample data
        transactions = create_sample_transaction_data()

        # Test pipeline processing
        result = pipeline.process_dataset(transactions, target_column='category')

        if result.get('success'):
            logger.info(f"Pipeline processing successful")
            logger.info(f"Features shape: {result['feature_matrix_shape']}")
            logger.info(f"Quality score: {result['quality_report'].get('overall_score', 'N/A')}")
        else:
            logger.warning(f"Pipeline processing failed: {result.get('error', 'Unknown error')}")

        assert result.get('success', False), "Pipeline should process successfully"
        assert 'features' in result, "Should return features"
        assert 'quality_report' in result, "Should return quality report"

        logger.info("✓ QualityAssuredFeaturePipeline test passed")
        return True

    except Exception as e:
        logger.error(f"✗ QualityAssuredFeaturePipeline test failed: {str(e)}")
        return False


def test_smote_implementation():
    """Test the SMOTE implementation"""
    logger.info("Testing SMOTE implementation...")

    try:
        # Initialize SMOTE engine
        smote = SMOTEImplementation()

        # Create imbalanced sample data
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.choice([0, 1], size=100, p=[0.7, 0.3])  # Imbalanced

        # Test imbalance detection
        imbalance_info = smote.detect_imbalance(X, y)

        logger.info(f"Original class distribution: {imbalance_info.get('class_distribution', {})}")
        logger.info(f"Imbalance ratio: {imbalance_info.get('imbalance_ratio', 'N/A')}")

        # Test SMOTE application
        X_resampled, y_resampled = smote.apply_smote(X, y)

        logger.info(f"Original samples: {len(X)}, Resampled: {len(X_resampled)}")

        assert len(X_resampled) >= len(X), "SMOTE should increase sample size"
        assert len(y_resampled) == len(X_resampled), "X and y should have same length"

        logger.info("✓ SMOTE implementation test passed")
        return True

    except Exception as e:
        logger.error(f"✗ SMOTE implementation test failed: {str(e)}")
        return False


def test_data_augmentation():
    """Test data augmentation pipeline"""
    logger.info("Testing data augmentation pipeline...")

    try:
        # Initialize augmentation pipeline
        config = {
            'general': {'augmentation_ratio': 1.5},
            'text_augmentation': {'enabled': True}
        }
        augmenter = DataAugmentationPipeline(config)

        # Create sample data
        data = create_sample_transaction_data()

        # Test data augmentation
        augmented_data, report = augmenter.augment_dataset(data, data_type='transaction')

        logger.info(f"Original data size: {len(data)}")
        logger.info(f"Augmented data size: {len(augmented_data)}")
        logger.info(f"Augmentation report: {report.get('augmentation_ratio', 'N/A')}")

        assert len(augmented_data) >= len(data), "Augmented data should be at least as large as original"
        assert 'augmentation_report' in report, "Should have augmentation report"

        logger.info("✓ Data augmentation test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Data augmentation test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all enhanced feature engineering tests"""
    logger.info("Starting enhanced feature engineering tests...")

    tests = [
        ("EnhancedFeatureEngineer", test_enhanced_feature_engineer),
        ("AdvancedTextFeatureExtractor", test_advanced_text_extractor),
        ("TemporalFeatureEnhancer", test_temporal_feature_enhancer),
        ("FinancialFeatureEngineer", test_financial_feature_engineer),
        ("QualityAssuredFeaturePipeline", test_quality_assured_pipeline),
        ("SMOTE Implementation", test_smote_implementation),
        ("Data Augmentation", test_data_augmentation)
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {str(e)}")
            results.append((test_name, False))

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)