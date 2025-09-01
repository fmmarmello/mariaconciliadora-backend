#!/usr/bin/env python3
"""
Test script for the advanced feature engineering implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.services.feature_engineer import FeatureEngineer
from src.services.ai_service import AIService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_sample_transactions():
    """Create sample transaction data for testing"""
    base_date = datetime.now() - timedelta(days=30)

    transactions = [
        {
            'description': 'Compra no supermercado Extra',
            'amount': -150.50,
            'date': (base_date + timedelta(days=i)).isoformat(),
            'category': 'alimentacao',
            'type': 'debito'
        }
        for i in range(10)
    ]

    # Add more diverse transactions
    transactions.extend([
        {
            'description': 'Pagamento de salário',
            'amount': 5000.00,
            'date': (base_date + timedelta(days=5)).isoformat(),
            'category': 'salario',
            'type': 'credito'
        },
        {
            'description': 'Uber para o trabalho',
            'amount': -25.00,
            'date': (base_date + timedelta(days=6)).isoformat(),
            'category': 'transporte',
            'type': 'debito'
        },
        {
            'description': 'Compra na farmácia',
            'amount': -89.90,
            'date': (base_date + timedelta(days=7)).isoformat(),
            'category': 'saude',
            'type': 'debito'
        },
        {
            'description': 'Netflix assinatura',
            'amount': -39.90,
            'date': (base_date + timedelta(days=8)).isoformat(),
            'category': 'lazer',
            'type': 'debito'
        }
    ])

    return transactions


def test_feature_engineer():
    """Test the FeatureEngineer class"""
    print("Testing FeatureEngineer...")

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()

    # Create sample data
    transactions = create_sample_transactions()

    print(f"Created {len(transactions)} sample transactions")

    # Test comprehensive feature extraction
    try:
        X, feature_names = feature_engineer.create_comprehensive_features(transactions)

        print("SUCCESS: Comprehensive feature extraction successful")
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Number of features: {len(feature_names)}")
        print(f"   Sample feature names: {feature_names[:10]}")

        # Test individual feature types
        print("\nTesting individual feature types...")

        # Text embeddings
        texts = [t['description'] for t in transactions]
        embeddings = feature_engineer.extract_text_embeddings(texts)
        print(f"SUCCESS: Text embeddings: {embeddings.shape}")

        # Temporal features
        dates = [t['date'] for t in transactions]
        temporal_df = feature_engineer.extract_temporal_features(dates)
        print(f"SUCCESS: Temporal features: {temporal_df.shape}")

        # Transaction patterns
        pattern_df = feature_engineer.extract_transaction_patterns(transactions)
        print(f"SUCCESS: Transaction patterns: {pattern_df.shape}")

        # Categorical features
        categories = [t['category'] for t in transactions]
        cat_df = feature_engineer.extract_categorical_features(categories)
        print(f"SUCCESS: Categorical features: {cat_df.shape}")

        return True

    except Exception as e:
        print(f"FAILED: Feature engineering test failed: {str(e)}")
        return False


def test_ai_service_integration():
    """Test AI service integration with feature engineering"""
    print("\nTesting AI Service integration...")

    try:
        # Initialize AI service
        ai_service = AIService()

        # Create sample training data
        transactions = create_sample_transactions()

        print(f"Training model with {len(transactions)} transactions...")

        # Train a custom model
        result = ai_service.train_custom_model(transactions, model_type='random_forest')

        if result['success']:
            print("SUCCESS: Model training successful")
            print(f"   Model type: {result['model_type']}")
            print(f"   Accuracy: {result.get('accuracy', 0):.3f}")
            print(f"   Training samples: {result['training_data_count']}")

            # Test prediction
            test_description = "Compra no restaurante"
            prediction = ai_service.categorize_with_custom_model(test_description)
            print(f"SUCCESS: Prediction test: '{test_description}' -> '{prediction}'")

            return True
        else:
            print(f"FAILED: Model training failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"FAILED: AI service integration test failed: {str(e)}")
        return False


def test_feature_importance():
    """Test feature importance extraction"""
    print("\nTesting feature importance...")

    try:
        ai_service = AIService()

        # Get feature importance info
        importance_info = ai_service.get_feature_importance_info()

        if 'error' not in importance_info:
            print("SUCCESS: Feature importance extraction successful")
            print(f"   Importance keys: {list(importance_info.keys())}")
        else:
            print(f"WARNING: Feature importance not available: {importance_info['error']}")

        # Get feature names
        feature_names = ai_service.get_feature_names()
        print(f"SUCCESS: Feature names: {len(feature_names)} features available")

        return True

    except Exception as e:
        print(f"FAILED: Feature importance test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("Starting Feature Engineering Tests")
    print("=" * 50)

    tests = [
        test_feature_engineer,
        test_ai_service_integration,
        test_feature_importance
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"FAILED: Test {test.__name__} crashed: {str(e)}")
            results.append(False)

    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASSED" if result else "FAILED"
        print(f"{i+1}. {test.__name__}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: All tests passed! Feature engineering is working correctly.")
        return 0
    else:
        print("WARNING: Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)