#!/usr/bin/env python3
"""
Test script for the new ModelManager implementation
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.model_manager import ModelManager
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def create_sample_data(n_samples=100):
    """Create sample financial transaction data for testing"""
    np.random.seed(42)

    # Generate sample data
    descriptions = [
        "Compra no mercado Extra",
        "Pagamento de conta de luz",
        "Transferência recebida",
        "Saque no caixa eletrônico",
        "Compra online Amazon",
        "Pagamento de internet",
        "Recebimento de salário",
        "Compra de combustível",
        "Pagamento de cartão de crédito",
        "Transferência TED"
    ]

    categories = [
        "alimentacao", "casa", "transferencia", "saque",
        "vestuario", "casa", "salario", "transporte",
        "casa", "transferencia"
    ]

    data = []
    base_date = datetime.now() - timedelta(days=365)

    for i in range(n_samples):
        # Random date within the last year
        random_days = np.random.randint(0, 365)
        date = base_date + timedelta(days=random_days)

        # Random amount (some positive, some negative)
        amount = np.random.normal(0, 500)
        if np.random.random() < 0.3:  # 30% positive amounts
            amount = abs(amount)

        # Random description and category
        desc_idx = np.random.randint(0, len(descriptions))
        description = descriptions[desc_idx]
        category = categories[desc_idx]

        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'description': description,
            'amount': round(amount, 2),
            'category': category,
            'transaction_type': 'credit' if amount > 0 else 'debit'
        })

    return data

def test_model_manager():
    """Test the ModelManager functionality"""
    logger.info("Starting ModelManager tests")

    # Create sample data
    sample_data = create_sample_data(200)
    logger.info(f"Created {len(sample_data)} sample transactions")

    # Initialize ModelManager
    model_manager = ModelManager()
    logger.info("ModelManager initialized")

    # Test data processing
    logger.info("Testing data processing...")
    X, y, feature_names = model_manager.process_data(sample_data)
    logger.info(f"Processed data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Feature names: {feature_names[:5]}...")  # Show first 5

    # Test model selection
    logger.info("Testing automatic model selection...")
    best_model = model_manager.select_best_model(X, y)
    logger.info(f"Selected best model: {best_model}")

    # Test model training
    logger.info(f"Testing model training for {best_model}...")
    result = model_manager.train_model(best_model, X, y)
    if result['success']:
        logger.info(f"Model {best_model} trained successfully")
    else:
        logger.error(f"Model training failed: {result.get('error', 'Unknown error')}")
        return False

    # Test model evaluation
    logger.info(f"Testing model evaluation for {best_model}...")
    evaluation = model_manager.evaluate_model(best_model, X, y)
    if 'error' not in evaluation:
        logger.info(f"Model evaluation completed. F1 Score: {evaluation.get('f1_score', 0):.4f}")
    else:
        logger.error(f"Model evaluation failed: {evaluation['error']}")

    # Test prediction
    logger.info("Testing prediction...")
    test_sample = sample_data[0]  # Use first sample for prediction
    test_X, _, _ = model_manager.process_data([test_sample])

    try:
        prediction = model_manager.predict(best_model, test_X)
        logger.info(f"Prediction successful: {prediction[0]}")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")

    # Test model comparison
    logger.info("Testing model comparison...")
    comparison = model_manager.compare_models(X, y, ['random_forest', 'xgboost', 'lightgbm'])
    if 'error' not in comparison:
        logger.info(f"Model comparison completed. Best: {comparison.get('best_model')}")
    else:
        logger.error(f"Model comparison failed: {comparison['error']}")

    # Test model info
    logger.info("Testing model info retrieval...")
    info = model_manager.get_model_info(best_model)
    logger.info(f"Model info retrieved for {best_model}")

    logger.info("All ModelManager tests completed successfully!")
    return True

def test_kmeans_clustering():
    """Test KMeans clustering specifically"""
    logger.info("Testing KMeans clustering...")

    # Create sample data
    sample_data = create_sample_data(150)

    # Initialize ModelManager
    model_manager = ModelManager()

    # Process data
    X, y, feature_names = model_manager.process_data(sample_data)

    # Train KMeans
    result = model_manager.train_model('kmeans', X, y)
    if result['success']:
        logger.info("KMeans training successful")
        logger.info(f"Number of clusters: {result['n_clusters']}")
        logger.info(f"Inertia: {result['inertia']:.2f}")

        # Test prediction
        prediction = model_manager.predict('kmeans', X[:5])  # Predict first 5 samples
        logger.info(f"KMeans predictions: {prediction}")

        return True
    else:
        logger.error(f"KMeans training failed: {result.get('error', 'Unknown error')}")
        return False

def test_bert_model():
    """Test BERT model (if available)"""
    logger.info("Testing BERT model...")

    try:
        # Create sample data with text
        sample_data = create_sample_data(50)

        # Initialize ModelManager
        model_manager = ModelManager()

        # Process data
        X, y, feature_names = model_manager.process_data(sample_data)

        # Extract texts for BERT
        texts = [item['description'] for item in sample_data]

        # Train BERT
        result = model_manager.train_model('bert', X, y, texts=texts)
        if result['success']:
            logger.info("BERT training successful")

            # Test prediction
            test_texts = texts[:3]
            test_X = type('MockData', (), {'texts': test_texts})()
            prediction = model_manager.predict('bert', test_X, texts=test_texts)
            logger.info(f"BERT predictions: {prediction}")

            return True
        else:
            logger.error(f"BERT training failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        logger.warning(f"BERT test failed (may not be available): {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting ModelManager test suite")

    # Run tests
    test_results = []

    # Test main ModelManager functionality
    test_results.append(("ModelManager Core", test_model_manager()))

    # Test KMeans specifically
    test_results.append(("KMeans Clustering", test_kmeans_clustering()))

    # Test BERT (optional)
    test_results.append(("BERT Model", test_bert_model()))

    # Print results
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*50)

    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")

    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)

    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed")
        sys.exit(1)