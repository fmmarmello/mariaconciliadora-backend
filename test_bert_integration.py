#!/usr/bin/env python3
"""
Test script for BERT integration in AI service
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pytest
from services.ai_service import AIService
from services.bert_service import BERTTextClassifier
from services.portuguese_preprocessor import PortugueseTextPreprocessor


def test_portuguese_preprocessor():
    """Test Portuguese text preprocessor functionality"""
    print("Testing Portuguese Text Preprocessor")
    print("=" * 50)

    preprocessor = PortugueseTextPreprocessor()

    # Test basic preprocessing
    test_texts = [
        "Compra no supermercado EXTRA",
        "PAGAMENTO DE CONTA DE LUZ",
        "Transferência PIX recebida R$ 150,00",
        "Restaurante: Jantar com amigos",
        "Débito em conta - Saque ATM"
    ]

    config = {
        'lowercase': True,
        'remove_accents': True,
        'remove_punctuation': True,
        'stopwords': True
    }

    processed_texts = preprocessor.preprocess_batch(test_texts, config)

    print("Original vs Processed texts:")
    for original, processed in zip(test_texts, processed_texts):
        print(f"  '{original}' -> '{processed}'")

    # Test preprocessing stats
    stats = preprocessor.get_preprocessing_stats(test_texts, config)
    print(f"\nPreprocessing stats: {stats}")

    print("\nPortuguese preprocessor test completed!")


def test_bert_classifier_initialization():
    """Test BERT classifier initialization"""
    print("\nTesting BERT Classifier Initialization")
    print("=" * 50)

    try:
        classifier = BERTTextClassifier()

        # Test model loading
        success = classifier.load_model()
        print(f"Model loading: {'SUCCESS' if success else 'FAILED'}")

        if success:
            info = classifier.get_model_info()
            print(f"Model info: {info}")

        print("BERT classifier initialization test completed!")

    except Exception as e:
        print(f"BERT classifier initialization failed: {str(e)}")


def test_bert_training():
    """Test BERT model training"""
    print("\nTesting BERT Model Training")
    print("=" * 50)

    # Sample training data
    train_texts = [
        "Compra no supermercado Extra",
        "Pagamento de conta de luz",
        "Transferência PIX recebida",
        "Compra de remédio na farmácia",
        "Pagamento de aluguel",
        "Compra no shopping",
        "Restaurante jantar",
        "Combustível posto",
        "Salário depositado",
        "Compra Netflix",
        "Pagamento de internet",
        "Compra de livro",
        "Saque no caixa eletrônico",
        "Aplicação em CDB",
        "Compra de passagem ônibus"
    ]

    train_labels = [
        'alimentacao', 'casa', 'transferencia', 'saude', 'casa',
        'vestuario', 'alimentacao', 'transporte', 'salario', 'lazer',
        'casa', 'educacao', 'saque', 'investimento', 'transporte'
    ]

    try:
        classifier = BERTTextClassifier()

        # Load model
        if not classifier.load_model():
            print("Failed to load base model")
            return

        # Train model (with smaller dataset for testing)
        result = classifier.train(train_texts, train_labels)

        if result['success']:
            print("BERT training: SUCCESS")
            print(f"Training metrics: {result.get('training_metrics', {})}")
            print(f"Evaluation metrics: {result.get('evaluation_metrics', {})}")

            # Test prediction
            test_texts = ["Compra no mercado", "Pagamento conta água"]
            predictions = classifier.predict(test_texts)
            print(f"Test predictions: {predictions}")

        else:
            print(f"BERT training failed: {result.get('error', 'Unknown error')}")

        print("BERT training test completed!")

    except Exception as e:
        print(f"BERT training test failed: {str(e)}")


def test_ai_service_bert_integration():
    """Test BERT integration with AI service"""
    print("\nTesting AI Service BERT Integration")
    print("=" * 50)

    ai_service = AIService()

    # Sample training data
    training_data = [
        {'description': 'Compra no supermercado Extra', 'category': 'alimentacao'},
        {'description': 'Pagamento de conta de luz', 'category': 'casa'},
        {'description': 'Transferência PIX recebida', 'category': 'transferencia'},
        {'description': 'Compra de remédio na farmácia', 'category': 'saude'},
        {'description': 'Pagamento de aluguel', 'category': 'casa'},
        {'description': 'Compra no shopping', 'category': 'vestuario'},
        {'description': 'Restaurante jantar', 'category': 'alimentacao'},
        {'description': 'Combustível posto', 'category': 'transporte'},
        {'description': 'Salário depositado', 'category': 'salario'},
        {'description': 'Compra Netflix', 'category': 'lazer'},
        {'description': 'Pagamento de internet', 'category': 'casa'},
        {'description': 'Compra de livro', 'category': 'educacao'},
        {'description': 'Saque no caixa eletrônico', 'category': 'saque'},
        {'description': 'Aplicação em CDB', 'category': 'investimento'},
        {'description': 'Compra de passagem ônibus', 'category': 'transporte'}
    ]

    try:
        # Test BERT training
        print("Training BERT model...")
        bert_result = ai_service.train_bert_model(training_data)

        if bert_result['success']:
            print("BERT training: SUCCESS")
            print(f"Training samples: {bert_result.get('training_samples', 0)}")
            print(f"Categories: {bert_result.get('num_labels', 0)}")

            # Test BERT categorization
            test_descriptions = [
                'Compra no mercado',
                'Pagamento conta água',
                'Depósito salário'
            ]

            print("\nBERT categorizations:")
            for desc in test_descriptions:
                category = ai_service.categorize_with_bert(desc)
                print(f"  '{desc}' -> {category}")

        else:
            print(f"BERT training failed: {bert_result.get('error', 'Unknown error')}")

        # Test model comparison
        print("\nTesting model comparison...")
        comparison = ai_service.compare_model_performance(training_data)

        if 'error' not in comparison:
            print("Model comparison: SUCCESS")
            print(f"Best model: {comparison.get('best_model', 'unknown')}")
            print(f"Recommendation: {comparison.get('recommendation', 'none')}")

            print("\nTraditional models performance:")
            for model, metrics in comparison.get('traditional_models', {}).items():
                if isinstance(metrics, dict) and 'f1_score' in metrics:
                    print(f"  {model}: F1={metrics['f1_score']:.4f}")

            bert_metrics = comparison.get('bert_model', {})
            if bert_metrics:
                print(f"\nBERT model performance:")
                print(f"  F1: {bert_metrics.get('eval_f1', 0):.4f}")
                print(f"  Accuracy: {bert_metrics.get('eval_accuracy', 0):.4f}")

        else:
            print(f"Model comparison failed: {comparison['error']}")

        print("AI service BERT integration test completed!")

    except Exception as e:
        print(f"AI service BERT integration test failed: {str(e)}")


def test_bert_error_handling():
    """Test BERT error handling and fallback mechanisms"""
    print("\nTesting BERT Error Handling")
    print("=" * 50)

    ai_service = AIService()

    try:
        # Test with empty data
        print("Testing with empty data...")
        result = ai_service.train_bert_model([])
        print(f"Empty data result: {result}")

        # Test with insufficient data
        print("\nTesting with insufficient data...")
        small_data = [
            {'description': 'Test', 'category': 'test'}
        ]
        result = ai_service.train_bert_model(small_data)
        print(f"Insufficient data result: {result}")

        # Test categorization with untrained model
        print("\nTesting categorization with untrained BERT...")
        category = ai_service.categorize_with_bert("Test description")
        print(f"Untrained BERT categorization: {category}")

        # Test with invalid text
        print("\nTesting with invalid text...")
        category = ai_service.categorize_with_bert("")
        print(f"Empty text categorization: {category}")

        print("BERT error handling test completed!")

    except Exception as e:
        print(f"BERT error handling test failed: {str(e)}")


def test_bert_model_info():
    """Test BERT model information retrieval"""
    print("\nTesting BERT Model Information")
    print("=" * 50)

    ai_service = AIService()

    try:
        # Get BERT model info
        bert_info = ai_service.get_bert_model_info()
        print(f"BERT model info: {bert_info}")

        # Get model comparison info
        try:
            comparison_info = ai_service.get_model_comparison_info()
            print(f"Model comparison info: {comparison_info}")
        except Exception as e:
            print(f"Error getting model comparison info: {str(e)}")
            # Try to get available models info directly
            try:
                models_info = ai_service.get_available_models_info()
                print(f"Available models info: {models_info}")
            except Exception as e2:
                print(f"Error getting available models info: {str(e2)}")

        print("BERT model information test completed!")

    except Exception as e:
        print(f"BERT model information test failed: {str(e)}")


def run_all_bert_tests():
    """Run all BERT integration tests"""
    print("Running BERT Integration Tests")
    print("=" * 60)

    try:
        test_portuguese_preprocessor()
        test_bert_classifier_initialization()
        test_bert_training()
        test_ai_service_bert_integration()
        test_bert_error_handling()
        test_bert_model_info()

        print("\n" + "=" * 60)
        print("All BERT integration tests completed!")

    except Exception as e:
        print(f"Test suite failed: {str(e)}")


if __name__ == "__main__":
    run_all_bert_tests()