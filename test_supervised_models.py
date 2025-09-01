#!/usr/bin/env python3
"""
Test script for supervised learning models in AI service
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from services.ai_service import AIService

def test_supervised_models():
    """Test the new supervised learning models"""

    # Initialize AI service
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

    print("Testing supervised learning models...")
    print("=" * 50)

    # Test different model types
    model_types = ['auto', 'random_forest', 'xgboost', 'lightgbm', 'bert', 'kmeans']

    for model_type in model_types:
        try:
            print(f"\nTesting {model_type} model:")
            print("-" * 30)

            # Train the model
            result = ai_service.train_custom_model(training_data, model_type)

            if result['success']:
                print(f"[SUCCESS] Model trained successfully!")
                print(f"   Model type: {result.get('model_type', 'unknown')}")
                print(f"   Accuracy: {result.get('accuracy', 0):.4f}")
                if 'f1_score' in result:
                    print(f"   F1-Score: {result['f1_score']:.4f}")
                print(f"   Training samples: {result['training_data_count']}")
                print(f"   Categories: {result['categories_count']}")

                # Test categorization
                test_descriptions = [
                    'Compra no mercado',
                    'Pagamento conta água',
                    'Depósito salário'
                ]

                print("   Test categorizations:")
                for desc in test_descriptions:
                    category = ai_service.categorize_with_custom_model(desc)
                    print(f"     '{desc}' -> {category}")

            else:
                print(f"[FAILED] Model training failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"[ERROR] Error testing {model_type}: {str(e)}")

    print("\n" + "=" * 50)
    print("Testing completed!")

def test_hyperparameter_optimization():
    """Test hyperparameter optimization functionality"""

    # Initialize AI service
    ai_service = AIService()

    # Sample training data (expanded for better optimization)
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
        {'description': 'Compra de passagem ônibus', 'category': 'transporte'},
        # Additional data for better optimization
        {'description': 'Compra no mercado municipal', 'category': 'alimentacao'},
        {'description': 'Pagamento conta água', 'category': 'casa'},
        {'description': 'Transferência TED', 'category': 'transferencia'},
        {'description': 'Consulta médica', 'category': 'saude'},
        {'description': 'Compra de roupa', 'category': 'vestuario'},
        {'description': 'Cinema com pipoca', 'category': 'lazer'},
        {'description': 'Pagamento condomínio', 'category': 'casa'},
        {'description': 'Compra de combustível', 'category': 'transporte'},
        {'description': 'Salário empresa', 'category': 'salario'},
        {'description': 'Compra Spotify', 'category': 'lazer'}
    ]

    print("Testing Hyperparameter Optimization")
    print("=" * 50)

    # Test optimization for each model type with fewer trials for testing
    model_types = ['random_forest', 'xgboost', 'lightgbm']

    for model_type in model_types:
        try:
            print(f"\nTesting optimization for {model_type}:")
            print("-" * 40)

            # Train optimized model with fewer trials for testing
            result = ai_service.train_custom_model_optimized(
                training_data,
                model_type=model_type,
                optimize_hyperparams=True,
                n_trials=10  # Reduced for testing
            )

            if result['success']:
                print(f"[SUCCESS] Optimized {model_type} training completed!")
                print(f"   Optimized: {result.get('optimized', False)}")
                print(f"   Accuracy: {result.get('accuracy', 0):.4f}")
                print(f"   F1-Score: {result.get('f1_score', 0):.4f}")

                if 'best_params' in result:
                    print(f"   Best parameters found: {len(result['best_params'])} parameters")
                    print(f"   Optimization score: {result.get('optimization_score', 0):.4f}")

                if 'performance_comparison' in result:
                    comparison = result['performance_comparison']
                    if comparison:
                        print("   Performance improvements:")
                        for metric, values in comparison.items():
                            improvement_pct = values.get('improvement_percentage', 0)
                            print(f"     {metric}: {improvement_pct:+.2f}%")

                # Test categorization with optimized model
                test_descriptions = [
                    'Compra no mercado',
                    'Pagamento conta água',
                    'Depósito salário'
                ]

                print("   Test categorizations with optimized model:")
                for desc in test_descriptions:
                    category = ai_service.categorize_with_custom_model(desc)
                    print(f"     '{desc}' -> {category}")

            else:
                print(f"[FAILED] Optimized {model_type} training failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"[ERROR] Error testing optimized {model_type}: {str(e)}")

    # Test optimization info retrieval
    print("\nTesting optimization info retrieval:")
    print("-" * 40)

    for model_type in model_types:
        try:
            opt_info = ai_service.get_optimization_info(model_type)
            if 'error' not in opt_info:
                print(f"   {model_type} optimization info retrieved successfully")
                print(f"     Best score: {opt_info.get('best_score', 0):.4f}")
                print(f"     Trials: {opt_info.get('n_trials', 0)}")
            else:
                print(f"   {model_type}: {opt_info['error']}")
        except Exception as e:
            print(f"   Error getting optimization info for {model_type}: {str(e)}")

    # Test available models info
    print("\nTesting available models info:")
    print("-" * 40)

    try:
        models_info = ai_service.get_available_models_info()
        for model_type, info in models_info.items():
            print(f"   {model_type}:")
            print(f"     Trained: {info.get('is_trained', False)}")
            print(f"     Optimized: {info.get('is_optimized', False)}")
            print(f"     Has optimization history: {info.get('has_optimization_history', False)}")
            if info.get('has_optimization_history'):
                print(f"     Best score: {info.get('best_score', 0):.4f}")
    except Exception as e:
        print(f"   Error getting available models info: {str(e)}")

    print("\n" + "=" * 50)
    print("Hyperparameter optimization testing completed!")

def test_optimization_comparison():
    """Test comparison between default and optimized models"""

    ai_service = AIService()

    training_data = [
        {'description': 'Compra no supermercado', 'category': 'alimentacao'},
        {'description': 'Pagamento conta luz', 'category': 'casa'},
        {'description': 'Transferência recebida', 'category': 'transferencia'},
        {'description': 'Compra remédio', 'category': 'saude'},
        {'description': 'Pagamento aluguel', 'category': 'casa'},
        {'description': 'Compra shopping', 'category': 'vestuario'},
        {'description': 'Restaurante', 'category': 'alimentacao'},
        {'description': 'Combustível', 'category': 'transporte'},
        {'description': 'Salário', 'category': 'salario'},
        {'description': 'Compra Netflix', 'category': 'lazer'},
        {'description': 'Pagamento internet', 'category': 'casa'},
        {'description': 'Compra livro', 'category': 'educacao'},
        {'description': 'Saque', 'category': 'saque'},
        {'description': 'Aplicação CDB', 'category': 'investimento'},
        {'description': 'Ônibus', 'category': 'transporte'}
    ]

    print("Testing Optimization vs Default Model Comparison")
    print("=" * 55)

    model_type = 'random_forest'

    try:
        # Train with optimization
        print(f"\nTraining optimized {model_type}...")
        opt_result = ai_service.train_custom_model_optimized(
            training_data,
            model_type=model_type,
            optimize_hyperparams=True,
            n_trials=5
        )

        if opt_result['success'] and 'performance_comparison' in opt_result:
            comparison = opt_result['performance_comparison']
            print(f"\nPerformance comparison for {model_type}:")
            print("-" * 40)

            for metric, values in comparison.items():
                default_val = values.get('default', 0)
                optimized_val = values.get('optimized', 0)
                improvement = values.get('improvement', 0)
                improvement_pct = values.get('improvement_percentage', 0)

                print(f"   {metric}:")
                print(f"     Default: {default_val:.4f}")
                print(f"     Optimized: {optimized_val:.4f}")
                print(f"     Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")

        else:
            print(f"   Could not perform comparison for {model_type}")

    except Exception as e:
        print(f"   Error in comparison test: {str(e)}")

    print("\n" + "=" * 55)
    print("Comparison testing completed!")

def test_bert_models():
    """Test BERT model specifically"""
    ai_service = AIService()

    # Expanded training data for better BERT performance
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
        {'description': 'Compra de passagem ônibus', 'category': 'transporte'},
        # Additional data
        {'description': 'Compra no mercado municipal', 'category': 'alimentacao'},
        {'description': 'Pagamento conta água', 'category': 'casa'},
        {'description': 'Transferência TED', 'category': 'transferencia'},
        {'description': 'Consulta médica', 'category': 'saude'},
        {'description': 'Compra de roupa', 'category': 'vestuario'},
        {'description': 'Cinema com pipoca', 'category': 'lazer'},
        {'description': 'Pagamento condomínio', 'category': 'casa'},
        {'description': 'Compra de combustível', 'category': 'transporte'},
        {'description': 'Salário empresa', 'category': 'salario'},
        {'description': 'Compra Spotify', 'category': 'lazer'}
    ]

    print("Testing BERT Model")
    print("=" * 50)

    try:
        # Test BERT training
        result = ai_service.train_custom_model(training_data, 'bert')

        if result['success']:
            print("[SUCCESS] BERT model trained successfully!")
            print(f"   Training samples: {result.get('training_samples', 0)}")
            print(f"   Categories: {result.get('num_labels', 0)}")

            # Test BERT categorization
            test_descriptions = [
                'Compra no mercado',
                'Pagamento conta água',
                'Depósito salário',
                'Compra remédio',
                'Transferência recebida'
            ]

            print("   BERT categorizations:")
            for desc in test_descriptions:
                category = ai_service.categorize_with_custom_model(desc)
                print(f"     '{desc}' -> {category}")

        else:
            print(f"[FAILED] BERT training failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"[ERROR] Error testing BERT: {str(e)}")

    print("\n" + "=" * 50)
    print("BERT model testing completed!")


if __name__ == "__main__":
    test_supervised_models()
    print("\n" + "=" * 60)
    test_bert_models()
    print("\n" + "=" * 60)
    test_hyperparameter_optimization()
    print("\n" + "=" * 60)
    test_optimization_comparison()