#!/usr/bin/env python3
"""
Simple test for BERT training functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from services.bert_service import BERTTextClassifier

def test_bert_simple():
    """Simple BERT training test"""

    # Sample data
    texts = [
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

    labels = [
        'alimentacao', 'casa', 'transferencia', 'saude', 'casa',
        'vestuario', 'alimentacao', 'transporte', 'salario', 'lazer',
        'casa', 'educacao', 'saque', 'investimento', 'transporte'
    ]

    print("Testing BERT Training")
    print("=" * 30)

    try:
        # Initialize classifier
        classifier = BERTTextClassifier()

        # Load model
        if not classifier.load_model():
            print("Failed to load model")
            return

        print("Model loaded successfully")

        # Train model
        result = classifier.train(texts, labels)

        if result['success']:
            print("Training successful!")
            print(f"F1 Score: {result.get('best_score', 0):.4f}")

            # Test prediction
            test_text = "Compra no mercado"
            prediction = classifier.predict([test_text])
            print(f"Prediction for '{test_text}': {prediction[0]}")

        else:
            print(f"Training failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bert_simple()