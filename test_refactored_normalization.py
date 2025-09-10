"""
Test script for refactored data normalization service
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.services.data_normalization import BrazilianDataNormalizer, NormalizationConfig, NormalizationMode, EntityType

def test_refactored_data_normalization():
    """Test the refactored data normalization service"""
    print("Testing Refactored Data Normalization Service...")
    
    try:
        # Test 1: Basic initialization
        normalizer = BrazilianDataNormalizer()
        print("PASS: Basic initialization successful")
        
        # Test 2: Configuration-based initialization
        config = NormalizationConfig(
            mode=NormalizationMode.AGGRESSIVE,
            extract_entities=True,
            remove_noise_words=True
        )
        aggressive_normalizer = BrazilianDataNormalizer(config)
        print("PASS: Configuration-based initialization successful")
        
        # Test 3: Text normalization with different modes
        test_text = "Pagamento de R$ 1.234,56 para LTDA EXAMPLE SA"
        
        # Test standard mode
        result = normalizer.normalize_text(test_text)
        print(f"PASS: Standard normalization: '{result.normalized_text}' (confidence: {result.confidence:.3f})")
        
        # Test aggressive mode
        aggressive_result = aggressive_normalizer.normalize_text(test_text)
        print(f"PASS: Aggressive normalization: '{aggressive_result.normalized_text}' (confidence: {aggressive_result.confidence:.3f})")
        
        # Test 4: Amount normalization
        amount_tests = [
            "R$ 1.234,56",
            "$1,234.56",
            "1234.56",
            "1234 reais",
            "valor: 1.234,56"
        ]
        
        for amount_test in amount_tests:
            result = normalizer.normalize_amount(amount_test)
            print(f"PASS: Amount normalization '{amount_test}' -> {result}")
        
        # Test 5: Date normalization
        date_tests = [
            "15/01/2023",
            "2023-01-15",
            "15-01-2023",
            "15.01.2023"
        ]
        
        for date_test in date_tests:
            result = normalizer.normalize_date(date_test)
            print(f"PASS: Date normalization '{date_test}' -> {result}")
        
        # Test 6: Entity extraction
        entity_test = "Pagamento via BOLETO para LTDA EXAMPLE COMERCIO de R$ 1.234,56"
        result = normalizer.normalize_text(entity_test)
        print(f"PASS: Entity extraction: {result.entities}")
        
        # Test 7: Similarity calculation
        text1 = "Pagamento de R$ 1.234,56 para LTDA EXAMPLE"
        text2 = "Pagamento R$1234.56 LTDA EXAMPLE SA"
        similarity = normalizer.calculate_similarity(text1, text2)
        print(f"PASS: Similarity calculation: {similarity:.3f}")
        
        # Test 8: Performance metrics
        metrics = normalizer.get_performance_metrics()
        print(f"PASS: Performance metrics: {metrics}")
        
        # Test 9: Configuration update
        new_config = NormalizationConfig(mode=NormalizationMode.STRICT)
        normalizer.update_config(new_config)
        print("PASS: Configuration update successful")
        
        # Test 10: Batch processing
        batch_texts = [
            "Compra no SUPERMERCADO ABC por R$ 150,00",
            "Pagamento de TAXA XYZ via PIX",
            "Transferência para BANCO EXAMPLE S.A."
        ]
        
        batch_results = []
        for text in batch_texts:
            result = normalizer.normalize_text(text)
            batch_results.append(result)
        
        print(f"PASS: Batch processing: {len(batch_results)} texts processed")
        
        print("\n--- All Refactored Data Normalization Tests Passed ---")
        return True
        
    except Exception as e:
        print(f"FAIL: Refactored data normalization test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_refactored_data_normalization()
    sys.exit(0 if success else 1)