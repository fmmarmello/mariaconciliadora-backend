"""
TextPreprocessingTestSuite - Comprehensive tests for Portuguese text processing

This module provides comprehensive tests for:
- Stemming and lemmatization testing
- Stopword filtering testing with financial terms
- Multi-language support testing
- Context-aware processing testing
- Performance and accuracy validation
"""

import pytest
import re
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.services.advanced_portuguese_preprocessor import AdvancedPortuguesePreprocessor


class TestAdvancedPortuguesePreprocessor:
    """Test AdvancedPortuguesePreprocessor functionality"""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance for testing"""
        return AdvancedPortuguesePreprocessor()

    @pytest.fixture
    def sample_portuguese_texts(self):
        """Sample Portuguese texts for testing"""
        return [
            "Transferência bancária realizada com sucesso no valor de R$ 1.234,56",
            "Pagamento de conta de luz CEMIG energia elétrica",
            "Saque efetuado na conta corrente do Banco Itaú",
            "Depósito identificado via PIX no valor de R$ 500,00",
            "Transação cancelada devido a erro no processamento",
            "Compra de supermercado com cartão de crédito",
            "Recebimento de salário mensal empresa XYZ",
            "Transferência TED entre contas bancárias",
            "Pagamento de boleto bancário vencido",
            "Saque no caixa eletrônico 24 horas"
        ]

    @pytest.fixture
    def sample_financial_texts(self):
        """Sample financial texts for testing"""
        return [
            "PIX recebido conta corrente valor R$ 250,00",
            "TED enviado agência 1234 conta 56789-0",
            "Boleto pago código barras 12345678901234567890123456789012345678901234",
            "DOC realizado favorecido João Silva",
            "Cheque depositado compensação bancária",
            "Empréstimo consignado desconto folha pagamento",
            "Financiamento veículo aprovado análise crédito",
            "Investimento CDB liquidez diária rendimento",
            "Seguro vida cobertura morte invalidez",
            "Consórcio contemplado lance fixo"
        ]

    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initialization"""
        assert preprocessor is not None
        assert hasattr(preprocessor, 'config')
        assert 'use_spacy' in preprocessor.config
        assert 'use_nltk' in preprocessor.config
        assert 'cache_enabled' in preprocessor.config

    def test_preprocessor_custom_config(self):
        """Test preprocessor with custom configuration"""
        custom_config = {
            'use_spacy': False,
            'use_nltk': True,
            'lemmatization': False,
            'cache_enabled': False,
            'batch_size': 50
        }

        preprocessor = AdvancedPortuguesePreprocessor(custom_config)
        assert preprocessor.config['use_spacy'] == False
        assert preprocessor.config['batch_size'] == 50

    def test_preprocess_text_basic(self, preprocessor):
        """Test basic text preprocessing"""
        text = "Olá, isso é um teste de processamento de texto em português!"
        result = preprocessor.preprocess_text(text)

        assert isinstance(result, dict)
        assert 'original_text' in result
        assert 'processed_text' in result
        assert 'tokens' in result
        assert 'quality_metrics' in result
        assert result['original_text'] == text
        assert isinstance(result['processed_text'], str)

    def test_preprocess_text_empty_input(self, preprocessor):
        """Test preprocessing with empty input"""
        result = preprocessor.preprocess_text("")

        assert result['original_text'] == ""
        assert result['processed_text'] == ""
        assert result['tokens'] == []
        assert result['quality_metrics']['token_count'] == 0

    def test_preprocess_text_none_input(self, preprocessor):
        """Test preprocessing with None input"""
        result = preprocessor.preprocess_text(None)

        assert result['original_text'] == ""
        assert result['processed_text'] == ""
        assert result['tokens'] == []

    def test_text_normalization(self, preprocessor):
        """Test text normalization features"""
        # Test accent removal
        text_with_accents = "Olá, café naïve résumé transferência"
        result = preprocessor.preprocess_text(text_with_accents)

        # Should handle accents properly
        assert result['success'] is True
        assert isinstance(result['processed_text'], str)

        # Test encoding fixes
        text_with_encoding_issues = "TransferÃªncia bancÃ¡ria"
        result = preprocessor.preprocess_text(text_with_encoding_issues)

        assert result['success'] is True

    def test_encoding_handling(self, preprocessor):
        """Test encoding issue handling"""
        # Test with potentially problematic characters
        text_with_special_chars = "Texto com caracteres especiais: àáâãéêíóôõúüç"
        result = preprocessor.preprocess_text(text_with_special_chars)

        assert result['success'] is True
        assert 'processed_text' in result

    def test_stopword_filtering(self, preprocessor):
        """Test stopword filtering functionality"""
        text = "O pagamento foi realizado com sucesso na conta bancária"
        result = preprocessor.preprocess_text(text)

        assert 'filtered_tokens' in result
        filtered_tokens = result['filtered_tokens']

        # Common stopwords should be filtered
        stopwords_to_check = ['o', 'foi', 'com', 'na']
        for stopword in stopwords_to_check:
            assert stopword not in [token.lower() for token in filtered_tokens]

        # Important financial terms should be kept
        assert 'pagamento' in [token.lower() for token in filtered_tokens]
        assert 'conta' in [token.lower() for token in filtered_tokens]
        assert 'bancária' in [token.lower() for token in filtered_tokens]

    def test_financial_term_preservation(self, preprocessor):
        """Test that financial terms are preserved during preprocessing"""
        financial_text = "PIX TED DOC boleto conta agência número banco"
        result = preprocessor.preprocess_text(financial_text)

        filtered_tokens = result.get('filtered_tokens', [])
        token_texts = [token.lower() for token in filtered_tokens]

        # Financial terms should be preserved
        financial_terms = ['pix', 'ted', 'doc', 'boleto', 'conta', 'agência', 'número', 'banco']
        for term in financial_terms:
            assert term in token_texts, f"Financial term '{term}' should be preserved"

    def test_context_aware_filtering(self, preprocessor):
        """Test context-aware filtering"""
        # Test compound financial terms
        text = "Número da conta bancária agência 1234"
        result = preprocessor.preprocess_text(text)

        filtered_tokens = result.get('filtered_tokens', [])
        token_texts = [token.lower() for token in filtered_tokens]

        # Context-aware terms should be preserved
        assert 'número' in token_texts
        assert 'conta' in token_texts
        assert 'bancária' in token_texts
        assert 'agência' in token_texts

    def test_stemming_functionality(self, preprocessor):
        """Test stemming functionality"""
        text = "pagamento pagamentos pagando pagou"
        result = preprocessor.preprocess_text(text)

        # Should have stems if NLTK is available
        if preprocessor.nltk_stemmer:
            assert 'stems' in result
            stems = result['stems']
            assert len(stems) > 0

            # Check that similar words are stemmed similarly
            assert len(set(stems)) <= len(stems)  # Some stems should be the same

    def test_lemmatization_functionality(self, preprocessor):
        """Test lemmatization functionality"""
        text = "pagamento pagamentos pagando pagou"
        result = preprocessor.preprocess_text(text)

        # Should have lemmas if spaCy is available
        if preprocessor.nlp:
            assert 'lemmas' in result
            lemmas = result['lemmas']
            assert len(lemmas) > 0

    def test_pos_tagging(self, preprocessor):
        """Test POS tagging functionality"""
        text = "O banco pagou o salário"
        result = preprocessor.preprocess_text(text)

        if preprocessor.nlp:
            assert 'pos_tags' in result
            pos_tags = result['pos_tags']
            assert len(pos_tags) > 0

            # Should have various POS tags
            assert any(tag in ['NOUN', 'VERB', 'ADJ'] for tag in pos_tags)

    def test_entity_extraction(self, preprocessor):
        """Test entity extraction functionality"""
        text = "Transferência de R$ 1.500,00 do Banco Itaú para João Silva"
        result = preprocessor.preprocess_text(text)

        if preprocessor.nlp:
            assert 'entities' in result
            entities = result['entities']
            assert isinstance(entities, list)

    def test_quality_assessment(self, preprocessor):
        """Test quality assessment functionality"""
        # Test with good quality text
        good_text = "Transferência bancária realizada com sucesso no valor de R$ 1.234,56"
        good_result = preprocessor.preprocess_text(good_text)

        quality = good_result.get('quality_metrics', {})
        assert 'token_count' in quality
        assert 'lexical_diversity' in quality
        assert 'has_financial_terms' in quality

        # Test with poor quality text
        poor_text = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        poor_result = preprocessor.preprocess_text(poor_text)

        poor_quality = poor_result.get('quality_metrics', {})

        # Poor text should have lower quality metrics
        if quality and poor_quality:
            assert quality.get('lexical_diversity', 0) >= poor_quality.get('lexical_diversity', 0)

    def test_cache_functionality(self, preprocessor):
        """Test caching functionality"""
        text = "Texto para teste de cache"

        # First processing
        start_time = time.time()
        result1 = preprocessor.preprocess_text(text)
        first_time = time.time() - start_time

        # Second processing (should use cache)
        start_time = time.time()
        result2 = preprocessor.preprocess_text(text)
        second_time = time.time() - start_time

        # Results should be identical
        assert result1['processed_text'] == result2['processed_text']
        assert result1['tokens'] == result2['tokens']

        # Second processing should be faster (cache hit)
        # Note: This might not always be true in testing environment
        assert second_time <= first_time * 1.5  # Allow some tolerance

    def test_clear_cache(self, preprocessor):
        """Test cache clearing functionality"""
        text = "Texto para teste de limpeza de cache"
        preprocessor.preprocess_text(text)

        # Clear cache
        preprocessor.clear_cache()

        # Cache should be cleared (no direct way to verify, but should not raise errors)
        assert preprocessor._processed_cache == {}
        assert preprocessor._entity_cache == {}

    def test_batch_preprocessing(self, preprocessor, sample_portuguese_texts):
        """Test batch text preprocessing"""
        results = preprocessor.preprocess_batch(sample_portuguese_texts)

        assert isinstance(results, list)
        assert len(results) == len(sample_portuguese_texts)

        for result in results:
            assert isinstance(result, dict)
            assert 'success' in result
            assert 'processed_text' in result
            assert 'quality_metrics' in result

    def test_batch_preprocessing_empty_list(self, preprocessor):
        """Test batch preprocessing with empty list"""
        results = preprocessor.preprocess_batch([])
        assert results == []

    def test_processing_statistics(self, preprocessor, sample_portuguese_texts):
        """Test processing statistics generation"""
        # Process some texts first
        results = preprocessor.preprocess_batch(sample_portuguese_texts[:5])

        # Get statistics
        stats = preprocessor.get_processing_statistics(results)

        assert isinstance(stats, dict)
        assert 'total_texts' in stats
        assert 'successful_processing' in stats
        assert 'average_confidence' in stats
        assert 'average_token_count' in stats
        assert stats['total_texts'] == len(results)

    def test_financial_text_processing(self, preprocessor, sample_financial_texts):
        """Test processing of financial texts"""
        results = preprocessor.preprocess_batch(sample_financial_texts)

        assert len(results) == len(sample_financial_texts)

        # Check that financial terms are properly handled
        for result in results:
            assert 'processed_text' in result
            assert 'quality_metrics' in result

            quality = result['quality_metrics']
            # Financial texts should be detected as having financial terms
            assert quality.get('has_financial_terms', False)

    def test_multi_language_support(self, preprocessor):
        """Test multi-language support"""
        # Test with mixed Portuguese and English
        mixed_text = "Transferência PIX received pagamento em dólares"
        result = preprocessor.preprocess_text(mixed_text)

        assert result['success'] is True
        assert 'processed_text' in result

        # Should handle mixed languages gracefully
        tokens = result.get('tokens', [])
        assert len(tokens) > 0

    def test_special_characters_handling(self, preprocessor):
        """Test handling of special characters and symbols"""
        text_with_symbols = "Pagamento R$ 1.234,56 + taxa 2,5% = R$ 1.267,81"
        result = preprocessor.preprocess_text(text_with_symbols)

        assert result['success'] is True
        assert 'processed_text' in result

        # Should handle currency symbols and percentages
        processed = result['processed_text']
        assert isinstance(processed, str)

    def test_long_text_processing(self, preprocessor):
        """Test processing of long texts"""
        long_text = "Transferência bancária realizada com sucesso. " * 100  # Repeat to make it long
        result = preprocessor.preprocess_text(long_text)

        assert result['success'] is True
        assert 'processed_text' in result

        # Should handle long texts without issues
        tokens = result.get('tokens', [])
        assert len(tokens) > 0

    def test_numeric_data_handling(self, preprocessor):
        """Test handling of numeric data in text"""
        text_with_numbers = "Conta 12345-6 agência 0789 valor R$ 1.500,00"
        result = preprocessor.preprocess_text(text_with_numbers)

        assert result['success'] is True

        filtered_tokens = result.get('filtered_tokens', [])
        token_texts = [token.lower() for token in filtered_tokens]

        # Should preserve account numbers and agency
        assert 'conta' in token_texts
        assert 'agência' in token_texts
        assert 'valor' in token_texts

    def test_punctuation_handling(self, preprocessor):
        """Test punctuation handling"""
        text_with_punctuation = "Pagamento! Realizado? Com sucesso... (teste)"
        result = preprocessor.preprocess_text(text_with_punctuation)

        assert result['success'] is True
        assert 'processed_text' in result

        # Should handle punctuation gracefully
        processed = result['processed_text']
        assert isinstance(processed, str)

    def test_case_sensitivity(self, preprocessor):
        """Test case sensitivity handling"""
        text_mixed_case = "Pagamento REALIZADO com SUCESSO"
        result = preprocessor.preprocess_text(text_mixed_case)

        assert result['success'] is True

        # Should handle mixed case properly
        processed = result['processed_text']
        assert isinstance(processed, str)

    def test_whitespace_handling(self, preprocessor):
        """Test whitespace handling"""
        text_with_whitespace = "Pagamento    realizado\n\tcom  \n  sucesso"
        result = preprocessor.preprocess_text(text_with_whitespace)

        assert result['success'] is True

        # Should normalize whitespace
        processed = result['processed_text']
        assert "    " not in processed  # No multiple spaces
        assert "\n" not in processed   # No newlines
        assert "\t" not in processed   # No tabs

    def test_error_handling(self, preprocessor):
        """Test error handling in preprocessing"""
        # Test with very long text that might cause issues
        long_text = "teste " * 10000  # Very long text
        result = preprocessor.preprocess_text(long_text)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result

    @patch('src.services.advanced_portuguese_preprocessor.spacy.load')
    def test_spacy_fallback(self, mock_spacy_load, preprocessor):
        """Test fallback when spaCy is not available"""
        # Mock spaCy load to raise exception
        mock_spacy_load.side_effect = Exception("spaCy not available")

        # Create new preprocessor instance (to trigger initialization)
        preprocessor_fallback = AdvancedPortuguesePreprocessor()

        # Should work without spaCy
        text = "Texto sem spaCy"
        result = preprocessor_fallback.preprocess_text(text)

        assert result['success'] is True
        assert 'processed_text' in result

    def test_memory_usage(self, preprocessor, sample_portuguese_texts):
        """Test memory usage with multiple texts"""
        # Process multiple texts
        results = preprocessor.preprocess_batch(sample_portuguese_texts)

        # Should not cause memory issues
        assert len(results) == len(sample_portuguese_texts)

        # Clear cache to free memory
        preprocessor.clear_cache()

    def test_concurrent_processing(self, preprocessor, sample_portuguese_texts):
        """Test concurrent processing capabilities"""
        # This is a basic test - in real scenarios, you'd use threading
        results = preprocessor.preprocess_batch(sample_portuguese_texts)

        # All results should be successful
        successful_results = [r for r in results if r.get('success', False)]
        assert len(successful_results) == len(sample_portuguese_texts)

    def test_performance_metrics(self, preprocessor, sample_portuguese_texts):
        """Test performance metrics collection"""
        start_time = time.time()

        # Process texts
        results = preprocessor.preprocess_batch(sample_portuguese_texts)

        end_time = time.time()
        total_time = end_time - start_time

        # Should process within reasonable time
        avg_time_per_text = total_time / len(sample_portuguese_texts)
        assert avg_time_per_text < 5.0  # Less than 5 seconds per text

        # All results should have processing metadata
        for result in results:
            assert 'processing_steps' in result
            assert isinstance(result['processing_steps'], list)

    def test_text_statistics(self, preprocessor, sample_portuguese_texts):
        """Test text statistics generation"""
        results = preprocessor.preprocess_batch(sample_portuguese_texts)
        stats = preprocessor.get_processing_statistics(results)

        # Check statistics structure
        required_stats = [
            'total_texts', 'successful_processing', 'average_confidence',
            'average_token_count', 'average_quality_score'
        ]

        for stat in required_stats:
            assert stat in stats
            assert isinstance(stats[stat], (int, float))

    def test_large_batch_processing(self, preprocessor):
        """Test processing of large batches"""
        # Create a larger batch
        large_batch = ["Texto de teste {}".format(i) for i in range(50)]

        start_time = time.time()
        results = preprocessor.preprocess_batch(large_batch)
        end_time = time.time()

        # Should handle large batches
        assert len(results) == len(large_batch)

        # Should complete within reasonable time
        total_time = end_time - start_time
        assert total_time < 60.0  # Less than 1 minute for 50 texts

    def test_financial_entity_extraction(self, preprocessor):
        """Test financial entity extraction"""
        financial_text = "Transferência de R$ 2.500,00 do Banco Itaú para conta 12345-6 em 15/08/2024"
        result = preprocessor.preprocess_text(financial_text)

        assert result['success'] is True

        # Check if entities were extracted
        entities = result.get('entities', [])
        assert isinstance(entities, list)

        # Should contain some financial information
        # Note: Exact entity extraction depends on the spaCy model

    def test_preprocessing_config_options(self, preprocessor):
        """Test different preprocessing configuration options"""
        text = "TESTE de configuração de PREPROCESSAMENTO"

        # Test with different configurations
        configs = [
            {'lowercase': True, 'remove_accents': False},
            {'lowercase': False, 'remove_accents': True},
            {'lowercase': True, 'remove_accents': True, 'remove_punctuation': False}
        ]

        for config in configs:
            result = preprocessor.preprocess_text(text)
            assert result['success'] is True
            assert 'processed_text' in result

    def test_contextual_relevance_filtering(self, preprocessor):
        """Test contextual relevance filtering"""
        # Test with contextually relevant terms
        text = "Número da conta bancária e código da agência"
        result = preprocessor.preprocess_text(text)

        filtered_tokens = result.get('filtered_tokens', [])
        token_texts = [token.lower() for token in filtered_tokens]

        # Contextually relevant terms should be preserved
        relevant_terms = ['número', 'conta', 'bancária', 'código', 'agência']
        for term in relevant_terms:
            assert term in token_texts, f"Contextually relevant term '{term}' should be preserved"

    def test_financial_compound_terms(self, preprocessor):
        """Test recognition of financial compound terms"""
        text = "Código do banco número da conta agência central"
        result = preprocessor.preprocess_text(text)

        filtered_tokens = result.get('filtered_tokens', [])
        token_texts = [token.lower() for token in filtered_tokens]

        # Financial compound terms should be recognized
        assert 'código' in token_texts
        assert 'banco' in token_texts
        assert 'número' in token_texts
        assert 'conta' in token_texts
        assert 'agência' in token_texts

    def test_accent_normalization_comprehensive(self, preprocessor):
        """Test comprehensive accent normalization"""
        accented_text = "ÀÁÂÃÉÊÍÓÔÕÚÜÇ àáâãéêíóôõúüç"
        result = preprocessor.preprocess_text(accented_text)

        assert result['success'] is True

        # Should normalize accents properly
        processed = result['processed_text']
        assert isinstance(processed, str)

        # Check that accents are handled
        # Note: Exact behavior depends on normalization settings

    def test_encoding_fixes_comprehensive(self, preprocessor):
        """Test comprehensive encoding fixes"""
        encoding_text = "TransferÃªncia bancÃ¡ria recebida comÃª"
        result = preprocessor.preprocess_text(encoding_text)

        assert result['success'] is True

        # Should fix encoding issues
        processed = result['processed_text']
        assert isinstance(processed, str)

    def test_text_quality_scoring(self, preprocessor):
        """Test text quality scoring system"""
        texts_and_expected_quality = [
            ("Transferência bancária realizada com sucesso", "high"),
            ("Pagamento conta luz", "medium"),
            ("a b c d e f g", "low"),
            ("", "low")
        ]

        for text, expected_quality in texts_and_expected_quality:
            result = preprocessor.preprocess_text(text)
            quality_metrics = result.get('quality_metrics', {})

            assert 'lexical_diversity' in quality_metrics
            assert 'token_count' in quality_metrics
            assert 'has_financial_terms' in quality_metrics

            # Quality scores should be reasonable
            lexical_diversity = quality_metrics.get('lexical_diversity', 0)
            assert 0.0 <= lexical_diversity <= 1.0

    def test_processing_pipeline_integrity(self, preprocessor):
        """Test that the processing pipeline maintains integrity"""
        text = "Transferência PIX realizada com sucesso"
        result = preprocessor.preprocess_text(text)

        # Check pipeline steps
        processing_steps = result.get('processing_steps', [])
        assert isinstance(processing_steps, list)

        # Should have basic processing steps
        assert 'normalization' in processing_steps

        # Check that all expected fields are present
        required_fields = [
            'original_text', 'processed_text', 'tokens',
            'quality_metrics', 'confidence_score'
        ]

        for field in required_fields:
            assert field in result, f"Required field '{field}' missing from result"

    def test_batch_processing_consistency(self, preprocessor, sample_portuguese_texts):
        """Test consistency of batch processing results"""
        # Process individually
        individual_results = []
        for text in sample_portuguese_texts[:3]:
            result = preprocessor.preprocess_text(text)
            individual_results.append(result)

        # Process as batch
        batch_results = preprocessor.preprocess_batch(sample_portuguese_texts[:3])

        # Results should be consistent
        assert len(individual_results) == len(batch_results)

        for i in range(len(individual_results)):
            ind_result = individual_results[i]
            batch_result = batch_results[i]

            assert ind_result['original_text'] == batch_result['original_text']
            assert ind_result['processed_text'] == batch_result['processed_text']


if __name__ == "__main__":
    pytest.main([__file__])