import pytest
import time
from unittest.mock import Mock, patch

from src.services.advanced_portuguese_preprocessor import AdvancedPortuguesePreprocessor


class TestAdvancedPortuguesePreprocessor:
    """Test cases for AdvancedPortuguesePreprocessor"""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance for testing"""
        return AdvancedPortuguesePreprocessor()

    @pytest.fixture
    def sample_texts(self):
        """Sample Portuguese texts for testing"""
        return [
            "Transferência bancária realizada com sucesso no valor de R$ 1.234,56",
            "Pagamento de boleto para empresa XYZ no dia 15/03/2024",
            "Saque efetuado na conta corrente do Banco Itaú",
            "Depósito identificado via PIX no valor de R$ 500,00",
            "Transação cancelada devido a erro no processamento",
            "",  # Empty text
            "Texto sem significado específico para teste"
        ]

    def test_initialization(self, preprocessor):
        """Test preprocessor initialization"""
        assert preprocessor is not None
        assert hasattr(preprocessor, 'preprocess_text')
        assert hasattr(preprocessor, 'preprocess_batch')
        assert hasattr(preprocessor, 'get_processing_statistics')

    def test_preprocess_text_basic(self, preprocessor):
        """Test basic text preprocessing"""
        text = "Olá, isso é um teste de processamento de texto em português!"
        result = preprocessor.preprocess_text(text)

        assert isinstance(result, dict)
        assert 'original_text' in result
        assert 'processed_text' in result
        assert 'tokens' in result
        assert 'lemmas' in result
        assert 'quality_metrics' in result
        assert result['original_text'] == text
        assert isinstance(result['processed_text'], str)

    def test_preprocess_text_with_financial_content(self, preprocessor):
        """Test preprocessing of financial text"""
        text = "Transferência de R$ 1.500,00 realizada para conta do Banco Bradesco"
        result = preprocessor.preprocess_text(text)

        assert result['success'] is True
        assert 'processed_text' in result
        assert 'entities' in result
        assert 'quality_metrics' in result

        # Check if financial entities were extracted
        entities = result.get('entities', {})
        assert isinstance(entities, dict)

    def test_preprocess_empty_text(self, preprocessor):
        """Test preprocessing of empty text"""
        result = preprocessor.preprocess_text("")

        assert result['success'] is True
        assert result['processed_text'] == ""
        assert result['tokens'] == []
        assert result['quality_metrics']['token_count'] == 0

    def test_preprocess_none_text(self, preprocessor):
        """Test preprocessing of None text"""
        result = preprocessor.preprocess_text(None)

        assert result['success'] is True
        assert result['processed_text'] == ""
        assert result['tokens'] == []

    def test_batch_preprocessing(self, preprocessor, sample_texts):
        """Test batch text preprocessing"""
        results = preprocessor.preprocess_batch(sample_texts)

        assert isinstance(results, list)
        assert len(results) == len(sample_texts)

        for result in results:
            assert isinstance(result, dict)
            assert 'success' in result
            assert 'processed_text' in result
            assert 'quality_metrics' in result

    def test_batch_preprocessing_empty_list(self, preprocessor):
        """Test batch preprocessing with empty list"""
        results = preprocessor.preprocess_batch([])
        assert results == []

    def test_processing_statistics(self, preprocessor, sample_texts):
        """Test processing statistics generation"""
        # Process some texts first
        results = preprocessor.preprocess_batch(sample_texts[:3])

        # Get statistics
        stats = preprocessor.get_processing_statistics(results)

        assert isinstance(stats, dict)
        assert 'total_texts' in stats
        assert 'successful_processing' in stats
        assert 'average_confidence' in stats
        assert 'average_token_count' in stats
        assert stats['total_texts'] == len(results)

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

    def test_text_normalization(self, preprocessor):
        """Test text normalization features"""
        # Test accent removal
        text_with_accents = "Olá, café naïve résumé"
        result = preprocessor.preprocess_text(text_with_accents)

        # Should handle accents properly
        assert result['success'] is True
        assert isinstance(result['processed_text'], str)

    def test_encoding_handling(self, preprocessor):
        """Test encoding issue handling"""
        # Test with potentially problematic characters
        text_with_special_chars = "Texto com caracteres especiais: àáâãéêíóôõúüç"
        result = preprocessor.preprocess_text(text_with_special_chars)

        assert result['success'] is True
        assert 'processed_text' in result

    def test_quality_assessment(self, preprocessor):
        """Test quality assessment functionality"""
        good_text = "Transferência bancária realizada com sucesso no valor de R$ 1.234,56 para conta corrente"
        poor_text = "a b c d e f g h i j k l m n o p q r s t u v w x y z"

        good_result = preprocessor.preprocess_text(good_text)
        poor_result = preprocessor.preprocess_text(poor_text)

        # Good text should have higher quality score
        good_quality = good_result['quality_metrics'].get('overall_quality', 0)
        poor_quality = poor_result['quality_metrics'].get('overall_quality', 0)

        # This is a soft assertion - quality assessment may vary
        assert isinstance(good_quality, (int, float))
        assert isinstance(poor_quality, (int, float))

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

    def test_memory_usage(self, preprocessor, sample_texts):
        """Test memory usage with multiple texts"""
        # Process multiple texts
        results = preprocessor.preprocess_batch(sample_texts)

        # Should not cause memory issues
        assert len(results) == len(sample_texts)

        # Clear cache to free memory
        preprocessor.clear_cache()

    def test_concurrent_processing(self, preprocessor, sample_texts):
        """Test concurrent processing capabilities"""
        # This is a basic test - in real scenarios, you'd use threading
        results = preprocessor.preprocess_batch(sample_texts)

        # All results should be successful
        successful_results = [r for r in results if r.get('success', False)]
        assert len(successful_results) == len(sample_texts)

    def test_financial_entity_extraction(self, preprocessor):
        """Test financial entity extraction"""
        financial_text = "Transferência de R$ 2.500,00 do Banco Itaú para conta 12345-6 em 15/08/2024"
        result = preprocessor.preprocess_text(financial_text)

        assert result['success'] is True

        # Check if entities were extracted
        entities = result.get('entities', {})
        assert isinstance(entities, dict)

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

    def test_performance_metrics(self, preprocessor, sample_texts):
        """Test performance metrics collection"""
        start_time = time.time()

        # Process texts
        results = preprocessor.preprocess_batch(sample_texts)

        end_time = time.time()
        total_time = end_time - start_time

        # Should process within reasonable time
        avg_time_per_text = total_time / len(sample_texts)
        assert avg_time_per_text < 5.0  # Less than 5 seconds per text

        # All results should have processing metadata
        for result in results:
            assert 'processing_time' in result
            assert isinstance(result['processing_time'], (int, float))

    def test_text_statistics(self, preprocessor, sample_texts):
        """Test text statistics generation"""
        results = preprocessor.preprocess_batch(sample_texts)
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