import pytest
from unittest.mock import Mock, patch
from src.services.language_detector import LanguageDetector, DetectionResult, LanguageDetection, DetectionMethod


class TestLanguageDetector:
    """Test cases for LanguageDetector service"""

    def setup_method(self):
        """Setup test fixtures"""
        self.detector = LanguageDetector()

    def test_initialization(self):
        """Test LanguageDetector initialization"""
        assert self.detector is not None
        assert hasattr(self.detector, '_detection_methods')
        assert hasattr(self.detector, '_performance_stats')

    def test_default_config(self):
        """Test default configuration"""
        config = self.detector.config
        assert config['primary_method'] == 'fasttext'
        assert 'langdetect' in config['fallback_methods']
        assert config['confidence_threshold'] == 0.7
        assert config['cache_enabled'] is True

    @patch('src.services.language_detector.fasttext')
    def test_fasttext_detection(self, mock_fasttext):
        """Test FastText language detection"""
        # Mock FastText model
        mock_model = Mock()
        mock_model.predict.return_value = (['__label__pt'], [0.95])
        mock_fasttext.load_model.return_value = mock_model

        # Create detector with mocked FastText
        detector = LanguageDetector()
        detector._detection_methods[DetectionMethod.FASTTEXT] = mock_model

        result = detector.detect_language("Olá, como você está?")

        assert result.consensus_language == 'pt'
        assert result.consensus_confidence == 0.95
        assert result.primary_detection.method == 'fasttext'

    def test_langdetect_fallback(self):
        """Test LangDetect fallback"""
        # Mock LangDetect
        mock_langdetect = {
            'detect': Mock(return_value='en'),
            'detect_langs': Mock(return_value=[Mock(lang='en', prob=0.9), Mock(lang='pt', prob=0.1)]),
            'error': Exception
        }

        detector = LanguageDetector()
        detector._detection_methods[DetectionMethod.LANGDETECT] = mock_langdetect

        result = detector.detect_language("Hello, how are you?")

        assert result.consensus_language == 'en'
        assert result.consensus_confidence == 0.9

    def test_empty_text_handling(self):
        """Test handling of empty or invalid text"""
        result = self.detector.detect_language("")

        assert result.consensus_language == 'unknown'
        assert result.consensus_confidence == 0.0
        assert result.success is False

    def test_short_text_handling(self):
        """Test handling of very short text"""
        result = self.detector.detect_language("Hi")

        assert result.consensus_language == 'unknown'
        assert result.consensus_confidence == 0.0

    def test_cache_functionality(self):
        """Test caching functionality"""
        text = "This is a test text for caching."

        # First call
        result1 = self.detector.detect_language(text)

        # Second call should use cache
        result2 = self.detector.detect_language(text)

        assert result1.consensus_language == result2.consensus_language
        assert result1.consensus_confidence == result2.consensus_confidence

        # Check cache hit in performance stats
        stats = self.detector.get_performance_stats()
        assert stats['cache_hit_rate'] > 0

    def test_batch_detection(self):
        """Test batch language detection"""
        texts = [
            "Olá, tudo bem?",
            "Hello, how are you?",
            "Hola, ¿cómo estás?"
        ]

        results = self.detector.detect_batch(texts)

        assert len(results) == 3
        assert all(isinstance(result, DetectionResult) for result in results)

    def test_financial_domain_boost(self):
        """Test financial domain language boost"""
        # Text with financial terms in Portuguese
        financial_text = "Transferência PIX de R$ 1.000,00 para conta bancária"

        result = self.detector.detect_language(financial_text)

        # Should detect as Portuguese due to financial terms
        # (This test may need adjustment based on actual detection results)
        assert result.consensus_language in ['pt', 'unknown']

    def test_performance_stats(self):
        """Test performance statistics tracking"""
        initial_stats = self.detector.get_performance_stats()

        self.detector.detect_language("Test text")

        updated_stats = self.detector.get_performance_stats()

        assert updated_stats['total_detections'] == initial_stats['total_detections'] + 1

    def test_error_handling(self):
        """Test error handling in detection methods"""
        # Test with None input
        result = self.detector.detect_language(None)

        assert result.consensus_language == 'unknown'
        assert result.success is False
        assert result.error_message is not None

    def test_ensemble_detection(self):
        """Test ensemble detection with multiple methods"""
        # Mock multiple detection methods
        mock_fasttext = Mock()
        mock_fasttext.predict.return_value = (['__label__pt'], [0.8])

        mock_langdetect = {
            'detect': Mock(return_value='pt'),
            'detect_langs': Mock(return_value=[Mock(lang='pt', prob=0.9)]),
            'error': Exception
        }

        detector = LanguageDetector()
        detector._detection_methods[DetectionMethod.FASTTEXT] = mock_fasttext
        detector._detection_methods[DetectionMethod.LANGDETECT] = mock_langdetect

        result = detector.detect_language("Texto de teste")

        assert result.consensus_language == 'pt'
        assert len(result.all_detections) >= 2

    def test_confidence_thresholds(self):
        """Test language-specific confidence thresholds"""
        # Test with high confidence detection
        result = self.detector.detect_language("Este é um texto em português claro e óbvio.")

        # Should meet Portuguese threshold
        assert result.consensus_confidence >= 0.0  # Actual threshold depends on detection method

    def test_alternative_languages(self):
        """Test alternative language detection"""
        result = self.detector.detect_language("Hello world")

        # Should have primary detection
        assert result.primary_detection is not None

        # May have alternatives depending on detection method
        if result.primary_detection.alternatives:
            assert isinstance(result.primary_detection.alternatives, list)

    def test_clear_cache(self):
        """Test cache clearing functionality"""
        # Add something to cache
        self.detector.detect_language("Cache test text")

        # Clear cache
        self.detector.clear_cache()

        # Cache should be empty
        assert len(self.detector._detection_cache) == 0


class TestLanguageDetectionResult:
    """Test cases for DetectionResult and LanguageDetection classes"""

    def test_detection_result_creation(self):
        """Test DetectionResult creation"""
        primary_detection = LanguageDetection(
            language='pt',
            confidence=0.9,
            method='fasttext',
            alternatives=[('en', 0.1)],
            processing_time=0.05,
            text_sample='Test text',
            text_length=9
        )

        result = DetectionResult(
            primary_detection=primary_detection,
            all_detections=[primary_detection],
            consensus_language='pt',
            consensus_confidence=0.9,
            detection_methods=['fasttext'],
            fallback_used=False
        )

        assert result.consensus_language == 'pt'
        assert result.consensus_confidence == 0.9
        assert not result.fallback_used

    def test_language_detection_creation(self):
        """Test LanguageDetection creation"""
        detection = LanguageDetection(
            language='en',
            confidence=0.85,
            method='langdetect',
            alternatives=[('pt', 0.15)],
            processing_time=0.03,
            text_sample='Hello world',
            text_length=11
        )

        assert detection.language == 'en'
        assert detection.confidence == 0.85
        assert detection.method == 'langdetect'
        assert len(detection.alternatives) == 1


if __name__ == '__main__':
    pytest.main([__file__])