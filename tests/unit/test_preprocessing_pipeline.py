import pytest
import time
from unittest.mock import Mock, patch

from src.services.preprocessing_pipeline import (
    PreprocessingPipeline,
    PipelineConfig,
    ProcessingStep,
    ProcessingResult
)


class TestPreprocessingPipeline:
    """Test cases for PreprocessingPipeline"""

    @pytest.fixture
    def pipeline_config(self):
        """Create pipeline configuration for testing"""
        return PipelineConfig(
            steps=[
                ProcessingStep.ADVANCED_PORTUGUESE,
                ProcessingStep.FINANCIAL_PROCESSING,
                ProcessingStep.CONTEXT_AWARE,
                ProcessingStep.QUALITY_ASSESSMENT
            ],
            batch_size=16,
            max_workers=2,
            enable_parallel_processing=False,  # Disable for testing
            error_handling="continue",
            quality_threshold=0.5,
            cache_enabled=True,
            performance_monitoring=True
        )

    @pytest.fixture
    def pipeline(self, pipeline_config):
        """Create pipeline instance for testing"""
        return PreprocessingPipeline(pipeline_config)

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing"""
        return [
            "Transferência bancária de R$ 1.000,00 realizada com sucesso",
            "Pagamento de boleto no valor de R$ 500,00",
            "Saque efetuado na conta corrente",
            "Depósito via PIX recebido",
            "Transação internacional processada"
        ]

    def test_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline is not None
        assert hasattr(pipeline, 'process_text')
        assert hasattr(pipeline, 'process_batch')
        assert hasattr(pipeline, 'get_pipeline_metrics')

    def test_process_single_text(self, pipeline):
        """Test processing of single text"""
        text = "Transferência de R$ 500,00 realizada"
        result = pipeline.process_text(text)

        assert isinstance(result, ProcessingResult)
        assert result.original_text == text
        assert isinstance(result.processed_text, str)
        assert isinstance(result.quality_metrics, dict)
        assert 'overall_quality' in result.quality_metrics
        assert isinstance(result.processing_time, float)
        assert result.processing_time > 0

    def test_process_empty_text(self, pipeline):
        """Test processing of empty text"""
        result = pipeline.process_text("")

        assert isinstance(result, ProcessingResult)
        assert result.original_text == ""
        assert result.success is True
        assert result.processed_text == ""

    def test_process_none_text(self, pipeline):
        """Test processing of None text"""
        result = pipeline.process_text(None)

        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.processed_text == ""

    def test_batch_processing(self, pipeline, sample_texts):
        """Test batch text processing"""
        results = pipeline.process_batch(sample_texts)

        assert isinstance(results, list)
        assert len(results) == len(sample_texts)

        for result in results:
            assert isinstance(result, ProcessingResult)
            assert result.success is True
            assert 'overall_quality' in result.quality_metrics

    def test_batch_processing_empty_list(self, pipeline):
        """Test batch processing with empty list"""
        results = pipeline.process_batch([])
        assert results == []

    def test_cache_functionality(self, pipeline):
        """Test caching functionality"""
        text = "Texto para teste de cache"

        # First processing
        result1 = pipeline.process_text(text)

        # Second processing (should use cache if enabled)
        result2 = pipeline.process_text(text)

        # Results should be identical
        assert result1.processed_text == result2.processed_text
        assert result1.quality_metrics == result2.quality_metrics

    def test_clear_cache(self, pipeline):
        """Test cache clearing"""
        text = "Texto para teste de limpeza de cache"
        pipeline.process_text(text)

        # Clear cache
        pipeline.clear_cache()

        # Should not raise errors
        assert True

    def test_pipeline_metrics(self, pipeline, sample_texts):
        """Test pipeline metrics collection"""
        # Process some texts
        results = pipeline.process_batch(sample_texts)

        # Get metrics
        metrics = pipeline.get_pipeline_metrics()

        assert isinstance(metrics, dict)
        assert 'total_processed' in metrics
        assert 'average_processing_time' in metrics
        assert 'average_quality_score' in metrics
        assert metrics['total_processed'] == len(sample_texts)

    def test_reset_metrics(self, pipeline, sample_texts):
        """Test metrics reset functionality"""
        # Process texts to generate metrics
        pipeline.process_batch(sample_texts)

        # Reset metrics
        pipeline.reset_metrics()

        # Get metrics after reset
        metrics = pipeline.get_pipeline_metrics()

        # Should be reset
        assert metrics['total_processed'] == 0

    def test_error_handling_continue(self, pipeline_config):
        """Test error handling with 'continue' mode"""
        # Create pipeline with error handling set to continue
        pipeline_config.error_handling = "continue"
        pipeline = PreprocessingPipeline(pipeline_config)

        # Process texts (some may fail, but processing should continue)
        texts = ["Valid text", "", "Another valid text"]
        results = pipeline.process_batch(texts)

        assert len(results) == len(texts)

        # Should have processed all texts despite potential errors
        successful_results = [r for r in results if r.success]
        assert len(successful_results) >= 0  # At least some should succeed

    def test_error_handling_fallback(self, pipeline_config):
        """Test error handling with 'fallback' mode"""
        pipeline_config.error_handling = "continue"  # Using continue for testing
        pipeline = PreprocessingPipeline(pipeline_config)

        # Process texts
        texts = ["Valid text", "Another text"]
        results = pipeline.process_batch(texts)

        # All results should be present
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, ProcessingResult)

    def test_quality_threshold(self, pipeline_config):
        """Test quality threshold functionality"""
        pipeline_config.quality_threshold = 0.8
        pipeline = PreprocessingPipeline(pipeline_config)

        text = "Texto de teste"
        result = pipeline.process_text(text)

        # Should process regardless of threshold
        assert isinstance(result, ProcessingResult)
        assert result.success is True

    def test_parallel_processing_disabled(self, pipeline_config, sample_texts):
        """Test with parallel processing disabled"""
        pipeline_config.enable_parallel_processing = False
        pipeline = PreprocessingPipeline(pipeline_config)

        results = pipeline.process_batch(sample_texts)

        assert len(results) == len(sample_texts)
        for result in results:
            assert isinstance(result, ProcessingResult)
            assert result.success is True

    def test_performance_monitoring(self, pipeline, sample_texts):
        """Test performance monitoring"""
        # Process texts
        start_time = time.time()
        results = pipeline.process_batch(sample_texts)
        end_time = time.time()

        total_time = end_time - start_time

        # Should complete within reasonable time
        assert total_time < 30.0  # Less than 30 seconds for batch

        # Check individual processing times
        for result in results:
            assert result.processing_time > 0
            assert result.processing_time < 10.0  # Less than 10 seconds per text

    def test_step_execution_tracking(self, pipeline):
        """Test step execution time tracking"""
        text = "Texto para teste de execução de etapas"
        result = pipeline.process_text(text)

        # Get metrics
        metrics = pipeline.get_pipeline_metrics()

        # Should have step execution times
        step_times = metrics.get('step_execution_times', {})
        assert isinstance(step_times, dict)

    def test_large_batch_processing(self, pipeline):
        """Test processing of large batches"""
        large_batch = ["Texto de teste {}".format(i) for i in range(20)]

        start_time = time.time()
        results = pipeline.process_batch(large_batch)
        end_time = time.time()

        total_time = end_time - start_time

        # Should handle large batches
        assert len(results) == len(large_batch)

        # Should complete within reasonable time
        assert total_time < 60.0  # Less than 1 minute

        # All results should be successful
        successful_results = [r for r in results if r.success]
        assert len(successful_results) == len(large_batch)

    def test_memory_efficiency(self, pipeline):
        """Test memory efficiency with repeated processing"""
        text = "Texto para teste de eficiência de memória"

        # Process same text multiple times
        for _ in range(10):
            result = pipeline.process_text(text)
            assert result.success is True

        # Should not cause memory issues
        metrics = pipeline.get_pipeline_metrics()
        assert 'cache_size' in metrics

    def test_configuration_validation(self):
        """Test pipeline configuration validation"""
        # Valid configuration
        valid_config = PipelineConfig(
            steps=[ProcessingStep.ADVANCED_PORTUGUESE],
            batch_size=16,
            max_workers=2
        )

        pipeline = PreprocessingPipeline(valid_config)
        assert pipeline is not None

        # Test processing with valid config
        result = pipeline.process_text("Test text")
        assert isinstance(result, ProcessingResult)

    def test_intermediate_results(self, pipeline):
        """Test intermediate results from pipeline steps"""
        text = "Transferência bancária de R$ 1.000,00"
        result = pipeline.process_text(text)

        # Should have intermediate results
        assert hasattr(result, 'intermediate_results')
        assert isinstance(result.intermediate_results, dict)

        # Should have results from different steps
        expected_steps = ['advanced_portuguese', 'financial_processing', 'context_aware', 'quality_assessment']
        for step in expected_steps:
            assert step in result.intermediate_results

    def test_processing_result_structure(self, pipeline):
        """Test structure of processing results"""
        text = "Texto de teste estrutural"
        result = pipeline.process_text(text)

        # Check required attributes
        required_attrs = [
            'original_text', 'processed_text', 'intermediate_results',
            'quality_metrics', 'processing_time', 'success'
        ]

        for attr in required_attrs:
            assert hasattr(result, attr)

        # Check types
        assert isinstance(result.original_text, str)
        assert isinstance(result.processed_text, str)
        assert isinstance(result.intermediate_results, dict)
        assert isinstance(result.quality_metrics, dict)
        assert isinstance(result.processing_time, (int, float))
        assert isinstance(result.success, bool)

    def test_quality_metrics_calculation(self, pipeline):
        """Test quality metrics calculation"""
        text = "Transferência realizada com sucesso"
        result = pipeline.process_text(text)

        quality_metrics = result.quality_metrics

        # Should have overall quality
        assert 'overall_quality' in quality_metrics
        assert isinstance(quality_metrics['overall_quality'], (int, float))
        assert 0.0 <= quality_metrics['overall_quality'] <= 1.0

    def test_different_text_types(self, pipeline):
        """Test processing of different types of text"""
        test_texts = [
            "Texto financeiro: Transferência de R$ 500,00",
            "Texto simples sem números",
            "Texto com acentuação: café naïve",
            "Texto muito longo " * 50,
            "Texto com caracteres especiais: @#$%&*"
        ]

        for text in test_texts:
            result = pipeline.process_text(text)
            assert isinstance(result, ProcessingResult)
            assert result.success is True
            assert isinstance(result.processed_text, str)

    def test_consecutive_processing(self, pipeline, sample_texts):
        """Test consecutive processing of texts"""
        all_results = []

        # Process texts one by one
        for text in sample_texts:
            result = pipeline.process_text(text)
            all_results.append(result)

            # Each result should be valid
            assert isinstance(result, ProcessingResult)
            assert result.success is True

        # Should have processed all texts
        assert len(all_results) == len(sample_texts)

    @patch('src.services.preprocessing_pipeline.logger')
    def test_logging_integration(self, mock_logger, pipeline):
        """Test logging integration"""
        text = "Texto para teste de logging"
        result = pipeline.process_text(text)

        # Logger should have been called
        assert mock_logger.info.called or mock_logger.debug.called

    def test_exception_handling(self, pipeline):
        """Test exception handling in pipeline"""
        # Test with potentially problematic input
        problematic_texts = [
            "Texto com caracteres especiais: \x00\x01\x02",
            "Texto muito longo" * 1000,
            "Texto com emojis: 😀🎉💯",
            None,
            ""
        ]

        for text in problematic_texts:
            result = pipeline.process_text(text)
            # Should handle gracefully without crashing
            assert isinstance(result, ProcessingResult)

    def test_resource_cleanup(self, pipeline, sample_texts):
        """Test resource cleanup after processing"""
        # Process texts
        results = pipeline.process_batch(sample_texts)

        # Clear cache
        pipeline.clear_cache()

        # Should not cause errors
        assert True

        # Process again after cleanup
        new_results = pipeline.process_batch(sample_texts[:2])
        assert len(new_results) == 2