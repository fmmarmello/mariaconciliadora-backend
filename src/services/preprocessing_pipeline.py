import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logging_config import get_logger
from src.services.advanced_portuguese_preprocessor import AdvancedPortuguesePreprocessor
from src.services.financial_text_processor import FinancialTextProcessor
from src.services.context_aware_processor import ContextAwareProcessor
from src.services.language_detector import LanguageDetector, DetectionResult
from src.services.multi_language_preprocessor import MultiLanguagePreprocessor
from src.services.language_aware_processor import LanguageAwareProcessor
from src.services.international_financial_processor import InternationalFinancialProcessor

logger = get_logger(__name__)


class ProcessingStep(Enum):
    """Enumeration of available processing steps"""
    LANGUAGE_DETECTION = "language_detection"
    MULTI_LANGUAGE_PREPROCESSING = "multi_language_preprocessing"
    LANGUAGE_AWARE_PROCESSING = "language_aware_processing"
    ADVANCED_PORTUGUESE = "advanced_portuguese"
    FINANCIAL_PROCESSING = "financial_processing"
    INTERNATIONAL_FINANCIAL = "international_financial"
    CONTEXT_AWARE = "context_aware"
    QUALITY_ASSESSMENT = "quality_assessment"


@dataclass
class PipelineConfig:
    """Configuration for preprocessing pipeline"""
    steps: List[ProcessingStep]
    batch_size: int = 32
    max_workers: int = 4
    enable_parallel_processing: bool = True
    error_handling: str = "continue"  # "continue", "stop", "fallback"
    quality_threshold: float = 0.6
    cache_enabled: bool = True
    performance_monitoring: bool = True
    # Multi-language configuration
    enable_multi_language: bool = True
    language_detection_threshold: float = 0.7
    enable_translation: bool = False
    enable_compliance_checking: bool = True
    enable_risk_assessment: bool = True
    supported_languages: List[str] = None

    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ['pt', 'en', 'es', 'fr', 'de', 'it']


@dataclass
class ProcessingResult:
    """Result of a single text processing"""
    original_text: str
    processed_text: str
    intermediate_results: Dict[str, Any]
    quality_metrics: Dict[str, float]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    # Multi-language fields
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    language_detection_result: Optional[DetectionResult] = None


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance"""
    total_processed: int = 0
    total_errors: int = 0
    average_processing_time: float = 0.0
    average_quality_score: float = 0.0
    step_execution_times: Dict[str, float] = None
    cache_hit_rate: float = 0.0

    def __post_init__(self):
        if self.step_execution_times is None:
            self.step_execution_times = {}


class PreprocessingPipeline:
    """
    Modular preprocessing pipeline for Portuguese financial text
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the preprocessing pipeline

        Args:
            config: Pipeline configuration
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize processing components
        self._initialize_components()

        # Initialize metrics and monitoring
        self.metrics = PipelineMetrics()
        self._processing_cache = {}

        # Performance tracking
        self._step_timers = {}
        self._start_time = time.time()

        self.logger.info("PreprocessingPipeline initialized")

    def _get_default_config(self) -> PipelineConfig:
        """Get default pipeline configuration"""
        return PipelineConfig(
            steps=[
                ProcessingStep.LANGUAGE_DETECTION,
                ProcessingStep.MULTI_LANGUAGE_PREPROCESSING,
                ProcessingStep.LANGUAGE_AWARE_PROCESSING,
                ProcessingStep.ADVANCED_PORTUGUESE,
                ProcessingStep.FINANCIAL_PROCESSING,
                ProcessingStep.INTERNATIONAL_FINANCIAL,
                ProcessingStep.CONTEXT_AWARE,
                ProcessingStep.QUALITY_ASSESSMENT
            ],
            batch_size=32,
            max_workers=4,
            enable_parallel_processing=True,
            error_handling="continue",
            quality_threshold=0.6,
            cache_enabled=True,
            performance_monitoring=True,
            # Multi-language defaults
            enable_multi_language=True,
            language_detection_threshold=0.7,
            enable_translation=False,
            enable_compliance_checking=True,
            enable_risk_assessment=True
        )

    def _initialize_components(self):
        """Initialize processing components"""
        try:
            self.components = {}

            # Initialize multi-language components
            if self.config.enable_multi_language:
                # Initialize Language Detector
                self.components[ProcessingStep.LANGUAGE_DETECTION] = LanguageDetector()

                # Initialize Multi-Language Preprocessor
                self.components[ProcessingStep.MULTI_LANGUAGE_PREPROCESSING] = MultiLanguagePreprocessor()

                # Initialize Language-Aware Processor
                self.components[ProcessingStep.LANGUAGE_AWARE_PROCESSING] = LanguageAwareProcessor()

                # Initialize International Financial Processor
                self.components[ProcessingStep.INTERNATIONAL_FINANCIAL] = InternationalFinancialProcessor()

            # Initialize traditional Portuguese components
            self.components[ProcessingStep.ADVANCED_PORTUGUESE] = AdvancedPortuguesePreprocessor()

            # Initialize Financial Text Processor
            self.components[ProcessingStep.FINANCIAL_PROCESSING] = FinancialTextProcessor()

            # Initialize Context-Aware Processor
            self.components[ProcessingStep.CONTEXT_AWARE] = ContextAwareProcessor()

            # Quality assessment is handled by the pipeline itself
            self.components[ProcessingStep.QUALITY_ASSESSMENT] = None

            self.logger.info("All processing components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise

    def process_text(self, text: str) -> ProcessingResult:
        """
        Process a single text through the pipeline

        Args:
            text: Input text to process

        Returns:
            ProcessingResult with processed text and metadata
        """
        start_time = time.time()

        try:
            # Check cache
            if self.config.cache_enabled:
                cache_key = hash(text)
                if cache_key in self._processing_cache:
                    cached_result = self._processing_cache[cache_key]
                    self.metrics.cache_hit_rate = (self.metrics.cache_hit_rate + 1) / 2  # Simple moving average
                    return cached_result

            # Initialize result
            result = ProcessingResult(
                original_text=text,
                processed_text=text,
                intermediate_results={},
                quality_metrics={},
                processing_time=0.0,
                success=True,
                detected_language=None,
                language_confidence=None,
                language_detection_result=None
            )

            # Set current result for step execution
            self._current_result = result

            # Execute pipeline steps
            for step in self.config.steps:
                step_start = time.time()

                try:
                    step_result = self._execute_step(step, result.processed_text, result.intermediate_results)
                    result.intermediate_results[step.value] = step_result

                    # Update processed text if step modified it
                    if 'processed_text' in step_result:
                        result.processed_text = step_result['processed_text']

                    # Track step execution time
                    step_time = time.time() - step_start
                    self._update_step_timer(step.value, step_time)

                except Exception as e:
                    error_msg = f"Error in step {step.value}: {str(e)}"
                    self.logger.error(error_msg)

                    if self.config.error_handling == "stop":
                        result.success = False
                        result.error_message = error_msg
                        break
                    elif self.config.error_handling == "fallback":
                        # Continue with original text
                        continue
                    # For "continue", just log and continue

            # Calculate final quality metrics
            result.quality_metrics = self._calculate_quality_metrics(result)

            # Check quality threshold
            overall_quality = result.quality_metrics.get('overall_quality', 0.0)
            if overall_quality < self.config.quality_threshold:
                self.logger.warning(f"Text quality below threshold: {overall_quality}")

            # Calculate processing time
            result.processing_time = time.time() - start_time

            # Update metrics
            self._update_metrics(result)

            # Cache result
            if self.config.cache_enabled and result.success:
                self._processing_cache[cache_key] = result

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing text: {str(e)}")

            return ProcessingResult(
                original_text=text,
                processed_text=text,
                intermediate_results={},
                quality_metrics={'overall_quality': 0.0},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

    def _execute_step(self, step: ProcessingStep, text: str, intermediate_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single processing step"""
        try:
            component = self.components[step]

            if step == ProcessingStep.LANGUAGE_DETECTION:
                detection_result = component.detect_language(text)
                # Store language detection result in the pipeline result
                if hasattr(self, '_current_result'):
                    self._current_result.detected_language = detection_result.consensus_language
                    self._current_result.language_confidence = detection_result.consensus_confidence
                    self._current_result.language_detection_result = detection_result
                return {
                    'language': detection_result.consensus_language,
                    'confidence': detection_result.consensus_confidence,
                    'detection_result': detection_result
                }

            elif step == ProcessingStep.MULTI_LANGUAGE_PREPROCESSING:
                # Get detected language from previous step
                detected_lang = None
                if 'language_detection' in intermediate_results:
                    detected_lang = intermediate_results['language_detection'].get('language')
                return component.preprocess_text(text, detected_lang)

            elif step == ProcessingStep.LANGUAGE_AWARE_PROCESSING:
                # Get detected language from previous step
                detected_lang = None
                if 'language_detection' in intermediate_results:
                    detected_lang = intermediate_results['language_detection'].get('language')
                return component.process_text(text, detected_lang)

            elif step == ProcessingStep.ADVANCED_PORTUGUESE:
                return component.preprocess_text(text)

            elif step == ProcessingStep.FINANCIAL_PROCESSING:
                return component.process_financial_text(text)

            elif step == ProcessingStep.INTERNATIONAL_FINANCIAL:
                # Get detected language from previous step
                detected_lang = None
                if 'language_detection' in intermediate_results:
                    detected_lang = intermediate_results['language_detection'].get('language')
                return component.process_financial_text(text, detected_lang)

            elif step == ProcessingStep.CONTEXT_AWARE:
                # Get context from previous steps
                context_history = self._extract_context_history(intermediate_results)
                return component.process_with_context(text, context_history)

            elif step == ProcessingStep.QUALITY_ASSESSMENT:
                return self._perform_quality_assessment(text, intermediate_results)

            else:
                raise ValueError(f"Unknown processing step: {step}")

        except Exception as e:
            self.logger.error(f"Error executing step {step.value}: {str(e)}")
            raise

    def _extract_context_history(self, intermediate_results: Dict[str, Any]) -> List[str]:
        """Extract context history from intermediate results"""
        context_texts = []

        # Extract processed texts from previous steps
        for step_result in intermediate_results.values():
            if isinstance(step_result, dict) and 'processed_text' in step_result:
                context_texts.append(step_result['processed_text'])

        return context_texts[-3:]  # Return last 3 texts for context

    def _perform_quality_assessment(self, text: str, intermediate_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform quality assessment on processing results"""
        try:
            quality_scores = {}

            # Assess each step's contribution
            for step_name, step_result in intermediate_results.items():
                if isinstance(step_result, dict):
                    # Extract quality/confidence scores from step results
                    confidence = step_result.get('confidence_score', 0.0)
                    quality = step_result.get('quality_score', confidence)

                    if quality > 0:
                        quality_scores[step_name] = quality

            # Calculate overall quality
            if quality_scores:
                overall_quality = sum(quality_scores.values()) / len(quality_scores)
            else:
                overall_quality = 0.5  # Default quality

            return {
                'quality_scores': quality_scores,
                'overall_quality': overall_quality,
                'assessment_method': 'weighted_average'
            }

        except Exception as e:
            self.logger.warning(f"Error in quality assessment: {str(e)}")
            return {'overall_quality': 0.5}

    def _calculate_quality_metrics(self, result: ProcessingResult) -> Dict[str, float]:
        """Calculate comprehensive quality metrics"""
        try:
            metrics = {}

            # Text length metrics
            original_length = len(result.original_text)
            processed_length = len(result.processed_text)
            metrics['compression_ratio'] = processed_length / original_length if original_length > 0 else 1.0

            # Processing success metrics
            successful_steps = sum(1 for step_result in result.intermediate_results.values()
                                 if isinstance(step_result, dict) and not step_result.get('error'))
            total_steps = len(self.config.steps)
            metrics['step_success_rate'] = successful_steps / total_steps if total_steps > 0 else 0.0

            # Quality from intermediate results
            quality_scores = []
            for step_result in result.intermediate_results.values():
                if isinstance(step_result, dict):
                    quality = step_result.get('quality_score') or step_result.get('confidence_score', 0.0)
                    if quality > 0:
                        quality_scores.append(quality)

            if quality_scores:
                metrics['average_step_quality'] = sum(quality_scores) / len(quality_scores)
            else:
                metrics['average_step_quality'] = 0.5

            # Overall quality (weighted combination)
            metrics['overall_quality'] = (
                metrics['step_success_rate'] * 0.4 +
                metrics['average_step_quality'] * 0.6
            )

            return metrics

        except Exception as e:
            self.logger.warning(f"Error calculating quality metrics: {str(e)}")
            return {'overall_quality': 0.5}

    def process_batch(self, texts: List[str]) -> List[ProcessingResult]:
        """
        Process a batch of texts through the pipeline

        Args:
            texts: List of input texts

        Returns:
            List of ProcessingResult objects
        """
        if not texts:
            return []

        start_time = time.time()

        try:
            if self.config.enable_parallel_processing and len(texts) > 1:
                return self._process_batch_parallel(texts)
            else:
                return self._process_batch_sequential(texts)

        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            return [ProcessingResult(
                original_text=text,
                processed_text=text,
                intermediate_results={},
                quality_metrics={'overall_quality': 0.0},
                processing_time=0.0,
                success=False,
                error_message=str(e)
            ) for text in texts]

        finally:
            batch_time = time.time() - start_time
            self.logger.info(f"Batch processing completed in {batch_time:.2f}s for {len(texts)} texts")

    def _process_batch_sequential(self, texts: List[str]) -> List[ProcessingResult]:
        """Process batch sequentially"""
        results = []
        for text in texts:
            result = self.process_text(text)
            results.append(result)
        return results

    def _process_batch_parallel(self, texts: List[str]) -> List[ProcessingResult]:
        """Process batch in parallel"""
        results = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.process_text, text): i
                for i, text in enumerate(texts)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.logger.error(f"Error processing text at index {index}: {str(e)}")
                    results[index] = ProcessingResult(
                        original_text=texts[index],
                        processed_text=texts[index],
                        intermediate_results={},
                        quality_metrics={'overall_quality': 0.0},
                        processing_time=0.0,
                        success=False,
                        error_message=str(e)
                    )

        return results

    def _update_step_timer(self, step_name: str, execution_time: float):
        """Update step execution time tracking"""
        if step_name not in self._step_timers:
            self._step_timers[step_name] = []

        self._step_timers[step_name].append(execution_time)

        # Keep only last 100 measurements
        if len(self._step_timers[step_name]) > 100:
            self._step_timers[step_name].pop(0)

    def _update_metrics(self, result: ProcessingResult):
        """Update pipeline metrics"""
        self.metrics.total_processed += 1

        if not result.success:
            self.metrics.total_errors += 1

        # Update average processing time
        current_avg = self.metrics.average_processing_time
        self.metrics.average_processing_time = (
            (current_avg * (self.metrics.total_processed - 1)) + result.processing_time
        ) / self.metrics.total_processed

        # Update average quality score
        current_quality_avg = self.metrics.average_quality_score
        quality = result.quality_metrics.get('overall_quality', 0.0)
        self.metrics.average_quality_score = (
            (current_quality_avg * (self.metrics.total_processed - 1)) + quality
        ) / self.metrics.total_processed

        # Update step execution times
        for step_name, times in self._step_timers.items():
            if times:
                self.metrics.step_execution_times[step_name] = sum(times) / len(times)

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics"""
        try:
            metrics_dict = {
                'total_processed': self.metrics.total_processed,
                'total_errors': self.metrics.total_errors,
                'error_rate': self.metrics.total_errors / self.metrics.total_processed if self.metrics.total_processed > 0 else 0.0,
                'average_processing_time': self.metrics.average_processing_time,
                'average_quality_score': self.metrics.average_quality_score,
                'cache_size': len(self._processing_cache),
                'cache_hit_rate': self.metrics.cache_hit_rate,
                'step_execution_times': self.metrics.step_execution_times.copy(),
                'pipeline_uptime': time.time() - self._start_time,
                'configuration': {
                    'steps': [step.value for step in self.config.steps],
                    'batch_size': self.config.batch_size,
                    'max_workers': self.config.max_workers,
                    'parallel_processing': self.config.enable_parallel_processing
                }
            }

            return metrics_dict

        except Exception as e:
            self.logger.error(f"Error getting pipeline metrics: {str(e)}")
            return {}

    def optimize_pipeline(self, sample_texts: List[str]) -> Dict[str, Any]:
        """
        Optimize pipeline configuration based on sample data

        Args:
            sample_texts: Sample texts for optimization

        Returns:
            Optimization recommendations
        """
        try:
            self.logger.info("Starting pipeline optimization")

            # Test different configurations
            configurations = [
                {'batch_size': 16, 'max_workers': 2},
                {'batch_size': 32, 'max_workers': 4},
                {'batch_size': 64, 'max_workers': 8}
            ]

            results = {}

            for config in configurations:
                # Temporarily change configuration
                original_batch_size = self.config.batch_size
                original_max_workers = self.config.max_workers

                self.config.batch_size = config['batch_size']
                self.config.max_workers = config['max_workers']

                # Test performance
                start_time = time.time()
                test_results = self.process_batch(sample_texts[:min(10, len(sample_texts))])
                test_time = time.time() - start_time

                # Calculate metrics
                avg_quality = sum(r.quality_metrics.get('overall_quality', 0.0) for r in test_results) / len(test_results)
                throughput = len(test_results) / test_time

                results[f"batch_{config['batch_size']}_workers_{config['max_workers']}"] = {
                    'throughput': throughput,
                    'average_quality': avg_quality,
                    'processing_time': test_time
                }

                # Restore original configuration
                self.config.batch_size = original_batch_size
                self.config.max_workers = original_max_workers

            # Find best configuration
            best_config = max(results.items(), key=lambda x: x[1]['throughput'])

            optimization_result = {
                'best_configuration': best_config[0],
                'performance_metrics': results,
                'recommendations': {
                    'suggested_batch_size': int(best_config[0].split('_')[1]),
                    'suggested_max_workers': int(best_config[0].split('_')[3])
                }
            }

            self.logger.info(f"Pipeline optimization completed. Best config: {best_config[0]}")
            return optimization_result

        except Exception as e:
            self.logger.error(f"Error optimizing pipeline: {str(e)}")
            return {}

    def clear_cache(self):
        """Clear processing cache"""
        self._processing_cache.clear()
        for component in self.components.values():
            if component and hasattr(component, 'clear_cache'):
                component.clear_cache()
        self.logger.info("Pipeline cache cleared")

    def reset_metrics(self):
        """Reset pipeline metrics"""
        self.metrics = PipelineMetrics()
        self._step_timers = {}
        self._start_time = time.time()
        self.logger.info("Pipeline metrics reset")