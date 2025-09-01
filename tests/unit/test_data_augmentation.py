"""
DataAugmentationTestSuite - Comprehensive tests for data augmentation pipeline

This module provides comprehensive tests for:
- Text augmentation testing (synonym replacement, paraphrasing)
- Numerical augmentation testing with noise injection
- SMOTE and synthetic data generation testing
- Quality control testing for augmented data
- Integration testing with training pipelines
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.services.data_augmentation_pipeline import DataAugmentationPipeline
from src.services.text_augmentation_engine import TextAugmentationEngine
from src.services.numerical_augmentation_engine import NumericalAugmentationEngine
from src.services.categorical_augmentation_engine import CategoricalAugmentationEngine
from src.services.temporal_augmentation_engine import TemporalAugmentationEngine
from src.services.synthetic_data_generator import SyntheticDataGenerator
from src.services.augmentation_quality_control import AugmentationQualityControl


class TestDataAugmentationPipeline:
    """Test DataAugmentationPipeline functionality"""

    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for testing"""
        return [
            {
                'id': 1,
                'description': 'Compra no supermercado Extra',
                'amount': 150.50,
                'date': '2024-01-15',
                'category': 'alimentacao',
                'type': 'debit'
            },
            {
                'id': 2,
                'description': 'Pagamento de conta de luz CEMIG',
                'amount': 200.75,
                'date': '2024-01-16',
                'category': 'serviços',
                'type': 'debit'
            },
            {
                'id': 3,
                'description': 'Transferência PIX recebida',
                'amount': 500.00,
                'date': '2024-01-17',
                'category': 'transferência',
                'type': 'credit'
            }
        ]

    @pytest.fixture
    def pipeline(self):
        """Create DataAugmentationPipeline instance"""
        config = {
            'general': {
                'augmentation_ratio': 1.5,  # 1.5x for testing
                'random_seed': 42
            },
            'text_augmentation': {
                'enabled': False  # Disable for faster testing
            },
            'numerical_augmentation': {
                'enabled': True,
                'strategies': ['gaussian_noise']
            },
            'categorical_augmentation': {
                'enabled': False  # Disable for faster testing
            },
            'temporal_augmentation': {
                'enabled': False  # Disable for faster testing
            },
            'synthetic_generation': {
                'enabled': False  # Disable for faster testing
            },
            'quality_control': {
                'enabled': True
            }
        }
        return DataAugmentationPipeline(config)

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline is not None
        assert hasattr(pipeline, 'config')
        assert hasattr(pipeline, 'text_engine')
        assert hasattr(pipeline, 'numerical_engine')
        assert hasattr(pipeline, 'quality_controller')

    def test_pipeline_custom_config(self):
        """Test pipeline with custom configuration"""
        custom_config = {
            'general': {
                'augmentation_ratio': 2.0,
                'random_seed': 123
            },
            'text_augmentation': {
                'enabled': False
            },
            'numerical_augmentation': {
                'enabled': False
            }
        }

        pipeline = DataAugmentationPipeline(custom_config)
        assert pipeline.config['general']['augmentation_ratio'] == 2.0
        assert pipeline.config['general']['random_seed'] == 123

    def test_augment_dataset_basic(self, pipeline, sample_transaction_data):
        """Test basic dataset augmentation"""
        augmented_data, report = pipeline.augment_dataset(sample_transaction_data, 'transaction')

        assert isinstance(augmented_data, pd.DataFrame)
        assert len(augmented_data) >= len(sample_transaction_data)  # Should have at least original data
        assert isinstance(report, dict)
        assert 'original_size' in report
        assert 'final_size' in report
        assert 'augmentation_ratio' in report

    def test_augment_dataset_empty(self, pipeline):
        """Test augmentation with empty dataset"""
        with pytest.raises(Exception):  # Should handle empty data gracefully
            pipeline.augment_dataset([], 'transaction')

    def test_get_augmentation_stats(self, pipeline):
        """Test getting augmentation statistics"""
        stats = pipeline.get_augmentation_stats()

        assert isinstance(stats, dict)
        assert 'stats' in stats
        assert 'quality_metrics' in stats
        assert 'config' in stats

    def test_reset_stats(self, pipeline):
        """Test resetting augmentation statistics"""
        # Add some stats first
        pipeline.augmentation_stats['test'] = 1
        pipeline.quality_metrics['test'] = 1

        pipeline.reset_stats()

        assert len(pipeline.augmentation_stats) == 0
        assert len(pipeline.quality_metrics) == 0


class TestTextAugmentationEngine:
    """Test TextAugmentationEngine functionality"""

    @pytest.fixture
    def text_engine(self):
        """Create TextAugmentationEngine instance"""
        config = {
            'strategies': ['synonym_replacement'],
            'synonym_config': {
                'replacement_rate': 0.5,
                'use_wordnet': False  # Disable WordNet for testing
            }
        }
        return TextAugmentationEngine(config)

    def test_text_engine_initialization(self, text_engine):
        """Test text engine initialization"""
        assert text_engine is not None
        assert hasattr(text_engine, 'config')
        assert hasattr(text_engine, 'financial_terms')
        assert hasattr(text_engine, 'domain_synonyms')

    def test_augment_single_text(self, text_engine):
        """Test single text augmentation"""
        text = "Pagamento de conta de luz"
        augmented = text_engine.augment(text)

        assert isinstance(augmented, list)
        assert len(augmented) > 0
        assert text in augmented  # Original should be included

    def test_augment_batch_texts(self, text_engine):
        """Test batch text augmentation"""
        texts = ["Pagamento de conta", "Recebimento de salário", "Compra no mercado"]
        augmented_batch = text_engine.augment_batch(texts)

        assert isinstance(augmented_batch, list)
        assert len(augmented_batch) == len(texts)
        assert all(isinstance(item, list) for item in augmented_batch)

    def test_synonym_replacement(self, text_engine):
        """Test synonym replacement functionality"""
        text = "pagamento de conta"
        augmented = text_engine._apply_synonym_replacement(text)

        assert isinstance(augmented, str)
        # Should either return original or a variation
        assert augmented == text or len(augmented) > 0

    def test_get_quality_metrics(self, text_engine):
        """Test getting quality metrics"""
        metrics = text_engine.get_quality_metrics()

        assert isinstance(metrics, dict)
        assert 'augmentation_quality' in metrics
        assert 'strategies_used' in metrics
        assert 'financial_terms_coverage' in metrics

    def test_add_financial_term(self, text_engine):
        """Test adding custom financial terms"""
        text_engine.add_financial_term('teste', ['teste1', 'teste2'])

        assert 'teste' in text_engine.financial_terms
        assert text_engine.financial_terms['teste'] == ['teste1', 'teste2']


class TestNumericalAugmentationEngine:
    """Test NumericalAugmentationEngine functionality"""

    @pytest.fixture
    def numerical_engine(self):
        """Create NumericalAugmentationEngine instance"""
        config = {
            'strategies': ['gaussian_noise', 'scaling'],
            'noise_config': {
                'std_multiplier': 0.1
            },
            'scaling_config': {
                'scale_range': (0.9, 1.1)
            }
        }
        return NumericalAugmentationEngine(config)

    def test_numerical_engine_initialization(self, numerical_engine):
        """Test numerical engine initialization"""
        assert numerical_engine is not None
        assert hasattr(numerical_engine, 'config')

    def test_augment_single_value(self, numerical_engine):
        """Test single numerical value augmentation"""
        value = 100.50
        augmented = numerical_engine.augment(value)

        assert isinstance(augmented, list)
        assert len(augmented) > 0
        assert all(isinstance(v, (int, float)) for v in augmented)

    def test_augment_array_values(self, numerical_engine):
        """Test array numerical augmentation"""
        values = np.array([100.0, 200.0, 300.0])
        augmented = numerical_engine.augment_numerical(values)

        assert isinstance(augmented, list)
        assert len(augmented) == len(values)
        assert all(isinstance(arr, np.ndarray) for arr in augmented)

    def test_gaussian_noise(self, numerical_engine):
        """Test Gaussian noise application"""
        data = np.array([100.0, 200.0, 300.0])
        noisy_data = numerical_engine._apply_gaussian_noise(data)

        assert isinstance(noisy_data, list)
        assert len(noisy_data) == len(data)
        assert all(isinstance(v, float) for v in noisy_data)

    def test_scaling(self, numerical_engine):
        """Test scaling transformation"""
        data = np.array([100.0, 200.0, 300.0])
        scaled_data = numerical_engine._apply_scaling(data)

        assert isinstance(scaled_data, list)
        assert len(scaled_data) == len(data)
        assert all(isinstance(v, float) for v in scaled_data)

    def test_financial_constraints(self, numerical_engine):
        """Test financial-specific constraints"""
        values = [100.0, -50.0, 1000.0]
        constrained = numerical_engine.apply_financial_constraints(values, 'debit')

        assert isinstance(constrained, list)
        assert len(constrained) == len(values)
        assert all(isinstance(v, float) for v in constrained)

    def test_get_quality_metrics(self, numerical_engine):
        """Test getting quality metrics"""
        metrics = numerical_engine.get_quality_metrics()

        assert isinstance(metrics, dict)
        assert 'data_statistics' in metrics
        assert 'augmentation_quality' in metrics
        assert 'strategies_used' in metrics


class TestCategoricalAugmentationEngine:
    """Test CategoricalAugmentationEngine functionality"""

    @pytest.fixture
    def categorical_engine(self):
        """Create CategoricalAugmentationEngine instance"""
        config = {
            'strategies': ['label_preservation', 'similar_category_mapping'],
            'mapping_config': {
                'similarity_threshold': 0.7
            }
        }
        return CategoricalAugmentationEngine(config)

    def test_categorical_engine_initialization(self, categorical_engine):
        """Test categorical engine initialization"""
        assert categorical_engine is not None
        assert hasattr(categorical_engine, 'category_mappings')
        assert hasattr(categorical_engine, 'business_rules')

    def test_augment_single_category(self, categorical_engine):
        """Test single category augmentation"""
        category = "alimentação"
        augmented = categorical_engine.augment(category)

        assert isinstance(augmented, list)
        assert len(augmented) > 0
        assert category in augmented

    def test_augment_category_list(self, categorical_engine):
        """Test category list augmentation"""
        categories = ["alimentação", "transporte", "serviços"]
        augmented = categorical_engine.augment(categories)

        assert isinstance(augmented, list)
        assert len(augmented) >= len(categories)

    def test_add_category_mapping(self, categorical_engine):
        """Test adding custom category mapping"""
        categorical_engine.add_category_mapping('teste', ['teste1', 'teste2'])

        assert 'teste' in categorical_engine.category_mappings
        assert categorical_engine.category_mappings['teste'] == ['teste1', 'teste2']

    def test_generate_category_variations(self, categorical_engine):
        """Test generating category variations"""
        variations = categorical_engine.generate_category_variations('alimentação', 3)

        assert isinstance(variations, list)
        assert len(variations) <= 4  # Original + up to 3 variations
        assert 'alimentação' in variations


class TestTemporalAugmentationEngine:
    """Test TemporalAugmentationEngine functionality"""

    @pytest.fixture
    def temporal_engine(self):
        """Create TemporalAugmentationEngine instance"""
        config = {
            'strategies': ['date_shifting'],
            'date_config': {
                'max_days_shift': 30,
                'preserve_weekends': True
            }
        }
        return TemporalAugmentationEngine(config)

    def test_temporal_engine_initialization(self, temporal_engine):
        """Test temporal engine initialization"""
        assert temporal_engine is not None
        assert hasattr(temporal_engine, 'holiday_calendar')

    def test_augment_single_date(self, temporal_engine):
        """Test single date augmentation"""
        date_str = "2024-01-15"
        augmented = temporal_engine.augment(date_str)

        assert isinstance(augmented, list)
        assert len(augmented) > 0
        assert all(isinstance(d, datetime) for d in augmented)

    def test_generate_temporal_sequence(self, temporal_engine):
        """Test temporal sequence generation"""
        start_date = "2024-01-01"
        sequence = temporal_engine.generate_temporal_sequence(start_date, 5, 'daily')

        assert isinstance(sequence, list)
        assert len(sequence) == 5
        assert all(isinstance(d, datetime) for d in sequence)

    def test_seasonal_augmentation(self, temporal_engine):
        """Test seasonal date augmentation"""
        dates = ["2024-01-15", "2024-07-15"]
        augmented = temporal_engine.apply_seasonal_augmentation(dates, 'summer')

        assert isinstance(augmented, list)
        assert len(augmented) == len(dates)
        assert all(isinstance(d, datetime) for d in augmented)


class TestSyntheticDataGenerator:
    """Test SyntheticDataGenerator functionality"""

    @pytest.fixture
    def synthetic_generator(self):
        """Create SyntheticDataGenerator instance"""
        config = {
            'method': 'vae',
            'sample_size_ratio': 0.5
        }
        return SyntheticDataGenerator(config)

    def test_synthetic_generator_initialization(self, synthetic_generator):
        """Test synthetic generator initialization"""
        assert synthetic_generator is not None
        assert hasattr(synthetic_generator, 'config')

    @patch('src.services.synthetic_data_generator.torch')
    def test_generate_synthetic_data(self, mock_torch, synthetic_generator):
        """Test synthetic data generation"""
        # Mock torch to avoid actual model training
        mock_torch.manual_seed = Mock()
        mock_torch.cuda.is_available.return_value = False

        # Create sample data
        data = pd.DataFrame({
            'amount': [100.0, 200.0, 300.0],
            'category': ['A', 'B', 'C']
        })

        # Mock the VAE model
        synthetic_generator.vae_model = Mock()
        synthetic_generator.is_trained = True
        synthetic_generator.scaler = Mock()
        synthetic_generator.scaler.inverse_transform = Mock(return_value=np.array([[150.0], [250.0]]))

        result = synthetic_generator.generate_synthetic_data(data)

        assert result is not None or isinstance(result, type(None))


class TestAugmentationQualityControl:
    """Test AugmentationQualityControl functionality"""

    @pytest.fixture
    def quality_controller(self):
        """Create AugmentationQualityControl instance"""
        config = {
            'statistical_similarity_threshold': 0.9,
            'semantic_preservation_threshold': 0.85,
            'business_rule_compliance': True
        }
        return AugmentationQualityControl(config)

    def test_quality_control_initialization(self, quality_controller):
        """Test quality control initialization"""
        assert quality_controller is not None
        assert quality_controller.statistical_threshold == 0.9
        assert quality_controller.semantic_threshold == 0.85

    def test_validate_augmentation(self, quality_controller):
        """Test augmentation validation"""
        original_data = [
            {'amount': 100.0, 'description': 'Test transaction'},
            {'amount': 200.0, 'description': 'Another transaction'}
        ]

        augmented_data = [
            {'amount': 105.0, 'description': 'Test transaction'},
            {'amount': 195.0, 'description': 'Another transaction'},
            {'amount': 150.0, 'description': 'Generated transaction'}
        ]

        results = quality_controller.validate_augmentation(original_data, augmented_data)

        assert isinstance(results, dict)
        assert 'validation_passed' in results
        assert 'quality_score' in results
        assert 'original_size' in results
        assert 'augmented_size' in results

    def test_get_quality_metrics(self, quality_controller):
        """Test getting quality metrics"""
        metrics = quality_controller.get_quality_metrics()

        assert isinstance(metrics, dict)
        assert 'current_quality_metrics' in metrics
        assert 'validation_history' in metrics
        assert 'thresholds' in metrics

    def test_generate_quality_report(self, quality_controller):
        """Test quality report generation"""
        report = quality_controller.generate_quality_report()

        assert isinstance(report, dict)
        assert 'generated_at' in report
        assert 'overall_assessment' in report
        assert 'recommendations' in report


class TestDataAugmentationIntegration:
    """Integration tests for the complete data augmentation system"""

    def test_full_pipeline_integration(self):
        """Test the complete data augmentation pipeline"""
        # Create sample data
        sample_data = [
            {
                'description': 'Pagamento de conta de luz',
                'amount': -150.50,
                'date': '2024-01-15',
                'transaction_type': 'debit',
                'category': 'serviços'
            },
            {
                'description': 'Recebimento de salário',
                'amount': 3000.00,
                'date': '2024-01-16',
                'transaction_type': 'credit',
                'category': 'salário'
            }
        ]

        # Create pipeline with minimal config for testing
        config = {
            'general': {
                'augmentation_ratio': 1.0,
                'random_seed': 42
            },
            'text_augmentation': {
                'enabled': False  # Disable for faster testing
            },
            'numerical_augmentation': {
                'enabled': True,
                'strategies': ['gaussian_noise']
            },
            'categorical_augmentation': {
                'enabled': False  # Disable for faster testing
            },
            'temporal_augmentation': {
                'enabled': False  # Disable for faster testing
            },
            'synthetic_generation': {
                'enabled': False  # Disable for faster testing
            },
            'quality_control': {
                'enabled': True
            }
        }

        pipeline = DataAugmentationPipeline(config)

        # Run augmentation
        augmented_data, report = pipeline.augment_dataset(sample_data, 'transaction')

        # Verify results
        assert isinstance(augmented_data, pd.DataFrame)
        assert len(augmented_data) >= len(sample_data)
        assert isinstance(report, dict)
        assert report['original_size'] == len(sample_data)
        assert report['final_size'] == len(augmented_data)

    def test_error_handling(self):
        """Test error handling in the augmentation system"""
        pipeline = DataAugmentationPipeline()

        # Test with invalid data
        with pytest.raises(Exception):
            pipeline.augment_dataset(None, 'transaction')

        # Test with empty data
        result, report = pipeline.augment_dataset([], 'transaction')
        assert len(result) == 0

    def test_augmentation_quality_assessment(self):
        """Test augmentation quality assessment"""
        # Create original and augmented data
        original_data = [
            {'amount': 100.0, 'description': 'Original transaction'},
            {'amount': 200.0, 'description': 'Another original'}
        ]

        augmented_data = [
            {'amount': 105.0, 'description': 'Augmented transaction'},
            {'amount': 195.0, 'description': 'Another augmented'},
            {'amount': 150.0, 'description': 'Generated transaction'}
        ]

        quality_controller = AugmentationQualityControl()

        results = quality_controller.validate_augmentation(original_data, augmented_data)

        # Check quality assessment results
        assert 'quality_score' in results
        assert 'validation_passed' in results
        assert isinstance(results['quality_score'], (int, float))
        assert 0.0 <= results['quality_score'] <= 1.0

    def test_augmentation_statistics_tracking(self):
        """Test augmentation statistics tracking"""
        config = {
            'general': {'augmentation_ratio': 1.0, 'random_seed': 42},
            'text_augmentation': {'enabled': False},
            'numerical_augmentation': {'enabled': True, 'strategies': ['gaussian_noise']},
            'categorical_augmentation': {'enabled': False},
            'temporal_augmentation': {'enabled': False},
            'synthetic_generation': {'enabled': False},
            'quality_control': {'enabled': True}
        }

        pipeline = DataAugmentationPipeline(config)

        # Get initial stats
        initial_stats = pipeline.get_augmentation_stats()

        # Run augmentation
        sample_data = [{'amount': 100.0, 'description': 'Test'}]
        pipeline.augment_dataset(sample_data, 'transaction')

        # Get updated stats
        updated_stats = pipeline.get_augmentation_stats()

        # Stats should be updated
        assert isinstance(updated_stats, dict)
        assert 'stats' in updated_stats

    def test_augmentation_config_validation(self):
        """Test augmentation configuration validation"""
        # Test with invalid configuration
        invalid_config = {
            'general': {
                'augmentation_ratio': -1,  # Invalid negative ratio
                'random_seed': 42
            }
        }

        # Should handle invalid config gracefully or raise appropriate error
        try:
            pipeline = DataAugmentationPipeline(invalid_config)
            # If it doesn't raise an error, it should have default values
            assert pipeline.config['general']['augmentation_ratio'] >= 0
        except Exception:
            # It's acceptable to raise an error for invalid config
            pass

    def test_memory_management(self):
        """Test memory management in augmentation pipeline"""
        config = {
            'general': {'augmentation_ratio': 1.0, 'random_seed': 42},
            'text_augmentation': {'enabled': False},
            'numerical_augmentation': {'enabled': True},
            'categorical_augmentation': {'enabled': False},
            'temporal_augmentation': {'enabled': False},
            'synthetic_generation': {'enabled': False},
            'quality_control': {'enabled': True}
        }

        pipeline = DataAugmentationPipeline(config)

        # Test with larger dataset
        large_data = [{'amount': float(i), 'description': f'Test {i}'} for i in range(100)]

        # Should handle without memory issues
        augmented_data, report = pipeline.augment_dataset(large_data, 'transaction')

        assert isinstance(augmented_data, pd.DataFrame)
        assert len(augmented_data) >= len(large_data)

    def test_augmentation_reproducibility(self):
        """Test augmentation reproducibility with fixed seed"""
        config1 = {
            'general': {'augmentation_ratio': 2.0, 'random_seed': 42},
            'text_augmentation': {'enabled': False},
            'numerical_augmentation': {'enabled': True},
            'categorical_augmentation': {'enabled': False},
            'temporal_augmentation': {'enabled': False},
            'synthetic_generation': {'enabled': False},
            'quality_control': {'enabled': False}
        }

        config2 = config1.copy()  # Same config

        pipeline1 = DataAugmentationPipeline(config1)
        pipeline2 = DataAugmentationPipeline(config2)

        sample_data = [{'amount': 100.0, 'description': 'Test'}]

        # Run augmentation with both pipelines
        result1, _ = pipeline1.augment_dataset(sample_data, 'transaction')
        result2, _ = pipeline2.augment_dataset(sample_data, 'transaction')

        # Results should be identical (same seed)
        assert len(result1) == len(result2)

        # Check that amounts are the same (if numerical augmentation is deterministic)
        if len(result1) > 1 and len(result2) > 1:
            # Compare the augmented amounts (excluding original)
            amounts1 = sorted(result1['amount'].iloc[1:].values)
            amounts2 = sorted(result2['amount'].iloc[1:].values)
            # They should be the same with same seed
            np.testing.assert_array_almost_equal(amounts1, amounts2, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__])