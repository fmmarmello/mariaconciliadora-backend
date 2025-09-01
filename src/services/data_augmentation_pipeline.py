import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import random
import re
from collections import defaultdict

# ML and NLP libraries
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.preprocessing import StandardScaler
import torch

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from src.services.portuguese_preprocessor import PortugueseTextPreprocessor

logger = get_logger(__name__)


class AugmentationStrategy(ABC):
    """Abstract base class for augmentation strategies"""

    @abstractmethod
    def augment(self, data: Any, config: Dict[str, Any] = None) -> Any:
        """Apply augmentation to data"""
        pass

    @abstractmethod
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics for the augmentation strategy"""
        pass


class DataAugmentationPipeline:
    """
    Comprehensive data augmentation pipeline for training data expansion
    Supports multiple augmentation strategies for different data types
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data augmentation pipeline

        Args:
            config: Configuration dictionary for augmentation settings
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize augmentation engines
        self.text_engine = None
        self.numerical_engine = None
        self.categorical_engine = None
        self.temporal_engine = None
        self.synthetic_generator = None
        self.quality_controller = None

        # Initialize components
        self._initialize_components()

        # Quality tracking
        self.augmentation_stats = defaultdict(int)
        self.quality_metrics = {}

        self.logger.info("DataAugmentationPipeline initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for data augmentation"""
        return {
            'text_augmentation': {
                'enabled': True,
                'strategies': ['synonym_replacement', 'back_translation', 'paraphrasing'],
                'synonym_config': {
                    'replacement_rate': 0.3,
                    'use_wordnet': True,
                    'domain_specific_dict': True
                },
                'back_translation_config': {
                    'languages': ['en', 'es', 'fr'],
                    'model_name': 'Helsinki-NLP/opus-mt-pt-en'
                },
                'paraphrasing_config': {
                    'model_name': 'tuner007/pegasus_paraphrase',
                    'max_length': 60,
                    'num_beams': 5,
                    'num_return_sequences': 1
                }
            },
            'numerical_augmentation': {
                'enabled': True,
                'strategies': ['gaussian_noise', 'scaling', 'outlier_generation'],
                'noise_config': {
                    'std_multiplier': 0.1,
                    'preserve_distribution': True
                },
                'scaling_config': {
                    'scale_range': (0.8, 1.2),
                    'preserve_zeros': True
                }
            },
            'categorical_augmentation': {
                'enabled': True,
                'strategies': ['label_preservation', 'similar_category_mapping'],
                'mapping_config': {
                    'similarity_threshold': 0.7,
                    'preserve_rare_categories': True
                }
            },
            'temporal_augmentation': {
                'enabled': True,
                'strategies': ['date_shifting', 'pattern_generation'],
                'date_config': {
                    'max_days_shift': 30,
                    'preserve_weekends': True,
                    'business_days_only': False
                }
            },
            'synthetic_generation': {
                'enabled': True,
                'method': 'vae',  # 'vae', 'gan', 'conditional'
                'sample_size_ratio': 0.5,  # Generate 50% of original data size
                'quality_threshold': 0.8
            },
            'quality_control': {
                'enabled': True,
                'statistical_similarity_threshold': 0.9,
                'semantic_preservation_threshold': 0.85,
                'business_rule_compliance': True
            },
            'general': {
                'augmentation_ratio': 2.0,  # Generate 2x the original data
                'random_seed': 42,
                'batch_size': 100,
                'cache_enabled': True
            }
        }

    def _initialize_components(self):
        """Initialize augmentation components"""
        try:
            # Set random seed for reproducibility
            random.seed(self.config['general']['random_seed'])
            np.random.seed(self.config['general']['random_seed'])
            torch.manual_seed(self.config['general']['random_seed'])

            # Initialize text augmentation engine
            if self.config['text_augmentation']['enabled']:
                self.text_engine = TextAugmentationEngine(self.config['text_augmentation'])

            # Initialize numerical augmentation engine
            if self.config['numerical_augmentation']['enabled']:
                self.numerical_engine = NumericalAugmentationEngine(self.config['numerical_augmentation'])

            # Initialize categorical augmentation engine
            if self.config['categorical_augmentation']['enabled']:
                self.categorical_engine = CategoricalAugmentationEngine(self.config['categorical_augmentation'])

            # Initialize temporal augmentation engine
            if self.config['temporal_augmentation']['enabled']:
                self.temporal_engine = TemporalAugmentationEngine(self.config['temporal_augmentation'])

            # Initialize synthetic data generator
            if self.config['synthetic_generation']['enabled']:
                self.synthetic_generator = SyntheticDataGenerator(self.config['synthetic_generation'])

            # Initialize quality control
            if self.config['quality_control']['enabled']:
                self.quality_controller = AugmentationQualityControl(self.config['quality_control'])

            self.logger.info("All augmentation components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing augmentation components: {str(e)}")
            raise ValidationError(f"Failed to initialize DataAugmentationPipeline: {str(e)}")

    def augment_dataset(self, data: Union[List[Dict], pd.DataFrame],
                       data_type: str = 'mixed') -> Tuple[Union[List[Dict], pd.DataFrame], Dict[str, Any]]:
        """
        Apply comprehensive data augmentation to a dataset

        Args:
            data: Input dataset (list of dicts or DataFrame)
            data_type: Type of data ('transaction', 'company_financial', 'mixed')

        Returns:
            Tuple of (augmented_data, augmentation_report)
        """
        try:
            self.logger.info(f"Starting data augmentation for {len(data)} records of type: {data_type}")

            # Convert to DataFrame for processing
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()

            original_size = len(df)
            augmented_data = []

            # Apply different augmentation strategies based on data type
            if data_type in ['transaction', 'mixed']:
                augmented_data.extend(self._augment_transaction_data(df))
            elif data_type == 'company_financial':
                augmented_data.extend(self._augment_company_financial_data(df))

            # Apply synthetic data generation if enabled
            if self.synthetic_generator:
                synthetic_data = self.synthetic_generator.generate_synthetic_data(df)
                if synthetic_data:
                    augmented_data.extend(synthetic_data)
                    self.augmentation_stats['synthetic_generated'] = len(synthetic_data)

            # Quality control
            if self.quality_controller:
                quality_report = self.quality_controller.validate_augmentation(df, augmented_data)
                self.quality_metrics.update(quality_report)

            # Combine original and augmented data
            final_data = pd.concat([df, pd.DataFrame(augmented_data)], ignore_index=True)

            # Generate augmentation report
            augmentation_report = self._generate_augmentation_report(original_size, len(final_data))

            self.logger.info(f"Data augmentation completed. Original: {original_size}, Augmented: {len(final_data)}")
            return final_data, augmentation_report

        except Exception as e:
            self.logger.error(f"Error during data augmentation: {str(e)}")
            raise ValidationError(f"Data augmentation failed: {str(e)}")

    def _augment_transaction_data(self, df: pd.DataFrame) -> List[Dict]:
        """Augment transaction-specific data"""
        augmented_records = []

        try:
            # Text augmentation for descriptions
            if 'description' in df.columns and self.text_engine:
                descriptions = df['description'].fillna('').tolist()
                augmented_descriptions = self.text_engine.augment_batch(descriptions)
                self.augmentation_stats['text_augmented'] += len(augmented_descriptions)

            # Numerical augmentation for amounts
            if 'amount' in df.columns and self.numerical_engine:
                amounts = df['amount'].fillna(0).values
                augmented_amounts = self.numerical_engine.augment_numerical(amounts)
                self.augmentation_stats['numerical_augmented'] += len(augmented_amounts)

            # Categorical augmentation for categories
            if 'category' in df.columns and self.categorical_engine:
                categories = df['category'].fillna('unknown').tolist()
                augmented_categories = self.categorical_engine.augment_categorical(categories)
                self.augmentation_stats['categorical_augmented'] += len(augmented_categories)

            # Temporal augmentation for dates
            if 'date' in df.columns and self.temporal_engine:
                dates = pd.to_datetime(df['date'], errors='coerce').tolist()
                augmented_dates = self.temporal_engine.augment_dates(dates)
                self.augmentation_stats['temporal_augmented'] += len(augmented_dates)

            # Create augmented records by combining different augmentations
            for i in range(len(df)):
                # Create multiple variations of each record
                for _ in range(self.config['general']['augmentation_ratio'] - 1):
                    record = df.iloc[i].copy()

                    # Apply random augmentations
                    if random.random() < 0.7 and 'description' in df.columns:
                        record['description'] = random.choice(augmented_descriptions[i]) if i < len(augmented_descriptions) else record['description']

                    if random.random() < 0.5 and 'amount' in df.columns:
                        record['amount'] = random.choice(augmented_amounts[i]) if i < len(augmented_amounts) else record['amount']

                    if random.random() < 0.3 and 'category' in df.columns:
                        record['category'] = random.choice(augmented_categories[i]) if i < len(augmented_categories) else record['category']

                    if random.random() < 0.4 and 'date' in df.columns:
                        record['date'] = random.choice(augmented_dates[i]) if i < len(augmented_dates) else record['date']

                    augmented_records.append(record.to_dict())

        except Exception as e:
            self.logger.error(f"Error augmenting transaction data: {str(e)}")

        return augmented_records

    def _augment_company_financial_data(self, df: pd.DataFrame) -> List[Dict]:
        """Augment company financial data"""
        # Similar logic but adapted for company financial records
        return self._augment_transaction_data(df)  # For now, use same logic

    def _generate_augmentation_report(self, original_size: int, final_size: int) -> Dict[str, Any]:
        """Generate comprehensive augmentation report"""
        return {
            'original_size': original_size,
            'final_size': final_size,
            'augmentation_ratio': final_size / original_size if original_size > 0 else 0,
            'stats': dict(self.augmentation_stats),
            'quality_metrics': self.quality_metrics,
            'timestamp': datetime.utcnow().isoformat(),
            'config_used': self.config
        }

    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get current augmentation statistics"""
        return {
            'stats': dict(self.augmentation_stats),
            'quality_metrics': self.quality_metrics,
            'config': self.config
        }

    def reset_stats(self):
        """Reset augmentation statistics"""
        self.augmentation_stats.clear()
        self.quality_metrics.clear()
        self.logger.info("Augmentation statistics reset")


# Import the engine classes here to avoid circular imports
from .text_augmentation_engine import TextAugmentationEngine
from .numerical_augmentation_engine import NumericalAugmentationEngine
from .categorical_augmentation_engine import CategoricalAugmentationEngine
from .temporal_augmentation_engine import TemporalAugmentationEngine
from .synthetic_data_generator import SyntheticDataGenerator
from .augmentation_quality_control import AugmentationQualityControl