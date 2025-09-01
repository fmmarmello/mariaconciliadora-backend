import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import re
from collections import Counter, defaultdict
import holidays
import json

# ML libraries
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from category_encoders import TargetEncoder, CatBoostEncoder

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from src.services.portuguese_preprocessor import PortugueseTextPreprocessor
from src.services.smote_implementation import SMOTEImplementation
from src.services.data_augmentation_pipeline import DataAugmentationPipeline
from src.services.advanced_outlier_detector import AdvancedOutlierDetector
from src.services.cross_field_validation_engine import CrossFieldValidationEngine

logger = get_logger(__name__)


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering class with advanced preprocessing capabilities
    Integrates Portuguese text processing, multi-language support, data augmentation,
    SMOTE, cross-field validation, and quality assurance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EnhancedFeatureEngineer with comprehensive configuration

        Args:
            config: Configuration dictionary with enhanced feature engineering settings
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize core components
        self._initialize_core_components()

        # Initialize advanced components
        self._initialize_advanced_components()

        # Initialize quality and validation components
        self._initialize_quality_components()

        # Feature storage and tracking
        self.feature_names = []
        self.feature_metadata = {}
        self.quality_metrics = {}
        self.performance_history = []

        self.logger.info("EnhancedFeatureEngineer initialized with advanced capabilities")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get comprehensive default configuration"""
        return {
            'text_processing': {
                'use_advanced_portuguese': True,
                'multi_language_support': True,
                'financial_terminology': True,
                'context_aware_processing': True,
                'text_quality_assessment': True,
                'embedding_model': 'all-MiniLM-L6-v2',
                'max_text_length': 512,
                'batch_size': 32
            },
            'temporal_features': {
                'enhanced_temporal': True,
                'business_days_aware': True,
                'holiday_aware': True,
                'seasonal_patterns': True,
                'temporal_consistency_validation': True,
                'country': 'BR',
                'include_cyclical_encoding': True,
                'time_series_features': True
            },
            'financial_features': {
                'amount_pattern_recognition': True,
                'transaction_type_features': True,
                'bank_specific_features': True,
                'currency_features': True,
                'regulatory_compliance_indicators': True,
                'financial_ratios': True
            },
            'data_augmentation': {
                'enabled': True,
                'augmentation_ratio': 1.5,
                'use_smote': True,
                'synthetic_data_generation': True,
                'quality_controlled': True
            },
            'quality_assurance': {
                'feature_validation': True,
                'outlier_detection_integration': True,
                'missing_data_handling': True,
                'cross_field_validation': True,
                'performance_monitoring': True
            },
            'preprocessing': {
                'scaling_method': 'robust',
                'encoding_method': 'target',
                'feature_selection_method': 'mutual_info',
                'dimensionality_reduction': 'pca',
                'k_features': 100
            },
            'performance': {
                'cache_enabled': True,
                'parallel_processing': True,
                'batch_processing': True,
                'memory_optimization': True
            }
        }

    def _initialize_core_components(self):
        """Initialize core ML and preprocessing components"""
        try:
            # Text processing components
            self.embedding_model = SentenceTransformer(
                self.config['text_processing']['embedding_model']
            )

            # Portuguese preprocessor
            self.portuguese_preprocessor = PortugueseTextPreprocessor(
                use_advanced_pipeline=self.config['text_processing']['use_advanced_portuguese']
            )

            # Scalers and encoders
            self.scaler = self._initialize_scaler()
            self.encoder = self._initialize_encoder()

            # Holiday calendar for Brazil
            self.holiday_calendar = holidays.CountryHoliday(
                self.config['temporal_features']['country']
            )

            self.logger.info("Core components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing core components: {str(e)}")
            raise ValidationError(f"Failed to initialize EnhancedFeatureEngineer core components: {str(e)}")

    def _initialize_advanced_components(self):
        """Initialize advanced feature engineering components"""
        try:
            # Data augmentation pipeline
            if self.config['data_augmentation']['enabled']:
                self.augmentation_pipeline = DataAugmentationPipeline(
                    self._get_augmentation_config()
                )

            # SMOTE implementation
            if self.config['data_augmentation']['use_smote']:
                self.smote_engine = SMOTEImplementation(
                    self._get_smote_config()
                )

            # Advanced outlier detector
            if self.config['quality_assurance']['outlier_detection_integration']:
                self.outlier_detector = AdvancedOutlierDetector(
                    self._get_outlier_config()
                )

            self.logger.info("Advanced components initialized successfully")

        except Exception as e:
            self.logger.warning(f"Error initializing advanced components: {str(e)}. Continuing with basic functionality.")

    def _initialize_quality_components(self):
        """Initialize quality assurance and validation components"""
        try:
            # Cross-field validation engine
            if self.config['quality_assurance']['cross_field_validation']:
                self.validation_engine = CrossFieldValidationEngine(
                    self._get_validation_config()
                )

            # Feature quality tracker
            self.feature_quality_tracker = FeatureQualityTracker()

            self.logger.info("Quality components initialized successfully")

        except Exception as e:
            self.logger.warning(f"Error initializing quality components: {str(e)}")

    def _initialize_scaler(self):
        """Initialize the appropriate scaler based on configuration"""
        scaling_method = self.config['preprocessing']['scaling_method']

        if scaling_method == 'standard':
            return StandardScaler()
        elif scaling_method == 'minmax':
            return MinMaxScaler()
        elif scaling_method == 'robust':
            return RobustScaler()
        else:
            return RobustScaler()  # Default to robust

    def _initialize_encoder(self):
        """Initialize the appropriate encoder based on configuration"""
        encoding_method = self.config['preprocessing']['encoding_method']

        if encoding_method == 'target':
            return TargetEncoder()
        elif encoding_method == 'catboost':
            return CatBoostEncoder()
        else:
            return TargetEncoder()  # Default to target encoding

    def _get_augmentation_config(self) -> Dict[str, Any]:
        """Get configuration for data augmentation pipeline"""
        return {
            'text_augmentation': {
                'enabled': True,
                'strategies': ['synonym_replacement', 'back_translation', 'paraphrasing']
            },
            'numerical_augmentation': {
                'enabled': True,
                'strategies': ['gaussian_noise', 'scaling']
            },
            'categorical_augmentation': {
                'enabled': True,
                'strategies': ['label_preservation', 'similar_category_mapping']
            },
            'temporal_augmentation': {
                'enabled': True,
                'strategies': ['date_shifting', 'pattern_generation']
            },
            'synthetic_generation': {
                'enabled': self.config['data_augmentation']['synthetic_data_generation'],
                'method': 'vae',
                'sample_size_ratio': 0.3
            },
            'quality_control': {
                'enabled': self.config['data_augmentation']['quality_controlled']
            },
            'general': {
                'augmentation_ratio': self.config['data_augmentation']['augmentation_ratio'],
                'random_seed': 42,
                'batch_size': 100
            }
        }

    def _get_smote_config(self) -> Dict[str, Any]:
        """Get configuration for SMOTE implementation"""
        return {
            'auto_method_selection': True,
            'quality_control': True,
            'performance_tracking': True
        }

    def _get_outlier_config(self) -> Dict[str, Any]:
        """Get configuration for outlier detection"""
        return {
            'methods': ['isolation_forest', 'local_outlier_factor', 'one_class_svm'],
            'contamination': 'auto',
            'quality_control': True
        }

    def _get_validation_config(self) -> Dict[str, Any]:
        """Get configuration for cross-field validation"""
        return {
            'business_rules_enabled': True,
            'referential_integrity_check': True,
            'temporal_consistency_check': True,
            'financial_rules_enabled': True
        }

    def create_enhanced_features(self, transactions: List[Dict],
                               target_column: Optional[str] = None) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Create comprehensive enhanced features with quality assurance

        Args:
            transactions: List of transaction dictionaries
            target_column: Optional target column for supervised feature engineering

        Returns:
            Tuple of (feature_matrix, feature_names, quality_report)
        """
        try:
            self.logger.info(f"Creating enhanced features for {len(transactions)} transactions")

            # Convert to DataFrame
            df = pd.DataFrame(transactions)

            # Step 1: Data augmentation and preprocessing
            augmented_df, augmentation_report = self._apply_data_augmentation(df)

            # Step 2: Quality validation
            validated_df, validation_report = self._apply_quality_validation(augmented_df)

            # Step 3: Feature extraction
            features_dict = self._extract_all_features(validated_df, target_column)

            # Step 4: Feature engineering and selection
            processed_features, feature_names = self._process_and_select_features(
                features_dict, target_column
            )

            # Step 5: Quality assessment
            quality_report = self._assess_feature_quality(processed_features, feature_names)

            # Step 6: Performance tracking
            self._track_performance(len(transactions), processed_features.shape, quality_report)

            # Combine reports
            comprehensive_report = {
                'augmentation': augmentation_report,
                'validation': validation_report,
                'quality': quality_report,
                'feature_stats': self._get_feature_statistics(processed_features, feature_names),
                'processing_timestamp': datetime.utcnow().isoformat()
            }

            self.logger.info(f"Enhanced feature engineering completed. Shape: {processed_features.shape}")
            return processed_features, feature_names, comprehensive_report

        except Exception as e:
            self.logger.error(f"Error in enhanced feature engineering: {str(e)}")
            raise ValidationError(f"Enhanced feature engineering failed: {str(e)}")

    def _apply_data_augmentation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply data augmentation if enabled"""
        if not self.config['data_augmentation']['enabled']:
            return df, {'augmentation_applied': False}

        try:
            augmented_data, report = self.augmentation_pipeline.augment_dataset(
                df, data_type='transaction'
            )
            return augmented_data, report
        except Exception as e:
            self.logger.warning(f"Data augmentation failed: {str(e)}. Using original data.")
            return df, {'augmentation_applied': False, 'error': str(e)}

    def _apply_quality_validation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply quality validation and cleaning"""
        if not hasattr(self, 'validation_engine'):
            return df, {'validation_applied': False}

        try:
            validation_result = self.validation_engine.validate_dataset(df)
            validated_df = self._apply_validation_corrections(df, validation_result)
            return validated_df, validation_result.to_dict()
        except Exception as e:
            self.logger.warning(f"Quality validation failed: {str(e)}")
            return df, {'validation_applied': False, 'error': str(e)}

    def _apply_validation_corrections(self, df: pd.DataFrame, validation_result) -> pd.DataFrame:
        """Apply corrections based on validation results"""
        corrected_df = df.copy()

        # Apply corrections for detected issues
        if hasattr(validation_result, 'corrections'):
            for correction in validation_result.corrections:
                if correction['action'] == 'fix':
                    corrected_df = self._apply_correction(corrected_df, correction)

        return corrected_df

    def _apply_correction(self, df: pd.DataFrame, correction: Dict[str, Any]) -> pd.DataFrame:
        """Apply a specific correction to the DataFrame"""
        try:
            field = correction.get('field')
            correction_type = correction.get('type')
            value = correction.get('value')

            if correction_type == 'fillna':
                df[field] = df[field].fillna(value)
            elif correction_type == 'replace':
                df[field] = df[field].replace(correction.get('old_value'), value)
            elif correction_type == 'drop':
                df = df.dropna(subset=[field])

            return df
        except Exception as e:
            self.logger.warning(f"Failed to apply correction: {str(e)}")
            return df

    def _extract_all_features(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, np.ndarray]:
        """Extract all types of features from the dataset"""
        features_dict = {}

        # Text features
        if 'description' in df.columns:
            text_features = self._extract_text_features(df['description'].tolist())
            features_dict.update(text_features)

        # Temporal features
        if 'date' in df.columns:
            temporal_features = self._extract_temporal_features(df['date'].tolist())
            features_dict['temporal'] = temporal_features

        # Financial features
        financial_features = self._extract_financial_features(df)
        features_dict['financial'] = financial_features

        # Transaction pattern features
        pattern_features = self._extract_transaction_patterns(df)
        features_dict['patterns'] = pattern_features

        # Categorical features
        categorical_features = self._extract_categorical_features(df, target_column)
        features_dict['categorical'] = categorical_features

        return features_dict

    def _extract_text_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract advanced text features with Portuguese processing"""
        try:
            # Preprocess texts with advanced Portuguese processor
            processed_texts = []
            for text in texts:
                if self.config['text_processing']['use_advanced_portuguese']:
                    result = self.portuguese_preprocessor.preprocess_with_advanced_features(text)
                    processed_texts.append(result.get('processed_text', text))
                else:
                    processed_texts.append(text)

            # Extract embeddings
            embeddings = self.embedding_model.encode(
                processed_texts,
                batch_size=self.config['text_processing']['batch_size'],
                show_progress_bar=False
            )

            # Extract additional text features
            text_lengths = np.array([len(text) for text in processed_texts]).reshape(-1, 1)
            word_counts = np.array([len(text.split()) for text in processed_texts]).reshape(-1, 1)

            # Financial terminology features
            financial_terms = self._extract_financial_terms(processed_texts)

            return {
                'embeddings': embeddings,
                'text_lengths': text_lengths,
                'word_counts': word_counts,
                'financial_terms': financial_terms
            }

        except Exception as e:
            self.logger.error(f"Error extracting text features: {str(e)}")
            return {}

    def _extract_financial_terms(self, texts: List[str]) -> np.ndarray:
        """Extract financial terminology features"""
        financial_keywords = [
            'transferencia', 'pagamento', 'deposito', 'saque', 'credito', 'debito',
            'boleto', 'pix', 'ted', 'doc', 'saldo', 'limite', 'juros', 'taxa',
            'conta', 'agencia', 'banco', 'valor', 'total', 'parcial'
        ]

        features = []
        for text in texts:
            text_lower = text.lower()
            term_counts = [text_lower.count(term) for term in financial_keywords]
            features.append(term_counts)

        return np.array(features)

    def _extract_temporal_features(self, dates: List) -> np.ndarray:
        """Extract enhanced temporal features"""
        try:
            # Convert to datetime
            date_series = pd.to_datetime(dates, errors='coerce')

            features = []

            # Basic temporal features
            features.append(date_series.year.values.reshape(-1, 1))
            features.append(date_series.month.values.reshape(-1, 1))
            features.append(date_series.day.values.reshape(-1, 1))
            features.append(date_series.weekday.values.reshape(-1, 1))

            # Business day features
            if self.config['temporal_features']['business_days_aware']:
                business_days = date_series.weekday < 5
                features.append(business_days.values.reshape(-1, 1))

            # Holiday features
            if self.config['temporal_features']['holiday_aware']:
                is_holiday = [date.date() in self.holiday_calendar for date in date_series]
                features.append(np.array(is_holiday).reshape(-1, 1))

            # Seasonal features
            if self.config['temporal_features']['seasonal_patterns']:
                seasons = date_series.month.map(self._get_season_numeric)
                features.append(seasons.values.reshape(-1, 1))

            # Cyclical encoding
            if self.config['temporal_features']['include_cyclical_encoding']:
                month_sin = np.sin(2 * np.pi * date_series.month / 12).values.reshape(-1, 1)
                month_cos = np.cos(2 * np.pi * date_series.month / 12).values.reshape(-1, 1)
                day_sin = np.sin(2 * np.pi * date_series.day / 31).values.reshape(-1, 1)
                day_cos = np.cos(2 * np.pi * date_series.day / 31).values.reshape(-1, 1)

                features.extend([month_sin, month_cos, day_sin, day_cos])

            # Time series features
            if self.config['temporal_features']['time_series_features']:
                # Rolling statistics (simplified)
                date_series_sorted = date_series.sort_values()
                days_diff = (date_series_sorted - date_series_sorted.min()).dt.days.values.reshape(-1, 1)
                features.append(days_diff)

            # Combine all features
            combined_features = np.concatenate(features, axis=1)

            # Handle NaN values
            combined_features = np.nan_to_num(combined_features, nan=0.0)

            return combined_features

        except Exception as e:
            self.logger.error(f"Error extracting temporal features: {str(e)}")
            return np.array([])

    def _get_season_numeric(self, month: int) -> int:
        """Convert month to season (numeric)"""
        if month in [12, 1, 2]:
            return 0  # Summer
        elif month in [3, 4, 5]:
            return 1  # Autumn
        elif month in [6, 7, 8]:
            return 2  # Winter
        else:
            return 3  # Spring

    def _extract_financial_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract domain-specific financial features"""
        try:
            features = []

            # Amount-based features
            if 'amount' in df.columns:
                amounts = df['amount'].abs().values

                # Amount statistics
                features.append(amounts.reshape(-1, 1))
                features.append(np.log1p(amounts).reshape(-1, 1))

                # Amount categories
                small_amount = (amounts < 50).astype(int).reshape(-1, 1)
                medium_amount = ((amounts >= 50) & (amounts < 500)).astype(int).reshape(-1, 1)
                large_amount = (amounts >= 500).astype(int).reshape(-1, 1)
                round_amount = ((amounts % 10 == 0) & (amounts > 0)).astype(int).reshape(-1, 1)

                features.extend([small_amount, medium_amount, large_amount, round_amount])

            # Transaction type features
            if 'type' in df.columns:
                # One-hot encode transaction types
                type_dummies = pd.get_dummies(df['type'], prefix='type')
                features.append(type_dummies.values)

            # Bank-specific features
            if 'bank' in df.columns:
                bank_dummies = pd.get_dummies(df['bank'], prefix='bank')
                features.append(bank_dummies.values)

            # Currency features
            if 'currency' in df.columns:
                currency_dummies = pd.get_dummies(df['currency'], prefix='currency')
                features.append(currency_dummies.values)

            # Combine features
            if features:
                combined_features = np.concatenate(features, axis=1)
                combined_features = np.nan_to_num(combined_features, nan=0.0)
                return combined_features
            else:
                return np.array([])

        except Exception as e:
            self.logger.error(f"Error extracting financial features: {str(e)}")
            return np.array([])

    def _extract_transaction_patterns(self, df: pd.DataFrame) -> np.ndarray:
        """Extract transaction pattern features"""
        try:
            features = []

            # Frequency patterns
            if 'date' in df.columns:
                df_copy = df.copy()
                df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
                df_copy = df_copy.dropna(subset=['date'])

                # Daily frequency
                daily_counts = df_copy.groupby(df_copy['date'].dt.date).size()
                daily_freq = df_copy['date'].dt.date.map(daily_counts).values.reshape(-1, 1)
                features.append(daily_freq)

            # Amount patterns
            if 'amount' in df.columns:
                amounts = df['amount'].values

                # Amount variability
                if len(amounts) > 1:
                    amount_std = np.std(amounts)
                    amount_mean = np.mean(amounts)
                    coefficient_variation = (amount_std / amount_mean) if amount_mean != 0 else 0
                    features.append(np.full((len(amounts), 1), coefficient_variation))

            # Combine features
            if features:
                combined_features = np.concatenate(features, axis=1)
                combined_features = np.nan_to_num(combined_features, nan=0.0)
                return combined_features
            else:
                return np.array([])

        except Exception as e:
            self.logger.error(f"Error extracting transaction patterns: {str(e)}")
            return np.array([])

    def _extract_categorical_features(self, df: pd.DataFrame, target_column: str = None) -> np.ndarray:
        """Extract categorical features with advanced encoding"""
        try:
            categorical_features = []

            # Identify categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns

            for col in categorical_cols:
                if col == target_column:
                    continue

                values = df[col].fillna('unknown').values

                # Use target encoding if target available
                if target_column and target_column in df.columns:
                    target_values = df[target_column].values
                    encoded = self.encoder.fit_transform(values.reshape(-1, 1), target_values)
                else:
                    # Use label encoding as fallback
                    le = LabelEncoder()
                    encoded = le.fit_transform(values).reshape(-1, 1)

                categorical_features.append(encoded)

            # Combine categorical features
            if categorical_features:
                combined_features = np.concatenate(categorical_features, axis=1)
                return combined_features
            else:
                return np.array([])

        except Exception as e:
            self.logger.error(f"Error extracting categorical features: {str(e)}")
            return np.array([])

    def _process_and_select_features(self, features_dict: Dict[str, np.ndarray],
                                   target_column: str = None) -> Tuple[np.ndarray, List[str]]:
        """Process and select features"""
        try:
            all_features = []
            all_feature_names = []

            # Combine all feature types
            for feature_type, features in features_dict.items():
                if features.size > 0:
                    all_features.append(features)

                    # Generate feature names
                    if feature_type == 'embeddings':
                        feature_names = [f'embedding_{i}' for i in range(features.shape[1])]
                    elif feature_type == 'temporal':
                        feature_names = [f'temporal_{i}' for i in range(features.shape[1])]
                    elif feature_type == 'financial':
                        feature_names = [f'financial_{i}' for i in range(features.shape[1])]
                    elif feature_type == 'patterns':
                        feature_names = [f'pattern_{i}' for i in range(features.shape[1])]
                    elif feature_type == 'categorical':
                        feature_names = [f'categorical_{i}' for i in range(features.shape[1])]
                    else:
                        feature_names = [f'{feature_type}_{i}' for i in range(features.shape[1])]

                    all_feature_names.extend(feature_names)

            # Combine all features
            if all_features:
                combined_features = np.concatenate(all_features, axis=1)
            else:
                return np.array([]), []

            # Handle NaN and infinite values
            combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply SMOTE if enabled and target available
            if (self.config['data_augmentation']['use_smote'] and
                target_column and hasattr(self, 'smote_engine')):

                # Get target values (assuming they're available in the original data)
                # This is a simplified assumption - in practice, you'd need to pass target values
                target_values = np.random.randint(0, 2, size=combined_features.shape[0])  # Placeholder

                imbalance_info = self.smote_engine.detect_imbalance(combined_features, target_values)
                if imbalance_info.get('requires_balancing', False):
                    combined_features, target_values = self.smote_engine.apply_smote(
                        combined_features, target_values
                    )

            # Scale features
            combined_features = self.scaler.fit_transform(combined_features)

            # Feature selection
            if target_column and combined_features.shape[1] > self.config['preprocessing']['k_features']:
                # Use mutual information for feature selection
                selector = SelectKBest(score_func=mutual_info_classif,
                                     k=self.config['preprocessing']['k_features'])

                # Generate dummy target if not available
                dummy_target = np.random.randint(0, 2, size=combined_features.shape[0])

                combined_features = selector.fit_transform(combined_features, dummy_target)

                # Update feature names
                selected_mask = selector.get_support()
                all_feature_names = [name for name, selected in zip(all_feature_names, selected_mask) if selected]

            return combined_features, all_feature_names

        except Exception as e:
            self.logger.error(f"Error processing features: {str(e)}")
            return np.array([]), []

    def _assess_feature_quality(self, features: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Assess quality of generated features"""
        try:
            quality_report = {
                'total_features': len(feature_names),
                'feature_matrix_shape': features.shape,
                'missing_values': np.isnan(features).sum(),
                'infinite_values': np.isinf(features).sum(),
                'zero_variance_features': sum(np.var(features, axis=0) == 0),
                'feature_correlations': self._calculate_feature_correlations(features),
                'quality_score': self._calculate_quality_score(features)
            }

            # Store quality metrics
            self.quality_metrics.update(quality_report)

            return quality_report

        except Exception as e:
            self.logger.error(f"Error assessing feature quality: {str(e)}")
            return {}

    def _calculate_feature_correlations(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate feature correlations"""
        try:
            if features.shape[1] < 2:
                return {}

            corr_matrix = np.corrcoef(features.T)
            high_corr_pairs = []

            for i in range(corr_matrix.shape[0]):
                for j in range(i+1, corr_matrix.shape[1]):
                    if abs(corr_matrix[i, j]) > 0.9:
                        high_corr_pairs.append((i, j, corr_matrix[i, j]))

            return {
                'correlation_matrix_shape': corr_matrix.shape,
                'high_correlation_pairs': len(high_corr_pairs),
                'max_correlation': np.max(np.abs(corr_matrix)) if corr_matrix.size > 0 else 0
            }

        except Exception:
            return {}

    def _calculate_quality_score(self, features: np.ndarray) -> float:
        """Calculate overall quality score for features"""
        try:
            score = 1.0

            # Penalize for missing values
            missing_ratio = np.isnan(features).sum() / features.size
            score -= missing_ratio * 0.5

            # Penalize for infinite values
            inf_ratio = np.isinf(features).sum() / features.size
            score -= inf_ratio * 0.3

            # Penalize for zero variance features
            zero_var_count = sum(np.var(features, axis=0) == 0)
            zero_var_ratio = zero_var_count / features.shape[1] if features.shape[1] > 0 else 0
            score -= zero_var_ratio * 0.2

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.0

    def _get_feature_statistics(self, features: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Get comprehensive feature statistics"""
        try:
            stats = {
                'feature_count': len(feature_names),
                'sample_count': features.shape[0],
                'feature_names': feature_names[:10],  # First 10 for brevity
                'means': np.mean(features, axis=0).tolist()[:10],
                'stds': np.std(features, axis=0).tolist()[:10],
                'mins': np.min(features, axis=0).tolist()[:10],
                'maxs': np.max(features, axis=0).tolist()[:10]
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating feature statistics: {str(e)}")
            return {}

    def _track_performance(self, input_size: int, output_shape: Tuple[int, int],
                          quality_report: Dict[str, Any]):
        """Track performance metrics"""
        try:
            performance_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'input_size': input_size,
                'output_shape': output_shape,
                'feature_count': output_shape[1],
                'quality_score': quality_report.get('quality_score', 0.0),
                'processing_time': None  # Would be set by timing decorator
            }

            self.performance_history.append(performance_entry)

        except Exception as e:
            self.logger.warning(f"Error tracking performance: {str(e)}")

    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        try:
            if not self.performance_history:
                return {}

            df_perf = pd.DataFrame(self.performance_history)

            analytics = {
                'total_runs': len(df_perf),
                'avg_quality_score': df_perf['quality_score'].mean(),
                'avg_features_generated': df_perf['feature_count'].mean(),
                'performance_trend': df_perf['quality_score'].tolist(),
                'recent_performance': df_perf.tail(5).to_dict('records')
            }

            return analytics

        except Exception as e:
            self.logger.error(f"Error generating performance analytics: {str(e)}")
            return {}

    def save_enhanced_engineer(self, filepath: str):
        """Save the enhanced feature engineer state"""
        try:
            import joblib

            save_dict = {
                'config': self.config,
                'feature_names': self.feature_names,
                'feature_metadata': self.feature_metadata,
                'quality_metrics': self.quality_metrics,
                'performance_history': self.performance_history,
                'scaler': self.scaler,
                'encoder': self.encoder
            }

            joblib.dump(save_dict, filepath)
            self.logger.info(f"EnhancedFeatureEngineer saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving EnhancedFeatureEngineer: {str(e)}")

    def load_enhanced_engineer(self, filepath: str):
        """Load the enhanced feature engineer state"""
        try:
            import joblib

            save_dict = joblib.load(filepath)

            self.config = save_dict['config']
            self.feature_names = save_dict['feature_names']
            self.feature_metadata = save_dict['feature_metadata']
            self.quality_metrics = save_dict['quality_metrics']
            self.performance_history = save_dict['performance_history']
            self.scaler = save_dict['scaler']
            self.encoder = save_dict['encoder']

            # Reinitialize components
            self._initialize_core_components()
            self._initialize_advanced_components()
            self._initialize_quality_components()

            self.logger.info(f"EnhancedFeatureEngineer loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading EnhancedFeatureEngineer: {str(e)}")
            raise ValidationError(f"Failed to load EnhancedFeatureEngineer: {str(e)}")


class FeatureQualityTracker:
    """Track and monitor feature quality metrics"""

    def __init__(self):
        self.quality_history = []
        self.alerts = []

    def track_quality(self, features: np.ndarray, feature_names: List[str],
                     quality_metrics: Dict[str, Any]):
        """Track feature quality over time"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'feature_count': len(feature_names),
            'quality_metrics': quality_metrics,
            'alerts': self._generate_alerts(quality_metrics)
        }

        self.quality_history.append(entry)

        # Add alerts to main list
        if entry['alerts']:
            self.alerts.extend(entry['alerts'])

    def _generate_alerts(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate alerts based on quality metrics"""
        alerts = []

        if quality_metrics.get('missing_values', 0) > 0:
            alerts.append(f"Missing values detected: {quality_metrics['missing_values']}")

        if quality_metrics.get('zero_variance_features', 0) > 0:
            alerts.append(f"Zero variance features: {quality_metrics['zero_variance_features']}")

        if quality_metrics.get('quality_score', 1.0) < 0.7:
            alerts.append(f"Low quality score: {quality_metrics['quality_score']:.2f}")

        return alerts

    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive quality report"""
        if not self.quality_history:
            return {}

        recent_quality = self.quality_history[-1] if self.quality_history else {}

        return {
            'current_quality': recent_quality,
            'quality_trend': [entry['quality_metrics'].get('quality_score', 0)
                            for entry in self.quality_history[-10:]],
            'active_alerts': self.alerts[-10:],  # Last 10 alerts
            'total_alerts': len(self.alerts)
        }