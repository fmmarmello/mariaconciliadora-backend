import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import re
from collections import Counter, defaultdict
import holidays

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

# Enhanced feature engineering components
from src.services.enhanced_feature_engineer import EnhancedFeatureEngineer
from src.services.advanced_text_feature_extractor import AdvancedTextFeatureExtractor
from src.services.temporal_feature_enhancer import TemporalFeatureEnhancer
from src.services.financial_feature_engineer import FinancialFeatureEngineer
from src.services.quality_assured_feature_pipeline import QualityAssuredFeaturePipeline
from src.services.smote_implementation import SMOTEImplementation
from src.services.data_augmentation_pipeline import DataAugmentationPipeline

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering class for ML models with comprehensive feature extraction
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FeatureEngineer with configuration

        Args:
            config: Configuration dictionary with feature engineering settings
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize components
        self._initialize_components()

        # Initialize advanced Portuguese preprocessor
        self._initialize_text_preprocessor()

        # Initialize enhanced feature engineering components
        self._initialize_enhanced_components()

        # Feature storage
        self.feature_names = []
        self.scalers = {}
        self.encoders = {}
        self.selectors = {}

        # Enhanced feature engineering state
        self.enhanced_features_created = 0
        self.quality_metrics_history = []
        self.performance_analytics = defaultdict(list)

        self.logger.info("FeatureEngineer initialized with enhanced capabilities")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for feature engineering"""
        return {
            'text_embeddings': {
                'model_name': 'all-MiniLM-L6-v2',
                'max_length': 512,
                'batch_size': 32
            },
            'temporal_features': {
                'include_holidays': True,
                'country': 'BR',  # Brazil
                'include_seasonal': True,
                'include_business_days': True
            },
            'transaction_patterns': {
                'amount_bins': [0, 50, 100, 500, 1000, 5000, float('inf')],
                'frequency_window_days': 30,
                'merchant_similarity_threshold': 0.8
            },
            'categorical_features': {
                'encoding_method': 'target',  # 'target', 'catboost', 'onehot'
                'handle_unknown': 'value',
                'unknown_value': -1
            },
            'scaling': {
                'method': 'standard',  # 'standard', 'minmax', 'robust'
                'with_mean': True,
                'with_std': True
            },
            'feature_selection': {
                'method': 'mutual_info',  # 'mutual_info', 'f_classif', 'rfe', 'model_based'
                'k_features': 50,  # int - number of features to select
                'estimator': 'random_forest'
            },
            'dimensionality_reduction': {
                'method': 'pca',  # 'pca', 'tsne', 'umap'
                'n_components': 'auto',  # 'auto' or int - will be set based on data size
                'random_state': 42
            },
            'text_preprocessing': {
                'use_advanced': True,  # Use advanced Portuguese preprocessor
                'batch_processing': True,  # Process texts in batches for better performance
                'cache_enabled': True  # Cache preprocessing results
            }
        }

    def _initialize_components(self):
        """Initialize ML components"""
        try:
            # Text embedding model
            if self.config['text_embeddings']['model_name']:
                self.embedding_model = SentenceTransformer(
                    self.config['text_embeddings']['model_name']
                )
                self.logger.info(f"Loaded SentenceTransformer model: {self.config['text_embeddings']['model_name']}")

            # Holiday calendar
            if self.config['temporal_features']['include_holidays']:
                self.holiday_calendar = holidays.CountryHoliday(
                    self.config['temporal_features']['country']
                )

            # Scalers
            self._initialize_scalers()

            # Feature selectors
            self._initialize_selectors()

        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise ValidationError(f"Failed to initialize FeatureEngineer components: {str(e)}")

    def _initialize_text_preprocessor(self):
        """Initialize the advanced Portuguese text preprocessor"""
        try:
            # Use advanced preprocessing by default
            use_advanced = self.config.get('text_preprocessing', {}).get('use_advanced', True)
            self.text_preprocessor = PortugueseTextPreprocessor(use_advanced_pipeline=use_advanced)

            # Set preprocessing configuration
            self.preprocessing_config = {
                'expand_abbreviations': True,
                'lowercase': True,
                'remove_accents': True,
                'remove_numbers': False,  # Keep numbers for financial data
                'remove_punctuation': True,
                'stopwords': True,
                'stemming': False  # Use lemmatization instead when available
            }

            self.logger.info(f"Text preprocessor initialized (advanced: {use_advanced})")

        except Exception as e:
            self.logger.warning(f"Failed to initialize text preprocessor: {str(e)}. Using basic preprocessing.")
            self.text_preprocessor = None

    def _initialize_enhanced_components(self):
        """Initialize enhanced feature engineering components"""
        try:
            # Enhanced feature engineer
            self.enhanced_engineer = EnhancedFeatureEngineer(self.config)

            # Advanced text feature extractor
            self.advanced_text_extractor = AdvancedTextFeatureExtractor(self.config)

            # Temporal feature enhancer
            self.temporal_enhancer = TemporalFeatureEnhancer(self.config)

            # Financial feature engineer
            self.financial_engineer = FinancialFeatureEngineer(self.config)

            # Quality-assured pipeline
            self.quality_pipeline = QualityAssuredFeaturePipeline(self.config)

            # SMOTE implementation
            self.smote_engine = SMOTEImplementation(self.config)

            # Data augmentation pipeline
            self.augmentation_pipeline = DataAugmentationPipeline(self.config)

            self.logger.info("Enhanced feature engineering components initialized")

        except Exception as e:
            self.logger.warning(f"Failed to initialize enhanced components: {str(e)}. Using basic functionality.")
            # Set to None so basic functionality still works
            self.enhanced_engineer = None
            self.advanced_text_extractor = None
            self.temporal_enhancer = None
            self.financial_engineer = None
            self.quality_pipeline = None
            self.smote_engine = None
            self.augmentation_pipeline = None

    def _initialize_scalers(self):
        """Initialize scaling methods"""
        scaling_config = self.config['scaling']

        if scaling_config['method'] == 'standard':
            self.scaler = StandardScaler(
                with_mean=scaling_config['with_mean'],
                with_std=scaling_config['with_std']
            )
        elif scaling_config['method'] == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_config['method'] == 'robust':
            self.scaler = RobustScaler()

    def _initialize_selectors(self):
        """Initialize feature selection methods"""
        selection_config = self.config['feature_selection']

        if selection_config['method'] == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_classif, k=selection_config['k_features'])
        elif selection_config['method'] == 'f_classif':
            self.selector = SelectKBest(score_func=f_classif, k=selection_config['k_features'])
        elif selection_config['method'] == 'rfe':
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            self.selector = RFE(estimator=estimator, n_features_to_select=selection_config['k_features'])
        elif selection_config['method'] == 'model_based':
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            self.selector = SelectFromModel(estimator=estimator)

    def extract_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Extract text embeddings using Sentence Transformers

        Args:
            texts: List of text strings to embed

        Returns:
            Array of text embeddings
        """
        if not texts:
            return np.array([])

        try:
            self.logger.info(f"Extracting embeddings for {len(texts)} texts")

            # Clean and preprocess texts
            if self.config['text_preprocessing']['batch_processing'] and self.text_preprocessor:
                # Use batch processing with advanced preprocessor
                try:
                    preprocessing_config = {
                        'expand_abbreviations': True,
                        'lowercase': True,
                        'remove_accents': True,
                        'remove_punctuation': True,
                        'stopwords': True
                    }
                    cleaned_texts = self.text_preprocessor.preprocess_batch(texts, preprocessing_config)
                    self.logger.debug(f"Processed {len(texts)} texts using advanced batch preprocessing")
                except Exception as e:
                    self.logger.warning(f"Batch preprocessing failed: {str(e)}. Falling back to individual processing.")
                    cleaned_texts = [self._clean_text(text) for text in texts]
            else:
                cleaned_texts = [self._clean_text(text) for text in texts]

            # Extract embeddings in batches
            embeddings = self.embedding_model.encode(
                cleaned_texts,
                batch_size=self.config['text_embeddings']['batch_size'],
                show_progress_bar=False,
                convert_to_numpy=True
            )

            self.logger.info(f"Extracted embeddings with shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error extracting text embeddings: {str(e)}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), self.embedding_model.get_sentence_embedding_dimension()))

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for embedding using advanced preprocessor"""
        if not text:
            return ""

        try:
            # Use advanced Portuguese preprocessor if available
            if self.text_preprocessor:
                processed_text = self.text_preprocessor.preprocess(text, self.preprocessing_config)
                return processed_text
            else:
                # Fallback to basic preprocessing
                return self._basic_clean_text(text)

        except Exception as e:
            self.logger.warning(f"Error in advanced text cleaning: {str(e)}. Using basic cleaning.")
            return self._basic_clean_text(text)

    def _basic_clean_text(self, text: str) -> str:
        """Basic text cleaning as fallback"""
        if not text:
            return ""

        # Convert to string and lowercase
        text = str(text).lower()

        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Remove numbers (optional - depending on use case)
        text = re.sub(r'\d+', '', text)

        return text.strip()

    def extract_temporal_features(self, dates: List[Union[str, datetime, pd.Timestamp]]) -> pd.DataFrame:
        """
        Extract comprehensive temporal features from dates

        Args:
            dates: List of date values

        Returns:
            DataFrame with temporal features
        """
        if not dates:
            return pd.DataFrame()

        try:
            self.logger.info(f"Extracting temporal features for {len(dates)} dates")

            # Convert to pandas datetime
            date_series = pd.to_datetime(dates, errors='coerce')

            features = {}

            # Basic temporal features
            features['year'] = date_series.year
            features['month'] = date_series.month
            features['day'] = date_series.day
            features['hour'] = date_series.hour.fillna(12)  # Default to noon
            features['weekday'] = date_series.weekday  # 0=Monday, 6=Sunday
            features['day_of_year'] = date_series.dayofyear
            features['week_of_year'] = date_series.isocalendar().week

            # Business day features
            if self.config['temporal_features']['include_business_days']:
                features['is_business_day'] = date_series.weekday < 5
                features['is_weekend'] = date_series.weekday >= 5

            # Seasonal features
            if self.config['temporal_features']['include_seasonal']:
                features['quarter'] = date_series.quarter
                features['is_month_start'] = date_series.is_month_start
                features['is_month_end'] = date_series.is_month_end
                features['is_quarter_start'] = date_series.is_quarter_start
                features['is_quarter_end'] = date_series.is_quarter_end

                # Season classification (Brazilian seasons) - convert to numeric
                season_mapping = {'summer': 0, 'autumn': 1, 'winter': 2, 'spring': 3}
                features['season'] = date_series.month.map(self._get_season).map(season_mapping)

            # Holiday features
            if self.config['temporal_features']['include_holidays']:
                features['is_holiday'] = [self._is_holiday(date) for date in date_series]
                features['days_to_next_holiday'] = [self._days_to_next_holiday(date) for date in date_series]
                features['days_since_last_holiday'] = [self._days_since_last_holiday(date) for date in date_series]

            # Cyclical encoding for periodic features
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            features['weekday_sin'] = np.sin(2 * np.pi * features['weekday'] / 7)
            features['weekday_cos'] = np.cos(2 * np.pi * features['weekday'] / 7)
            features['day_sin'] = np.sin(2 * np.pi * features['day'] / 31)
            features['day_cos'] = np.cos(2 * np.pi * features['day'] / 31)

            # Convert to DataFrame
            temporal_df = pd.DataFrame(features)

            # Fill NaN values
            temporal_df = temporal_df.fillna(0)

            # Ensure no infinite values
            temporal_df = temporal_df.replace([np.inf, -np.inf], 0)

            self.logger.info(f"Extracted {len(temporal_df.columns)} temporal features")
            return temporal_df

        except Exception as e:
            self.logger.error(f"Error extracting temporal features: {str(e)}")
            return pd.DataFrame()

    def _get_season(self, month: int) -> str:
        """Get season based on month (Brazilian seasons)"""
        if month in [12, 1, 2]:
            return 'summer'
        elif month in [3, 4, 5]:
            return 'autumn'
        elif month in [6, 7, 8]:
            return 'winter'
        else:  # 9, 10, 11
            return 'spring'

    def _is_holiday(self, date: pd.Timestamp) -> bool:
        """Check if date is a holiday"""
        try:
            return date.date() in self.holiday_calendar
        except:
            return False

    def _days_to_next_holiday(self, date: pd.Timestamp) -> int:
        """Calculate days to next holiday"""
        try:
            current_date = date.date()
            for days in range(1, 31):  # Look ahead 30 days
                check_date = current_date + timedelta(days=days)
                if check_date in self.holiday_calendar:
                    return days
            return 30  # Default if no holiday found
        except:
            return 30

    def _days_since_last_holiday(self, date: pd.Timestamp) -> int:
        """Calculate days since last holiday"""
        try:
            current_date = date.date()
            for days in range(1, 31):  # Look back 30 days
                check_date = current_date - timedelta(days=days)
                if check_date in self.holiday_calendar:
                    return days
            return 30  # Default if no holiday found
        except:
            return 30

    def extract_transaction_patterns(self, transactions: List[Dict]) -> pd.DataFrame:
        """
        Extract transaction pattern features

        Args:
            transactions: List of transaction dictionaries

        Returns:
            DataFrame with transaction pattern features
        """
        if not transactions:
            return pd.DataFrame()

        try:
            self.logger.info(f"Extracting transaction patterns for {len(transactions)} transactions")

            df = pd.DataFrame(transactions)

            features = {}

            # Amount-based features
            if 'amount' in df.columns:
                amounts = df['amount'].abs()

                # Amount bins
                bins = self.config['transaction_patterns']['amount_bins']
                features['amount_bin'] = pd.cut(amounts, bins=bins, labels=False).fillna(0)

                # Amount statistics
                features['amount_log'] = np.log1p(amounts)
                features['amount_zscore'] = (amounts - amounts.mean()) / amounts.std()
                features['amount_percentile'] = amounts.rank(pct=True)

                # Amount categories
                features['is_small_amount'] = amounts < 50
                features['is_medium_amount'] = (amounts >= 50) & (amounts < 500)
                features['is_large_amount'] = amounts >= 500
                features['is_round_amount'] = (amounts % 10 == 0) & (amounts > 0)

            # Frequency patterns
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                df = df.sort_values('date')  # Ensure dates are sorted

                # Daily transaction frequency
                daily_counts = df.groupby(df['date'].dt.date).size()
                features['daily_transaction_freq'] = df['date'].dt.date.map(daily_counts)

                # Rolling frequency (last N days)
                window_days = self.config['transaction_patterns']['frequency_window_days']
                try:
                    # Use a simpler approach for rolling frequency
                    df_sorted = df.sort_values('date')
                    rolling_freq = df_sorted.set_index('date').rolling(f'{window_days}D').count()['amount']
                    features['rolling_freq'] = rolling_freq.values
                except Exception as e:
                    self.logger.warning(f"Rolling frequency calculation failed: {str(e)}")
                    features['rolling_freq'] = [1] * len(df)  # Default to 1

            # Merchant analysis
            if 'description' in df.columns:
                # Extract merchant names (simplified)
                merchants = df['description'].apply(self._extract_merchant)
                features['merchant_frequency'] = merchants.map(merchants.value_counts())
                features['merchant_diversity'] = merchants.nunique() / len(df)

                # Merchant similarity (simplified clustering)
                merchant_embeddings = self.extract_text_embeddings(merchants.tolist())
                if len(merchant_embeddings) > 0:
                    from sklearn.cluster import KMeans
                    n_clusters = min(10, len(merchants.unique()))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    features['merchant_cluster'] = kmeans.fit_predict(merchant_embeddings)

            # Transaction type patterns
            if 'type' in df.columns:
                type_counts = df['type'].value_counts()
                features['transaction_type_freq'] = df['type'].map(type_counts)
                features['is_common_type'] = df['type'].map(type_counts) > type_counts.median()

            # Convert to DataFrame
            pattern_df = pd.DataFrame(features)

            # Fill NaN values
            pattern_df = pattern_df.fillna(0)

            # Ensure no infinite values
            pattern_df = pattern_df.replace([np.inf, -np.inf], 0)

            self.logger.info(f"Extracted {len(pattern_df.columns)} transaction pattern features")
            return pattern_df

        except Exception as e:
            self.logger.error(f"Error extracting transaction patterns: {str(e)}")
            return pd.DataFrame()

    def _extract_merchant(self, description: str) -> str:
        """Extract merchant name from transaction description"""
        if not description:
            return "unknown"

        # Simple merchant extraction (can be improved with ML)
        description = str(description).upper()

        # Common merchant patterns
        merchants = ['MERCADO', 'PADARIA', 'FARMACIA', 'POSTO', 'SHOPPING', 'RESTAURANTE']

        for merchant in merchants:
            if merchant in description:
                return merchant

        # Return first word as fallback
        words = description.split()
        return words[0] if words else "unknown"

    def extract_categorical_features(self, categories: List[str], target: Optional[List] = None) -> pd.DataFrame:
        """
        Extract categorical features with various encoding methods

        Args:
            categories: List of categorical values
            target: Target values for supervised encoding (optional)

        Returns:
            DataFrame with encoded categorical features
        """
        if not categories:
            return pd.DataFrame()

        try:
            self.logger.info(f"Extracting categorical features for {len(categories)} items")

            encoding_method = self.config['categorical_features']['encoding_method']

            if encoding_method == 'onehot':
                encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown='ignore',
                    drop='first'  # Avoid multicollinearity
                )
                encoded = encoder.fit_transform(np.array(categories).reshape(-1, 1))
                feature_names = [f"cat_{i}" for i in range(encoded.shape[1])]

            elif encoding_method == 'target' and target is not None:
                encoder = TargetEncoder(
                    handle_unknown='value',
                    handle_missing='value'
                )
                encoded = encoder.fit_transform(np.array(categories).reshape(-1, 1), target)
                feature_names = ['target_encoded']

            elif encoding_method == 'catboost' and target is not None:
                encoder = CatBoostEncoder(
                    handle_unknown='value',
                    handle_missing='value'
                )
                encoded = encoder.fit_transform(np.array(categories).reshape(-1, 1), target)
                feature_names = ['catboost_encoded']

            else:
                # Fallback to label encoding
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(categories).reshape(-1, 1)
                feature_names = ['label_encoded']

            # Convert to DataFrame
            cat_df = pd.DataFrame(encoded, columns=feature_names)

            # Store encoder for later use
            self.encoders['categorical'] = encoder

            self.logger.info(f"Extracted {len(cat_df.columns)} categorical features using {encoding_method}")
            return cat_df

        except Exception as e:
            self.logger.error(f"Error extracting categorical features: {str(e)}")
            return pd.DataFrame()

    def scale_features(self, features: np.ndarray, feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Scale/normalize features

        Args:
            features: Feature matrix
            feature_names: Names of features

        Returns:
            Tuple of (scaled_features, feature_names)
        """
        if features.size == 0:
            return features, feature_names or []

        try:
            self.logger.info(f"Scaling features with shape: {features.shape}")

            # Fit and transform
            scaled_features = self.scaler.fit_transform(features)

            # Store scaler
            self.scalers['main'] = self.scaler

            self.logger.info("Feature scaling completed")
            return scaled_features, feature_names

        except Exception as e:
            self.logger.error(f"Error scaling features: {str(e)}")
            return features, feature_names

    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Perform feature selection

        Args:
            X: Feature matrix
            y: Target values
            feature_names: Names of features

        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        if X.size == 0 or len(y) == 0:
            return X, feature_names or []

        try:
            self.logger.info(f"Selecting features from {X.shape[1]} features")

            # Fit selector
            self.selector.fit(X, y)

            # Transform features
            selected_features = self.selector.transform(X)

            # Get selected feature names
            if hasattr(self.selector, 'get_support'):
                support = self.selector.get_support()
                if feature_names:
                    selected_names = [name for name, selected in zip(feature_names, support) if selected]
                else:
                    selected_names = [f"feature_{i}" for i, selected in enumerate(support) if selected]
            else:
                selected_names = feature_names or [f"feature_{i}" for i in range(selected_features.shape[1])]

            # Store selector
            self.selectors['main'] = self.selector

            self.logger.info(f"Selected {selected_features.shape[1]} features")
            return selected_features, selected_names

        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            return X, feature_names

    def reduce_dimensionality(self, features: np.ndarray, feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Perform dimensionality reduction

        Args:
            features: Feature matrix
            feature_names: Names of features

        Returns:
            Tuple of (reduced_features, component_names)
        """
        if features.size == 0:
            return features, feature_names or []

        try:
            method = self.config['dimensionality_reduction']['method']
            n_components_config = self.config['dimensionality_reduction']['n_components']

            # Determine n_components based on data size
            if n_components_config == 'auto':
                n_samples, n_features = features.shape
                n_components = min(50, n_features // 2, n_samples - 1)
                n_components = max(2, n_components)  # At least 2 components
            else:
                n_components = n_components_config

            self.logger.info(f"Reducing dimensionality from {features.shape[1]} to {n_components}")

            if method == 'pca':
                reducer = PCA(n_components=n_components, random_state=42)
            elif method == 'tsne':
                reducer = TSNE(n_components=min(n_components, 3), random_state=42)  # t-SNE limited to 3 components
            elif method == 'umap':
                reducer = umap.UMAP(n_components=n_components, random_state=42)
            else:
                self.logger.warning(f"Unknown dimensionality reduction method: {method}")
                return features, feature_names

            # Fit and transform
            reduced_features = reducer.fit_transform(features)

            # Generate component names
            component_names = [f"{method.upper()}_{i+1}" for i in range(reduced_features.shape[1])]

            self.logger.info(f"Dimensionality reduction completed. New shape: {reduced_features.shape}")
            return reduced_features, component_names

        except Exception as e:
            self.logger.error(f"Error reducing dimensionality: {str(e)}")
            return features, feature_names

    def create_comprehensive_features(self, transactions: List[Dict], target_column: str = None) -> Tuple[np.ndarray, List[str]]:
        """
        Create comprehensive feature set combining all feature types

        Args:
            transactions: List of transaction dictionaries
            target_column: Name of target column for supervised encoding

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not transactions:
            return np.array([]), []

        try:
            self.logger.info(f"Creating comprehensive features for {len(transactions)} transactions")

            df = pd.DataFrame(transactions)
            all_features = []
            all_feature_names = []

            # 1. Text embeddings
            if 'description' in df.columns:
                text_embeddings = self.extract_text_embeddings(df['description'].tolist())
                if text_embeddings.size > 0:
                    all_features.append(text_embeddings)
                    embedding_dim = text_embeddings.shape[1]
                    all_feature_names.extend([f"embedding_{i}" for i in range(embedding_dim)])

            # 2. Temporal features
            if 'date' in df.columns:
                temporal_features = self.extract_temporal_features(df['date'].tolist())
                if not temporal_features.empty:
                    temporal_array = temporal_features.values
                    all_features.append(temporal_array)
                    all_feature_names.extend(temporal_features.columns.tolist())

            # 3. Transaction pattern features
            pattern_features = self.extract_transaction_patterns(transactions)
            if not pattern_features.empty:
                pattern_array = pattern_features.values
                all_features.append(pattern_array)
                all_feature_names.extend(pattern_features.columns.tolist())

            # 4. Categorical features
            target_values = None
            if target_column and target_column in df.columns:
                target_values = df[target_column].values

            categorical_features_list = []
            if 'category' in df.columns:
                categorical_features_list.append(('category', df['category'].tolist()))
            if 'type' in df.columns:
                categorical_features_list.append(('type', df['type'].tolist()))

            for cat_name, cat_values in categorical_features_list:
                cat_features = self.extract_categorical_features(cat_values, target_values)
                if not cat_features.empty:
                    cat_array = cat_features.values
                    all_features.append(cat_array)
                    all_feature_names.extend([f"{cat_name}_{name}" for name in cat_features.columns])

            # Combine all features
            if all_features:
                combined_features = np.concatenate(all_features, axis=1)
            else:
                combined_features = np.array([])

            # Handle NaN and infinite values before scaling
            if combined_features.size > 0:
                combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)

            # 5. Feature scaling
            if combined_features.size > 0:
                combined_features, all_feature_names = self.scale_features(combined_features, all_feature_names)

            # 6. Feature selection (if target available)
            if target_values is not None and combined_features.size > 0:
                # Ensure no NaN values before feature selection
                if np.any(np.isnan(combined_features)):
                    self.logger.warning("NaN values found before feature selection, filling with 0")
                    combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)

                combined_features, all_feature_names = self.select_features(
                    combined_features, target_values, all_feature_names
                )

            # 7. Dimensionality reduction (optional)
            n_components_config = self.config['dimensionality_reduction']['n_components']
            if (isinstance(n_components_config, int) and n_components_config < len(all_feature_names) and
                combined_features.size > 0):
                combined_features, all_feature_names = self.reduce_dimensionality(
                    combined_features, all_feature_names
                )

            self.logger.info(f"Comprehensive feature engineering completed. Final shape: {combined_features.shape}")
            return combined_features, all_feature_names

        except Exception as e:
            self.logger.error(f"Error creating comprehensive features: {str(e)}")
            return np.array([]), []

    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance information from trained components

        Returns:
            Dictionary with feature importance information
        """
        importance_info = {}

        try:
            # Scaler information
            if hasattr(self.scaler, 'mean_'):
                importance_info['scaler_mean'] = self.scaler.mean_.tolist()
            if hasattr(self.scaler, 'scale_'):
                importance_info['scaler_scale'] = self.scaler.scale_.tolist()

            # Feature selection information
            if hasattr(self.selector, 'scores_'):
                importance_info['selector_scores'] = self.selector.scores_.tolist()
            if hasattr(self.selector, 'feature_importances_'):
                importance_info['feature_importances'] = self.selector.feature_importances_.tolist()

            # Encoder information
            for name, encoder in self.encoders.items():
                if hasattr(encoder, 'classes_'):
                    importance_info[f'{name}_classes'] = encoder.classes_.tolist()

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")

        return importance_info

    def save_feature_engineer(self, filepath: str):
        """
        Save the feature engineer state (simplified - in production use joblib/pickle)

        Args:
            filepath: Path to save the feature engineer
        """
        try:
            import joblib

            # Create a serializable version
            save_dict = {
                'config': self.config,
                'feature_names': self.feature_names,
                'scaler': self.scalers.get('main'),
                'selector': self.selectors.get('main'),
                'encoders': self.encoders
            }

            joblib.dump(save_dict, filepath)
            self.logger.info(f"FeatureEngineer saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving FeatureEngineer: {str(e)}")

    def load_feature_engineer(self, filepath: str):
        """
        Load the feature engineer state

        Args:
            filepath: Path to load the feature engineer from
        """
        try:
            import joblib

            save_dict = joblib.load(filepath)

            self.config = save_dict['config']
            self.feature_names = save_dict['feature_names']
            self.scalers['main'] = save_dict['scaler']
            self.selectors['main'] = save_dict['selector']
            self.encoders = save_dict['encoders']

            # Reinitialize components
            self._initialize_components()

            self.logger.info(f"FeatureEngineer loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading FeatureEngineer: {str(e)}")
            raise ValidationError(f"Failed to load FeatureEngineer: {str(e)}")

    # Enhanced Methods using Advanced Components

    def create_enhanced_features(self, transactions: List[Dict],
                               target_column: Optional[str] = None,
                               use_quality_pipeline: bool = True) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Create enhanced features using advanced components with quality assurance

        Args:
            transactions: List of transaction dictionaries
            target_column: Optional target column for supervised feature engineering
            use_quality_pipeline: Whether to use the quality-assured pipeline

        Returns:
            Tuple of (feature_matrix, feature_names, quality_report)
        """
        if self.enhanced_engineer and use_quality_pipeline and self.quality_pipeline:
            # Use quality-assured pipeline
            result = self.quality_pipeline.process_dataset(transactions, target_column)
            if result.get('success'):
                features = np.array(result['features'])
                feature_names = result['feature_names']
                quality_report = result['quality_report']
                return features, feature_names, quality_report
            else:
                self.logger.warning("Quality pipeline failed, falling back to enhanced engineer")

        if self.enhanced_engineer:
            # Use enhanced feature engineer directly
            return self.enhanced_engineer.create_enhanced_features(transactions, target_column)
        else:
            # Fall back to original method
            return self.create_comprehensive_features(transactions, target_column), [], {}

    def extract_advanced_text_features(self, texts: List[str],
                                     feature_types: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Extract advanced text features using specialized text processor

        Args:
            texts: List of input texts
            feature_types: Types of features to extract

        Returns:
            Dictionary of extracted text features
        """
        if self.advanced_text_extractor:
            return self.advanced_text_extractor.extract_text_features(texts, feature_types)
        else:
            # Fall back to basic text processing
            return {'embeddings': self.extract_text_embeddings(texts)}

    def extract_enhanced_temporal_features(self, dates: List[Union[str, datetime, pd.Timestamp]],
                                         context_data: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Extract enhanced temporal features with business awareness

        Args:
            dates: List of date values
            context_data: Additional context data

        Returns:
            Tuple of (temporal_features, feature_names)
        """
        if self.temporal_enhancer:
            return self.temporal_enhancer.extract_temporal_features(dates, context_data)
        else:
            # Fall back to basic temporal features
            return self.extract_temporal_features(dates), []

    def extract_domain_financial_features(self, transactions: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract domain-specific financial features

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Tuple of (financial_features, feature_names)
        """
        if self.financial_engineer:
            return self.financial_engineer.extract_financial_features(transactions)
        else:
            # Fall back to basic transaction patterns
            return self.extract_transaction_patterns(transactions), []

    def apply_data_augmentation(self, data: Union[List[Dict], pd.DataFrame],
                              data_type: str = 'transaction',
                              augmentation_config: Optional[Dict[str, Any]] = None) -> Tuple[Union[List[Dict], pd.DataFrame], Dict[str, Any]]:
        """
        Apply data augmentation techniques to expand training data

        Args:
            data: Input data to augment
            data_type: Type of data ('transaction', 'company_financial', 'mixed')
            augmentation_config: Configuration for augmentation

        Returns:
            Tuple of (augmented_data, augmentation_report)
        """
        if self.augmentation_pipeline:
            if augmentation_config:
                self.augmentation_pipeline.config.update(augmentation_config)
            return self.augmentation_pipeline.augment_dataset(data, data_type)
        else:
            self.logger.warning("Data augmentation pipeline not available")
            return data, {'augmentation_applied': False}

    def apply_smote_balancing(self, X: np.ndarray, y: np.ndarray,
                            method: str = 'auto', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE for dataset balancing

        Args:
            X: Feature matrix
            y: Target labels
            method: SMOTE method to use
            **kwargs: Additional parameters

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if self.smote_engine:
            return self.smote_engine.apply_smote(X, y, method=method, **kwargs)
        else:
            self.logger.warning("SMOTE engine not available")
            return X, y

    def get_enhanced_quality_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive quality metrics from enhanced components

        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            'basic_metrics': self.get_feature_importance(),
            'enhanced_features_created': self.enhanced_features_created,
            'quality_history': self.quality_metrics_history[-10:],  # Last 10 entries
        }

        # Add metrics from enhanced components if available
        if self.quality_pipeline:
            metrics['pipeline_metrics'] = self.quality_pipeline.get_quality_metrics()

        if self.advanced_text_extractor:
            metrics['text_extraction_stats'] = self.advanced_text_extractor.get_extraction_stats()
            metrics['text_quality_metrics'] = self.advanced_text_extractor.get_quality_metrics()

        if self.temporal_enhancer:
            metrics['temporal_stats'] = self.temporal_enhancer.get_temporal_stats()

        if self.financial_engineer:
            metrics['financial_stats'] = self.financial_engineer.get_feature_stats()

        return metrics

    def get_performance_analytics(self) -> Dict[str, Any]:
        """
        Get performance analytics for feature engineering operations

        Returns:
            Dictionary with performance analytics
        """
        analytics = dict(self.performance_analytics)

        # Add analytics from enhanced components
        if self.enhanced_engineer:
            analytics['enhanced_engineer'] = self.enhanced_engineer.get_performance_analytics()

        if self.quality_pipeline:
            analytics['quality_pipeline'] = self.quality_pipeline.get_quality_metrics()

        return analytics

    def validate_feature_quality(self, features: np.ndarray,
                               feature_names: List[str]) -> Dict[str, Any]:
        """
        Validate quality of generated features

        Args:
            features: Feature matrix
            feature_names: Feature names

        Returns:
            Dictionary with quality validation results
        """
        quality_report = {
            'total_features': len(feature_names),
            'feature_matrix_shape': features.shape,
            'quality_checks': []
        }

        # Basic quality checks
        if features.size == 0:
            quality_report['quality_checks'].append({
                'check': 'feature_matrix_empty',
                'status': 'failed',
                'message': 'No features generated'
            })
        else:
            quality_report['quality_checks'].append({
                'check': 'feature_matrix_exists',
                'status': 'passed',
                'message': f'Generated {features.shape[0]} samples with {features.shape[1]} features'
            })

        # Check for NaN values
        nan_count = np.isnan(features).sum()
        if nan_count > 0:
            quality_report['quality_checks'].append({
                'check': 'nan_values',
                'status': 'warning',
                'message': f'Found {nan_count} NaN values in feature matrix'
            })
        else:
            quality_report['quality_checks'].append({
                'check': 'nan_values',
                'status': 'passed',
                'message': 'No NaN values found'
            })

        # Check for infinite values
        inf_count = np.isinf(features).sum()
        if inf_count > 0:
            quality_report['quality_checks'].append({
                'check': 'infinite_values',
                'status': 'warning',
                'message': f'Found {inf_count} infinite values in feature matrix'
            })

        # Check for zero variance features
        if features.shape[1] > 0:
            variances = np.var(features, axis=0)
            zero_var_count = np.sum(variances == 0)
            if zero_var_count > 0:
                quality_report['quality_checks'].append({
                    'check': 'zero_variance_features',
                    'status': 'warning',
                    'message': f'Found {zero_var_count} features with zero variance'
                })

        # Overall quality score
        passed_checks = sum(1 for check in quality_report['quality_checks'] if check['status'] == 'passed')
        total_checks = len(quality_report['quality_checks'])
        quality_report['overall_quality_score'] = passed_checks / total_checks if total_checks > 0 else 0

        return quality_report

    def enable_enhanced_mode(self):
        """Enable enhanced feature engineering mode"""
        if self.enhanced_engineer:
            self.logger.info("Enhanced feature engineering mode already enabled")
        else:
            try:
                self._initialize_enhanced_components()
                self.logger.info("Enhanced feature engineering mode enabled")
            except Exception as e:
                self.logger.error(f"Failed to enable enhanced mode: {str(e)}")

    def disable_enhanced_mode(self):
        """Disable enhanced feature engineering mode"""
        self.enhanced_engineer = None
        self.advanced_text_extractor = None
        self.temporal_enhancer = None
        self.financial_engineer = None
        self.quality_pipeline = None
        self.smote_engine = None
        self.augmentation_pipeline = None
        self.logger.info("Enhanced feature engineering mode disabled")

    def clear_enhanced_cache(self):
        """Clear caches of enhanced components"""
        try:
            if self.enhanced_engineer:
                self.enhanced_engineer.clear_cache()
            if self.advanced_text_extractor:
                self.advanced_text_extractor.clear_cache()
            if self.temporal_enhancer:
                self.temporal_enhancer.clear_cache()
            if self.financial_engineer:
                self.financial_engineer.clear_cache()
            if self.quality_pipeline:
                self.quality_pipeline.clear_cache()

            self.logger.info("Enhanced component caches cleared")

        except Exception as e:
            self.logger.warning(f"Error clearing enhanced caches: {str(e)}")