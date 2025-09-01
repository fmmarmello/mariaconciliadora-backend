#!/usr/bin/env python3
"""
Unit tests for feature engineering components
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.services.feature_engineer import FeatureEngineer
from src.utils.exceptions import ValidationError


@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for testing"""
    base_date = datetime.now() - timedelta(days=365)

    return [
        {
            'description': 'Compra no supermercado Extra',
            'amount': 150.50,
            'date': (base_date + timedelta(days=i*10)).strftime('%Y-%m-%d'),
            'category': 'alimentacao',
            'type': 'debit'
        }
        for i in range(20)
    ]


@pytest.fixture
def sample_text_data():
    """Sample text data for testing"""
    return [
        'Compra no supermercado Extra hiper bom preco',
        'Pagamento de conta de luz CEMIG',
        'Transferência PIX recebida valor R$ 500,00',
        'Restaurante jantar com familia',
        'Combustível posto Ipiranga'
    ]


@pytest.fixture
def sample_date_data():
    """Sample date data for testing"""
    base_date = datetime(2024, 1, 1)
    return [
        (base_date + timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(10)
    ]


class TestFeatureEngineerInitialization:
    """Test FeatureEngineer initialization"""

    def test_initialization_default_config(self):
        """Test initialization with default config"""
        engineer = FeatureEngineer()

        assert engineer.config is not None
        assert 'text_embeddings' in engineer.config
        assert 'temporal_features' in engineer.config
        assert 'transaction_patterns' in engineer.config

    def test_initialization_custom_config(self):
        """Test initialization with custom config"""
        custom_config = {
            'text_embeddings': {
                'model_name': 'custom-model',
                'batch_size': 16
            }
        }

        engineer = FeatureEngineer(custom_config)

        assert engineer.config['text_embeddings']['model_name'] == 'custom-model'
        assert engineer.config['text_embeddings']['batch_size'] == 16

    @patch('src.services.feature_engineer.SentenceTransformer')
    def test_initialization_with_embeddings(self, mock_sentence_transformer):
        """Test initialization with text embeddings"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        config = {
            'text_embeddings': {
                'model_name': 'all-MiniLM-L6-v2'
            }
        }

        engineer = FeatureEngineer(config)

        mock_sentence_transformer.assert_called_once_with('all-MiniLM-L6-v2')
        assert engineer.embedding_model == mock_model

    @patch('src.services.feature_engineer.holidays.CountryHoliday')
    def test_initialization_with_holidays(self, mock_holidays):
        """Test initialization with holiday calendar"""
        mock_calendar = Mock()
        mock_holidays.return_value = mock_calendar

        config = {
            'temporal_features': {
                'include_holidays': True,
                'country': 'BR'
            }
        }

        engineer = FeatureEngineer(config)

        mock_holidays.assert_called_once_with('BR')
        assert engineer.holiday_calendar == mock_calendar


class TestTextEmbeddings:
    """Test text embedding functionality"""

    @patch('src.services.feature_engineer.SentenceTransformer')
    def test_extract_text_embeddings_success(self, mock_sentence_transformer, sample_text_data):
        """Test successful text embedding extraction"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(len(sample_text_data), 384)
        mock_sentence_transformer.return_value = mock_model

        engineer = FeatureEngineer()
        embeddings = engineer.extract_text_embeddings(sample_text_data)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sample_text_data)
        assert embeddings.shape[1] == 384  # Typical embedding dimension

        mock_model.encode.assert_called_once()

    def test_extract_text_embeddings_empty_input(self):
        """Test text embeddings with empty input"""
        engineer = FeatureEngineer()
        embeddings = engineer.extract_text_embeddings([])

        assert embeddings.size == 0
        assert isinstance(embeddings, np.ndarray)

    def test_extract_text_embeddings_with_special_chars(self):
        """Test text embeddings with special characters"""
        engineer = FeatureEngineer()

        texts_with_special = [
            'Compra R$ 150,00 no mercado!',
            'Transferência @ PIX #123',
            'Pagamento conta luz (CEMIG)'
        ]

        # Should not raise exception
        embeddings = engineer.extract_text_embeddings(texts_with_special)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts_with_special)

    @patch('src.services.feature_engineer.SentenceTransformer')
    def test_extract_text_embeddings_error_handling(self, mock_sentence_transformer):
        """Test error handling in text embeddings"""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Embedding error")
        mock_sentence_transformer.return_value = mock_model

        engineer = FeatureEngineer()

        # Should return zero embeddings as fallback
        embeddings = engineer.extract_text_embeddings(sample_text_data)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sample_text_data)
        # Should be zero embeddings due to error
        assert np.all(embeddings == 0)

    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        engineer = FeatureEngineer()

        text = "Compra R$ 150,00 no MERCADO!"
        cleaned = engineer._clean_text(text)

        assert cleaned == "compra r   no mercado"
        assert "$" not in cleaned
        assert "!" not in cleaned
        assert "150,00" not in cleaned  # Numbers removed

    def test_clean_text_edge_cases(self):
        """Test text cleaning edge cases"""
        engineer = FeatureEngineer()

        # Empty text
        assert engineer._clean_text("") == ""

        # None input
        assert engineer._clean_text(None) == ""

        # Only special characters
        assert engineer._clean_text("!@#$%") == ""

        # Only numbers
        assert engineer._clean_text("12345") == ""


class TestTemporalFeatures:
    """Test temporal feature extraction"""

    def test_extract_temporal_features_success(self, sample_date_data):
        """Test successful temporal feature extraction"""
        engineer = FeatureEngineer()
        temporal_df = engineer.extract_temporal_features(sample_date_data)

        assert isinstance(temporal_df, pd.DataFrame)
        assert len(temporal_df) == len(sample_date_data)

        # Check expected columns
        expected_cols = ['year', 'month', 'day', 'weekday', 'month_sin', 'month_cos']
        for col in expected_cols:
            assert col in temporal_df.columns

    def test_extract_temporal_features_empty_input(self):
        """Test temporal features with empty input"""
        engineer = FeatureEngineer()
        temporal_df = engineer.extract_temporal_features([])

        assert isinstance(temporal_df, pd.DataFrame)
        assert len(temporal_df) == 0

    def test_extract_temporal_features_with_business_days(self):
        """Test temporal features with business day features"""
        config = {
            'temporal_features': {
                'include_business_days': True
            }
        }

        engineer = FeatureEngineer(config)
        dates = ['2024-01-01', '2024-01-02', '2024-01-06']  # Mon, Tue, Sat

        temporal_df = engineer.extract_temporal_features(dates)

        assert 'is_business_day' in temporal_df.columns
        assert 'is_weekend' in temporal_df.columns

        # Monday and Tuesday should be business days
        assert temporal_df.iloc[0]['is_business_day'] == True
        assert temporal_df.iloc[1]['is_business_day'] == True
        # Saturday should be weekend
        assert temporal_df.iloc[2]['is_weekend'] == True

    def test_extract_temporal_features_with_seasonal(self):
        """Test temporal features with seasonal features"""
        config = {
            'temporal_features': {
                'include_seasonal': True
            }
        }

        engineer = FeatureEngineer(config)
        dates = ['2024-01-15', '2024-04-15', '2024-07-15', '2024-10-15']

        temporal_df = engineer.extract_temporal_features(dates)

        assert 'season' in temporal_df.columns
        assert 'quarter' in temporal_df.columns
        assert 'is_month_start' in temporal_df.columns
        assert 'is_month_end' in temporal_df.columns

    @patch('src.services.feature_engineer.holidays.CountryHoliday')
    def test_extract_temporal_features_with_holidays(self, mock_holidays, sample_date_data):
        """Test temporal features with holiday features"""
        mock_calendar = Mock()
        mock_calendar.__contains__ = Mock(return_value=True)
        mock_holidays.return_value = mock_calendar

        config = {
            'temporal_features': {
                'include_holidays': True,
                'country': 'BR'
            }
        }

        engineer = FeatureEngineer(config)
        temporal_df = engineer.extract_temporal_features(sample_date_data)

        assert 'is_holiday' in temporal_df.columns
        assert 'days_to_next_holiday' in temporal_df.columns
        assert 'days_since_last_holiday' in temporal_df.columns

    def test_cyclical_encoding(self, sample_date_data):
        """Test cyclical encoding of periodic features"""
        engineer = FeatureEngineer()
        temporal_df = engineer.extract_temporal_features(sample_date_data)

        # Check cyclical encoding columns
        cyclical_cols = ['month_sin', 'month_cos', 'weekday_sin', 'weekday_cos', 'day_sin', 'day_cos']
        for col in cyclical_cols:
            assert col in temporal_df.columns

        # Check that sin/cos values are in valid range [-1, 1]
        for col in cyclical_cols:
            assert temporal_df[col].min() >= -1.0
            assert temporal_df[col].max() <= 1.0

    def test_season_classification(self):
        """Test season classification for Brazilian seasons"""
        engineer = FeatureEngineer()

        # Test different months
        assert engineer._get_season(1) == 'summer'   # January
        assert engineer._get_season(4) == 'autumn'   # April
        assert engineer._get_season(7) == 'winter'   # July
        assert engineer._get_season(10) == 'spring'  # October

    def test_holiday_detection(self):
        """Test holiday detection"""
        config = {
            'temporal_features': {
                'include_holidays': True,
                'country': 'BR'
            }
        }

        with patch('src.services.feature_engineer.holidays.CountryHoliday') as mock_holidays:
            mock_calendar = Mock()
            mock_calendar.__contains__ = Mock(return_value=True)
            mock_holidays.return_value = mock_calendar

            engineer = FeatureEngineer(config)

            test_date = pd.Timestamp('2024-12-25')  # Christmas
            assert engineer._is_holiday(test_date) == True

            mock_calendar.__contains__ = Mock(return_value=False)
            assert engineer._is_holiday(test_date) == False


class TestTransactionPatterns:
    """Test transaction pattern feature extraction"""

    def test_extract_transaction_patterns_success(self, sample_transaction_data):
        """Test successful transaction pattern extraction"""
        engineer = FeatureEngineer()
        pattern_df = engineer.extract_transaction_patterns(sample_transaction_data)

        assert isinstance(pattern_df, pd.DataFrame)
        assert len(pattern_df) == len(sample_transaction_data)

        # Check expected pattern columns
        expected_cols = ['amount_log', 'amount_zscore', 'merchant_frequency']
        for col in expected_cols:
            assert col in pattern_df.columns

    def test_extract_transaction_patterns_empty_input(self):
        """Test transaction patterns with empty input"""
        engineer = FeatureEngineer()
        pattern_df = engineer.extract_transaction_patterns([])

        assert isinstance(pattern_df, pd.DataFrame)
        assert len(pattern_df) == 0

    def test_extract_transaction_patterns_with_amounts(self):
        """Test transaction patterns with amount features"""
        data = [
            {'amount': 100.0, 'description': 'Test 1'},
            {'amount': 200.0, 'description': 'Test 2'},
            {'amount': 50.0, 'description': 'Test 3'}
        ]

        engineer = FeatureEngineer()
        pattern_df = engineer.extract_transaction_patterns(data)

        assert 'amount_bin' in pattern_df.columns
        assert 'is_small_amount' in pattern_df.columns
        assert 'is_medium_amount' in pattern_df.columns
        assert 'is_large_amount' in pattern_df.columns
        assert 'is_round_amount' in pattern_df.columns

    def test_extract_transaction_patterns_with_dates(self):
        """Test transaction patterns with date-based features"""
        base_date = datetime(2024, 1, 1)
        data = [
            {'date': (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
             'amount': 100.0, 'description': f'Test {i}'}
            for i in range(10)
        ]

        engineer = FeatureEngineer()
        pattern_df = engineer.extract_transaction_patterns(data)

        assert 'daily_transaction_freq' in pattern_df.columns

    def test_extract_transaction_patterns_with_merchants(self):
        """Test transaction patterns with merchant analysis"""
        data = [
            {'description': 'Compra no mercado', 'amount': 100.0},
            {'description': 'Compra no mercado', 'amount': 150.0},
            {'description': 'Posto de combustível', 'amount': 200.0}
        ]

        engineer = FeatureEngineer()
        pattern_df = engineer.extract_transaction_patterns(data)

        assert 'merchant_frequency' in pattern_df.columns
        assert 'merchant_diversity' in pattern_df.columns

        # First two should have same merchant
        assert pattern_df.iloc[0]['merchant_frequency'] == pattern_df.iloc[1]['merchant_frequency']

    def test_merchant_extraction(self):
        """Test merchant name extraction"""
        engineer = FeatureEngineer()

        # Test various merchant patterns
        assert engineer._extract_merchant('Compra no MERCADO') == 'MERCADO'
        assert engineer._extract_merchant('Posto Ipiranga') == 'POSTO'
        assert engineer._extract_merchant('Restaurante XYZ') == 'RESTAURANTE'

        # Test fallback
        assert engineer._extract_merchant('Unknown transaction') == 'unknown'


class TestCategoricalFeatures:
    """Test categorical feature extraction"""

    def test_extract_categorical_features_onehot(self):
        """Test one-hot encoding"""
        categories = ['cat_a', 'cat_b', 'cat_a', 'cat_c']

        config = {
            'categorical_features': {
                'encoding_method': 'onehot'
            }
        }

        engineer = FeatureEngineer(config)
        cat_df = engineer.extract_categorical_features(categories)

        assert isinstance(cat_df, pd.DataFrame)
        assert len(cat_df) == len(categories)
        assert cat_df.shape[1] >= 2  # At least 2 columns for 3 categories (drop first)

    @patch('src.services.feature_engineer.TargetEncoder')
    def test_extract_categorical_features_target_encoding(self, mock_target_encoder):
        """Test target encoding"""
        mock_encoder = Mock()
        mock_encoder.fit_transform.return_value = np.random.rand(10, 1)
        mock_target_encoder.return_value = mock_encoder

        categories = ['cat_a', 'cat_b'] * 5
        targets = [0, 1] * 5

        config = {
            'categorical_features': {
                'encoding_method': 'target'
            }
        }

        engineer = FeatureEngineer(config)
        cat_df = engineer.extract_categorical_features(categories, targets)

        assert isinstance(cat_df, pd.DataFrame)
        assert len(cat_df) == len(categories)
        assert cat_df.shape[1] == 1

    def test_extract_categorical_features_label_encoding_fallback(self):
        """Test label encoding fallback"""
        categories = ['cat_a', 'cat_b', 'cat_c']

        config = {
            'categorical_features': {
                'encoding_method': 'invalid_method'
            }
        }

        engineer = FeatureEngineer(config)
        cat_df = engineer.extract_categorical_features(categories)

        assert isinstance(cat_df, pd.DataFrame)
        assert len(cat_df) == len(categories)
        assert cat_df.shape[1] == 1  # Single column for label encoding


class TestFeatureScaling:
    """Test feature scaling functionality"""

    def test_scale_features_standard_scaler(self):
        """Test standard scaling"""
        config = {
            'scaling': {
                'method': 'standard'
            }
        }

        engineer = FeatureEngineer(config)

        features = np.random.rand(20, 5)
        scaled_features, feature_names = engineer.scale_features(features)

        assert isinstance(scaled_features, np.ndarray)
        assert scaled_features.shape == features.shape
        assert feature_names == list(range(5))  # Default names

        # Check that scaling worked (mean close to 0, std close to 1)
        assert abs(np.mean(scaled_features)) < 0.1
        assert abs(np.std(scaled_features) - 1.0) < 0.1

    def test_scale_features_minmax_scaler(self):
        """Test min-max scaling"""
        config = {
            'scaling': {
                'method': 'minmax'
            }
        }

        engineer = FeatureEngineer(config)

        features = np.random.rand(20, 5)
        scaled_features, feature_names = engineer.scale_features(features)

        assert isinstance(scaled_features, np.ndarray)
        assert scaled_features.shape == features.shape

        # Check that values are in [0, 1] range
        assert np.all(scaled_features >= 0)
        assert np.all(scaled_features <= 1)

    def test_scale_features_robust_scaler(self):
        """Test robust scaling"""
        config = {
            'scaling': {
                'method': 'robust'
            }
        }

        engineer = FeatureEngineer(config)

        features = np.random.rand(20, 5)
        scaled_features, feature_names = engineer.scale_features(features)

        assert isinstance(scaled_features, np.ndarray)
        assert scaled_features.shape == features.shape

    def test_scale_features_empty_input(self):
        """Test scaling with empty input"""
        engineer = FeatureEngineer()

        features = np.array([])
        scaled_features, feature_names = engineer.scale_features(features)

        assert scaled_features.size == 0
        assert feature_names == []


class TestFeatureSelection:
    """Test feature selection functionality"""

    def test_select_features_mutual_info(self):
        """Test mutual information feature selection"""
        config = {
            'feature_selection': {
                'method': 'mutual_info',
                'k_features': 3
            }
        }

        engineer = FeatureEngineer(config)

        X = np.random.rand(50, 10)
        y = np.random.choice([0, 1], 50)

        selected_features, selected_names = engineer.select_features(X, y)

        assert isinstance(selected_features, np.ndarray)
        assert selected_features.shape[0] == X.shape[0]
        assert selected_features.shape[1] == 3  # k_features
        assert len(selected_names) == 3

    def test_select_features_f_classif(self):
        """Test F-classif feature selection"""
        config = {
            'feature_selection': {
                'method': 'f_classif',
                'k_features': 2
            }
        }

        engineer = FeatureEngineer(config)

        X = np.random.rand(50, 8)
        y = np.random.choice([0, 1], 50)

        selected_features, selected_names = engineer.select_features(X, y)

        assert selected_features.shape[1] == 2
        assert len(selected_names) == 2

    def test_select_features_empty_input(self):
        """Test feature selection with empty input"""
        engineer = FeatureEngineer()

        X = np.array([])
        y = np.array([])

        selected_features, selected_names = engineer.select_features(X, y)

        assert selected_features.size == 0
        assert selected_names == []


class TestComprehensiveFeatures:
    """Test comprehensive feature engineering"""

    def test_create_comprehensive_features_success(self, sample_transaction_data):
        """Test successful comprehensive feature creation"""
        engineer = FeatureEngineer()
        features, feature_names = engineer.create_comprehensive_features(sample_transaction_data)

        assert isinstance(features, np.ndarray)
        assert isinstance(feature_names, list)
        assert features.shape[0] == len(sample_transaction_data)
        assert len(feature_names) > 0

    def test_create_comprehensive_features_empty_input(self):
        """Test comprehensive features with empty input"""
        engineer = FeatureEngineer()
        features, feature_names = engineer.create_comprehensive_features([])

        assert features.size == 0
        assert feature_names == []

    def test_create_comprehensive_features_with_target_column(self, sample_transaction_data):
        """Test comprehensive features with target column"""
        engineer = FeatureEngineer()
        features, feature_names = engineer.create_comprehensive_features(
            sample_transaction_data, target_column='category'
        )

        assert isinstance(features, np.ndarray)
        assert len(feature_names) > 0

    def test_comprehensive_features_nan_handling(self):
        """Test NaN handling in comprehensive features"""
        data = [
            {'description': 'Test', 'amount': np.nan, 'date': '2024-01-01', 'category': 'test'},
            {'description': 'Test 2', 'amount': 100.0, 'date': '2024-01-02', 'category': 'test'}
        ]

        engineer = FeatureEngineer()
        features, feature_names = engineer.create_comprehensive_features(data)

        assert isinstance(features, np.ndarray)
        assert not np.any(np.isnan(features))  # No NaN values should remain

    def test_comprehensive_features_error_handling(self):
        """Test error handling in comprehensive features"""
        # Invalid data that might cause errors
        data = [
            {'invalid_field': 'value'}
        ]

        engineer = FeatureEngineer()
        features, feature_names = engineer.create_comprehensive_features(data)

        # Should handle gracefully
        assert isinstance(features, np.ndarray)
        assert isinstance(feature_names, list)


class TestFeatureEngineerErrorHandling:
    """Test error handling in FeatureEngineer"""

    def test_initialization_error_handling(self):
        """Test initialization error handling"""
        with patch('src.services.feature_engineer.SentenceTransformer') as mock_st:
            mock_st.side_effect = Exception("Model loading failed")

            # Should still initialize but with warnings
            engineer = FeatureEngineer()

            assert engineer.config is not None

    def test_temporal_features_error_handling(self):
        """Test temporal features error handling"""
        engineer = FeatureEngineer()

        # Invalid dates
        invalid_dates = ['invalid-date', None, '']

        temporal_df = engineer.extract_temporal_features(invalid_dates)

        assert isinstance(temporal_df, pd.DataFrame)
        # Should handle invalid dates gracefully

    def test_transaction_patterns_error_handling(self):
        """Test transaction patterns error handling"""
        engineer = FeatureEngineer()

        # Invalid transaction data
        invalid_data = [
            {'invalid_field': 'value'},
            None,
            {}
        ]

        pattern_df = engineer.extract_transaction_patterns(invalid_data)

        assert isinstance(pattern_df, pd.DataFrame)
        # Should handle invalid data gracefully


class TestFeatureImportance:
    """Test feature importance functionality"""

    def test_get_feature_importance(self):
        """Test getting feature importance information"""
        engineer = FeatureEngineer()

        importance = engineer.get_feature_importance()

        assert isinstance(importance, dict)

        # Should have some importance information even without trained components
        assert 'scaler_mean' in importance or len(importance) == 0