"""
FeatureEngineeringTestSuite - Comprehensive tests for enhanced feature engineering

This module provides comprehensive tests for:
- Advanced text feature extraction testing
- Temporal feature enhancement testing
- Financial feature engineering testing
- Quality assurance pipeline testing
- Integration testing with ML pipelines
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.services.enhanced_feature_engineer import (
    EnhancedFeatureEngineer,
    FeatureQualityTracker
)


class TestEnhancedFeatureEngineer:
    """Test EnhancedFeatureEngineer functionality"""

    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for testing"""
        return [
            {
                'id': 1,
                'description': 'Compra no supermercado Extra hiper bom preco',
                'amount': 150.50,
                'date': '2024-01-15',
                'category': 'alimentacao',
                'type': 'debit',
                'balance': 850.00,
                'bank': '001',
                'currency': 'BRL'
            },
            {
                'id': 2,
                'description': 'Pagamento de conta de luz CEMIG energia elétrica',
                'amount': 200.75,
                'date': '2024-01-16',
                'category': 'serviços',
                'type': 'debit',
                'balance': 649.25,
                'bank': '001',
                'currency': 'BRL'
            },
            {
                'id': 3,
                'description': 'Transferência PIX recebida valor R$ 500,00',
                'amount': 500.00,
                'date': '2024-01-17',
                'category': 'transferência',
                'type': 'credit',
                'balance': 1149.25,
                'bank': '001',
                'currency': 'BRL'
            }
        ]

    @pytest.fixture
    def engineer(self):
        """Create EnhancedFeatureEngineer instance"""
        config = {
            'text_processing': {
                'use_advanced_portuguese': False,  # Disable for faster testing
                'multi_language_support': False,
                'batch_size': 2
            },
            'temporal_features': {
                'enhanced_temporal': True,
                'business_days_aware': True,
                'holiday_aware': True,
                'include_cyclical_encoding': True
            },
            'financial_features': {
                'amount_pattern_recognition': True,
                'transaction_type_features': True
            },
            'data_augmentation': {
                'enabled': False  # Disable for faster testing
            },
            'quality_assurance': {
                'feature_validation': True,
                'cross_field_validation': False  # Disable for faster testing
            },
            'preprocessing': {
                'scaling_method': 'standard',
                'encoding_method': 'label',  # Use label encoding for testing
                'k_features': 50
            }
        }
        return EnhancedFeatureEngineer(config)

    def test_engineer_initialization(self, engineer):
        """Test engineer initialization"""
        assert engineer is not None
        assert hasattr(engineer, 'config')
        assert hasattr(engineer, 'embedding_model')
        assert hasattr(engineer, 'scaler')
        assert hasattr(engineer, 'encoder')

    def test_engineer_custom_config(self):
        """Test engineer with custom configuration"""
        custom_config = {
            'text_processing': {
                'use_advanced_portuguese': False,
                'batch_size': 1
            },
            'temporal_features': {
                'enhanced_temporal': False
            },
            'data_augmentation': {
                'enabled': False
            },
            'quality_assurance': {
                'feature_validation': False
            },
            'preprocessing': {
                'scaling_method': 'minmax'
            }
        }

        engineer = EnhancedFeatureEngineer(custom_config)
        assert engineer.config['preprocessing']['scaling_method'] == 'minmax'

    def test_create_enhanced_features_basic(self, engineer, sample_transaction_data):
        """Test basic enhanced feature creation"""
        features, feature_names, report = engineer.create_enhanced_features(sample_transaction_data)

        assert isinstance(features, np.ndarray)
        assert len(feature_names) > 0
        assert isinstance(report, dict)
        assert 'quality' in report
        assert 'feature_stats' in report

    def test_create_enhanced_features_with_target(self, engineer, sample_transaction_data):
        """Test enhanced feature creation with target column"""
        features, feature_names, report = engineer.create_enhanced_features(
            sample_transaction_data, target_column='category'
        )

        assert isinstance(features, np.ndarray)
        assert len(feature_names) > 0
        assert 'category' not in [name.split('_')[0] for name in feature_names]  # Target should be excluded

    def test_text_feature_extraction(self, engineer, sample_transaction_data):
        """Test text feature extraction"""
        features_dict = engineer._extract_all_features(
            pd.DataFrame(sample_transaction_data), None
        )

        # Should have text features if text processing is enabled
        if engineer.config['text_processing']['use_advanced_portuguese']:
            assert 'embeddings' in features_dict
            assert isinstance(features_dict['embeddings'], np.ndarray)
            assert features_dict['embeddings'].shape[0] == len(sample_transaction_data)

    def test_temporal_feature_extraction(self, engineer, sample_transaction_data):
        """Test temporal feature extraction"""
        features_dict = engineer._extract_all_features(
            pd.DataFrame(sample_transaction_data), None
        )

        assert 'temporal' in features_dict
        temporal_features = features_dict['temporal']

        assert isinstance(temporal_features, np.ndarray)
        assert temporal_features.shape[0] == len(sample_transaction_data)
        assert temporal_features.shape[1] > 0  # Should have multiple temporal features

    def test_financial_feature_extraction(self, engineer, sample_transaction_data):
        """Test financial feature extraction"""
        features_dict = engineer._extract_all_features(
            pd.DataFrame(sample_transaction_data), None
        )

        assert 'financial' in features_dict
        financial_features = features_dict['financial']

        assert isinstance(financial_features, np.ndarray)
        assert financial_features.shape[0] == len(sample_transaction_data)

        # Should include amount-based features
        assert financial_features.shape[1] >= 4  # At least amount, log_amount, and categories

    def test_transaction_pattern_extraction(self, engineer, sample_transaction_data):
        """Test transaction pattern extraction"""
        features_dict = engineer._extract_all_features(
            pd.DataFrame(sample_transaction_data), None
        )

        assert 'patterns' in features_dict
        pattern_features = features_dict['patterns']

        assert isinstance(pattern_features, np.ndarray)
        assert pattern_features.shape[0] == len(sample_transaction_data)

    def test_categorical_feature_extraction(self, engineer, sample_transaction_data):
        """Test categorical feature extraction"""
        features_dict = engineer._extract_all_features(
            pd.DataFrame(sample_transaction_data), None
        )

        assert 'categorical' in features_dict
        categorical_features = features_dict['categorical']

        assert isinstance(categorical_features, np.ndarray)
        assert categorical_features.shape[0] == len(sample_transaction_data)

    def test_feature_processing_and_selection(self, engineer):
        """Test feature processing and selection"""
        # Create sample features
        features_dict = {
            'temporal': np.random.rand(10, 5),
            'financial': np.random.rand(10, 3),
            'patterns': np.random.rand(10, 2)
        }

        processed_features, feature_names = engineer._process_and_select_features(features_dict)

        assert isinstance(processed_features, np.ndarray)
        assert isinstance(feature_names, list)
        assert processed_features.shape[0] == 10  # Same number of samples
        assert len(feature_names) == processed_features.shape[1]

    def test_feature_quality_assessment(self, engineer):
        """Test feature quality assessment"""
        # Create sample features
        features = np.random.rand(20, 10)
        feature_names = [f'feature_{i}' for i in range(10)]

        quality_report = engineer._assess_feature_quality(features, feature_names)

        assert isinstance(quality_report, dict)
        assert 'total_features' in quality_report
        assert 'feature_matrix_shape' in quality_report
        assert 'quality_score' in quality_report
        assert 'missing_values' in quality_report
        assert 'zero_variance_features' in quality_report

        # Quality score should be between 0 and 1
        assert 0.0 <= quality_report['quality_score'] <= 1.0

    def test_feature_correlation_analysis(self, engineer):
        """Test feature correlation analysis"""
        # Create correlated features
        np.random.seed(42)
        base_feature = np.random.rand(100)
        correlated_feature = base_feature + 0.1 * np.random.rand(100)
        features = np.column_stack([base_feature, correlated_feature, np.random.rand(100)])

        correlations = engineer._calculate_feature_correlations(features)

        assert isinstance(correlations, dict)
        assert 'correlation_matrix_shape' in correlations
        assert correlations['correlation_matrix_shape'] == (3, 3)

    def test_feature_statistics_calculation(self, engineer):
        """Test feature statistics calculation"""
        features = np.random.rand(50, 8)
        feature_names = [f'feature_{i}' for i in range(8)]

        stats = engineer._get_feature_statistics(features, feature_names)

        assert isinstance(stats, dict)
        assert 'feature_count' in stats
        assert 'sample_count' in stats
        assert 'means' in stats
        assert 'stds' in stats
        assert 'mins' in stats
        assert 'maxs' in stats

        assert stats['feature_count'] == 8
        assert stats['sample_count'] == 50
        assert len(stats['means']) == 8

    def test_performance_tracking(self, engineer):
        """Test performance tracking"""
        # Simulate feature engineering run
        input_size = 100
        output_shape = (100, 50)
        quality_report = {'quality_score': 0.85}

        engineer._track_performance(input_size, output_shape, quality_report)

        assert len(engineer.performance_history) > 0

        last_entry = engineer.performance_history[-1]
        assert last_entry['input_size'] == input_size
        assert last_entry['output_shape'] == output_shape
        assert last_entry['quality_score'] == 0.85

    def test_performance_analytics(self, engineer):
        """Test performance analytics generation"""
        # Add some performance data
        for i in range(5):
            engineer.performance_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'input_size': 100 + i * 10,
                'output_shape': (100 + i * 10, 50),
                'feature_count': 50,
                'quality_score': 0.8 + i * 0.03
            })

        analytics = engineer.get_performance_analytics()

        assert isinstance(analytics, dict)
        assert 'total_runs' in analytics
        assert 'avg_quality_score' in analytics
        assert 'avg_features_generated' in analytics
        assert 'performance_trend' in analytics

        assert analytics['total_runs'] == 5
        assert len(analytics['performance_trend']) == 5

    def test_financial_term_extraction(self, engineer):
        """Test financial term extraction"""
        texts = [
            'PIX TED DOC boleto conta agência número banco',
            'Transferência depósito saque valor total',
            'Pagamento juros multa taxa desconto'
        ]

        financial_terms = engineer._extract_financial_terms(texts)

        assert isinstance(financial_terms, np.ndarray)
        assert financial_terms.shape[0] == len(texts)
        assert financial_terms.shape[1] == 11  # Number of financial keywords

    def test_season_calculation(self, engineer):
        """Test season calculation from months"""
        # Test different months
        assert engineer._get_season_numeric(1) == 0   # January - Summer
        assert engineer._get_season_numeric(4) == 1   # April - Autumn
        assert engineer._get_season_numeric(7) == 2   # July - Winter
        assert engineer._get_season_numeric(10) == 3  # October - Spring

    def test_temporal_features_comprehensive(self, engineer):
        """Test comprehensive temporal feature extraction"""
        # Create test data with various dates
        dates = [
            '2024-01-15',  # Monday
            '2024-01-16',  # Tuesday
            '2024-01-20',  # Saturday
            '2024-01-25',  # Thursday
            '2024-12-25'   # Christmas (holiday)
        ]

        temporal_features = engineer._extract_temporal_features(dates)

        assert isinstance(temporal_features, np.ndarray)
        assert temporal_features.shape[0] == len(dates)

        # Should include basic features (year, month, day, weekday)
        assert temporal_features.shape[1] >= 4

        # Should include cyclical encoding if enabled
        if engineer.config['temporal_features']['include_cyclical_encoding']:
            # Should have additional sin/cos features
            assert temporal_features.shape[1] >= 8

    def test_financial_features_comprehensive(self, engineer):
        """Test comprehensive financial feature extraction"""
        df = pd.DataFrame({
            'amount': [100.0, 250.0, 1000.0, 50.0, 500.0],
            'type': ['debit', 'credit', 'debit', 'debit', 'credit'],
            'bank': ['001', '001', '237', '001', '341'],
            'currency': ['BRL', 'BRL', 'USD', 'BRL', 'EUR']
        })

        financial_features = engineer._extract_financial_features(df)

        assert isinstance(financial_features, np.ndarray)
        assert financial_features.shape[0] == len(df)

        # Should include amount features, type encoding, bank encoding, currency encoding
        expected_features = 1 + 1 + 2 + 3 + 3  # amount + log_amount + type_dummies + bank_dummies + currency_dummies
        assert financial_features.shape[1] >= expected_features

    def test_transaction_patterns_comprehensive(self, engineer):
        """Test comprehensive transaction pattern extraction"""
        # Create data with date patterns
        base_date = datetime(2024, 1, 1)
        data = []
        for i in range(20):
            data.append({
                'date': (base_date + timedelta(days=i % 5)).strftime('%Y-%m-%d'),  # Repeat every 5 days
                'amount': float(100 + i * 10)
            })

        df = pd.DataFrame(data)
        pattern_features = engineer._extract_transaction_patterns(df)

        assert isinstance(pattern_features, np.ndarray)
        assert pattern_features.shape[0] == len(data)

    def test_categorical_features_with_target(self, engineer):
        """Test categorical feature extraction with target encoding"""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A'],
            'target': [1, 0, 1, 0, 0, 1]
        })

        categorical_features = engineer._extract_categorical_features(df, 'target')

        assert isinstance(categorical_features, np.ndarray)
        assert categorical_features.shape[0] == len(df)
        assert categorical_features.shape[1] >= 1  # At least one encoded feature

    def test_feature_scaling_methods(self):
        """Test different feature scaling methods"""
        test_configs = [
            {'preprocessing': {'scaling_method': 'standard'}},
            {'preprocessing': {'scaling_method': 'minmax'}},
            {'preprocessing': {'scaling_method': 'robust'}}
        ]

        for config in test_configs:
            engineer = EnhancedFeatureEngineer(config)
            features = np.random.rand(20, 5)

            scaled_features = engineer.scaler.fit_transform(features)

            assert isinstance(scaled_features, np.ndarray)
            assert scaled_features.shape == features.shape

    def test_feature_encoding_methods(self):
        """Test different feature encoding methods"""
        # Test label encoding (fallback)
        config = {
            'preprocessing': {'encoding_method': 'label'},
            'text_processing': {'use_advanced_portuguese': False},
            'data_augmentation': {'enabled': False},
            'quality_assurance': {'feature_validation': False}
        }

        engineer = EnhancedFeatureEngineer(config)

        categories = np.array(['A', 'B', 'C', 'A', 'B'])
        df = pd.DataFrame({'category': categories})

        encoded_features = engineer._extract_categorical_features(df)

        assert isinstance(encoded_features, np.ndarray)
        assert encoded_features.shape[0] == len(categories)

    def test_quality_score_calculation(self, engineer):
        """Test quality score calculation"""
        # Test with perfect features
        perfect_features = np.random.rand(100, 10)
        quality_score = engineer._calculate_quality_score(perfect_features)

        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0

        # Test with features containing NaN
        features_with_nan = perfect_features.copy()
        features_with_nan[0, 0] = np.nan
        quality_score_nan = engineer._calculate_quality_score(features_with_nan)

        assert quality_score_nan < quality_score  # Should be lower with NaN

        # Test with zero variance features
        features_zero_var = perfect_features.copy()
        features_zero_var[:, 0] = 1.0  # Constant feature
        quality_score_zero_var = engineer._calculate_quality_score(features_zero_var)

        assert quality_score_zero_var < quality_score  # Should be lower with zero variance

    def test_feature_selection_functionality(self, engineer):
        """Test feature selection functionality"""
        # Create features with some irrelevant ones
        np.random.seed(42)
        relevant_features = np.random.rand(100, 3)
        irrelevant_features = np.random.rand(100, 7) * 0.01  # Low variance

        features = np.concatenate([relevant_features, irrelevant_features], axis=1)
        target = np.random.randint(0, 2, 100)

        # Select top 5 features
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(score_func=f_classif, k=5)
        selected_features = selector.fit_transform(features, target)

        assert selected_features.shape == (100, 5)

        # Should select mostly relevant features
        selected_mask = selector.get_support()
        relevant_selected = sum(selected_mask[:3])  # First 3 are relevant
        assert relevant_selected >= 2  # Should select at least 2 relevant features

    def test_error_handling_feature_extraction(self, engineer):
        """Test error handling in feature extraction"""
        # Test with empty data
        empty_df = pd.DataFrame()
        features_dict = engineer._extract_all_features(empty_df, None)

        assert isinstance(features_dict, dict)
        # Should handle empty data gracefully

        # Test with missing columns
        df_missing_cols = pd.DataFrame({'id': [1, 2, 3]})
        features_dict = engineer._extract_all_features(df_missing_cols, None)

        assert isinstance(features_dict, dict)
        # Should handle missing columns gracefully

    def test_memory_efficiency_large_dataset(self, engineer):
        """Test memory efficiency with large dataset"""
        # Create moderately large dataset
        np.random.seed(42)
        large_data = []
        for i in range(500):
            large_data.append({
                'description': f'Transaction {i} description text',
                'amount': float(np.random.normal(1000, 200)),
                'date': (datetime(2024, 1, 1) + timedelta(days=i % 365)).strftime('%Y-%m-%d'),
                'category': np.random.choice(['food', 'transport', 'utilities']),
                'type': np.random.choice(['debit', 'credit'])
            })

        # Should handle large dataset without memory issues
        features, feature_names, report = engineer.create_enhanced_features(large_data)

        assert isinstance(features, np.ndarray)
        assert len(feature_names) > 0
        assert isinstance(report, dict)

    def test_feature_engineering_reproducibility(self, engineer, sample_transaction_data):
        """Test reproducibility of feature engineering"""
        # Run feature engineering twice with same data
        features1, names1, report1 = engineer.create_enhanced_features(sample_transaction_data)
        features2, names2, report2 = engineer.create_enhanced_features(sample_transaction_data)

        # Results should be identical (same random seed)
        assert features1.shape == features2.shape
        assert names1 == names2

        # Features should be very similar (allowing for small numerical differences)
        np.testing.assert_allclose(features1, features2, rtol=1e-10, atol=1e-10)

    def test_save_load_functionality(self, engineer, tmp_path):
        """Test save and load functionality"""
        import os

        # Save engineer state
        save_path = tmp_path / "test_engineer.pkl"
        engineer.save_enhanced_engineer(str(save_path))

        assert os.path.exists(save_path)

        # Create new engineer and load state
        new_engineer = EnhancedFeatureEngineer()
        new_engineer.load_enhanced_engineer(str(save_path))

        # Should have loaded configuration
        assert new_engineer.config is not None

    def test_integration_with_data_augmentation(self, engineer, sample_transaction_data):
        """Test integration with data augmentation"""
        # Enable data augmentation
        engineer.config['data_augmentation']['enabled'] = True

        # Mock the augmentation pipeline
        with patch.object(engineer, 'augmentation_pipeline') as mock_pipeline:
            mock_pipeline.augment_dataset.return_value = (
                pd.DataFrame(sample_transaction_data * 2),  # Doubled data
                {'augmentation_applied': True, 'ratio': 2.0}
            )

            features, names, report = engineer.create_enhanced_features(sample_transaction_data)

            # Should have called augmentation
            mock_pipeline.augment_dataset.assert_called_once()

            # Should have augmented data in report
            assert 'augmentation' in report

    def test_integration_with_quality_validation(self, engineer, sample_transaction_data):
        """Test integration with quality validation"""
        # Enable quality validation
        engineer.config['quality_assurance']['cross_field_validation'] = True

        # Mock the validation engine
        with patch.object(engineer, 'validation_engine') as mock_validation:
            mock_validation.validate_dataset.return_value = Mock(
                to_dict=lambda: {'validation_passed': True, 'errors': []}
            )

            features, names, report = engineer.create_enhanced_features(sample_transaction_data)

            # Should have called validation
            mock_validation.validate_dataset.assert_called_once()

            # Should have validation in report
            assert 'validation' in report

    def test_end_to_end_feature_engineering_pipeline(self, engineer, sample_transaction_data):
        """Test end-to-end feature engineering pipeline"""
        # Run complete pipeline
        features, feature_names, comprehensive_report = engineer.create_enhanced_features(
            sample_transaction_data, target_column='category'
        )

        # Verify all components are present
        assert isinstance(features, np.ndarray)
        assert isinstance(feature_names, list)
        assert isinstance(comprehensive_report, dict)

        # Check report structure
        required_report_keys = ['quality', 'feature_stats', 'processing_timestamp']
        for key in required_report_keys:
            assert key in comprehensive_report

        # Check quality report
        quality_report = comprehensive_report['quality']
        required_quality_keys = ['total_features', 'quality_score', 'feature_matrix_shape']
        for key in required_quality_keys:
            assert key in quality_report

        # Check feature stats
        feature_stats = comprehensive_report['feature_stats']
        required_stats_keys = ['feature_count', 'sample_count', 'means', 'stds']
        for key in required_stats_keys:
            assert key in feature_stats

        # Verify feature matrix properties
        assert features.shape[0] == len(sample_transaction_data)
        assert features.shape[1] == len(feature_names)
        assert not np.any(np.isnan(features))  # No NaN values
        assert not np.any(np.isinf(features))  # No infinite values


class TestFeatureQualityTracker:
    """Test FeatureQualityTracker functionality"""

    @pytest.fixture
    def quality_tracker(self):
        """Create FeatureQualityTracker instance"""
        return FeatureQualityTracker()

    def test_quality_tracker_initialization(self, quality_tracker):
        """Test quality tracker initialization"""
        assert quality_tracker is not None
        assert hasattr(quality_tracker, 'quality_history')
        assert hasattr(quality_tracker, 'alerts')

    def test_track_quality(self, quality_tracker):
        """Test quality tracking"""
        features = np.random.rand(50, 10)
        feature_names = [f'feature_{i}' for i in range(10)]
        quality_metrics = {
            'quality_score': 0.85,
            'missing_values': 0,
            'zero_variance_features': 1
        }

        quality_tracker.track_quality(features, feature_names, quality_metrics)

        assert len(quality_tracker.quality_history) == 1

        entry = quality_tracker.quality_history[0]
        assert entry['feature_count'] == 10
        assert entry['quality_metrics'] == quality_metrics
        assert 'timestamp' in entry

    def test_generate_alerts(self, quality_tracker):
        """Test alert generation"""
        # Test with issues that should generate alerts
        quality_metrics_with_issues = {
            'missing_values': 5,
            'zero_variance_features': 2,
            'quality_score': 0.6
        }

        alerts = quality_tracker._generate_alerts(quality_metrics_with_issues)

        assert isinstance(alerts, list)
        assert len(alerts) >= 3  # Should have alerts for missing values, zero variance, and low quality

        # Test with good metrics
        quality_metrics_good = {
            'missing_values': 0,
            'zero_variance_features': 0,
            'quality_score': 0.9
        }

        alerts_good = quality_tracker._generate_alerts(quality_metrics_good)

        assert isinstance(alerts_good, list)
        assert len(alerts_good) == 0  # Should have no alerts

    def test_get_quality_report(self, quality_tracker):
        """Test quality report generation"""
        # Add some quality tracking data
        for i in range(3):
            quality_metrics = {
                'quality_score': 0.8 + i * 0.05,
                'missing_values': i,
                'zero_variance_features': 0
            }
            quality_tracker.track_quality(
                np.random.rand(20, 5),
                [f'feature_{j}' for j in range(5)],
                quality_metrics
            )

        report = quality_tracker.get_quality_report()

        assert isinstance(report, dict)
        assert 'current_quality' in report
        assert 'quality_trend' in report
        assert 'active_alerts' in report
        assert 'total_alerts' in report

        assert len(report['quality_trend']) == 3
        assert report['total_alerts'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])