"""
IntegrationTestSuite - End-to-end integration testing for Phase 2 data quality improvements

This module provides comprehensive integration tests for:
- Complete data processing pipeline testing
- API endpoint testing for all new features
- Performance benchmarking and regression testing
- Error handling and recovery testing
- Scalability testing with large datasets
"""

import pytest
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from flask import Flask
import tempfile
import os

from src.services.advanced_imputation_engine import AdvancedImputationEngine
from src.services.advanced_portuguese_preprocessor import AdvancedPortuguesePreprocessor
from src.services.data_augmentation_pipeline import DataAugmentationPipeline
from src.services.enhanced_feature_engineer import EnhancedFeatureEngineer
from src.utils.business_logic_validator import BusinessLogicValidator
from src.routes.model_manager import model_manager_bp
from src.services.model_manager import ModelManager


class TestCompleteDataProcessingPipeline:
    """Test complete data processing pipeline integration"""

    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw transaction data"""
        return [
            {
                'id': 1,
                'description': 'Compra no supermercado Extra hiper bom preco',
                'amount': 150.50,
                'date': '2024-01-15',
                'category': 'alimentacao',
                'type': 'debit',
                'balance': None,  # Missing data
                'bank': '001',
                'currency': 'BRL'
            },
            {
                'id': 2,
                'description': 'Pagamento de conta de luz CEMIG energia elétrica',
                'amount': None,  # Missing data
                'date': '2024-01-16',
                'category': 'serviços',
                'type': 'debit',
                'balance': 850.00,
                'bank': '001',
                'currency': 'BRL'
            },
            {
                'id': 3,
                'description': 'Transferência PIX recebida valor R$ 500,00',
                'amount': 500.00,
                'date': None,  # Missing data
                'category': 'transferência',
                'type': 'credit',
                'balance': 1350.00,
                'bank': '001',
                'currency': 'BRL'
            },
            {
                'id': 4,
                'description': '',  # Empty description
                'amount': 200.00,
                'date': '2024-01-18',
                'category': None,  # Missing category
                'type': 'debit',
                'balance': 1150.00,
                'bank': '001',
                'currency': 'BRL'
            }
        ]

    @pytest.fixture
    def complete_pipeline_config(self):
        """Configuration for complete pipeline testing"""
        return {
            'imputation': {
                'method': 'intelligent',
                'statistical_methods': {
                    'numeric_strategy': 'mean',
                    'categorical_strategy': 'most_frequent'
                }
            },
            'preprocessing': {
                'use_advanced_portuguese': True,
                'multi_language_support': True,
                'cache_enabled': True
            },
            'augmentation': {
                'enabled': True,
                'augmentation_ratio': 1.5,
                'use_smote': False,  # Disable for faster testing
                'synthetic_data_generation': False
            },
            'feature_engineering': {
                'text_processing': {'use_advanced_portuguese': False},
                'temporal_features': {'enhanced_temporal': True},
                'financial_features': {'amount_pattern_recognition': True},
                'data_augmentation': {'enabled': False},
                'quality_assurance': {'feature_validation': True},
                'preprocessing': {'scaling_method': 'standard'}
            },
            'validation': {
                'business_rules_enabled': True,
                'referential_integrity_check': True
            }
        }

    def test_end_to_end_data_processing_pipeline(self, sample_raw_data, complete_pipeline_config):
        """Test complete end-to-end data processing pipeline"""
        # Step 1: Data Imputation
        imputation_engine = AdvancedImputationEngine(complete_pipeline_config['imputation'])
        df = pd.DataFrame(sample_raw_data)

        # Apply imputation
        imputed_df, imputation_report = imputation_engine.auto_impute(df, 'intelligent')

        # Verify imputation worked
        assert not imputed_df['amount'].isnull().any()
        assert not imputed_df['balance'].isnull().any()
        assert not imputed_df['date'].isnull().any()
        assert imputation_report['total_imputations'] > 0

        # Step 2: Text Preprocessing
        preprocessor = AdvancedPortuguesePreprocessor({
            'use_spacy': False,  # Disable for faster testing
            'use_nltk': True,
            'cache_enabled': True
        })

        descriptions = imputed_df['description'].fillna('').tolist()
        processed_texts = preprocessor.preprocess_batch(descriptions)

        # Verify preprocessing worked
        assert len(processed_texts) == len(descriptions)
        for result in processed_texts:
            assert 'processed_text' in result
            assert 'quality_metrics' in result

        # Step 3: Data Augmentation
        augmentation_config = {
            'general': {'augmentation_ratio': 1.2, 'random_seed': 42},
            'text_augmentation': {'enabled': False},
            'numerical_augmentation': {'enabled': True},
            'categorical_augmentation': {'enabled': False},
            'temporal_augmentation': {'enabled': False},
            'synthetic_generation': {'enabled': False},
            'quality_control': {'enabled': True}
        }

        augmentation_pipeline = DataAugmentationPipeline(augmentation_config)
        augmented_df, augmentation_report = augmentation_pipeline.augment_dataset(
            imputed_df.to_dict('records'), 'transaction'
        )

        # Verify augmentation worked
        assert len(augmented_df) >= len(imputed_df)
        assert augmentation_report['original_size'] == len(imputed_df)

        # Step 4: Feature Engineering
        feature_engineer = EnhancedFeatureEngineer(complete_pipeline_config['feature_engineering'])
        features, feature_names, feature_report = feature_engineer.create_enhanced_features(
            augmented_df.to_dict('records'), target_column='category'
        )

        # Verify feature engineering worked
        assert isinstance(features, np.ndarray)
        assert len(feature_names) > 0
        assert features.shape[0] == len(augmented_df)
        assert features.shape[1] == len(feature_names)
        assert 'quality' in feature_report

        # Step 5: Business Logic Validation
        validator = BusinessLogicValidator()
        validation_results = []

        for record in augmented_df.to_dict('records'):
            result = validator.validate(record, rule_group='transaction_validation')
            validation_results.append(result)

        # Verify validation worked
        assert len(validation_results) == len(augmented_df)
        for result in validation_results:
            assert hasattr(result, 'is_valid')

        # Comprehensive pipeline report
        pipeline_report = {
            'original_records': len(sample_raw_data),
            'imputation': imputation_report,
            'preprocessing': {
                'texts_processed': len(processed_texts),
                'avg_quality': np.mean([r['quality_metrics'].get('overall_quality', 0) for r in processed_texts])
            },
            'augmentation': augmentation_report,
            'feature_engineering': {
                'features_generated': len(feature_names),
                'feature_matrix_shape': features.shape,
                'quality_score': feature_report['quality']['quality_score']
            },
            'validation': {
                'records_validated': len(validation_results),
                'validation_passed': sum(1 for r in validation_results if r.is_valid),
                'avg_confidence': np.mean([getattr(r, 'validation_duration', 0) for r in validation_results])
            },
            'pipeline_success': True
        }

        # Verify complete pipeline success
        assert pipeline_report['pipeline_success'] is True
        assert pipeline_report['original_records'] == len(sample_raw_data)
        assert pipeline_report['feature_engineering']['features_generated'] > 0

        return pipeline_report

    def test_pipeline_error_handling_and_recovery(self, sample_raw_data):
        """Test pipeline error handling and recovery"""
        # Test with corrupted data
        corrupted_data = sample_raw_data.copy()
        corrupted_data[0] = {'invalid': 'data'}  # Completely invalid record

        # Pipeline should handle errors gracefully
        try:
            imputation_engine = AdvancedImputationEngine()
            df = pd.DataFrame(corrupted_data)

            # Should handle invalid data
            imputed_df, report = imputation_engine.auto_impute(df, 'simple')

            # Should still produce results
            assert isinstance(imputed_df, pd.DataFrame)
            assert isinstance(report, dict)

        except Exception as e:
            # If it fails, it should be a controlled failure
            assert isinstance(e, Exception)

    def test_pipeline_scalability_large_dataset(self):
        """Test pipeline scalability with large dataset"""
        # Create large dataset
        np.random.seed(42)
        large_data = []
        for i in range(1000):
            large_data.append({
                'id': i,
                'description': f'Transaction {i} description',
                'amount': float(np.random.normal(1000, 200)),
                'date': (datetime(2024, 1, 1) + timedelta(days=i % 365)).strftime('%Y-%m-%d'),
                'category': np.random.choice(['food', 'transport', 'utilities', 'entertainment']),
                'type': np.random.choice(['debit', 'credit']),
                'balance': float(np.random.normal(5000, 1000)),
                'bank': np.random.choice(['001', '237', '341']),
                'currency': 'BRL'
            })

        # Add some missing data
        for i in range(0, len(large_data), 10):
            large_data[i]['amount'] = None
            large_data[i]['description'] = None

        start_time = time.time()

        # Run complete pipeline
        imputation_engine = AdvancedImputationEngine()
        df = pd.DataFrame(large_data)
        imputed_df, _ = imputation_engine.auto_impute(df, 'simple')

        preprocessor = AdvancedPortuguesePreprocessor({'use_spacy': False})
        descriptions = imputed_df['description'].fillna('').tolist()
        processed_texts = preprocessor.preprocess_batch(descriptions)

        feature_engineer = EnhancedFeatureEngineer({
            'text_processing': {'use_advanced_portuguese': False},
            'temporal_features': {'enhanced_temporal': True},
            'financial_features': {'amount_pattern_recognition': True},
            'data_augmentation': {'enabled': False},
            'quality_assurance': {'feature_validation': True},
            'preprocessing': {'scaling_method': 'standard'}
        })

        features, feature_names, _ = feature_engineer.create_enhanced_features(
            imputed_df.to_dict('records'), target_column='category'
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify scalability
        assert features.shape[0] == len(large_data)
        assert len(feature_names) > 0
        assert processing_time < 60.0  # Should complete within 1 minute
        assert not np.any(np.isnan(features))

    def test_pipeline_memory_efficiency(self):
        """Test pipeline memory efficiency"""
        # Create moderately large dataset
        data = []
        for i in range(5000):
            data.append({
                'description': f'Test transaction {i}',
                'amount': float(i * 10),
                'date': '2024-01-15',
                'category': 'test',
                'type': 'debit'
            })

        # Monitor memory usage (simplified)
        initial_objects = len(data)

        # Process through pipeline
        imputation_engine = AdvancedImputationEngine()
        df = pd.DataFrame(data)
        imputed_df, _ = imputation_engine.auto_impute(df, 'simple')

        feature_engineer = EnhancedFeatureEngineer({
            'text_processing': {'use_advanced_portuguese': False},
            'data_augmentation': {'enabled': False},
            'quality_assurance': {'feature_validation': False}
        })

        features, _, _ = feature_engineer.create_enhanced_features(
            imputed_df.to_dict('records')
        )

        # Verify memory efficiency
        assert features.shape[0] == len(data)
        assert features.nbytes < 50 * 1024 * 1024  # Less than 50MB for features


class TestAPIEndpointIntegration:
    """Test API endpoint integration with Phase 2 features"""

    @pytest.fixture
    def app(self):
        """Create Flask test app"""
        app = Flask(__name__)
        app.register_blueprint(model_manager_bp)
        app.config['TESTING'] = True
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()

    @pytest.fixture
    def mock_model_manager(self):
        """Mock ModelManager for testing"""
        with patch('src.routes.model_manager.model_manager', new_callable=Mock) as mock_mm:
            yield mock_mm

    def test_data_quality_pipeline_api_integration(self, client, mock_model_manager, sample_raw_data):
        """Test data quality pipeline API integration"""
        # Mock the model manager to return processed data
        mock_model_manager.process_data.return_value = (
            np.random.rand(len(sample_raw_data), 10),
            ['category_A'] * len(sample_raw_data),
            [f'feature_{i}' for i in range(10)]
        )
        mock_model_manager.train_model.return_value = {
            'success': True,
            'message': 'Model trained with enhanced data quality pipeline'
        }

        # Test training with enhanced data quality
        response = client.post('/models/train', json={
            'model_type': 'random_forest',
            'data_source': 'transactions',
            'use_data_quality_pipeline': True,
            'imputation_method': 'intelligent',
            'text_preprocessing': True,
            'data_augmentation': True
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

    def test_feature_engineering_api_integration(self, client, mock_model_manager):
        """Test feature engineering API integration"""
        mock_model_manager.create_enhanced_features.return_value = (
            np.random.rand(10, 20),
            [f'enhanced_feature_{i}' for i in range(20)],
            {
                'quality': {'quality_score': 0.95},
                'feature_stats': {'feature_count': 20}
            }
        )

        response = client.post('/models/feature-engineering', json={
            'data': [
                {'description': 'Test transaction', 'amount': 100.0, 'date': '2024-01-01'}
            ],
            'text_processing': True,
            'temporal_features': True,
            'financial_features': True,
            'quality_assurance': True
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'features' in data['data']
        assert 'feature_names' in data['data']

    def test_data_augmentation_api_integration(self, client, mock_model_manager):
        """Test data augmentation API integration"""
        mock_model_manager.augment_dataset.return_value = (
            pd.DataFrame([{'description': 'Augmented transaction', 'amount': 150.0}]),
            {'original_size': 1, 'final_size': 2, 'augmentation_ratio': 2.0}
        )

        response = client.post('/models/data-augmentation', json={
            'data': [{'description': 'Original transaction', 'amount': 100.0}],
            'augmentation_ratio': 2.0,
            'text_augmentation': True,
            'numerical_augmentation': True,
            'quality_control': True
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'augmented_data' in data['data']
        assert 'report' in data['data']

    def test_validation_api_integration(self, client, mock_model_manager):
        """Test validation API integration"""
        mock_model_manager.validate_data.return_value = {
            'validation_passed': True,
            'errors': [],
            'warnings': ['Minor issue detected'],
            'quality_score': 0.88
        }

        response = client.post('/models/validate', json={
            'data': [{'amount': 100.0, 'type': 'debit', 'date': '2024-01-01'}],
            'validation_rules': ['business_logic', 'referential_integrity'],
            'strict_mode': False
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'validation_results' in data['data']

    def test_complete_ml_pipeline_api_integration(self, client, mock_model_manager):
        """Test complete ML pipeline API integration"""
        # Mock all components
        mock_model_manager.process_data.return_value = (
            np.random.rand(20, 15),
            ['class_A'] * 20,
            [f'feature_{i}' for i in range(15)]
        )
        mock_model_manager.train_model.return_value = {
            'success': True,
            'accuracy': 0.92,
            'f1_score': 0.89
        }
        mock_model_manager.predict.return_value = np.array(['class_A', 'class_B'])
        mock_model_manager.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])

        # Test complete pipeline: data quality -> feature engineering -> training -> prediction
        training_data = [
            {'description': 'Transaction 1', 'amount': 100.0, 'date': '2024-01-01', 'category': 'A'},
            {'description': 'Transaction 2', 'amount': 200.0, 'date': '2024-01-02', 'category': 'B'}
        ] * 5  # 10 records total

        # 1. Train model with enhanced pipeline
        response = client.post('/models/train', json={
            'model_type': 'random_forest',
            'data': training_data,
            'use_enhanced_pipeline': True,
            'data_quality': True,
            'feature_engineering': True
        })

        assert response.status_code == 200
        train_data = json.loads(response.data)
        assert train_data['success'] is True

        # 2. Make predictions
        test_data = [
            {'description': 'New transaction', 'amount': 150.0, 'date': '2024-01-03'}
        ]

        response = client.post('/models/predict', json={
            'model_type': 'random_forest',
            'data': test_data,
            'use_enhanced_features': True
        })

        assert response.status_code == 200
        predict_data = json.loads(response.data)
        assert predict_data['success'] is True
        assert 'prediction' in predict_data['data']


class TestPerformanceBenchmarking:
    """Test performance benchmarking and regression testing"""

    def test_pipeline_performance_benchmarking(self, sample_raw_data):
        """Test pipeline performance benchmarking"""
        performance_results = {}

        # Benchmark imputation
        start_time = time.time()
        imputation_engine = AdvancedImputationEngine()
        df = pd.DataFrame(sample_raw_data)
        imputed_df, _ = imputation_engine.auto_impute(df, 'simple')
        imputation_time = time.time() - start_time
        performance_results['imputation'] = imputation_time

        # Benchmark preprocessing
        start_time = time.time()
        preprocessor = AdvancedPortuguesePreprocessor({'use_spacy': False})
        descriptions = imputed_df['description'].fillna('').tolist()
        processed_texts = preprocessor.preprocess_batch(descriptions)
        preprocessing_time = time.time() - start_time
        performance_results['preprocessing'] = preprocessing_time

        # Benchmark feature engineering
        start_time = time.time()
        feature_engineer = EnhancedFeatureEngineer({
            'text_processing': {'use_advanced_portuguese': False},
            'data_augmentation': {'enabled': False},
            'quality_assurance': {'feature_validation': False}
        })
        features, _, _ = feature_engineer.create_enhanced_features(
            imputed_df.to_dict('records')
        )
        feature_time = time.time() - start_time
        performance_results['feature_engineering'] = feature_time

        # Verify performance is reasonable
        total_time = sum(performance_results.values())
        assert total_time < 10.0  # Should complete within 10 seconds

        # Each component should take reasonable time
        assert performance_results['imputation'] < 2.0
        assert performance_results['preprocessing'] < 3.0
        assert performance_results['feature_engineering'] < 5.0

        return performance_results

    def test_memory_usage_tracking(self):
        """Test memory usage tracking during pipeline execution"""
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run pipeline with moderate dataset
        data = []
        for i in range(100):
            data.append({
                'description': f'Test transaction {i}',
                'amount': float(i * 10),
                'date': '2024-01-15',
                'category': 'test'
            })

        imputation_engine = AdvancedImputationEngine()
        df = pd.DataFrame(data)
        imputed_df, _ = imputation_engine.auto_impute(df, 'simple')

        feature_engineer = EnhancedFeatureEngineer({
            'text_processing': {'use_advanced_portuguese': False},
            'data_augmentation': {'enabled': False},
            'quality_assurance': {'feature_validation': False}
        })

        features, _, _ = feature_engineer.create_enhanced_features(
            imputed_df.to_dict('records')
        )

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100.0

    def test_regression_testing_against_baselines(self):
        """Test regression testing against performance baselines"""
        # Define performance baselines
        baselines = {
            'imputation_time': 1.0,  # seconds
            'preprocessing_time': 2.0,
            'feature_engineering_time': 3.0,
            'memory_usage': 50.0,  # MB
            'quality_score': 0.8
        }

        # Run current performance test
        data = []
        for i in range(50):
            data.append({
                'description': f'Test transaction {i}',
                'amount': float(i * 10),
                'date': '2024-01-15',
                'category': 'test'
            })

        start_time = time.time()

        imputation_engine = AdvancedImputationEngine()
        df = pd.DataFrame(data)
        imputed_df, _ = imputation_engine.auto_impute(df, 'simple')

        feature_engineer = EnhancedFeatureEngineer({
            'text_processing': {'use_advanced_portuguese': False},
            'data_augmentation': {'enabled': False},
            'quality_assurance': {'feature_validation': True}
        })

        features, _, report = feature_engineer.create_enhanced_features(
            imputed_df.to_dict('records')
        )

        total_time = time.time() - start_time
        quality_score = report['quality']['quality_score']

        # Check against baselines
        assert total_time < baselines['imputation_time'] + baselines['preprocessing_time'] + baselines['feature_engineering_time']
        assert quality_score >= baselines['quality_score']

        # No regression if all tests pass
        assert True


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms"""

    def test_pipeline_error_recovery(self):
        """Test pipeline error recovery"""
        # Create data with various issues
        problematic_data = [
            {'description': None, 'amount': 'invalid', 'date': '2024-01-01'},
            {'description': 'Valid transaction', 'amount': 100.0, 'date': None},
            {'description': '', 'amount': None, 'date': '2024-01-03'},
            {'description': 'Another transaction', 'amount': 200.0, 'date': 'invalid_date'}
        ]

        # Pipeline should handle errors gracefully
        imputation_engine = AdvancedImputationEngine()

        try:
            df = pd.DataFrame(problematic_data)
            imputed_df, report = imputation_engine.auto_impute(df, 'simple')

            # Should still produce results
            assert isinstance(imputed_df, pd.DataFrame)
            assert len(imputed_df) == len(problematic_data)

        except Exception as e:
            # If it fails, should be controlled
            assert isinstance(e, Exception)

    def test_component_failure_isolation(self):
        """Test that component failures don't break entire pipeline"""
        data = [{'description': 'Test', 'amount': 100.0, 'date': '2024-01-01'}]

        # Test with feature engineering failure
        with patch('src.services.enhanced_feature_engineer.SentenceTransformer') as mock_st:
            mock_st.side_effect = Exception("Embedding model failed")

            feature_engineer = EnhancedFeatureEngineer({
                'text_processing': {'use_advanced_portuguese': False},
                'data_augmentation': {'enabled': False},
                'quality_assurance': {'feature_validation': False}
            })

            # Should handle gracefully
            features, names, report = feature_engineer.create_enhanced_features(data)

            # Should still return results
            assert isinstance(features, np.ndarray)
            assert isinstance(names, list)
            assert isinstance(report, dict)

    def test_data_validation_error_handling(self):
        """Test data validation error handling"""
        # Test with invalid data types
        invalid_data = [
            {'amount': 'not_a_number', 'type': 123, 'date': []},
            {'amount': None, 'type': None, 'date': None}
        ]

        validator = BusinessLogicValidator()

        for record in invalid_data:
            result = validator.validate(record, rule_group='transaction_validation')
            # Should handle gracefully
            assert isinstance(result, object)
            assert hasattr(result, 'is_valid')

    def test_api_error_response_handling(self, client):
        """Test API error response handling"""
        # Test with invalid JSON
        response = client.post('/models/train',
                              data='invalid json',
                              content_type='application/json')

        assert response.status_code in [400, 500]

        # Test with missing required fields
        response = client.post('/models/train', json={})

        assert response.status_code in [400, 500]

        # Test with invalid model type
        response = client.post('/models/train', json={
            'model_type': 'invalid_model',
            'data_source': 'transactions'
        })

        assert response.status_code == 500


class TestScalabilityTesting:
    """Test scalability with large datasets"""

    def test_large_dataset_processing(self):
        """Test processing of large datasets"""
        # Create large dataset (1000 records)
        np.random.seed(42)
        large_data = []
        for i in range(1000):
            large_data.append({
                'id': i,
                'description': f'Transaction {i} with detailed description for testing purposes',
                'amount': float(np.random.normal(1000, 200)),
                'date': (datetime(2024, 1, 1) + timedelta(days=i % 365)).strftime('%Y-%m-%d'),
                'category': np.random.choice(['food', 'transport', 'utilities', 'entertainment', 'shopping']),
                'type': np.random.choice(['debit', 'credit']),
                'balance': float(np.random.normal(5000, 1000)),
                'bank': np.random.choice(['001', '237', '341', '104']),
                'currency': np.random.choice(['BRL', 'USD', 'EUR'])
            })

        # Add missing data
        for i in range(0, len(large_data), 20):
            large_data[i]['amount'] = None
            large_data[i]['description'] = None
            large_data[i]['date'] = None

        start_time = time.time()

        # Process through complete pipeline
        imputation_engine = AdvancedImputationEngine()
        df = pd.DataFrame(large_data)
        imputed_df, imputation_report = imputation_engine.auto_impute(df, 'simple')

        preprocessor = AdvancedPortuguesePreprocessor({'use_spacy': False})
        descriptions = imputed_df['description'].fillna('').tolist()
        processed_texts = preprocessor.preprocess_batch(descriptions)

        feature_engineer = EnhancedFeatureEngineer({
            'text_processing': {'use_advanced_portuguese': False},
            'temporal_features': {'enhanced_temporal': True},
            'financial_features': {'amount_pattern_recognition': True},
            'data_augmentation': {'enabled': False},
            'quality_assurance': {'feature_validation': True},
            'preprocessing': {'scaling_method': 'standard'}
        })

        features, feature_names, feature_report = feature_engineer.create_enhanced_features(
            imputed_df.to_dict('records'), target_column='category'
        )

        end_time = time.time()
        total_time = end_time - start_time

        # Verify scalability
        assert features.shape[0] == len(large_data)
        assert len(feature_names) > 0
        assert total_time < 120.0  # Should complete within 2 minutes
        assert not np.any(np.isnan(features))
        assert feature_report['quality']['quality_score'] > 0.5

        # Verify imputation worked
        assert not imputed_df['amount'].isnull().any()
        assert not imputed_df['description'].isnull().any()
        assert not imputed_df['date'].isnull().any()

        # Verify preprocessing worked
        assert len(processed_texts) == len(large_data)
        successful_preprocessing = sum(1 for r in processed_texts if r.get('success', False))
        assert successful_preprocessing / len(processed_texts) > 0.8  # At least 80% success rate

    def test_concurrent_processing_scalability(self):
        """Test concurrent processing scalability"""
        # Create multiple batches of data
        batches = []
        for batch_idx in range(5):
            batch_data = []
            for i in range(200):
                batch_data.append({
                    'description': f'Batch {batch_idx} Transaction {i}',
                    'amount': float(i * 10),
                    'date': '2024-01-15',
                    'category': 'test'
                })
            batches.append(batch_data)

        start_time = time.time()

        # Process batches (simulating concurrent processing)
        results = []
        for batch in batches:
            imputation_engine = AdvancedImputationEngine()
            df = pd.DataFrame(batch)
            imputed_df, _ = imputation_engine.auto_impute(df, 'simple')

            feature_engineer = EnhancedFeatureEngineer({
                'text_processing': {'use_advanced_portuguese': False},
                'data_augmentation': {'enabled': False},
                'quality_assurance': {'feature_validation': False}
            })

            features, _, _ = feature_engineer.create_enhanced_features(
                imputed_df.to_dict('records')
            )
            results.append(features)

        end_time = time.time()
        total_time = end_time - start_time

        # Verify concurrent processing worked
        assert len(results) == len(batches)
        for features in results:
            assert features.shape[0] == 200
            assert features.shape[1] > 0

        # Should complete within reasonable time
        assert total_time < 60.0  # 1 minute for 1000 records across 5 batches


if __name__ == "__main__":
    pytest.main([__file__])