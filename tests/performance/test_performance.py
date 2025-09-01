#!/usr/bin/env python3
"""
Performance tests for ML algorithms and feature engineering
"""

import pytest
import time
import numpy as np
import psutil
import os
from unittest.mock import Mock, patch
from memory_profiler import profile as memory_profile

from src.services.model_manager import ModelManager, RandomForestModel, XGBoostModel, LightGBMModel
from src.services.feature_engineer import FeatureEngineer


@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing"""
    np.random.seed(42)
    n_samples = 10000
    n_features = 50

    # Generate transaction-like data
    descriptions = [
        'Compra no supermercado Extra',
        'Pagamento de conta de luz CEMIG',
        'Transferência PIX recebida',
        'Restaurante jantar familia',
        'Combustível posto Ipiranga',
        'Compra online Amazon',
        'Pagamento internet',
        'Salário depositado',
        'Compra farmacia',
        'Pagamento aluguel'
    ] * 1000

    categories = ['alimentacao', 'casa', 'transferencia', 'lazer', 'transporte'] * 2000

    data = []
    base_date = "2024-01-01"

    for i in range(n_samples):
        data.append({
            'description': descriptions[i % len(descriptions)],
            'amount': np.random.normal(100, 50),
            'date': base_date,
            'category': categories[i % len(categories)],
            'type': 'debit' if np.random.random() < 0.8 else 'credit'
        })

    return data


@pytest.fixture
def medium_dataset():
    """Generate medium dataset for performance testing"""
    np.random.seed(42)
    n_samples = 1000

    descriptions = ['Compra mercado', 'Pagamento luz', 'Transferência', 'Restaurante']
    categories = ['alimentacao', 'casa', 'transferencia', 'lazer']

    data = []
    for i in range(n_samples):
        data.append({
            'description': descriptions[i % len(descriptions)],
            'amount': np.random.uniform(10, 500),
            'date': '2024-01-01',
            'category': categories[i % len(categories)]
        })

    return data


@pytest.fixture
def mock_feature_engineer():
    """Mock FeatureEngineer for performance testing"""
    mock_fe = Mock()
    mock_fe.create_comprehensive_features.return_value = (
        np.random.rand(1000, 100),  # Features
        [f'feature_{i}' for i in range(100)]  # Feature names
    )
    return mock_fe


@pytest.fixture
def performance_model_manager(mock_feature_engineer):
    """ModelManager instance for performance testing"""
    with patch('src.services.model_manager.FeatureEngineer', return_value=mock_feature_engineer), \
         patch('src.services.model_manager.ModelSelector'):

        manager = ModelManager()
        return manager


class TestFeatureEngineeringPerformance:
    """Performance tests for feature engineering"""

    def test_feature_engineering_large_dataset(self, large_dataset):
        """Test feature engineering performance with large dataset"""
        engineer = FeatureEngineer()

        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        features, feature_names = engineer.create_comprehensive_features(large_dataset)

        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        processing_time = end_time - start_time
        memory_usage = end_memory - start_memory

        print(f"Feature engineering performance:")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Memory usage: {memory_usage:.2f}MB")
        print(f"  Features shape: {features.shape}")

        # Performance assertions
        assert processing_time < 30.0, f"Feature engineering too slow: {processing_time:.2f}s"
        assert features.shape[0] == len(large_dataset)
        assert len(feature_names) > 0

    def test_text_embedding_performance(self):
        """Test text embedding extraction performance"""
        engineer = FeatureEngineer()

        # Generate many texts
        texts = [f"Transaction description {i} with some random text" for i in range(1000)]

        start_time = time.time()
        embeddings = engineer.extract_text_embeddings(texts)
        processing_time = time.time() - start_time

        print(f"Text embedding performance:")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Embeddings shape: {embeddings.shape}")

        assert processing_time < 10.0, f"Text embedding too slow: {processing_time:.2f}s"
        assert embeddings.shape[0] == len(texts)

    def test_temporal_feature_performance(self):
        """Test temporal feature extraction performance"""
        engineer = FeatureEngineer()

        # Generate many dates
        dates = [f"2024-{month:02d}-{day:02d}" for month in range(1, 13) for day in range(1, 28)] * 10

        start_time = time.time()
        temporal_df = engineer.extract_temporal_features(dates)
        processing_time = time.time() - start_time

        print(f"Temporal feature performance:")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Features shape: {temporal_df.shape}")

        assert processing_time < 5.0, f"Temporal features too slow: {processing_time:.2f}s"
        assert temporal_df.shape[0] == len(dates)

    def test_transaction_pattern_performance(self, large_dataset):
        """Test transaction pattern extraction performance"""
        engineer = FeatureEngineer()

        start_time = time.time()
        pattern_df = engineer.extract_transaction_patterns(large_dataset)
        processing_time = time.time() - start_time

        print(f"Transaction pattern performance:")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Features shape: {pattern_df.shape}")

        assert processing_time < 10.0, f"Transaction patterns too slow: {processing_time:.2f}s"
        assert pattern_df.shape[0] == len(large_dataset)


class TestModelTrainingPerformance:
    """Performance tests for model training"""

    def test_random_forest_training_performance(self, performance_model_manager, medium_dataset, mock_feature_engineer):
        """Test Random Forest training performance"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(len(medium_dataset), 50),
            [f'feature_{i}' for i in range(50)]
        )

        X, y, _ = performance_model_manager.process_data(medium_dataset)

        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        result = performance_model_manager.train_model('random_forest', X, y)

        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        training_time = end_time - start_time
        memory_usage = end_memory - start_memory

        print(f"Random Forest training performance:")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Memory usage: {memory_usage:.2f}MB")

        assert result['success'] is True
        assert training_time < 10.0, f"Random Forest training too slow: {training_time:.2f}s"

    def test_xgboost_training_performance(self, performance_model_manager, medium_dataset, mock_feature_engineer):
        """Test XGBoost training performance"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(len(medium_dataset), 50),
            [f'feature_{i}' for i in range(50)]
        )

        X, y, _ = performance_model_manager.process_data(medium_dataset)

        start_time = time.time()
        result = performance_model_manager.train_model('xgboost', X, y)
        training_time = time.time() - start_time

        print(f"XGBoost training performance:")
        print(f"  Training time: {training_time:.2f}s")

        assert result['success'] is True
        assert training_time < 15.0, f"XGBoost training too slow: {training_time:.2f}s"

    def test_lightgbm_training_performance(self, performance_model_manager, medium_dataset, mock_feature_engineer):
        """Test LightGBM training performance"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(len(medium_dataset), 50),
            [f'feature_{i}' for i in range(50)]
        )

        X, y, _ = performance_model_manager.process_data(medium_dataset)

        start_time = time.time()
        result = performance_model_manager.train_model('lightgbm', X, y)
        training_time = time.time() - start_time

        print(f"LightGBM training performance:")
        print(f"  Training time: {training_time:.2f}s")

        assert result['success'] is True
        assert training_time < 10.0, f"LightGBM training too slow: {training_time:.2f}s"

    def test_kmeans_training_performance(self, performance_model_manager, medium_dataset, mock_feature_engineer):
        """Test KMeans training performance"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(len(medium_dataset), 50),
            [f'feature_{i}' for i in range(50)]
        )

        X, y, _ = performance_model_manager.process_data(medium_dataset)

        start_time = time.time()
        result = performance_model_manager.train_model('kmeans', X, y)
        training_time = time.time() - start_time

        print(f"KMeans training performance:")
        print(f"  Training time: {training_time:.2f}s")

        assert result['success'] is True
        assert training_time < 5.0, f"KMeans training too slow: {training_time:.2f}s"


class TestModelPredictionPerformance:
    """Performance tests for model prediction"""

    def test_batch_prediction_performance(self, performance_model_manager, medium_dataset, mock_feature_engineer):
        """Test batch prediction performance"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(len(medium_dataset), 50),
            [f'feature_{i}' for i in range(50)]
        )

        # Train model first
        X, y, _ = performance_model_manager.process_data(medium_dataset)
        performance_model_manager.train_model('random_forest', X, y)

        # Test batch prediction
        batch_sizes = [10, 50, 100, 500]

        for batch_size in batch_sizes:
            test_X = np.random.rand(batch_size, 50)

            start_time = time.time()
            predictions = performance_model_manager.predict('random_forest', test_X)
            prediction_time = time.time() - start_time

            print(f"Batch prediction performance (size={batch_size}):")
            print(f"  Prediction time: {prediction_time:.4f}s")
            print(f"  Time per sample: {prediction_time/batch_size:.6f}s")

            assert len(predictions) == batch_size
            assert prediction_time < 1.0, f"Batch prediction too slow: {prediction_time:.4f}s"

    def test_prediction_probability_performance(self, performance_model_manager, medium_dataset, mock_feature_engineer):
        """Test prediction probability performance"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(len(medium_dataset), 50),
            [f'feature_{i}' for i in range(50)]
        )

        # Train model first
        X, y, _ = performance_model_manager.process_data(medium_dataset)
        performance_model_manager.train_model('random_forest', X, y)

        # Test probability prediction
        test_X = np.random.rand(100, 50)

        start_time = time.time()
        probabilities = performance_model_manager.predict_proba('random_forest', test_X)
        prediction_time = time.time() - start_time

        print(f"Probability prediction performance:")
        print(f"  Prediction time: {prediction_time:.4f}s")
        print(f"  Probabilities shape: {probabilities.shape}")

        assert probabilities.shape[0] == 100
        assert prediction_time < 0.5, f"Probability prediction too slow: {prediction_time:.4f}s"


class TestModelComparisonPerformance:
    """Performance tests for model comparison"""

    def test_model_comparison_performance(self, performance_model_manager, medium_dataset, mock_feature_engineer):
        """Test model comparison performance"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(len(medium_dataset), 50),
            [f'feature_{i}' for i in range(50)]
        )

        X, y, _ = performance_model_manager.process_data(medium_dataset)

        models_to_compare = ['random_forest', 'xgboost', 'lightgbm']

        start_time = time.time()
        comparison_result = performance_model_manager.compare_models(X, y, models_to_compare)
        comparison_time = time.time() - start_time

        print(f"Model comparison performance:")
        print(f"  Comparison time: {comparison_time:.2f}s")
        print(f"  Models compared: {len(models_to_compare)}")

        assert 'best_model' in comparison_result
        assert comparison_time < 60.0, f"Model comparison too slow: {comparison_time:.2f}s"


class TestHyperparameterOptimizationPerformance:
    """Performance tests for hyperparameter optimization"""

    def test_optimization_performance_small(self, performance_model_manager, mock_feature_engineer):
        """Test optimization performance with small dataset"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(200, 20),
            [f'feature_{i}' for i in range(20)]
        )

        X = np.random.rand(200, 20)
        y = np.random.choice(['class_a', 'class_b'], 200)

        start_time = time.time()
        result = performance_model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=5)
        optimization_time = time.time() - start_time

        print(f"Hyperparameter optimization performance (small):")
        print(f"  Optimization time: {optimization_time:.2f}s")
        print(f"  Trials: 5")

        assert result['success'] is True
        assert optimization_time < 30.0, f"Optimization too slow: {optimization_time:.2f}s"

    def test_optimization_performance_large_trials(self, performance_model_manager):
        """Test optimization performance with more trials"""
        X = np.random.rand(300, 15)
        y = np.random.choice(['class_a', 'class_b', 'class_c'], 300)

        start_time = time.time()
        result = performance_model_manager.optimize_hyperparameters('random_forest', X, y, n_trials=20)
        optimization_time = time.time() - start_time

        print(f"Hyperparameter optimization performance (large):")
        print(f"  Optimization time: {optimization_time:.2f}s")
        print(f"  Trials: 20")

        assert result['success'] is True
        assert optimization_time < 120.0, f"Optimization too slow: {optimization_time:.2f}s"


class TestMemoryUsage:
    """Test memory usage of different operations"""

    def test_memory_usage_during_training(self, performance_model_manager, medium_dataset, mock_feature_engineer):
        """Test memory usage during model training"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(len(medium_dataset), 100),
            [f'feature_{i}' for i in range(100)]
        )

        X, y, _ = performance_model_manager.process_data(medium_dataset)

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Train model
        result = performance_model_manager.train_model('random_forest', X, y)

        peak_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory

        print(f"Memory usage during training:")
        print(f"  Initial memory: {initial_memory:.2f}MB")
        print(f"  Peak memory: {peak_memory:.2f}MB")
        print(f"  Memory increase: {memory_increase:.2f}MB")

        assert result['success'] is True
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f}MB"

    def test_memory_usage_during_feature_engineering(self, large_dataset):
        """Test memory usage during feature engineering"""
        engineer = FeatureEngineer()

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        features, _ = engineer.create_comprehensive_features(large_dataset)

        peak_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory

        print(f"Memory usage during feature engineering:")
        print(f"  Initial memory: {initial_memory:.2f}MB")
        print(f"  Peak memory: {peak_memory:.2f}MB")
        print(f"  Memory increase: {memory_increase:.2f}MB")
        print(f"  Features shape: {features.shape}")

        assert memory_increase < 1000, f"Memory usage too high: {memory_increase:.2f}MB"


class TestScalability:
    """Test scalability of algorithms with different data sizes"""

    @pytest.mark.parametrize("n_samples", [100, 500, 1000, 5000])
    def test_scalability_training_time(self, performance_model_manager, mock_feature_engineer, n_samples):
        """Test how training time scales with data size"""
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(n_samples, 50),
            [f'feature_{i}' for i in range(50)]
        )

        # Create dataset of specified size
        data = [{'description': f'Test {i}', 'amount': 100.0, 'date': '2024-01-01', 'category': 'test'}
                for i in range(n_samples)]

        X, y, _ = performance_model_manager.process_data(data)

        start_time = time.time()
        result = performance_model_manager.train_model('random_forest', X, y)
        training_time = time.time() - start_time

        print(f"Scalability test (n_samples={n_samples}):")
        print(f"  Training time: {training_time:.2f}s")

        assert result['success'] is True

        # Training time should scale reasonably (not exponentially)
        if n_samples <= 1000:
            assert training_time < 30.0, f"Training too slow for {n_samples} samples: {training_time:.2f}s"

    @pytest.mark.parametrize("n_features", [10, 50, 100, 200])
    def test_scalability_feature_count(self, performance_model_manager, mock_feature_engineer, n_features):
        """Test how performance scales with feature count"""
        n_samples = 500
        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(n_samples, n_features),
            [f'feature_{i}' for i in range(n_features)]
        )

        data = [{'description': f'Test {i}', 'amount': 100.0, 'date': '2024-01-01', 'category': 'test'}
                for i in range(n_samples)]

        X, y, _ = performance_model_manager.process_data(data)

        start_time = time.time()
        result = performance_model_manager.train_model('random_forest', X, y)
        training_time = time.time() - start_time

        print(f"Feature scalability test (n_features={n_features}):")
        print(f"  Training time: {training_time:.2f}s")

        assert result['success'] is True
        assert training_time < 60.0, f"Training too slow for {n_features} features: {training_time:.2f}s"


class TestConcurrentPerformance:
    """Test performance under concurrent operations"""

    def test_concurrent_predictions(self, performance_model_manager, medium_dataset, mock_feature_engineer):
        """Test concurrent prediction performance"""
        import threading

        mock_feature_engineer.create_comprehensive_features.return_value = (
            np.random.rand(len(medium_dataset), 50),
            [f'feature_{i}' for i in range(50)]
        )

        # Train model
        X, y, _ = performance_model_manager.process_data(medium_dataset)
        performance_model_manager.train_model('random_forest', X, y)

        results = []
        errors = []

        def predict_worker(worker_id):
            try:
                test_X = np.random.rand(50, 50)
                start_time = time.time()
                predictions = performance_model_manager.predict('random_forest', test_X)
                end_time = time.time()

                results.append({
                    'worker_id': worker_id,
                    'time': end_time - start_time,
                    'predictions': len(predictions)
                })
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")

        # Start concurrent predictions
        threads = []
        for i in range(5):  # 5 concurrent predictions
            thread = threading.Thread(target=predict_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        print(f"Concurrent prediction results:")
        print(f"  Successful predictions: {len(results)}")
        print(f"  Errors: {len(errors)}")
        print(f"  Average time: {np.mean([r['time'] for r in results]):.4f}s")

        assert len(results) == 5, f"Expected 5 successful predictions, got {len(results)}"
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Average prediction time should be reasonable
        avg_time = np.mean([r['time'] for r in results])
        assert avg_time < 2.0, f"Average prediction time too slow: {avg_time:.4f}s"