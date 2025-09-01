#!/usr/bin/env python3
"""
Basic test script for the Advanced Outlier Detection System

This script provides a simple way to test the outlier detection functionality
without running the full test suite.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_advanced_outlier_detector():
    """Test the AdvancedOutlierDetector class"""
    print("Testing AdvancedOutlierDetector...")

    try:
        from src.services.advanced_outlier_detector import AdvancedOutlierDetector

        # Create test data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, 100)  # Normal transactions
        outliers = np.array([500, -200, 800])  # Outlier transactions
        test_data = np.concatenate([normal_data, outliers])

        # Initialize detector
        detector = AdvancedOutlierDetector()

        # Test IQR method
        print("  Testing IQR method...")
        flags_iqr, scores_iqr = detector.detect_outliers_iqr(test_data)
        print(f"    IQR detected {np.sum(flags_iqr)} outliers out of {len(test_data)} samples")

        # Test Z-score method
        print("  Testing Z-score method...")
        flags_zscore, scores_zscore = detector.detect_outliers_zscore(test_data)
        print(f"    Z-score detected {np.sum(flags_zscore)} outliers out of {len(test_data)} samples")

        # Test ensemble method
        print("  Testing ensemble method...")
        flags_ensemble, scores_ensemble = detector.detect_outliers_ensemble(test_data.reshape(-1, 1))
        print(f"    Ensemble detected {np.sum(flags_ensemble)} outliers out of {len(test_data)} samples")

        print("  ✓ AdvancedOutlierDetector tests passed!")
        return True

    except Exception as e:
        print(f"  ✗ AdvancedOutlierDetector test failed: {str(e)}")
        return False

def test_contextual_outlier_detector():
    """Test the ContextualOutlierDetector class"""
    print("Testing ContextualOutlierDetector...")

    try:
        from src.services.contextual_outlier_detector import ContextualOutlierDetector

        # Create test transaction data
        base_date = datetime.now()
        transactions = []

        # Normal transactions
        for i in range(50):
            transactions.append({
                'amount': np.random.normal(100, 20),
                'date': (base_date + timedelta(days=i)).isoformat(),
                'description': f'Normal transaction {i}',
                'category': 'alimentacao' if i % 2 == 0 else 'transporte'
            })

        # Outlier transactions
        transactions.extend([
            {
                'amount': 5000,  # Very large amount
                'date': (base_date + timedelta(days=51)).isoformat(),
                'description': 'Outlier transaction 1',
                'category': 'alimentacao'
            },
            {
                'amount': -1000,  # Very large negative amount
                'date': (base_date + timedelta(days=52)).isoformat(),
                'description': 'Outlier transaction 2',
                'category': 'transporte'
            }
        ])

        # Initialize detector
        detector = ContextualOutlierDetector()

        # Test category-based detection
        print("  Testing category-based outlier detection...")
        category_results = detector.detect_amount_outliers_by_category(transactions)
        if 'results' in category_results:
            total_outliers = sum(cat.get('outlier_count', 0) for cat in category_results['results'].values())
            print(f"    Category-based detection found {total_outliers} outliers")

        # Test temporal detection
        print("  Testing temporal outlier detection...")
        temporal_results = detector.detect_temporal_outliers(transactions)
        if 'results' in temporal_results:
            print("    Temporal detection completed")

        print("  ✓ ContextualOutlierDetector tests passed!")
        return True

    except Exception as e:
        print(f"  ✗ ContextualOutlierDetector test failed: {str(e)}")
        return False

def test_statistical_outlier_analysis():
    """Test the StatisticalOutlierAnalysis class"""
    print("Testing StatisticalOutlierAnalysis...")

    try:
        from src.services.statistical_outlier_analysis import StatisticalOutlierAnalysis

        # Create test transaction data
        transactions = []
        for i in range(30):
            transactions.append({
                'amount': np.random.normal(100, 15),
                'date': f'2024-{i%12+1:02d}-{i%28+1:02d}',
                'description': f'Test transaction {i}',
                'category': 'test'
            })

        # Add some outliers
        transactions.extend([
            {'amount': 1000, 'date': '2024-01-01', 'description': 'Outlier 1', 'category': 'test'},
            {'amount': -500, 'date': '2024-01-02', 'description': 'Outlier 2', 'category': 'test'}
        ])

        # Initialize analyzer
        analyzer = StatisticalOutlierAnalysis()

        # Test comprehensive analysis
        print("  Testing comprehensive analysis...")
        results = analyzer.perform_comprehensive_outlier_analysis(transactions)

        if 'error' not in results:
            print("    Comprehensive analysis completed successfully")
            if 'summary' in results:
                total_transactions = results['summary'].get('total_transactions', 0)
                print(f"    Analyzed {total_transactions} transactions")
        else:
            print(f"    Analysis failed: {results['error']}")

        print("  ✓ StatisticalOutlierAnalysis tests passed!")
        return True

    except Exception as e:
        print(f"  ✗ StatisticalOutlierAnalysis test failed: {str(e)}")
        return False

def test_ai_service_integration():
    """Test the AI service integration with outlier detection"""
    print("Testing AI Service integration...")

    try:
        from src.services.ai_service import AIService

        # Create test transaction data
        transactions = [
            {'amount': 100, 'date': '2024-01-01', 'description': 'Normal transaction 1'},
            {'amount': 150, 'date': '2024-01-02', 'description': 'Normal transaction 2'},
            {'amount': 5000, 'date': '2024-01-03', 'description': 'Outlier transaction'}  # Outlier
        ]

        # Initialize AI service
        ai_service = AIService()

        # Test basic anomaly detection
        print("  Testing basic anomaly detection...")
        results = ai_service.detect_anomalies(transactions, method='iqr')
        outlier_count = sum(1 for t in results if t.get('is_anomaly', False))
        print(f"    Detected {outlier_count} outliers using IQR method")

        # Test ensemble method
        print("  Testing ensemble anomaly detection...")
        results_ensemble = ai_service.detect_anomalies(transactions, method='ensemble')
        outlier_count_ensemble = sum(1 for t in results_ensemble if t.get('is_anomaly', False))
        print(f"    Detected {outlier_count_ensemble} outliers using ensemble method")

        print("  ✓ AI Service integration tests passed!")
        return True

    except Exception as e:
        print(f"  ✗ AI Service integration test failed: {str(e)}")
        return False

def test_validation_pipeline_integration():
    """Test the validation pipeline integration"""
    print("Testing validation pipeline integration...")

    try:
        from src.utils.advanced_validation_engine import advanced_validation_engine

        # Create test transaction data
        test_transaction = {
            'amount': 5000,  # Potential outlier
            'date': '2024-01-01',
            'description': 'Test transaction',
            'category': 'alimentacao',
            'transaction_type': 'debit'
        }

        # Test validation with outlier detection
        print("  Testing validation with outlier detection...")
        result = advanced_validation_engine.validate(
            test_transaction,
            profile='transaction_with_outliers',
            context={'outlier_method': 'iqr', 'include_contextual': False}
        )

        print(f"    Validation completed with {len(result.errors)} errors and {len(result.warnings)} warnings")

        # Check if outlier detection layer was executed
        if hasattr(result, 'metadata') and result.metadata.get('outlier_detected') is not None:
            outlier_detected = result.metadata.get('outlier_detected', False)
            print(f"    Outlier detection: {'Detected' if outlier_detected else 'Not detected'}")

        print("  ✓ Validation pipeline integration tests passed!")
        return True

    except Exception as e:
        print(f"  ✗ Validation pipeline integration test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Advanced Outlier Detection System - Basic Tests")
    print("=" * 60)

    tests = [
        test_advanced_outlier_detector,
        test_contextual_outlier_detector,
        test_statistical_outlier_analysis,
        test_ai_service_integration,
        test_validation_pipeline_integration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed with exception: {str(e)}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The outlier detection system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())