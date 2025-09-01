import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.services.data_completeness_analyzer import DataCompletenessAnalyzer
from src.utils.exceptions import ValidationError


class TestDataCompletenessAnalyzer:
    """Test suite for DataCompletenessAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        """Create a DataCompletenessAnalyzer instance for testing"""
        return DataCompletenessAnalyzer()

    @pytest.fixture
    def sample_complete_data(self):
        """Create sample complete transaction data"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'amount': [100.50, -50.25, 200.00, -75.10, 150.75],
            'description': ['Payment A', 'Purchase B', 'Income C', 'Expense D', 'Payment E'],
            'transaction_type': ['credit', 'debit', 'credit', 'debit', 'credit'],
            'balance': [1000.00, 949.75, 1149.75, 1074.65, 1225.40]
        })

    @pytest.fixture
    def sample_incomplete_data(self):
        """Create sample incomplete transaction data"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'date': ['2023-01-01', None, '2023-01-03', '2023-01-04', None],
            'amount': [100.50, -50.25, None, -75.10, 150.75],
            'description': ['Payment A', None, 'Income C', 'Expense D', ''],
            'transaction_type': ['credit', 'debit', None, 'debit', 'credit'],
            'balance': [1000.00, None, 1149.75, 1074.65, None]
        })

    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert hasattr(analyzer, 'completeness_results')
        assert hasattr(analyzer, 'missing_patterns')
        assert hasattr(analyzer, 'trend_analysis')
        assert analyzer.config is not None

    def test_analyze_field_completeness_complete_data(self, analyzer, sample_complete_data):
        """Test field completeness analysis with complete data"""
        result = analyzer.analyze_field_completeness(sample_complete_data, 'amount')

        assert result['field_name'] == 'amount'
        assert result['completeness_score'] == 1.0
        assert result['total_records'] == 5
        assert result['missing_records'] == 0
        assert result['missing_percentage'] == 0.0
        assert result['is_critical'] is True

    def test_analyze_field_completeness_incomplete_data(self, analyzer, sample_incomplete_data):
        """Test field completeness analysis with incomplete data"""
        result = analyzer.analyze_field_completeness(sample_incomplete_data, 'amount')

        assert result['field_name'] == 'amount'
        assert result['completeness_score'] == 0.8  # 4 out of 5 records have amount
        assert result['total_records'] == 5
        assert result['missing_records'] == 1
        assert result['missing_percentage'] == 20.0

    def test_analyze_field_completeness_nonexistent_field(self, analyzer, sample_complete_data):
        """Test field completeness analysis with nonexistent field"""
        result = analyzer.analyze_field_completeness(sample_complete_data, 'nonexistent_field')

        assert result['field_name'] == 'nonexistent_field'
        assert result['completeness_score'] == 0.0
        assert 'error' in result

    def test_analyze_record_completeness(self, analyzer, sample_incomplete_data):
        """Test record completeness analysis"""
        results = analyzer.analyze_record_completeness(sample_incomplete_data)

        assert len(results) == 5
        assert all('completeness_score' in r for r in results)
        assert all('missing_fields' in r for r in results)
        assert all('is_complete' in r for r in results)

        # Check that results are sorted by completeness (worst first)
        scores = [r['completeness_score'] for r in results]
        assert scores == sorted(scores)

    def test_analyze_dataset_completeness_complete(self, analyzer, sample_complete_data):
        """Test dataset completeness analysis with complete data"""
        result = analyzer.analyze_dataset_completeness(sample_complete_data)

        assert 'dataset_info' in result
        assert 'overall_completeness' in result
        assert 'field_completeness' in result
        assert 'critical_fields' in result
        assert 'summary' in result

        assert result['overall_completeness']['score'] == 1.0
        assert result['overall_completeness']['is_acceptable'] is True

    def test_analyze_dataset_completeness_incomplete(self, analyzer, sample_incomplete_data):
        """Test dataset completeness analysis with incomplete data"""
        result = analyzer.analyze_dataset_completeness(sample_incomplete_data)

        assert result['overall_completeness']['score'] < 1.0
        assert len(result['field_completeness']) == 6  # All columns
        assert 'critical_fields' in result

    def test_identify_missing_patterns(self, analyzer, sample_incomplete_data):
        """Test missing pattern identification"""
        patterns = analyzer._identify_missing_patterns(sample_incomplete_data)

        assert isinstance(patterns, list)
        if patterns:  # Only if patterns are found
            for pattern in patterns:
                assert 'fields' in pattern
                assert 'correlation' in pattern
                assert 'support' in pattern

    def test_analyze_completeness_trends(self, analyzer):
        """Test completeness trend analysis"""
        # Create historical data
        base_date = datetime.now()
        historical_data = [
            (base_date - timedelta(days=2), pd.DataFrame({
                'date': ['2023-01-01', '2023-01-02'],
                'amount': [100, 200],
                'description': ['A', 'B']
            })),
            (base_date - timedelta(days=1), pd.DataFrame({
                'date': ['2023-01-01', None],
                'amount': [100, None],
                'description': ['A', None]
            })),
            (base_date, pd.DataFrame({
                'date': ['2023-01-01', '2023-01-02', None],
                'amount': [100, 200, None],
                'description': ['A', 'B', None]
            }))
        ]

        result = analyzer.analyze_completeness_trends(historical_data)

        assert 'field_trends' in result
        assert 'overall_trends' in result
        assert 'improvement_areas' in result
        assert len(result['overall_trends']) == 3

    def test_generate_completeness_report(self, analyzer, sample_incomplete_data):
        """Test completeness report generation"""
        report = analyzer.generate_completeness_report(sample_incomplete_data)

        assert 'timestamp' in report
        assert 'dataset_analysis' in report
        assert 'record_analysis_summary' in report
        assert 'recommendations' in report

        assert isinstance(report['recommendations'], list)

    def test_generate_completeness_report_with_recommendations(self, analyzer, sample_incomplete_data):
        """Test completeness report generation with recommendations"""
        report = analyzer.generate_completeness_report(sample_incomplete_data, include_recommendations=True)

        assert 'recommendations' in report
        assert isinstance(report['recommendations'], list)

        if report['recommendations']:
            for rec in report['recommendations']:
                assert 'priority' in rec
                assert 'category' in rec
                assert 'recommendation' in rec

    def test_empty_dataframe_handling(self, analyzer):
        """Test handling of empty dataframes"""
        empty_df = pd.DataFrame()

        result = analyzer.analyze_dataset_completeness(empty_df)
        assert 'error' in result or result['dataset_info']['total_records'] == 0

    def test_single_record_handling(self, analyzer):
        """Test handling of single record dataframes"""
        single_record = pd.DataFrame({
            'date': ['2023-01-01'],
            'amount': [100.0],
            'description': ['Test']
        })

        result = analyzer.analyze_dataset_completeness(single_record)
        assert result['dataset_info']['total_records'] == 1
        assert result['overall_completeness']['score'] == 1.0

    def test_numeric_field_metrics(self, analyzer):
        """Test numeric field-specific metrics"""
        data = pd.DataFrame({
            'amount': [100, 200, 300, None, 500],
            'balance': [1000, 1100, 1200, 1300, 1400]
        })

        result = analyzer.analyze_field_completeness(data, 'amount')

        assert 'mean_value' in result
        assert 'median_value' in result
        assert 'std_value' in result
        assert 'min_value' in result
        assert 'max_value' in result

    def test_text_field_metrics(self, analyzer):
        """Test text field-specific metrics"""
        data = pd.DataFrame({
            'description': ['Payment A', 'Purchase B', '', None, 'Income C']
        })

        result = analyzer.analyze_field_completeness(data, 'description')

        assert 'avg_length' in result
        assert 'empty_strings' in result
        assert 'most_common_value' in result

    def test_date_field_metrics(self, analyzer):
        """Test date field-specific metrics"""
        data = pd.DataFrame({
            'date': ['2023-01-01', '2023-06-15', '2023-12-31', None, '2024-01-01']
        })

        result = analyzer.analyze_field_completeness(data, 'date')

        assert 'date_range_days' in result
        assert 'future_dates' in result
        assert 'past_dates' in result

    def test_config_customization(self):
        """Test analyzer with custom configuration"""
        custom_config = {
            'completeness_thresholds': {
                'field_level': 0.9,
                'record_level': 0.8,
                'dataset_level': 0.85
            },
            'critical_fields': ['date', 'amount']
        }

        analyzer = DataCompletenessAnalyzer(custom_config)
        assert analyzer.config['completeness_thresholds']['field_level'] == 0.9
        assert analyzer.config['critical_fields'] == ['date', 'amount']

    @patch('src.services.data_completeness_analyzer.get_logger')
    def test_logging_integration(self, mock_get_logger, analyzer, sample_incomplete_data):
        """Test that logging is properly integrated"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        analyzer.analyze_dataset_completeness(sample_incomplete_data)

        mock_logger.info.assert_called()

    def test_error_handling_invalid_data_type(self, analyzer):
        """Test error handling with invalid data types"""
        with pytest.raises(ValidationError):
            analyzer._ensure_dataframe("invalid_data_type")

    def test_missing_data_pattern_correlation_calculation(self, analyzer):
        """Test missing data pattern correlation calculation"""
        # Create data with correlated missing patterns
        data = pd.DataFrame({
            'field1': [1, 2, None, 4, None],
            'field2': [1, 2, None, 4, None],  # Perfect correlation with field1
            'field3': [1, None, 3, None, 5]  # No correlation
        })

        patterns = analyzer._identify_missing_patterns(data)

        # Should find correlation between field1 and field2
        correlated_patterns = [p for p in patterns if p.get('correlation', 0) > 0.5]
        assert len(correlated_patterns) > 0

    def test_trend_analysis_edge_cases(self, analyzer):
        """Test trend analysis with edge cases"""
        # Empty historical data
        result = analyzer.analyze_completeness_trends([])
        assert 'field_trends' in result
        assert 'overall_trends' in result
        assert len(result['overall_trends']) == 0

        # Single data point
        single_point = [(datetime.now(), pd.DataFrame({'col': [1, 2, 3]}))]
        result = analyzer.analyze_completeness_trends(single_point)
        assert len(result['overall_trends']) == 1