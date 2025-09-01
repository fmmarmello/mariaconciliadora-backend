import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime

from src.services.quality_assessment_engine import QualityAssessmentEngine
from src.utils.exceptions import ValidationError


class TestQualityAssessmentEngine:
    """Test cases for QualityAssessmentEngine class"""

    @pytest.fixture
    def quality_engine(self):
        """Create quality assessment engine instance"""
        config = {}
        return QualityAssessmentEngine(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample original and synthetic datasets"""
        np.random.seed(42)

        # Original data
        n_samples = 1000
        n_features = 5

        original_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(2, 1.5, n_samples),
            'feature_3': np.random.normal(-1, 0.5, n_samples),
            'amount': np.random.uniform(10, 1000, n_samples),
            'description': [f'transaction_{i}' for i in range(n_samples)],
            'category': np.random.choice(['receita', 'despesa', 'transferencia'], n_samples)
        })

        # Synthetic data (similar but with some differences)
        synthetic_data = pd.DataFrame({
            'feature_1': np.random.normal(0.1, 1.1, n_samples),
            'feature_2': np.random.normal(2.1, 1.6, n_samples),
            'feature_3': np.random.normal(-0.9, 0.6, n_samples),
            'amount': np.random.uniform(15, 950, n_samples),
            'description': [f'synthetic_transaction_{i}' for i in range(n_samples)],
            'category': np.random.choice(['receita', 'despesa', 'transferencia'], n_samples)
        })

        return original_data, synthetic_data

    def test_initialization(self, quality_engine):
        """Test quality engine initialization"""
        assert quality_engine.config == {}
        assert quality_engine.quality_reports == []
        assert quality_engine.baseline_metrics == {}
        assert 'statistical_tests' in dir(quality_engine)
        assert 'quality_thresholds' in dir(quality_engine)

    def test_assess_synthetic_quality(self, quality_engine, sample_data):
        """Test comprehensive quality assessment"""
        original_data, synthetic_data = sample_data

        metadata = {
            'generation_method': 'smote',
            'target_variable': 'category'
        }

        report = quality_engine.assess_synthetic_quality(
            original_data, synthetic_data, metadata
        )

        # Check report structure
        required_fields = [
            'timestamp', 'data_characteristics', 'statistical_similarity',
            'distribution_preservation', 'business_rule_compliance',
            'model_performance_impact', 'privacy_preservation',
            'overall_quality_score', 'quality_grade', 'recommendations'
        ]

        for field in required_fields:
            assert field in report

        # Check score is between 0 and 1
        assert 0 <= report['overall_quality_score'] <= 1

        # Check grade is valid
        assert report['quality_grade'] in ['A', 'B', 'C', 'D', 'F']

    def test_analyze_data_characteristics(self, quality_engine, sample_data):
        """Test data characteristics analysis"""
        original_data, synthetic_data = sample_data

        characteristics = quality_engine._analyze_data_characteristics(
            original_data, synthetic_data
        )

        assert 'original' in characteristics
        assert 'synthetic' in characteristics
        assert 'compatibility' in characteristics

        # Check original data characteristics
        orig = characteristics['original']
        assert 'n_samples' in orig
        assert 'n_features' in orig
        assert orig['n_samples'] == 1000
        assert orig['n_features'] == 6  # 5 features + description + category

    def test_assess_statistical_similarity(self, quality_engine, sample_data):
        """Test statistical similarity assessment"""
        original_data, synthetic_data = sample_data

        similarity = quality_engine._assess_statistical_similarity(
            original_data, synthetic_data
        )

        assert 'column_similarities' in similarity
        assert 'overall_similarity' in similarity
        assert 'numerical_columns_assessed' in similarity

        # Check that numerical columns were assessed
        assert similarity['numerical_columns_assessed'] > 0
        assert 0 <= similarity['overall_similarity'] <= 1

    def test_assess_distribution_preservation(self, quality_engine, sample_data):
        """Test distribution preservation assessment"""
        original_data, synthetic_data = sample_data

        preservation = quality_engine._assess_distribution_preservation(
            original_data, synthetic_data
        )

        assert 'column_distributions' in preservation
        assert 'overall_preservation' in preservation
        assert 'categorical_columns_assessed' in preservation

        assert 0 <= preservation['overall_preservation'] <= 1

    def test_assess_business_rule_compliance(self, quality_engine, sample_data):
        """Test business rule compliance assessment"""
        original_data, synthetic_data = sample_data

        compliance = quality_engine._assess_business_rule_compliance(synthetic_data)

        assert 'rule_compliance' in compliance
        assert 'overall_compliance' in compliance
        assert 'rules_checked' in compliance

        assert 0 <= compliance['overall_compliance'] <= 1

    def test_calculate_overall_quality_score(self, quality_engine):
        """Test overall quality score calculation"""
        # Mock assessment report
        report = {
            'statistical_similarity': {'overall_similarity': 0.8},
            'distribution_preservation': {'overall_preservation': 0.75},
            'business_rule_compliance': {'overall_compliance': 0.9},
            'model_performance_impact': {'overall_impact': 0.85},
            'privacy_preservation': {'overall_privacy': 0.8}
        }

        score = quality_engine._calculate_overall_quality_score(report)

        assert 0 <= score <= 1
        # Should be close to weighted average
        expected_score = (0.8 * 0.3 + 0.75 * 0.25 + 0.9 * 0.2 + 0.85 * 0.15 + 0.8 * 0.1)
        assert abs(score - expected_score) < 0.01

    def test_assign_quality_grade(self, quality_engine):
        """Test quality grade assignment"""
        test_cases = [
            (0.95, 'A'),
            (0.85, 'B'),
            (0.75, 'C'),
            (0.65, 'D'),
            (0.55, 'F'),
            (0.0, 'F')
        ]

        for score, expected_grade in test_cases:
            grade = quality_engine._assign_quality_grade(score)
            assert grade == expected_grade

    def test_generate_recommendations(self, quality_engine):
        """Test recommendations generation"""
        # High quality report
        good_report = {
            'overall_quality_score': 0.9,
            'statistical_similarity': {'overall_similarity': 0.9},
            'distribution_preservation': {'overall_preservation': 0.9},
            'business_rule_compliance': {'overall_compliance': 0.9}
        }

        recommendations = quality_engine._generate_recommendations(good_report)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Poor quality report
        poor_report = {
            'overall_quality_score': 0.3,
            'statistical_similarity': {'overall_similarity': 0.3},
            'distribution_preservation': {'overall_preservation': 0.3}
        }

        poor_recommendations = quality_engine._generate_recommendations(poor_report)
        assert len(poor_recommendations) > 0
        # Should contain improvement suggestions
        assert any('improve' in rec.lower() or 'adjust' in rec.lower()
                  for rec in poor_recommendations)

    def test_get_quality_history(self, quality_engine, sample_data):
        """Test quality history retrieval"""
        original_data, synthetic_data = sample_data

        # Perform assessment
        quality_engine.assess_synthetic_quality(original_data, synthetic_data)

        history = quality_engine.get_quality_history()
        assert len(history) == 1

        report = history[0]
        assert 'timestamp' in report
        assert 'overall_quality_score' in report
        assert 'quality_grade' in report

    def test_get_quality_summary(self, quality_engine, sample_data):
        """Test quality summary generation"""
        original_data, synthetic_data = sample_data

        # Perform multiple assessments
        quality_engine.assess_synthetic_quality(original_data, synthetic_data)
        quality_engine.assess_synthetic_quality(original_data, synthetic_data)

        summary = quality_engine.get_quality_summary()

        assert 'total_assessments' in summary
        assert 'average_score' in summary
        assert 'median_score' in summary
        assert 'best_score' in summary
        assert 'worst_score' in summary
        assert 'grade_distribution' in summary
        assert 'most_common_grade' in summary

        assert summary['total_assessments'] == 2

    def test_kolmogorov_smirnov_test(self, quality_engine):
        """Test Kolmogorov-Smirnov test implementation"""
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(0.5, 1, 100)

        result = quality_engine._kolmogorov_smirnov_test(data1, data2)

        assert 'statistic' in result
        assert 'p_value' in result
        assert 0 <= result['p_value'] <= 1

    def test_jensen_shannon_divergence(self, quality_engine):
        """Test Jensen-Shannon divergence calculation"""
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(0.5, 1, 100)

        result = quality_engine._jensen_shannon_divergence(data1, data2)

        assert 'divergence' in result
        assert 'p_value' in result
        assert result['divergence'] >= 0
        assert 0 <= result['p_value'] <= 1

    def test_wasserstein_distance(self, quality_engine):
        """Test Wasserstein distance calculation"""
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(0.5, 1, 100)

        result = quality_engine._wasserstein_distance(data1, data2)

        assert 'distance' in result
        assert 'p_value' in result
        assert result['distance'] >= 0
        assert 0 <= result['p_value'] <= 1

    def test_empty_data_handling(self, quality_engine):
        """Test handling of empty data"""
        original_data = pd.DataFrame()
        synthetic_data = pd.DataFrame()

        report = quality_engine.assess_synthetic_quality(original_data, synthetic_data)

        # Should handle gracefully
        assert 'error' in report or report['overall_quality_score'] == 0.0

    def test_invalid_data_format(self, quality_engine):
        """Test handling of invalid data formats"""
        # Test with non-DataFrame inputs that can't be converted
        with pytest.raises(ValidationError):
            quality_engine.assess_synthetic_quality(None, None)

    def test_save_quality_report(self, quality_engine, sample_data, tmp_path):
        """Test saving quality report"""
        original_data, synthetic_data = sample_data

        report = quality_engine.assess_synthetic_quality(original_data, synthetic_data)

        filepath = tmp_path / "quality_report.json"
        quality_engine.save_quality_report(str(filepath), report)

        # Check file was created
        assert filepath.exists()

        # Check file content
        import json
        with open(filepath, 'r') as f:
            saved_report = json.load(f)

        assert 'overall_quality_score' in saved_report
        assert 'quality_grade' in saved_report

    @patch('src.services.quality_assessment_engine.datetime')
    def test_timestamp_in_report(self, mock_datetime, quality_engine, sample_data):
        """Test timestamp inclusion in reports"""
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

        original_data, synthetic_data = sample_data
        report = quality_engine.assess_synthetic_quality(original_data, synthetic_data)

        assert 'timestamp' in report

    def test_privacy_preservation_assessment(self, quality_engine, sample_data):
        """Test privacy preservation assessment"""
        original_data, synthetic_data = sample_data

        privacy = quality_engine._assess_privacy_preservation(original_data, synthetic_data)

        assert 'privacy_metrics' in privacy
        assert 'overall_privacy' in privacy
        assert 0 <= privacy['overall_privacy'] <= 1

    def test_model_performance_impact_assessment(self, quality_engine, sample_data):
        """Test model performance impact assessment"""
        original_data, synthetic_data = sample_data

        # Create simple numerical data for testing
        X_orig = original_data.select_dtypes(include=[np.number]).values
        X_synth = synthetic_data.select_dtypes(include=[np.number]).values

        impact = quality_engine._assess_model_performance_impact(
            pd.DataFrame(X_orig), pd.DataFrame(X_synth)
        )

        assert 'model_performance' in impact
        assert 'overall_impact' in impact
        assert 0 <= impact['overall_impact'] <= 1