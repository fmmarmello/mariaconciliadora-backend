"""
Unit tests for ValidationReportingService.

This module contains comprehensive tests for the ValidationReportingService
including metrics tracking, report generation, dashboard data, and analytics.
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock

from src.services.validation_reporting_service import (
    ValidationReportingService,
    ValidationMetrics,
    ValidationReport
)
from src.utils.validation_result import ValidationResult, ValidationSeverity


class TestValidationReportingService:
    """Test cases for ValidationReportingService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = ValidationReportingService()

    def test_initialization(self):
        """Test service initialization."""
        assert isinstance(self.service.metrics_history, list)
        assert isinstance(self.service.current_metrics, ValidationMetrics)
        assert isinstance(self.service.report_cache, dict)
        assert isinstance(self.service.performance_thresholds, dict)

    def test_record_validation_result(self):
        """Test recording validation results."""
        # Create a mock validation result
        result = ValidationResult()
        result.errors = ["Test error 1", "Test error 2"]
        result.warnings = ["Test warning"]
        result.validation_duration_ms = 150.0

        # Record the result
        self.service.record_validation_result(result, "test_engine")

        # Check that metrics were updated
        assert self.service.current_metrics.total_validations == 1
        assert self.service.current_metrics.total_errors == 2
        assert self.service.current_metrics.total_warnings == 1
        assert self.service.current_metrics.validation_duration_ms == 150.0
        assert self.service.current_metrics.engine_usage["test_engine"] == 1

        # Check that result was added to history
        assert len(self.service.metrics_history) == 1
        assert self.service.metrics_history[0]['engine'] == "test_engine"

    def test_generate_report_daily(self):
        """Test generating daily reports."""
        # Add some test data
        result1 = ValidationResult()
        result1.errors = ["Error 1"]
        result1.warnings = ["Warning 1"]
        result1.validation_duration_ms = 100.0

        result2 = ValidationResult()
        result2.errors = []
        result2.warnings = ["Warning 2"]
        result2.validation_duration_ms = 50.0

        self.service.record_validation_result(result1, "engine1")
        self.service.record_validation_result(result2, "engine2")

        # Generate daily report
        report = self.service.generate_report('daily')

        assert isinstance(report, ValidationReport)
        assert report.report_type == 'daily'
        assert report.summary['total_validations'] == 2
        assert report.summary['total_issues'] == 3  # 1 error + 2 warnings
        assert report.summary['error_rate'] == 0.5  # 1 error out of 2 validations

    def test_generate_report_custom_dates(self):
        """Test generating reports with custom date ranges."""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        # Generate custom report
        report = self.service.generate_report('custom', start_date, end_date)

        assert isinstance(report, ValidationReport)
        assert report.report_type == 'custom'
        assert report.start_date.date() == start_date.date()
        assert report.end_date.date() == end_date.date()

    def test_generate_report_with_no_data(self):
        """Test generating reports when no data is available."""
        report = self.service.generate_report('daily')

        assert isinstance(report, ValidationReport)
        assert report.summary['total_validations'] == 0
        assert report.summary['error_rate'] == 0
        assert report.summary['warning_rate'] == 0

    def test_get_dashboard_data(self):
        """Test getting dashboard data."""
        # Add some test data
        result = ValidationResult()
        result.errors = ["Test error"]
        result.warnings = ["Test warning"]
        result.validation_duration_ms = 200.0

        self.service.record_validation_result(result, "test_engine")

        dashboard_data = self.service.get_dashboard_data()

        assert isinstance(dashboard_data, dict)
        assert 'overview' in dashboard_data
        assert 'trends' in dashboard_data
        assert 'top_issues' in dashboard_data
        assert 'performance' in dashboard_data

        # Check overview data
        assert dashboard_data['overview']['total_validations'] == 1
        assert dashboard_data['overview']['error_rate'] == 1.0
        assert dashboard_data['overview']['warning_rate'] == 1.0

    def test_get_dashboard_data_with_errors(self):
        """Test dashboard data generation with error conditions."""
        # Mock the generate_report method to raise an exception
        with patch.object(self.service, 'generate_report', side_effect=Exception("Test error")):
            dashboard_data = self.service.get_dashboard_data()

            assert 'error' in dashboard_data
            assert dashboard_data['error'] == "Test error"

    def test_get_health_status_healthy(self):
        """Test health status when system is healthy."""
        # Add successful validations
        result = ValidationResult()
        result.errors = []
        result.warnings = []
        result.validation_duration_ms = 50.0

        for _ in range(10):
            self.service.record_validation_result(result, "test_engine")

        health_status = self.service.get_health_status()

        assert health_status['status'] == 'healthy'
        assert health_status['score'] >= 85
        assert 'issues' in health_status

    def test_get_health_status_warning(self):
        """Test health status when system has warnings."""
        # Add validations with high warning rate
        result = ValidationResult()
        result.errors = []
        result.warnings = ["Warning"] * 5  # High warning rate
        result.validation_duration_ms = 50.0

        for _ in range(10):
            self.service.record_validation_result(result, "test_engine")

        health_status = self.service.get_health_status()

        assert health_status['status'] in ['warning', 'critical']
        assert health_status['score'] < 100
        assert len(health_status['issues']) > 0

    def test_get_health_status_critical(self):
        """Test health status when system is critical."""
        # Add validations with high error rate and slow performance
        result = ValidationResult()
        result.errors = ["Critical error"] * 10  # Very high error rate
        result.warnings = []
        result.validation_duration_ms = 10000.0  # Very slow

        for _ in range(5):
            self.service.record_validation_result(result, "test_engine")

        health_status = self.service.get_health_status()

        assert health_status['status'] == 'critical'
        assert health_status['score'] < 70
        assert len(health_status['issues']) > 0

    def test_export_report_json(self):
        """Test exporting reports in JSON format."""
        # Generate a test report
        report = self.service.generate_report('daily')

        # Export as JSON
        exported_data = self.service.export_report(report, 'json')

        # Should be valid JSON
        import json
        parsed_data = json.loads(exported_data)
        assert isinstance(parsed_data, dict)
        assert 'report_type' in parsed_data

    def test_export_report_csv(self):
        """Test exporting reports in CSV format."""
        # Generate a test report
        report = self.service.generate_report('daily')

        # Export as CSV
        exported_data = self.service.export_report(report, 'csv')

        # Should contain CSV headers
        assert 'Metric,Value' in exported_data
        assert 'Total Validations' in exported_data

    def test_export_report_invalid_format(self):
        """Test exporting reports with invalid format."""
        report = self.service.generate_report('daily')

        # Export with invalid format should return JSON
        exported_data = self.service.export_report(report, 'invalid')

        import json
        parsed_data = json.loads(exported_data)
        assert isinstance(parsed_data, dict)

    def test_calculate_daily_trends(self):
        """Test calculating daily trends."""
        # Add data for multiple days
        today = date.today()
        yesterday = today - timedelta(days=1)

        # Add data for today
        result = ValidationResult()
        result.errors = ["Error"]
        result.validation_duration_ms = 100.0

        # Mock the temporal trends
        self.service.current_metrics.temporal_trends[today.isoformat()] = {
            'validations': 5,
            'errors': 1,
            'warnings': 2
        }

        trends = self.service._calculate_daily_trends('error_rate', 7)

        assert len(trends) == 7
        assert isinstance(trends[0], dict)
        assert 'date' in trends[0]
        assert 'value' in trends[0]

    def test_get_most_common_issues(self):
        """Test getting most common validation issues."""
        # Add validation results with specific errors
        result1 = ValidationResult()
        result1.errors = ["Database connection failed", "Invalid input format"]
        result1.warnings = ["Slow performance detected"]

        result2 = ValidationResult()
        result2.errors = ["Database connection failed", "Timeout error"]
        result2.warnings = ["Slow performance detected"]

        self.service.record_validation_result(result1)
        self.service.record_validation_result(result2)

        common_errors = self.service._get_most_common_issues('errors', 5)
        common_warnings = self.service._get_most_common_issues('warnings', 5)

        assert len(common_errors) > 0
        assert len(common_warnings) > 0

        # Check that "Database connection failed" appears in errors
        error_messages = [item['issue'] for item in common_errors]
        assert any("Database connection failed" in msg for msg in error_messages)

    def test_get_problematic_fields(self):
        """Test getting problematic fields."""
        # Create validation results with field-specific errors
        result = ValidationResult()
        result.errors = ["Amount validation failed"]
        result.warnings = ["Date format warning"]

        # Mock field results
        from src.utils.validation_result import FieldValidationResult
        field_result = FieldValidationResult("amount", ValidationSeverity.HIGH)
        field_result.errors = ["Amount validation failed"]
        result.field_results["amount"] = field_result

        self.service.record_validation_result(result)

        problematic_fields = self.service._get_problematic_fields(5)

        assert len(problematic_fields) > 0
        assert problematic_fields[0]['field'] == 'amount'
        assert problematic_fields[0]['error_rate'] > 0

    def test_get_slowest_rules(self):
        """Test getting slowest performing rules."""
        # This would require rule performance data
        slowest_rules = self.service._get_slowest_rules(5)

        # Should return empty list if no rule performance data
        assert isinstance(slowest_rules, list)

    def test_calculate_engine_performance(self):
        """Test calculating engine performance metrics."""
        # Add validation results from different engines
        result1 = ValidationResult()
        result1.errors = ["Error"]
        result1.warnings = ["Warning"]
        result1.validation_duration_ms = 100.0

        result2 = ValidationResult()
        result2.errors = []
        result2.warnings = []
        result2.validation_duration_ms = 50.0

        self.service.record_validation_result(result1, "engine1")
        self.service.record_validation_result(result2, "engine1")
        self.service.record_validation_result(result2, "engine2")

        engine_performance = self.service._calculate_engine_performance()

        assert 'engine1' in engine_performance
        assert 'engine2' in engine_performance

        assert engine_performance['engine1']['usage_count'] == 2
        assert engine_performance['engine2']['usage_count'] == 1
        assert engine_performance['engine1']['total_errors'] == 1

    def test_metrics_history_limit(self):
        """Test that metrics history is limited."""
        # Add more than 1000 records
        result = ValidationResult()
        result.errors = ["Test error"]

        for i in range(1010):
            self.service.record_validation_result(result)

        # Should only keep last 1000 records
        assert len(self.service.metrics_history) == 1000

    def test_empty_metrics(self):
        """Test behavior with empty metrics."""
        # Test with no data
        dashboard_data = self.service.get_dashboard_data()

        assert isinstance(dashboard_data, dict)
        assert 'overview' in dashboard_data
        assert dashboard_data['overview']['total_validations'] == 0

    def test_report_caching(self):
        """Test report caching functionality."""
        # Generate a report
        report1 = self.service.generate_report('daily')

        # Generate the same report again (should use cache)
        report2 = self.service.generate_report('daily')

        # Should be the same object (from cache)
        assert report1 is report2

        # Check cache key exists
        cache_key = f"daily_{date.today()}_{date.today() + timedelta(days=1)}"
        assert cache_key in self.service.report_cache


class TestValidationMetrics:
    """Test cases for ValidationMetrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = ValidationMetrics()

    def test_initialization(self):
        """Test metrics initialization."""
        assert self.metrics.total_validations == 0
        assert self.metrics.total_errors == 0
        assert self.metrics.total_warnings == 0
        assert isinstance(self.metrics.engine_usage, dict)
        assert isinstance(self.metrics.rule_performance, dict)

    def test_update_from_result(self):
        """Test updating metrics from validation result."""
        result = ValidationResult()
        result.errors = ["Error 1", "Error 2"]
        result.warnings = ["Warning 1"]
        result.validation_duration_ms = 150.0

        self.metrics.update_from_result(result, "test_engine")

        assert self.metrics.total_validations == 1
        assert self.metrics.total_errors == 2
        assert self.metrics.total_warnings == 1
        assert self.metrics.validation_duration_ms == 150.0
        assert self.metrics.engine_usage["test_engine"] == 1

    def test_update_rule_performance(self):
        """Test updating rule performance metrics."""
        self.metrics.update_rule_performance("test_rule", 100.0, 1, 2)

        assert "test_rule" in self.metrics.rule_performance
        rule_metrics = self.metrics.rule_performance["test_rule"]
        assert rule_metrics['executions'] == 1
        assert rule_metrics['errors'] == 1
        assert rule_metrics['warnings'] == 2
        assert rule_metrics['avg_duration_ms'] == 100.0

    def test_add_temporal_data(self):
        """Test adding temporal trend data."""
        self.metrics.add_temporal_data("2023-01-15", "validations", 5)
        self.metrics.add_temporal_data("2023-01-15", "errors", 2)

        assert "2023-01-15" in self.metrics.temporal_trends
        assert self.metrics.temporal_trends["2023-01-15"]["validations"] == 5
        assert self.metrics.temporal_trends["2023-01-15"]["errors"] == 2

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        # Add some test data
        result = ValidationResult()
        result.errors = ["Test error"]
        self.metrics.update_from_result(result, "test_engine")

        metrics_dict = self.metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict['total_validations'] == 1
        assert metrics_dict['total_errors'] == 1
        assert 'engine_usage' in metrics_dict
        assert 'rule_performance' in metrics_dict


class TestValidationReport:
    """Test cases for ValidationReport."""

    def setup_method(self):
        """Set up test fixtures."""
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        self.report = ValidationReport('test', start_date, end_date)

    def test_initialization(self):
        """Test report initialization."""
        assert self.report.report_type == 'test'
        assert isinstance(self.report.metrics, ValidationMetrics)
        assert isinstance(self.report.summary, dict)
        assert isinstance(self.report.recommendations, list)
        assert isinstance(self.report.alerts, list)

    def test_generate_summary(self):
        """Test summary generation."""
        # Add some test data
        result = ValidationResult()
        result.errors = ["Error"]
        result.warnings = ["Warning"]
        result.validation_duration_ms = 100.0

        self.report.metrics.update_from_result(result)

        self.report.generate_summary()

        assert self.report.summary['total_validations'] == 1
        assert self.report.summary['total_issues'] == 2
        assert self.report.summary['error_rate'] == 1.0
        assert self.report.summary['warning_rate'] == 1.0

    def test_generate_recommendations(self):
        """Test recommendations generation."""
        # Set up high error rate scenario
        self.report.summary = {
            'error_rate': 0.15,  # Above 10% threshold
            'avg_validation_duration_ms': 1500.0  # Above 1000ms threshold
        }

        self.report.generate_recommendations()

        assert len(self.report.recommendations) > 0

        # Check for specific recommendations
        recommendation_types = [rec['category'] for rec in self.report.recommendations]
        assert 'error_rate' in recommendation_types
        assert 'performance' in recommendation_types

    def test_generate_alerts(self):
        """Test alerts generation."""
        # Set up critical scenario
        self.report.summary = {
            'error_rate': 0.8,  # Above 50% threshold
            'avg_validation_duration_ms': 6000.0  # Above 5000ms threshold
        }

        self.report.generate_alerts()

        assert len(self.report.alerts) > 0

        # Check for critical alert
        alert_levels = [alert['level'] for alert in self.report.alerts]
        assert 'critical' in alert_levels

    def test_to_dict(self):
        """Test converting report to dictionary."""
        report_dict = self.report.to_dict()

        assert isinstance(report_dict, dict)
        assert report_dict['report_type'] == 'test'
        assert 'start_date' in report_dict
        assert 'end_date' in report_dict
        assert 'summary' in report_dict
        assert 'recommendations' in report_dict
        assert 'alerts' in report_dict


if __name__ == '__main__':
    pytest.main([__file__])