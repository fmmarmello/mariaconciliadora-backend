"""
Validation Reporting Service for Maria Conciliadora system.

This module provides comprehensive validation reporting with:
- Validation metrics tracking and aggregation
- Performance monitoring and analytics
- Rule effectiveness analysis
- Trend analysis and forecasting
- Compliance reporting
- Dashboard data generation
"""

import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from collections import defaultdict, Counter
import json
import statistics

from src.utils.logging_config import get_logger
from src.utils.validation_result import ValidationResult, ValidationSeverity

logger = get_logger(__name__)


class ValidationMetrics:
    """Container for validation metrics data."""

    def __init__(self):
        self.total_validations = 0
        self.total_errors = 0
        self.total_warnings = 0
        self.validation_duration_ms = 0
        self.engine_usage = defaultdict(int)
        self.rule_performance = defaultdict(lambda: {
            'executions': 0,
            'errors': 0,
            'warnings': 0,
            'avg_duration_ms': 0
        })
        self.severity_distribution = defaultdict(int)
        self.temporal_trends = defaultdict(lambda: defaultdict(int))
        self.field_error_rates = defaultdict(lambda: {
            'total_validations': 0,
            'errors': 0,
            'warnings': 0
        })

    def update_from_result(self, result: ValidationResult, engine_name: str = None):
        """Update metrics from a validation result."""
        self.total_validations += 1
        self.total_errors += len(result.errors)
        self.total_warnings += len(result.warnings)
        self.validation_duration_ms += result.validation_duration_ms

        if engine_name:
            self.engine_usage[engine_name] += 1

        # Update severity distribution
        for error in result.errors:
            self.severity_distribution['error'] += 1
        for warning in result.warnings:
            self.severity_distribution['warning'] += 1

        # Update field-level metrics
        for field_result in result.field_results.values():
            field_name = field_result.field_name
            self.field_error_rates[field_name]['total_validations'] += 1
            self.field_error_rates[field_name]['errors'] += len(field_result.errors)
            self.field_error_rates[field_name]['warnings'] += len(field_result.warnings)

    def update_rule_performance(self, rule_id: str, duration_ms: float,
                               errors: int = 0, warnings: int = 0):
        """Update rule performance metrics."""
        rule_metrics = self.rule_performance[rule_id]
        rule_metrics['executions'] += 1
        rule_metrics['errors'] += errors
        rule_metrics['warnings'] += warnings

        # Update average duration
        if rule_metrics['executions'] == 1:
            rule_metrics['avg_duration_ms'] = duration_ms
        else:
            rule_metrics['avg_duration_ms'] = (
                (rule_metrics['avg_duration_ms'] * (rule_metrics['executions'] - 1)) + duration_ms
            ) / rule_metrics['executions']

    def add_temporal_data(self, date_key: str, metric_type: str, value: int):
        """Add temporal trend data."""
        self.temporal_trends[date_key][metric_type] += value

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_validations': self.total_validations,
            'total_errors': self.total_errors,
            'total_warnings': self.total_warnings,
            'validation_duration_ms': self.validation_duration_ms,
            'engine_usage': dict(self.engine_usage),
            'rule_performance': dict(self.rule_performance),
            'severity_distribution': dict(self.severity_distribution),
            'temporal_trends': {date: dict(metrics) for date, metrics in self.temporal_trends.items()},
            'field_error_rates': dict(self.field_error_rates)
        }


class ValidationReport:
    """Container for validation reports."""

    def __init__(self, report_type: str, start_date: datetime, end_date: datetime):
        self.report_type = report_type
        self.start_date = start_date
        self.end_date = end_date
        self.generated_at = datetime.now()
        self.metrics = ValidationMetrics()
        self.summary = {}
        self.recommendations = []
        self.alerts = []

    def generate_summary(self):
        """Generate report summary."""
        total_issues = self.metrics.total_errors + self.metrics.total_warnings

        self.summary = {
            'total_validations': self.metrics.total_validations,
            'total_issues': total_issues,
            'error_rate': (self.metrics.total_errors / self.metrics.total_validations) if self.metrics.total_validations > 0 else 0,
            'warning_rate': (self.metrics.total_warnings / self.metrics.total_validations) if self.metrics.total_validations > 0 else 0,
            'avg_validation_duration_ms': (self.metrics.validation_duration_ms / self.metrics.total_validations) if self.metrics.total_validations > 0 else 0,
            'most_used_engine': max(self.metrics.engine_usage.items(), key=lambda x: x[1]) if self.metrics.engine_usage else None,
            'period_days': (self.end_date - self.start_date).days
        }

    def generate_recommendations(self):
        """Generate recommendations based on metrics."""
        self.recommendations = []

        # High error rate recommendations
        if self.summary.get('error_rate', 0) > 0.1:  # > 10% error rate
            self.recommendations.append({
                'priority': 'high',
                'category': 'error_rate',
                'message': 'High validation error rate detected. Review data quality and validation rules.',
                'suggested_actions': [
                    'Review most common error types',
                    'Update validation rules if needed',
                    'Improve data preprocessing'
                ]
            })

        # Performance recommendations
        avg_duration = self.summary.get('avg_validation_duration_ms', 0)
        if avg_duration > 1000:  # > 1 second
            self.recommendations.append({
                'priority': 'medium',
                'category': 'performance',
                'message': 'Validation performance is slow. Consider optimization.',
                'suggested_actions': [
                    'Review rule complexity',
                    'Consider parallel processing',
                    'Optimize database queries'
                ]
            })

        # Engine usage recommendations
        if len(self.metrics.engine_usage) > 3:
            self.recommendations.append({
                'priority': 'low',
                'category': 'complexity',
                'message': 'Multiple validation engines in use. Consider consolidation.',
                'suggested_actions': [
                    'Review if all engines are necessary',
                    'Consider unified validation approach'
                ]
            })

    def generate_alerts(self):
        """Generate alerts based on metrics."""
        self.alerts = []

        # Critical error rate alert
        if self.summary.get('error_rate', 0) > 0.5:  # > 50% error rate
            self.alerts.append({
                'level': 'critical',
                'message': 'Critical: Extremely high validation error rate',
                'details': f"Error rate: {self.summary['error_rate']:.1%}"
            })

        # Performance degradation alert
        if self.summary.get('avg_validation_duration_ms', 0) > 5000:  # > 5 seconds
            self.alerts.append({
                'level': 'warning',
                'message': 'Warning: Validation performance severely degraded',
                'details': f"Average duration: {self.summary['avg_validation_duration_ms']:.0f}ms"
            })

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'report_type': self.report_type,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'generated_at': self.generated_at.isoformat(),
            'summary': self.summary,
            'recommendations': self.recommendations,
            'alerts': self.alerts,
            'metrics': self.metrics.to_dict()
        }


class ValidationReportingService:
    """
    Service for generating validation reports and analytics.
    """

    def __init__(self):
        self.metrics_history = []
        self.current_metrics = ValidationMetrics()
        self.report_cache = {}
        self.performance_thresholds = {
            'max_error_rate': 0.1,
            'max_warning_rate': 0.3,
            'max_avg_duration_ms': 1000,
            'min_validation_success_rate': 0.8
        }

    def record_validation_result(self, result: ValidationResult,
                               engine_name: str = None, context: Dict[str, Any] = None):
        """Record a validation result for metrics tracking."""
        try:
            # Update current metrics
            self.current_metrics.update_from_result(result, engine_name)

            # Add temporal data
            today = date.today().isoformat()
            self.current_metrics.add_temporal_data(today, 'validations', 1)
            self.current_metrics.add_temporal_data(today, 'errors', len(result.errors))
            self.current_metrics.add_temporal_data(today, 'warnings', len(result.warnings))

            # Store in history (keep last 1000 records)
            if len(self.metrics_history) >= 1000:
                self.metrics_history.pop(0)

            self.metrics_history.append({
                'timestamp': datetime.now(),
                'result': result.to_dict(),
                'engine': engine_name,
                'context': context or {}
            })

            logger.debug(f"Recorded validation result: {len(result.errors)} errors, {len(result.warnings)} warnings")

        except Exception as e:
            logger.error(f"Error recording validation result: {str(e)}")

    def generate_report(self, report_type: str = 'daily',
                       start_date: datetime = None, end_date: datetime = None) -> ValidationReport:
        """
        Generate a validation report.

        Args:
            report_type: Type of report ('daily', 'weekly', 'monthly', 'custom')
            start_date: Start date for custom reports
            end_date: End date for custom reports

        Returns:
            ValidationReport object
        """
        try:
            # Determine date range
            if report_type == 'daily':
                start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = start_date + timedelta(days=1)
            elif report_type == 'weekly':
                today = date.today()
                start_date = datetime.combine(today - timedelta(days=today.weekday()), datetime.min.time())
                end_date = start_date + timedelta(days=7)
            elif report_type == 'monthly':
                today = date.today()
                start_date = datetime.combine(today.replace(day=1), datetime.min.time())
                end_date = datetime.combine((today.replace(day=28) + timedelta(days=4)).replace(day=1), datetime.min.time())
            else:  # custom
                if not start_date or not end_date:
                    start_date = datetime.now() - timedelta(days=1)
                    end_date = datetime.now()

            # Create report
            report = ValidationReport(report_type, start_date, end_date)

            # Aggregate metrics from history within date range
            for record in self.metrics_history:
                if start_date <= record['timestamp'] <= end_date:
                    if 'result' in record:
                        # Reconstruct ValidationResult from stored data
                        result_data = record['result']
                        temp_result = ValidationResult()
                        temp_result.errors = result_data.get('errors', [])
                        temp_result.warnings = result_data.get('warnings', [])
                        temp_result.validation_duration_ms = result_data.get('validation_duration_ms', 0)

                        report.metrics.update_from_result(temp_result, record.get('engine'))

            # Generate report components
            report.generate_summary()
            report.generate_recommendations()
            report.generate_alerts()

            # Cache report
            cache_key = f"{report_type}_{start_date.date()}_{end_date.date()}"
            self.report_cache[cache_key] = report

            logger.info(f"Generated {report_type} validation report: {len(report.metrics.total_validations)} validations")

            return report

        except Exception as e:
            logger.error(f"Error generating validation report: {str(e)}")
            # Return empty report on error
            return ValidationReport('error', datetime.now(), datetime.now())

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Generate dashboard data for validation monitoring."""
        try:
            # Get recent metrics (last 30 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            recent_report = self.generate_report('custom', start_date, end_date)

            # Calculate key metrics
            dashboard_data = {
                'overview': {
                    'total_validations': recent_report.summary.get('total_validations', 0),
                    'error_rate': recent_report.summary.get('error_rate', 0),
                    'warning_rate': recent_report.summary.get('warning_rate', 0),
                    'avg_duration_ms': recent_report.summary.get('avg_validation_duration_ms', 0),
                    'success_rate': 1 - recent_report.summary.get('error_rate', 0)
                },
                'trends': {
                    'daily_error_rates': self._calculate_daily_trends('error_rate', 7),
                    'daily_validation_counts': self._calculate_daily_trends('validations', 7),
                    'engine_usage': dict(recent_report.metrics.engine_usage)
                },
                'top_issues': {
                    'most_common_errors': self._get_most_common_issues('errors', 5),
                    'most_common_warnings': self._get_most_common_issues('warnings', 5),
                    'problematic_fields': self._get_problematic_fields(5)
                },
                'performance': {
                    'slowest_rules': self._get_slowest_rules(5),
                    'engine_performance': self._calculate_engine_performance()
                },
                'alerts': recent_report.alerts,
                'recommendations': recent_report.recommendations
            }

            return dashboard_data

        except Exception as e:
            logger.error(f"Error generating dashboard data: {str(e)}")
            return {'error': str(e)}

    def _calculate_daily_trends(self, metric_type: str, days: int) -> List[Dict[str, Any]]:
        """Calculate daily trends for a metric."""
        trends = []
        today = date.today()

        for i in range(days):
            target_date = today - timedelta(days=i)
            date_key = target_date.isoformat()

            if date_key in self.current_metrics.temporal_trends:
                day_metrics = self.current_metrics.temporal_trends[date_key]
                total_validations = day_metrics.get('validations', 0)

                if metric_type == 'error_rate':
                    errors = day_metrics.get('errors', 0)
                    rate = (errors / total_validations) if total_validations > 0 else 0
                    value = rate
                elif metric_type == 'validations':
                    value = total_validations
                else:
                    value = day_metrics.get(metric_type, 0)

                trends.append({
                    'date': date_key,
                    'value': value
                })

        return trends

    def _get_most_common_issues(self, issue_type: str, limit: int) -> List[Dict[str, Any]]:
        """Get most common validation issues."""
        issue_counts = Counter()

        for record in self.metrics_history[-1000:]:  # Last 1000 records
            if 'result' in record:
                result_data = record['result']
                issues = result_data.get(issue_type, [])

                for issue in issues:
                    # Extract key part of error message
                    if ': ' in issue:
                        key = issue.split(': ', 1)[1]
                    else:
                        key = issue[:50]  # First 50 chars
                    issue_counts[key] += 1

        return [
            {'issue': issue, 'count': count}
            for issue, count in issue_counts.most_common(limit)
        ]

    def _get_problematic_fields(self, limit: int) -> List[Dict[str, Any]]:
        """Get fields with highest error rates."""
        field_stats = []

        for field_name, stats in self.current_metrics.field_error_rates.items():
            total = stats['total_validations']
            if total > 0:
                error_rate = stats['errors'] / total
                warning_rate = stats['warnings'] / total

                field_stats.append({
                    'field': field_name,
                    'error_rate': error_rate,
                    'warning_rate': warning_rate,
                    'total_validations': total
                })

        # Sort by error rate descending
        field_stats.sort(key=lambda x: x['error_rate'], reverse=True)

        return field_stats[:limit]

    def _get_slowest_rules(self, limit: int) -> List[Dict[str, Any]]:
        """Get slowest performing rules."""
        rule_stats = []

        for rule_id, metrics in self.current_metrics.rule_performance.items():
            if metrics['executions'] > 0:
                rule_stats.append({
                    'rule_id': rule_id,
                    'avg_duration_ms': metrics['avg_duration_ms'],
                    'executions': metrics['executions'],
                    'error_rate': (metrics['errors'] / metrics['executions']) if metrics['executions'] > 0 else 0
                })

        # Sort by average duration descending
        rule_stats.sort(key=lambda x: x['avg_duration_ms'], reverse=True)

        return rule_stats[:limit]

    def _calculate_engine_performance(self) -> Dict[str, Any]:
        """Calculate performance metrics for each validation engine."""
        engine_performance = {}

        for engine_name, usage_count in self.current_metrics.engine_usage.items():
            # Calculate average performance for this engine
            engine_records = [r for r in self.metrics_history if r.get('engine') == engine_name]

            if engine_records:
                durations = [r['result'].get('validation_duration_ms', 0) for r in engine_records if 'result' in r]
                avg_duration = statistics.mean(durations) if durations else 0

                total_errors = sum(len(r['result'].get('errors', [])) for r in engine_records if 'result' in r)
                total_warnings = sum(len(r['result'].get('warnings', [])) for r in engine_records if 'result' in r)

                engine_performance[engine_name] = {
                    'usage_count': usage_count,
                    'avg_duration_ms': avg_duration,
                    'total_errors': total_errors,
                    'total_warnings': total_warnings,
                    'error_rate': (total_errors / len(engine_records)) if engine_records else 0
                }

        return engine_performance

    def export_report(self, report: ValidationReport, format_type: str = 'json') -> str:
        """Export validation report in specified format."""
        try:
            if format_type == 'json':
                return json.dumps(report.to_dict(), indent=2, default=str)
            elif format_type == 'csv':
                # Simple CSV export of key metrics
                lines = [
                    "Metric,Value",
                    f"Total Validations,{report.summary.get('total_validations', 0)}",
                    f"Total Errors,{report.metrics.total_errors}",
                    f"Total Warnings,{report.metrics.total_warnings}",
                    f"Error Rate,{report.summary.get('error_rate', 0):.3f}",
                    f"Average Duration (ms),{report.summary.get('avg_validation_duration_ms', 0):.2f}"
                ]
                return '\n'.join(lines)
            else:
                return json.dumps(report.to_dict(), default=str)

        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            return f"Error exporting report: {str(e)}"

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of validation system."""
        try:
            # Generate recent report for health assessment
            report = self.generate_report('daily')

            health_status = {
                'status': 'healthy',
                'score': 100,
                'issues': [],
                'last_check': datetime.now().isoformat()
            }

            # Assess error rate
            error_rate = report.summary.get('error_rate', 0)
            if error_rate > self.performance_thresholds['max_error_rate']:
                health_status['issues'].append({
                    'type': 'error_rate',
                    'severity': 'high' if error_rate > 0.2 else 'medium',
                    'message': f"High error rate: {error_rate:.1%}"
                })
                health_status['score'] -= 20

            # Assess performance
            avg_duration = report.summary.get('avg_validation_duration_ms', 0)
            if avg_duration > self.performance_thresholds['max_avg_duration_ms']:
                health_status['issues'].append({
                    'type': 'performance',
                    'severity': 'medium',
                    'message': f"Slow performance: {avg_duration:.0f}ms average"
                })
                health_status['score'] -= 15

            # Assess warning rate
            warning_rate = report.summary.get('warning_rate', 0)
            if warning_rate > self.performance_thresholds['max_warning_rate']:
                health_status['issues'].append({
                    'type': 'warning_rate',
                    'severity': 'low',
                    'message': f"High warning rate: {warning_rate:.1%}"
                })
                health_status['score'] -= 10

            # Determine overall status
            if health_status['score'] < 70:
                health_status['status'] = 'critical'
            elif health_status['score'] < 85:
                health_status['status'] = 'warning'
            else:
                health_status['status'] = 'healthy'

            return health_status

        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {
                'status': 'error',
                'score': 0,
                'issues': [{'type': 'system_error', 'message': str(e)}],
                'last_check': datetime.now().isoformat()
            }


# Global validation reporting service instance
validation_reporting_service = ValidationReportingService()