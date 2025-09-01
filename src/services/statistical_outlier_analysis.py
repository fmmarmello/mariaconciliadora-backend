"""
Statistical Outlier Analysis for Comprehensive Outlier Evaluation

This module provides comprehensive statistical analysis of outlier detection results,
including method comparison, confidence scoring, severity ranking, and significance testing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2, ttest_ind, mannwhitneyu
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
import warnings
import logging
from datetime import datetime
import json

from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from .advanced_outlier_detector import AdvancedOutlierDetector
from .contextual_outlier_detector import ContextualOutlierDetector

logger = get_logger(__name__)


class StatisticalOutlierAnalysis:
    """
    Comprehensive statistical analysis for outlier detection results
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the statistical outlier analysis

        Args:
            config: Configuration dictionary for analysis parameters
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize detectors
        self.advanced_detector = AdvancedOutlierDetector(self.config.get('advanced_config', {}))
        self.contextual_detector = ContextualOutlierDetector(self.config.get('contextual_config', {}))

        # Analysis results storage
        self.method_comparison_results = {}
        self.confidence_scores = {}
        self.severity_rankings = {}
        self.significance_tests = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for statistical analysis"""
        return {
            'method_comparison': {
                'metrics': ['precision', 'recall', 'f1_score', 'accuracy'],
                'statistical_tests': ['mcnemar', 'paired_ttest'],
                'confidence_level': 0.95
            },
            'confidence_scoring': {
                'method_weights': {
                    'iqr': 0.8,
                    'zscore': 0.7,
                    'lof': 0.9,
                    'isolation_forest': 0.85,
                    'mahalanobis': 0.9,
                    'ensemble': 0.95
                },
                'agreement_bonus': 0.1,
                'score_normalization': 'minmax'
            },
            'severity_ranking': {
                'factors': ['deviation_magnitude', 'frequency', 'context_importance'],
                'weights': [0.4, 0.3, 0.3],
                'severity_levels': ['low', 'medium', 'high', 'critical']
            },
            'significance_testing': {
                'alpha': 0.05,
                'tests': ['chisquare', 'mannwhitney', 'ks_test'],
                'multiple_testing_correction': 'bonferroni'
            },
            'visual_analysis': {
                'generate_plots': True,
                'plot_format': 'json',
                'max_points': 1000
            },
            'advanced_config': {},
            'contextual_config': {}
        }

    def perform_comprehensive_analysis(self, transactions: List[Dict],
                                    ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive outlier analysis on transaction data

        Args:
            transactions: List of transaction dictionaries
            ground_truth: Optional ground truth outlier labels

        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            self.logger.info(f"Starting comprehensive outlier analysis for {len(transactions)} transactions")

            # Step 1: Run multiple detection methods
            detection_results = self._run_multiple_detection_methods(transactions)

            # Step 2: Compare detection methods
            method_comparison = self._compare_detection_methods(detection_results, ground_truth)

            # Step 3: Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(detection_results)

            # Step 4: Rank outlier severity
            severity_ranking = self._rank_outlier_severity(transactions, detection_results)

            # Step 5: Perform significance testing
            significance_tests = self._perform_significance_testing(detection_results, ground_truth)

            # Step 6: Generate visual analysis data
            visual_data = self._generate_visual_analysis_data(transactions, detection_results)

            # Step 7: Generate comprehensive report
            comprehensive_report = self._generate_comprehensive_report(
                detection_results, method_comparison, confidence_scores,
                severity_ranking, significance_tests, visual_data
            )

            return comprehensive_report

        except Exception as e:
            self.logger.error(f"Error in comprehensive outlier analysis: {str(e)}")
            return {'error': str(e)}

    def _run_multiple_detection_methods(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Run multiple outlier detection methods"""
        try:
            df = pd.DataFrame(transactions)

            if len(df) == 0:
                return {'error': 'No transaction data provided'}

            # Prepare data for detection
            if 'amount' in df.columns:
                amounts = np.abs(df['amount'].values)
            else:
                return {'error': 'No amount data found'}

            # Run advanced detection methods
            advanced_results = self.advanced_detector.detect_outliers_comprehensive(
                amounts.reshape(-1, 1), return_details=True
            )

            # Run contextual detection methods
            contextual_results = {}

            if 'category' in df.columns:
                category_results = self.contextual_detector.detect_amount_outliers_by_category(transactions)
                contextual_results['category_based'] = category_results

            if 'date' in df.columns:
                temporal_results = self.contextual_detector.detect_temporal_outliers(transactions)
                contextual_results['temporal'] = temporal_results

                frequency_results = self.contextual_detector.detect_frequency_outliers(transactions)
                contextual_results['frequency'] = frequency_results

                balance_results = self.contextual_detector.detect_balance_outliers(transactions)
                contextual_results['balance'] = balance_results

            if 'description' in df.columns:
                merchant_results = self.contextual_detector.detect_merchant_outliers(transactions)
                contextual_results['merchant'] = merchant_results

            return {
                'advanced_methods': advanced_results,
                'contextual_methods': contextual_results,
                'data_summary': {
                    'total_transactions': len(transactions),
                    'amount_range': [df['amount'].min(), df['amount'].max()],
                    'categories': df.get('category', pd.Series()).value_counts().to_dict(),
                    'date_range': [df['date'].min(), df['date'].max()] if 'date' in df.columns else None
                }
            }

        except Exception as e:
            self.logger.error(f"Error running multiple detection methods: {str(e)}")
            return {'error': str(e)}

    def _compare_detection_methods(self, detection_results: Dict[str, Any],
                                 ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compare performance of different detection methods"""
        try:
            comparison = {
                'method_performance': {},
                'pairwise_comparison': {},
                'best_methods': {},
                'consensus_analysis': {}
            }

            # Extract method results
            advanced_results = detection_results.get('advanced_methods', {}).get('results', {})

            if not advanced_results:
                return comparison

            # Calculate performance metrics for each method
            for method_name, method_result in advanced_results.items():
                if 'error' in method_result:
                    continue

                outlier_flags = np.array(method_result['outlier_flags'])
                outlier_scores = np.array(method_result['outlier_scores'])

                perf = {
                    'outlier_count': method_result['outlier_count'],
                    'outlier_percentage': method_result['outlier_percentage'],
                    'mean_score': np.mean(outlier_scores),
                    'std_score': np.std(outlier_scores),
                    'score_range': [np.min(outlier_scores), np.max(outlier_scores)]
                }

                # Add classification metrics if ground truth available
                if ground_truth is not None:
                    try:
                        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

                        precision = precision_score(ground_truth, outlier_flags, zero_division=0)
                        recall = recall_score(ground_truth, outlier_flags, zero_division=0)
                        f1 = f1_score(ground_truth, outlier_flags, zero_division=0)
                        accuracy = accuracy_score(ground_truth, outlier_flags)

                        perf.update({
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'accuracy': accuracy
                        })
                    except Exception as e:
                        self.logger.warning(f"Error calculating classification metrics for {method_name}: {str(e)}")

                comparison['method_performance'][method_name] = perf

            # Find consensus outliers (detected by multiple methods)
            if len(advanced_results) > 1:
                outlier_flags_matrix = np.column_stack([
                    result['outlier_flags'] for result in advanced_results.values()
                    if 'outlier_flags' in result
                ])

                consensus_flags = np.sum(outlier_flags_matrix, axis=1) >= 2  # At least 2 methods agree
                consensus_count = np.sum(consensus_flags)

                comparison['consensus_analysis'] = {
                    'consensus_outlier_count': consensus_count,
                    'consensus_percentage': consensus_count / len(consensus_flags) * 100,
                    'agreement_level': np.mean(np.sum(outlier_flags_matrix, axis=1) / len(advanced_results))
                }

            # Determine best methods
            if comparison['method_performance']:
                # Sort by F1 score if available, otherwise by outlier percentage
                if 'f1_score' in list(comparison['method_performance'].values())[0]:
                    best_method = max(comparison['method_performance'].items(),
                                    key=lambda x: x[1]['f1_score'])
                else:
                    best_method = max(comparison['method_performance'].items(),
                                    key=lambda x: x[1]['outlier_percentage'])

                comparison['best_methods']['overall'] = best_method[0]

            self.method_comparison_results = comparison
            return comparison

        except Exception as e:
            self.logger.error(f"Error comparing detection methods: {str(e)}")
            return {'error': str(e)}

    def _calculate_confidence_scores(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence scores for outlier classifications"""
        try:
            confidence_scores = {
                'method_confidence': {},
                'ensemble_confidence': {},
                'overall_confidence': {}
            }

            advanced_results = detection_results.get('advanced_methods', {}).get('results', {})

            if not advanced_results:
                return confidence_scores

            n_samples = len(list(advanced_results.values())[0]['outlier_flags'])
            method_weights = self.config['confidence_scoring']['method_weights']

            # Calculate confidence for each method
            method_confidences = {}
            outlier_flags_matrix = []

            for method_name, method_result in advanced_results.items():
                if 'error' in method_result:
                    continue

                outlier_flags = np.array(method_result['outlier_flags'])
                outlier_scores = np.array(method_result['outlier_scores'])

                # Base confidence from outlier scores
                base_confidence = np.abs(outlier_scores)

                # Method-specific weight
                method_weight = method_weights.get(method_name, 0.5)

                # Calculate weighted confidence
                method_confidence = base_confidence * method_weight

                method_confidences[method_name] = method_confidence
                outlier_flags_matrix.append(outlier_flags)

                confidence_scores['method_confidence'][method_name] = {
                    'mean_confidence': np.mean(method_confidence),
                    'std_confidence': np.std(method_confidence),
                    'confidence_range': [np.min(method_confidence), np.max(method_confidence)]
                }

            # Calculate ensemble confidence
            if method_confidences:
                outlier_flags_matrix = np.column_stack(outlier_flags_matrix)
                confidence_matrix = np.column_stack(list(method_confidences.values()))

                # Agreement bonus
                agreement_bonus = self.config['confidence_scoring']['agreement_bonus']
                agreement_scores = np.sum(outlier_flags_matrix, axis=1) / len(method_confidences)
                agreement_bonus_values = agreement_scores * agreement_bonus

                # Ensemble confidence as weighted average + agreement bonus
                ensemble_confidence = np.mean(confidence_matrix, axis=1) + agreement_bonus_values

                confidence_scores['ensemble_confidence'] = {
                    'mean_confidence': np.mean(ensemble_confidence),
                    'std_confidence': np.std(ensemble_confidence),
                    'confidence_distribution': np.histogram(ensemble_confidence, bins=10)[0].tolist()
                }

                # Overall confidence scores for each sample
                confidence_scores['overall_confidence'] = {
                    'sample_confidences': ensemble_confidence.tolist(),
                    'high_confidence_threshold': np.percentile(ensemble_confidence, 90),
                    'low_confidence_threshold': np.percentile(ensemble_confidence, 10)
                }

            self.confidence_scores = confidence_scores
            return confidence_scores

        except Exception as e:
            self.logger.error(f"Error calculating confidence scores: {str(e)}")
            return {'error': str(e)}

    def _rank_outlier_severity(self, transactions: List[Dict],
                             detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Rank outliers by severity"""
        try:
            severity_ranking = {
                'severity_levels': {},
                'top_outliers': [],
                'severity_distribution': {}
            }

            df = pd.DataFrame(transactions)
            advanced_results = detection_results.get('advanced_methods', {}).get('results', {})

            if not advanced_results or len(df) == 0:
                return severity_ranking

            # Get ensemble outlier flags and confidence scores
            ensemble_flags = None
            ensemble_confidence = None

            if 'ensemble' in advanced_results:
                ensemble_result = advanced_results['ensemble']
                ensemble_flags = np.array(ensemble_result['outlier_flags'])
                ensemble_confidence = np.array(ensemble_result['outlier_scores'])
            else:
                # Use majority voting as fallback
                outlier_flags_matrix = np.column_stack([
                    result['outlier_flags'] for result in advanced_results.values()
                    if 'outlier_flags' in result
                ])
                ensemble_flags = np.sum(outlier_flags_matrix, axis=1) >= (len(advanced_results) // 2 + 1)

            if ensemble_flags is None:
                return severity_ranking

            outlier_indices = np.where(ensemble_flags)[0]

            if len(outlier_indices) == 0:
                return severity_ranking

            # Calculate severity scores
            severity_scores = []

            for idx in outlier_indices:
                transaction = df.iloc[idx]

                # Factor 1: Deviation magnitude
                amount = abs(transaction.get('amount', 0))
                deviation_magnitude = amount / (df['amount'].abs().mean() + 1e-10)

                # Factor 2: Frequency/context
                frequency_factor = 1.0
                if 'category' in transaction:
                    category_count = df[df['category'] == transaction['category']].shape[0]
                    frequency_factor = 1.0 / (category_count / len(df) + 0.1)

                # Factor 3: Context importance
                context_factor = 1.0
                if transaction.get('category') in ['transferencia', 'investimento']:
                    context_factor = 1.5  # Higher importance for financial transfers

                # Combine factors with weights
                weights = self.config['severity_ranking']['weights']
                severity_score = (
                    weights[0] * deviation_magnitude +
                    weights[1] * frequency_factor +
                    weights[2] * context_factor
                )

                # Add confidence bonus if available
                if ensemble_confidence is not None:
                    confidence_bonus = ensemble_confidence[idx] * 0.2
                    severity_score += confidence_bonus

                severity_scores.append({
                    'index': int(idx),
                    'severity_score': float(severity_score),
                    'transaction': transaction.to_dict(),
                    'factors': {
                        'deviation_magnitude': float(deviation_magnitude),
                        'frequency_factor': float(frequency_factor),
                        'context_factor': float(context_factor)
                    }
                })

            # Sort by severity score
            severity_scores.sort(key=lambda x: x['severity_score'], reverse=True)

            # Assign severity levels
            severity_levels = self.config['severity_ranking']['severity_levels']
            n_levels = len(severity_levels)

            for i, outlier in enumerate(severity_scores):
                level_index = min(i * n_levels // len(severity_scores), n_levels - 1)
                outlier['severity_level'] = severity_levels[level_index]

            # Group by severity levels
            for level in severity_levels:
                level_outliers = [o for o in severity_scores if o['severity_level'] == level]
                severity_ranking['severity_levels'][level] = {
                    'count': len(level_outliers),
                    'percentage': len(level_outliers) / len(severity_scores) * 100,
                    'outliers': level_outliers[:10]  # Top 10 per level
                }

            # Overall distribution
            severity_ranking['severity_distribution'] = {
                level: len([o for o in severity_scores if o['severity_level'] == level])
                for level in severity_levels
            }

            # Top outliers
            severity_ranking['top_outliers'] = severity_scores[:20]

            self.severity_rankings = severity_ranking
            return severity_ranking

        except Exception as e:
            self.logger.error(f"Error ranking outlier severity: {str(e)}")
            return {'error': str(e)}

    def _perform_significance_testing(self, detection_results: Dict[str, Any],
                                   ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform statistical significance testing"""
        try:
            significance_tests = {
                'method_differences': {},
                'distribution_tests': {},
                'overall_significance': {}
            }

            advanced_results = detection_results.get('advanced_methods', {}).get('results', {})

            if len(advanced_results) < 2:
                return significance_tests

            alpha = self.config['significance_testing']['alpha']
            tests = self.config['significance_testing']['tests']

            # Test for differences between methods
            method_names = list(advanced_results.keys())
            method_scores = []

            for method_name in method_names:
                result = advanced_results[method_name]
                if 'outlier_scores' in result:
                    method_scores.append(np.array(result['outlier_scores']))

            if len(method_scores) >= 2:
                # Perform pairwise significance tests
                for i in range(len(method_names)):
                    for j in range(i + 1, len(method_names)):
                        method1, method2 = method_names[i], method_names[j]
                        scores1, scores2 = method_scores[i], method_scores[j]

                        test_results = {}

                        # Mann-Whitney U test (non-parametric)
                        if 'mannwhitney' in tests:
                            try:
                                stat, p_value = mannwhitneyu(scores1, scores2, alternative='two-sided')
                                test_results['mann_whitney'] = {
                                    'statistic': float(stat),
                                    'p_value': float(p_value),
                                    'significant': p_value < alpha
                                }
                            except Exception as e:
                                test_results['mann_whitney'] = {'error': str(e)}

                        # Kolmogorov-Smirnov test for distribution differences
                        if 'ks_test' in tests:
                            try:
                                stat, p_value = stats.ks_2samp(scores1, scores2)
                                test_results['ks_test'] = {
                                    'statistic': float(stat),
                                    'p_value': float(p_value),
                                    'significant': p_value < alpha
                                }
                            except Exception as e:
                                test_results['ks_test'] = {'error': str(e)}

                        significance_tests['method_differences'][f'{method1}_vs_{method2}'] = test_results

            # Test against ground truth if available
            if ground_truth is not None:
                significance_tests['ground_truth_tests'] = {}

                for method_name, result in advanced_results.items():
                    if 'outlier_flags' in result:
                        outlier_flags = np.array(result['outlier_flags'])

                        # Chi-square test for independence
                        if 'chisquare' in tests:
                            try:
                                contingency_table = np.array([
                                    [np.sum((outlier_flags == 1) & (ground_truth == 1)),
                                     np.sum((outlier_flags == 1) & (ground_truth == 0))],
                                    [np.sum((outlier_flags == 0) & (ground_truth == 1)),
                                     np.sum((outlier_flags == 0) & (ground_truth == 0))]
                                ])

                                if np.all(contingency_table > 0):  # Ensure no zero cells
                                    stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                                    significance_tests['ground_truth_tests'][method_name] = {
                                        'chi_square': {
                                            'statistic': float(stat),
                                            'p_value': float(p_value),
                                            'dof': int(dof),
                                            'significant': p_value < alpha
                                        }
                                    }
                            except Exception as e:
                                significance_tests['ground_truth_tests'][method_name] = {
                                    'chi_square': {'error': str(e)}
                                }

            self.significance_tests = significance_tests
            return significance_tests

        except Exception as e:
            self.logger.error(f"Error performing significance testing: {str(e)}")
            return {'error': str(e)}

    def _generate_visual_analysis_data(self, transactions: List[Dict],
                                     detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for visual analysis"""
        try:
            visual_data = {
                'scatter_plots': {},
                'histograms': {},
                'time_series': {},
                'heatmaps': {}
            }

            df = pd.DataFrame(transactions)

            if len(df) == 0:
                return visual_data

            # Limit data points for performance
            max_points = self.config['visual_analysis']['max_points']
            if len(df) > max_points:
                df = df.sample(n=max_points, random_state=42)

            # Scatter plot data (amount vs outlier score)
            advanced_results = detection_results.get('advanced_methods', {}).get('results', {})

            if advanced_results:
                amounts = df['amount'].abs().values

                for method_name, result in advanced_results.items():
                    if 'outlier_scores' in result:
                        scores = np.array(result['outlier_scores'])
                        outlier_flags = np.array(result['outlier_flags'])

                        # Create scatter plot data
                        visual_data['scatter_plots'][method_name] = {
                            'x': amounts.tolist(),
                            'y': scores.tolist(),
                            'colors': outlier_flags.astype(int).tolist(),
                            'labels': ['Normal', 'Outlier']
                        }

                # Histogram data
                visual_data['histograms']['amount_distribution'] = {
                    'data': amounts.tolist(),
                    'bins': 30
                }

                # Time series data if dates available
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.dropna(subset=['date'])
                    df = df.sort_values('date')

                    visual_data['time_series']['amount_over_time'] = {
                        'dates': df['date'].dt.strftime('%Y-%m-%d').tolist(),
                        'amounts': df['amount'].abs().tolist()
                    }

            return visual_data

        except Exception as e:
            self.logger.error(f"Error generating visual analysis data: {str(e)}")
            return {'error': str(e)}

    def _generate_comprehensive_report(self, detection_results: Dict[str, Any],
                                     method_comparison: Dict[str, Any],
                                     confidence_scores: Dict[str, Any],
                                     severity_ranking: Dict[str, Any],
                                     significance_tests: Dict[str, Any],
                                     visual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        try:
            report = {
                'summary': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_transactions': detection_results.get('data_summary', {}).get('total_transactions', 0),
                    'methods_used': list(detection_results.get('advanced_methods', {}).get('results', {}).keys()),
                    'contextual_analyses': list(detection_results.get('contextual_methods', {}).keys())
                },
                'detection_results': detection_results,
                'method_comparison': method_comparison,
                'confidence_analysis': confidence_scores,
                'severity_analysis': severity_ranking,
                'statistical_significance': significance_tests,
                'visual_data': visual_data,
                'recommendations': self._generate_recommendations(
                    method_comparison, severity_ranking, significance_tests
                )
            }

            return report

        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {str(e)}")
            return {'error': str(e)}

    def _generate_recommendations(self, method_comparison: Dict[str, Any],
                                severity_ranking: Dict[str, Any],
                                significance_tests: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        try:
            # Method recommendations
            if 'best_methods' in method_comparison:
                best_method = method_comparison['best_methods'].get('overall')
                if best_method:
                    recommendations.append(
                        f"Use {best_method.replace('_', ' ').title()} as the primary outlier detection method based on performance analysis."
                    )

            # Severity-based recommendations
            if 'severity_levels' in severity_ranking:
                critical_count = severity_ranking['severity_levels'].get('critical', {}).get('count', 0)
                if critical_count > 0:
                    recommendations.append(
                        f"Immediate attention required: {critical_count} critical severity outliers detected. Review these transactions urgently."
                    )

            # Consensus recommendations
            if 'consensus_analysis' in method_comparison:
                consensus_pct = method_comparison['consensus_analysis'].get('consensus_percentage', 0)
                if consensus_pct > 20:
                    recommendations.append(
                        f"High consensus among detection methods ({consensus_pct:.1f}%). Results are reliable and consistent."
                    )
                elif consensus_pct < 5:
                    recommendations.append(
                        "Low consensus among detection methods. Consider using ensemble approach or adjusting parameters."
                    )

            # Statistical significance recommendations
            if significance_tests.get('method_differences'):
                significant_differences = [
                    method for method, tests in significance_tests['method_differences'].items()
                    if any(test.get('significant', False) for test in tests.values() if isinstance(test, dict))
                ]
                if significant_differences:
                    recommendations.append(
                        f"Statistically significant differences found between methods: {', '.join(significant_differences)}. "
                        "Consider method selection based on specific use case requirements."
                    )

            # General recommendations
            recommendations.extend([
                "Implement automated monitoring for outlier detection in production pipelines.",
                "Establish threshold tuning procedures based on business requirements and false positive rates.",
                "Consider domain expert validation of high-severity outliers before taking action.",
                "Regularly update detection models with new transaction patterns and seasonal variations."
            ])

        except Exception as e:
            self.logger.warning(f"Error generating recommendations: {str(e)}")
            recommendations = ["Analysis completed. Review results manually for specific recommendations."]

        return recommendations

    def export_analysis_report(self, filepath: str, report: Dict[str, Any]) -> bool:
        """Export comprehensive analysis report to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"Analysis report exported to {filepath}")

            # Also export summary as CSV if applicable
            summary_data = self._extract_summary_for_csv(report)
            if summary_data:
                csv_filepath = filepath.replace('.json', '_summary.csv')
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(csv_filepath, index=False)
                self.logger.info(f"Summary CSV exported to {csv_filepath}")

            return True

        except Exception as e:
            self.logger.error(f"Error exporting analysis report: {str(e)}")
            return False

    def _extract_summary_for_csv(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract summary data for CSV export"""
        try:
            summary_data = []

            # Method performance summary
            method_perf = report.get('method_comparison', {}).get('method_performance', {})
            for method, perf in method_perf.items():
                summary_data.append({
                    'category': 'method_performance',
                    'method': method,
                    'outlier_count': perf.get('outlier_count', 0),
                    'outlier_percentage': perf.get('outlier_percentage', 0),
                    'precision': perf.get('precision', 0),
                    'recall': perf.get('recall', 0),
                    'f1_score': perf.get('f1_score', 0)
                })

            # Severity distribution
            severity_dist = report.get('severity_analysis', {}).get('severity_distribution', {})
            for level, count in severity_dist.items():
                summary_data.append({
                    'category': 'severity_distribution',
                    'severity_level': level,
                    'count': count,
                    'method': '',
                    'outlier_count': 0,
                    'outlier_percentage': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0
                })

            return summary_data

        except Exception as e:
            self.logger.warning(f"Error extracting summary for CSV: {str(e)}")
            return []