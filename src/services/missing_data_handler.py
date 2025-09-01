import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from src.services.data_completeness_analyzer import DataCompletenessAnalyzer
from src.services.advanced_imputation_engine import AdvancedImputationEngine

logger = get_logger(__name__)


class ImputationStrategy(Enum):
    """Enumeration of available imputation strategies"""
    STATISTICAL = "statistical"
    KNN = "knn"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    CONTEXT_AWARE = "context_aware"
    AUTO = "auto"


class ConfidenceLevel(Enum):
    """Enumeration of confidence levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ImputationResult:
    """Data class for imputation results"""
    original_data: pd.DataFrame
    imputed_data: pd.DataFrame
    strategy_used: ImputationStrategy
    confidence_score: float
    confidence_level: ConfidenceLevel
    imputation_count: int
    columns_affected: List[str]
    quality_metrics: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class CompletenessReport:
    """Data class for completeness reports"""
    dataset_completeness: float
    field_completeness: Dict[str, float]
    record_completeness: Dict[str, float]
    missing_patterns: List[Dict[str, Any]]
    critical_issues: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    timestamp: datetime


class MissingDataHandler:
    """
    Intelligent missing data management system that orchestrates completeness analysis
    and imputation strategies for optimal data quality.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MissingDataHandler with configuration

        Args:
            config: Configuration dictionary with missing data handling settings
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize core components
        self.completeness_analyzer = DataCompletenessAnalyzer(self.config.get('completeness_config', {}))
        self.imputation_engine = AdvancedImputationEngine(self.config.get('imputation_config', {}))

        # Track imputation history
        self.imputation_history: List[ImputationResult] = []
        self.completeness_reports: List[CompletenessReport] = []

        # Performance tracking
        self.performance_metrics = {
            'total_imputations': 0,
            'successful_imputations': 0,
            'average_confidence': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in ImputationStrategy}
        }

        self.logger.info("MissingDataHandler initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for missing data handler"""
        return {
            'completeness_config': {
                'completeness_thresholds': {
                    'field_level': 0.8,
                    'record_level': 0.7,
                    'dataset_level': 0.75
                },
                'critical_fields': ['date', 'amount', 'description', 'transaction_type']
            },
            'imputation_config': {
                'statistical_methods': {
                    'numeric_strategy': 'median',
                    'categorical_strategy': 'most_frequent'
                },
                'knn_imputation': {
                    'n_neighbors': 5
                },
                'confidence_scoring': {
                    'method': 'variance'
                }
            },
            'strategy_selection': {
                'auto_strategy': 'intelligent',
                'min_confidence_threshold': 0.6,
                'max_imputation_ratio': 0.5  # Maximum ratio of missing values before flagging
            },
            'quality_assurance': {
                'validate_imputation': True,
                'check_data_integrity': True,
                'performance_tracking': True
            }
        }

    def analyze_and_impute(self, data: Union[pd.DataFrame, List[Dict]],
                          strategy: Union[str, ImputationStrategy] = 'auto',
                          target_columns: Optional[List[str]] = None) -> ImputationResult:
        """
        Comprehensive analysis and imputation of missing data

        Args:
            data: Data to analyze and impute
            strategy: Imputation strategy to use
            target_columns: Specific columns to focus on (None for all)

        Returns:
            ImputationResult with complete analysis and imputation details
        """
        try:
            self.logger.info(f"Starting comprehensive analysis and imputation with strategy: {strategy}")

            # Convert to DataFrame
            df = self._ensure_dataframe(data)
            original_df = df.copy()

            # Step 1: Analyze completeness
            completeness_report = self.generate_completeness_report(df)
            self.completeness_reports.append(completeness_report)

            # Step 2: Determine imputation strategy
            if strategy == 'auto' or strategy == ImputationStrategy.AUTO:
                strategy = self._select_optimal_strategy(completeness_report, df)

            strategy_enum = strategy if isinstance(strategy, ImputationStrategy) else ImputationStrategy(strategy)

            # Step 3: Apply imputation
            imputed_df, imputation_details = self._apply_imputation_strategy(
                df, strategy_enum, target_columns
            )

            # Step 4: Calculate confidence and quality metrics
            confidence_score = self._calculate_overall_confidence(imputation_details)
            confidence_level = self._determine_confidence_level(confidence_score)

            quality_metrics = self.imputation_engine.get_imputation_quality_metrics(
                original_df, imputed_df
            )

            # Step 5: Validate results
            if self.config['quality_assurance']['validate_imputation']:
                validation_results = self._validate_imputation_results(
                    original_df, imputed_df, imputation_details
                )
                quality_metrics['validation_results'] = validation_results

            # Step 6: Create result object
            result = ImputationResult(
                original_data=original_df,
                imputed_data=imputed_df,
                strategy_used=strategy_enum,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                imputation_count=sum(imputation_details.get('imputation_counts', {}).values())
                              if isinstance(imputation_details, dict) else 0,
                columns_affected=imputation_details.get('columns_imputed', [])
                              if isinstance(imputation_details, dict) else [],
                quality_metrics=quality_metrics,
                timestamp=datetime.now(),
                metadata={
                    'strategy_details': imputation_details,
                    'completeness_report': {
                        'dataset_completeness': completeness_report.dataset_completeness,
                        'critical_issues_count': len(completeness_report.critical_issues)
                    }
                }
            )

            # Step 7: Update performance tracking
            self._update_performance_metrics(result)

            # Step 8: Store in history
            self.imputation_history.append(result)

            self.logger.info(f"Analysis and imputation completed: {result.imputation_count} values imputed "
                           f"with {result.confidence_level.value} confidence")

            return result

        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis and imputation: {str(e)}")
            raise ValidationError(f"Failed to complete analysis and imputation: {str(e)}")

    def generate_completeness_report(self, data: Union[pd.DataFrame, List[Dict]]) -> CompletenessReport:
        """
        Generate a comprehensive completeness report

        Args:
            data: Data to analyze

        Returns:
            CompletenessReport with detailed analysis
        """
        try:
            df = self._ensure_dataframe(data)

            # Get comprehensive completeness analysis
            analysis = self.completeness_analyzer.generate_completeness_report(df)

            # Extract key metrics
            dataset_completeness = analysis['dataset_analysis']['overall_completeness']['score']

            field_completeness = {
                field['field_name']: field['completeness_score']
                for field in analysis['dataset_analysis']['field_completeness']
            }

            # Calculate record completeness
            record_analysis = analysis['record_analysis_summary']
            record_completeness = {
                'average_completeness': sum(r['completeness_score'] for r in self.completeness_analyzer.analyze_record_completeness(df)) / len(df),
                'complete_records': record_analysis['complete_records'],
                'incomplete_records': record_analysis['incomplete_records']
            }

            # Extract missing patterns and critical issues
            missing_patterns = analysis['dataset_analysis']['missing_patterns']
            critical_issues = analysis['recommendations']

            return CompletenessReport(
                dataset_completeness=dataset_completeness,
                field_completeness=field_completeness,
                record_completeness=record_completeness,
                missing_patterns=missing_patterns,
                critical_issues=[issue for issue in critical_issues if issue.get('priority') == 'high'],
                recommendations=analysis['recommendations'],
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Error generating completeness report: {str(e)}")
            return CompletenessReport(
                dataset_completeness=0.0,
                field_completeness={},
                record_completeness={},
                missing_patterns=[],
                critical_issues=[{'error': str(e)}],
                recommendations=[],
                timestamp=datetime.now()
            )

    def _select_optimal_strategy(self, completeness_report: CompletenessReport,
                               df: pd.DataFrame) -> ImputationStrategy:
        """
        Select the optimal imputation strategy based on data characteristics

        Args:
            completeness_report: Completeness analysis results
            df: DataFrame to analyze

        Returns:
            Optimal imputation strategy
        """
        try:
            # Analyze missing data patterns
            missing_percentage = 1.0 - completeness_report.dataset_completeness
            critical_missing = any(
                completeness_report.field_completeness.get(field, 1.0) < 0.5
                for field in self.config['completeness_config']['critical_fields']
                if field in completeness_report.field_completeness
            )

            # High missing data (>30%) - use comprehensive approach
            if missing_percentage > 0.3:
                return ImputationStrategy.AUTO

            # Critical fields missing - use context-aware approach
            if critical_missing:
                return ImputationStrategy.CONTEXT_AWARE

            # Time series data detected
            if self._is_time_series_data(df):
                return ImputationStrategy.TIME_SERIES

            # Mixed data types with correlations
            if self._has_correlated_missing(df):
                return ImputationStrategy.KNN

            # Default to statistical for simple cases
            return ImputationStrategy.STATISTICAL

        except Exception as e:
            self.logger.warning(f"Error selecting optimal strategy: {str(e)}, using default")
            return ImputationStrategy.STATISTICAL

    def _is_time_series_data(self, df: pd.DataFrame) -> bool:
        """Check if data appears to be time series"""
        # Look for date/datetime columns
        date_columns = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_columns.append(col)
            elif df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(), errors='coerce')
                    date_columns.append(col)
                except:
                    pass

        return len(date_columns) > 0

    def _has_correlated_missing(self, df: pd.DataFrame) -> bool:
        """Check if missing values are correlated between columns"""
        try:
            missing_matrix = df.isnull().astype(int)
            if missing_matrix.shape[1] > 1:
                corr_matrix = missing_matrix.corr()
                # Check for strong correlations (>0.3)
                strong_corr = (corr_matrix.abs() > 0.3).sum().sum() - len(corr_matrix)
                return strong_corr > 0
        except:
            pass
        return False

    def _apply_imputation_strategy(self, df: pd.DataFrame,
                                 strategy: ImputationStrategy,
                                 target_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply the selected imputation strategy

        Args:
            df: DataFrame to impute
            strategy: Imputation strategy to apply
            target_columns: Specific columns to target

        Returns:
            Tuple of (imputed_data, imputation_details)
        """
        try:
            if strategy == ImputationStrategy.STATISTICAL:
                return self.imputation_engine.impute_statistical(df, target_columns)

            elif strategy == ImputationStrategy.KNN:
                return self.imputation_engine.impute_knn(df, target_columns)

            elif strategy == ImputationStrategy.REGRESSION:
                # Apply regression to each target column
                results = {}
                for col in target_columns or df.select_dtypes(include=[np.number]).columns:
                    if df[col].isnull().sum() > 0:
                        df, result = self.imputation_engine.impute_regression(df, col)
                        results[col] = result
                return df, {'method': 'regression', 'results': results}

            elif strategy == ImputationStrategy.TIME_SERIES:
                # Apply time series imputation (assuming date and value columns exist)
                date_col = None
                value_cols = []

                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        date_col = col
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        value_cols.append(col)

                if date_col and value_cols:
                    for value_col in value_cols:
                        df, _ = self.imputation_engine.impute_time_series(df, date_col, value_col)

                return df, {'method': 'time_series', 'date_column': date_col, 'value_columns': value_cols}

            elif strategy == ImputationStrategy.CONTEXT_AWARE:
                # Apply context-aware imputation to each column with missing data
                for col in df.columns:
                    if df[col].isnull().sum() > 0:
                        df, _ = self.imputation_engine.impute_context_aware(df, col)
                return df, {'method': 'context_aware'}

            elif strategy == ImputationStrategy.AUTO:
                return self.imputation_engine.auto_impute(df, 'intelligent')

            else:
                raise ValidationError(f"Unknown imputation strategy: {strategy}")

        except Exception as e:
            self.logger.error(f"Error applying imputation strategy {strategy}: {str(e)}")
            return df, {'error': str(e)}

    def _calculate_overall_confidence(self, imputation_details: Dict[str, Any]) -> float:
        """Calculate overall confidence score for imputation results"""
        try:
            if isinstance(imputation_details, dict):
                # Extract confidence scores from different methods
                confidences = []

                # Direct confidence score
                if 'confidence_score' in imputation_details:
                    confidences.append(imputation_details['confidence_score'])

                # Column-wise confidence scores
                if 'confidence_scores' in imputation_details:
                    confidences.extend(imputation_details['confidence_scores'].values())

                # Method-specific confidence
                if 'results' in imputation_details:
                    for result in imputation_details['results'].values():
                        if isinstance(result, dict) and 'confidence_score' in result:
                            confidences.append(result['confidence_score'])

                if confidences:
                    return sum(confidences) / len(confidences)

            return 0.5  # Default confidence

        except Exception as e:
            self.logger.warning(f"Error calculating overall confidence: {str(e)}")
            return 0.5

    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level from confidence score"""
        if confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _validate_imputation_results(self, original_df: pd.DataFrame,
                                   imputed_df: pd.DataFrame,
                                   imputation_details: Dict[str, Any]) -> Dict[str, Any]:
        """Validate imputation results for data integrity"""
        validation_results = {
            'data_integrity': True,
            'warnings': [],
            'errors': []
        }

        try:
            # Check data shape preservation
            if original_df.shape != imputed_df.shape:
                validation_results['data_integrity'] = False
                validation_results['errors'].append(
                    f"Data shape changed: {original_df.shape} -> {imputed_df.shape}"
                )

            # Check for new missing values
            original_missing = original_df.isnull().sum().sum()
            final_missing = imputed_df.isnull().sum().sum()

            if final_missing > original_missing:
                validation_results['warnings'].append(
                    f"New missing values introduced: {final_missing - original_missing}"
                )

            # Check data type consistency
            for col in original_df.columns:
                if col in imputed_df.columns:
                    if original_df[col].dtype != imputed_df[col].dtype:
                        validation_results['warnings'].append(
                            f"Data type changed for column {col}: {original_df[col].dtype} -> {imputed_df[col].dtype}"
                        )

            # Check for extreme outliers in numeric columns
            for col in original_df.select_dtypes(include=[np.number]).columns:
                if col in imputed_df.columns:
                    original_stats = original_df[col].describe()
                    imputed_stats = imputed_df[col].describe()

                    # Check if imputed values are within reasonable range
                    if pd.notna(imputed_stats['min']) and pd.notna(imputed_stats['max']):
                        original_range = original_stats['max'] - original_stats['min']
                        if original_range > 0:
                            # Flag if imputed values extend range by more than 50%
                            range_extension = max(
                                abs(imputed_stats['min'] - original_stats['min']),
                                abs(imputed_stats['max'] - original_stats['max'])
                            )
                            if range_extension > original_range * 0.5:
                                validation_results['warnings'].append(
                                    f"Extreme values detected in column {col} after imputation"
                                )

        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['data_integrity'] = False

        return validation_results

    def _update_performance_metrics(self, result: ImputationResult):
        """Update performance tracking metrics"""
        try:
            self.performance_metrics['total_imputations'] += 1

            if result.confidence_level in [ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]:
                self.performance_metrics['successful_imputations'] += 1

            # Update strategy usage
            self.performance_metrics['strategy_usage'][result.strategy_used.value] += 1

            # Update average confidence
            total_imputations = self.performance_metrics['total_imputations']
            current_avg = self.performance_metrics['average_confidence']
            new_avg = (current_avg * (total_imputations - 1) + result.confidence_score) / total_imputations
            self.performance_metrics['average_confidence'] = new_avg

        except Exception as e:
            self.logger.warning(f"Error updating performance metrics: {str(e)}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the missing data handler"""
        try:
            total = self.performance_metrics['total_imputations']
            successful = self.performance_metrics['successful_imputations']

            return {
                'total_imputations': total,
                'successful_imputations': successful,
                'success_rate': successful / total if total > 0 else 0.0,
                'average_confidence': self.performance_metrics['average_confidence'],
                'strategy_usage': self.performance_metrics['strategy_usage'],
                'history_length': len(self.imputation_history),
                'reports_count': len(self.completeness_reports)
            }

        except Exception as e:
            self.logger.error(f"Error generating performance summary: {str(e)}")
            return {'error': str(e)}

    def get_imputation_recommendations(self, data: Union[pd.DataFrame, List[Dict]]) -> List[Dict[str, Any]]:
        """
        Get recommendations for handling missing data in the given dataset

        Args:
            data: Data to analyze

        Returns:
            List of recommendations
        """
        try:
            df = self._ensure_dataframe(data)
            completeness_report = self.generate_completeness_report(df)

            recommendations = []

            # Dataset-level recommendations
            if completeness_report.dataset_completeness < self.config['completeness_config']['dataset_level']:
                recommendations.append({
                    'type': 'dataset',
                    'priority': 'high',
                    'issue': 'Low overall data completeness',
                    'current_score': completeness_report.dataset_completeness,
                    'recommended_action': 'Implement comprehensive imputation strategy',
                    'suggested_strategy': ImputationStrategy.AUTO.value
                })

            # Field-level recommendations
            for field, completeness in completeness_report.field_completeness.items():
                if completeness < self.config['completeness_config']['field_level']:
                    priority = 'high' if field in self.config['completeness_config']['critical_fields'] else 'medium'

                    recommendations.append({
                        'type': 'field',
                        'priority': priority,
                        'field': field,
                        'issue': f'Low completeness in {field}',
                        'current_score': completeness,
                        'recommended_action': 'Apply targeted imputation',
                        'suggested_strategy': self._get_field_specific_strategy(df, field)
                    })

            # Pattern-based recommendations
            for pattern in completeness_report.missing_patterns:
                if pattern.get('support', 0) > 0.1:  # More than 10% support
                    recommendations.append({
                        'type': 'pattern',
                        'priority': 'medium',
                        'issue': f'Correlated missing values in {pattern.get("fields", [])}',
                        'correlation': pattern.get('correlation', 0),
                        'recommended_action': 'Use multivariate imputation method',
                        'suggested_strategy': ImputationStrategy.KNN.value
                    })

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return [{'error': str(e)}]

    def _get_field_specific_strategy(self, df: pd.DataFrame, field: str) -> str:
        """Get field-specific imputation strategy recommendation"""
        try:
            if pd.api.types.is_numeric_dtype(df[field]):
                if self._is_time_series_data(df):
                    return ImputationStrategy.TIME_SERIES.value
                else:
                    return ImputationStrategy.CONTEXT_AWARE.value
            else:
                return ImputationStrategy.STATISTICAL.value
        except:
            return ImputationStrategy.STATISTICAL.value

    def _ensure_dataframe(self, data: Union[pd.DataFrame, List[Dict]]) -> pd.DataFrame:
        """Ensure data is in DataFrame format"""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValidationError(f"Unsupported data type: {type(data)}")

    def clear_history(self):
        """Clear imputation history and reset performance metrics"""
        self.imputation_history.clear()
        self.completeness_reports.clear()
        self.performance_metrics = {
            'total_imputations': 0,
            'successful_imputations': 0,
            'average_confidence': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in ImputationStrategy}
        }
        self.logger.info("Imputation history and performance metrics cleared")