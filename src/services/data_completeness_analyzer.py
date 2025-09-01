import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError

logger = get_logger(__name__)


class DataCompletenessAnalyzer:
    """
    Comprehensive data completeness assessment system for Maria Conciliadora.
    Provides field-level, record-level, and dataset-level completeness analysis.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DataCompletenessAnalyzer with configuration

        Args:
            config: Configuration dictionary with completeness analysis settings
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Analysis results storage
        self.completeness_results = {}
        self.missing_patterns = {}
        self.trend_analysis = {}

        self.logger.info("DataCompletenessAnalyzer initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for completeness analysis"""
        return {
            'completeness_thresholds': {
                'field_level': 0.8,  # 80% completeness required per field
                'record_level': 0.7,  # 70% completeness required per record
                'dataset_level': 0.75  # 75% overall completeness required
            },
            'critical_fields': [
                'date', 'amount', 'description', 'transaction_type'
            ],
            'trend_analysis': {
                'window_days': 30,
                'min_samples': 10
            },
            'pattern_detection': {
                'min_pattern_support': 0.05,  # 5% minimum pattern support
                'max_patterns': 20
            }
        }

    def analyze_field_completeness(self, data: Union[pd.DataFrame, List[Dict]],
                                 field_name: str) -> Dict[str, Any]:
        """
        Analyze completeness of a specific field

        Args:
            data: DataFrame or list of dictionaries
            field_name: Name of the field to analyze

        Returns:
            Dictionary with field completeness metrics
        """
        try:
            self.logger.info(f"Analyzing field completeness for: {field_name}")

            # Convert to DataFrame if needed
            df = self._ensure_dataframe(data)

            if field_name not in df.columns:
                return {
                    'field_name': field_name,
                    'completeness_score': 0.0,
                    'total_records': len(df),
                    'missing_records': len(df),
                    'missing_percentage': 100.0,
                    'error': f"Field '{field_name}' not found in data"
                }

            # Calculate completeness metrics
            total_records = len(df)
            missing_records = df[field_name].isnull().sum()
            completeness_score = 1.0 - (missing_records / total_records) if total_records > 0 else 0.0

            # Additional field-specific metrics
            field_metrics = self._calculate_field_specific_metrics(df, field_name)

            result = {
                'field_name': field_name,
                'completeness_score': completeness_score,
                'total_records': total_records,
                'missing_records': missing_records,
                'missing_percentage': (missing_records / total_records * 100) if total_records > 0 else 100.0,
                'data_type': str(df[field_name].dtype),
                'unique_values': df[field_name].nunique() if not df[field_name].isnull().all() else 0,
                'is_critical': field_name in self.config['critical_fields'],
                **field_metrics
            }

            self.logger.info(f"Field '{field_name}' completeness: {completeness_score:.3f}")
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing field completeness for {field_name}: {str(e)}")
            return {
                'field_name': field_name,
                'error': str(e),
                'completeness_score': 0.0
            }

    def _calculate_field_specific_metrics(self, df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """Calculate field-specific completeness metrics"""
        metrics = {}

        try:
            series = df[field_name]

            # For numeric fields
            if pd.api.types.is_numeric_dtype(series):
                non_null_values = series.dropna()
                if len(non_null_values) > 0:
                    metrics.update({
                        'mean_value': non_null_values.mean(),
                        'median_value': non_null_values.median(),
                        'std_value': non_null_values.std(),
                        'min_value': non_null_values.min(),
                        'max_value': non_null_values.max(),
                        'zero_values': (non_null_values == 0).sum(),
                        'negative_values': (non_null_values < 0).sum()
                    })

            # For string/text fields
            elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                non_null_values = series.dropna().astype(str)
                if len(non_null_values) > 0:
                    metrics.update({
                        'avg_length': non_null_values.str.len().mean(),
                        'empty_strings': (non_null_values == '').sum(),
                        'whitespace_only': non_null_values.str.match(r'^\s*$').sum(),
                        'most_common_value': non_null_values.mode().iloc[0] if len(non_null_values) > 0 else None
                    })

            # For datetime fields
            elif pd.api.types.is_datetime64_any_dtype(series):
                non_null_values = series.dropna()
                if len(non_null_values) > 0:
                    metrics.update({
                        'date_range_days': (non_null_values.max() - non_null_values.min()).days,
                        'future_dates': (non_null_values > pd.Timestamp.now()).sum(),
                        'past_dates': (non_null_values < pd.Timestamp.now() - pd.DateOffset(years=10)).sum()
                    })

        except Exception as e:
            self.logger.warning(f"Error calculating field-specific metrics for {field_name}: {str(e)}")

        return metrics

    def analyze_record_completeness(self, data: Union[pd.DataFrame, List[Dict]],
                                  record_id_field: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Analyze completeness of individual records

        Args:
            data: DataFrame or list of dictionaries
            record_id_field: Field to use as record identifier

        Returns:
            List of dictionaries with record completeness metrics
        """
        try:
            self.logger.info("Analyzing record-level completeness")

            df = self._ensure_dataframe(data)
            total_fields = len(df.columns)

            record_completeness = []

            for idx, row in df.iterrows():
                # Calculate completeness for this record
                non_null_count = row.notna().sum()
                completeness_score = non_null_count / total_fields if total_fields > 0 else 0.0

                # Identify missing fields
                missing_fields = row[row.isnull()].index.tolist()

                record_info = {
                    'record_index': idx,
                    'record_id': row.get(record_id_field, f"record_{idx}") if record_id_field else f"record_{idx}",
                    'completeness_score': completeness_score,
                    'total_fields': total_fields,
                    'filled_fields': non_null_count,
                    'missing_fields': missing_fields,
                    'missing_count': len(missing_fields),
                    'is_complete': completeness_score >= self.config['completeness_thresholds']['record_level']
                }

                # Add field values for context
                if len(missing_fields) <= 5:  # Only include if not too many missing
                    record_info['field_values'] = {
                        field: row[field] for field in df.columns
                        if pd.notna(row[field]) or field in missing_fields[:3]  # Include some missing for context
                    }

                record_completeness.append(record_info)

            # Sort by completeness score (worst first)
            record_completeness.sort(key=lambda x: x['completeness_score'])

            self.logger.info(f"Analyzed completeness for {len(record_completeness)} records")
            return record_completeness

        except Exception as e:
            self.logger.error(f"Error analyzing record completeness: {str(e)}")
            return []

    def analyze_dataset_completeness(self, data: Union[pd.DataFrame, List[Dict]]) -> Dict[str, Any]:
        """
        Analyze overall dataset completeness

        Args:
            data: DataFrame or list of dictionaries

        Returns:
            Dictionary with dataset completeness metrics
        """
        try:
            self.logger.info("Analyzing dataset-level completeness")

            df = self._ensure_dataframe(data)

            # Field-level completeness
            field_completeness = []
            for column in df.columns:
                field_result = self.analyze_field_completeness(df, column)
                field_completeness.append(field_result)

            # Overall metrics
            total_records = len(df)
            total_fields = len(df.columns)
            total_cells = total_records * total_fields
            missing_cells = df.isnull().sum().sum()

            overall_completeness = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0

            # Critical fields analysis
            critical_fields_analysis = self._analyze_critical_fields(df)

            # Missing patterns
            missing_patterns = self._identify_missing_patterns(df)

            result = {
                'dataset_info': {
                    'total_records': total_records,
                    'total_fields': total_fields,
                    'total_cells': total_cells,
                    'missing_cells': missing_cells
                },
                'overall_completeness': {
                    'score': overall_completeness,
                    'percentage': overall_completeness * 100,
                    'is_acceptable': overall_completeness >= self.config['completeness_thresholds']['dataset_level']
                },
                'field_completeness': field_completeness,
                'critical_fields': critical_fields_analysis,
                'missing_patterns': missing_patterns,
                'summary': {
                    'fields_below_threshold': len([f for f in field_completeness
                                                 if f['completeness_score'] < self.config['completeness_thresholds']['field_level']]),
                    'records_below_threshold': len([r for r in self.analyze_record_completeness(df)
                                                  if r['completeness_score'] < self.config['completeness_thresholds']['record_level']]),
                    'most_problematic_field': min(field_completeness, key=lambda x: x['completeness_score']) if field_completeness else None
                }
            }

            self.logger.info(f"Dataset completeness analysis completed: {overall_completeness:.3f}")
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing dataset completeness: {str(e)}")
            return {'error': str(e)}

    def _analyze_critical_fields(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze completeness of critical fields"""
        critical_analysis = {}

        for field in self.config['critical_fields']:
            if field in df.columns:
                completeness = 1.0 - (df[field].isnull().sum() / len(df))
                critical_analysis[field] = {
                    'completeness_score': completeness,
                    'is_critical': True,
                    'meets_threshold': completeness >= self.config['completeness_thresholds']['field_level']
                }
            else:
                critical_analysis[field] = {
                    'completeness_score': 0.0,
                    'is_critical': True,
                    'meets_threshold': False,
                    'error': f"Critical field '{field}' not found in dataset"
                }

        return critical_analysis

    def _identify_missing_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify patterns in missing data"""
        try:
            # Create missing data matrix
            missing_matrix = df.isnull().astype(int)

            # Find correlations between missing values
            if missing_matrix.shape[1] > 1:
                missing_corr = missing_matrix.corr()

                # Extract strong correlations (missing together)
                patterns = []
                for i in range(len(missing_corr.columns)):
                    for j in range(i+1, len(missing_corr.columns)):
                        corr_value = missing_corr.iloc[i, j]
                        if abs(corr_value) > 0.3:  # Significant correlation
                            field1 = missing_corr.columns[i]
                            field2 = missing_corr.columns[j]

                            # Calculate pattern support
                            both_missing = ((missing_matrix[field1] == 1) & (missing_matrix[field2] == 1)).sum()
                            support = both_missing / len(df)

                            if support >= self.config['pattern_detection']['min_pattern_support']:
                                patterns.append({
                                    'fields': [field1, field2],
                                    'correlation': corr_value,
                                    'support': support,
                                    'both_missing_count': both_missing,
                                    'pattern_type': 'correlated_missing'
                                })

                # Sort by support and return top patterns
                patterns.sort(key=lambda x: x['support'], reverse=True)
                return patterns[:self.config['pattern_detection']['max_patterns']]

        except Exception as e:
            self.logger.warning(f"Error identifying missing patterns: {str(e)}")

        return []

    def analyze_completeness_trends(self, historical_data: List[Tuple[datetime, Union[pd.DataFrame, List[Dict]]]]) -> Dict[str, Any]:
        """
        Analyze completeness trends over time

        Args:
            historical_data: List of tuples (timestamp, data) for trend analysis

        Returns:
            Dictionary with trend analysis results
        """
        try:
            self.logger.info(f"Analyzing completeness trends for {len(historical_data)} time points")

            trends = {
                'field_trends': {},
                'overall_trends': [],
                'improvement_areas': []
            }

            # Process each time point
            for timestamp, data in historical_data:
                df = self._ensure_dataframe(data)
                completeness = self.analyze_dataset_completeness(df)

                # Store overall trend
                trends['overall_trends'].append({
                    'timestamp': timestamp.isoformat(),
                    'completeness_score': completeness['overall_completeness']['score'],
                    'missing_cells': completeness['dataset_info']['missing_cells']
                })

                # Store field-level trends
                for field_result in completeness['field_completeness']:
                    field_name = field_result['field_name']
                    if field_name not in trends['field_trends']:
                        trends['field_trends'][field_name] = []

                    trends['field_trends'][field_name].append({
                        'timestamp': timestamp.isoformat(),
                        'completeness_score': field_result['completeness_score'],
                        'missing_records': field_result['missing_records']
                    })

            # Analyze trends for improvement areas
            trends['improvement_areas'] = self._identify_trend_improvements(trends)

            self.logger.info("Completeness trend analysis completed")
            return trends

        except Exception as e:
            self.logger.error(f"Error analyzing completeness trends: {str(e)}")
            return {'error': str(e)}

    def _identify_trend_improvements(self, trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify areas that need improvement based on trends"""
        improvements = []

        try:
            # Check overall trend
            if len(trends['overall_trends']) >= 2:
                recent_scores = [t['completeness_score'] for t in trends['overall_trends'][-3:]]
                avg_recent = sum(recent_scores) / len(recent_scores)

                if avg_recent < self.config['completeness_thresholds']['dataset_level']:
                    improvements.append({
                        'type': 'overall_completeness',
                        'current_score': avg_recent,
                        'target_score': self.config['completeness_thresholds']['dataset_level'],
                        'priority': 'high'
                    })

            # Check field trends
            for field_name, field_trend in trends['field_trends'].items():
                if len(field_trend) >= 2:
                    recent_scores = [t['completeness_score'] for t in field_trend[-3:]]
                    avg_recent = sum(recent_scores) / len(recent_scores)

                    if avg_recent < self.config['completeness_thresholds']['field_level']:
                        improvements.append({
                            'type': 'field_completeness',
                            'field_name': field_name,
                            'current_score': avg_recent,
                            'target_score': self.config['completeness_thresholds']['field_level'],
                            'priority': 'medium'
                        })

        except Exception as e:
            self.logger.warning(f"Error identifying trend improvements: {str(e)}")

        return improvements

    def _ensure_dataframe(self, data: Union[pd.DataFrame, List[Dict]]) -> pd.DataFrame:
        """Ensure data is in DataFrame format"""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValidationError(f"Unsupported data type: {type(data)}")

    def generate_completeness_report(self, data: Union[pd.DataFrame, List[Dict]],
                                   include_recommendations: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive completeness report

        Args:
            data: Data to analyze
            include_recommendations: Whether to include improvement recommendations

        Returns:
            Comprehensive completeness report
        """
        try:
            self.logger.info("Generating comprehensive completeness report")

            # Perform all analyses
            dataset_analysis = self.analyze_dataset_completeness(data)
            record_analysis = self.analyze_record_completeness(data)

            report = {
                'timestamp': datetime.now().isoformat(),
                'dataset_analysis': dataset_analysis,
                'record_analysis_summary': {
                    'total_records': len(record_analysis),
                    'complete_records': len([r for r in record_analysis if r['is_complete']]),
                    'incomplete_records': len([r for r in record_analysis if not r['is_complete']]),
                    'worst_records': record_analysis[:5] if len(record_analysis) >= 5 else record_analysis
                },
                'recommendations': [] if include_recommendations else None
            }

            if include_recommendations:
                report['recommendations'] = self._generate_recommendations(dataset_analysis, record_analysis)

            self.logger.info("Completeness report generated successfully")
            return report

        except Exception as e:
            self.logger.error(f"Error generating completeness report: {str(e)}")
            return {'error': str(e)}

    def _generate_recommendations(self, dataset_analysis: Dict[str, Any],
                                record_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations for improving data completeness"""
        recommendations = []

        try:
            # Dataset-level recommendations
            overall_score = dataset_analysis['overall_completeness']['score']
            if overall_score < self.config['completeness_thresholds']['dataset_level']:
                recommendations.append({
                    'priority': 'high',
                    'category': 'dataset',
                    'issue': 'Low overall completeness',
                    'current_score': overall_score,
                    'target_score': self.config['completeness_thresholds']['dataset_level'],
                    'recommendation': 'Implement data validation rules and automated imputation for missing values'
                })

            # Field-level recommendations
            for field in dataset_analysis['field_completeness']:
                if field['completeness_score'] < self.config['completeness_thresholds']['field_level']:
                    priority = 'high' if field.get('is_critical', False) else 'medium'
                    recommendations.append({
                        'priority': priority,
                        'category': 'field',
                        'field_name': field['field_name'],
                        'issue': f'Low completeness for field {field["field_name"]}',
                        'current_score': field['completeness_score'],
                        'target_score': self.config['completeness_thresholds']['field_level'],
                        'recommendation': f'Consider making {field["field_name"]} a required field or implement imputation strategy'
                    })

            # Critical fields recommendations
            for field_name, analysis in dataset_analysis['critical_fields'].items():
                if not analysis.get('meets_threshold', True):
                    recommendations.append({
                        'priority': 'high',
                        'category': 'critical_field',
                        'field_name': field_name,
                        'issue': f'Critical field {field_name} below completeness threshold',
                        'current_score': analysis['completeness_score'],
                        'recommendation': 'Immediate attention required - implement mandatory validation for this field'
                    })

            # Sort by priority
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))

        except Exception as e:
            self.logger.warning(f"Error generating recommendations: {str(e)}")

        return recommendations