import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import re

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError

logger = get_logger(__name__)


class AdvancedImputationEngine:
    """
    Advanced imputation engine with multiple strategies for handling missing data.
    Supports statistical, machine learning, and context-aware imputation methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AdvancedImputationEngine with configuration

        Args:
            config: Configuration dictionary with imputation settings
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize imputers and models
        self._initialize_imputers()

        # Track imputation results
        self.imputation_history = []
        self.confidence_scores = {}

        self.logger.info("AdvancedImputationEngine initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for imputation engine"""
        return {
            'statistical_methods': {
                'numeric_strategy': 'median',  # 'mean', 'median', 'most_frequent', 'constant'
                'categorical_strategy': 'most_frequent',  # 'most_frequent', 'constant'
                'constant_value': 0
            },
            'knn_imputation': {
                'n_neighbors': 5,
                'weights': 'uniform',  # 'uniform', 'distance'
                'metric': 'nan_euclidean'
            },
            'regression_imputation': {
                'model_type': 'random_forest',  # 'linear', 'ridge', 'random_forest'
                'test_size': 0.2,
                'random_state': 42
            },
            'time_series_imputation': {
                'method': 'interpolation',  # 'interpolation', 'forward_fill', 'backward_fill', 'moving_average'
                'interpolation_order': 2,
                'moving_window': 7
            },
            'categorical_imputation': {
                'method': 'frequency',  # 'frequency', 'model_based'
                'min_samples': 10
            },
            'context_aware': {
                'correlation_threshold': 0.3,
                'max_related_features': 5
            },
            'confidence_scoring': {
                'method': 'variance',  # 'variance', 'prediction_interval', 'bootstrap'
                'n_bootstraps': 100
            }
        }

    def _initialize_imputers(self):
        """Initialize various imputation methods"""
        try:
            # Statistical imputers
            self.statistical_imputer_numeric = SimpleImputer(
                strategy=self.config['statistical_methods']['numeric_strategy'],
                fill_value=self.config['statistical_methods']['constant_value']
            )

            self.statistical_imputer_categorical = SimpleImputer(
                strategy=self.config['statistical_methods']['categorical_strategy'],
                fill_value='unknown'
            )

            # KNN imputer
            self.knn_imputer = KNNImputer(
                n_neighbors=self.config['knn_imputation']['n_neighbors'],
                weights=self.config['knn_imputation']['weights'],
                metric=self.config['knn_imputation']['metric']
            )

            # Regression models for imputation
            if self.config['regression_imputation']['model_type'] == 'linear':
                self.regression_model = LinearRegression()
            elif self.config['regression_imputation']['model_type'] == 'ridge':
                self.regression_model = Ridge(alpha=1.0)
            else:  # random_forest
                self.regression_model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.config['regression_imputation']['random_state']
                )

            # Scaler for preprocessing
            self.scaler = StandardScaler()

            self.logger.info("Imputation methods initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing imputers: {str(e)}")
            raise ValidationError(f"Failed to initialize imputation methods: {str(e)}")

    def impute_statistical(self, data: Union[pd.DataFrame, List[Dict]],
                          columns: Optional[List[str]] = None,
                          method: str = 'auto') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform statistical imputation (mean, median, mode)

        Args:
            data: Data to impute
            columns: Specific columns to impute (None for all)
            method: Imputation method ('mean', 'median', 'mode', 'auto')

        Returns:
            Tuple of (imputed_data, imputation_info)
        """
        try:
            self.logger.info(f"Performing statistical imputation with method: {method}")

            df = self._ensure_dataframe(data)
            imputation_info = {
                'method': 'statistical',
                'strategy': method,
                'columns_imputed': [],
                'imputation_counts': {},
                'confidence_scores': {}
            }

            # Determine columns to impute
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            # Impute each column
            for column in columns:
                if column not in df.columns:
                    continue

                original_missing = df[column].isnull().sum()
                if original_missing == 0:
                    continue

                # Choose strategy based on data type and method
                if pd.api.types.is_numeric_dtype(df[column]):
                    strategy = self._choose_numeric_strategy(df[column], method)
                    imputer = SimpleImputer(strategy=strategy)

                    # Fit and transform
                    imputed_values = imputer.fit_transform(df[[column]]).ravel()
                    df[column] = imputed_values

                elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == 'object':
                    strategy = 'most_frequent' if method == 'auto' else method
                    imputer = SimpleImputer(strategy=strategy, fill_value='unknown')

                    imputed_values = imputer.fit_transform(df[[column]]).ravel()
                    df[column] = imputed_values

                # Track imputation
                final_missing = df[column].isnull().sum()
                imputation_info['columns_imputed'].append(column)
                imputation_info['imputation_counts'][column] = original_missing - final_missing

                # Calculate confidence score
                confidence = self._calculate_statistical_confidence(df[column], original_missing)
                imputation_info['confidence_scores'][column] = confidence

            self.logger.info(f"Statistical imputation completed for {len(imputation_info['columns_imputed'])} columns")
            return df, imputation_info

        except Exception as e:
            self.logger.error(f"Error in statistical imputation: {str(e)}")
            return self._ensure_dataframe(data), {'error': str(e)}

    def _choose_numeric_strategy(self, series: pd.Series, method: str) -> str:
        """Choose appropriate statistical strategy for numeric data"""
        if method != 'auto':
            return method

        # Auto-choose based on data characteristics
        if series.skew() > 1:  # Highly skewed
            return 'median'
        elif series.std() / series.mean() > 0.5:  # High variance
            return 'median'
        else:
            return 'mean'

    def _calculate_statistical_confidence(self, series: pd.Series, original_missing: int) -> float:
        """Calculate confidence score for statistical imputation"""
        try:
            if len(series) == 0 or original_missing == 0:
                return 1.0

            # Confidence based on data variability
            if pd.api.types.is_numeric_dtype(series):
                cv = series.std() / series.mean() if series.mean() != 0 else 0
                confidence = max(0.1, 1.0 - min(cv, 1.0))  # Higher confidence for lower variability
            else:
                # For categorical, confidence based on mode frequency
                mode_freq = series.value_counts().iloc[0] / len(series)
                confidence = min(1.0, mode_freq * 2)  # Scale mode frequency to confidence

            return confidence

        except:
            return 0.5  # Default confidence

    def impute_knn(self, data: Union[pd.DataFrame, List[Dict]],
                  columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform KNN-based imputation

        Args:
            data: Data to impute
            columns: Specific columns to impute (None for all numeric)

        Returns:
            Tuple of (imputed_data, imputation_info)
        """
        try:
            self.logger.info("Performing KNN imputation")

            df = self._ensure_dataframe(data)
            imputation_info = {
                'method': 'knn',
                'n_neighbors': self.config['knn_imputation']['n_neighbors'],
                'columns_imputed': [],
                'imputation_counts': {},
                'confidence_scores': {}
            }

            # Select numeric columns only
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            numeric_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

            if not numeric_columns:
                self.logger.warning("No numeric columns found for KNN imputation")
                return df, imputation_info

            # Prepare data for KNN
            numeric_data = df[numeric_columns].copy()

            # Track original missing values
            original_missing_mask = numeric_data.isnull()

            # Scale the data
            scaled_data = self.scaler.fit_transform(numeric_data)

            # Perform KNN imputation
            imputed_scaled = self.knn_imputer.fit_transform(scaled_data)

            # Inverse transform
            imputed_data = self.scaler.inverse_transform(imputed_scaled)

            # Update original dataframe
            for i, column in enumerate(numeric_columns):
                original_missing_count = original_missing_mask[column].sum()
                if original_missing_count > 0:
                    df[column] = imputed_data[:, i]
                    imputation_info['columns_imputed'].append(column)
                    imputation_info['imputation_counts'][column] = original_missing_count

                    # Calculate confidence based on KNN distances
                    confidence = self._calculate_knn_confidence(imputed_data[:, i], original_missing_mask[column])
                    imputation_info['confidence_scores'][column] = confidence

            self.logger.info(f"KNN imputation completed for {len(imputation_info['columns_imputed'])} columns")
            return df, imputation_info

        except Exception as e:
            self.logger.error(f"Error in KNN imputation: {str(e)}")
            return self._ensure_dataframe(data), {'error': str(e)}

    def _calculate_knn_confidence(self, imputed_values: np.ndarray, missing_mask: pd.Series) -> float:
        """Calculate confidence score for KNN imputation"""
        try:
            if not missing_mask.any():
                return 1.0

            # Calculate variance of imputed values as inverse confidence measure
            imputed_subset = imputed_values[missing_mask]
            if len(imputed_subset) > 1:
                variance = np.var(imputed_subset)
                # Higher variance = lower confidence
                confidence = max(0.1, 1.0 - min(variance / (np.mean(imputed_subset)**2 + 1), 0.9))
            else:
                confidence = 0.5

            return confidence

        except:
            return 0.5

    def impute_regression(self, data: Union[pd.DataFrame, List[Dict]],
                         target_column: str,
                         predictor_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform regression-based imputation for a target column

        Args:
            data: Data to impute
            target_column: Column to impute
            predictor_columns: Columns to use as predictors (auto-selected if None)

        Returns:
            Tuple of (imputed_data, imputation_info)
        """
        try:
            self.logger.info(f"Performing regression imputation for column: {target_column}")

            df = self._ensure_dataframe(data)
            imputation_info = {
                'method': 'regression',
                'target_column': target_column,
                'predictor_columns': [],
                'model_type': self.config['regression_imputation']['model_type'],
                'imputation_count': 0,
                'model_performance': {},
                'confidence_score': 0.0
            }

            if target_column not in df.columns:
                return df, {'error': f"Target column '{target_column}' not found"}

            # Separate complete and incomplete data
            complete_mask = ~df[target_column].isnull()
            incomplete_mask = df[target_column].isnull()

            complete_data = df[complete_mask].copy()
            incomplete_data = df[incomplete_mask].copy()

            if len(complete_data) == 0:
                self.logger.warning(f"No complete data available for regression imputation of {target_column}")
                return df, imputation_info

            # Select predictor columns
            if predictor_columns is None:
                predictor_columns = self._select_predictor_columns(df, target_column, complete_data)

            imputation_info['predictor_columns'] = predictor_columns

            if not predictor_columns:
                self.logger.warning(f"No suitable predictor columns found for {target_column}")
                return df, imputation_info

            # Prepare training data
            X_train = complete_data[predictor_columns]
            y_train = complete_data[target_column]

            # Handle missing values in predictors (use statistical imputation)
            X_train_imputed, _ = self.impute_statistical(X_train)

            # Prepare prediction data
            X_pred = incomplete_data[predictor_columns]
            X_pred_imputed, _ = self.impute_statistical(X_pred)

            # Train model and make predictions
            if pd.api.types.is_numeric_dtype(df[target_column]):
                # Regression for numeric target
                self.regression_model.fit(X_train_imputed, y_train)
                predictions = self.regression_model.predict(X_pred_imputed)

                # Calculate model performance
                if len(complete_data) > 10:
                    from sklearn.model_selection import cross_val_score
                    scores = cross_val_score(self.regression_model, X_train_imputed, y_train,
                                           cv=min(5, len(complete_data)), scoring='r2')
                    imputation_info['model_performance'] = {
                        'r2_score': scores.mean(),
                        'r2_std': scores.std()
                    }
            else:
                # Classification for categorical target
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train)

                classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                classifier.fit(X_train_imputed, y_train_encoded)
                predictions_encoded = classifier.predict(X_pred_imputed)
                predictions = le.inverse_transform(predictions_encoded)

            # Update original dataframe
            df.loc[incomplete_mask, target_column] = predictions
            imputation_info['imputation_count'] = len(predictions)

            # Calculate confidence
            imputation_info['confidence_score'] = self._calculate_regression_confidence(
                predictions, complete_data[target_column]
            )

            self.logger.info(f"Regression imputation completed for {target_column}: {len(predictions)} values imputed")
            return df, imputation_info

        except Exception as e:
            self.logger.error(f"Error in regression imputation: {str(e)}")
            return self._ensure_dataframe(data), {'error': str(e)}

    def _select_predictor_columns(self, df: pd.DataFrame, target_column: str,
                                complete_data: pd.DataFrame) -> List[str]:
        """Select best predictor columns based on correlation"""
        try:
            predictors = []

            # Calculate correlations with target
            if pd.api.types.is_numeric_dtype(df[target_column]):
                correlations = complete_data.corr()[target_column].abs().sort_values(ascending=False)
                # Select top correlated numeric columns (excluding target)
                numeric_cols = [col for col in correlations.index
                              if col != target_column and pd.api.types.is_numeric_dtype(df[col])]
                predictors.extend(numeric_cols[:self.config['context_aware']['max_related_features']])
            else:
                # For categorical target, use chi-square or mutual information
                from sklearn.feature_selection import mutual_info_classif
                from sklearn.preprocessing import LabelEncoder

                le = LabelEncoder()
                y_encoded = le.fit_transform(complete_data[target_column])

                numeric_predictors = complete_data.select_dtypes(include=[np.number]).columns
                if len(numeric_predictors) > 0:
                    X_numeric = complete_data[numeric_predictors]
                    if X_numeric.shape[1] > 0:
                        mi_scores = mutual_info_classif(X_numeric, y_encoded)
                        top_indices = np.argsort(mi_scores)[-self.config['context_aware']['max_related_features']:]
                        predictors.extend([numeric_predictors[i] for i in top_indices])

            return predictors

        except Exception as e:
            self.logger.warning(f"Error selecting predictor columns: {str(e)}")
            return []

    def _calculate_regression_confidence(self, predictions: np.ndarray,
                                       training_target: pd.Series) -> float:
        """Calculate confidence score for regression imputation"""
        try:
            if len(predictions) == 0:
                return 0.0

            # Confidence based on prediction variance vs training variance
            pred_std = np.std(predictions)
            train_std = training_target.std()

            if train_std > 0:
                confidence = max(0.1, 1.0 - (pred_std / train_std))
            else:
                confidence = 0.5

            return min(confidence, 1.0)

        except:
            return 0.5

    def impute_time_series(self, data: Union[pd.DataFrame, List[Dict]],
                          time_column: str,
                          value_column: str,
                          method: str = 'auto') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform time series imputation

        Args:
            data: Time series data to impute
            time_column: Column containing timestamps
            value_column: Column to impute
            method: Imputation method ('interpolation', 'forward_fill', 'backward_fill', 'moving_average')

        Returns:
            Tuple of (imputed_data, imputation_info)
        """
        try:
            self.logger.info(f"Performing time series imputation for {value_column}")

            df = self._ensure_dataframe(data)
            imputation_info = {
                'method': 'time_series',
                'time_column': time_column,
                'value_column': value_column,
                'imputation_method': method,
                'imputation_count': 0,
                'confidence_score': 0.0
            }

            if time_column not in df.columns or value_column not in df.columns:
                return df, {'error': f"Required columns not found: {time_column}, {value_column}"}

            # Sort by time
            df = df.sort_values(time_column).copy()

            # Convert time column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                df[time_column] = pd.to_datetime(df[time_column], errors='coerce')

            original_missing = df[value_column].isnull().sum()

            if method == 'auto':
                method = self.config['time_series_imputation']['method']

            # Apply imputation method
            if method == 'interpolation':
                df[value_column] = df[value_column].interpolate(
                    method='polynomial',
                    order=self.config['time_series_imputation']['interpolation_order']
                )
            elif method == 'forward_fill':
                df[value_column] = df[value_column].fillna(method='ffill')
            elif method == 'backward_fill':
                df[value_column] = df[value_column].fillna(method='bfill')
            elif method == 'moving_average':
                window = self.config['time_series_imputation']['moving_window']
                df[value_column] = df[value_column].fillna(df[value_column].rolling(window=window, center=True).mean())

            # Fill any remaining NaN with statistical imputation
            if df[value_column].isnull().sum() > 0:
                df, _ = self.impute_statistical(df, [value_column])

            final_missing = df[value_column].isnull().sum()
            imputation_info['imputation_count'] = original_missing - final_missing
            imputation_info['confidence_score'] = self._calculate_time_series_confidence(df[value_column])

            self.logger.info(f"Time series imputation completed: {imputation_info['imputation_count']} values imputed")
            return df, imputation_info

        except Exception as e:
            self.logger.error(f"Error in time series imputation: {str(e)}")
            return self._ensure_dataframe(data), {'error': str(e)}

    def _calculate_time_series_confidence(self, series: pd.Series) -> float:
        """Calculate confidence score for time series imputation"""
        try:
            if series.isnull().sum() > 0:
                return 0.0

            # Confidence based on interpolation smoothness
            if len(series) > 2:
                # Calculate second derivative (smoothness measure)
                first_diff = series.diff().abs()
                second_diff = first_diff.diff().abs()

                # Lower second derivative = smoother = higher confidence
                smoothness = 1.0 / (1.0 + second_diff.mean())
                confidence = min(smoothness, 1.0)
            else:
                confidence = 0.8

            return confidence

        except:
            return 0.5

    def impute_context_aware(self, data: Union[pd.DataFrame, List[Dict]],
                           target_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform context-aware imputation using related fields

        Args:
            data: Data to impute
            target_column: Column to impute

        Returns:
            Tuple of (imputed_data, imputation_info)
        """
        try:
            self.logger.info(f"Performing context-aware imputation for {target_column}")

            df = self._ensure_dataframe(data)
            imputation_info = {
                'method': 'context_aware',
                'target_column': target_column,
                'related_fields': [],
                'imputation_count': 0,
                'confidence_score': 0.0
            }

            if target_column not in df.columns:
                return df, {'error': f"Target column '{target_column}' not found"}

            # Find related fields based on correlation or patterns
            related_fields = self._find_related_fields(df, target_column)
            imputation_info['related_fields'] = related_fields

            if not related_fields:
                # Fallback to statistical imputation
                self.logger.info(f"No related fields found, using statistical imputation for {target_column}")
                return self.impute_statistical(df, [target_column])

            # Use regression-based imputation with related fields
            return self.impute_regression(df, target_column, related_fields)

        except Exception as e:
            self.logger.error(f"Error in context-aware imputation: {str(e)}")
            return self._ensure_dataframe(data), {'error': str(e)}

    def _find_related_fields(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """Find fields related to the target column"""
        try:
            related_fields = []

            # Calculate correlations for numeric columns
            if pd.api.types.is_numeric_dtype(df[target_column]):
                correlations = df.corr()[target_column].abs().sort_values(ascending=False)
                threshold = self.config['context_aware']['correlation_threshold']

                for col in correlations.index:
                    if (col != target_column and
                        correlations[col] >= threshold and
                        pd.api.types.is_numeric_dtype(df[col])):
                        related_fields.append(col)

            else:
                # For categorical target, use mutual information
                from sklearn.feature_selection import mutual_info_classif
                from sklearn.preprocessing import LabelEncoder

                le = LabelEncoder()
                target_encoded = le.fit_transform(df[target_column].dropna())

                # Get complete cases for target
                complete_mask = ~df[target_column].isnull()
                numeric_cols = df.select_dtypes(include=[np.number]).columns

                if len(numeric_cols) > 0 and len(target_encoded) > 0:
                    X_numeric = df.loc[complete_mask, numeric_cols]
                    if X_numeric.shape[1] > 0:
                        mi_scores = mutual_info_classif(X_numeric, target_encoded)
                        threshold = np.mean(mi_scores) + np.std(mi_scores)  # Above average + 1 std

                        for i, col in enumerate(numeric_cols):
                            if mi_scores[i] >= threshold:
                                related_fields.append(col)

            return related_fields[:self.config['context_aware']['max_related_features']]

        except Exception as e:
            self.logger.warning(f"Error finding related fields: {str(e)}")
            return []

    def auto_impute(self, data: Union[pd.DataFrame, List[Dict]],
                   strategy: str = 'intelligent') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Automatically choose and apply imputation strategies

        Args:
            data: Data to impute
            strategy: Auto-imputation strategy ('simple', 'intelligent', 'comprehensive')

        Returns:
            Tuple of (imputed_data, imputation_summary)
        """
        try:
            self.logger.info(f"Performing auto-imputation with strategy: {strategy}")

            df = self._ensure_dataframe(data)
            imputation_summary = {
                'strategy': strategy,
                'methods_used': [],
                'total_imputations': 0,
                'columns_imputed': [],
                'overall_confidence': 0.0,
                'method_details': []
            }

            # Analyze missing data patterns
            missing_analysis = self._analyze_missing_patterns(df)

            if strategy == 'simple':
                # Simple statistical imputation for all
                df, method_info = self.impute_statistical(df)
                imputation_summary['methods_used'].append('statistical')
                imputation_summary['method_details'].append(method_info)

            elif strategy == 'intelligent':
                # Intelligent strategy based on data characteristics
                df = self._apply_intelligent_strategy(df, missing_analysis, imputation_summary)

            elif strategy == 'comprehensive':
                # Comprehensive imputation with multiple methods
                df = self._apply_comprehensive_strategy(df, missing_analysis, imputation_summary)

            # Calculate overall statistics
            imputation_summary['total_imputations'] = sum(
                sum(details.get('imputation_counts', {}).values())
                for details in imputation_summary['method_details']
                if isinstance(details, dict)
            )

            imputation_summary['columns_imputed'] = list(set(
                col for details in imputation_summary['method_details']
                if isinstance(details, dict)
                for col in details.get('columns_imputed', [])
            ))

            # Calculate overall confidence
            confidences = [
                details.get('confidence_score', 0.5)
                for details in imputation_summary['method_details']
                if isinstance(details, dict) and 'confidence_score' in details
            ]
            if confidences:
                imputation_summary['overall_confidence'] = sum(confidences) / len(confidences)

            self.logger.info(f"Auto-imputation completed: {imputation_summary['total_imputations']} values imputed")
            return df, imputation_summary

        except Exception as e:
            self.logger.error(f"Error in auto-imputation: {str(e)}")
            return self._ensure_dataframe(data), {'error': str(e)}

    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing data"""
        missing_patterns = {
            'missing_by_column': {},
            'missing_by_row': {},
            'correlation_patterns': []
        }

        # Column-wise missing analysis
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = missing_count / len(df)
            missing_patterns['missing_by_column'][col] = {
                'count': missing_count,
                'percentage': missing_percentage,
                'severity': 'high' if missing_percentage > 0.5 else 'medium' if missing_percentage > 0.2 else 'low'
            }

        # Row-wise missing analysis
        row_missing_counts = df.isnull().sum(axis=1)
        missing_patterns['missing_by_row'] = {
            'avg_missing_per_row': row_missing_counts.mean(),
            'max_missing_per_row': row_missing_counts.max(),
            'rows_with_missing': (row_missing_counts > 0).sum()
        }

        return missing_patterns

    def _apply_intelligent_strategy(self, df: pd.DataFrame,
                                  missing_analysis: Dict[str, Any],
                                  summary: Dict[str, Any]) -> pd.DataFrame:
        """Apply intelligent imputation strategy"""
        # Process columns by missing severity
        high_missing_cols = [
            col for col, info in missing_analysis['missing_by_column'].items()
            if info['severity'] == 'high'
        ]
        medium_missing_cols = [
            col for col, info in missing_analysis['missing_by_column'].items()
            if info['severity'] == 'medium'
        ]
        low_missing_cols = [
            col for col, info in missing_analysis['missing_by_column'].items()
            if info['severity'] == 'low'
        ]

        # High missing: Use statistical imputation
        if high_missing_cols:
            df, method_info = self.impute_statistical(df, high_missing_cols)
            summary['methods_used'].append('statistical_high_missing')
            summary['method_details'].append(method_info)

        # Medium missing: Try KNN if suitable
        numeric_cols = [col for col in medium_missing_cols if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_cols and len(numeric_cols) > 1:
            df, method_info = self.impute_knn(df, numeric_cols)
            summary['methods_used'].append('knn_medium_missing')
            summary['method_details'].append(method_info)

        # Low missing: Use context-aware imputation
        for col in low_missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df, method_info = self.impute_context_aware(df, col)
                summary['methods_used'].append(f'context_aware_{col}')
                summary['method_details'].append(method_info)

        return df

    def _apply_comprehensive_strategy(self, df: pd.DataFrame,
                                    missing_analysis: Dict[str, Any],
                                    summary: Dict[str, Any]) -> pd.DataFrame:
        """Apply comprehensive imputation strategy"""
        # First pass: Statistical imputation for all
        df, method_info = self.impute_statistical(df)
        summary['methods_used'].append('statistical_all')
        summary['method_details'].append(method_info)

        # Second pass: KNN for numeric columns with remaining missing
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols and df[numeric_cols].isnull().any().any():
            df, method_info = self.impute_knn(df, numeric_cols)
            summary['methods_used'].append('knn_refinement')
            summary['method_details'].append(method_info)

        # Third pass: Context-aware for any remaining missing
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df, method_info = self.impute_context_aware(df, col)
                summary['methods_used'].append(f'context_aware_final_{col}')
                summary['method_details'].append(method_info)

        return df

    def _ensure_dataframe(self, data: Union[pd.DataFrame, List[Dict]]) -> pd.DataFrame:
        """Ensure data is in DataFrame format"""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValidationError(f"Unsupported data type: {type(data)}")

    def get_imputation_quality_metrics(self, original_data: Union[pd.DataFrame, List[Dict]],
                                     imputed_data: Union[pd.DataFrame, List[Dict]]) -> Dict[str, Any]:
        """
        Calculate quality metrics for imputation results

        Args:
            original_data: Original data with missing values
            imputed_data: Data after imputation

        Returns:
            Dictionary with quality metrics
        """
        try:
            original_df = self._ensure_dataframe(original_data)
            imputed_df = self._ensure_dataframe(imputed_data)

            quality_metrics = {
                'overall_metrics': {},
                'column_metrics': {},
                'data_integrity_checks': {}
            }

            # Overall metrics
            original_missing = original_df.isnull().sum().sum()
            final_missing = imputed_df.isnull().sum().sum()
            quality_metrics['overall_metrics'] = {
                'original_missing_values': original_missing,
                'final_missing_values': final_missing,
                'imputation_success_rate': 1.0 - (final_missing / original_missing) if original_missing > 0 else 1.0,
                'data_preservation_rate': (len(imputed_df) - final_missing) / len(imputed_df)
            }

            # Column-wise metrics
            for col in original_df.columns:
                if col in imputed_df.columns:
                    original_missing_col = original_df[col].isnull().sum()
                    final_missing_col = imputed_df[col].isnull().sum()

                    col_metrics = {
                        'original_missing': original_missing_col,
                        'final_missing': final_missing_col,
                        'imputation_success': original_missing_col - final_missing_col,
                        'success_rate': 1.0 - (final_missing_col / original_missing_col) if original_missing_col > 0 else 1.0
                    }

                    # Data type consistency
                    if pd.api.types.is_numeric_dtype(original_df[col]):
                        col_metrics['data_type_consistency'] = pd.api.types.is_numeric_dtype(imputed_df[col])
                    else:
                        col_metrics['data_type_consistency'] = original_df[col].dtype == imputed_df[col].dtype

                    quality_metrics['column_metrics'][col] = col_metrics

            # Data integrity checks
            quality_metrics['data_integrity_checks'] = {
                'no_new_missing_values': final_missing <= original_missing,
                'row_count_preserved': len(original_df) == len(imputed_df),
                'column_count_preserved': len(original_df.columns) == len(imputed_df.columns),
                'index_preserved': original_df.index.equals(imputed_df.index)
            }

            return quality_metrics

        except Exception as e:
            self.logger.error(f"Error calculating imputation quality metrics: {str(e)}")
            return {'error': str(e)}