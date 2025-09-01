"""
Contextual Outlier Detection for Financial Transactions

This module provides domain-specific outlier detection methods tailored
for financial transaction data, including category-based, temporal,
frequency-based, and merchant-specific analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
import warnings
import logging
from scipy import stats
from scipy.spatial.distance import mahalanobis

from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from .advanced_outlier_detector import AdvancedOutlierDetector

logger = get_logger(__name__)


class ContextualOutlierDetector:
    """
    Domain-specific outlier detection for financial transactions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the contextual outlier detector

        Args:
            config: Configuration dictionary for contextual detection
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize base detector
        self.base_detector = AdvancedOutlierDetector(self.config.get('base_config', {}))

        # Contextual analysis storage
        self.category_profiles = {}
        self.temporal_profiles = {}
        self.merchant_profiles = {}
        self.frequency_profiles = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for contextual outlier detection"""
        return {
            'amount_outliers': {
                'method': 'iqr',  # 'iqr', 'zscore', 'robust'
                'threshold': 2.5,
                'min_samples_per_category': 10
            },
            'temporal_outliers': {
                'time_windows': ['hour', 'day', 'week', 'month'],
                'seasonal_decomposition': True,
                'holiday_sensitivity': True,
                'weekend_factor': 1.2
            },
            'frequency_outliers': {
                'window_days': 30,
                'burst_threshold': 3.0,
                'pattern_recognition': True
            },
            'merchant_outliers': {
                'similarity_threshold': 0.8,
                'min_transactions_per_merchant': 5,
                'category_context': True
            },
            'balance_outliers': {
                'volatility_threshold': 0.15,
                'trend_analysis': True,
                'shock_detection': True
            },
            'base_config': {
                'iqr': {'multiplier': 1.5},
                'zscore': {'threshold': 2.5},
                'lof': {'n_neighbors': 15}
            }
        }

    def detect_amount_outliers_by_category(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Detect amount outliers within each transaction category

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Dictionary with category-specific outlier detection results
        """
        try:
            df = pd.DataFrame(transactions)

            if 'category' not in df.columns or 'amount' not in df.columns:
                raise ValidationError("Transactions must have 'category' and 'amount' fields")

            results = {}
            min_samples = self.config['amount_outliers']['min_samples_per_category']

            # Group by category
            for category, group in df.groupby('category'):
                if len(group) < min_samples:
                    self.logger.debug(f"Skipping category {category}: insufficient samples ({len(group)})")
                    continue

                amounts = np.abs(group['amount'].values)
                method = self.config['amount_outliers']['method']

                # Detect outliers using specified method
                if method == 'iqr':
                    outlier_flags, outlier_scores = self.base_detector.detect_outliers_iqr(amounts)
                elif method == 'zscore':
                    outlier_flags, outlier_scores = self.base_detector.detect_outliers_zscore(amounts)
                elif method == 'robust':
                    outlier_flags, outlier_scores = self.base_detector.detect_outliers_iqr(amounts, method='robust')
                else:
                    outlier_flags, outlier_scores = self.base_detector.detect_outliers_iqr(amounts)

                # Store category profile
                self.category_profiles[category] = {
                    'mean_amount': np.mean(amounts),
                    'std_amount': np.std(amounts),
                    'median_amount': np.median(amounts),
                    'q25_amount': np.percentile(amounts, 25),
                    'q75_amount': np.percentile(amounts, 75),
                    'total_transactions': len(amounts)
                }

                # Get outlier transactions
                outlier_indices = np.where(outlier_flags)[0]
                outlier_transactions = group.iloc[outlier_indices].to_dict('records')

                results[category] = {
                    'outlier_count': np.sum(outlier_flags),
                    'outlier_percentage': np.sum(outlier_flags) / len(amounts) * 100,
                    'total_transactions': len(amounts),
                    'outlier_transactions': outlier_transactions,
                    'outlier_scores': outlier_scores[outlier_indices].tolist(),
                    'category_profile': self.category_profiles[category]
                }

            return {
                'results': results,
                'summary': self._generate_category_outlier_summary(results),
                'detection_method': method
            }

        except Exception as e:
            self.logger.error(f"Error in category-based amount outlier detection: {str(e)}")
            return {'error': str(e)}

    def detect_temporal_outliers(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Detect temporal outliers (unusual patterns by time/day/month)

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Dictionary with temporal outlier detection results
        """
        try:
            df = pd.DataFrame(transactions)

            if 'date' not in df.columns or 'amount' not in df.columns:
                raise ValidationError("Transactions must have 'date' and 'amount' fields")

            # Convert dates
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])

            if len(df) == 0:
                return {'error': 'No valid dates found in transactions'}

            results = {}

            # Analyze different time windows
            time_windows = self.config['temporal_outliers']['time_windows']

            for window in time_windows:
                window_results = self._analyze_temporal_window(df, window)
                if window_results:
                    results[window] = window_results

            # Seasonal analysis
            if self.config['temporal_outliers']['seasonal_decomposition']:
                seasonal_results = self._analyze_seasonal_patterns(df)
                results['seasonal'] = seasonal_results

            return {
                'results': results,
                'summary': self._generate_temporal_outlier_summary(results)
            }

        except Exception as e:
            self.logger.error(f"Error in temporal outlier detection: {str(e)}")
            return {'error': str(e)}

    def _analyze_temporal_window(self, df: pd.DataFrame, window: str) -> Dict[str, Any]:
        """Analyze outliers within a specific temporal window"""
        try:
            if window == 'hour':
                groupby_col = df['date'].dt.hour
                expected_periods = 24
            elif window == 'day':
                groupby_col = df['date'].dt.day
                expected_periods = 31
            elif window == 'week':
                groupby_col = df['date'].dt.isocalendar().week
                expected_periods = 52
            elif window == 'month':
                groupby_col = df['date'].dt.month
                expected_periods = 12
            else:
                return None

            # Group by time period and calculate statistics
            period_stats = df.groupby(groupby_col)['amount'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).fillna(0)

            # Detect unusual patterns
            counts = period_stats['count'].values
            means = period_stats['mean'].values

            # Outliers in transaction frequency
            freq_outlier_flags, freq_scores = self.base_detector.detect_outliers_iqr(counts)

            # Outliers in transaction amounts
            amount_outlier_flags, amount_scores = self.base_detector.detect_outliers_iqr(np.abs(means))

            # Identify periods with unusual activity
            unusual_periods = []
            for i, (period, row) in enumerate(period_stats.iterrows()):
                if freq_outlier_flags[i] or amount_outlier_flags[i]:
                    unusual_periods.append({
                        'period': period,
                        'transaction_count': row['count'],
                        'avg_amount': row['mean'],
                        'freq_score': freq_scores[i],
                        'amount_score': amount_scores[i],
                        'is_freq_outlier': freq_outlier_flags[i],
                        'is_amount_outlier': amount_outlier_flags[i]
                    })

            return {
                'period_stats': period_stats.to_dict(),
                'unusual_periods': unusual_periods,
                'freq_outlier_count': np.sum(freq_outlier_flags),
                'amount_outlier_count': np.sum(amount_outlier_flags),
                'expected_periods': expected_periods
            }

        except Exception as e:
            self.logger.warning(f"Error analyzing temporal window {window}: {str(e)}")
            return None

    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns and detect anomalies"""
        try:
            # Extract seasonal components
            df['month'] = df['date'].dt.month
            df['weekday'] = df['date'].dt.weekday
            df['hour'] = df['date'].dt.hour

            seasonal_analysis = {}

            # Monthly patterns
            monthly = df.groupby('month')['amount'].agg(['count', 'mean', 'sum']).fillna(0)
            seasonal_analysis['monthly'] = monthly.to_dict()

            # Weekly patterns
            weekly = df.groupby('weekday')['amount'].agg(['count', 'mean', 'sum']).fillna(0)
            seasonal_analysis['weekly'] = weekly.to_dict()

            # Hourly patterns (if available)
            if df['hour'].notna().any():
                hourly = df.groupby('hour')['amount'].agg(['count', 'mean', 'sum']).fillna(0)
                seasonal_analysis['hourly'] = hourly.to_dict()

            # Detect seasonal anomalies
            anomalies = []

            # Check for unusual monthly activity
            monthly_counts = monthly['count'].values
            outlier_flags, scores = self.base_detector.detect_outliers_iqr(monthly_counts)
            if np.any(outlier_flags):
                for i, is_outlier in enumerate(outlier_flags):
                    if is_outlier:
                        anomalies.append({
                            'type': 'monthly',
                            'period': i + 1,
                            'count': monthly_counts[i],
                            'score': scores[i]
                        })

            seasonal_analysis['anomalies'] = anomalies

            return seasonal_analysis

        except Exception as e:
            self.logger.warning(f"Error in seasonal pattern analysis: {str(e)}")
            return {'error': str(e)}

    def detect_frequency_outliers(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Detect frequency-based outliers (unusual transaction volumes)

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Dictionary with frequency outlier detection results
        """
        try:
            df = pd.DataFrame(transactions)

            if 'date' not in df.columns:
                raise ValidationError("Transactions must have 'date' field")

            # Convert dates
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')

            if len(df) == 0:
                return {'error': 'No valid dates found in transactions'}

            window_days = self.config['frequency_outliers']['window_days']
            burst_threshold = self.config['frequency_outliers']['burst_threshold']

            # Calculate rolling frequency
            df['date_only'] = df['date'].dt.date
            daily_counts = df.groupby('date_only').size()

            # Rolling window analysis
            rolling_freq = daily_counts.rolling(window=window_days, center=True).mean()
            rolling_std = daily_counts.rolling(window=window_days, center=True).std()

            # Detect bursts (unusually high frequency)
            z_scores = (daily_counts - rolling_freq) / (rolling_std + 1e-10)
            burst_flags = z_scores > burst_threshold

            # Detect gaps (unusually low frequency)
            gap_flags = z_scores < -burst_threshold

            # Pattern recognition
            pattern_anomalies = []
            if self.config['frequency_outliers']['pattern_recognition']:
                pattern_anomalies = self._detect_frequency_patterns(daily_counts)

            # Compile results
            burst_days = daily_counts[burst_flags].index.tolist()
            gap_days = daily_counts[gap_flags].index.tolist()

            results = {
                'daily_counts': daily_counts.to_dict(),
                'rolling_stats': {
                    'mean': rolling_freq.mean(),
                    'std': rolling_freq.std(),
                    'min': rolling_freq.min(),
                    'max': rolling_freq.max()
                },
                'bursts': {
                    'count': len(burst_days),
                    'days': burst_days,
                    'threshold': burst_threshold
                },
                'gaps': {
                    'count': len(gap_days),
                    'days': gap_days,
                    'threshold': -burst_threshold
                },
                'pattern_anomalies': pattern_anomalies
            }

            return results

        except Exception as e:
            self.logger.error(f"Error in frequency outlier detection: {str(e)}")
            return {'error': str(e)}

    def _detect_frequency_patterns(self, daily_counts: pd.Series) -> List[Dict]:
        """Detect unusual patterns in transaction frequency"""
        try:
            anomalies = []

            # Check for sudden spikes
            diff = daily_counts.diff()
            spike_threshold = daily_counts.std() * 2

            spikes = diff > spike_threshold
            if spikes.any():
                for date in daily_counts[spikes].index:
                    anomalies.append({
                        'type': 'sudden_spike',
                        'date': str(date),
                        'count': daily_counts[date],
                        'change': diff[date]
                    })

            # Check for unusual weekends/weekdays
            weekday_counts = daily_counts.groupby(daily_counts.index.to_series().dt.weekday).mean()
            weekend_factor = self.config['temporal_outliers']['weekend_factor']

            # Weekend analysis (assuming Saturday=5, Sunday=6)
            weekend_avg = weekday_counts.loc[[5, 6]].mean()
            weekday_avg = weekday_counts.loc[0:4].mean()

            if weekend_avg > weekday_avg * weekend_factor:
                anomalies.append({
                    'type': 'high_weekend_activity',
                    'weekend_avg': weekend_avg,
                    'weekday_avg': weekday_avg,
                    'ratio': weekend_avg / weekday_avg
                })

            return anomalies

        except Exception as e:
            self.logger.warning(f"Error in frequency pattern detection: {str(e)}")
            return []

    def detect_merchant_outliers(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Detect merchant-specific outliers

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Dictionary with merchant outlier detection results
        """
        try:
            df = pd.DataFrame(transactions)

            if 'description' not in df.columns or 'amount' not in df.columns:
                raise ValidationError("Transactions must have 'description' and 'amount' fields")

            min_transactions = self.config['merchant_outliers']['min_transactions_per_merchant']

            # Extract merchant names (simplified)
            df['merchant'] = df['description'].apply(self._extract_merchant_name)

            # Group by merchant
            merchant_groups = df.groupby('merchant')

            results = {}
            merchant_profiles = {}

            for merchant, group in merchant_groups:
                if len(group) < min_transactions:
                    continue

                amounts = np.abs(group['amount'].values)

                # Calculate merchant profile
                profile = {
                    'transaction_count': len(amounts),
                    'total_amount': np.sum(amounts),
                    'avg_amount': np.mean(amounts),
                    'std_amount': np.std(amounts),
                    'min_amount': np.min(amounts),
                    'max_amount': np.max(amounts)
                }

                merchant_profiles[merchant] = profile

                # Detect outliers within merchant transactions
                outlier_flags, outlier_scores = self.base_detector.detect_outliers_iqr(amounts)

                # Category context analysis
                category_context = None
                if self.config['merchant_outliers']['category_context'] and 'category' in group.columns:
                    category_context = self._analyze_merchant_category_context(group)

                outlier_transactions = group.iloc[np.where(outlier_flags)[0]].to_dict('records')

                results[merchant] = {
                    'profile': profile,
                    'outlier_count': np.sum(outlier_flags),
                    'outlier_percentage': np.sum(outlier_flags) / len(amounts) * 100,
                    'outlier_transactions': outlier_transactions,
                    'category_context': category_context
                }

            # Cross-merchant analysis
            cross_merchant_anomalies = self._detect_cross_merchant_anomalies(merchant_profiles)

            return {
                'merchant_results': results,
                'cross_merchant_anomalies': cross_merchant_anomalies,
                'summary': self._generate_merchant_outlier_summary(results)
            }

        except Exception as e:
            self.logger.error(f"Error in merchant outlier detection: {str(e)}")
            return {'error': str(e)}

    def _extract_merchant_name(self, description: str) -> str:
        """Extract merchant name from transaction description"""
        if not description:
            return "unknown"

        # Simple merchant extraction (can be enhanced)
        description = str(description).upper()

        # Common patterns
        if 'MERCADO' in description or 'SUPERMERCADO' in description:
            return 'mercado'
        elif 'FARMACIA' in description:
            return 'farmacia'
        elif 'POSTO' in description or 'COMBUSTIVEL' in description:
            return 'posto'
        elif 'RESTAURANTE' in description or 'IFOOD' in description:
            return 'restaurante'
        else:
            # Return first significant word
            words = [w for w in description.split() if len(w) > 2]
            return words[0] if words else "unknown"

    def _analyze_merchant_category_context(self, merchant_group: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how merchant transactions fit within their category context"""
        try:
            if 'category' not in merchant_group.columns:
                return None

            category = merchant_group['category'].iloc[0]
            amounts = np.abs(merchant_group['amount'].values)

            # Compare with category profile if available
            if category in self.category_profiles:
                cat_profile = self.category_profiles[category]

                # Check if merchant amounts are unusual for the category
                merchant_avg = np.mean(amounts)
                category_avg = cat_profile['mean_amount']

                deviation = abs(merchant_avg - category_avg) / (category_avg + 1e-10)

                return {
                    'category': category,
                    'merchant_avg': merchant_avg,
                    'category_avg': category_avg,
                    'deviation_ratio': deviation,
                    'is_unusual': deviation > 0.5  # More than 50% deviation
                }

            return None

        except Exception as e:
            self.logger.warning(f"Error in merchant category context analysis: {str(e)}")
            return None

    def _detect_cross_merchant_anomalies(self, merchant_profiles: Dict) -> List[Dict]:
        """Detect anomalies across merchants"""
        try:
            anomalies = []

            if len(merchant_profiles) < 3:
                return anomalies

            # Extract features for cross-merchant analysis
            features = []
            merchant_names = []

            for merchant, profile in merchant_profiles.items():
                features.append([
                    profile['avg_amount'],
                    profile['std_amount'],
                    profile['transaction_count']
                ])
                merchant_names.append(merchant)

            features = np.array(features)

            # Detect multivariate outliers
            outlier_flags, outlier_scores = self.base_detector.detect_outliers_mahalanobis(features)

            # Identify anomalous merchants
            for i, (merchant, is_outlier) in enumerate(zip(merchant_names, outlier_flags)):
                if is_outlier:
                    anomalies.append({
                        'merchant': merchant,
                        'profile': merchant_profiles[merchant],
                        'outlier_score': outlier_scores[i],
                        'reason': 'multivariate_outlier'
                    })

            return anomalies

        except Exception as e:
            self.logger.warning(f"Error in cross-merchant anomaly detection: {str(e)}")
            return []

    def detect_balance_outliers(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Detect account balance outliers and unusual patterns

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Dictionary with balance outlier detection results
        """
        try:
            df = pd.DataFrame(transactions)

            if 'date' not in df.columns or 'amount' not in df.columns:
                raise ValidationError("Transactions must have 'date' and 'amount' fields")

            # Sort by date
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')

            if len(df) == 0:
                return {'error': 'No valid transactions found'}

            # Calculate running balance
            df['balance_change'] = df['amount']
            df['running_balance'] = df['balance_change'].cumsum()

            balance_series = df['running_balance'].values

            results = {}

            # Volatility analysis
            volatility_threshold = self.config['balance_outliers']['volatility_threshold']
            balance_changes = np.diff(balance_series)
            volatility = np.std(balance_changes) / (np.mean(np.abs(balance_series)) + 1e-10)

            results['volatility'] = {
                'value': volatility,
                'threshold': volatility_threshold,
                'is_high': volatility > volatility_threshold
            }

            # Trend analysis
            if self.config['balance_outliers']['trend_analysis']:
                trend_results = self._analyze_balance_trend(df)
                results['trend'] = trend_results

            # Shock detection
            if self.config['balance_outliers']['shock_detection']:
                shock_results = self._detect_balance_shocks(df)
                results['shocks'] = shock_results

            # Balance outlier detection
            outlier_flags, outlier_scores = self.base_detector.detect_outliers_iqr(balance_series)

            balance_outliers = df.iloc[np.where(outlier_flags)[0]][['date', 'amount', 'running_balance']].to_dict('records')

            results['balance_outliers'] = {
                'count': np.sum(outlier_flags),
                'percentage': np.sum(outlier_flags) / len(balance_series) * 100,
                'outliers': balance_outliers,
                'scores': outlier_scores[np.where(outlier_flags)[0]].tolist()
            }

            return results

        except Exception as e:
            self.logger.error(f"Error in balance outlier detection: {str(e)}")
            return {'error': str(e)}

    def _analyze_balance_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze balance trend for anomalies"""
        try:
            balance = df['running_balance'].values
            dates = df['date'].values

            # Simple linear trend
            x = np.arange(len(balance))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, balance)

            # Detect trend changes
            balance_changes = np.diff(balance)
            change_points = []

            # Simple change point detection
            window_size = min(10, len(balance_changes) // 3)
            if window_size > 3:
                rolling_mean = pd.Series(balance_changes).rolling(window=window_size).mean()
                rolling_std = pd.Series(balance_changes).rolling(window=window_size).std()

                change_flags = np.abs(balance_changes - rolling_mean) > 2 * rolling_std
                change_indices = np.where(change_flags)[0]

                for idx in change_indices:
                    if idx < len(df) - 1:
                        change_points.append({
                            'date': str(df.iloc[idx + 1]['date']),
                            'change_amount': balance_changes[idx],
                            'balance_before': balance[idx],
                            'balance_after': balance[idx + 1]
                        })

            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'change_points': change_points
            }

        except Exception as e:
            self.logger.warning(f"Error in balance trend analysis: {str(e)}")
            return {'error': str(e)}

    def _detect_balance_shocks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect sudden balance shocks"""
        try:
            balance_changes = df['balance_change'].values

            # Detect large sudden changes
            median_change = np.median(np.abs(balance_changes))
            shock_threshold = median_change * 5  # 5x median change

            shock_flags = np.abs(balance_changes) > shock_threshold
            shock_indices = np.where(shock_flags)[0]

            shocks = []
            for idx in shock_indices:
                shocks.append({
                    'date': str(df.iloc[idx]['date']),
                    'amount': df.iloc[idx]['amount'],
                    'balance_change': balance_changes[idx],
                    'shock_magnitude': abs(balance_changes[idx]) / (median_change + 1e-10)
                })

            return {
                'shock_count': len(shocks),
                'shock_threshold': shock_threshold,
                'median_change': median_change,
                'shocks': shocks
            }

        except Exception as e:
            self.logger.warning(f"Error in balance shock detection: {str(e)}")
            return {'error': str(e)}

    def _generate_category_outlier_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary for category outlier detection"""
        if not results:
            return {}

        total_outliers = sum(cat['outlier_count'] for cat in results.values())
        total_transactions = sum(cat['total_transactions'] for cat in results.values())

        return {
            'total_categories': len(results),
            'total_outliers': total_outliers,
            'overall_outlier_percentage': total_outliers / total_transactions * 100 if total_transactions > 0 else 0,
            'categories_with_outliers': len([cat for cat in results.values() if cat['outlier_count'] > 0])
        }

    def _generate_temporal_outlier_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary for temporal outlier detection"""
        if not results:
            return {}

        summary = {}
        for window, window_results in results.items():
            if isinstance(window_results, dict) and 'unusual_periods' in window_results:
                summary[window] = {
                    'unusual_periods_count': len(window_results['unusual_periods']),
                    'freq_outliers': window_results.get('freq_outlier_count', 0),
                    'amount_outliers': window_results.get('amount_outlier_count', 0)
                }

        return summary

    def _generate_merchant_outlier_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary for merchant outlier detection"""
        if not results:
            return {}

        total_outliers = sum(merchant['outlier_count'] for merchant in results.values())
        total_merchants = len(results)

        return {
            'total_merchants': total_merchants,
            'merchants_with_outliers': len([m for m in results.values() if m['outlier_count'] > 0]),
            'total_outliers': total_outliers,
            'avg_outliers_per_merchant': total_outliers / total_merchants if total_merchants > 0 else 0
        }