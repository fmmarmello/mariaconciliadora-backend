import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple, Union
import holidays
from collections import defaultdict, Counter
import calendar
import math

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError

logger = get_logger(__name__)


class TemporalFeatureEnhancer:
    """
    Advanced temporal feature enhancer for time-series and temporal data
    Provides comprehensive temporal feature extraction with business awareness
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TemporalFeatureEnhancer

        Args:
            config: Configuration dictionary for temporal feature enhancement
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize temporal components
        self._initialize_components()

        # Temporal patterns and knowledge
        self.business_patterns = self._get_business_patterns()
        self.seasonal_patterns = self._get_seasonal_patterns()

        # Feature storage and caching
        self.feature_cache = {}
        self.temporal_stats = defaultdict(int)
        self.pattern_history = []

        self.logger.info("TemporalFeatureEnhancer initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for temporal feature enhancement"""
        return {
            'temporal_features': {
                'basic_features': True,
                'cyclical_encoding': True,
                'business_days': True,
                'holidays': True,
                'seasonal_patterns': True,
                'temporal_consistency': True,
                'time_series_features': True,
                'pattern_recognition': True
            },
            'business_calendar': {
                'country': 'BR',
                'state': None,  # Brazilian state for regional holidays
                'include_optional_holidays': True,
                'business_hours': {
                    'start': 9,
                    'end': 18,
                    'timezone': 'America/Sao_Paulo'
                }
            },
            'seasonal_analysis': {
                'method': 'trigonometric',  # 'trigonometric', 'categorical', 'fourier'
                'harmonics': 3,  # Number of harmonics for Fourier analysis
                'include_trends': True,
                'detect_outliers': True
            },
            'pattern_recognition': {
                'clustering_enabled': True,
                'n_clusters': 5,
                'similarity_threshold': 0.8,
                'pattern_memory_days': 90
            },
            'time_series': {
                'rolling_windows': [7, 14, 30, 90],  # Days
                'lag_features': [1, 2, 3, 7, 14, 30],
                'difference_features': True,
                'momentum_features': True
            },
            'quality_assurance': {
                'validate_consistency': True,
                'detect_anomalies': True,
                'temporal_outlier_threshold': 3.0,  # Standard deviations
                'missing_data_handling': 'interpolate'
            },
            'performance': {
                'cache_enabled': True,
                'parallel_processing': True,
                'batch_size': 1000
            }
        }

    def _initialize_components(self):
        """Initialize temporal processing components"""
        try:
            # Holiday calendar for Brazil
            self.holiday_calendar = holidays.CountryHoliday(
                self.config['business_calendar']['country'],
                state=self.config['business_calendar']['state']
            )

            # Add optional holidays if configured
            if self.config['business_calendar']['include_optional_holidays']:
                self._add_optional_holidays()

            # Scaler for normalization
            self.scaler = StandardScaler()

            # Pattern recognition model
            if self.config['pattern_recognition']['clustering_enabled']:
                self.pattern_clusterer = KMeans(
                    n_clusters=self.config['pattern_recognition']['n_clusters'],
                    random_state=42,
                    n_init=10
                )

            self.logger.info("Temporal components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing temporal components: {str(e)}")
            raise ValidationError(f"Failed to initialize TemporalFeatureEnhancer: {str(e)}")

    def _add_optional_holidays(self):
        """Add optional holidays to the calendar"""
        try:
            # Brazilian optional holidays and special dates
            optional_holidays = {
                'Carnaval': self._get_carnival_dates,
                'Corpus Christi': self._get_corpus_christi_dates,
                'Black Friday': self._get_black_friday_dates,
                'Christmas Eve': self._get_christmas_eve_dates,
                'New Year Eve': self._get_new_year_eve_dates
            }

            current_year = datetime.now().year
            for holiday_name, date_func in optional_holidays.items():
                try:
                    dates = date_func(current_year)
                    for holiday_date in dates:
                        self.holiday_calendar[holiday_date] = holiday_name
                except Exception as e:
                    self.logger.warning(f"Error adding {holiday_name}: {str(e)}")

        except Exception as e:
            self.logger.warning(f"Error adding optional holidays: {str(e)}")

    def _get_carnival_dates(self, year: int) -> List[date]:
        """Get Carnival dates for a given year"""
        # Carnival is 47 days before Easter
        easter = self._get_easter_date(year)
        carnival = easter - timedelta(days=47)
        return [carnival]

    def _get_corpus_christi_dates(self, year: int) -> List[date]:
        """Get Corpus Christi dates for a given year"""
        # Corpus Christi is 60 days after Easter
        easter = self._get_easter_date(year)
        corpus_christi = easter + timedelta(days=60)
        return [corpus_christi]

    def _get_black_friday_dates(self, year: int) -> List[date]:
        """Get Black Friday dates for a given year"""
        # Black Friday is the Friday after Thanksgiving (US), but we'll use a simple approximation
        nov_1 = date(year, 11, 1)
        # Find the 4th Thursday in November, then Black Friday is the next day
        thanksgiving = self._get_nth_weekday_of_month(year, 11, 3, 4)  # 4th Thursday
        black_friday = thanksgiving + timedelta(days=1)
        return [black_friday]

    def _get_christmas_eve_dates(self, year: int) -> List[date]:
        """Get Christmas Eve dates"""
        return [date(year, 12, 24)]

    def _get_new_year_eve_dates(self, year: int) -> List[date]:
        """Get New Year Eve dates"""
        return [date(year, 12, 31)]

    def _get_easter_date(self, year: int) -> date:
        """Calculate Easter date for a given year (simplified algorithm)"""
        # This is a simplified Easter calculation
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)

    def _get_nth_weekday_of_month(self, year: int, month: int, weekday: int, n: int) -> date:
        """Get the nth weekday of a month"""
        first_day = date(year, month, 1)
        first_weekday = first_day.weekday()

        # Calculate days to first occurrence of the desired weekday
        days_to_first = (weekday - first_weekday) % 7
        first_occurrence = first_day + timedelta(days=days_to_first)

        # Calculate the nth occurrence
        target_date = first_occurrence + timedelta(days=7 * (n - 1))
        return target_date

    def _get_business_patterns(self) -> Dict[str, Any]:
        """Get business-specific temporal patterns"""
        return {
            'payroll_days': [1, 5, 10, 15, 20, 25, 30],  # Common payroll dates
            'tax_deadlines': {
                'monthly': [20],  # Approximate monthly tax deadlines
                'quarterly': [31, 30, 31],  # End of quarters
                'yearly': [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Monthly deadlines
            },
            'business_cycles': {
                'short': 7,  # Weekly
                'medium': 30,  # Monthly
                'long': 365  # Yearly
            },
            'peak_hours': [9, 10, 11, 14, 15, 16, 17],  # Business hours
            'low_activity_periods': [1, 2, 3, 4, 5, 6, 22, 23, 24]  # Early morning, late night
        }

    def _get_seasonal_patterns(self) -> Dict[str, Any]:
        """Get seasonal patterns for Brazil"""
        return {
            'summer': {'months': [12, 1, 2], 'peak': 'high_season'},
            'autumn': {'months': [3, 4, 5], 'peak': 'shoulder_season'},
            'winter': {'months': [6, 7, 8], 'peak': 'low_season'},
            'spring': {'months': [9, 10, 11], 'peak': 'shoulder_season'},
            'holiday_peaks': {
                'christmas_new_year': [12, 1],
                'carnival': [2, 3],
                'easter': [3, 4]
            }
        }

    def extract_temporal_features(self, dates: List[Union[str, datetime, pd.Timestamp]],
                                context_data: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Extract comprehensive temporal features from dates

        Args:
            dates: List of date values
            context_data: Additional context data for enhanced features

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        try:
            if not dates:
                return np.array([]), []

            self.logger.info(f"Extracting temporal features for {len(dates)} dates")

            # Convert to pandas datetime
            date_series = pd.to_datetime(dates, errors='coerce')

            # Remove invalid dates
            valid_mask = ~date_series.isna()
            date_series = date_series[valid_mask]

            if len(date_series) == 0:
                return np.array([]), []

            features = []
            feature_names = []

            # Basic temporal features
            if self.config['temporal_features']['basic_features']:
                basic_features, basic_names = self._extract_basic_features(date_series)
                features.append(basic_features)
                feature_names.extend(basic_names)

            # Cyclical encoding
            if self.config['temporal_features']['cyclical_encoding']:
                cyclical_features, cyclical_names = self._extract_cyclical_features(date_series)
                features.append(cyclical_features)
                feature_names.extend(cyclical_names)

            # Business day features
            if self.config['temporal_features']['business_days']:
                business_features, business_names = self._extract_business_features(date_series)
                features.append(business_features)
                feature_names.extend(business_names)

            # Holiday features
            if self.config['temporal_features']['holidays']:
                holiday_features, holiday_names = self._extract_holiday_features(date_series)
                features.append(holiday_features)
                feature_names.extend(holiday_names)

            # Seasonal features
            if self.config['temporal_features']['seasonal_patterns']:
                seasonal_features, seasonal_names = self._extract_seasonal_features(date_series)
                features.append(seasonal_features)
                feature_names.extend(seasonal_names)

            # Time series features
            if self.config['temporal_features']['time_series_features'] and context_data:
                ts_features, ts_names = self._extract_time_series_features(date_series, context_data)
                features.append(ts_features)
                feature_names.extend(ts_names)

            # Pattern recognition features
            if self.config['temporal_features']['pattern_recognition']:
                pattern_features, pattern_names = self._extract_pattern_features(date_series)
                features.append(pattern_features)
                feature_names.extend(pattern_names)

            # Combine all features
            if features:
                combined_features = np.concatenate(features, axis=1)
            else:
                combined_features = np.array([])

            # Handle NaN values
            combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Update statistics
            self._update_temporal_stats(feature_names)

            self.logger.info(f"Extracted {len(feature_names)} temporal features")
            return combined_features, feature_names

        except Exception as e:
            self.logger.error(f"Error extracting temporal features: {str(e)}")
            return np.array([]), []

    def _extract_basic_features(self, date_series: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Extract basic temporal features"""
        try:
            features = []

            # Year, month, day
            features.append(date_series.year.values.reshape(-1, 1))
            features.append(date_series.month.values.reshape(-1, 1))
            features.append(date_series.day.values.reshape(-1, 1))

            # Hour, minute (if available)
            if date_series.dt.hour.notna().any():
                features.append(date_series.dt.hour.values.reshape(-1, 1))
                features.append(date_series.dt.minute.values.reshape(-1, 1))

            # Day of week
            features.append(date_series.dt.weekday.values.reshape(-1, 1))

            # Day of year
            features.append(date_series.dt.dayofyear.values.reshape(-1, 1))

            # Week of year
            features.append(date_series.dt.isocalendar().week.values.reshape(-1, 1))

            # Quarter
            features.append(date_series.dt.quarter.values.reshape(-1, 1))

            # Is weekend
            is_weekend = (date_series.dt.weekday >= 5).astype(int).values.reshape(-1, 1)
            features.append(is_weekend)

            # Is month start/end
            is_month_start = date_series.dt.is_month_start.astype(int).values.reshape(-1, 1)
            is_month_end = date_series.dt.is_month_end.astype(int).values.reshape(-1, 1)
            features.append(is_month_start)
            features.append(is_month_end)

            # Is quarter start/end
            is_quarter_start = date_series.dt.is_quarter_start.astype(int).values.reshape(-1, 1)
            is_quarter_end = date_series.dt.is_quarter_end.astype(int).values.reshape(-1, 1)
            features.append(is_quarter_start)
            features.append(is_quarter_end)

            feature_names = [
                'year', 'month', 'day', 'hour', 'minute', 'weekday', 'day_of_year',
                'week_of_year', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end',
                'is_quarter_start', 'is_quarter_end'
            ]

            # Filter out features that don't exist (e.g., if no time component)
            existing_features = []
            existing_names = []
            for i, (feature, name) in enumerate(zip(features, feature_names)):
                if feature.shape[1] > 0 and not np.all(np.isnan(feature)):
                    existing_features.append(feature)
                    existing_names.append(name)

            if existing_features:
                combined_features = np.concatenate(existing_features, axis=1)
                return combined_features, existing_names
            else:
                return np.array([]), []

        except Exception as e:
            self.logger.error(f"Error extracting basic features: {str(e)}")
            return np.array([]), []

    def _extract_cyclical_features(self, date_series: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Extract cyclical temporal features"""
        try:
            features = []

            # Cyclical encoding for periodic features
            # Month (12 months)
            month_sin = np.sin(2 * np.pi * date_series.dt.month / 12).values.reshape(-1, 1)
            month_cos = np.cos(2 * np.pi * date_series.dt.month / 12).values.reshape(-1, 1)
            features.extend([month_sin, month_cos])

            # Day of month (28-31 days, use 30 as approximation)
            day_sin = np.sin(2 * np.pi * date_series.dt.day / 30).values.reshape(-1, 1)
            day_cos = np.cos(2 * np.pi * date_series.dt.day / 30).values.reshape(-1, 1)
            features.extend([day_sin, day_cos])

            # Day of week (7 days)
            weekday_sin = np.sin(2 * np.pi * date_series.dt.weekday / 7).values.reshape(-1, 1)
            weekday_cos = np.cos(2 * np.pi * date_series.dt.weekday / 7).values.reshape(-1, 1)
            features.extend([weekday_sin, weekday_cos])

            # Hour (24 hours, if available)
            if date_series.dt.hour.notna().any():
                hour_sin = np.sin(2 * np.pi * date_series.dt.hour / 24).values.reshape(-1, 1)
                hour_cos = np.cos(2 * np.pi * date_series.dt.hour / 24).values.reshape(-1, 1)
                features.extend([hour_sin, hour_cos])

            feature_names = ['month_sin', 'month_cos', 'day_sin', 'day_cos',
                           'weekday_sin', 'weekday_cos']

            if date_series.dt.hour.notna().any():
                feature_names.extend(['hour_sin', 'hour_cos'])

            combined_features = np.concatenate(features, axis=1)
            return combined_features, feature_names

        except Exception as e:
            self.logger.error(f"Error extracting cyclical features: {str(e)}")
            return np.array([]), []

    def _extract_business_features(self, date_series: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Extract business-specific temporal features"""
        try:
            features = []

            # Is business day
            is_business_day = (date_series.dt.weekday < 5).astype(int).values.reshape(-1, 1)
            features.append(is_business_day)

            # Business day of week (1-5 for Mon-Fri)
            business_day = date_series.dt.weekday.clip(upper=4) + 1
            business_day = business_day.values.reshape(-1, 1)
            features.append(business_day)

            # Is business hour (if time component exists)
            if date_series.dt.hour.notna().any():
                business_hours = self.config['business_calendar']['business_hours']
                is_business_hour = (
                    (date_series.dt.hour >= business_hours['start']) &
                    (date_series.dt.hour < business_hours['end'])
                ).astype(int).values.reshape(-1, 1)
                features.append(is_business_hour)

            # Days since last business day
            business_days = date_series[date_series.dt.weekday < 5]
            days_since_business = []
            for date_val in date_series:
                last_business = business_days[business_days <= date_val].max()
                if pd.isna(last_business):
                    days_since = 0
                else:
                    days_since = (date_val - last_business).days
                days_since_business.append(days_since)

            features.append(np.array(days_since_business).reshape(-1, 1))

            # Is payroll day
            is_payroll_day = date_series.dt.day.isin(
                self.business_patterns['payroll_days']
            ).astype(int).values.reshape(-1, 1)
            features.append(is_payroll_day)

            feature_names = ['is_business_day', 'business_day_of_week', 'days_since_business_day', 'is_payroll_day']

            if date_series.dt.hour.notna().any():
                feature_names.insert(2, 'is_business_hour')

            combined_features = np.concatenate(features, axis=1)
            return combined_features, feature_names

        except Exception as e:
            self.logger.error(f"Error extracting business features: {str(e)}")
            return np.array([]), []

    def _extract_holiday_features(self, date_series: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Extract holiday-related temporal features"""
        try:
            features = []

            # Is holiday
            is_holiday = np.array([date.date() in self.holiday_calendar for date in date_series]).reshape(-1, 1)
            features.append(is_holiday)

            # Days to next holiday
            days_to_holiday = []
            for date_val in date_series:
                days = 0
                check_date = date_val.date()
                while days < 30:  # Look ahead 30 days
                    check_date += timedelta(days=1)
                    if check_date in self.holiday_calendar:
                        break
                    days += 1
                days_to_holiday.append(min(days, 30))

            features.append(np.array(days_to_holiday).reshape(-1, 1))

            # Days since last holiday
            days_since_holiday = []
            for date_val in date_series:
                days = 0
                check_date = date_val.date()
                while days < 30:  # Look back 30 days
                    check_date -= timedelta(days=1)
                    if check_date in self.holiday_calendar:
                        break
                    days += 1
                days_since_holiday.append(min(days, 30))

            features.append(np.array(days_since_holiday).reshape(-1, 1))

            # Is long weekend (holiday on Friday or Monday)
            is_long_weekend = []
            for date_val in date_series:
                date_only = date_val.date()
                is_friday_holiday = (date_val.weekday() == 4 and date_only in self.holiday_calendar)
                is_monday_holiday = (date_val.weekday() == 0 and date_only in self.holiday_calendar)

                # Check if adjacent days are also holidays or weekends
                prev_day = date_only - timedelta(days=1)
                next_day = date_only + timedelta(days=1)

                has_adjacent_off = (
                    (prev_day in self.holiday_calendar or prev_day.weekday() >= 5) or
                    (next_day in self.holiday_calendar or next_day.weekday() >= 5)
                )

                is_long_weekend.append(int((is_friday_holiday or is_monday_holiday) and has_adjacent_off))

            features.append(np.array(is_long_weekend).reshape(-1, 1))

            feature_names = ['is_holiday', 'days_to_next_holiday', 'days_since_last_holiday', 'is_long_weekend']

            combined_features = np.concatenate(features, axis=1)
            return combined_features, feature_names

        except Exception as e:
            self.logger.error(f"Error extracting holiday features: {str(e)}")
            return np.array([]), []

    def _extract_seasonal_features(self, date_series: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Extract seasonal temporal features"""
        try:
            features = []

            # Season classification
            seasons = []
            for date_val in date_series:
                month = date_val.month
                if month in [12, 1, 2]:
                    seasons.append(0)  # Summer
                elif month in [3, 4, 5]:
                    seasons.append(1)  # Autumn
                elif month in [6, 7, 8]:
                    seasons.append(2)  # Winter
                else:
                    seasons.append(3)  # Spring

            season_features = np.array(seasons).reshape(-1, 1)
            features.append(season_features)

            # One-hot encoding for seasons
            season_dummies = pd.get_dummies(seasons, prefix='season')
            features.append(season_dummies.values)

            # Holiday season indicators
            holiday_season_indicators = []
            for date_val in date_series:
                month = date_val.month
                indicators = [
                    int(month in [12, 1]),  # Christmas/New Year season
                    int(month in [2, 3]),    # Carnival season
                    int(month in [3, 4]),    # Easter season
                    int(month in [6, 7]),    # School holidays
                    int(month in [12, 1, 7]) # Peak vacation season
                ]
                holiday_season_indicators.append(indicators)

            features.append(np.array(holiday_season_indicators))

            # Seasonal strength (based on historical patterns)
            seasonal_strength = []
            for date_val in date_series:
                month = date_val.month
                # Simplified seasonal strength based on Brazilian patterns
                if month in [12, 1, 7]:  # Peak season
                    strength = 1.0
                elif month in [2, 3, 11]:  # Shoulder season
                    strength = 0.7
                elif month in [6, 8]:  # Low season
                    strength = 0.3
                else:  # Regular season
                    strength = 0.5
                seasonal_strength.append(strength)

            features.append(np.array(seasonal_strength).reshape(-1, 1))

            feature_names = ['season', 'season_summer', 'season_autumn', 'season_winter', 'season_spring',
                           'holiday_season_christmas', 'holiday_season_carnival', 'holiday_season_easter',
                           'holiday_season_school', 'holiday_season_peak', 'seasonal_strength']

            combined_features = np.concatenate(features, axis=1)
            return combined_features, feature_names

        except Exception as e:
            self.logger.error(f"Error extracting seasonal features: {str(e)}")
            return np.array([]), []

    def _extract_time_series_features(self, date_series: pd.Series,
                                    context_data: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Extract time series specific features"""
        try:
            features = []

            # Sort dates for time series analysis
            sorted_dates = date_series.sort_values()
            sorted_indices = sorted_dates.index

            # Rolling statistics
            for window in self.config['time_series']['rolling_windows']:
                # Rolling mean (placeholder - would need actual values)
                rolling_mean = np.random.rand(len(date_series)).reshape(-1, 1)  # Placeholder
                features.append(rolling_mean)

                # Rolling std
                rolling_std = np.random.rand(len(date_series)).reshape(-1, 1)  # Placeholder
                features.append(rolling_std)

            # Lag features
            for lag in self.config['time_series']['lag_features']:
                lag_feature = np.random.rand(len(date_series)).reshape(-1, 1)  # Placeholder
                features.append(lag_feature)

            # Difference features
            if self.config['time_series']['difference_features']:
                diff_feature = np.random.rand(len(date_series)).reshape(-1, 1)  # Placeholder
                features.append(diff_feature)

            # Momentum features
            if self.config['time_series']['momentum_features']:
                momentum_feature = np.random.rand(len(date_series)).reshape(-1, 1)  # Placeholder
                features.append(momentum_feature)

            # Generate feature names
            feature_names = []
            for window in self.config['time_series']['rolling_windows']:
                feature_names.extend([f'rolling_mean_{window}d', f'rolling_std_{window}d'])

            for lag in self.config['time_series']['lag_features']:
                feature_names.append(f'lag_{lag}d')

            if self.config['time_series']['difference_features']:
                feature_names.append('diff_1d')

            if self.config['time_series']['momentum_features']:
                feature_names.append('momentum_7d')

            combined_features = np.concatenate(features, axis=1)
            return combined_features, feature_names

        except Exception as e:
            self.logger.error(f"Error extracting time series features: {str(e)}")
            return np.array([]), []

    def _extract_pattern_features(self, date_series: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Extract pattern recognition features"""
        try:
            features = []

            # Time of day patterns
            if date_series.dt.hour.notna().any():
                hour_patterns = self._analyze_hourly_patterns(date_series)
                features.append(hour_patterns)

            # Weekly patterns
            weekly_patterns = self._analyze_weekly_patterns(date_series)
            features.append(weekly_patterns)

            # Monthly patterns
            monthly_patterns = self._analyze_monthly_patterns(date_series)
            features.append(monthly_patterns)

            # Clustering-based patterns
            if hasattr(self, 'pattern_clusterer'):
                cluster_features = self._extract_cluster_patterns(date_series)
                features.append(cluster_features)

            feature_names = ['hour_pattern', 'weekly_pattern', 'monthly_pattern', 'cluster_pattern']

            combined_features = np.concatenate(features, axis=1)
            return combined_features, feature_names

        except Exception as e:
            self.logger.error(f"Error extracting pattern features: {str(e)}")
            return np.array([]), []

    def _analyze_hourly_patterns(self, date_series: pd.Series) -> np.ndarray:
        """Analyze hourly patterns in the data"""
        try:
            hours = date_series.dt.hour.dropna()
            if len(hours) == 0:
                return np.zeros((len(date_series), 1))

            # Simple pattern: distance from peak business hours
            peak_hour = 14  # 2 PM
            hour_patterns = np.abs(hours - peak_hour).values.reshape(-1, 1)

            # Normalize to 0-1 scale
            max_diff = 12  # Max difference from peak
            hour_patterns = hour_patterns / max_diff

            return hour_patterns

        except Exception:
            return np.zeros((len(date_series), 1))

    def _analyze_weekly_patterns(self, date_series: pd.Series) -> np.ndarray:
        """Analyze weekly patterns in the data"""
        try:
            weekdays = date_series.dt.weekday.values.reshape(-1, 1)

            # Pattern: business week vs weekend
            weekly_patterns = (weekdays < 5).astype(int).reshape(-1, 1)

            return weekly_patterns

        except Exception:
            return np.zeros((len(date_series), 1))

    def _analyze_monthly_patterns(self, date_series: pd.Series) -> np.ndarray:
        """Analyze monthly patterns in the data"""
        try:
            days = date_series.dt.day.values.reshape(-1, 1)

            # Pattern: month timing (early, mid, late month)
            monthly_patterns = np.digitize(days.flatten(), [10, 20]).reshape(-1, 1)

            return monthly_patterns

        except Exception:
            return np.zeros((len(date_series), 1))

    def _extract_cluster_patterns(self, date_series: pd.Series) -> np.ndarray:
        """Extract clustering-based pattern features"""
        try:
            # Create simple time-based features for clustering
            time_features = np.column_stack([
                date_series.dt.hour.fillna(12).values,
                date_series.dt.weekday.values,
                date_series.dt.month.values
            ])

            # Apply clustering
            cluster_labels = self.pattern_clusterer.fit_predict(time_features)

            # One-hot encode cluster labels
            cluster_dummies = pd.get_dummies(cluster_labels, prefix='temporal_cluster')
            cluster_features = cluster_dummies.values

            return cluster_features

        except Exception as e:
            self.logger.error(f"Error in cluster pattern extraction: {str(e)}")
            return np.zeros((len(date_series), self.config['pattern_recognition']['n_clusters']))

    def validate_temporal_consistency(self, dates: List[Union[str, datetime, pd.Timestamp]]) -> Dict[str, Any]:
        """
        Validate temporal consistency of date data

        Args:
            dates: List of dates to validate

        Returns:
            Dictionary with validation results
        """
        try:
            date_series = pd.to_datetime(dates, errors='coerce')

            validation_results = {
                'total_dates': len(dates),
                'valid_dates': date_series.notna().sum(),
                'invalid_dates': date_series.isna().sum(),
                'date_range': {
                    'start': date_series.min().isoformat() if date_series.notna().any() else None,
                    'end': date_series.max().isoformat() if date_series.notna().any() else None
                },
                'temporal_gaps': self._detect_temporal_gaps(date_series),
                'consistency_score': self._calculate_consistency_score(date_series)
            }

            return validation_results

        except Exception as e:
            self.logger.error(f"Error validating temporal consistency: {str(e)}")
            return {'error': str(e)}

    def _detect_temporal_gaps(self, date_series: pd.Series) -> Dict[str, Any]:
        """Detect gaps in temporal data"""
        try:
            valid_dates = date_series.dropna().sort_values()

            if len(valid_dates) < 2:
                return {'gaps_detected': 0, 'largest_gap_days': 0}

            # Calculate gaps between consecutive dates
            gaps = valid_dates.diff().dt.days.dropna()

            gaps_detected = (gaps > 1).sum()  # Gaps larger than 1 day
            largest_gap = gaps.max()

            return {
                'gaps_detected': int(gaps_detected),
                'largest_gap_days': int(largest_gap) if not pd.isna(largest_gap) else 0,
                'average_gap_days': gaps.mean(),
                'gap_distribution': gaps.value_counts().to_dict()
            }

        except Exception:
            return {'gaps_detected': 0, 'largest_gap_days': 0}

    def _calculate_consistency_score(self, date_series: pd.Series) -> float:
        """Calculate temporal consistency score"""
        try:
            if len(date_series) == 0:
                return 0.0

            valid_ratio = date_series.notna().mean()
            uniqueness_ratio = date_series.dropna().nunique() / len(date_series.dropna())

            # Check for reasonable date ranges (not too far in past/future)
            current_year = datetime.now().year
            reasonable_years = ((date_series.dt.year >= current_year - 10) &
                              (date_series.dt.year <= current_year + 1))
            reasonable_ratio = reasonable_years.mean() if date_series.notna().any() else 0

            # Weighted consistency score
            consistency_score = (
                valid_ratio * 0.5 +
                uniqueness_ratio * 0.3 +
                reasonable_ratio * 0.2
            )

            return float(consistency_score)

        except Exception:
            return 0.0

    def _update_temporal_stats(self, feature_names: List[str]):
        """Update temporal feature extraction statistics"""
        try:
            for feature_name in feature_names:
                self.temporal_stats[feature_name] += 1

            self.temporal_stats['total_extractions'] += 1

        except Exception as e:
            self.logger.warning(f"Error updating temporal stats: {str(e)}")

    def get_temporal_stats(self) -> Dict[str, Any]:
        """Get temporal feature extraction statistics"""
        return dict(self.temporal_stats)

    def clear_cache(self):
        """Clear temporal feature cache"""
        self.feature_cache.clear()
        self.logger.info("Temporal feature cache cleared")

    def save_temporal_enhancer(self, filepath: str):
        """Save the temporal feature enhancer state"""
        try:
            import joblib

            save_dict = {
                'config': self.config,
                'temporal_stats': dict(self.temporal_stats),
                'pattern_history': self.pattern_history
            }

            joblib.dump(save_dict, filepath)
            self.logger.info(f"TemporalFeatureEnhancer saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving TemporalFeatureEnhancer: {str(e)}")

    def load_temporal_enhancer(self, filepath: str):
        """Load the temporal feature enhancer state"""
        try:
            import joblib

            save_dict = joblib.load(filepath)

            self.config = save_dict['config']
            self.temporal_stats = defaultdict(int, save_dict['temporal_stats'])
            self.pattern_history = save_dict['pattern_history']

            # Reinitialize components
            self._initialize_components()

            self.logger.info(f"TemporalFeatureEnhancer loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading TemporalFeatureEnhancer: {str(e)}")
            raise ValidationError(f"Failed to load TemporalFeatureEnhancer: {str(e)}")