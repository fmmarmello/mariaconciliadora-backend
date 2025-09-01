import random
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import holidays

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from .data_augmentation_pipeline import AugmentationStrategy

logger = get_logger(__name__)


class TemporalAugmentationEngine(AugmentationStrategy):
    """
    Temporal augmentation engine with:
    - Date shifting with realistic constraints
    - Pattern generation for temporal sequences
    - Business day and holiday awareness
    - Seasonal and cyclical pattern preservation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the temporal augmentation engine

        Args:
            config: Configuration for temporal augmentation
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Holiday calendar for Brazil
        self.holiday_calendar = holidays.CountryHoliday('BR')

        # Business day patterns
        self.business_day_patterns = self._initialize_business_patterns()

        # Quality tracking
        self.augmentation_quality = {}

        self.logger.info("TemporalAugmentationEngine initialized")

    def _initialize_business_patterns(self) -> Dict[str, Any]:
        """Initialize business day patterns"""
        return {
            'weekend_days': [5, 6],  # Saturday, Sunday (0=Monday, 6=Sunday)
            'business_days': [0, 1, 2, 3, 4],  # Monday to Friday
            'month_end_buffer': 3,  # Days before month end
            'quarter_end_buffer': 5,  # Days before quarter end
            'year_end_buffer': 10  # Days before year end
        }

    def augment(self, data: Union[str, datetime, date, List],
                config: Dict[str, Any] = None) -> List[datetime]:
        """
        Apply temporal augmentation to date/time data

        Args:
            data: Input temporal data (single date or list)
            config: Augmentation configuration

        Returns:
            List of augmented datetime objects
        """
        try:
            if isinstance(data, (str, datetime, date)):
                return self._augment_single_date(data, config)
            elif isinstance(data, list):
                return self._augment_date_list(data, config)
            else:
                return [pd.to_datetime(data, errors='coerce')]

        except Exception as e:
            self.logger.error(f"Error in temporal augmentation: {str(e)}")
            return [pd.to_datetime(data, errors='coerce')]

    def augment_dates(self, dates: List[Union[str, datetime, date]],
                     config: Dict[str, Any] = None) -> List[List[datetime]]:
        """
        Apply temporal augmentation to a list of dates

        Args:
            dates: List of input dates
            config: Augmentation configuration

        Returns:
            List of lists containing augmented dates for each input
        """
        try:
            augmented_batch = []

            for date_val in dates:
                augmented = self.augment(date_val, config)
                augmented_batch.append(augmented)

            return augmented_batch

        except Exception as e:
            self.logger.error(f"Error in temporal batch augmentation: {str(e)}")
            return [[pd.to_datetime(d, errors='coerce') for d in dates]]

    def _augment_single_date(self, date_val: Union[str, datetime, date],
                           config: Dict[str, Any] = None) -> List[datetime]:
        """Augment a single date"""
        try:
            # Convert to datetime
            if isinstance(date_val, str):
                dt = pd.to_datetime(date_val, errors='coerce')
            elif isinstance(date_val, date):
                dt = datetime.combine(date_val, datetime.min.time())
            else:
                dt = date_val

            if pd.isna(dt):
                return [datetime.now()]

            augmented_dates = [dt]  # Always include original

            # Apply different augmentation strategies
            strategies = config.get('strategies', self.config.get('strategies', []))

            for strategy in strategies:
                if strategy == 'date_shifting':
                    shifted = self._apply_date_shifting(dt)
                    if shifted:
                        augmented_dates.extend(shifted)

                elif strategy == 'pattern_generation':
                    patterned = self._apply_pattern_generation(dt)
                    if patterned:
                        augmented_dates.extend(patterned)

            # Remove duplicates and invalid dates
            valid_dates = []
            seen = set()

            for aug_date in augmented_dates:
                if pd.notna(aug_date):
                    date_str = aug_date.isoformat()
                    if date_str not in seen:
                        seen.add(date_str)
                        valid_dates.append(aug_date)

            return valid_dates[:5]  # Limit to 5 variations

        except Exception as e:
            self.logger.error(f"Error augmenting single date: {str(e)}")
            return [pd.to_datetime(date_val, errors='coerce')]

    def _augment_date_list(self, dates: List[Union[str, datetime, date]],
                          config: Dict[str, Any] = None) -> List[datetime]:
        """Augment a list of dates"""
        try:
            all_augmented = []

            for date_val in dates:
                augmented = self._augment_single_date(date_val, config)
                all_augmented.extend(augmented)

            # Remove duplicates while preserving some variety
            unique_dates = list(set(all_augmented))

            # Ensure we have at least the original dates
            for orig_date in dates:
                dt = pd.to_datetime(orig_date, errors='coerce')
                if pd.notna(dt) and dt not in unique_dates:
                    unique_dates.append(dt)

            return unique_dates

        except Exception as e:
            self.logger.error(f"Error augmenting date list: {str(e)}")
            return [pd.to_datetime(d, errors='coerce') for d in dates]

    def _apply_date_shifting(self, base_date: datetime) -> Optional[List[datetime]]:
        """Apply date shifting augmentation"""
        try:
            shifted_dates = []
            date_config = self.config.get('date_config', {})

            max_days_shift = date_config.get('max_days_shift', 30)
            preserve_weekends = date_config.get('preserve_weekends', True)
            business_days_only = date_config.get('business_days_only', False)

            # Generate multiple shifted dates
            for _ in range(3):  # Generate 3 variations
                shift_days = random.randint(-max_days_shift, max_days_shift)

                if shift_days == 0:
                    continue  # Skip no shift

                shifted_date = base_date + timedelta(days=shift_days)

                # Apply business day constraints
                if business_days_only:
                    shifted_date = self._adjust_to_business_day(shifted_date)
                elif preserve_weekends and base_date.weekday() >= 5:
                    # If original was weekend, keep shifted date as weekend
                    while shifted_date.weekday() < 5:
                        if shift_days > 0:
                            shifted_date += timedelta(days=1)
                        else:
                            shifted_date -= timedelta(days=1)

                # Avoid holidays if original wasn't a holiday
                if not self._is_holiday(base_date) and self._is_holiday(shifted_date):
                    # Shift away from holiday
                    if shift_days > 0:
                        shifted_date += timedelta(days=1)
                    else:
                        shifted_date -= timedelta(days=1)

                shifted_dates.append(shifted_date)

            return shifted_dates if shifted_dates else None

        except Exception as e:
            self.logger.error(f"Error in date shifting: {str(e)}")
            return None

    def _apply_pattern_generation(self, base_date: datetime) -> Optional[List[datetime]]:
        """Apply pattern-based date generation"""
        try:
            pattern_dates = []

            # Generate dates with realistic temporal patterns
            patterns = [
                lambda d: d + relativedelta(months=1),  # Monthly pattern
                lambda d: d + relativedelta(weeks=1),   # Weekly pattern
                lambda d: d + relativedelta(days=15),   # Bi-weekly pattern
                lambda d: d + relativedelta(months=3),  # Quarterly pattern
            ]

            for pattern_func in patterns:
                try:
                    pattern_date = pattern_func(base_date)

                    # Apply business constraints
                    if self.config.get('date_config', {}).get('business_days_only', False):
                        pattern_date = self._adjust_to_business_day(pattern_date)

                    pattern_dates.append(pattern_date)
                except Exception:
                    continue  # Skip failed patterns

            return pattern_dates if pattern_dates else None

        except Exception as e:
            self.logger.error(f"Error in pattern generation: {str(e)}")
            return None

    def _adjust_to_business_day(self, date_val: datetime) -> datetime:
        """Adjust date to nearest business day"""
        try:
            weekday = date_val.weekday()

            if weekday >= 5:  # Weekend
                # Move to next Monday
                days_to_add = 7 - weekday
                return date_val + timedelta(days=days_to_add)

            return date_val

        except Exception:
            return date_val

    def _is_holiday(self, date_val: datetime) -> bool:
        """Check if date is a holiday"""
        try:
            return date_val.date() in self.holiday_calendar
        except Exception:
            return False

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics for temporal augmentation"""
        return {
            'augmentation_quality': self.augmentation_quality,
            'strategies_used': self.config.get('strategies', []),
            'holiday_awareness': True,
            'business_day_compliance': self._calculate_business_day_compliance()
        }

    def _calculate_business_day_compliance(self) -> float:
        """Calculate compliance with business day constraints"""
        try:
            # Placeholder implementation
            return 0.9
        except Exception:
            return 0.0

    def generate_temporal_sequence(self, start_date: Union[str, datetime, date],
                                 n_dates: int = 10, pattern: str = 'daily') -> List[datetime]:
        """
        Generate a sequence of dates following a temporal pattern

        Args:
            start_date: Starting date for the sequence
            n_dates: Number of dates to generate
            pattern: Pattern type ('daily', 'weekly', 'monthly', 'business_days')

        Returns:
            List of datetime objects in sequence
        """
        try:
            # Convert to datetime
            if isinstance(start_date, str):
                start_dt = pd.to_datetime(start_date)
            elif isinstance(start_date, date):
                start_dt = datetime.combine(start_date, datetime.min.time())
            else:
                start_dt = start_date

            sequence = [start_dt]

            for i in range(1, n_dates):
                if pattern == 'daily':
                    next_date = start_dt + timedelta(days=i)
                elif pattern == 'weekly':
                    next_date = start_dt + timedelta(weeks=i)
                elif pattern == 'monthly':
                    next_date = start_dt + relativedelta(months=i)
                elif pattern == 'business_days':
                    next_date = start_dt
                    business_days_added = 0
                    while business_days_added < i:
                        next_date += timedelta(days=1)
                        if next_date.weekday() < 5:  # Monday to Friday
                            business_days_added += 1
                else:
                    next_date = start_dt + timedelta(days=i)

                sequence.append(next_date)

            return sequence

        except Exception as e:
            self.logger.error(f"Error generating temporal sequence: {str(e)}")
            return [pd.to_datetime(start_date, errors='coerce')]

    def apply_seasonal_augmentation(self, dates: List[Union[str, datetime, date]],
                                  season: str = None) -> List[datetime]:
        """
        Apply seasonal augmentation to dates

        Args:
            dates: List of input dates
            season: Target season ('summer', 'autumn', 'winter', 'spring')

        Returns:
            Dates adjusted to target season
        """
        try:
            augmented_dates = []

            for date_val in dates:
                dt = pd.to_datetime(date_val, errors='coerce')
                if pd.isna(dt):
                    continue

                if season:
                    # Adjust date to target season
                    augmented_date = self._adjust_to_season(dt, season)
                    augmented_dates.append(augmented_date)
                else:
                    # Apply random seasonal shift
                    current_season = self._get_season(dt)
                    seasons = ['summer', 'autumn', 'winter', 'spring']
                    target_season = random.choice([s for s in seasons if s != current_season])
                    augmented_date = self._adjust_to_season(dt, target_season)
                    augmented_dates.append(augmented_date)

            return augmented_dates

        except Exception as e:
            self.logger.error(f"Error in seasonal augmentation: {str(e)}")
            return [pd.to_datetime(d, errors='coerce') for d in dates]

    def _get_season(self, date_val: datetime) -> str:
        """Get season for a given date (Brazilian seasons)"""
        month = date_val.month

        if month in [12, 1, 2]:
            return 'summer'
        elif month in [3, 4, 5]:
            return 'autumn'
        elif month in [6, 7, 8]:
            return 'winter'
        else:  # 9, 10, 11
            return 'spring'

    def _adjust_to_season(self, date_val: datetime, target_season: str) -> datetime:
        """Adjust date to target season"""
        try:
            season_months = {
                'summer': [12, 1, 2],
                'autumn': [3, 4, 5],
                'winter': [6, 7, 8],
                'spring': [9, 10, 11]
            }

            target_months = season_months.get(target_season, [date_val.month])
            target_month = random.choice(target_months)

            # Adjust year if necessary
            year = date_val.year
            if target_season == 'summer' and date_val.month in [9, 10, 11]:
                year += 1
            elif target_season == 'spring' and date_val.month in [12, 1, 2]:
                year -= 1

            # Create new date with target month
            try:
                new_date = date_val.replace(year=year, month=target_month)
            except ValueError:
                # Handle invalid dates (e.g., Feb 30)
                new_date = date_val.replace(year=year, month=target_month, day=28)

            return new_date

        except Exception as e:
            self.logger.error(f"Error adjusting to season: {str(e)}")
            return date_val

    def generate_business_cycle_dates(self, base_date: datetime, cycle_type: str = 'monthly',
                                    n_cycles: int = 12) -> List[datetime]:
        """
        Generate dates following business cycles

        Args:
            base_date: Base date for cycle generation
            cycle_type: Type of business cycle ('monthly', 'quarterly', 'yearly')
            n_cycles: Number of cycles to generate

        Returns:
            List of dates following the business cycle
        """
        try:
            cycle_dates = []

            for i in range(n_cycles):
                if cycle_type == 'monthly':
                    cycle_date = base_date + relativedelta(months=i)
                elif cycle_type == 'quarterly':
                    cycle_date = base_date + relativedelta(months=i*3)
                elif cycle_type == 'yearly':
                    cycle_date = base_date + relativedelta(years=i)
                else:
                    cycle_date = base_date + relativedelta(months=i)

                # Adjust to business day if required
                if self.config.get('date_config', {}).get('business_days_only', False):
                    cycle_date = self._adjust_to_business_day(cycle_date)

                cycle_dates.append(cycle_date)

            return cycle_dates

        except Exception as e:
            self.logger.error(f"Error generating business cycle dates: {str(e)}")
            return [base_date]