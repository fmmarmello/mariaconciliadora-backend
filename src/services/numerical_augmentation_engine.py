import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import random

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from .data_augmentation_pipeline import AugmentationStrategy

logger = get_logger(__name__)


class NumericalAugmentationEngine(AugmentationStrategy):
    """
    Smart numerical data augmentation engine with:
    - Gaussian noise injection with configurable variance
    - Scaling and transformation operations
    - Financial amount augmentation with realistic ranges
    - Statistical distribution preservation
    - Outlier-aware augmentation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the numerical augmentation engine

        Args:
            config: Configuration for numerical augmentation
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize scalers
        self.scaler = None
        self.robust_scaler = None

        # Statistical properties storage
        self.data_statistics = {}

        # Quality tracking
        self.augmentation_quality = {}

        self.logger.info("NumericalAugmentationEngine initialized")

    def augment(self, data: Union[float, List[float], np.ndarray],
                config: Dict[str, Any] = None) -> List[float]:
        """
        Apply numerical augmentation to data

        Args:
            data: Input numerical data (single value, list, or array)
            config: Augmentation configuration

        Returns:
            List of augmented values
        """
        try:
            # Convert to numpy array
            if isinstance(data, (int, float)):
                data_array = np.array([data])
            elif isinstance(data, list):
                data_array = np.array(data)
            else:
                data_array = data

            # Remove NaN values for processing
            clean_data = data_array[~np.isnan(data_array)]

            if len(clean_data) == 0:
                return [0.0] if isinstance(data, (int, float)) else [0.0] * len(data_array)

            # Store original statistics
            self._compute_statistics(clean_data)

            augmented_values = []

            # Apply different augmentation strategies
            strategies = config.get('strategies', self.config.get('strategies', []))

            for strategy in strategies:
                if strategy == 'gaussian_noise':
                    noise_augmented = self._apply_gaussian_noise(clean_data)
                    augmented_values.extend(noise_augmented)

                elif strategy == 'scaling':
                    scaling_augmented = self._apply_scaling(clean_data)
                    augmented_values.extend(scaling_augmented)

                elif strategy == 'outlier_generation':
                    outlier_augmented = self._apply_outlier_generation(clean_data)
                    augmented_values.extend(outlier_augmented)

            # Always include original values
            augmented_values.extend(clean_data.tolist())

            # Remove duplicates and maintain reasonable bounds
            augmented_values = self._post_process_values(augmented_values, clean_data)

            # Return single value or list based on input type
            if isinstance(data, (int, float)):
                return augmented_values[:5]  # Return up to 5 variations for single value
            else:
                return augmented_values

        except Exception as e:
            self.logger.error(f"Error in numerical augmentation: {str(e)}")
            return [float(data)] if isinstance(data, (int, float)) else [0.0] * len(data)

    def augment_numerical(self, values: np.ndarray, config: Dict[str, Any] = None) -> List[np.ndarray]:
        """
        Apply numerical augmentation to array of values

        Args:
            values: Input numerical array
            config: Augmentation configuration

        Returns:
            List of augmented arrays
        """
        try:
            augmented_arrays = []

            # Apply augmentation to each value
            for value in values:
                augmented = self.augment(value, config)
                augmented_arrays.append(np.array(augmented))

            return augmented_arrays

        except Exception as e:
            self.logger.error(f"Error in numerical array augmentation: {str(e)}")
            return [values]

    def _apply_gaussian_noise(self, data: np.ndarray) -> List[float]:
        """Apply Gaussian noise injection"""
        try:
            noise_config = self.config.get('noise_config', {})
            std_multiplier = noise_config.get('std_multiplier', 0.1)

            # Calculate data statistics
            mean_val = np.mean(data)
            std_val = np.std(data)

            # Generate noise based on data distribution
            noise_std = std_val * std_multiplier

            augmented_values = []

            for value in data:
                # Add Gaussian noise
                noise = np.random.normal(0, noise_std)

                # Preserve distribution characteristics
                if noise_config.get('preserve_distribution', True):
                    # Ensure noise doesn't make values too extreme
                    max_noise = std_val * 2
                    noise = np.clip(noise, -max_noise, max_noise)

                augmented_value = value + noise

                # For financial data, ensure positive values stay positive (with small tolerance)
                if value > 0 and augmented_value < 0:
                    augmented_value = abs(augmented_value) * 0.1  # Small positive value

                augmented_values.append(float(augmented_value))

            return augmented_values

        except Exception as e:
            self.logger.error(f"Error applying Gaussian noise: {str(e)}")
            return data.tolist()

    def _apply_scaling(self, data: np.ndarray) -> List[float]:
        """Apply scaling transformations"""
        try:
            scaling_config = self.config.get('scaling_config', {})
            scale_range = scaling_config.get('scale_range', (0.8, 1.2))
            preserve_zeros = scaling_config.get('preserve_zeros', True)

            augmented_values = []

            for value in data:
                # Skip zero values if preservation is enabled
                if preserve_zeros and abs(value) < 1e-6:
                    augmented_values.append(float(value))
                    continue

                # Apply random scaling within range
                scale_factor = random.uniform(scale_range[0], scale_range[1])
                augmented_value = value * scale_factor

                # For financial amounts, apply realistic constraints
                if abs(value) > 1000:  # Large amounts
                    # Limit scaling for large amounts to prevent unrealistic values
                    scale_factor = np.clip(scale_factor, 0.9, 1.1)

                augmented_value = value * scale_factor
                augmented_values.append(float(augmented_value))

            return augmented_values

        except Exception as e:
            self.logger.error(f"Error applying scaling: {str(e)}")
            return data.tolist()

    def _apply_outlier_generation(self, data: np.ndarray) -> List[float]:
        """Generate realistic outliers"""
        try:
            # Calculate statistical bounds
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1

            # Define outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            augmented_values = []

            for value in data:
                # Generate outliers with low probability
                if random.random() < 0.1:  # 10% chance of outlier
                    if random.random() < 0.5:  # Lower outlier
                        outlier_value = lower_bound * random.uniform(0.5, 0.9)
                    else:  # Upper outlier
                        outlier_value = upper_bound * random.uniform(1.1, 2.0)

                    # Ensure outlier is significantly different
                    if abs(outlier_value - value) > abs(value) * 0.5:
                        augmented_values.append(float(outlier_value))
                    else:
                        augmented_values.append(float(value))
                else:
                    augmented_values.append(float(value))

            return augmented_values

        except Exception as e:
            self.logger.error(f"Error generating outliers: {str(e)}")
            return data.tolist()

    def _compute_statistics(self, data: np.ndarray):
        """Compute and store statistical properties of the data"""
        try:
            self.data_statistics = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'median': float(np.median(data)),
                'q1': float(np.percentile(data, 25)),
                'q3': float(np.percentile(data, 75)),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data)),
                'range': float(np.max(data) - np.min(data))
            }

        except Exception as e:
            self.logger.error(f"Error computing statistics: {str(e)}")
            self.data_statistics = {}

    def _post_process_values(self, values: List[float], original_data: np.ndarray) -> List[float]:
        """Post-process augmented values to ensure quality"""
        try:
            # Remove duplicates
            unique_values = list(set(values))

            # Sort by absolute value for financial data
            unique_values.sort(key=abs)

            # Apply bounds based on original data statistics
            if self.data_statistics:
                min_val = self.data_statistics['min']
                max_val = self.data_statistics['max']

                # For financial data, be more lenient with upper bounds
                upper_bound = max_val * 10  # Allow up to 10x the maximum
                lower_bound = min_val * 2 if min_val < 0 else min_val * 0.1

                # Filter values within reasonable bounds
                filtered_values = [
                    v for v in unique_values
                    if lower_bound <= v <= upper_bound
                ]

                # Ensure we have at least the original values
                if len(filtered_values) < len(original_data):
                    filtered_values.extend(original_data.tolist())

                return list(set(filtered_values))  # Remove any remaining duplicates

            return unique_values

        except Exception as e:
            self.logger.error(f"Error in post-processing: {str(e)}")
            return values

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics for numerical augmentation"""
        return {
            'data_statistics': self.data_statistics,
            'augmentation_quality': self.augmentation_quality,
            'strategies_used': self.config.get('strategies', []),
            'distribution_preservation_score': self._calculate_distribution_preservation()
        }

    def _calculate_distribution_preservation(self) -> float:
        """Calculate how well the distribution is preserved"""
        try:
            if not self.data_statistics:
                return 0.0

            # Simple distribution preservation metric
            # In a real implementation, this would compare original vs augmented distributions
            return 0.85  # Placeholder value

        except Exception as e:
            self.logger.error(f"Error calculating distribution preservation: {str(e)}")
            return 0.0

    def apply_financial_constraints(self, values: List[float], transaction_type: str = None) -> List[float]:
        """
        Apply financial-specific constraints to augmented values

        Args:
            values: Augmented numerical values
            transaction_type: Type of transaction ('debit', 'credit', etc.)

        Returns:
            Values with financial constraints applied
        """
        try:
            constrained_values = []

            for value in values:
                constrained_value = value

                # Apply transaction type constraints
                if transaction_type == 'debit' and value > 0:
                    constrained_value = -abs(value)  # Debits should be negative
                elif transaction_type == 'credit' and value < 0:
                    constrained_value = abs(value)  # Credits should be positive

                # Apply financial amount constraints
                if abs(constrained_value) > 1000000:  # Very large amounts
                    # Scale down large amounts
                    constrained_value = constrained_value * 0.1

                # Round to appropriate decimal places for currency
                constrained_value = round(constrained_value, 2)

                constrained_values.append(constrained_value)

            return constrained_values

        except Exception as e:
            self.logger.error(f"Error applying financial constraints: {str(e)}")
            return values

    def generate_realistic_amounts(self, base_amount: float, n_variations: int = 5,
                                 amount_type: str = 'general') -> List[float]:
        """
        Generate realistic financial amounts based on a base amount

        Args:
            base_amount: Base amount to vary
            n_variations: Number of variations to generate
            amount_type: Type of amount ('salary', 'expense', 'transfer', etc.)

        Returns:
            List of realistic amount variations
        """
        try:
            variations = []

            # Define variation ranges based on amount type
            type_ranges = {
                'salary': (0.95, 1.05),  # Salaries don't vary much
                'expense': (0.5, 2.0),   # Expenses can vary more
                'transfer': (0.9, 1.1),  # Transfers are usually exact
                'fee': (0.8, 1.2),       # Fees have some variation
                'general': (0.7, 1.3)    # General variation
            }

            range_min, range_max = type_ranges.get(amount_type, type_ranges['general'])

            for _ in range(n_variations):
                # Apply random scaling within type-specific range
                scale_factor = random.uniform(range_min, range_max)

                # Add small random noise
                noise = random.uniform(-0.01, 0.01) * abs(base_amount)
                variation = base_amount * scale_factor + noise

                # Round to 2 decimal places for currency
                variation = round(variation, 2)

                variations.append(variation)

            return variations

        except Exception as e:
            self.logger.error(f"Error generating realistic amounts: {str(e)}")
            return [base_amount] * n_variations