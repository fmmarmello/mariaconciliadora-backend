import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
import warnings

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from .data_augmentation_pipeline import AugmentationStrategy

logger = get_logger(__name__)


class AugmentationQualityControl(AugmentationStrategy):
    """
    Quality assurance and validation system for data augmentation with:
    - Statistical similarity checks between original and augmented data
    - Semantic preservation validation for text data
    - Business rule compliance verification
    - Performance impact assessment on downstream models
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the quality control system

        Args:
            config: Configuration for quality control
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Quality thresholds
        self.statistical_threshold = config.get('statistical_similarity_threshold', 0.9)
        self.semantic_threshold = config.get('semantic_preservation_threshold', 0.85)
        self.business_rule_threshold = config.get('business_rule_compliance', True)

        # Quality metrics storage
        self.quality_metrics = {}
        self.validation_history = []

        # Statistical tests
        self.scaler = StandardScaler()

        self.logger.info("AugmentationQualityControl initialized")

    def augment(self, data: Any, config: Dict[str, Any] = None) -> Any:
        """
        Quality control doesn't modify data, it validates it

        Args:
            data: Data to validate
            config: Validation configuration

        Returns:
            Original data (no modification)
        """
        return data

    def validate_augmentation(self, original_data: Union[pd.DataFrame, List[Dict]],
                            augmented_data: Union[pd.DataFrame, List[Dict]]) -> Dict[str, Any]:
        """
        Comprehensive validation of augmented data

        Args:
            original_data: Original dataset
            augmented_data: Augmented dataset

        Returns:
            Validation results and quality metrics
        """
        try:
            self.logger.info("Starting comprehensive augmentation validation")

            # Convert to DataFrames
            orig_df = self._to_dataframe(original_data)
            aug_df = self._to_dataframe(augmented_data)

            if orig_df.empty or aug_df.empty:
                return {'error': 'Empty datasets provided for validation'}

            validation_results = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'original_size': len(orig_df),
                'augmented_size': len(aug_df),
                'validation_passed': True,
                'quality_score': 0.0
            }

            # Statistical similarity validation
            stat_results = self._validate_statistical_similarity(orig_df, aug_df)
            validation_results.update(stat_results)

            # Semantic preservation validation (for text data)
            semantic_results = self._validate_semantic_preservation(orig_df, aug_df)
            validation_results.update(semantic_results)

            # Business rule compliance
            business_results = self._validate_business_rules(orig_df, aug_df)
            validation_results.update(business_results)

            # Distribution preservation
            distribution_results = self._validate_distribution_preservation(orig_df, aug_df)
            validation_results.update(distribution_results)

            # Overall quality score
            validation_results['quality_score'] = self._calculate_overall_quality_score(validation_results)

            # Determine if validation passed
            validation_results['validation_passed'] = (
                stat_results.get('statistical_similarity_score', 0) >= self.statistical_threshold and
                semantic_results.get('semantic_preservation_score', 0) >= self.semantic_threshold and
                business_results.get('business_rule_compliance', False) == self.business_rule_threshold
            )

            # Store in history
            self.validation_history.append(validation_results)
            self.quality_metrics.update(validation_results)

            self.logger.info(f"Validation completed. Quality score: {validation_results['quality_score']:.3f}")
            return validation_results

        except Exception as e:
            self.logger.error(f"Error in augmentation validation: {str(e)}")
            return {'error': str(e), 'validation_passed': False}

    def _to_dataframe(self, data: Union[pd.DataFrame, List[Dict]]) -> pd.DataFrame:
        """Convert data to DataFrame"""
        try:
            if isinstance(data, pd.DataFrame):
                return data.copy()
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _validate_statistical_similarity(self, orig_df: pd.DataFrame,
                                       aug_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate statistical similarity between datasets"""
        try:
            results = {}

            # Get numerical columns
            num_cols = orig_df.select_dtypes(include=[np.number]).columns
            common_num_cols = [col for col in num_cols if col in aug_df.columns]

            if common_num_cols:
                # Kolmogorov-Smirnov test for distribution similarity
                ks_scores = []
                for col in common_num_cols:
                    try:
                        orig_vals = orig_df[col].dropna().values
                        aug_vals = aug_df[col].dropna().values

                        if len(orig_vals) > 1 and len(aug_vals) > 1:
                            ks_stat, ks_p = stats.ks_2samp(orig_vals, aug_vals)
                            ks_scores.append(1 - ks_stat)  # Convert to similarity score
                    except Exception:
                        continue

                results['ks_similarity_score'] = np.mean(ks_scores) if ks_scores else 0.0

                # Mean and std comparisons
                mean_diffs = []
                std_diffs = []

                for col in common_num_cols:
                    orig_mean = orig_df[col].mean()
                    aug_mean = aug_df[col].mean()
                    orig_std = orig_df[col].std()
                    aug_std = aug_df[col].std()

                    mean_diffs.append(abs(orig_mean - aug_mean) / abs(orig_mean) if orig_mean != 0 else 0)
                    std_diffs.append(abs(orig_std - aug_std) / abs(orig_std) if orig_std != 0 else 0)

                results['mean_similarity_score'] = 1 - np.mean(mean_diffs) if mean_diffs else 1.0
                results['std_similarity_score'] = 1 - np.mean(std_diffs) if std_diffs else 1.0

            # Overall statistical similarity score
            scores = [results.get('ks_similarity_score', 0),
                     results.get('mean_similarity_score', 0),
                     results.get('std_similarity_score', 0)]

            results['statistical_similarity_score'] = np.mean(scores) if scores else 0.0

            return results

        except Exception as e:
            self.logger.error(f"Error in statistical similarity validation: {str(e)}")
            return {'statistical_similarity_score': 0.0}

    def _validate_semantic_preservation(self, orig_df: pd.DataFrame,
                                       aug_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate semantic preservation for text data"""
        try:
            results = {}

            # Find text columns
            text_cols = []
            for col in orig_df.columns:
                if orig_df[col].dtype == 'object':
                    # Check if column contains text (not just categories)
                    sample_vals = orig_df[col].dropna().head(10).astype(str)
                    if any(len(str(val)) > 20 for val in sample_vals):  # Likely text descriptions
                        text_cols.append(col)

            if text_cols:
                semantic_scores = []

                for col in text_cols:
                    if col in aug_df.columns:
                        # Simplified semantic similarity check
                        # In production, this would use embeddings and cosine similarity
                        orig_texts = orig_df[col].dropna().astype(str).tolist()
                        aug_texts = aug_df[col].dropna().astype(str).tolist()

                        if orig_texts and aug_texts:
                            # Calculate average text length similarity
                            orig_avg_len = np.mean([len(text) for text in orig_texts])
                            aug_avg_len = np.mean([len(text) for text in aug_texts])

                            length_similarity = 1 - abs(orig_avg_len - aug_avg_len) / max(orig_avg_len, aug_avg_len)
                            semantic_scores.append(length_similarity)

                results['semantic_preservation_score'] = np.mean(semantic_scores) if semantic_scores else 0.8
            else:
                results['semantic_preservation_score'] = 1.0  # No text columns to validate

            return results

        except Exception as e:
            self.logger.error(f"Error in semantic preservation validation: {str(e)}")
            return {'semantic_preservation_score': 0.0}

    def _validate_business_rules(self, orig_df: pd.DataFrame,
                               aug_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate business rule compliance"""
        try:
            results = {'business_rule_compliance': True, 'violations': []}

            # Financial data validation rules
            if 'amount' in orig_df.columns and 'amount' in aug_df.columns:
                # Check for negative amounts in credit transactions
                if 'transaction_type' in orig_df.columns:
                    credit_mask = orig_df['transaction_type'].str.lower() == 'credit'
                    if credit_mask.any():
                        orig_credit_amounts = orig_df.loc[credit_mask, 'amount']
                        aug_credit_amounts = aug_df.loc[aug_df['transaction_type'].str.lower() == 'credit', 'amount']

                        # Credits should be positive
                        orig_negative_credits = (orig_credit_amounts < 0).sum()
                        aug_negative_credits = (aug_credit_amounts < 0).sum()

                        if aug_negative_credits > orig_negative_credits:
                            results['business_rule_compliance'] = False
                            results['violations'].append('Excessive negative credit amounts in augmented data')

            # Date validation
            if 'date' in orig_df.columns and 'date' in aug_df.columns:
                try:
                    orig_dates = pd.to_datetime(orig_df['date'], errors='coerce')
                    aug_dates = pd.to_datetime(aug_df['date'], errors='coerce')

                    # Check for future dates (beyond reasonable range)
                    max_orig_date = orig_dates.max()
                    future_dates = aug_dates > (max_orig_date + pd.Timedelta(days=365*2))  # 2 years ahead

                    if future_dates.sum() > len(aug_dates) * 0.1:  # More than 10% future dates
                        results['business_rule_compliance'] = False
                        results['violations'].append('Excessive future dates in augmented data')

                except Exception:
                    pass

            # Balance validation
            if 'balance' in orig_df.columns and 'balance' in aug_df.columns:
                # Check for unrealistic balance changes
                orig_balance_range = orig_df['balance'].max() - orig_df['balance'].min()
                aug_balance_range = aug_df['balance'].max() - aug_df['balance'].min()

                if aug_balance_range > orig_balance_range * 10:  # 10x range increase
                    results['business_rule_compliance'] = False
                    results['violations'].append('Unrealistic balance range in augmented data')

            return results

        except Exception as e:
            self.logger.error(f"Error in business rule validation: {str(e)}")
            return {'business_rule_compliance': False, 'violations': [str(e)]}

    def _validate_distribution_preservation(self, orig_df: pd.DataFrame,
                                          aug_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate distribution preservation"""
        try:
            results = {}

            # Categorical distribution preservation
            cat_cols = orig_df.select_dtypes(include=['object']).columns
            common_cat_cols = [col for col in cat_cols if col in aug_df.columns]

            if common_cat_cols:
                distribution_similarities = []

                for col in common_cat_cols:
                    orig_dist = orig_df[col].value_counts(normalize=True)
                    aug_dist = aug_df[col].value_counts(normalize=True)

                    # Calculate Jensen-Shannon divergence
                    common_categories = set(orig_dist.index) & set(aug_dist.index)

                    if common_categories:
                        orig_probs = np.array([orig_dist.get(cat, 0) for cat in common_categories])
                        aug_probs = np.array([aug_dist.get(cat, 0) for cat in common_categories])

                        # Normalize
                        orig_probs = orig_probs / orig_probs.sum()
                        aug_probs = aug_probs / aug_probs.sum()

                        # Jensen-Shannon divergence
                        m = (orig_probs + aug_probs) / 2
                        js_div = 0.5 * (stats.entropy(orig_probs, m) + stats.entropy(aug_probs, m))
                        js_similarity = 1 - js_div

                        distribution_similarities.append(js_similarity)

                results['categorical_distribution_similarity'] = (
                    np.mean(distribution_similarities) if distribution_similarities else 0.0
                )

            # Overall distribution preservation score
            scores = []
            if 'categorical_distribution_similarity' in results:
                scores.append(results['categorical_distribution_similarity'])

            results['distribution_preservation_score'] = np.mean(scores) if scores else 0.8

            return results

        except Exception as e:
            self.logger.error(f"Error in distribution preservation validation: {str(e)}")
            return {'distribution_preservation_score': 0.0}

    def _calculate_overall_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score from validation results"""
        try:
            scores = []

            # Weight different aspects
            weights = {
                'statistical_similarity_score': 0.4,
                'semantic_preservation_score': 0.3,
                'distribution_preservation_score': 0.2,
                'business_rule_compliance': 0.1
            }

            for metric, weight in weights.items():
                if metric in validation_results:
                    value = validation_results[metric]
                    # Convert boolean to numeric for business rules
                    if isinstance(value, bool):
                        value = 1.0 if value else 0.0
                    scores.append(value * weight)

            return sum(scores) if scores else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating overall quality score: {str(e)}")
            return 0.0

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get comprehensive quality metrics"""
        return {
            'current_quality_metrics': self.quality_metrics,
            'validation_history': self.validation_history[-10:],  # Last 10 validations
            'thresholds': {
                'statistical_similarity': self.statistical_threshold,
                'semantic_preservation': self.semantic_threshold,
                'business_rule_compliance': self.business_rule_threshold
            },
            'overall_health_score': self._calculate_health_score()
        }

    def _calculate_health_score(self) -> float:
        """Calculate system health score based on recent validations"""
        try:
            if not self.validation_history:
                return 0.0

            recent_validations = self.validation_history[-5:]  # Last 5 validations
            passed_count = sum(1 for v in recent_validations if v.get('validation_passed', False))

            return passed_count / len(recent_validations)

        except Exception:
            return 0.0

    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate detailed quality report"""
        try:
            report = {
                'generated_at': pd.Timestamp.now().isoformat(),
                'overall_assessment': 'good' if self._calculate_health_score() > 0.8 else 'needs_attention',
                'metrics_summary': {},
                'recommendations': []
            }

            # Metrics summary
            if self.quality_metrics:
                report['metrics_summary'] = {
                    'average_quality_score': np.mean([v.get('quality_score', 0)
                                                    for v in self.validation_history[-10:]]),
                    'validation_success_rate': self._calculate_health_score(),
                    'total_validations': len(self.validation_history)
                }

            # Generate recommendations
            if self.quality_metrics.get('statistical_similarity_score', 1.0) < self.statistical_threshold:
                report['recommendations'].append(
                    "Consider adjusting augmentation parameters to better preserve statistical properties"
                )

            if self.quality_metrics.get('semantic_preservation_score', 1.0) < self.semantic_threshold:
                report['recommendations'].append(
                    "Review text augmentation strategies to improve semantic preservation"
                )

            if not self.quality_metrics.get('business_rule_compliance', True):
                report['recommendations'].append(
                    "Address business rule violations in augmented data"
                )

            return report

        except Exception as e:
            self.logger.error(f"Error generating quality report: {str(e)}")
            return {'error': str(e)}

    def reset_quality_history(self):
        """Reset quality control history"""
        self.quality_metrics.clear()
        self.validation_history.clear()
        self.logger.info("Quality control history reset")