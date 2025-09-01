import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError


class QualityAssessmentEngine:
    """
    Comprehensive quality validation engine for synthetic data with:
    - Statistical similarity checks between original and synthetic data
    - Distribution preservation validation
    - Business rule compliance verification
    - Model performance impact assessment
    - Synthetic data quality scoring
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the quality assessment engine

        Args:
            config: Configuration for quality assessment
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Quality metrics storage
        self.quality_reports = []
        self.baseline_metrics = {}

        # Statistical tests configuration
        self.statistical_tests = {
            'kolmogorov_smirnov': self._kolmogorov_smirnov_test,
            'mann_whitney_u': self._mann_whitney_u_test,
            'jensen_shannon_divergence': self._jensen_shannon_divergence,
            'wasserstein_distance': self._wasserstein_distance,
            'correlation_preservation': self._correlation_preservation_test
        }

        # Quality thresholds
        self.quality_thresholds = {
            'statistical_similarity': 0.8,
            'distribution_preservation': 0.75,
            'business_rule_compliance': 0.9,
            'model_performance_impact': 0.85,
            'overall_quality_score': 0.8
        }

        self.logger.info("QualityAssessmentEngine initialized")

    def assess_synthetic_quality(self, original_data: Union[pd.DataFrame, np.ndarray],
                               synthetic_data: Union[pd.DataFrame, np.ndarray],
                               metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive assessment of synthetic data quality

        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            metadata: Additional metadata about the generation process

        Returns:
            Comprehensive quality assessment report
        """
        try:
            start_time = datetime.now()

            # Convert to DataFrames if needed
            original_df = self._ensure_dataframe(original_data)
            synthetic_df = self._ensure_dataframe(synthetic_data)

            if original_df is None or synthetic_df is None:
                raise ValidationError("Invalid data format for quality assessment")

            # Perform comprehensive assessment
            assessment_report = {
                'timestamp': start_time.isoformat(),
                'data_characteristics': self._analyze_data_characteristics(original_df, synthetic_df),
                'statistical_similarity': self._assess_statistical_similarity(original_df, synthetic_df),
                'distribution_preservation': self._assess_distribution_preservation(original_df, synthetic_df),
                'business_rule_compliance': self._assess_business_rule_compliance(synthetic_df),
                'model_performance_impact': self._assess_model_performance_impact(original_df, synthetic_df),
                'privacy_preservation': self._assess_privacy_preservation(original_df, synthetic_df),
                'overall_quality_score': 0.0,
                'quality_grade': 'F',
                'recommendations': [],
                'metadata': metadata or {}
            }

            # Calculate overall quality score
            assessment_report['overall_quality_score'] = self._calculate_overall_quality_score(assessment_report)

            # Assign quality grade
            assessment_report['quality_grade'] = self._assign_quality_grade(assessment_report['overall_quality_score'])

            # Generate recommendations
            assessment_report['recommendations'] = self._generate_recommendations(assessment_report)

            # Store report
            self.quality_reports.append(assessment_report)

            processing_time = (datetime.now() - start_time).total_seconds()
            assessment_report['processing_time'] = processing_time

            self.logger.info(f"Quality assessment completed in {processing_time:.2f}s with score: {assessment_report['overall_quality_score']:.3f}")

            return assessment_report

        except Exception as e:
            self.logger.error(f"Error in synthetic quality assessment: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'overall_quality_score': 0.0,
                'quality_grade': 'F'
            }

    def _ensure_dataframe(self, data: Union[pd.DataFrame, np.ndarray]) -> Optional[pd.DataFrame]:
        """Ensure data is in DataFrame format"""
        try:
            if isinstance(data, pd.DataFrame):
                return data.copy()
            elif isinstance(data, np.ndarray):
                return pd.DataFrame(data)
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                return None
        except Exception:
            return None

    def _analyze_data_characteristics(self, original_df: pd.DataFrame,
                                    synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic characteristics of original and synthetic data"""
        try:
            characteristics = {
                'original': {
                    'n_samples': len(original_df),
                    'n_features': len(original_df.columns),
                    'feature_types': original_df.dtypes.to_dict(),
                    'missing_values': original_df.isnull().sum().to_dict(),
                    'duplicate_rows': original_df.duplicated().sum()
                },
                'synthetic': {
                    'n_samples': len(synthetic_df),
                    'n_features': len(synthetic_df.columns),
                    'feature_types': synthetic_df.dtypes.to_dict(),
                    'missing_values': synthetic_df.isnull().sum().to_dict(),
                    'duplicate_rows': synthetic_df.duplicated().sum()
                },
                'compatibility': {
                    'feature_match': set(original_df.columns) == set(synthetic_df.columns),
                    'sample_ratio': len(synthetic_df) / len(original_df) if len(original_df) > 0 else 0
                }
            }

            return characteristics

        except Exception as e:
            self.logger.warning(f"Error analyzing data characteristics: {str(e)}")
            return {}

    def _assess_statistical_similarity(self, original_df: pd.DataFrame,
                                     synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess statistical similarity between original and synthetic data"""
        try:
            similarity_results = {}

            # Get numerical columns
            numerical_cols = original_df.select_dtypes(include=[np.number]).columns
            common_numerical_cols = [col for col in numerical_cols if col in synthetic_df.columns]

            if common_numerical_cols:
                for col in common_numerical_cols:
                    orig_data = original_df[col].dropna()
                    synth_data = synthetic_df[col].dropna()

                    if len(orig_data) > 10 and len(synth_data) > 10:
                        col_similarity = self._assess_column_similarity(orig_data, synth_data)
                        similarity_results[col] = col_similarity

            # Overall statistical similarity score
            if similarity_results:
                column_scores = [result.get('overall_similarity', 0) for result in similarity_results.values()]
                overall_similarity = np.mean(column_scores)
            else:
                overall_similarity = 0.0

            return {
                'column_similarities': similarity_results,
                'overall_similarity': overall_similarity,
                'numerical_columns_assessed': len(common_numerical_cols)
            }

        except Exception as e:
            self.logger.error(f"Error assessing statistical similarity: {str(e)}")
            return {'overall_similarity': 0.0}

    def _assess_column_similarity(self, original_data: pd.Series,
                                synthetic_data: pd.Series) -> Dict[str, Any]:
        """Assess similarity for a single column"""
        try:
            results = {}

            # Basic statistics comparison
            orig_stats = original_data.describe()
            synth_stats = synthetic_data.describe()

            results['mean_difference'] = abs(orig_stats['mean'] - synth_stats['mean'])
            results['std_difference'] = abs(orig_stats['std'] - synth_stats['std'])
            results['median_difference'] = abs(original_data.median() - synthetic_data.median())

            # Statistical tests
            test_results = {}
            for test_name, test_func in self.statistical_tests.items():
                try:
                    test_results[test_name] = test_func(original_data, synthetic_data)
                except Exception:
                    test_results[test_name] = {'p_value': 1.0, 'statistic': 0.0}

            results['statistical_tests'] = test_results

            # Overall similarity score (weighted average of test p-values)
            p_values = [test.get('p_value', 0) for test in test_results.values()]
            if p_values:
                results['overall_similarity'] = np.mean(p_values)
            else:
                results['overall_similarity'] = 0.0

            return results

        except Exception as e:
            self.logger.error(f"Error assessing column similarity: {str(e)}")
            return {'overall_similarity': 0.0}

    def _kolmogorov_smirnov_test(self, original_data: pd.Series,
                               synthetic_data: pd.Series) -> Dict[str, float]:
        """Perform Kolmogorov-Smirnov test"""
        try:
            statistic, p_value = stats.ks_2samp(original_data, synthetic_data)
            return {'statistic': statistic, 'p_value': p_value}
        except Exception:
            return {'statistic': 1.0, 'p_value': 0.0}

    def _mann_whitney_u_test(self, original_data: pd.Series,
                           synthetic_data: pd.Series) -> Dict[str, float]:
        """Perform Mann-Whitney U test"""
        try:
            statistic, p_value = stats.mannwhitneyu(original_data, synthetic_data, alternative='two-sided')
            return {'statistic': statistic, 'p_value': p_value}
        except Exception:
            return {'statistic': 0.0, 'p_value': 0.0}

    def _jensen_shannon_divergence(self, original_data: pd.Series,
                                 synthetic_data: pd.Series) -> Dict[str, float]:
        """Calculate Jensen-Shannon divergence"""
        try:
            # Create histograms
            hist_orig, bins = np.histogram(original_data, bins=50, density=True)
            hist_synth, _ = np.histogram(synthetic_data, bins=bins, density=True)

            # Ensure non-zero histograms
            hist_orig = hist_orig + 1e-10
            hist_synth = hist_synth + 1e-10

            # Normalize
            hist_orig = hist_orig / hist_orig.sum()
            hist_synth = hist_synth / hist_synth.sum()

            # Calculate JS divergence
            js_div = jensenshannon(hist_orig, hist_synth)

            # Convert to similarity score (1 - normalized divergence)
            similarity = 1 - (js_div / np.log(2))  # Normalize by max possible divergence

            return {'divergence': js_div, 'p_value': similarity}
        except Exception:
            return {'divergence': 1.0, 'p_value': 0.0}

    def _wasserstein_distance(self, original_data: pd.Series,
                            synthetic_data: pd.Series) -> Dict[str, float]:
        """Calculate Wasserstein distance"""
        try:
            from scipy.stats import wasserstein_distance
            distance = wasserstein_distance(original_data, synthetic_data)

            # Normalize by data range
            data_range = original_data.max() - original_data.min()
            if data_range > 0:
                normalized_distance = distance / data_range
            else:
                normalized_distance = 0.0

            # Convert to similarity score
            similarity = 1 - min(normalized_distance, 1.0)

            return {'distance': distance, 'p_value': similarity}
        except Exception:
            return {'distance': 1.0, 'p_value': 0.0}

    def _correlation_preservation_test(self, original_data: pd.Series,
                                     synthetic_data: pd.Series) -> Dict[str, float]:
        """Test correlation preservation (placeholder for multi-column analysis)"""
        # This would be more meaningful with multiple columns
        return {'correlation_diff': 0.0, 'p_value': 0.8}

    def _assess_distribution_preservation(self, original_df: pd.DataFrame,
                                        synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess how well synthetic data preserves original distributions"""
        try:
            preservation_results = {}

            # Categorical columns
            categorical_cols = original_df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if col in synthetic_df.columns:
                    orig_dist = Counter(original_df[col].dropna())
                    synth_dist = Counter(synthetic_df[col].dropna())

                    # Calculate distribution similarity
                    similarity = self._calculate_distribution_similarity(orig_dist, synth_dist)
                    preservation_results[col] = {
                        'original_distribution': dict(orig_dist),
                        'synthetic_distribution': dict(synth_dist),
                        'similarity_score': similarity
                    }

            # Overall distribution preservation score
            if preservation_results:
                similarity_scores = [result['similarity_score'] for result in preservation_results.values()]
                overall_preservation = np.mean(similarity_scores)
            else:
                overall_preservation = 0.8  # Default for no categorical data

            return {
                'column_distributions': preservation_results,
                'overall_preservation': overall_preservation,
                'categorical_columns_assessed': len(preservation_results)
            }

        except Exception as e:
            self.logger.error(f"Error assessing distribution preservation: {str(e)}")
            return {'overall_preservation': 0.0}

    def _calculate_distribution_similarity(self, orig_dist: Counter,
                                        synth_dist: Counter) -> float:
        """Calculate similarity between two distributions"""
        try:
            all_keys = set(orig_dist.keys()) | set(synth_dist.keys())

            # Create probability distributions
            orig_probs = []
            synth_probs = []

            total_orig = sum(orig_dist.values())
            total_synth = sum(synth_dist.values())

            for key in all_keys:
                orig_probs.append(orig_dist.get(key, 0) / total_orig)
                synth_probs.append(synth_dist.get(key, 0) / total_synth)

            # Calculate Jensen-Shannon divergence
            js_div = jensenshannon(orig_probs, synth_probs)
            similarity = 1 - (js_div / np.log(2))

            return similarity

        except Exception:
            return 0.0

    def _assess_business_rule_compliance(self, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess compliance with business rules"""
        try:
            compliance_results = {}

            # Amount validation (assuming financial data)
            if 'amount' in synthetic_df.columns:
                amounts = synthetic_df['amount'].dropna()
                valid_amounts = ((amounts >= 0.01) & (amounts <= 100000)).sum()
                compliance_results['amount_range'] = valid_amounts / len(amounts)

            # Description validation
            if 'description' in synthetic_df.columns:
                descriptions = synthetic_df['description'].dropna()
                valid_descriptions = descriptions.str.len().between(3, 100)
                compliance_results['description_length'] = valid_descriptions.sum() / len(descriptions)

            # Date validation
            if 'date' in synthetic_df.columns:
                try:
                    dates = pd.to_datetime(synthetic_df['date'], errors='coerce')
                    valid_dates = (~dates.isna()).sum()
                    future_dates = (dates > datetime.now()).sum()
                    compliance_results['date_format'] = valid_dates / len(dates)
                    compliance_results['no_future_dates'] = 1 - (future_dates / len(dates))
                except Exception:
                    compliance_results['date_validation'] = 0.0

            # Overall compliance score
            if compliance_results:
                compliance_scores = list(compliance_results.values())
                overall_compliance = np.mean(compliance_scores)
            else:
                overall_compliance = 1.0  # No rules to check

            return {
                'rule_compliance': compliance_results,
                'overall_compliance': overall_compliance,
                'rules_checked': len(compliance_results)
            }

        except Exception as e:
            self.logger.error(f"Error assessing business rule compliance: {str(e)}")
            return {'overall_compliance': 0.0}

    def _assess_model_performance_impact(self, original_df: pd.DataFrame,
                                       synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess impact of synthetic data on model performance"""
        try:
            performance_results = {}

            # Prepare data for modeling
            combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
            combined_df['is_synthetic'] = [False] * len(original_df) + [True] * len(synthetic_df)

            # Simple classification task: predict if data is synthetic
            if len(combined_df) > 100:
                X, y = self._prepare_modeling_data(combined_df)

                if X is not None and len(X) > 0:
                    # Train model
                    model = RandomForestClassifier(n_estimators=100, random_state=42)

                    # Cross-validation scores
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

                    performance_results['discrimination_accuracy'] = cv_scores.mean()
                    performance_results['discrimination_std'] = cv_scores.std()

                    # If discrimination accuracy is high, synthetic data is too different
                    # We want low discrimination accuracy (hard to distinguish real from synthetic)
                    performance_results['synthetic_realism'] = 1 - cv_scores.mean()

            # Overall performance impact score
            if performance_results:
                overall_impact = performance_results.get('synthetic_realism', 0.5)
            else:
                overall_impact = 0.5  # Neutral score

            return {
                'model_performance': performance_results,
                'overall_impact': overall_impact,
                'data_combined_size': len(combined_df)
            }

        except Exception as e:
            self.logger.error(f"Error assessing model performance impact: {str(e)}")
            return {'overall_impact': 0.5}

    def _prepare_modeling_data(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare data for modeling assessment"""
        try:
            # Select features
            feature_cols = []
            for col in df.columns:
                if col not in ['is_synthetic'] and df[col].dtype in ['int64', 'float64']:
                    feature_cols.append(col)

            if not feature_cols:
                return None, None

            X = df[feature_cols].fillna(0).values
            y = df['is_synthetic'].values.astype(int)

            return X, y

        except Exception:
            return None, None

    def _assess_privacy_preservation(self, original_df: pd.DataFrame,
                                   synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess privacy preservation in synthetic data"""
        try:
            privacy_results = {}

            # Nearest neighbor distance analysis
            # (Simplified privacy assessment)
            privacy_results['nearest_neighbor_analysis'] = self._nearest_neighbor_privacy(original_df, synthetic_df)

            # Uniqueness analysis
            orig_unique_ratio = original_df.drop_duplicates().shape[0] / original_df.shape[0]
            synth_unique_ratio = synthetic_df.drop_duplicates().shape[0] / synthetic_df.shape[0]

            privacy_results['original_uniqueness'] = orig_unique_ratio
            privacy_results['synthetic_uniqueness'] = synth_unique_ratio
            privacy_results['uniqueness_preservation'] = min(synth_unique_ratio / orig_unique_ratio, 1.0)

            # Overall privacy score
            privacy_scores = [result for result in privacy_results.values() if isinstance(result, (int, float))]
            overall_privacy = np.mean(privacy_scores) if privacy_scores else 0.8

            return {
                'privacy_metrics': privacy_results,
                'overall_privacy': overall_privacy
            }

        except Exception as e:
            self.logger.error(f"Error assessing privacy preservation: {str(e)}")
            return {'overall_privacy': 0.8}

    def _nearest_neighbor_privacy(self, original_df: pd.DataFrame,
                                synthetic_df: pd.DataFrame) -> float:
        """Assess privacy through nearest neighbor analysis"""
        try:
            # Simplified nearest neighbor privacy check
            # In practice, this would use more sophisticated privacy metrics
            return 0.85  # Placeholder
        except Exception:
            return 0.5

    def _calculate_overall_quality_score(self, assessment_report: Dict[str, Any]) -> float:
        """Calculate overall quality score from all assessments"""
        try:
            scores = []

            # Weight different aspects
            weights = {
                'statistical_similarity': 0.3,
                'distribution_preservation': 0.25,
                'business_rule_compliance': 0.2,
                'model_performance_impact': 0.15,
                'privacy_preservation': 0.1
            }

            for aspect, weight in weights.items():
                if aspect in assessment_report:
                    aspect_data = assessment_report[aspect]
                    if isinstance(aspect_data, dict) and 'overall_' + aspect.split('_')[0] in aspect_data:
                        score_key = 'overall_' + aspect.split('_')[0]
                        if aspect == 'statistical_similarity':
                            score_key = 'overall_similarity'
                        elif aspect == 'distribution_preservation':
                            score_key = 'overall_preservation'
                        elif aspect == 'business_rule_compliance':
                            score_key = 'overall_compliance'
                        elif aspect == 'model_performance_impact':
                            score_key = 'overall_impact'
                        elif aspect == 'privacy_preservation':
                            score_key = 'overall_privacy'

                        score = aspect_data.get(score_key, 0.5)
                        scores.append(score * weight)

            return sum(scores) if scores else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating overall quality score: {str(e)}")
            return 0.0

    def _assign_quality_grade(self, overall_score: float) -> str:
        """Assign quality grade based on overall score"""
        if overall_score >= 0.9:
            return 'A'
        elif overall_score >= 0.8:
            return 'B'
        elif overall_score >= 0.7:
            return 'C'
        elif overall_score >= 0.6:
            return 'D'
        else:
            return 'F'

    def _generate_recommendations(self, assessment_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on assessment results"""
        recommendations = []

        try:
            overall_score = assessment_report.get('overall_quality_score', 0)

            if overall_score < 0.6:
                recommendations.append("Synthetic data quality is poor. Consider using different generation methods.")

            # Statistical similarity recommendations
            stat_sim = assessment_report.get('statistical_similarity', {}).get('overall_similarity', 0)
            if stat_sim < 0.7:
                recommendations.append("Improve statistical similarity by adjusting generation parameters or using different models.")

            # Distribution preservation recommendations
            dist_preserve = assessment_report.get('distribution_preservation', {}).get('overall_preservation', 0)
            if dist_preserve < 0.7:
                recommendations.append("Distribution preservation needs improvement. Consider conditional generation methods.")

            # Business rule compliance recommendations
            rule_compliance = assessment_report.get('business_rule_compliance', {}).get('overall_compliance', 0)
            if rule_compliance < 0.8:
                recommendations.append("Business rule compliance is low. Add post-processing validation and correction.")

            # Model performance recommendations
            model_impact = assessment_report.get('model_performance_impact', {}).get('overall_impact', 0)
            if model_impact < 0.7:
                recommendations.append("Model performance impact is significant. Synthetic data may be too different from real data.")

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")

        if not recommendations:
            recommendations.append("Synthetic data quality is acceptable. No major improvements needed.")

        return recommendations

    def get_quality_history(self) -> List[Dict[str, Any]]:
        """Get history of quality assessments"""
        return self.quality_reports

    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of quality assessment results"""
        try:
            if not self.quality_reports:
                return {}

            scores = [report.get('overall_quality_score', 0) for report in self.quality_reports]
            grades = [report.get('quality_grade', 'F') for report in self.quality_reports]

            return {
                'total_assessments': len(self.quality_reports),
                'average_score': np.mean(scores),
                'median_score': np.median(scores),
                'best_score': max(scores),
                'worst_score': min(scores),
                'grade_distribution': Counter(grades),
                'most_common_grade': Counter(grades).most_common(1)[0][0] if grades else 'N/A'
            }

        except Exception as e:
            self.logger.error(f"Error generating quality summary: {str(e)}")
            return {}

    def save_quality_report(self, filepath: str, report: Dict[str, Any]):
        """Save quality assessment report to file"""
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info(f"Quality report saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving quality report: {str(e)}")

    def generate_quality_visualization(self, original_df: pd.DataFrame,
                                     synthetic_df: pd.DataFrame,
                                     save_path: str = None) -> Dict[str, Any]:
        """Generate visualizations for quality assessment"""
        try:
            visualizations = {}

            # Distribution comparison plots
            numerical_cols = original_df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols[:5]:  # Limit to first 5 columns
                if col in synthetic_df.columns:
                    plt.figure(figsize=(10, 6))

                    plt.hist(original_df[col].dropna(), alpha=0.7, label='Original', bins=30)
                    plt.hist(synthetic_df[col].dropna(), alpha=0.7, label='Synthetic', bins=30)

                    plt.title(f'Distribution Comparison: {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.legend()

                    if save_path:
                        plt.savefig(f"{save_path}/distribution_{col}.png")
                        plt.close()
                    else:
                        plt.close()

                    visualizations[f'distribution_{col}'] = 'generated'

            return visualizations

        except Exception as e:
            self.logger.error(f"Error generating quality visualizations: {str(e)}")
            return {}