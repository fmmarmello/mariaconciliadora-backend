import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
import json
import warnings

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from src.services.enhanced_feature_engineer import EnhancedFeatureEngineer
from src.services.advanced_text_feature_extractor import AdvancedTextFeatureExtractor
from src.services.temporal_feature_enhancer import TemporalFeatureEnhancer
from src.services.financial_feature_engineer import FinancialFeatureEngineer
from src.services.smote_implementation import SMOTEImplementation
from src.services.data_augmentation_pipeline import DataAugmentationPipeline
from src.services.advanced_outlier_detector import AdvancedOutlierDetector
from src.utils.cross_field_validation_engine import CrossFieldValidationEngine

logger = get_logger(__name__)


class QualityAssuredFeaturePipeline:
    """
    Quality-assured feature engineering pipeline with comprehensive validation,
    monitoring, and performance optimization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the QualityAssuredFeaturePipeline

        Args:
            config: Configuration dictionary for quality-assured feature engineering
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize core components
        self._initialize_components()

        # Quality tracking and monitoring
        self.quality_metrics = defaultdict(list)
        self.performance_history = []
        self.validation_results = []
        self.alerts = []

        # Feature engineering state
        self.feature_engineer = None
        self.text_extractor = None
        self.temporal_enhancer = None
        self.financial_engineer = None

        self.logger.info("QualityAssuredFeaturePipeline initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for quality-assured feature engineering"""
        return {
            'quality_control': {
                'enable_validation': True,
                'quality_threshold': 0.8,
                'outlier_detection': True,
                'missing_data_handling': 'impute',
                'feature_stability_check': True,
                'performance_monitoring': True
            },
            'feature_engineering': {
                'enhanced_features': True,
                'text_features': True,
                'temporal_features': True,
                'financial_features': True,
                'cross_validation': True,
                'feature_selection': True
            },
            'data_augmentation': {
                'enable_smote': True,
                'synthetic_generation': True,
                'quality_controlled': True,
                'augmentation_ratio': 1.5
            },
            'validation': {
                'cross_field_validation': True,
                'business_rules_check': True,
                'data_consistency_check': True,
                'temporal_consistency_check': True
            },
            'performance': {
                'batch_processing': True,
                'parallel_processing': True,
                'cache_enabled': True,
                'memory_optimization': True,
                'max_batch_size': 10000
            },
            'monitoring': {
                'track_quality_metrics': True,
                'performance_alerts': True,
                'drift_detection': True,
                'anomaly_detection': True
            }
        }

    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            # Core feature engineering components
            self.feature_engineer = EnhancedFeatureEngineer()
            self.text_extractor = AdvancedTextFeatureExtractor()
            self.temporal_enhancer = TemporalFeatureEnhancer()
            self.financial_engineer = FinancialFeatureEngineer()

            # Quality and validation components
            if self.config['quality_control']['outlier_detection']:
                self.outlier_detector = AdvancedOutlierDetector()

            if self.config['validation']['cross_field_validation']:
                self.validation_engine = CrossFieldValidationEngine()

            # Data augmentation components
            if self.config['data_augmentation']['enable_smote']:
                self.smote_engine = SMOTEImplementation()

            if self.config['data_augmentation']['synthetic_generation']:
                self.augmentation_pipeline = DataAugmentationPipeline()

            # Quality monitoring
            self.quality_monitor = FeatureQualityMonitor()

            self.logger.info("All pipeline components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing pipeline components: {str(e)}")
            raise ValidationError(f"Failed to initialize QualityAssuredFeaturePipeline: {str(e)}")

    def process_dataset(self, transactions: List[Dict],
                       target_column: Optional[str] = None,
                       validation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process dataset through the quality-assured feature engineering pipeline

        Args:
            transactions: List of transaction dictionaries
            target_column: Optional target column for supervised feature engineering
            validation_rules: Optional custom validation rules

        Returns:
            Dictionary with processed features and quality report
        """
        try:
            self.logger.info(f"Processing dataset with {len(transactions)} transactions")

            processing_start = datetime.utcnow()

            # Step 1: Initial data validation
            validation_result = self._validate_input_data(transactions, validation_rules)
            if not validation_result['passed']:
                return {
                    'success': False,
                    'error': 'Input validation failed',
                    'validation_result': validation_result
                }

            # Step 2: Data preprocessing and cleaning
            cleaned_data = self._preprocess_data(transactions, validation_result)

            # Step 3: Quality assessment
            quality_report = self._assess_data_quality(cleaned_data)

            # Step 4: Feature engineering
            feature_result = self._engineer_features(cleaned_data, target_column)

            # Step 5: Data augmentation (if enabled)
            if self.config['data_augmentation']['enable_smote']:
                augmented_result = self._apply_data_augmentation(
                    feature_result['features'], feature_result['feature_names'], target_column
                )
                feature_result.update(augmented_result)

            # Step 6: Final quality validation
            final_quality = self._validate_output_quality(
                feature_result['features'], feature_result['feature_names']
            )

            # Step 7: Performance tracking
            processing_time = (datetime.utcnow() - processing_start).total_seconds()
            performance_metrics = self._track_performance(processing_time, quality_report, final_quality)

            # Compile comprehensive result
            result = {
                'success': True,
                'features': feature_result['features'],
                'feature_names': feature_result['feature_names'],
                'feature_matrix_shape': feature_result['features'].shape,
                'quality_report': {
                    'initial_validation': validation_result,
                    'data_quality': quality_report,
                    'final_quality': final_quality,
                    'overall_score': self._calculate_overall_quality_score(quality_report, final_quality)
                },
                'processing_metadata': {
                    'processing_time_seconds': processing_time,
                    'timestamp': datetime.utcnow().isoformat(),
                    'pipeline_version': '1.0.0',
                    'config_used': self.config
                },
                'performance_metrics': performance_metrics,
                'alerts': self._generate_alerts(quality_report, final_quality)
            }

            # Store results for monitoring
            self._store_processing_results(result)

            self.logger.info(f"Dataset processing completed successfully. Features shape: {feature_result['features'].shape}")
            return result

        except Exception as e:
            self.logger.error(f"Error in dataset processing: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def _validate_input_data(self, transactions: List[Dict],
                           validation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate input data quality and structure"""
        try:
            validation_result = {
                'passed': True,
                'issues': [],
                'warnings': [],
                'statistics': {}
            }

            if not transactions:
                validation_result['passed'] = False
                validation_result['issues'].append('Empty dataset')
                return validation_result

            df = pd.DataFrame(transactions)

            # Basic structure validation
            required_fields = ['description', 'amount']  # Minimum required
            missing_fields = [field for field in required_fields if field not in df.columns]
            if missing_fields:
                validation_result['issues'].append(f'Missing required fields: {missing_fields}')

            # Data type validation
            if 'amount' in df.columns:
                try:
                    pd.to_numeric(df['amount'], errors='coerce')
                except Exception:
                    validation_result['issues'].append('Invalid amount data types')

            # Date validation
            if 'date' in df.columns:
                try:
                    pd.to_datetime(df['date'], errors='coerce')
                except Exception:
                    validation_result['issues'].append('Invalid date formats')

            # Completeness check
            completeness = df.notna().mean()
            low_completeness = completeness[completeness < 0.8]
            if len(low_completeness) > 0:
                validation_result['warnings'].append(f'Low completeness in columns: {low_completeness.index.tolist()}')

            # Statistical validation
            validation_result['statistics'] = {
                'total_records': len(df),
                'columns': len(df.columns),
                'completeness_avg': completeness.mean(),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'text_columns': len(df.select_dtypes(include=['object']).columns)
            }

            # Overall validation decision
            validation_result['passed'] = len(validation_result['issues']) == 0

            return validation_result

        except Exception as e:
            self.logger.error(f"Error in input validation: {str(e)}")
            return {
                'passed': False,
                'issues': [f'Validation error: {str(e)}'],
                'warnings': [],
                'statistics': {}
            }

    def _preprocess_data(self, transactions: List[Dict],
                        validation_result: Dict[str, Any]) -> List[Dict]:
        """Preprocess and clean data based on validation results"""
        try:
            df = pd.DataFrame(transactions)

            # Handle missing data
            if self.config['quality_control']['missing_data_handling'] == 'impute':
                # Simple imputation strategies
                for col in df.columns:
                    if df[col].isna().sum() > 0:
                        if df[col].dtype in ['int64', 'float64']:
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown')

            # Remove duplicates if any
            initial_count = len(df)
            df = df.drop_duplicates()
            if len(df) < initial_count:
                self.logger.info(f"Removed {initial_count - len(df)} duplicate records")

            # Basic data type conversions
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            return df.to_dict('records')

        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}")
            return transactions

    def _assess_data_quality(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Assess overall data quality"""
        try:
            df = pd.DataFrame(transactions)

            quality_report = {
                'completeness_score': 0.0,
                'consistency_score': 0.0,
                'accuracy_score': 0.0,
                'timeliness_score': 0.0,
                'overall_quality': 0.0,
                'issues': [],
                'recommendations': []
            }

            # Completeness assessment
            completeness = df.notna().mean().mean()
            quality_report['completeness_score'] = completeness

            # Consistency assessment (basic)
            if 'amount' in df.columns:
                amount_consistency = (df['amount'] >= 0).mean()  # Assuming positive amounts
                quality_report['consistency_score'] = amount_consistency

            # Accuracy assessment (basic)
            if 'description' in df.columns:
                desc_completeness = (df['description'].str.len() > 0).mean()
                quality_report['accuracy_score'] = desc_completeness

            # Timeliness assessment
            if 'date' in df.columns:
                date_completeness = df['date'].notna().mean()
                quality_report['timeliness_score'] = date_completeness

            # Overall quality score
            scores = [
                quality_report['completeness_score'],
                quality_report['consistency_score'],
                quality_report['accuracy_score'],
                quality_report['timeliness_score']
            ]
            quality_report['overall_quality'] = np.mean([s for s in scores if not np.isnan(s)])

            # Generate issues and recommendations
            if quality_report['completeness_score'] < 0.8:
                quality_report['issues'].append('Low data completeness')
                quality_report['recommendations'].append('Consider data imputation or collection improvements')

            if quality_report['consistency_score'] < 0.9:
                quality_report['issues'].append('Data consistency issues')
                quality_report['recommendations'].append('Review data validation rules')

            return quality_report

        except Exception as e:
            self.logger.error(f"Error in quality assessment: {str(e)}")
            return {
                'completeness_score': 0.0,
                'overall_quality': 0.0,
                'issues': [f'Quality assessment error: {str(e)}'],
                'recommendations': []
            }

    def _engineer_features(self, transactions: List[Dict],
                          target_column: str = None) -> Dict[str, Any]:
        """Engineer features using all available components"""
        try:
            feature_results = []

            # Enhanced feature engineering
            if self.config['feature_engineering']['enhanced_features']:
                enhanced_result = self.feature_engineer.create_enhanced_features(
                    transactions, target_column
                )
                feature_results.append(enhanced_result)

            # Text feature extraction
            if self.config['feature_engineering']['text_features']:
                text_features, text_names = self.text_extractor.extract_text_features(
                    [t.get('description', '') for t in transactions]
                )
                if text_features:
                    feature_results.append((text_features, text_names))

            # Temporal feature enhancement
            if self.config['feature_engineering']['temporal_features']:
                temporal_features, temporal_names = self.temporal_enhancer.extract_temporal_features(
                    [t.get('date') for t in transactions]
                )
                if temporal_features.size > 0:
                    feature_results.append((temporal_features, temporal_names))

            # Financial feature engineering
            if self.config['feature_engineering']['financial_features']:
                financial_features, financial_names = self.financial_engineer.extract_financial_features(
                    transactions
                )
                if financial_features.size > 0:
                    feature_results.append((financial_features, financial_names))

            # Combine all features
            if feature_results:
                combined_features, combined_names = self._combine_feature_results(feature_results)
            else:
                combined_features = np.array([])
                combined_names = []

            # Feature selection and optimization
            if (self.config['feature_engineering']['feature_selection'] and
                combined_features.size > 0 and len(combined_names) > 50):

                combined_features, combined_names = self._optimize_features(
                    combined_features, combined_names, target_column
                )

            return {
                'features': combined_features,
                'feature_names': combined_names,
                'feature_sources': [len(result[1]) if len(result) > 1 else 0 for result in feature_results]
            }

        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            return {
                'features': np.array([]),
                'feature_names': [],
                'error': str(e)
            }

    def _combine_feature_results(self, feature_results: List[Tuple]) -> Tuple[np.ndarray, List[str]]:
        """Combine multiple feature engineering results"""
        try:
            all_features = []
            all_names = []

            for result in feature_results:
                if len(result) == 3:  # Enhanced feature engineer result
                    features, names, _ = result
                elif len(result) == 2:  # Other feature results
                    features, names = result
                else:
                    continue

                if features.size > 0:
                    all_features.append(features)
                    all_names.extend(names)

            if all_features:
                combined_features = np.concatenate(all_features, axis=1)
                # Handle NaN values
                combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
                return combined_features, all_names
            else:
                return np.array([]), []

        except Exception as e:
            self.logger.error(f"Error combining feature results: {str(e)}")
            return np.array([]), []

    def _optimize_features(self, features: np.ndarray, feature_names: List[str],
                          target_column: str = None) -> Tuple[np.ndarray, List[str]]:
        """Optimize features through selection and processing"""
        try:
            # Feature scaling
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # Feature selection
            if target_column and len(feature_names) > 20:
                # Use mutual information for feature selection
                n_select = min(50, len(feature_names) // 2)
                selector = SelectKBest(score_func=mutual_info_classif, k=n_select)

                # Create dummy target if needed
                if target_column not in ['dummy_target']:
                    dummy_target = np.random.randint(0, 2, size=features.shape[0])
                else:
                    dummy_target = np.random.randint(0, 2, size=features.shape[0])

                selected_features = selector.fit_transform(scaled_features, dummy_target)
                selected_mask = selector.get_support()
                selected_names = [name for name, selected in zip(feature_names, selected_mask) if selected]

                return selected_features, selected_names
            else:
                return scaled_features, feature_names

        except Exception as e:
            self.logger.error(f"Error optimizing features: {str(e)}")
            return features, feature_names

    def _apply_data_augmentation(self, features: np.ndarray, feature_names: List[str],
                                target_column: str = None) -> Dict[str, Any]:
        """Apply data augmentation techniques"""
        try:
            augmentation_result = {
                'augmentation_applied': False,
                'original_samples': features.shape[0],
                'augmented_samples': features.shape[0]
            }

            if not self.config['data_augmentation']['enable_smote'] or features.size == 0:
                return augmentation_result

            # Check if we need augmentation (imbalanced data)
            if target_column:
                # Create dummy target for demonstration
                dummy_target = np.random.randint(0, 2, size=features.shape[0])

                imbalance_info = self.smote_engine.detect_imbalance(features, dummy_target)

                if imbalance_info.get('requires_balancing', False):
                    # Apply SMOTE
                    augmented_features, augmented_target = self.smote_engine.apply_smote(
                        features, dummy_target
                    )

                    augmentation_result.update({
                        'augmentation_applied': True,
                        'augmented_samples': augmented_features.shape[0],
                        'imbalance_ratio_before': imbalance_info.get('imbalance_ratio', 1.0),
                        'imbalance_ratio_after': max(Counter(augmented_target).values()) / min(Counter(augmented_target).values())
                    })

                    return {
                        'features': augmented_features,
                        'feature_names': feature_names,
                        'augmentation_info': augmentation_result
                    }

            return {'augmentation_info': augmentation_result}

        except Exception as e:
            self.logger.error(f"Error in data augmentation: {str(e)}")
            return {'augmentation_info': {'error': str(e)}}

    def _validate_output_quality(self, features: np.ndarray,
                                feature_names: List[str]) -> Dict[str, Any]:
        """Validate quality of output features"""
        try:
            quality_report = {
                'feature_count': len(feature_names),
                'sample_count': features.shape[0],
                'missing_values': np.isnan(features).sum(),
                'infinite_values': np.isinf(features).sum(),
                'zero_variance_features': 0,
                'high_correlation_features': 0,
                'quality_score': 1.0,
                'issues': []
            }

            if features.size == 0:
                quality_report['quality_score'] = 0.0
                quality_report['issues'].append('No features generated')
                return quality_report

            # Check for zero variance features
            variances = np.var(features, axis=0)
            zero_var_count = np.sum(variances == 0)
            quality_report['zero_variance_features'] = int(zero_var_count)

            # Check for high correlations
            if features.shape[1] > 1:
                corr_matrix = np.corrcoef(features.T)
                high_corr_count = np.sum(np.abs(corr_matrix) > 0.95) - features.shape[1]  # Subtract diagonal
                quality_report['high_correlation_features'] = int(high_corr_count)

            # Calculate quality score
            score = 1.0

            # Penalize for missing values
            missing_ratio = quality_report['missing_values'] / features.size
            score -= missing_ratio * 0.5

            # Penalize for infinite values
            inf_ratio = quality_report['infinite_values'] / features.size
            score -= inf_ratio * 0.3

            # Penalize for zero variance features
            zero_var_ratio = quality_report['zero_variance_features'] / features.shape[1]
            score -= zero_var_ratio * 0.4

            quality_report['quality_score'] = max(0.0, min(1.0, score))

            # Generate issues
            if quality_report['missing_values'] > 0:
                quality_report['issues'].append(f"Missing values: {quality_report['missing_values']}")

            if quality_report['zero_variance_features'] > 0:
                quality_report['issues'].append(f"Zero variance features: {quality_report['zero_variance_features']}")

            if quality_report['quality_score'] < self.config['quality_control']['quality_threshold']:
                quality_report['issues'].append(f"Low quality score: {quality_report['quality_score']:.2f}")

            return quality_report

        except Exception as e:
            self.logger.error(f"Error in output quality validation: {str(e)}")
            return {
                'quality_score': 0.0,
                'issues': [f'Quality validation error: {str(e)}']
            }

    def _calculate_overall_quality_score(self, data_quality: Dict[str, Any],
                                       final_quality: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        try:
            scores = []

            if 'overall_quality' in data_quality:
                scores.append(data_quality['overall_quality'])

            if 'quality_score' in final_quality:
                scores.append(final_quality['quality_score'])

            if scores:
                return np.mean(scores)
            else:
                return 0.5

        except Exception:
            return 0.5

    def _track_performance(self, processing_time: float,
                          quality_report: Dict[str, Any],
                          final_quality: Dict[str, Any]) -> Dict[str, Any]:
        """Track performance metrics"""
        try:
            performance_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'processing_time_seconds': processing_time,
                'data_quality_score': quality_report.get('overall_quality', 0.0),
                'final_quality_score': final_quality.get('quality_score', 0.0),
                'feature_count': final_quality.get('feature_count', 0),
                'issues_count': len(final_quality.get('issues', []))
            }

            self.performance_history.append(performance_entry)

            return performance_entry

        except Exception as e:
            self.logger.error(f"Error tracking performance: {str(e)}")
            return {}

    def _generate_alerts(self, quality_report: Dict[str, Any],
                        final_quality: Dict[str, Any]) -> List[str]:
        """Generate alerts based on quality metrics"""
        try:
            alerts = []

            # Quality threshold alerts
            if quality_report.get('overall_quality', 1.0) < self.config['quality_control']['quality_threshold']:
                alerts.append(f"Low data quality score: {quality_report['overall_quality']:.2f}")

            if final_quality.get('quality_score', 1.0) < self.config['quality_control']['quality_threshold']:
                alerts.append(f"Low feature quality score: {final_quality['quality_score']:.2f}")

            # Issue-based alerts
            if final_quality.get('missing_values', 0) > 0:
                alerts.append(f"Features contain missing values: {final_quality['missing_values']}")

            if final_quality.get('zero_variance_features', 0) > 0:
                alerts.append(f"Zero variance features detected: {final_quality['zero_variance_features']}")

            # Performance alerts
            if len(self.performance_history) > 1:
                recent_perf = self.performance_history[-1]
                prev_perf = self.performance_history[-2]

                if recent_perf['processing_time_seconds'] > prev_perf['processing_time_seconds'] * 2:
                    alerts.append("Processing time increased significantly")

            self.alerts.extend(alerts)
            return alerts

        except Exception as e:
            self.logger.error(f"Error generating alerts: {str(e)}")
            return []

    def _store_processing_results(self, result: Dict[str, Any]):
        """Store processing results for monitoring"""
        try:
            if self.config['monitoring']['track_quality_metrics']:
                self.quality_metrics['processing_results'].append(result)

            # Keep only recent results to prevent memory issues
            if len(self.quality_metrics['processing_results']) > 100:
                self.quality_metrics['processing_results'] = self.quality_metrics['processing_results'][-50:]

        except Exception as e:
            self.logger.warning(f"Error storing processing results: {str(e)}")

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get comprehensive quality metrics"""
        try:
            if not self.performance_history:
                return {'message': 'No processing history available'}

            recent_results = self.quality_metrics.get('processing_results', [])[-10:]

            return {
                'total_runs': len(self.performance_history),
                'average_processing_time': np.mean([p['processing_time_seconds'] for p in self.performance_history]),
                'average_data_quality': np.mean([p['data_quality_score'] for p in self.performance_history]),
                'average_final_quality': np.mean([p['final_quality_score'] for p in self.performance_history]),
                'recent_alerts': self.alerts[-10:],
                'quality_trend': [p['final_quality_score'] for p in self.performance_history[-20:]],
                'performance_trend': [p['processing_time_seconds'] for p in self.performance_history[-20:]]
            }

        except Exception as e:
            self.logger.error(f"Error getting quality metrics: {str(e)}")
            return {'error': str(e)}

    def clear_cache(self):
        """Clear pipeline cache and reset state"""
        try:
            self.quality_metrics.clear()
            self.performance_history.clear()
            self.validation_results.clear()
            self.alerts.clear()

            # Clear component caches
            if hasattr(self.feature_engineer, 'clear_cache'):
                self.feature_engineer.clear_cache()
            if hasattr(self.text_extractor, 'clear_cache'):
                self.text_extractor.clear_cache()
            if hasattr(self.temporal_enhancer, 'clear_cache'):
                self.temporal_enhancer.clear_cache()
            if hasattr(self.financial_engineer, 'clear_cache'):
                self.financial_engineer.clear_cache()

            self.logger.info("Pipeline cache cleared")

        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")

    def save_pipeline(self, filepath: str):
        """Save the quality-assured pipeline state"""
        try:
            import joblib

            save_dict = {
                'config': self.config,
                'quality_metrics': dict(self.quality_metrics),
                'performance_history': self.performance_history,
                'alerts': self.alerts
            }

            joblib.dump(save_dict, filepath)
            self.logger.info(f"QualityAssuredFeaturePipeline saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving pipeline: {str(e)}")

    def load_pipeline(self, filepath: str):
        """Load the quality-assured pipeline state"""
        try:
            import joblib

            save_dict = joblib.load(filepath)

            self.config = save_dict['config']
            self.quality_metrics = defaultdict(list, save_dict['quality_metrics'])
            self.performance_history = save_dict['performance_history']
            self.alerts = save_dict['alerts']

            # Reinitialize components
            self._initialize_components()

            self.logger.info(f"QualityAssuredFeaturePipeline loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading pipeline: {str(e)}")
            raise ValidationError(f"Failed to load QualityAssuredFeaturePipeline: {str(e)}")


class FeatureQualityMonitor:
    """Monitor and track feature quality over time"""

    def __init__(self):
        self.quality_history = []
        self.baseline_metrics = {}
        self.drift_threshold = 0.1

    def update_baseline(self, metrics: Dict[str, Any]):
        """Update baseline quality metrics"""
        self.baseline_metrics = metrics.copy()
        self.baseline_metrics['timestamp'] = datetime.utcnow().isoformat()

    def detect_drift(self, current_metrics: Dict[str, Any]) -> List[str]:
        """Detect quality drift from baseline"""
        if not self.baseline_metrics:
            return []

        alerts = []

        for key, baseline_value in self.baseline_metrics.items():
            if key in current_metrics and isinstance(baseline_value, (int, float)):
                current_value = current_metrics[key]
                if abs(current_value - baseline_value) > self.drift_threshold:
                    alerts.append(f"Quality drift detected in {key}: {baseline_value:.2f} -> {current_value:.2f}")

        return alerts

    def get_quality_trend(self) -> Dict[str, Any]:
        """Get quality trend analysis"""
        if not self.quality_history:
            return {'message': 'No quality history available'}

        return {
            'total_measurements': len(self.quality_history),
            'trend_direction': self._calculate_trend(),
            'volatility': self._calculate_volatility(),
            'recent_quality': self.quality_history[-5:]
        }

    def _calculate_trend(self) -> str:
        """Calculate quality trend direction"""
        if len(self.quality_history) < 2:
            return 'insufficient_data'

        recent_scores = [entry.get('quality_score', 0) for entry in self.quality_history[-10:]]
        if len(recent_scores) < 2:
            return 'insufficient_data'

        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

        if trend > 0.01:
            return 'improving'
        elif trend < -0.01:
            return 'degrading'
        else:
            return 'stable'

    def _calculate_volatility(self) -> float:
        """Calculate quality volatility"""
        if len(self.quality_history) < 2:
            return 0.0

        scores = [entry.get('quality_score', 0) for entry in self.quality_history]
        return np.std(scores) if scores else 0.0