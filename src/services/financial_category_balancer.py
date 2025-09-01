import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re
import random

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from .smote_implementation import SMOTEImplementation
from .synthetic_data_generator import AdvancedSyntheticDataGenerator


class FinancialCategoryBalancer:
    """
    Domain-specific balancer for financial transaction categories with:
    - Category-specific SMOTE parameters
    - Financial transaction pattern preservation
    - Amount distribution maintenance
    - Temporal pattern preservation
    - Business rule compliance in synthetic data
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the financial category balancer

        Args:
            config: Configuration for financial balancing
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize components
        self.smote_handler = SMOTEImplementation(config.get('smote_config', {}))
        self.synthetic_generator = AdvancedSyntheticDataGenerator(config.get('synthetic_config', {}))

        # Financial domain knowledge
        self.category_patterns = self._load_category_patterns()
        self.business_rules = self._load_business_rules()
        self.temporal_patterns = {}

        # Scalers and encoders
        self.amount_scaler = StandardScaler()
        self.category_encoder = LabelEncoder()

        # Pattern preservation data
        self.category_stats = {}
        self.temporal_stats = {}
        self.amount_distributions = {}

        self.logger.info("FinancialCategoryBalancer initialized")

    def _load_category_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load financial category patterns and characteristics"""
        return {
            'receita': {
                'typical_amount_range': (0.01, 10000.0),
                'common_descriptions': ['salario', 'freelance', 'dividendos', 'aluguel', 'investimento'],
                'temporal_patterns': ['monthly', 'weekly', 'irregular'],
                'amount_distribution': 'right_skewed'
            },
            'despesa': {
                'typical_amount_range': (0.01, 5000.0),
                'common_descriptions': ['mercado', 'transporte', 'saude', 'educacao', 'lazer'],
                'temporal_patterns': ['weekly', 'monthly', 'daily'],
                'amount_distribution': 'right_skewed'
            },
            'transferencia': {
                'typical_amount_range': (0.01, 10000.0),
                'common_descriptions': ['pix', 'ted', 'doc', 'transferencia'],
                'temporal_patterns': ['irregular', 'monthly'],
                'amount_distribution': 'normal'
            },
            'investimento': {
                'typical_amount_range': (0.01, 50000.0),
                'common_descriptions': ['acao', 'fundo', 'cdb', 'poupanca', 'tesouro'],
                'temporal_patterns': ['monthly', 'quarterly', 'irregular'],
                'amount_distribution': 'left_skewed'
            },
            'outros': {
                'typical_amount_range': (0.01, 1000.0),
                'common_descriptions': ['taxa', 'juros', 'estorno', 'ajuste'],
                'temporal_patterns': ['irregular'],
                'amount_distribution': 'uniform'
            }
        }

    def _load_business_rules(self) -> Dict[str, Any]:
        """Load financial business rules for synthetic data validation"""
        return {
            'amount_constraints': {
                'minimum_amount': 0.01,
                'maximum_amount': 100000.0,
                'decimal_precision': 2
            },
            'temporal_constraints': {
                'future_dates_not_allowed': True,
                'reasonable_date_range': (-365, 0),  # days from today
                'business_days_only': False
            },
            'description_constraints': {
                'minimum_length': 3,
                'maximum_length': 100,
                'allowed_characters': r'^[a-zA-Z0-9\s\.\-\,\+\(\)\[\]\{\}\'\"]+$'
            },
            'category_consistency': {
                'amount_category_correlation': True,
                'description_category_matching': True
            }
        }

    def balance_financial_categories(self, transactions: List[Dict[str, Any]],
                                   target_category: str = None,
                                   method: str = 'auto') -> List[Dict[str, Any]]:
        """
        Balance financial categories with domain-specific constraints

        Args:
            transactions: List of transaction dictionaries
            target_category: Specific category to balance (None for all)
            method: Balancing method ('auto', 'smote', 'synthetic', 'pattern_based')

        Returns:
            List of balanced transactions including synthetic ones
        """
        try:
            # Convert to DataFrame for processing
            df = pd.DataFrame(transactions)

            # Analyze current category distribution
            category_dist = self._analyze_category_distribution(df)

            if target_category:
                # Balance specific category
                balanced_transactions = self._balance_specific_category(
                    df, target_category, method
                )
            else:
                # Balance all categories
                balanced_transactions = self._balance_all_categories(df, method)

            self.logger.info(f"Financial category balancing completed. Generated {len(balanced_transactions) - len(transactions)} synthetic transactions")
            return balanced_transactions

        except Exception as e:
            self.logger.error(f"Error balancing financial categories: {str(e)}")
            return transactions

    def _analyze_category_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the distribution of financial categories"""
        try:
            if 'category' not in df.columns:
                raise ValidationError("DataFrame must contain 'category' column")

            category_counts = Counter(df['category'])
            total_transactions = len(df)

            distribution = {}
            for category, count in category_counts.items():
                distribution[category] = {
                    'count': count,
                    'percentage': (count / total_transactions) * 100,
                    'needs_balancing': (count / total_transactions) < 0.1  # Less than 10%
                }

            # Identify minority and majority categories
            minority_categories = [cat for cat, info in distribution.items() if info['needs_balancing']]
            majority_category = max(category_counts, key=category_counts.get)

            return {
                'distribution': distribution,
                'minority_categories': minority_categories,
                'majority_category': majority_category,
                'total_transactions': total_transactions,
                'imbalance_ratio': max(category_counts.values()) / min(category_counts.values()) if category_counts else 1.0
            }

        except Exception as e:
            self.logger.error(f"Error analyzing category distribution: {str(e)}")
            return {}

    def _balance_specific_category(self, df: pd.DataFrame, target_category: str,
                                 method: str) -> List[Dict[str, Any]]:
        """Balance a specific financial category"""
        try:
            # Filter data for target category
            target_data = df[df['category'] == target_category].copy()
            other_data = df[df['category'] != target_category].copy()

            if len(target_data) == 0:
                self.logger.warning(f"No data found for category: {target_category}")
                return df.to_dict('records')

            # Generate synthetic data for target category
            synthetic_data = self._generate_category_specific_synthetic(
                target_data, target_category, method
            )

            # Combine original and synthetic data
            if synthetic_data:
                synthetic_df = pd.DataFrame(synthetic_data)
                balanced_df = pd.concat([df, synthetic_df], ignore_index=True)
                return balanced_df.to_dict('records')
            else:
                return df.to_dict('records')

        except Exception as e:
            self.logger.error(f"Error balancing specific category {target_category}: {str(e)}")
            return df.to_dict('records')

    def _balance_all_categories(self, df: pd.DataFrame, method: str) -> List[Dict[str, Any]]:
        """Balance all financial categories"""
        try:
            category_dist = self._analyze_category_distribution(df)
            minority_categories = category_dist.get('minority_categories', [])

            all_balanced_data = [df.copy()]

            for category in minority_categories:
                self.logger.info(f"Balancing category: {category}")

                # Generate synthetic data for this category
                category_data = df[df['category'] == category].copy()
                synthetic_data = self._generate_category_specific_synthetic(
                    category_data, category, method
                )

                if synthetic_data:
                    synthetic_df = pd.DataFrame(synthetic_data)
                    all_balanced_data.append(synthetic_df)

            # Combine all data
            if len(all_balanced_data) > 1:
                balanced_df = pd.concat(all_balanced_data, ignore_index=True)
                return balanced_df.to_dict('records')
            else:
                return df.to_dict('records')

        except Exception as e:
            self.logger.error(f"Error balancing all categories: {str(e)}")
            return df.to_dict('records')

    def _generate_category_specific_synthetic(self, category_data: pd.DataFrame,
                                            category: str, method: str) -> List[Dict[str, Any]]:
        """Generate synthetic data specific to a financial category"""
        try:
            if len(category_data) == 0:
                return []

            # Extract category patterns
            category_pattern = self.category_patterns.get(category, self.category_patterns.get('outros', {}))

            # Choose generation method
            if method == 'auto':
                method = self._select_financial_method(category_data, category)

            synthetic_transactions = []

            if method == 'pattern_based':
                synthetic_transactions = self._generate_pattern_based_synthetic(
                    category_data, category, category_pattern
                )
            elif method == 'smote':
                synthetic_transactions = self._generate_smote_based_synthetic(
                    category_data, category
                )
            elif method == 'synthetic':
                synthetic_transactions = self._generate_advanced_synthetic(
                    category_data, category
                )

            # Validate and clean synthetic data
            validated_transactions = self._validate_synthetic_transactions(
                synthetic_transactions, category
            )

            return validated_transactions

        except Exception as e:
            self.logger.error(f"Error generating category-specific synthetic data: {str(e)}")
            return []

    def _select_financial_method(self, category_data: pd.DataFrame, category: str) -> str:
        """Select the most appropriate method for financial data generation"""
        try:
            n_samples = len(category_data)
            n_features = len(category_data.columns)

            # For small datasets, use pattern-based generation
            if n_samples < 50:
                return 'pattern_based'
            # For medium datasets, use SMOTE
            elif n_samples < 500:
                return 'smote'
            # For larger datasets, use advanced synthetic generation
            else:
                return 'synthetic'

        except Exception:
            return 'pattern_based'

    def _generate_pattern_based_synthetic(self, category_data: pd.DataFrame,
                                        category: str, category_pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate synthetic transactions based on learned patterns"""
        try:
            synthetic_transactions = []
            n_samples = len(category_data)

            # Extract statistical patterns
            amount_stats = category_data['amount'].describe() if 'amount' in category_data.columns else None
            description_patterns = self._extract_description_patterns(category_data)

            # Generate synthetic samples
            for _ in range(max(10, n_samples // 2)):  # Generate at least 10 samples
                transaction = self._generate_single_pattern_transaction(
                    category, category_pattern, amount_stats, description_patterns
                )
                if transaction:
                    synthetic_transactions.append(transaction)

            return synthetic_transactions

        except Exception as e:
            self.logger.error(f"Error in pattern-based generation: {str(e)}")
            return []

    def _generate_single_pattern_transaction(self, category: str, category_pattern: Dict[str, Any],
                                           amount_stats: pd.Series = None,
                                           description_patterns: List[str] = None) -> Optional[Dict[str, Any]]:
        """Generate a single synthetic transaction based on patterns"""
        try:
            # Generate amount
            amount = self._generate_realistic_amount(category_pattern, amount_stats)

            # Generate description
            description = self._generate_realistic_description(category, description_patterns)

            # Generate date (recent past)
            date = self._generate_realistic_date()

            # Create transaction
            transaction = {
                'amount': round(amount, 2),
                'description': description,
                'category': category,
                'date': date.isoformat(),
                'is_synthetic': True
            }

            return transaction

        except Exception as e:
            self.logger.error(f"Error generating single transaction: {str(e)}")
            return None

    def _generate_realistic_amount(self, category_pattern: Dict[str, Any],
                                 amount_stats: pd.Series = None) -> float:
        """Generate a realistic amount for the category"""
        try:
            min_amount, max_amount = category_pattern.get('typical_amount_range', (0.01, 1000.0))

            if amount_stats is not None:
                # Use statistical distribution from real data
                mean = amount_stats.get('mean', (min_amount + max_amount) / 2)
                std = amount_stats.get('std', (max_amount - min_amount) / 4)

                # Generate from normal distribution
                amount = np.random.normal(mean, std)

                # Ensure within bounds
                amount = np.clip(amount, min_amount, max_amount)
            else:
                # Use uniform distribution
                amount = np.random.uniform(min_amount, max_amount)

            return abs(amount)  # Ensure positive

        except Exception:
            return np.random.uniform(0.01, 100.0)

    def _generate_realistic_description(self, category: str,
                                      description_patterns: List[str] = None) -> str:
        """Generate a realistic description for the category"""
        try:
            if description_patterns:
                # Use existing patterns
                base_description = random.choice(description_patterns)
            else:
                # Use category defaults
                category_pattern = self.category_patterns.get(category, self.category_patterns.get('outros', {}))
                common_descriptions = category_pattern.get('common_descriptions', ['transacao'])
                base_description = random.choice(common_descriptions)

            # Add some variation
            variations = ['', ' - ref', ' - parcela', ' - ajuste', ' - estorno']
            variation = random.choice(variations)

            return f"{base_description}{variation}"

        except Exception:
            return f"transacao {category}"

    def _generate_realistic_date(self) -> datetime:
        """Generate a realistic date for the transaction"""
        try:
            # Generate date within the last year
            days_back = random.randint(0, 365)
            date = datetime.now() - timedelta(days=days_back)
            return date

        except Exception:
            return datetime.now() - timedelta(days=random.randint(1, 30))

    def _extract_description_patterns(self, category_data: pd.DataFrame) -> List[str]:
        """Extract common description patterns from category data"""
        try:
            if 'description' not in category_data.columns:
                return []

            descriptions = category_data['description'].dropna().tolist()

            # Extract common words/phrases
            patterns = []
            for desc in descriptions:
                # Clean and split description
                clean_desc = re.sub(r'[^\w\s]', '', str(desc).lower())
                words = clean_desc.split()

                if len(words) > 0:
                    patterns.append(' '.join(words[:3]))  # First 3 words

            # Return unique patterns
            return list(set(patterns))[:10]  # Limit to 10 patterns

        except Exception:
            return []

    def _generate_smote_based_synthetic(self, category_data: pd.DataFrame,
                                      category: str) -> List[Dict[str, Any]]:
        """Generate synthetic transactions using SMOTE"""
        try:
            # Prepare data for SMOTE
            if len(category_data) < 6:  # SMOTE needs at least 6 samples
                return self._generate_pattern_based_synthetic(
                    category_data, category, self.category_patterns.get(category, {})
                )

            # Convert to numerical features
            numerical_data = self._prepare_numerical_features(category_data)

            if numerical_data is None or len(numerical_data) == 0:
                return []

            # Apply SMOTE
            X_balanced, _ = self.smote_handler.apply_smote(
                numerical_data, np.ones(len(numerical_data)),  # Dummy target
                method='classic',
                k_neighbors=min(5, len(numerical_data) - 1)
            )

            # Convert back to transactions
            synthetic_transactions = []
            for i in range(len(X_balanced) - len(numerical_data)):
                transaction = self._numerical_to_transaction(
                    X_balanced[len(numerical_data) + i], category_data, category
                )
                if transaction:
                    synthetic_transactions.append(transaction)

            return synthetic_transactions

        except Exception as e:
            self.logger.error(f"Error in SMOTE-based generation: {str(e)}")
            return []

    def _generate_advanced_synthetic(self, category_data: pd.DataFrame,
                                   category: str) -> List[Dict[str, Any]]:
        """Generate synthetic transactions using advanced generative models"""
        try:
            # Use the advanced synthetic data generator
            synthetic_df = self.synthetic_generator.generate_synthetic_data(category_data)

            if synthetic_df is not None and not synthetic_df.empty:
                # Convert to transaction format
                synthetic_transactions = []
                for _, row in synthetic_df.iterrows():
                    transaction = self._dataframe_row_to_transaction(row, category)
                    if transaction:
                        synthetic_transactions.append(transaction)

                return synthetic_transactions
            else:
                return []

        except Exception as e:
            self.logger.error(f"Error in advanced synthetic generation: {str(e)}")
            return []

    def _prepare_numerical_features(self, category_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare numerical features for SMOTE"""
        try:
            # Select relevant columns
            feature_columns = []
            for col in category_data.columns:
                if col in ['amount', 'date']:
                    feature_columns.append(col)
                elif category_data[col].dtype in ['int64', 'float64']:
                    feature_columns.append(col)

            if not feature_columns:
                return None

            # Extract features
            features_df = category_data[feature_columns].copy()

            # Process date column
            if 'date' in features_df.columns:
                features_df['date'] = pd.to_datetime(features_df['date']).astype(int) / 10**9

            # Handle missing values
            features_df = features_df.fillna(0)

            # Scale features
            features_array = self.amount_scaler.fit_transform(features_df.values)

            return features_array

        except Exception as e:
            self.logger.error(f"Error preparing numerical features: {str(e)}")
            return None

    def _numerical_to_transaction(self, numerical_features: np.ndarray,
                                original_data: pd.DataFrame, category: str) -> Optional[Dict[str, Any]]:
        """Convert numerical features back to transaction format"""
        try:
            # This is a simplified conversion
            # In production, this would be more sophisticated

            amount = abs(numerical_features[0]) if len(numerical_features) > 0 else random.uniform(0.01, 100.0)

            transaction = {
                'amount': round(amount, 2),
                'description': self._generate_realistic_description(category),
                'category': category,
                'date': self._generate_realistic_date().isoformat(),
                'is_synthetic': True
            }

            return transaction

        except Exception as e:
            return None

    def _dataframe_row_to_transaction(self, row: pd.Series, category: str) -> Optional[Dict[str, Any]]:
        """Convert DataFrame row to transaction format"""
        try:
            transaction = {
                'amount': abs(row.get('amount', random.uniform(0.01, 100.0))),
                'description': row.get('description', self._generate_realistic_description(category)),
                'category': category,
                'date': row.get('date', self._generate_realistic_date().isoformat()),
                'is_synthetic': True
            }

            return transaction

        except Exception as e:
            return None

    def _validate_synthetic_transactions(self, transactions: List[Dict[str, Any]],
                                       category: str) -> List[Dict[str, Any]]:
        """Validate synthetic transactions against business rules"""
        try:
            validated_transactions = []

            rules = self.business_rules

            for transaction in transactions:
                if self._validate_single_transaction(transaction, rules):
                    validated_transactions.append(transaction)

            self.logger.info(f"Validated {len(validated_transactions)}/{len(transactions)} synthetic transactions")
            return validated_transactions

        except Exception as e:
            self.logger.error(f"Error validating synthetic transactions: {str(e)}")
            return transactions

    def _validate_single_transaction(self, transaction: Dict[str, Any],
                                   rules: Dict[str, Any]) -> bool:
        """Validate a single synthetic transaction"""
        try:
            # Amount validation
            amount = transaction.get('amount', 0)
            amount_rules = rules.get('amount_constraints', {})

            if not (amount_rules.get('minimum_amount', 0) <= amount <= amount_rules.get('maximum_amount', float('inf'))):
                return False

            # Description validation
            description = transaction.get('description', '')
            desc_rules = rules.get('description_constraints', {})

            if not (desc_rules.get('minimum_length', 0) <= len(description) <= desc_rules.get('maximum_length', float('inf'))):
                return False

            if desc_rules.get('allowed_characters'):
                if not re.match(desc_rules['allowed_characters'], description):
                    return False

            # Date validation
            date_str = transaction.get('date', '')
            if date_str:
                try:
                    date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    temporal_rules = rules.get('temporal_constraints', {})

                    if temporal_rules.get('future_dates_not_allowed', True):
                        if date > datetime.now():
                            return False

                except Exception:
                    return False

            return True

        except Exception:
            return False

    def get_category_statistics(self) -> Dict[str, Any]:
        """Get statistics about category balancing operations"""
        return {
            'category_patterns': self.category_patterns,
            'business_rules': self.business_rules,
            'category_stats': self.category_stats,
            'temporal_stats': self.temporal_stats,
            'amount_distributions': self.amount_distributions
        }

    def update_category_patterns(self, new_patterns: Dict[str, Dict[str, Any]]):
        """Update category patterns with new domain knowledge"""
        try:
            self.category_patterns.update(new_patterns)
            self.logger.info("Category patterns updated successfully")

        except Exception as e:
            self.logger.error(f"Error updating category patterns: {str(e)}")

    def save_balancer_state(self, filepath: str):
        """Save the current state of the financial balancer"""
        try:
            state = {
                'config': self.config,
                'category_patterns': self.category_patterns,
                'business_rules': self.business_rules,
                'category_stats': self.category_stats,
                'temporal_stats': self.temporal_stats,
                'amount_distributions': self.amount_distributions
            }

            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            self.logger.info(f"Financial balancer state saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving balancer state: {str(e)}")

    def load_balancer_state(self, filepath: str):
        """Load financial balancer state from file"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.config = state.get('config', self.config)
            self.category_patterns = state.get('category_patterns', self.category_patterns)
            self.business_rules = state.get('business_rules', self.business_rules)
            self.category_stats = state.get('category_stats', {})
            self.temporal_stats = state.get('temporal_stats', {})
            self.amount_distributions = state.get('amount_distributions', {})

            self.logger.info(f"Financial balancer state loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading balancer state: {str(e)}")