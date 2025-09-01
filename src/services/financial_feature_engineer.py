import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import re
from collections import Counter, defaultdict
import math

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError

logger = get_logger(__name__)


class FinancialFeatureEngineer:
    """
    Domain-specific financial feature engineer for transaction data
    Provides comprehensive financial feature extraction and analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FinancialFeatureEngineer

        Args:
            config: Configuration dictionary for financial feature engineering
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize financial knowledge base
        self._initialize_financial_knowledge()

        # Initialize feature extractors
        self._initialize_feature_extractors()

        # Feature storage and tracking
        self.feature_stats = defaultdict(int)
        self.financial_patterns = {}
        self.currency_rates = {}

        self.logger.info("FinancialFeatureEngineer initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for financial feature engineering"""
        return {
            'amount_processing': {
                'round_amount_detection': True,
                'amount_pattern_recognition': True,
                'decimal_precision_analysis': True,
                'amount_range_categorization': True
            },
            'transaction_analysis': {
                'type_classification': True,
                'frequency_analysis': True,
                'velocity_patterns': True,
                'merchant_categorization': True
            },
            'bank_specific': {
                'bank_code_recognition': True,
                'branch_pattern_analysis': True,
                'account_type_detection': True,
                'bank_fee_identification': True
            },
            'currency_features': {
                'currency_conversion': True,
                'exchange_rate_fluctuation': True,
                'multi_currency_support': True,
                'currency_risk_indicators': True
            },
            'regulatory_compliance': {
                'aml_indicators': True,
                'fraud_pattern_detection': True,
                'compliance_thresholds': True,
                'reporting_requirements': True
            },
            'financial_ratios': {
                'liquidity_ratios': True,
                'efficiency_ratios': True,
                'profitability_indicators': True,
                'risk_assessment': True
            },
            'quality_assurance': {
                'feature_validation': True,
                'outlier_detection': True,
                'consistency_checks': True,
                'data_quality_scoring': True
            }
        }

    def _initialize_financial_knowledge(self):
        """Initialize financial domain knowledge"""
        try:
            # Brazilian bank codes
            self.bank_codes = {
                '001': 'Banco do Brasil',
                '033': 'Santander',
                '104': 'Caixa Econômica Federal',
                '237': 'Bradesco',
                '341': 'Itaú Unibanco',
                '399': 'HSBC',
                '745': 'Citibank',
                '422': 'Banco Safra',
                '318': 'Banco BMG',
                '077': 'Banco Inter'
            }

            # Transaction type patterns
            self.transaction_patterns = {
                'credit': r'\b(credito|crédito|recebimento|depósito|deposito)\b',
                'debit': r'\b(debito|débito|pagamento|transferencia|saque)\b',
                'fee': r'\b(tarifa|taxa|custo|juros|multa)\b',
                'reversal': r'\b(estorno|cancelamento|devolucao|devolução)\b',
                'adjustment': r'\b(ajuste|correcao|correção|acertos)\b'
            }

            # Financial amount patterns
            self.amount_patterns = {
                'round_numbers': lambda x: x % 10 == 0,
                'psychological_prices': lambda x: x % 1 == 0.99,
                'tax_amounts': lambda x: x in [0.99, 1.99, 2.99, 4.99, 9.99, 19.99, 49.99, 99.99],
                'common_denominations': lambda x: x in [0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00, 50.00, 100.00]
            }

            # Merchant categories
            self.merchant_categories = {
                'supermarket': r'\b(mercado|supermercado|extra|carrefour|pao|pão)\b',
                'restaurant': r'\b(restaurante|bar|lanchonete|cafe|café|pizza)\b',
                'pharmacy': r'\b(farmacia|farmacia|drogaria|droga)\b',
                'gas_station': r'\b(posto|combustivel|shell|petrobras|ipiranga)\b',
                'shopping': r'\b(shopping|mall|loja|store)\b',
                'utilities': r'\b(energia|luz|agua|água|telefone|internet|celular)\b',
                'transport': r'\b(uber|taxi|metro|onibus|ônibus|trem|aviao|avião)\b',
                'entertainment': r'\b(cinema|teatro|show|concerto|festival)\b'
            }

            # Currency information
            self.currency_info = {
                'BRL': {'symbol': 'R$', 'decimal_places': 2, 'country': 'Brazil'},
                'USD': {'symbol': '$', 'decimal_places': 2, 'country': 'United States'},
                'EUR': {'symbol': '€', 'decimal_places': 2, 'country': 'Eurozone'},
                'GBP': {'symbol': '£', 'decimal_places': 2, 'country': 'United Kingdom'}
            }

            self.logger.info("Financial knowledge base initialized")

        except Exception as e:
            self.logger.error(f"Error initializing financial knowledge: {str(e)}")

    def _initialize_feature_extractors(self):
        """Initialize specialized feature extractors"""
        try:
            # Amount pattern extractor
            self.amount_extractor = AmountPatternExtractor()

            # Transaction analyzer
            self.transaction_analyzer = TransactionAnalyzer()

            # Bank feature extractor
            self.bank_extractor = BankFeatureExtractor()

            # Currency processor
            self.currency_processor = CurrencyProcessor()

            # Regulatory compliance checker
            self.compliance_checker = RegulatoryComplianceChecker()

            self.logger.info("Feature extractors initialized")

        except Exception as e:
            self.logger.warning(f"Error initializing feature extractors: {str(e)}")

    def extract_financial_features(self, transactions: List[Dict],
                                 context_data: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Extract comprehensive financial features from transactions

        Args:
            transactions: List of transaction dictionaries
            context_data: Additional context data for enhanced features

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        try:
            if not transactions:
                return np.array([]), []

            self.logger.info(f"Extracting financial features for {len(transactions)} transactions")

            # Convert to DataFrame for processing
            df = pd.DataFrame(transactions)

            features = []
            feature_names = []

            # Amount-based features
            if self.config['amount_processing']['amount_pattern_recognition']:
                amount_features, amount_names = self._extract_amount_features(df)
                features.append(amount_features)
                feature_names.extend(amount_names)

            # Transaction analysis features
            if self.config['transaction_analysis']['type_classification']:
                transaction_features, transaction_names = self._extract_transaction_features(df)
                features.append(transaction_features)
                feature_names.extend(transaction_names)

            # Bank-specific features
            if self.config['bank_specific']['bank_code_recognition']:
                bank_features, bank_names = self._extract_bank_features(df)
                features.append(bank_features)
                feature_names.extend(bank_names)

            # Currency features
            if self.config['currency_features']['currency_conversion']:
                currency_features, currency_names = self._extract_currency_features(df)
                features.append(currency_features)
                feature_names.extend(currency_names)

            # Regulatory compliance features
            if self.config['regulatory_compliance']['aml_indicators']:
                compliance_features, compliance_names = self._extract_compliance_features(df)
                features.append(compliance_features)
                feature_names.extend(compliance_names)

            # Financial ratios
            if self.config['financial_ratios']['liquidity_ratios']:
                ratio_features, ratio_names = self._extract_ratio_features(df, context_data)
                features.append(ratio_features)
                feature_names.extend(ratio_names)

            # Combine all features
            if features:
                combined_features = np.concatenate(features, axis=1)
            else:
                combined_features = np.array([])

            # Handle NaN values
            combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Update feature statistics
            self._update_feature_stats(feature_names)

            self.logger.info(f"Extracted {len(feature_names)} financial features")
            return combined_features, feature_names

        except Exception as e:
            self.logger.error(f"Error extracting financial features: {str(e)}")
            return np.array([]), []

    def _extract_amount_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract amount-based financial features"""
        try:
            features = []

            if 'amount' not in df.columns:
                return np.array([]), []

            amounts = df['amount'].fillna(0).abs().values

            # Basic amount statistics
            features.append(amounts.reshape(-1, 1))  # Raw amount
            features.append(np.log1p(amounts).reshape(-1, 1))  # Log amount

            # Amount categorization
            amount_bins = [0, 10, 50, 100, 500, 1000, 5000, float('inf')]
            amount_categories = np.digitize(amounts, amount_bins).reshape(-1, 1)
            features.append(amount_categories)

            # Round amount detection
            is_round_amount = np.array([self.amount_patterns['round_numbers'](x) for x in amounts]).reshape(-1, 1)
            features.append(is_round_amount.astype(int))

            # Psychological price detection
            is_psychological = np.array([self.amount_patterns['psychological_prices'](x) for x in amounts]).reshape(-1, 1)
            features.append(is_psychological.astype(int))

            # Tax amount detection
            is_tax_amount = np.array([self.amount_patterns['tax_amounts'](x) for x in amounts]).reshape(-1, 1)
            features.append(is_tax_amount.astype(int))

            # Common denomination detection
            is_common_denomination = np.array([self.amount_patterns['common_denominations'](x) for x in amounts]).reshape(-1, 1)
            features.append(is_common_denomination.astype(int))

            # Amount decimal analysis
            decimals = (amounts * 100) % 100
            features.append(decimals.reshape(-1, 1))

            # Amount magnitude categories
            magnitude = np.log10(amounts + 1).astype(int).reshape(-1, 1)
            features.append(magnitude)

            feature_names = [
                'amount_raw', 'amount_log', 'amount_category', 'is_round_amount',
                'is_psychological_price', 'is_tax_amount', 'is_common_denomination',
                'amount_decimals', 'amount_magnitude'
            ]

            combined_features = np.concatenate(features, axis=1)
            return combined_features, feature_names

        except Exception as e:
            self.logger.error(f"Error extracting amount features: {str(e)}")
            return np.array([]), []

    def _extract_transaction_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract transaction analysis features"""
        try:
            features = []

            # Transaction type classification
            if 'description' in df.columns:
                descriptions = df['description'].fillna('').str.lower()

                # Credit transaction detection
                is_credit = descriptions.str.contains(self.transaction_patterns['credit'], regex=True).astype(int).values.reshape(-1, 1)
                features.append(is_credit)

                # Debit transaction detection
                is_debit = descriptions.str.contains(self.transaction_patterns['debit'], regex=True).astype(int).values.reshape(-1, 1)
                features.append(is_debit)

                # Fee detection
                is_fee = descriptions.str.contains(self.transaction_patterns['fee'], regex=True).astype(int).values.reshape(-1, 1)
                features.append(is_fee)

                # Reversal detection
                is_reversal = descriptions.str.contains(self.transaction_patterns['reversal'], regex=True).astype(int).values.reshape(-1, 1)
                features.append(is_reversal)

            # Merchant category detection
            merchant_categories = []
            for desc in df['description'].fillna(''):
                category_found = False
                for category, pattern in self.merchant_categories.items():
                    if re.search(pattern, desc.lower()):
                        merchant_categories.append(list(self.merchant_categories.keys()).index(category))
                        category_found = True
                        break
                if not category_found:
                    merchant_categories.append(-1)  # Unknown category

            features.append(np.array(merchant_categories).reshape(-1, 1))

            # Transaction frequency patterns (simplified)
            if 'date' in df.columns:
                df_copy = df.copy()
                df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
                df_copy = df_copy.dropna(subset=['date'])

                # Daily transaction count
                daily_counts = df_copy.groupby(df_copy['date'].dt.date).size()
                daily_freq = df_copy['date'].dt.date.map(daily_counts).fillna(1).values.reshape(-1, 1)
                features.append(daily_freq)

            feature_names = [
                'is_credit_transaction', 'is_debit_transaction', 'is_fee_transaction',
                'is_reversal_transaction', 'merchant_category', 'daily_transaction_frequency'
            ]

            combined_features = np.concatenate(features, axis=1)
            return combined_features, feature_names

        except Exception as e:
            self.logger.error(f"Error extracting transaction features: {str(e)}")
            return np.array([]), []

    def _extract_bank_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract bank-specific features"""
        try:
            features = []

            # Bank code extraction
            bank_codes = []
            for desc in df['description'].fillna(''):
                bank_found = False
                for code, bank_name in self.bank_codes.items():
                    if code in desc or bank_name.lower() in desc.lower():
                        bank_codes.append(int(code))
                        bank_found = True
                        break
                if not bank_found:
                    bank_codes.append(0)  # Unknown bank

            features.append(np.array(bank_codes).reshape(-1, 1))

            # Account number patterns
            account_patterns = []
            for desc in df['description'].fillna(''):
                # Look for account number patterns (XXXX-X or XXXXX-X)
                account_match = re.search(r'\b\d{4,5}[-]?\d{1,2}\b', desc)
                if account_match:
                    account_patterns.append(1)
                else:
                    account_patterns.append(0)

            features.append(np.array(account_patterns).reshape(-1, 1))

            # Agency patterns
            agency_patterns = []
            for desc in df['description'].fillna(''):
                agency_match = re.search(r'\bag\.?\s*\d{4}\b', desc.lower())
                if agency_match:
                    agency_patterns.append(1)
                else:
                    agency_patterns.append(0)

            features.append(np.array(agency_patterns).reshape(-1, 1))

            feature_names = ['bank_code', 'has_account_number', 'has_agency_number']

            combined_features = np.concatenate(features, axis=1)
            return combined_features, feature_names

        except Exception as e:
            self.logger.error(f"Error extracting bank features: {str(e)}")
            return np.array([]), []

    def _extract_currency_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract currency-related features"""
        try:
            features = []

            # Currency detection
            currencies = []
            for desc in df['description'].fillna(''):
                currency_found = False
                for currency_code, info in self.currency_info.items():
                    if info['symbol'] in desc or currency_code in desc.upper():
                        currencies.append(list(self.currency_info.keys()).index(currency_code))
                        currency_found = True
                        break
                if not currency_found:
                    currencies.append(0)  # Default to BRL

            features.append(np.array(currencies).reshape(-1, 1))

            # Amount in different currencies (simplified)
            if 'amount' in df.columns:
                amounts = df['amount'].fillna(0).values

                # Convert to USD equivalent (simplified rates)
                usd_conversion = amounts * 0.2  # Simplified BRL to USD rate
                features.append(usd_conversion.reshape(-1, 1))

                # Currency volatility indicator (simplified)
                currency_volatility = np.random.rand(len(amounts)) * 0.1  # Placeholder
                features.append(currency_volatility.reshape(-1, 1))

            feature_names = ['currency_type', 'amount_usd_equivalent', 'currency_volatility']

            combined_features = np.concatenate(features, axis=1)
            return combined_features, feature_names

        except Exception as e:
            self.logger.error(f"Error extracting currency features: {str(e)}")
            return np.array([]), []

    def _extract_compliance_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract regulatory compliance features"""
        try:
            features = []

            if 'amount' not in df.columns:
                return np.array([]), []

            amounts = df['amount'].fillna(0).values

            # AML threshold indicators
            aml_threshold_1k = (amounts >= 1000).astype(int).reshape(-1, 1)
            features.append(aml_threshold_1k)

            aml_threshold_10k = (amounts >= 10000).astype(int).reshape(-1, 1)
            features.append(aml_threshold_10k)

            # Suspicious pattern detection
            # Round amounts over certain thresholds
            suspicious_round = ((amounts % 1000 == 0) & (amounts >= 5000)).astype(int).reshape(-1, 1)
            features.append(suspicious_round)

            # High frequency same amount (simplified)
            amount_counts = Counter(amounts)
            high_freq_amount = np.array([amount_counts[amount] > 3 for amount in amounts]).astype(int).reshape(-1, 1)
            features.append(high_freq_amount)

            # Unusual timing (simplified - would need time data)
            unusual_timing = np.random.rand(len(amounts)) > 0.95  # 5% unusual
            features.append(unusual_timing.astype(int).reshape(-1, 1))

            feature_names = [
                'aml_threshold_1k', 'aml_threshold_10k', 'suspicious_round_amount',
                'high_frequency_amount', 'unusual_timing'
            ]

            combined_features = np.concatenate(features, axis=1)
            return combined_features, feature_names

        except Exception as e:
            self.logger.error(f"Error extracting compliance features: {str(e)}")
            return np.array([]), []

    def _extract_ratio_features(self, df: pd.DataFrame, context_data: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """Extract financial ratio features"""
        try:
            features = []

            if 'amount' not in df.columns:
                return np.array([]), []

            amounts = df['amount'].fillna(0).values

            # Basic liquidity ratios (simplified)
            if context_data and 'total_balance' in context_data:
                total_balance = context_data['total_balance']
                liquidity_ratio = amounts / total_balance if total_balance != 0 else 0
                features.append(liquidity_ratio.reshape(-1, 1))

            # Efficiency ratios
            if len(amounts) > 1:
                amount_volatility = np.std(amounts) / np.mean(amounts) if np.mean(amounts) != 0 else 0
                features.append(np.full((len(amounts), 1), amount_volatility))

            # Transaction efficiency (amount per transaction)
            avg_transaction_size = np.mean(amounts)
            efficiency_ratio = amounts / avg_transaction_size if avg_transaction_size != 0 else 1
            features.append(efficiency_ratio.reshape(-1, 1))

            # Risk indicators
            risk_score = np.abs(amounts) / 1000  # Simplified risk based on amount
            features.append(risk_score.reshape(-1, 1))

            feature_names = ['liquidity_ratio', 'amount_volatility', 'efficiency_ratio', 'risk_score']

            # Filter out features that couldn't be calculated
            existing_features = []
            existing_names = []
            for i, (feature, name) in enumerate(zip(features, feature_names)):
                if feature.shape[1] > 0:
                    existing_features.append(feature)
                    existing_names.append(name)

            if existing_features:
                combined_features = np.concatenate(existing_features, axis=1)
                return combined_features, existing_names
            else:
                return np.array([]), []

        except Exception as e:
            self.logger.error(f"Error extracting ratio features: {str(e)}")
            return np.array([]), []

    def _update_feature_stats(self, feature_names: List[str]):
        """Update feature extraction statistics"""
        try:
            for feature_name in feature_names:
                self.feature_stats[feature_name] += 1

            self.feature_stats['total_extractions'] += 1

        except Exception as e:
            self.logger.warning(f"Error updating feature stats: {str(e)}")

    def get_feature_stats(self) -> Dict[str, Any]:
        """Get feature extraction statistics"""
        return dict(self.feature_stats)

    def validate_financial_data(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Validate financial data quality and consistency

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Dictionary with validation results
        """
        try:
            df = pd.DataFrame(transactions)

            validation_results = {
                'total_transactions': len(transactions),
                'amount_validation': self._validate_amounts(df),
                'description_validation': self._validate_descriptions(df),
                'temporal_validation': self._validate_temporal_consistency(df),
                'overall_quality_score': 0.0
            }

            # Calculate overall quality score
            scores = []
            if validation_results['amount_validation']['valid_ratio'] is not None:
                scores.append(validation_results['amount_validation']['valid_ratio'])
            if validation_results['description_validation']['completeness'] is not None:
                scores.append(validation_results['description_validation']['completeness'])
            if validation_results['temporal_validation']['consistency_score'] is not None:
                scores.append(validation_results['temporal_validation']['consistency_score'])

            if scores:
                validation_results['overall_quality_score'] = np.mean(scores)

            return validation_results

        except Exception as e:
            self.logger.error(f"Error validating financial data: {str(e)}")
            return {'error': str(e)}

    def _validate_amounts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate amount data"""
        try:
            if 'amount' not in df.columns:
                return {'valid_ratio': None, 'issues': ['no_amount_column']}

            amounts = df['amount']
            valid_amounts = pd.to_numeric(amounts, errors='coerce')

            valid_ratio = valid_amounts.notna().mean()
            negative_ratio = (valid_amounts < 0).mean() if valid_amounts.notna().any() else 0
            zero_ratio = (valid_amounts == 0).mean() if valid_amounts.notna().any() else 0

            issues = []
            if valid_ratio < 0.9:
                issues.append('low_valid_amount_ratio')
            if negative_ratio > 0.5:
                issues.append('high_negative_amount_ratio')
            if zero_ratio > 0.1:
                issues.append('high_zero_amount_ratio')

            return {
                'valid_ratio': valid_ratio,
                'negative_ratio': negative_ratio,
                'zero_ratio': zero_ratio,
                'issues': issues
            }

        except Exception:
            return {'valid_ratio': None, 'issues': ['validation_error']}

    def _validate_descriptions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate description data"""
        try:
            if 'description' not in df.columns:
                return {'completeness': None, 'issues': ['no_description_column']}

            descriptions = df['description'].fillna('')
            completeness = (descriptions.str.len() > 0).mean()

            avg_length = descriptions.str.len().mean()
            unique_ratio = descriptions.nunique() / len(descriptions)

            issues = []
            if completeness < 0.8:
                issues.append('low_description_completeness')
            if avg_length < 5:
                issues.append('very_short_descriptions')
            if unique_ratio < 0.1:
                issues.append('low_description_diversity')

            return {
                'completeness': completeness,
                'avg_length': avg_length,
                'unique_ratio': unique_ratio,
                'issues': issues
            }

        except Exception:
            return {'completeness': None, 'issues': ['validation_error']}

    def _validate_temporal_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal consistency"""
        try:
            if 'date' not in df.columns:
                return {'consistency_score': None, 'issues': ['no_date_column']}

            dates = pd.to_datetime(df['date'], errors='coerce')
            valid_dates = dates.notna()

            consistency_score = valid_dates.mean()

            # Check for reasonable date ranges
            current_year = datetime.now().year
            reasonable_dates = (
                (dates.dt.year >= current_year - 5) &
                (dates.dt.year <= current_year + 1) &
                valid_dates
            )
            reasonable_ratio = reasonable_dates.mean() if valid_dates.any() else 0

            issues = []
            if consistency_score < 0.8:
                issues.append('low_date_validity')
            if reasonable_ratio < 0.8:
                issues.append('unreasonable_date_ranges')

            return {
                'consistency_score': consistency_score,
                'reasonable_ratio': reasonable_ratio,
                'issues': issues
            }

        except Exception:
            return {'consistency_score': None, 'issues': ['validation_error']}

    def clear_cache(self):
        """Clear financial feature cache"""
        self.financial_patterns.clear()
        self.logger.info("Financial feature cache cleared")

    def save_financial_engineer(self, filepath: str):
        """Save the financial feature engineer state"""
        try:
            import joblib

            save_dict = {
                'config': self.config,
                'feature_stats': dict(self.feature_stats),
                'financial_patterns': self.financial_patterns,
                'currency_rates': self.currency_rates
            }

            joblib.dump(save_dict, filepath)
            self.logger.info(f"FinancialFeatureEngineer saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving FinancialFeatureEngineer: {str(e)}")

    def load_financial_engineer(self, filepath: str):
        """Load the financial feature engineer state"""
        try:
            import joblib

            save_dict = joblib.load(filepath)

            self.config = save_dict['config']
            self.feature_stats = defaultdict(int, save_dict['feature_stats'])
            self.financial_patterns = save_dict['financial_patterns']
            self.currency_rates = save_dict['currency_rates']

            # Reinitialize components
            self._initialize_financial_knowledge()
            self._initialize_feature_extractors()

            self.logger.info(f"FinancialFeatureEngineer loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading FinancialFeatureEngineer: {str(e)}")
            raise ValidationError(f"Failed to load FinancialFeatureEngineer: {str(e)}")


class AmountPatternExtractor:
    """Specialized extractor for amount patterns"""

    def __init__(self):
        self.patterns = {
            'psychological': [0.99, 1.99, 2.99, 4.99, 9.99, 19.99, 49.99, 99.99],
            'tax_common': [0.99, 1.99, 2.99, 4.99, 9.99, 19.99, 49.99, 99.99],
            'denominations': [0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00, 50.00, 100.00]
        }


class TransactionAnalyzer:
    """Analyzer for transaction patterns"""

    def __init__(self):
        self.transaction_types = ['credit', 'debit', 'fee', 'transfer', 'payment']


class BankFeatureExtractor:
    """Extractor for bank-specific features"""

    def __init__(self):
        self.brazilian_banks = {
            '001': 'Banco do Brasil',
            '237': 'Bradesco',
            '341': 'Itaú',
            '033': 'Santander',
            '104': 'Caixa'
        }


class CurrencyProcessor:
    """Processor for currency-related features"""

    def __init__(self):
        self.exchange_rates = {'BRL': 1.0, 'USD': 5.0, 'EUR': 6.0}  # Simplified


class RegulatoryComplianceChecker:
    """Checker for regulatory compliance indicators"""

    def __init__(self):
        self.aml_thresholds = [1000, 10000, 50000]  # BRL thresholds
        self.suspicious_patterns = ['round_amounts', 'high_frequency', 'unusual_timing']