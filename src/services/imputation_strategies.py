import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import re
from collections import Counter

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError

logger = get_logger(__name__)


class FinancialImputationStrategy:
    """
    Specialized imputation strategies for financial data (amounts, balances, etc.)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'amount_imputation': {
                'method': 'context_aware',  # 'median', 'mean', 'context_aware', 'distribution_based'
                'consider_transaction_type': True,
                'use_category_patterns': True,
                'round_to_cents': True
            },
            'balance_imputation': {
                'method': 'forward_fill',  # 'forward_fill', 'interpolation', 'calculated'
                'max_gap_fill': 10  # Maximum number of consecutive missing balances to fill
            }
        }

    def impute_financial_amounts(self, df: pd.DataFrame,
                               amount_column: str = 'amount',
                               context_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Impute missing financial amounts using context-aware strategies

        Args:
            df: DataFrame with financial data
            amount_column: Column containing amounts
            context_columns: Columns to use as context for imputation

        Returns:
            Tuple of (imputed_data, imputation_info)
        """
        try:
            self.logger.info(f"Imputing financial amounts in column: {amount_column}")

            if amount_column not in df.columns:
                return df, {'error': f"Amount column '{amount_column}' not found"}

            df = df.copy()
            original_missing = df[amount_column].isnull().sum()

            if original_missing == 0:
                return df, {'imputation_count': 0, 'method': 'none_required'}

            # Choose imputation method
            method = self.config['amount_imputation']['method']

            if method == 'context_aware':
                df, info = self._context_aware_amount_imputation(df, amount_column, context_columns)
            elif method == 'distribution_based':
                df, info = self._distribution_based_amount_imputation(df, amount_column)
            elif method == 'median':
                median_value = df[amount_column].median()
                df[amount_column] = df[amount_column].fillna(median_value)
                info = {'method': 'median', 'fill_value': median_value}
            elif method == 'mean':
                mean_value = df[amount_column].mean()
                df[amount_column] = df[amount_column].fillna(mean_value)
                info = {'method': 'mean', 'fill_value': mean_value}
            else:
                return df, {'error': f"Unknown imputation method: {method}"}

            # Round to cents if configured
            if self.config['amount_imputation']['round_to_cents']:
                df[amount_column] = df[amount_column].round(2)

            final_missing = df[amount_column].isnull().sum()
            imputation_count = original_missing - final_missing

            result_info = {
                'method': method,
                'imputation_count': imputation_count,
                'original_missing': original_missing,
                'final_missing': final_missing,
                'success_rate': 1.0 - (final_missing / original_missing) if original_missing > 0 else 1.0,
                **info
            }

            self.logger.info(f"Financial amount imputation completed: {imputation_count} values imputed")
            return df, result_info

        except Exception as e:
            self.logger.error(f"Error in financial amount imputation: {str(e)}")
            return df, {'error': str(e)}

    def _context_aware_amount_imputation(self, df: pd.DataFrame,
                                       amount_column: str,
                                       context_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Context-aware imputation using transaction type, category, etc."""
        try:
            df = df.copy()
            imputation_info = {'context_rules': [], 'fallback_used': False}

            # Define context columns if not provided
            if context_columns is None:
                context_columns = []
                if 'transaction_type' in df.columns:
                    context_columns.append('transaction_type')
                if 'category' in df.columns:
                    context_columns.append('category')
                if 'description' in df.columns:
                    context_columns.append('description')

            # Group by context and impute
            missing_mask = df[amount_column].isnull()

            if context_columns:
                # Create context groups
                context_data = df[context_columns].fillna('unknown')

                # Find most similar complete records for each missing record
                for idx in df[missing_mask].index:
                    context_values = context_data.loc[idx]

                    # Find similar records
                    similarities = []
                    for complete_idx in df[~missing_mask].index:
                        complete_context = context_data.loc[complete_idx]
                        similarity = self._calculate_context_similarity(context_values, complete_context)
                        similarities.append((complete_idx, similarity))

                    # Sort by similarity
                    similarities.sort(key=lambda x: x[1], reverse=True)

                    # Use amount from most similar record
                    if similarities:
                        most_similar_idx = similarities[0][0]
                        imputed_value = df.loc[most_similar_idx, amount_column]
                        df.loc[idx, amount_column] = imputed_value

                        imputation_info['context_rules'].append({
                            'record_index': idx,
                            'similar_record': most_similar_idx,
                            'similarity_score': similarities[0][1],
                            'imputed_value': imputed_value
                        })

            # Fallback to median if context-aware failed
            still_missing = df[amount_column].isnull().sum()
            if still_missing > 0:
                median_value = df[amount_column].median()
                df[amount_column] = df[amount_column].fillna(median_value)
                imputation_info['fallback_used'] = True
                imputation_info['fallback_value'] = median_value

            return df, imputation_info

        except Exception as e:
            self.logger.warning(f"Context-aware imputation failed: {str(e)}, using median fallback")
            median_value = df[amount_column].median()
            df[amount_column] = df[amount_column].fillna(median_value)
            return df, {'fallback_used': True, 'fallback_value': median_value}

    def _calculate_context_similarity(self, context1: pd.Series, context2: pd.Series) -> float:
        """Calculate similarity between two context records"""
        try:
            similarity = 0.0
            total_weight = 0.0

            for col in context1.index:
                if col in context2.index:
                    val1, val2 = context1[col], context2[col]

                    if pd.isna(val1) and pd.isna(val2):
                        weight = 1.0
                    elif pd.isna(val1) or pd.isna(val2):
                        weight = 0.0
                    elif val1 == val2:
                        weight = 1.0
                    else:
                        # Partial similarity for text fields
                        if isinstance(val1, str) and isinstance(val2, str):
                            # Simple text similarity
                            words1 = set(val1.lower().split())
                            words2 = set(val2.lower().split())
                            if words1 or words2:
                                weight = len(words1.intersection(words2)) / len(words1.union(words2))
                            else:
                                weight = 0.0
                        else:
                            weight = 0.0

                    similarity += weight
                    total_weight += 1.0

            return similarity / total_weight if total_weight > 0 else 0.0

        except:
            return 0.0

    def _distribution_based_amount_imputation(self, df: pd.DataFrame,
                                            amount_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Impute amounts based on statistical distribution"""
        try:
            df = df.copy()
            missing_mask = df[amount_column].isnull()

            if missing_mask.sum() == 0:
                return df, {}

            # Fit distribution to existing data
            existing_amounts = df.loc[~missing_mask, amount_column]

            # Use quantile-based imputation to maintain distribution
            quantiles = existing_amounts.quantile([0.25, 0.5, 0.75])

            # Generate imputed values from similar distribution
            n_missing = missing_mask.sum()
            imputed_values = np.random.choice(
                existing_amounts,
                size=n_missing,
                replace=True
            )

            df.loc[missing_mask, amount_column] = imputed_values

            return df, {
                'distribution_method': 'empirical_sampling',
                'sample_size': len(existing_amounts),
                'quantiles': quantiles.to_dict()
            }

        except Exception as e:
            self.logger.warning(f"Distribution-based imputation failed: {str(e)}, using median")
            median_value = df[amount_column].median()
            df[amount_column] = df[amount_column].fillna(median_value)
            return df, {'fallback': 'median', 'fallback_value': median_value}

    def impute_balances(self, df: pd.DataFrame,
                       balance_column: str = 'balance',
                       date_column: str = 'date') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Impute missing balance values using temporal patterns

        Args:
            df: DataFrame with balance data
            balance_column: Column containing balances
            date_column: Column containing dates

        Returns:
            Tuple of (imputed_data, imputation_info)
        """
        try:
            self.logger.info(f"Imputing balances in column: {balance_column}")

            if balance_column not in df.columns:
                return df, {'error': f"Balance column '{balance_column}' not found"}

            df = df.copy()
            original_missing = df[balance_column].isnull().sum()

            if original_missing == 0:
                return df, {'imputation_count': 0, 'method': 'none_required'}

            # Sort by date if available
            if date_column in df.columns:
                df = df.sort_values(date_column).copy()

            method = self.config['balance_imputation']['method']

            if method == 'forward_fill':
                df[balance_column] = df[balance_column].fillna(method='ffill')
            elif method == 'interpolation':
                df[balance_column] = df[balance_column].interpolate(method='linear')
            elif method == 'calculated':
                df = self._calculate_balances_from_transactions(df, balance_column)

            # Limit consecutive fills
            max_gap = self.config['balance_imputation']['max_gap_fill']
            df = self._limit_consecutive_fills(df, balance_column, max_gap)

            final_missing = df[balance_column].isnull().sum()
            imputation_count = original_missing - final_missing

            result_info = {
                'method': method,
                'imputation_count': imputation_count,
                'max_gap_limit': max_gap,
                'consecutive_fills_limited': final_missing > 0
            }

            self.logger.info(f"Balance imputation completed: {imputation_count} values imputed")
            return df, result_info

        except Exception as e:
            self.logger.error(f"Error in balance imputation: {str(e)}")
            return df, {'error': str(e)}

    def _calculate_balances_from_transactions(self, df: pd.DataFrame,
                                            balance_column: str) -> pd.DataFrame:
        """Calculate missing balances from transaction amounts"""
        try:
            df = df.copy()

            # Assume we have amount and transaction_type columns
            if 'amount' not in df.columns or 'transaction_type' not in df.columns:
                return df

            # Find gaps in balance data
            missing_mask = df[balance_column].isnull()

            for idx in df[missing_mask].index:
                # Try to calculate from previous balance and current transaction
                prev_idx = idx - 1
                if prev_idx >= 0 and not pd.isna(df.loc[prev_idx, balance_column]):
                    prev_balance = df.loc[prev_idx, balance_column]
                    amount = df.loc[idx, 'amount'] if not pd.isna(df.loc[idx, 'amount']) else 0
                    transaction_type = df.loc[idx, 'transaction_type']

                    # Calculate new balance
                    if transaction_type == 'credit':
                        new_balance = prev_balance + amount
                    elif transaction_type == 'debit':
                        new_balance = prev_balance - amount
                    else:
                        new_balance = prev_balance

                    df.loc[idx, balance_column] = new_balance

            return df

        except Exception as e:
            self.logger.warning(f"Balance calculation failed: {str(e)}")
            return df

    def _limit_consecutive_fills(self, df: pd.DataFrame,
                               balance_column: str,
                               max_gap: int) -> pd.DataFrame:
        """Limit the number of consecutive missing values that can be filled"""
        try:
            df = df.copy()

            # Find consecutive missing sequences
            missing_mask = df[balance_column].isnull()
            missing_groups = (missing_mask != missing_mask.shift()).cumsum()

            for group_id in missing_groups[missing_mask].unique():
                group_mask = (missing_groups == group_id) & missing_mask
                group_size = group_mask.sum()

                if group_size > max_gap:
                    # Keep only the first max_gap values in this group
                    group_indices = df[group_mask].index
                    keep_indices = group_indices[:max_gap]
                    remove_indices = group_indices[max_gap:]

                    # Set the remaining values back to NaN
                    df.loc[remove_indices, balance_column] = np.nan

            return df

        except Exception as e:
            self.logger.warning(f"Error limiting consecutive fills: {str(e)}")
            return df


class DateImputationStrategy:
    """
    Specialized imputation strategies for date/time data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'date_imputation': {
                'method': 'interpolation',  # 'interpolation', 'pattern_based', 'context_aware'
                'max_extrapolation_days': 30,
                'consider_business_days': True,
                'holiday_calendar': 'BR'  # Brazil holidays
            }
        }

    def impute_dates(self, df: pd.DataFrame,
                    date_column: str = 'date',
                    context_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Impute missing dates using temporal patterns and context

        Args:
            df: DataFrame with date data
            date_column: Column containing dates
            context_columns: Columns to use as context

        Returns:
            Tuple of (imputed_data, imputation_info)
        """
        try:
            self.logger.info(f"Imputing dates in column: {date_column}")

            if date_column not in df.columns:
                return df, {'error': f"Date column '{date_column}' not found"}

            df = df.copy()
            original_missing = df[date_column].isnull().sum()

            if original_missing == 0:
                return df, {'imputation_count': 0, 'method': 'none_required'}

            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

            method = self.config['date_imputation']['method']

            if method == 'interpolation':
                df, info = self._interpolate_dates(df, date_column)
            elif method == 'pattern_based':
                df, info = self._pattern_based_date_imputation(df, date_column)
            elif method == 'context_aware':
                df, info = self._context_aware_date_imputation(df, date_column, context_columns)

            final_missing = df[date_column].isnull().sum()
            imputation_count = original_missing - final_missing

            result_info = {
                'method': method,
                'imputation_count': imputation_count,
                'original_missing': original_missing,
                'final_missing': final_missing,
                **info
            }

            self.logger.info(f"Date imputation completed: {imputation_count} values imputed")
            return df, result_info

        except Exception as e:
            self.logger.error(f"Error in date imputation: {str(e)}")
            return df, {'error': str(e)}

    def _interpolate_dates(self, df: pd.DataFrame, date_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Interpolate missing dates"""
        try:
            df = df.copy()

            # Create a numeric representation for interpolation
            df['_date_numeric'] = df[date_column].astype('int64') // 10**9  # Convert to seconds since epoch

            # Interpolate
            df['_date_numeric'] = df['_date_numeric'].interpolate(method='linear')

            # Convert back to datetime
            df[date_column] = pd.to_datetime(df['_date_numeric'], unit='s')

            # Clean up
            df = df.drop('_date_numeric', axis=1)

            return df, {'interpolation_method': 'linear'}

        except Exception as e:
            self.logger.warning(f"Date interpolation failed: {str(e)}")
            return df, {'error': str(e)}

    def _pattern_based_date_imputation(self, df: pd.DataFrame, date_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Impute dates based on patterns (e.g., monthly, weekly)"""
        try:
            df = df.copy()

            # Analyze existing date patterns
            existing_dates = df[date_column].dropna()

            if len(existing_dates) < 2:
                return df, {'error': 'Insufficient data for pattern analysis'}

            # Calculate intervals between dates
            sorted_dates = existing_dates.sort_values()
            intervals = sorted_dates.diff().dropna()

            # Find most common interval
            most_common_interval = intervals.mode().iloc[0] if len(intervals) > 0 else pd.Timedelta(days=1)

            # Impute missing dates using the pattern
            missing_mask = df[date_column].isnull()

            for idx in df[missing_mask].index:
                # Find nearest non-missing dates
                prev_date = None
                next_date = None

                # Look backwards
                for prev_idx in range(idx - 1, -1, -1):
                    if not pd.isna(df.loc[prev_idx, date_column]):
                        prev_date = df.loc[prev_idx, date_column]
                        break

                # Look forwards
                for next_idx in range(idx + 1, len(df)):
                    if not pd.isna(df.loc[next_idx, date_column]):
                        next_date = df.loc[next_idx, date_column]
                        break

                # Impute based on available neighbors
                if prev_date is not None and next_date is not None:
                    # Interpolate between prev and next
                    imputed_date = prev_date + (next_date - prev_date) / 2
                elif prev_date is not None:
                    # Extrapolate forward
                    imputed_date = prev_date + most_common_interval
                elif next_date is not None:
                    # Extrapolate backward
                    imputed_date = next_date - most_common_interval
                else:
                    # No reference dates available
                    continue

                df.loc[idx, date_column] = imputed_date

            return df, {
                'pattern_interval': str(most_common_interval),
                'pattern_method': 'interval_based'
            }

        except Exception as e:
            self.logger.warning(f"Pattern-based date imputation failed: {str(e)}")
            return df, {'error': str(e)}

    def _context_aware_date_imputation(self, df: pd.DataFrame,
                                     date_column: str,
                                     context_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Impute dates using context from other columns"""
        try:
            df = df.copy()

            if context_columns is None:
                context_columns = []

            # Look for temporal context in other columns
            temporal_context = {}

            # Check for relative date references in text columns
            for col in context_columns:
                if col in df.columns and df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
                    # Look for patterns like "last month", "next week", etc.
                    temporal_patterns = self._extract_temporal_patterns(df[col])
                    if temporal_patterns:
                        temporal_context[col] = temporal_patterns

            # Use temporal context to impute dates
            missing_mask = df[date_column].isnull()

            for idx in df[missing_mask].index:
                imputed_date = self._infer_date_from_context(df.loc[idx], temporal_context, df[date_column])

                if imputed_date is not None:
                    df.loc[idx, date_column] = imputed_date

            return df, {
                'context_columns_used': list(temporal_context.keys()),
                'temporal_patterns_found': len(temporal_context)
            }

        except Exception as e:
            self.logger.warning(f"Context-aware date imputation failed: {str(e)}")
            return df, {'error': str(e)}

    def _extract_temporal_patterns(self, text_series: pd.Series) -> Dict[str, Any]:
        """Extract temporal patterns from text data"""
        try:
            patterns = {
                'relative_dates': [],
                'absolute_dates': [],
                'recurring_patterns': []
            }

            # Common Portuguese temporal expressions
            relative_patterns = {
                r'ontem': -1,  # yesterday
                r'hoje': 0,    # today
                r'amanhã': 1,  # tomorrow
                r'semana passada': -7,  # last week
                r'próxima semana': 7,   # next week
                r'mês passado': -30,    # last month
                r'próximo mês': 30,     # next month
            }

            for text in text_series.dropna():
                text_lower = text.lower()

                for pattern, days_offset in relative_patterns.items():
                    if re.search(pattern, text_lower):
                        patterns['relative_dates'].append({
                            'pattern': pattern,
                            'days_offset': days_offset,
                            'text': text
                        })

            return patterns if patterns['relative_dates'] else None

        except Exception as e:
            return None

    def _infer_date_from_context(self, row: pd.Series,
                               temporal_context: Dict[str, Any],
                               date_series: pd.Series) -> Optional[datetime]:
        """Infer date from contextual information"""
        try:
            base_date = datetime.now()  # Default to current date

            # Use existing dates as reference
            existing_dates = date_series.dropna()
            if len(existing_dates) > 0:
                base_date = existing_dates.mean()  # Use mean date as reference

            # Apply temporal offsets from context
            for col, patterns in temporal_context.items():
                if col in row.index and not pd.isna(row[col]):
                    text = str(row[col]).lower()

                    for pattern_info in patterns.get('relative_dates', []):
                        if re.search(pattern_info['pattern'], text):
                            offset_days = pattern_info['days_offset']
                            inferred_date = base_date + timedelta(days=offset_days)
                            return inferred_date

            return None

        except Exception as e:
            return None


class TextImputationStrategy:
    """
    Specialized imputation strategies for text data (descriptions, categories, etc.)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'text_imputation': {
                'method': 'pattern_based',  # 'pattern_based', 'similarity', 'category_based'
                'min_pattern_length': 3,
                'similarity_threshold': 0.6,
                'use_embeddings': False  # Could be extended with BERT embeddings
            }
        }

    def impute_text_descriptions(self, df: pd.DataFrame,
                               text_column: str = 'description',
                               context_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Impute missing text descriptions using pattern recognition

        Args:
            df: DataFrame with text data
            text_column: Column containing text descriptions
            context_columns: Columns to use as context

        Returns:
            Tuple of (imputed_data, imputation_info)
        """
        try:
            self.logger.info(f"Imputing text descriptions in column: {text_column}")

            if text_column not in df.columns:
                return df, {'error': f"Text column '{text_column}' not found"}

            df = df.copy()
            original_missing = df[text_column].isnull().sum()

            if original_missing == 0:
                return df, {'imputation_count': 0, 'method': 'none_required'}

            method = self.config['text_imputation']['method']

            if method == 'pattern_based':
                df, info = self._pattern_based_text_imputation(df, text_column, context_columns)
            elif method == 'similarity':
                df, info = self._similarity_based_text_imputation(df, text_column)
            elif method == 'category_based':
                df, info = self._category_based_text_imputation(df, text_column, context_columns)

            final_missing = df[text_column].isnull().sum()
            imputation_count = original_missing - final_missing

            result_info = {
                'method': method,
                'imputation_count': imputation_count,
                'original_missing': original_missing,
                'final_missing': final_missing,
                **info
            }

            self.logger.info(f"Text description imputation completed: {imputation_count} values imputed")
            return df, result_info

        except Exception as e:
            self.logger.error(f"Error in text description imputation: {str(e)}")
            return df, {'error': str(e)}

    def _pattern_based_text_imputation(self, df: pd.DataFrame,
                                     text_column: str,
                                     context_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Impute text using pattern recognition from context"""
        try:
            df = df.copy()

            # Extract patterns from existing text
            existing_texts = df[text_column].dropna().astype(str)
            patterns = self._extract_text_patterns(existing_texts)

            # Use context to generate appropriate text
            missing_mask = df[text_column].isnull()

            for idx in df[missing_mask].index:
                context = {}
                if context_columns:
                    for col in context_columns:
                        if col in df.columns:
                            context[col] = df.loc[idx, col]

                # Generate text based on context and patterns
                imputed_text = self._generate_text_from_context(context, patterns)
                if imputed_text:
                    df.loc[idx, text_column] = imputed_text

            return df, {
                'patterns_extracted': len(patterns),
                'context_columns_used': context_columns or []
            }

        except Exception as e:
            self.logger.warning(f"Pattern-based text imputation failed: {str(e)}")
            return df, {'error': str(e)}

    def _extract_text_patterns(self, texts: pd.Series) -> Dict[str, Any]:
        """Extract common patterns from text data"""
        try:
            patterns = {
                'common_words': [],
                'common_phrases': [],
                'structure_patterns': []
            }

            # Common words
            all_words = []
            for text in texts:
                words = re.findall(r'\b\w+\b', text.lower())
                all_words.extend(words)

            word_counts = Counter(all_words)
            patterns['common_words'] = [word for word, count in word_counts.most_common(20)
                                      if len(word) > 2]

            # Common phrases (2-3 word combinations)
            phrases = []
            for text in texts:
                words = re.findall(r'\b\w+\b', text.lower())
                for i in range(len(words) - 1):
                    phrases.append(' '.join(words[i:i+2]))
                for i in range(len(words) - 2):
                    phrases.append(' '.join(words[i:i+3]))

            phrase_counts = Counter(phrases)
            patterns['common_phrases'] = [phrase for phrase, count in phrase_counts.most_common(10)
                                        if count > 1]

            return patterns

        except Exception as e:
            return {}

    def _generate_text_from_context(self, context: Dict[str, Any],
                                  patterns: Dict[str, Any]) -> Optional[str]:
        """Generate text description from context"""
        try:
            text_parts = []

            # Use category/type information
            if 'category' in context and not pd.isna(context['category']):
                text_parts.append(str(context['category']))

            if 'transaction_type' in context and not pd.isna(context['transaction_type']):
                text_parts.append(str(context['transaction_type']))

            # Add common words/phrases
            if patterns.get('common_words'):
                # Add 1-2 random common words
                import random
                n_words = random.randint(1, min(2, len(patterns['common_words'])))
                selected_words = random.sample(patterns['common_words'], n_words)
                text_parts.extend(selected_words)

            if text_parts:
                return ' '.join(text_parts)
            else:
                return "Transaction"

        except Exception as e:
            return "Transaction"

    def _similarity_based_text_imputation(self, df: pd.DataFrame,
                                        text_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Impute text using similarity to existing descriptions"""
        try:
            df = df.copy()

            existing_texts = df[text_column].dropna().astype(str)
            missing_mask = df[text_column].isnull()

            for idx in df[missing_mask].index:
                # Find most similar existing text
                best_match = None
                best_similarity = 0

                target_text = self._get_context_text(df.loc[idx])

                for existing_text in existing_texts:
                    similarity = self._calculate_text_similarity(target_text, existing_text)
                    if similarity > best_similarity and similarity >= self.config['text_imputation']['similarity_threshold']:
                        best_match = existing_text
                        best_similarity = similarity

                if best_match:
                    df.loc[idx, text_column] = best_match

            return df, {'similarity_threshold': self.config['text_imputation']['similarity_threshold']}

        except Exception as e:
            self.logger.warning(f"Similarity-based text imputation failed: {str(e)}")
            return df, {'error': str(e)}

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        try:
            if not text1 or not text2:
                return 0.0

            # Simple word overlap similarity
            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))

            if not words1 or not words2:
                return 0.0

            intersection = words1.intersection(words2)
            union = words1.union(words2)

            return len(intersection) / len(union)

        except:
            return 0.0

    def _get_context_text(self, row: pd.Series) -> str:
        """Generate context text from row data"""
        try:
            context_parts = []
            for col, value in row.items():
                if not pd.isna(value) and col != 'description':
                    context_parts.append(str(value))
            return ' '.join(context_parts)
        except:
            return ""

    def _category_based_text_imputation(self, df: pd.DataFrame,
                                      text_column: str,
                                      context_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Impute text based on category patterns"""
        try:
            df = df.copy()

            # Group by category and find most common descriptions
            category_column = None
            if context_columns and 'category' in context_columns and 'category' in df.columns:
                category_column = 'category'
            elif 'transaction_type' in df.columns:
                category_column = 'transaction_type'

            if category_column:
                # Create mapping of category to most common description
                category_descriptions = {}
                for category, group in df.groupby(category_column):
                    descriptions = group[text_column].dropna().astype(str)
                    if len(descriptions) > 0:
                        most_common = descriptions.mode().iloc[0] if len(descriptions.mode()) > 0 else descriptions.iloc[0]
                        category_descriptions[category] = most_common

                # Impute using category mapping
                missing_mask = df[text_column].isnull()
                for idx in df[missing_mask].index:
                    category_value = df.loc[idx, category_column]
                    if category_value in category_descriptions:
                        df.loc[idx, text_column] = category_descriptions[category_value]

                return df, {
                    'category_column': category_column,
                    'categories_mapped': len(category_descriptions)
                }

            return df, {'error': 'No suitable category column found'}

        except Exception as e:
            self.logger.warning(f"Category-based text imputation failed: {str(e)}")
            return df, {'error': str(e)}


class CategoricalImputationStrategy:
    """
    Specialized imputation strategies for categorical data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'categorical_imputation': {
                'method': 'frequency',  # 'frequency', 'model_based', 'context_aware'
                'min_samples_per_category': 5,
                'handle_rare_categories': True,
                'rare_category_threshold': 0.05  # 5% of data
            }
        }

    def impute_categorical_fields(self, df: pd.DataFrame,
                                categorical_columns: List[str],
                                context_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Impute missing categorical values using specialized strategies

        Args:
            df: DataFrame with categorical data
            categorical_columns: List of categorical columns to impute
            context_columns: Columns to use as context

        Returns:
            Tuple of (imputed_data, imputation_info)
        """
        try:
            self.logger.info(f"Imputing categorical fields: {categorical_columns}")

            df = df.copy()
            imputation_info = {
                'columns_processed': [],
                'total_imputations': 0,
                'method_used': self.config['categorical_imputation']['method']
            }

            method = self.config['categorical_imputation']['method']

            for column in categorical_columns:
                if column not in df.columns:
                    continue

                original_missing = df[column].isnull().sum()
                if original_missing == 0:
                    continue

                if method == 'frequency':
                    df, column_info = self._frequency_based_imputation(df, column)
                elif method == 'model_based':
                    df, column_info = self._model_based_categorical_imputation(df, column, context_columns)
                elif method == 'context_aware':
                    df, column_info = self._context_aware_categorical_imputation(df, column, context_columns)

                final_missing = df[column].isnull().sum()
                imputations = original_missing - final_missing

                imputation_info['columns_processed'].append(column)
                imputation_info['total_imputations'] += imputations
                imputation_info[f'{column}_info'] = column_info

            self.logger.info(f"Categorical imputation completed: {imputation_info['total_imputations']} values imputed")
            return df, imputation_info

        except Exception as e:
            self.logger.error(f"Error in categorical imputation: {str(e)}")
            return df, {'error': str(e)}

    def _frequency_based_imputation(self, df: pd.DataFrame,
                                  column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Impute categorical values based on frequency distribution"""
        try:
            df = df.copy()

            # Calculate frequency distribution
            value_counts = df[column].value_counts()
            total_valid = value_counts.sum()

            # Handle rare categories if configured
            if self.config['categorical_imputation']['handle_rare_categories']:
                threshold = self.config['categorical_imputation']['rare_category_threshold']
                min_samples = int(threshold * total_valid)

                # Group rare categories
                rare_mask = value_counts < min_samples
                if rare_mask.any():
                    rare_categories = value_counts[rare_mask].index
                    # Replace rare categories with 'Other'
                    df[column] = df[column].replace(rare_categories, 'Other')

                    # Recalculate frequencies
                    value_counts = df[column].value_counts()

            # Use most frequent value for imputation
            most_frequent = value_counts.index[0]

            df[column] = df[column].fillna(most_frequent)

            return df, {
                'most_frequent_value': most_frequent,
                'unique_values': len(value_counts),
                'rare_categories_handled': rare_mask.sum() if 'rare_mask' in locals() else 0
            }

        except Exception as e:
            self.logger.warning(f"Frequency-based imputation failed for {column}: {str(e)}")
            # Fallback to simple mode imputation
            mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else 'Unknown'
            df[column] = df[column].fillna(mode_value)
            return df, {'fallback': 'mode', 'fallback_value': mode_value}

    def _model_based_categorical_imputation(self, df: pd.DataFrame,
                                          column: str,
                                          context_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Use predictive model to impute categorical values"""
        try:
            df = df.copy()

            if not context_columns:
                # Fallback to frequency-based
                return self._frequency_based_imputation(df, column)

            # Prepare training data
            missing_mask = df[column].isnull()
            train_data = df[~missing_mask].copy()
            pred_data = df[missing_mask].copy()

            if len(train_data) == 0:
                return self._frequency_based_imputation(df, column)

            # Prepare features
            feature_columns = [col for col in context_columns if col in df.columns and col != column]

            if not feature_columns:
                return self._frequency_based_imputation(df, column)

            # Simple rule-based prediction for categorical target
            # Group by context features and find most common category
            context_category_map = {}

            # Create context key from feature columns
            train_data['_context_key'] = train_data[feature_columns].astype(str).agg('_'.join, axis=1)

            for context_key, group in train_data.groupby('_context_key'):
                if len(group) >= self.config['categorical_imputation']['min_samples_per_category']:
                    most_common = group[column].mode().iloc[0] if not group[column].mode().empty else None
                    if most_common:
                        context_category_map[context_key] = most_common

            # Apply predictions
            pred_data['_context_key'] = pred_data[feature_columns].astype(str).agg('_'.join, axis=1)

            for idx in pred_data.index:
                context_key = pred_data.loc[idx, '_context_key']
                if context_key in context_category_map:
                    df.loc[idx, column] = context_category_map[context_key]
                else:
                    # Fallback to overall most frequent
                    most_frequent = train_data[column].mode().iloc[0] if not train_data[column].mode().empty else 'Unknown'
                    df.loc[idx, column] = most_frequent

            # Clean up temporary columns
            for temp_col in ['_context_key']:
                if temp_col in df.columns:
                    df = df.drop(temp_col, axis=1)

            return df, {
                'context_rules_created': len(context_category_map),
                'feature_columns_used': feature_columns
            }

        except Exception as e:
            self.logger.warning(f"Model-based categorical imputation failed for {column}: {str(e)}")
            return self._frequency_based_imputation(df, column)

    def _context_aware_categorical_imputation(self, df: pd.DataFrame,
                                            column: str,
                                            context_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Impute categorical values using contextual relationships"""
        try:
            df = df.copy()

            # Similar to model-based but with more sophisticated context analysis
            # For now, delegate to model-based implementation
            return self._model_based_categorical_imputation(df, column, context_columns)

        except Exception as e:
            self.logger.warning(f"Context-aware categorical imputation failed for {column}: {str(e)}")
            return self._frequency_based_imputation(df, column)


class ImputationStrategies:
    """
    Main class providing access to all specialized imputation strategies
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)

        # Initialize all strategies
        self.financial_strategy = FinancialImputationStrategy(self.config.get('financial', {}))
        self.date_strategy = DateImputationStrategy(self.config.get('date', {}))
        self.text_strategy = TextImputationStrategy(self.config.get('text', {}))
        self.categorical_strategy = CategoricalImputationStrategy(self.config.get('categorical', {}))

        self.logger.info("ImputationStrategies initialized")

    def impute_financial_data(self, df: pd.DataFrame,
                            amount_column: str = 'amount',
                            balance_column: Optional[str] = 'balance',
                            date_column: str = 'date') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive imputation for financial transaction data

        Args:
            df: DataFrame with financial data
            amount_column: Column containing transaction amounts
            balance_column: Column containing account balances (optional)
            date_column: Column containing transaction dates

        Returns:
            Tuple of (imputed_data, imputation_summary)
        """
        try:
            self.logger.info("Starting comprehensive financial data imputation")

            df = df.copy()
            imputation_summary = {
                'strategies_used': [],
                'total_imputations': 0,
                'columns_processed': []
            }

            # 1. Impute amounts
            if amount_column in df.columns:
                df, amount_info = self.financial_strategy.impute_financial_amounts(
                    df, amount_column, [date_column, 'transaction_type', 'category']
                )
                imputation_summary['strategies_used'].append('financial_amounts')
                imputation_summary['total_imputations'] += amount_info.get('imputation_count', 0)
                imputation_summary['columns_processed'].append(amount_column)
                imputation_summary['amount_imputation'] = amount_info

            # 2. Impute balances if available
            if balance_column and balance_column in df.columns:
                df, balance_info = self.financial_strategy.impute_balances(
                    df, balance_column, date_column
                )
                imputation_summary['strategies_used'].append('financial_balances')
                imputation_summary['total_imputations'] += balance_info.get('imputation_count', 0)
                imputation_summary['columns_processed'].append(balance_column)
                imputation_summary['balance_imputation'] = balance_info

            # 3. Impute dates if needed
            if date_column in df.columns and df[date_column].isnull().sum() > 0:
                df, date_info = self.date_strategy.impute_dates(
                    df, date_column, [amount_column, 'transaction_type']
                )
                imputation_summary['strategies_used'].append('date_imputation')
                imputation_summary['total_imputations'] += date_info.get('imputation_count', 0)
                imputation_summary['columns_processed'].append(date_column)
                imputation_summary['date_imputation'] = date_info

            self.logger.info(f"Financial data imputation completed: {imputation_summary['total_imputations']} total imputations")
            return df, imputation_summary

        except Exception as e:
            self.logger.error(f"Error in comprehensive financial imputation: {str(e)}")
            return df, {'error': str(e)}

    def impute_mixed_data(self, df: pd.DataFrame,
                         column_types: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Impute mixed data types using appropriate strategies

        Args:
            df: DataFrame with mixed data types
            column_types: Manual specification of column types (optional)

        Returns:
            Tuple of (imputed_data, imputation_summary)
        """
        try:
            self.logger.info("Starting mixed data imputation")

            df = df.copy()
            imputation_summary = {
                'strategies_used': [],
                'total_imputations': 0,
                'columns_processed': []
            }

            # Auto-detect or use provided column types
            if column_types is None:
                column_types = self._infer_column_types(df)

            # Group columns by type
            financial_columns = []
            date_columns = []
            text_columns = []
            categorical_columns = []

            for column, col_type in column_types.items():
                if col_type == 'financial' and column in df.columns:
                    financial_columns.append(column)
                elif col_type == 'date' and column in df.columns:
                    date_columns.append(column)
                elif col_type == 'text' and column in df.columns:
                    text_columns.append(column)
                elif col_type == 'categorical' and column in df.columns:
                    categorical_columns.append(column)

            # Apply appropriate strategies
            context_columns = ['date', 'transaction_type', 'category', 'description']

            # Financial columns
            for col in financial_columns:
                if col == 'amount':
                    df, info = self.financial_strategy.impute_financial_amounts(df, col, context_columns)
                elif col == 'balance':
                    df, info = self.financial_strategy.impute_balances(df, col, 'date')
                else:
                    # Generic financial imputation
                    df, info = self.financial_strategy.impute_financial_amounts(df, col, context_columns)

                imputation_summary['strategies_used'].append(f'financial_{col}')
                imputation_summary['total_imputations'] += info.get('imputation_count', 0)
                imputation_summary['columns_processed'].append(col)

            # Date columns
            for col in date_columns:
                df, info = self.date_strategy.impute_dates(df, col, context_columns)
                imputation_summary['strategies_used'].append(f'date_{col}')
                imputation_summary['total_imputations'] += info.get('imputation_count', 0)
                imputation_summary['columns_processed'].append(col)

            # Text columns
            for col in text_columns:
                df, info = self.text_strategy.impute_text_descriptions(df, col, context_columns)
                imputation_summary['strategies_used'].append(f'text_{col}')
                imputation_summary['total_imputations'] += info.get('imputation_count', 0)
                imputation_summary['columns_processed'].append(col)

            # Categorical columns
            if categorical_columns:
                df, info = self.categorical_strategy.impute_categorical_fields(df, categorical_columns, context_columns)
                imputation_summary['strategies_used'].append('categorical')
                imputation_summary['total_imputations'] += info.get('total_imputations', 0)
                imputation_summary['columns_processed'].extend(categorical_columns)

            self.logger.info(f"Mixed data imputation completed: {imputation_summary['total_imputations']} total imputations")
            return df, imputation_summary

        except Exception as e:
            self.logger.error(f"Error in mixed data imputation: {str(e)}")
            return df, {'error': str(e)}

    def _infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Infer column types for imputation strategy selection"""
        try:
            column_types = {}

            for col in df.columns:
                # Financial columns (amounts, balances)
                if any(keyword in col.lower() for keyword in ['amount', 'balance', 'value', 'price', 'cost']):
                    if pd.api.types.is_numeric_dtype(df[col]):
                        column_types[col] = 'financial'

                # Date columns
                elif pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower():
                    column_types[col] = 'date'

                # Text columns
                elif df[col].dtype == 'object':
                    # Check if it's mostly text or categorical
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio > 0.5:  # High uniqueness suggests text
                        column_types[col] = 'text'
                    else:  # Low uniqueness suggests categorical
                        column_types[col] = 'categorical'

                # Default to categorical for other types
                else:
                    column_types[col] = 'categorical'

            return column_types

        except Exception as e:
            self.logger.warning(f"Error inferring column types: {str(e)}")
            # Default all to categorical
            return {col: 'categorical' for col in df.columns}