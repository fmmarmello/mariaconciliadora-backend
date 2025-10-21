import pandas as pd
from datetime import datetime, timedelta
from src.models.transaction import Transaction, ReconciliationRecord
from src.models.company_financial import CompanyFinancial
from src.models.user import db
from typing import List, Dict, Any, Optional, Tuple
from src.utils.logging_config import get_logger, get_audit_logger
from src.utils.exceptions import (
    ReconciliationError, DatabaseError, RecordNotFoundError,
    InsufficientDataError, ValidationError
)
from src.utils.error_handler import (
    handle_service_errors, with_database_transaction, with_timeout
)
from src.services.data_normalization import BrazilianDataNormalizer, NormalizationConfig, NormalizationMode
from src.services.context_aware_matching import ContextAwareMatcher, MatchingConfig, MatchingMode
from src.services.reconciliation_anomaly_detector import ReconciliationAnomalyDetector
# Lazy import to avoid circular dependency
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize loggers
logger = get_logger(__name__)
audit_logger = get_audit_logger()

class ReconciliationConfig:
    """Configuration class for reconciliation parameters"""
    
    def __init__(self):
        # Matching weights (should sum to 1.0)
        self.amount_exact_weight = 0.35
        self.amount_close_weight = 0.15
        self.date_same_day_weight = 0.25
        self.date_close_weight = 0.10
        self.description_weight = 0.30
        
        # Matching thresholds
        self.minimum_match_threshold = 0.65
        self.minimum_amount_threshold = 0.01
        self.maximum_date_diff_days = 30
        self.maximum_amount_diff_percent = 0.02  # 2%
        
        # Fuzzy matching parameters
        self.enable_fuzzy_matching = True
        self.fuzzy_threshold = 0.85
        self.description_fuzzy_threshold = 0.7

        # Semantic (TF-IDF) description similarity
        self.enable_semantic_description = True
        self.semantic_threshold = 0.7
        self.semantic_use_max = True  # if True, use max(baseline, semantic) else average

        # Optional reference/ID bonus
        self.enable_reference_bonus = True
        self.reference_bonus_weight = 0.05  # small bonus when references match strongly
        
        # Validation parameters
        self.validate_date_proximity = True
        self.validate_amount_proximity = True
        self.enable_manual_override = True
        
        # Performance parameters
        self.max_matches_per_transaction = 3
        self.enable_batch_processing = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for API response"""
        return {
            'amount_exact_weight': self.amount_exact_weight,
            'amount_close_weight': self.amount_close_weight,
            'date_same_day_weight': self.date_same_day_weight,
            'date_close_weight': self.date_close_weight,
            'description_weight': self.description_weight,
            'minimum_match_threshold': self.minimum_match_threshold,
            'minimum_amount_threshold': self.minimum_amount_threshold,
            'maximum_date_diff_days': self.maximum_date_diff_days,
            'maximum_amount_diff_percent': self.maximum_amount_diff_percent,
            'enable_fuzzy_matching': self.enable_fuzzy_matching,
            'fuzzy_threshold': self.fuzzy_threshold,
            'description_fuzzy_threshold': self.description_fuzzy_threshold,
            'enable_semantic_description': self.enable_semantic_description,
            'semantic_threshold': self.semantic_threshold,
            'semantic_use_max': self.semantic_use_max,
            'enable_reference_bonus': self.enable_reference_bonus,
            'reference_bonus_weight': self.reference_bonus_weight,
            'validate_date_proximity': self.validate_date_proximity,
            'validate_amount_proximity': self.validate_amount_proximity,
            'enable_manual_override': self.enable_manual_override,
            'max_matches_per_transaction': self.max_matches_per_transaction,
            'enable_batch_processing': self.enable_batch_processing
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate_weights(self) -> bool:
        """Validate that weights sum to approximately 1.0"""
        total_weight = (
            self.amount_exact_weight + 
            self.amount_close_weight + 
            self.date_same_day_weight + 
            self.date_close_weight + 
            self.description_weight
        )
        return abs(total_weight - 1.0) < 0.01

class ReconciliationService:
    """
    Enhanced reconciliation service with configurable parameters and user feedback
    """
    
    def __init__(self, config: Optional[ReconciliationConfig] = None):
        self.config = config or ReconciliationConfig()
        self.user_feedback_history = []
        # Initialize Phase 2 services with configuration
        self.normalizer = BrazilianDataNormalizer()
        self.context_matcher = ContextAwareMatcher()
        # Initialize TF-IDF vectorizer for semantic similarity (PT-BR handled by normalizer)
        try:
            self._tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=2000,
                stop_words=None  # rely on BrazilianDataNormalizer to clean/normalize
            )
        except Exception:
            self._tfidf_vectorizer = None
        
        # Initialize anomaly detection services
        self.anomaly_detector = ReconciliationAnomalyDetector()
        # Lazy initialize interactive anomaly manager to avoid circular import
        self.anomaly_manager = None
        
        # Performance metrics for enhanced integration
        self.performance_metrics = {
            'total_matches_processed': 0,
            'contextual_matches': 0,
            'pattern_based_matches': 0,
            'average_processing_time': 0.0,
            'integration_accuracy': 0.0,
            'anomalies_detected': 0,
            'anomalies_resolved': 0,
            'human_supervision_interactions': 0
        }
    
    def _get_anomaly_manager(self):
        """Lazy initialization of interactive anomaly manager"""
        if self.anomaly_manager is None:
            from src.services.interactive_anomaly_manager import InteractiveAnomalyManager
            self.anomaly_manager = InteractiveAnomalyManager()
        return self.anomaly_manager
    
    @handle_service_errors('reconciliation_service')
    @with_timeout(300)  # 5 minute timeout for matching
    def find_matches(self, bank_transactions: List[Transaction], company_entries: List[CompanyFinancial]) -> List[Dict[str, Any]]:
        """
        Enhanced matching with configurable parameters, multiple matches per transaction, and anomaly detection
        """
        # Validate input data
        if not bank_transactions:
            raise InsufficientDataError('reconciliation matching', 1, 0)
        
        if not company_entries:
            raise InsufficientDataError('reconciliation matching', 1, 0)
        
        logger.info(f"Starting enhanced reconciliation matching with anomaly detection. Bank transactions: {len(bank_transactions)}, Company entries: {len(company_entries)}")
        logger.info(f"Configuration: threshold={self.config.minimum_match_threshold}, fuzzy_matching={self.config.enable_fuzzy_matching}")
        
        try:
            matches = []
            processed_transactions = 0
            matching_errors = 0
            
            # Validate transaction data before processing
            valid_bank_transactions = self._validate_transactions_for_matching(bank_transactions)
            valid_company_entries = self._validate_entries_for_matching(company_entries)
            
            if not valid_bank_transactions or not valid_company_entries:
                raise InsufficientDataError('reconciliation matching', 1, 0)
            
            logger.info(f"Validated data: {len(valid_bank_transactions)} bank transactions, {len(valid_company_entries)} company entries")
            
            for bank_transaction in valid_bank_transactions:
                try:
                    processed_transactions += 1
                    
                    # Find all potential matches for this transaction
                    potential_matches = []
                    
                    for company_entry in valid_company_entries:
                        try:
                            score = self._calculate_enhanced_match_score(bank_transaction, company_entry)
                            
                            if score >= self.config.minimum_match_threshold:
                                potential_matches.append({
                                    'company_entry': company_entry,
                                    'match_score': score,
                                    'score_breakdown': self._get_score_breakdown(bank_transaction, company_entry, score)
                                })
                                
                        except Exception as e:
                            logger.debug(f"Error calculating match score: {str(e)}")
                            continue
                    
                    # Sort by score and take top N matches
                    potential_matches.sort(key=lambda x: x['match_score'], reverse=True)
                    top_matches = potential_matches[:self.config.max_matches_per_transaction]
                    
                    # Add all valid matches with anomaly detection
                    for match in top_matches:
                        match_data = {
                            'bank_transaction': bank_transaction,
                            'company_entry': match['company_entry'],
                            'match_score': match['match_score'],
                            'score_breakdown': match['score_breakdown'],
                            'match_rank': len(matches) + 1
                        }
                        
                        # Validate match before adding
                        if self._validate_enhanced_match(match_data):
                            # Perform anomaly detection on the match (non-blocking for preview)
                            try:
                                anomaly_analysis = self.anomaly_detector.analyze_reconciliation_match(
                                    bank_transaction, match['company_entry'], match['match_score'], match['score_breakdown']
                                )
                                match_data['anomaly_analysis'] = anomaly_analysis
                            except Exception as e:
                                logger.debug(f"Anomaly detection failed (non-blocking): {str(e)}")
                                match_data['anomaly_analysis'] = {'error': str(e)}
                            
                            # Update performance metrics
                            anomaly = match_data.get('anomaly_analysis', {})
                            if isinstance(anomaly, dict) and anomaly.get('is_anomaly', False):
                                self.performance_metrics['anomalies_detected'] += 1
                                logger.warning(f"Anomaly detected in match for transaction {bank_transaction.id}: {anomaly.get('anomaly_type', 'unknown')}")
                            
                            matches.append(match_data)
                            logger.debug(f"Match found for transaction {bank_transaction.id} with score {match['match_score']:.2f}")
                        else:
                            logger.debug(f"Match validation failed for transaction {bank_transaction.id}")
                    
                except Exception as e:
                    logger.warning(f"Error processing bank transaction {bank_transaction.id}: {str(e)}")
                    matching_errors += 1
                    continue
            
            logger.info(f"Enhanced reconciliation matching completed. Processed: {processed_transactions}, "
                       f"Matches found: {len(matches)}, Anomalies detected: {self.performance_metrics['anomalies_detected']}, Errors: {matching_errors}")
            
            if matching_errors > 0:
                logger.warning(f"Encountered {matching_errors} errors during matching process")
            
            return matches
            
        except Exception as e:
            if isinstance(e, (InsufficientDataError, ValidationError)):
                raise
            
            logger.error(f"Critical error in enhanced reconciliation matching: {str(e)}", exc_info=True)
            raise ReconciliationError(f"Enhanced matching process failed: {str(e)}")
    
    def find_matches_with_config(self, bank_transactions: List[Transaction], company_entries: List[CompanyFinancial], 
                               config: ReconciliationConfig) -> List[Dict[str, Any]]:
        """
        Find matches using a specific configuration (for real-time parameter adjustment)
        """
        # Store original config
        original_config = self.config
        # Use temporary config
        self.config = config
        
        try:
            matches = self.find_matches(bank_transactions, company_entries)
            return matches
        finally:
            # Restore original config
            self.config = original_config
    
    def _validate_transactions_for_matching(self, transactions: List[Transaction]) -> List[Transaction]:
        """Validate bank transactions for matching process."""
        valid_transactions = []
        
        for transaction in transactions:
            try:
                # Check required fields
                if not all([
                    transaction.id,
                    transaction.date,
                    transaction.amount is not None,
                    transaction.description
                ]):
                    logger.debug(f"Transaction {transaction.id} missing required fields")
                    continue
                
                # Check data validity
                if abs(transaction.amount) < 0.01:  # Skip very small amounts
                    logger.debug(f"Transaction {transaction.id} has very small amount: {transaction.amount}")
                    continue
                
                valid_transactions.append(transaction)
                
            except Exception as e:
                logger.debug(f"Error validating transaction {getattr(transaction, 'id', 'unknown')}: {str(e)}")
                continue
        
        logger.debug(f"Transaction validation: {len(valid_transactions)}/{len(transactions)} valid")
        return valid_transactions
    
    def _validate_entries_for_matching(self, entries: List[CompanyFinancial]) -> List[CompanyFinancial]:
        """Validate company entries for matching process."""
        valid_entries = []
        
        for entry in entries:
            try:
                # Check required fields
                if not all([
                    entry.id,
                    entry.date,
                    entry.amount is not None,
                    entry.description
                ]):
                    logger.debug(f"Entry {entry.id} missing required fields")
                    continue
                
                # Check data validity
                if abs(entry.amount) < 0.01:  # Skip very small amounts
                    logger.debug(f"Entry {entry.id} has very small amount: {entry.amount}")
                    continue
                
                valid_entries.append(entry)
                
            except Exception as e:
                logger.debug(f"Error validating entry {getattr(entry, 'id', 'unknown')}: {str(e)}")
                continue
        
        logger.debug(f"Entry validation: {len(valid_entries)}/{len(entries)} valid")
        return valid_entries
    
    def _validate_match(self, match_data: Dict[str, Any]) -> bool:
        """Validate a potential match (legacy method for backward compatibility)."""
        return self._validate_enhanced_match(match_data)
    
    def _validate_enhanced_match(self, match_data: Dict[str, Any]) -> bool:
        """Enhanced match validation using configurable parameters."""
        try:
            bank_transaction = match_data['bank_transaction']
            company_entry = match_data['company_entry']
            match_score = match_data.get('match_score', 0.0)
            
            # Basic validation
            if not all([bank_transaction, company_entry, match_score >= self.config.minimum_match_threshold]):
                return False
            
            # Check for reasonable date difference (configurable)
            if self.config.validate_date_proximity and bank_transaction.date and company_entry.date:
                date_diff = abs((bank_transaction.date - company_entry.date).days)
                if date_diff > self.config.maximum_date_diff_days:
                    logger.debug(f"Match rejected: date difference too large ({date_diff} days, max: {self.config.maximum_date_diff_days})")
                    return False
            
            # Check for reasonable amount difference (configurable)
            if self.config.validate_amount_proximity:
                amount_diff = abs(abs(bank_transaction.amount) - abs(company_entry.amount))
                if abs(bank_transaction.amount) > 0 and abs(company_entry.amount) > 0:
                    amount_ratio = amount_diff / max(abs(bank_transaction.amount), abs(company_entry.amount))
                    if amount_ratio > self.config.maximum_amount_diff_percent:
                        logger.debug(f"Match rejected: amount difference too large ({amount_ratio:.2%}, max: {self.config.maximum_amount_diff_percent})")
                        return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating enhanced match: {str(e)}")
            return False
    
    def _calculate_enhanced_match_score(self, bank_transaction: Transaction, company_entry: CompanyFinancial) -> float:
        """
        Enhanced match score calculation with Phase 2 integration and confidence scoring
        """
        import time
        start_time = time.time()
        
        try:
            # Calculate base score using existing logic
            base_score = self._calculate_base_match_score(bank_transaction, company_entry)
            
            # Apply enhanced context-aware matching with Phase 2 capabilities
            contextual_result = self.context_matcher.get_contextual_match_score(
                bank_transaction, company_entry, base_score
            )
            
            # Calculate enhanced confidence score using Phase 2 features
            # Support dataclass return by extracting numeric score
            try:
                ctx_score_numeric = getattr(contextual_result, 'score', contextual_result)
            except Exception:
                ctx_score_numeric = base_score

            confidence_score = self._calculate_enhanced_confidence_score(
                bank_transaction, company_entry, ctx_score_numeric
            )
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, contextual_result)
            
            return min(confidence_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error in enhanced match score calculation: {str(e)}")
            return self._calculate_base_match_score(bank_transaction, company_entry)
    
    def _calculate_base_match_score(self, bank_transaction: Transaction, company_entry: CompanyFinancial) -> float:
        """
        Calculate base match score using configurable parameters and Phase 2 normalization
        """
        score = 0.0
        
        # Amount matching with configurable weights
        amount_diff = abs(abs(bank_transaction.amount) - abs(company_entry.amount))
        if amount_diff < self.config.minimum_amount_threshold:
            score += self.config.amount_exact_weight
        elif amount_diff <= (abs(bank_transaction.amount) * self.config.maximum_amount_diff_percent):
            score += self.config.amount_close_weight
        
        # Date matching with configurable weights
        if bank_transaction.date and company_entry.date:
            date_diff = abs((bank_transaction.date - company_entry.date).days)
            if date_diff == 0:
                score += self.config.date_same_day_weight
            elif date_diff <= 3:
                score += self.config.date_close_weight
        
        # Enhanced description matching using Phase 2 normalizer + optional semantic (TF-IDF)
        description_similarity = self.normalizer.calculate_similarity(
            bank_transaction.description,
            company_entry.description
        )

        if self.config.enable_semantic_description:
            try:
                semantic_similarity = self._calculate_semantic_description_similarity(
                    bank_transaction.description, company_entry.description
                )
            except Exception:
                semantic_similarity = 0.0

            # Combine similarities without altering global weights
            if semantic_similarity >= self.config.semantic_threshold:
                if self.config.semantic_use_max:
                    final_desc_similarity = max(description_similarity, semantic_similarity)
                else:
                    # simple average if not using max
                    final_desc_similarity = (description_similarity + semantic_similarity) / 2.0
            else:
                final_desc_similarity = description_similarity
        else:
            final_desc_similarity = description_similarity

        score += final_desc_similarity * self.config.description_weight
        
        # Bonus for entity matching using Phase 2 entity extraction
        entity_bonus = self._calculate_entity_matching_bonus(
            bank_transaction.description, company_entry.description
        )
        score += entity_bonus * 0.1  # 10% bonus for entity matching

        # Optional reference/ID bonus (e.g., transaction_id vs an internal reference)
        if self.config.enable_reference_bonus:
            ref_bonus = self._calculate_reference_bonus(bank_transaction, company_entry)
            score += ref_bonus

        return min(score, 1.0)

    def _calculate_semantic_description_similarity(self, bank_description: str, company_description: str) -> float:
        """
        Calculate TF-IDF cosine similarity between normalized PT-BR descriptions.
        Safe fallback to 0.0 on errors or empty inputs.
        """
        if not bank_description or not company_description:
            return 0.0

        if self._tfidf_vectorizer is None:
            return 0.0

        # Normalize using BrazilianDataNormalizer for PT-BR coherence
        try:
            norm_bank = self.normalizer.normalize_text(bank_description).normalized_text
            norm_comp = self.normalizer.normalize_text(company_description).normalized_text
        except Exception:
            norm_bank = (bank_description or '').lower().strip()
            norm_comp = (company_description or '').lower().strip()

        if not norm_bank or not norm_comp:
            return 0.0

        if norm_bank == norm_comp:
            return 1.0

        try:
            tfidf = self._tfidf_vectorizer.fit_transform([norm_bank, norm_comp])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0]
            return float(sim)
        except Exception:
            return 0.0

    def _calculate_reference_bonus(self, bank_transaction: Transaction, company_entry: CompanyFinancial) -> float:
        """
        Small bonus when references/IDs strongly match. Uses transaction_id vs possible identifiers
        in company_entry description. Conservative weight to avoid overpowering base weights.
        """
        try:
            if not getattr(bank_transaction, 'transaction_id', None):
                return 0.0

            ref = str(bank_transaction.transaction_id).strip()
            if not ref:
                return 0.0

            # Simple presence or strong fuzzy match in company description
            company_desc = (company_entry.description or '').strip()
            if not company_desc:
                return 0.0

            company_desc_lower = company_desc.lower()
            if ref.lower() in company_desc_lower:
                return self.config.reference_bonus_weight

            # Fuzzy fallback
            sim = SequenceMatcher(None, ref.lower(), company_desc_lower).ratio()
            if sim >= 0.95:
                return self.config.reference_bonus_weight
            if sim >= 0.85:
                return self.config.reference_bonus_weight * 0.5

            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_enhanced_confidence_score(self, bank_transaction: Transaction, company_entry: CompanyFinancial, 
                                           contextual_result) -> float:
        """
        Calculate confidence score with detailed breakdown and explanations
        """
        confidence_factors = {
            'data_quality_score': self._calculate_data_quality_score(bank_transaction, company_entry),
            'pattern_consistency_score': self._calculate_pattern_consistency_score(bank_transaction, company_entry),
            'outlier_detection_score': self._calculate_outlier_detection_score(bank_transaction, company_entry),
            'temporal_consistency_score': self._calculate_temporal_consistency_score(bank_transaction, company_entry)
        }
        
        # Calculate weighted confidence
        confidence_weights = {
            'data_quality_score': 0.3,
            'pattern_consistency_score': 0.25,
            'outlier_detection_score': 0.25,
            'temporal_consistency_score': 0.2
        }
        
        weighted_confidence = sum(
            confidence_factors[factor] * weight 
            for factor, weight in confidence_weights.items()
        )
        
        # Combine base score with confidence
        final_score = (contextual_result * 0.7) + (weighted_confidence * 0.3)
        
        # Store confidence breakdown for explanation
        confidence_factors['final_confidence'] = weighted_confidence
        confidence_factors['base_score'] = contextual_result
        confidence_factors['final_score'] = final_score
        
        return final_score
    
    def _calculate_data_quality_score(self, bank_transaction: Transaction, company_entry: CompanyFinancial) -> float:
        """Calculate data quality score for both records"""
        score = 0.0
        
        # Check bank transaction data quality
        bank_quality = 0.0
        if bank_transaction.description and len(bank_transaction.description.strip()) > 5:
            bank_quality += 0.3
        if bank_transaction.amount and abs(bank_transaction.amount) > 0.01:
            bank_quality += 0.3
        if bank_transaction.date:
            bank_quality += 0.2
        if bank_transaction.category:
            bank_quality += 0.2
        
        # Check company entry data quality
        company_quality = 0.0
        if company_entry.description and len(company_entry.description.strip()) > 5:
            company_quality += 0.3
        if company_entry.amount and abs(company_entry.amount) > 0.01:
            company_quality += 0.3
        if company_entry.date:
            company_quality += 0.2
        if company_entry.category:
            company_quality += 0.1
        if company_entry.cost_center:
            company_quality += 0.1
        
        score = (bank_quality + company_quality) / 2
        return min(score, 1.0)
    
    def _calculate_pattern_consistency_score(self, bank_transaction: Transaction, company_entry: CompanyFinancial) -> float:
        """Calculate pattern consistency score based on historical data"""
        try:
            # Use context matcher to get pattern consistency (robust to dataclass return)
            _ctx_result = self.context_matcher.get_contextual_match_score(
                bank_transaction, company_entry, 0.5
            )
            context_factors = {}
            try:
                context_factors = getattr(_ctx_result, 'context_factors', {}) or {}
            except Exception:
                context_factors = {}
            
            # Calculate consistency based on pattern frequency
            consistency_bonus = 0.0
            if context_factors.get('description_pattern_bonus', 0) > 0:
                consistency_bonus += 0.3
            if context_factors.get('entity_pattern_bonus', 0) > 0:
                consistency_bonus += 0.3
            if context_factors.get('supplier_consistency_bonus', 0) > 0:
                consistency_bonus += 0.2
            if context_factors.get('amount_pattern_likelihood', 0) > 0:
                consistency_bonus += 0.2
            
            return min(consistency_bonus, 1.0)
            
        except Exception as e:
            logger.debug(f"Error calculating pattern consistency: {str(e)}")
            return 0.5
    
    def _calculate_outlier_detection_score(self, bank_transaction: Transaction, company_entry: CompanyFinancial) -> float:
        """Calculate outlier detection score"""
        try:
            score = 1.0
            
            # Check for amount outliers
            amount_diff = abs(abs(bank_transaction.amount) - abs(company_entry.amount))
            if amount_diff > 0:
                avg_amount = (abs(bank_transaction.amount) + abs(company_entry.amount)) / 2
                relative_diff = amount_diff / avg_amount if avg_amount > 0 else 0
                
                if relative_diff > 0.5:  # More than 50% difference
                    score -= 0.4
                elif relative_diff > 0.2:  # More than 20% difference
                    score -= 0.2
                elif relative_diff > 0.1:  # More than 10% difference
                    score -= 0.1
            
            # Check for date outliers
            if bank_transaction.date and company_entry.date:
                date_diff = abs((bank_transaction.date - company_entry.date).days)
                if date_diff > 30:
                    score -= 0.3
                elif date_diff > 14:
                    score -= 0.2
                elif date_diff > 7:
                    score -= 0.1
            
            return max(score, 0.0)
            
        except Exception as e:
            logger.debug(f"Error calculating outlier detection: {str(e)}")
            return 0.5
    
    def _calculate_temporal_consistency_score(self, bank_transaction: Transaction, company_entry: CompanyFinancial) -> float:
        """Calculate temporal consistency score"""
        try:
            score = 1.0
            
            # Check if dates are in reasonable temporal order
            if bank_transaction.date and company_entry.date:
                date_diff = (company_entry.date - bank_transaction.date).days
                
                # Company entry should typically be after or same day as bank transaction
                if date_diff < -7:  # Company entry much earlier than bank transaction
                    score -= 0.3
                elif date_diff < 0:  # Company entry slightly earlier
                    score -= 0.1
                elif date_diff > 30:  # Company entry much later than bank transaction
                    score -= 0.2
                elif date_diff > 14:  # Company entry somewhat later
                    score -= 0.1
            
            return max(score, 0.0)
            
        except Exception as e:
            logger.debug(f"Error calculating temporal consistency: {str(e)}")
            return 0.5
    
    def _get_score_breakdown(self, bank_transaction: Transaction, company_entry: CompanyFinancial, total_score: float) -> Dict[str, Any]:
        """
        Get detailed breakdown of match score components with confidence explanations
        """
        # Calculate base components
        amount_diff = abs(abs(bank_transaction.amount) - abs(company_entry.amount))
        amount_score = 0.0
        if amount_diff < self.config.minimum_amount_threshold:
            amount_score = self.config.amount_exact_weight
        elif amount_diff <= (abs(bank_transaction.amount) * self.config.maximum_amount_diff_percent):
            amount_score = self.config.amount_close_weight
        
        date_diff = 0
        date_score = 0.0
        if bank_transaction.date and company_entry.date:
            date_diff = abs((bank_transaction.date - company_entry.date).days)
            if date_diff == 0:
                date_score = self.config.date_same_day_weight
            elif date_diff <= 3:
                date_score = self.config.date_close_weight
        
        # Recompute description similarities for explanation
        baseline_desc_similarity = self.normalizer.calculate_similarity(
            bank_transaction.description,
            company_entry.description
        )
        semantic_desc_similarity = 0.0
        final_desc_similarity = baseline_desc_similarity
        if self.config.enable_semantic_description:
            semantic_desc_similarity = self._calculate_semantic_description_similarity(
                bank_transaction.description, company_entry.description
            )
            if semantic_desc_similarity >= self.config.semantic_threshold:
                final_desc_similarity = max(baseline_desc_similarity, semantic_desc_similarity) if self.config.semantic_use_max else (baseline_desc_similarity + semantic_desc_similarity) / 2.0
        description_score = final_desc_similarity * self.config.description_weight
        
        # Get context-aware information (robust to dataclass return)
        _ctx_result = self.context_matcher.get_contextual_match_score(
            bank_transaction, company_entry, total_score
        )
        try:
            contextual_score = getattr(_ctx_result, 'score', _ctx_result)
            context_info = {
                'base_score': getattr(_ctx_result, 'base_score', None),
                'context_factors': getattr(_ctx_result, 'context_factors', {}),
                'total_bonus': getattr(_ctx_result, 'total_bonus', 0.0),
                'explanation': getattr(_ctx_result, 'explanation', ''),
                'patterns_used': getattr(_ctx_result, 'patterns_used', 0),
            }
        except Exception:
            contextual_score = 0.0
            context_info = {'context_factors': {}, 'total_bonus': 0.0, 'explanation': ''}
        
        # Calculate confidence factors
        confidence_factors = {
            'data_quality_score': self._calculate_data_quality_score(bank_transaction, company_entry),
            'pattern_consistency_score': self._calculate_pattern_consistency_score(bank_transaction, company_entry),
            'outlier_detection_score': self._calculate_outlier_detection_score(bank_transaction, company_entry),
            'temporal_consistency_score': self._calculate_temporal_consistency_score(bank_transaction, company_entry)
        }
        
        # Generate explanations
        explanations = self._generate_score_explanations(
            bank_transaction, company_entry, amount_diff, date_diff,
            final_desc_similarity, confidence_factors, context_info
        )
        
        return {
            'total_score': total_score,
            'base_score': amount_score + date_score + description_score,
            'contextual_adjustment': context_info.get('total_bonus', 0),
            'components': {
                'amount_score': amount_score,
                'date_score': date_score,
                'description_score': description_score,
                'amount_difference': amount_diff,
                'date_difference': date_diff,
                'description_similarity': baseline_desc_similarity,
                'description_semantic_similarity': semantic_desc_similarity,
                'description_similarity_used': final_desc_similarity
            },
            'confidence_factors': confidence_factors,
            'context_info': context_info,
            'explanations': explanations,
            'weights': {
                'amount_exact': self.config.amount_exact_weight,
                'amount_close': self.config.amount_close_weight,
                'date_same_day': self.config.date_same_day_weight,
                'date_close': self.config.date_close_weight,
                'description': self.config.description_weight
            },
            'recommendation': self._generate_recommendation(total_score, confidence_factors, context_info)
        }
    
    def _generate_score_explanations(self, bank_transaction: Transaction, company_entry: CompanyFinancial,
                                   amount_diff: float, date_diff: int, description_similarity: float,
                                   confidence_factors: Dict[str, float], context_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable explanations for score components"""
        explanations = {}
        
        # Amount explanation
        if amount_diff < 0.01:
            explanations['amount'] = "Valores idênticos - correspondência perfeita"
        elif amount_diff < (abs(bank_transaction.amount) * 0.02):
            explanations['amount'] = f"Valores muito próximos (diferença de R$ {amount_diff:.2f})"
        elif amount_diff < (abs(bank_transaction.amount) * 0.05):
            explanations['amount'] = f"Valores próximos (diferença de R$ {amount_diff:.2f})"
        else:
            explanations['amount'] = f"Diferença significativa de valores (R$ {amount_diff:.2f})"
        
        # Date explanation
        if date_diff == 0:
            explanations['date'] = "Mesma data - correspondência perfeita"
        elif date_diff <= 1:
            explanations['date'] = f"Diferença de {date_diff} dia - muito próximo"
        elif date_diff <= 3:
            explanations['date'] = f"Diferença de {date_diff} dias - aceitável"
        elif date_diff <= 7:
            explanations['date'] = f"Diferença de {date_diff} dias - moderada"
        else:
            explanations['date'] = f"Diferença significativa de {date_diff} dias"
        
        # Description explanation
        if description_similarity >= 0.9:
            explanations['description'] = "Descrições muito similares ou idênticas"
        elif description_similarity >= 0.7:
            explanations['description'] = "Descrições similares com boa correspondência"
        elif description_similarity >= 0.5:
            explanations['description'] = "Descrições parcialmente similares"
        else:
            explanations['description'] = "Descrições com baixa similaridade"
        
        # Data quality explanation
        data_quality = confidence_factors.get('data_quality_score', 0)
        if data_quality >= 0.8:
            explanations['data_quality'] = "Dados de alta qualidade em ambos os registros"
        elif data_quality >= 0.6:
            explanations['data_quality'] = "Dados de boa qualidade"
        elif data_quality >= 0.4:
            explanations['data_quality'] = "Dados de qualidade moderada"
        else:
            explanations['data_quality'] = "Dados com problemas de qualidade"
        
        # Pattern consistency explanation
        pattern_consistency = confidence_factors.get('pattern_consistency_score', 0)
        if pattern_consistency >= 0.7:
            explanations['pattern_consistency'] = "Alta consistência com padrões históricos"
        elif pattern_consistency >= 0.5:
            explanations['pattern_consistency'] = "Consistência moderada com padrões históricos"
        else:
            explanations['pattern_consistency'] = "Baixa consistência com padrões históricos"
        
        # Contextual explanation
        context_explanation = context_info.get('explanation', '')
        if context_explanation and "No significant" not in context_explanation:
            explanations['context'] = context_explanation
        
        return explanations
    
    def _generate_recommendation(self, total_score: float, confidence_factors: Dict[str, float], 
                               context_info: Dict[str, Any]) -> str:
        """Generate recommendation based on score and confidence factors"""
        if total_score >= 0.9:
            return "CORRESPONDÊNCIA EXCELENTE - Altíssima confiança, recomendada aprovação automática"
        elif total_score >= 0.8:
            return "CORRESPONDÊNCIA FORTE - Alta confiança, recomendada revisão rápida"
        elif total_score >= 0.7:
            return "CORRESPONDÊNCIA BOA - Confiança moderada, recomendada revisão humana"
        elif total_score >= 0.6:
            return "CORRESPONDÊNCIA POSSÍVEL - Baixa confiança, requer investigação detalhada"
        else:
            return "CORRESPONDÊNCIA FRACA - Muito baixa confiança, não recomendada"
    
    def _calculate_fuzzy_description_similarity(self, bank_description: str, company_description: str) -> float:
        """
        Enhanced description similarity using fuzzy matching
        """
        if not bank_description or not company_description:
            return 0.0
        
        # Normalize descriptions
        bank_desc = bank_description.lower().strip()
        company_desc = company_description.lower().strip()
        
        # Exact match
        if bank_desc == company_desc:
            return 1.0
        
        # Containment check
        if bank_desc in company_desc or company_desc in bank_desc:
            return 0.9
        
        # Use SequenceMatcher for fuzzy matching
        similarity = SequenceMatcher(None, bank_desc, company_desc).ratio()
        
        if similarity >= self.config.fuzzy_threshold:
            return similarity
        
        # Try word-based matching
        return self._calculate_word_based_similarity(bank_desc, company_desc)
    
    def _calculate_word_based_similarity(self, desc1: str, desc2: str) -> float:
        """
        Calculate similarity based on word matching with variations
        """
        # Remove special characters and split into words
        words1 = set(re.findall(r'\b\w+\b', desc1))
        words2 = set(re.findall(r'\b\w+\b', desc2))
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate intersection
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Jaccard similarity
        jaccard_similarity = len(intersection) / len(union) if union else 0
        
        # Check for partial word matches (typos, abbreviations)
        partial_matches = 0
        for word1 in words1:
            for word2 in words2:
                if self._are_words_similar(word1, word2):
                    partial_matches += 1
                    break
        
        # Combine exact and partial matches
        total_possible = max(len(words1), len(words2))
        enhanced_similarity = (len(intersection) + partial_matches * 0.5) / total_possible
        
        return max(jaccard_similarity, enhanced_similarity)
    
    def _are_words_similar(self, word1: str, word2: str) -> bool:
        """
        Check if two words are similar (handles typos, abbreviations)
        """
        if len(word1) < 3 or len(word2) < 3:
            return word1 == word2
        
        # Check if one is substring of the other (for abbreviations)
        if word1 in word2 or word2 in word1:
            return True
        
        # Use Levenshtein distance-like approach with SequenceMatcher
        similarity = SequenceMatcher(None, word1, word2).ratio()
        return similarity >= self.config.description_fuzzy_threshold
    
    def _calculate_description_similarity(self, bank_description: str, company_description: str) -> float:
        """
        Calcula a similaridade entre duas descrições usando uma abordagem simples
        """
        # Converte para minúsculas e remove espaços extras
        bank_desc = bank_description.lower().strip()
        company_desc = company_description.lower().strip()
        
        # Verifica se uma descrição está contida na outra
        if bank_desc in company_desc or company_desc in bank_desc:
            return 0.8
        
        # Verifica palavras em comum
        bank_words = set(bank_desc.split())
        company_words = set(company_desc.split())
        
        if len(bank_words) > 0 and len(company_words) > 0:
            common_words = bank_words.intersection(company_words)
            similarity = len(common_words) / max(len(bank_words), len(company_words))
            return similarity * 0.6
        
        return 0.0
    
    @handle_service_errors('reconciliation_service')
    @with_database_transaction
    def create_reconciliation_records(self, matches: List[Dict[str, Any]]) -> List[ReconciliationRecord]:
        """
        Cria registros de reconciliação no banco de dados com detecção de anomalias
        """
        if not matches:
            raise InsufficientDataError('reconciliation record creation', 1, 0)
        
        logger.info(f"Creating {len(matches)} reconciliation records with anomaly detection")
        
        try:
            records = []
            creation_errors = 0
            anomalies_found = 0
            
            for i, match in enumerate(matches):
                try:
                    # Validate match data
                    if not self._validate_match_for_record_creation(match):
                        logger.warning(f"Match {i+1} failed validation for record creation")
                        creation_errors += 1
                        continue
                    
                    # Check for existing reconciliation record
                    existing_record = ReconciliationRecord.query.filter_by(
                        bank_transaction_id=match['bank_transaction'].id,
                        company_entry_id=match['company_entry'].id
                    ).first()
                    
                    if existing_record:
                        logger.debug(f"Reconciliation record already exists for transaction {match['bank_transaction'].id}")
                        records.append(existing_record)
                        continue
                    
                    # Create new record with anomaly detection
                    record = ReconciliationRecord(
                        bank_transaction_id=match['bank_transaction'].id,
                        company_entry_id=match['company_entry'].id,
                        match_score=match['match_score'],
                        status='pending'
                    )
                    
                    # Apply anomaly detection if available
                    anomaly_analysis = match.get('anomaly_analysis')
                    if anomaly_analysis and anomaly_analysis.get('is_anomaly', False):
                        # Flag the record as anomalous
                        record.flag_anomaly(
                            anomaly_type=anomaly_analysis.get('anomaly_type', 'unknown'),
                            severity=anomaly_analysis.get('anomaly_severity', 'medium'),
                            score=anomaly_analysis.get('anomaly_score', 0.5),
                            reason=anomaly_analysis.get('anomaly_reason', 'Anomaly detected during reconciliation')
                        )
                        
                        # Store detailed anomaly breakdown
                        if 'score_breakdown' in anomaly_analysis:
                            import json
                            record.score_breakdown = json.dumps(anomaly_analysis['score_breakdown'])
                        
                        if 'risk_factors' in anomaly_analysis:
                            import json
                            record.risk_factors = json.dumps(anomaly_analysis['risk_factors'])
                        
                        anomalies_found += 1
                        logger.warning(f"Created anomalous reconciliation record {i+1}: {anomaly_analysis.get('anomaly_type', 'unknown')}")
                    
                    db.session.add(record)
                    records.append(record)
                    
                except Exception as e:
                    logger.error(f"Error creating reconciliation record for match {i+1}: {str(e)}")
                    creation_errors += 1
                    continue
            
            if not records:
                raise ReconciliationError("No valid reconciliation records could be created")
            
            # Commit handled by @with_database_transaction decorator
            
            logger.info(f"Successfully created {len(records)} reconciliation records (errors: {creation_errors}, anomalies: {anomalies_found})")
            audit_logger.log_database_operation('insert', 'reconciliation_records', len(records), True)
            
            return records
            
        except Exception as e:
            if isinstance(e, (InsufficientDataError, ReconciliationError)):
                raise
            
            logger.error(f"Critical error creating reconciliation records: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to create reconciliation records: {str(e)}")
    
    def _validate_match_for_record_creation(self, match: Dict[str, Any]) -> bool:
        """Validate match data for record creation."""
        try:
            required_keys = ['bank_transaction', 'company_entry', 'match_score']
            if not all(key in match for key in required_keys):
                return False
            
            bank_transaction = match['bank_transaction']
            company_entry = match['company_entry']
            match_score = match['match_score']
            
            # Validate objects have required attributes
            if not all([
                hasattr(bank_transaction, 'id') and bank_transaction.id,
                hasattr(company_entry, 'id') and company_entry.id,
                isinstance(match_score, (int, float)) and 0 <= match_score <= 1
            ]):
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating match for record creation: {str(e)}")
            return False
    
    @handle_service_errors('reconciliation_service')
    def get_pending_reconciliations(self) -> List[ReconciliationRecord]:
        """
        Obtém todos os registros de reconciliação pendentes
        """
        try:
            records = ReconciliationRecord.query.filter_by(status='pending').all()
            logger.info(f"Retrieved {len(records)} pending reconciliation records")
            return records
            
        except Exception as e:
            logger.error(f"Error retrieving pending reconciliations: {str(e)}", exc_info=True)
            raise DatabaseError("Failed to retrieve pending reconciliations")
    
    @handle_service_errors('reconciliation_service')
    @with_database_transaction
    def confirm_reconciliation(self, reconciliation_id: int) -> bool:
        """
        Confirma uma reconciliação
        """
        if not reconciliation_id or reconciliation_id <= 0:
            raise ValidationError("Invalid reconciliation ID")
        
        logger.info(f"Confirming reconciliation record {reconciliation_id}")
        
        try:
            record = ReconciliationRecord.query.get(reconciliation_id)
            if not record:
                raise RecordNotFoundError('ReconciliationRecord', reconciliation_id)
            
            # Validate current status
            if record.status == 'confirmed':
                logger.info(f"Reconciliation {reconciliation_id} already confirmed")
                return True
            
            if record.status == 'rejected':
                logger.warning(f"Cannot confirm rejected reconciliation {reconciliation_id}")
                raise ReconciliationError("Cannot confirm a rejected reconciliation")
            
            # Update status
            record.status = 'confirmed'
            # Commit handled by @with_database_transaction decorator
            
            logger.info(f"Reconciliation {reconciliation_id} confirmed successfully")
            audit_logger.log_reconciliation_operation('confirm', 1, 'confirmed', {
                'reconciliation_id': reconciliation_id,
                'match_score': record.match_score
            })
            
            return True
            
        except Exception as e:
            if isinstance(e, (RecordNotFoundError, ValidationError, ReconciliationError)):
                raise
            
            logger.error(f"Error confirming reconciliation {reconciliation_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to confirm reconciliation: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    @with_database_transaction
    def reject_reconciliation(self, reconciliation_id: int) -> bool:
        """
        Rejeita uma reconciliação
        """
        if not reconciliation_id or reconciliation_id <= 0:
            raise ValidationError("Invalid reconciliation ID")
        
        logger.info(f"Rejecting reconciliation record {reconciliation_id}")
        
        try:
            record = ReconciliationRecord.query.get(reconciliation_id)
            if not record:
                raise RecordNotFoundError('ReconciliationRecord', reconciliation_id)
            
            # Validate current status
            if record.status == 'rejected':
                logger.info(f"Reconciliation {reconciliation_id} already rejected")
                return True
            
            if record.status == 'confirmed':
                logger.warning(f"Cannot reject confirmed reconciliation {reconciliation_id}")
                raise ReconciliationError("Cannot reject a confirmed reconciliation")
            
            # Update status
            record.status = 'rejected'
            # Commit handled by @with_database_transaction decorator
            
            logger.info(f"Reconciliation {reconciliation_id} rejected successfully")
            audit_logger.log_reconciliation_operation('reject', 1, 'rejected', {
                'reconciliation_id': reconciliation_id,
                'match_score': record.match_score
            })
            
            return True
            
        except Exception as e:
            if isinstance(e, (RecordNotFoundError, ValidationError, ReconciliationError)):
                raise
            
            logger.error(f"Error rejecting reconciliation {reconciliation_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to reject reconciliation: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    def get_reconciliation_report(self) -> Dict[str, Any]:
        """
        Gera um relatório de reconciliação
        """
        logger.info("Generating reconciliation report")
        
        try:
            # Get basic counts
            total_records = ReconciliationRecord.query.count()
            confirmed_records = ReconciliationRecord.query.filter_by(status='confirmed').count()
            pending_records = ReconciliationRecord.query.filter_by(status='pending').count()
            rejected_records = ReconciliationRecord.query.filter_by(status='rejected').count()
            
            # Calculate financial metrics
            financial_metrics = self._calculate_financial_metrics()
            
            # Calculate reconciliation rate
            reconciliation_rate = confirmed_records / total_records if total_records > 0 else 0
            
            # Get additional statistics
            avg_match_score = self._calculate_average_match_score()
            recent_activity = self._get_recent_reconciliation_activity()
            
            report = {
                'summary': {
                    'total_records': total_records,
                    'confirmed': confirmed_records,
                    'pending': pending_records,
                    'rejected': rejected_records,
                    'reconciliation_rate': round(reconciliation_rate, 4),
                    'average_match_score': round(avg_match_score, 4) if avg_match_score else 0
                },
                'financials': financial_metrics,
                'recent_activity': recent_activity,
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Reconciliation report generated. Total records: {total_records}, "
                       f"Confirmed: {confirmed_records}, Rate: {reconciliation_rate:.2%}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating reconciliation report: {str(e)}", exc_info=True)
            raise ReconciliationError(f"Failed to generate reconciliation report: {str(e)}")
    
    def _calculate_financial_metrics(self) -> Dict[str, float]:
        """Calculate financial metrics for reconciliation report."""
        try:
            confirmed_matches = ReconciliationRecord.query.filter_by(status='confirmed').all()
            
            total_reconciled_value = 0
            reconciled_credits = 0
            reconciled_debits = 0
            
            for record in confirmed_matches:
                try:
                    if record.bank_transaction and record.bank_transaction.amount is not None:
                        amount = record.bank_transaction.amount
                        total_reconciled_value += abs(amount)
                        
                        if amount > 0:
                            reconciled_credits += amount
                        else:
                            reconciled_debits += abs(amount)
                            
                except Exception as e:
                    logger.debug(f"Error processing record {record.id}: {str(e)}")
                    continue
            
            return {
                'total_reconciled_value': round(total_reconciled_value, 2),
                'reconciled_credits': round(reconciled_credits, 2),
                'reconciled_debits': round(reconciled_debits, 2),
                'net_reconciled_flow': round(reconciled_credits - reconciled_debits, 2)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating financial metrics: {str(e)}")
            return {
                'total_reconciled_value': 0,
                'reconciled_credits': 0,
                'reconciled_debits': 0,
                'net_reconciled_flow': 0
            }
    
    def _calculate_average_match_score(self) -> Optional[float]:
        """Calculate average match score for confirmed reconciliations."""
        try:
            confirmed_records = ReconciliationRecord.query.filter_by(status='confirmed').all()
            
            if not confirmed_records:
                return None
            
            total_score = sum(record.match_score for record in confirmed_records if record.match_score)
            return total_score / len(confirmed_records)
            
        except Exception as e:
            logger.debug(f"Error calculating average match score: {str(e)}")
            return None
    
    def _get_recent_reconciliation_activity(self) -> Dict[str, int]:
        """Get recent reconciliation activity (last 7 days)."""
        try:
            seven_days_ago = datetime.now() - timedelta(days=7)
            
            recent_confirmed = ReconciliationRecord.query.filter(
                ReconciliationRecord.status == 'confirmed',
                ReconciliationRecord.created_at >= seven_days_ago
            ).count()
            
            recent_rejected = ReconciliationRecord.query.filter(
                ReconciliationRecord.status == 'rejected',
                ReconciliationRecord.created_at >= seven_days_ago
            ).count()
            
            return {
                'confirmed_last_7_days': recent_confirmed,
                'rejected_last_7_days': recent_rejected,
                'total_activity_last_7_days': recent_confirmed + recent_rejected
            }
            
        except Exception as e:
            logger.debug(f"Error getting recent activity: {str(e)}")
            return {
                'confirmed_last_7_days': 0,
                'rejected_last_7_days': 0,
                'total_activity_last_7_days': 0
            }
    
    @handle_service_errors('reconciliation_service')
    @with_database_transaction
    def create_manual_match(self, bank_transaction_id: int, company_entry_id: int, 
                           user_confidence: float = 1.0, justification: str = None) -> ReconciliationRecord:
        """
        Create a manual match between bank transaction and company entry
        """
        if not all([bank_transaction_id, company_entry_id]):
            raise ValidationError("Both bank transaction ID and company entry ID are required")
        
        if user_confidence < 0 or user_confidence > 1:
            raise ValidationError("User confidence must be between 0 and 1")
        
        logger.info(f"Creating manual match between bank transaction {bank_transaction_id} and company entry {company_entry_id}")
        
        try:
            # Verify records exist
            bank_transaction = Transaction.query.get(bank_transaction_id)
            if not bank_transaction:
                raise RecordNotFoundError('Transaction', bank_transaction_id)
            
            company_entry = CompanyFinancial.query.get(company_entry_id)
            if not company_entry:
                raise RecordNotFoundError('CompanyFinancial', company_entry_id)
            
            # Check for existing match
            existing_record = ReconciliationRecord.query.filter_by(
                bank_transaction_id=bank_transaction_id,
                company_entry_id=company_entry_id
            ).first()
            
            if existing_record:
                logger.info(f"Match already exists between transaction {bank_transaction_id} and entry {company_entry_id}")
                return existing_record
            
            # Create manual match record
            record = ReconciliationRecord(
                bank_transaction_id=bank_transaction_id,
                company_entry_id=company_entry_id,
                match_score=user_confidence,  # Use user's confidence as the score
                status='pending',
                justification=justification
            )
            
            db.session.add(record)
            
            logger.info(f"Manual match created successfully with score {user_confidence}")
            audit_logger.log_reconciliation_operation('manual_match', 1, 'pending', {
                'bank_transaction_id': bank_transaction_id,
                'company_entry_id': company_entry_id,
                'user_confidence': user_confidence,
                'justification': justification
            })
            
            return record
            
        except Exception as e:
            if isinstance(e, (ValidationError, RecordNotFoundError)):
                raise
            
            logger.error(f"Error creating manual match: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to create manual match: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    @with_database_transaction
    def batch_confirm_reconciliations(self, reconciliation_ids: List[int]) -> Dict[str, Any]:
        """
        Confirm multiple reconciliation records in batch
        """
        if not reconciliation_ids:
            raise ValidationError("At least one reconciliation ID is required")
        
        logger.info(f"Batch confirming {len(reconciliation_ids)} reconciliation records")
        
        try:
            confirmed_count = 0
            error_count = 0
            errors = []
            
            for rec_id in reconciliation_ids:
                try:
                    success = self.confirm_reconciliation(rec_id)
                    if success:
                        confirmed_count += 1
                    else:
                        error_count += 1
                        errors.append(f"Failed to confirm reconciliation {rec_id}")
                except Exception as e:
                    error_count += 1
                    errors.append(f"Error confirming reconciliation {rec_id}: {str(e)}")
                    continue
            
            result = {
                'total_processed': len(reconciliation_ids),
                'confirmed_count': confirmed_count,
                'error_count': error_count,
                'errors': errors,
                'success_rate': confirmed_count / len(reconciliation_ids) if reconciliation_ids else 0
            }
            
            logger.info(f"Batch confirmation completed: {confirmed_count} confirmed, {error_count} errors")
            audit_logger.log_reconciliation_operation('batch_confirm', confirmed_count, 'confirmed', {
                'total_processed': len(reconciliation_ids),
                'success_rate': result['success_rate']
            })
            
            return result
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            
            logger.error(f"Error in batch confirmation: {str(e)}", exc_info=True)
            raise DatabaseError(f"Batch confirmation failed: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    @with_database_transaction
    def batch_reject_reconciliations(self, reconciliation_ids: List[int]) -> Dict[str, Any]:
        """
        Reject multiple reconciliation records in batch
        """
        if not reconciliation_ids:
            raise ValidationError("At least one reconciliation ID is required")
        
        logger.info(f"Batch rejecting {len(reconciliation_ids)} reconciliation records")
        
        try:
            rejected_count = 0
            error_count = 0
            errors = []
            
            for rec_id in reconciliation_ids:
                try:
                    success = self.reject_reconciliation(rec_id)
                    if success:
                        rejected_count += 1
                    else:
                        error_count += 1
                        errors.append(f"Failed to reject reconciliation {rec_id}")
                except Exception as e:
                    error_count += 1
                    errors.append(f"Error rejecting reconciliation {rec_id}: {str(e)}")
                    continue
            
            result = {
                'total_processed': len(reconciliation_ids),
                'rejected_count': rejected_count,
                'error_count': error_count,
                'errors': errors,
                'success_rate': rejected_count / len(reconciliation_ids) if reconciliation_ids else 0
            }
            
            logger.info(f"Batch rejection completed: {rejected_count} rejected, {error_count} errors")
            audit_logger.log_reconciliation_operation('batch_reject', rejected_count, 'rejected', {
                'total_processed': len(reconciliation_ids),
                'success_rate': result['success_rate']
            })
            
            return result
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            
            logger.error(f"Error in batch rejection: {str(e)}", exc_info=True)
            raise DatabaseError(f"Batch rejection failed: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    def adjust_match_score(self, reconciliation_id: int, new_score: float, justification: str = None) -> bool:
        """
        Adjust the match score of an existing reconciliation record
        """
        if not reconciliation_id or reconciliation_id <= 0:
            raise ValidationError("Invalid reconciliation ID")
        
        if new_score < 0 or new_score > 1:
            raise ValidationError("Match score must be between 0 and 1")
        
        logger.info(f"Adjusting match score for reconciliation {reconciliation_id} to {new_score}")
        
        try:
            record = ReconciliationRecord.query.get(reconciliation_id)
            if not record:
                raise RecordNotFoundError('ReconciliationRecord', reconciliation_id)
            
            if record.status in ['confirmed', 'rejected']:
                raise ReconciliationError(f"Cannot adjust score for {record.status} reconciliation")
            
            # Store old score for audit
            old_score = record.match_score
            
            # Update score
            record.match_score = new_score
            if justification:
                record.justification = justification
            
            db.session.commit()
            
            logger.info(f"Match score adjusted from {old_score} to {new_score} for reconciliation {reconciliation_id}")
            audit_logger.log_reconciliation_operation('adjust_score', 1, 'adjusted', {
                'reconciliation_id': reconciliation_id,
                'old_score': old_score,
                'new_score': new_score,
                'justification': justification
            })
            
            return True
            
        except Exception as e:
            if isinstance(e, (ValidationError, RecordNotFoundError, ReconciliationError)):
                raise
            
            logger.error(f"Error adjusting match score: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to adjust match score: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    def get_reconciliation_suggestions(self, bank_transaction_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get AI-powered matching suggestions for a specific bank transaction
        """
        if not bank_transaction_id or bank_transaction_id <= 0:
            raise ValidationError("Invalid bank transaction ID")
        
        logger.info(f"Getting reconciliation suggestions for bank transaction {bank_transaction_id}")
        
        try:
            # Get the bank transaction
            bank_transaction = Transaction.query.get(bank_transaction_id)
            if not bank_transaction:
                raise RecordNotFoundError('Transaction', bank_transaction_id)
            
            # Get all potential company entries that aren't already matched
            already_matched_entries = db.session.query(ReconciliationRecord.company_entry_id).filter(
                ReconciliationRecord.bank_transaction_id == bank_transaction_id,
                ReconciliationRecord.status.in_(['pending', 'confirmed'])
            ).all()
            
            already_matched_ids = [entry[0] for entry in already_matched_entries]
            
            # Get available company entries
            available_entries = CompanyFinancial.query.filter(
                ~CompanyFinancial.id.in_(already_matched_ids)
            ).all()
            
            if not available_entries:
                return []
            
            # Calculate scores for all available entries
            suggestions = []
            for company_entry in available_entries:
                try:
                    score = self._calculate_enhanced_match_score(bank_transaction, company_entry)
                    
                    if score >= self.config.minimum_match_threshold:
                        suggestion = {
                            'company_entry': company_entry,
                            'match_score': score,
                            'score_breakdown': self._get_score_breakdown(bank_transaction, company_entry, score),
                            'suggestion_reason': self._generate_suggestion_reason(bank_transaction, company_entry, score)
                        }
                        suggestions.append(suggestion)
                        
                except Exception as e:
                    logger.debug(f"Error calculating suggestion for entry {company_entry.id}: {str(e)}")
                    continue
            
            # Sort by score and limit results
            suggestions.sort(key=lambda x: x['match_score'], reverse=True)
            return suggestions[:limit]
            
        except Exception as e:
            if isinstance(e, (ValidationError, RecordNotFoundError)):
                raise
            
            logger.error(f"Error getting reconciliation suggestions: {str(e)}", exc_info=True)
            raise ReconciliationError(f"Failed to get reconciliation suggestions: {str(e)}")
    
    def _generate_suggestion_reason(self, bank_transaction: Transaction, company_entry: CompanyFinancial, score: float) -> str:
        """Generate human-readable reason for match suggestion"""
        reasons = []
        
        # Amount comparison
        amount_diff = abs(abs(bank_transaction.amount) - abs(company_entry.amount))
        if amount_diff < self.config.minimum_amount_threshold:
            reasons.append("Valores idênticos")
        elif amount_diff <= (abs(bank_transaction.amount) * self.config.maximum_amount_diff_percent):
            reasons.append("Valores muito próximos")
        
        # Date comparison
        if bank_transaction.date and company_entry.date:
            date_diff = abs((bank_transaction.date - company_entry.date).days)
            if date_diff == 0:
                reasons.append("Mesma data")
            elif date_diff <= 3:
                reasons.append("Datas próximas")
        
        # Description comparison
        if self.config.enable_fuzzy_matching:
            desc_similarity = self._calculate_fuzzy_description_similarity(
                bank_transaction.description, company_entry.description
            )
        else:
            desc_similarity = self._calculate_description_similarity(
                bank_transaction.description, company_entry.description
            )
        
        if desc_similarity >= 0.8:
            reasons.append("Descrições muito similares")
        elif desc_similarity >= 0.6:
            reasons.append("Descrições similares")
        
        # Overall score
        if score >= 0.9:
            reasons.insert(0, "Correspondência excelente")
        elif score >= 0.8:
            reasons.insert(0, "Correspondência forte")
        elif score >= 0.7:
            reasons.insert(0, "Correspondência provável")
        
        return reasons
    
    # Phase 2 Integration Enhancement Methods
    
    def _calculate_entity_matching_bonus(self, bank_desc: str, company_desc: str) -> float:
        """Calculate entity matching bonus using Phase 2 entity extraction"""
        try:
            # Use Phase 2 normalizer to extract entities
            bank_result = self.normalizer.normalize_text(bank_desc)
            company_result = self.normalizer.normalize_text(company_desc)
            
            bank_entities = bank_result.entities
            company_entities = company_result.entities
            
            # Calculate entity overlap score
            entity_matches = 0
            total_entities = 0
            
            for entity_type in ['company', 'payment_method', 'tax', 'location']:
                bank_set = set(bank_entities.get(entity_type, []))
                company_set = set(company_entities.get(entity_type, []))
                
                if bank_set or company_set:
                    # Calculate Jaccard similarity for this entity type
                    intersection = bank_set.intersection(company_set)
                    union = bank_set.union(company_set)
                    
                    if union:
                        entity_matches += len(intersection)
                        total_entities += len(union)
            
            # Return entity matching bonus
            if total_entities > 0:
                return entity_matches / total_entities
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating entity matching bonus: {str(e)}")
            return 0.0
    
    def _update_performance_metrics(self, processing_time: float, contextual_result):
        """Update performance metrics for Phase 2 integration"""
        self.performance_metrics['total_matches_processed'] += 1
        
        # Track contextual matches
        if contextual_result.total_bonus > 0:
            self.performance_metrics['contextual_matches'] += 1
        
        # Track pattern-based matches
        if contextual_result.patterns_used > 0:
            self.performance_metrics['pattern_based_matches'] += 1
        
        # Update average processing time
        if self.performance_metrics['average_processing_time'] == 0:
            self.performance_metrics['average_processing_time'] = processing_time
        else:
            alpha = 0.1
            current_avg = self.performance_metrics['average_processing_time']
            self.performance_metrics['average_processing_time'] = (
                alpha * processing_time + (1 - alpha) * current_avg
            )
        
        # Update integration accuracy
        total = self.performance_metrics['total_matches_processed']
        if total > 0:
            contextual_ratio = self.performance_metrics['contextual_matches'] / total
            pattern_ratio = self.performance_metrics['pattern_based_matches'] / total
            self.performance_metrics['integration_accuracy'] = (contextual_ratio + pattern_ratio) / 2
    
    def get_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics including Phase 2 integration stats"""
        return {
            **self.performance_metrics,
            'context_matcher_metrics': self.context_matcher.get_performance_metrics(),
            'normalizer_metrics': self.normalizer.get_performance_metrics(),
            'integration_efficiency': {
                'contextual_match_rate': (
                    self.performance_metrics['contextual_matches'] / 
                    max(self.performance_metrics['total_matches_processed'], 1)
                ),
                'pattern_utilization_rate': (
                    self.performance_metrics['pattern_based_matches'] / 
                    max(self.performance_metrics['total_matches_processed'], 1)
                ),
                'average_processing_time': self.performance_metrics['average_processing_time']
            }
        }
    
    def configure_phase2_integration(self, 
                                   normalization_config: NormalizationConfig = None,
                                   matching_config: MatchingConfig = None):
        """Configure Phase 2 integration with custom settings"""
        try:
            # Update normalizer configuration
            if normalization_config:
                self.normalizer.update_config(normalization_config)
                logger.info("Updated normalizer configuration for Phase 2 integration")
            
            # Update matcher configuration
            if matching_config:
                self.context_matcher.update_config(matching_config)
                logger.info("Updated context matcher configuration for Phase 2 integration")
            
            return True
            
        except Exception as e:
            logger.error(f"Error configuring Phase 2 integration: {str(e)}")
            return False
    
    def get_contextual_suggestions(self, bank_transaction: Transaction, 
                                 available_entries: List[CompanyFinancial]) -> List[Dict[str, Any]]:
        """Get contextual matching suggestions using Phase 2 capabilities"""
        try:
            # Use enhanced context matcher to get suggestions
            suggestions = self.context_matcher.get_matching_suggestions(
                bank_transaction, available_entries
            )
            
            # Convert to enhanced format with reconciliation-specific data
            enhanced_suggestions = []
            for suggestion in suggestions:
                enhanced_suggestion = {
                    'company_entry': suggestion.company_entry,
                    'confidence_score': suggestion.confidence_score,
                    'context_factors': suggestion.context_factors,
                    'explanation': suggestion.explanation,
                    'patterns_used': suggestion.patterns_used,
                    'predicted_missing_info': suggestion.predicted_missing_info,
                    'user_behavior_score': suggestion.user_behavior_score,
                    'reconciliation_score': self._calculate_enhanced_match_score(
                        bank_transaction, suggestion.company_entry
                    )
                }
                enhanced_suggestions.append(enhanced_suggestion)
            
            return enhanced_suggestions
            
        except Exception as e:
            logger.error(f"Error getting contextual suggestions: {str(e)}")
            return []
    
    def predict_missing_information(self, transaction: Transaction) -> Dict[str, Any]:
        """Predict missing information using Phase 2 capabilities"""
        try:
            # Use Phase 2 prediction capabilities
            prediction = self.context_matcher.predict_missing_information(transaction)
            
            # Convert to dictionary format
            return {
                'predicted_suppliers': prediction.predicted_suppliers,
                'predicted_payment_methods': prediction.predicted_payment_methods,
                'predicted_categories': prediction.predicted_categories,
                'amount_variance_pattern': prediction.amount_variance_pattern,
                'confidence_score': prediction.confidence_score,
                'based_on_patterns': prediction.based_on_patterns
            }
            
        except Exception as e:
            logger.error(f"Error predicting missing information: {str(e)}")
            return {}
    
    def export_phase2_patterns(self, file_path: str) -> bool:
        """Export Phase 2 learned patterns"""
        try:
            # Export context matcher patterns
            matcher_success = self.context_matcher.export_patterns(file_path)
            
            if matcher_success:
                logger.info(f"Exported Phase 2 patterns to {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error exporting Phase 2 patterns: {str(e)}")
            return False
    
    def import_phase2_patterns(self, file_path: str) -> bool:
        """Import Phase 2 learned patterns"""
        try:
            # Import context matcher patterns
            matcher_success = self.context_matcher.import_patterns(file_path)
            
            if matcher_success:
                logger.info(f"Imported Phase 2 patterns from {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error importing Phase 2 patterns: {str(e)}")
            return False
    
    # Anomaly Detection Integration Methods
    
    @handle_service_errors('reconciliation_service')
    @with_database_transaction
    def process_reconciliation_with_anomaly_detection(self, bank_transactions: List[Transaction], 
                                                    company_entries: List[CompanyFinancial], 
                                                    user_id: int = None) -> Dict[str, Any]:
        """
        Process reconciliation with comprehensive anomaly detection and human supervision
        """
        if not bank_transactions or not company_entries:
            raise InsufficientDataError('reconciliation with anomaly detection', 1, 0)
        
        logger.info(f"Starting reconciliation with anomaly detection. Bank transactions: {len(bank_transactions)}, Company entries: {len(company_entries)}")
        
        try:
            # Use interactive anomaly manager for comprehensive processing
            anomaly_manager = self._get_anomaly_manager()
            result = anomaly_manager.process_reconciliation_with_anomaly_detection(
                bank_transactions, company_entries, user_id
            )
            
            # Update performance metrics
            self.performance_metrics['anomalies_detected'] += result.get('anomalies_detected', 0)
            self.performance_metrics['human_supervision_interactions'] += result.get('human_interventions', 0)
            
            logger.info(f"Reconciliation with anomaly detection completed. "
                       f"Matches: {result.get('total_matches', 0)}, "
                       f"Anomalies: {result.get('anomalies_detected', 0)}, "
                       f"Human interventions: {result.get('human_interventions', 0)}")
            
            return result
            
        except Exception as e:
            if isinstance(e, InsufficientDataError):
                raise
            
            logger.error(f"Error in reconciliation with anomaly detection: {str(e)}", exc_info=True)
            raise ReconciliationError(f"Reconciliation with anomaly detection failed: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    @with_database_transaction
    def review_anomaly(self, reconciliation_id: int, user_id: int, action: str, 
                      justification: str = None) -> Dict[str, Any]:
        """
        Review and resolve an anomaly with human supervision
        """
        if not reconciliation_id or reconciliation_id <= 0:
            raise ValidationError("Invalid reconciliation ID")
        
        if not user_id or user_id <= 0:
            raise ValidationError("Valid user ID is required for anomaly review")
        
        if action not in ['confirmed', 'dismissed', 'escalated']:
            raise ValidationError("Action must be 'confirmed', 'dismissed', or 'escalated'")
        
        logger.info(f"User {user_id} reviewing anomaly in reconciliation {reconciliation_id} with action: {action}")
        
        try:
            # Get the reconciliation record
            record = ReconciliationRecord.query.get(reconciliation_id)
            if not record:
                raise RecordNotFoundError('ReconciliationRecord', reconciliation_id)
            
            if not record.is_anomaly:
                raise ReconciliationError(f"Reconciliation {reconciliation_id} is not flagged as anomalous")
            
            # Use interactive anomaly manager to process the review
            anomaly_manager = self._get_anomaly_manager()
            result = anomaly_manager.review_anomaly(record, user_id, action, justification)
            
            # Update the record based on the review
            record.resolve_anomaly(action, justification or f"Anomaly {action} by user {user_id}", user_id)
            
            # Update performance metrics
            self.performance_metrics['anomalies_resolved'] += 1
            self.performance_metrics['human_supervision_interactions'] += 1
            
            logger.info(f"Anomaly review completed for reconciliation {reconciliation_id}. Action: {action}")
            audit_logger.log_anomaly_operation('review', reconciliation_id, action, {
                'user_id': user_id,
                'anomaly_type': record.anomaly_type,
                'severity': record.anomaly_severity,
                'justification': justification
            })
            
            return result
            
        except Exception as e:
            if isinstance(e, (ValidationError, RecordNotFoundError, ReconciliationError)):
                raise
            
            logger.error(f"Error reviewing anomaly for reconciliation {reconciliation_id}: {str(e)}", exc_info=True)
            raise ReconciliationError(f"Anomaly review failed: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    def get_anomaly_workflow_suggestions(self, reconciliation_id: int) -> Dict[str, Any]:
        """
        Get AI-powered workflow suggestions for anomaly resolution
        """
        if not reconciliation_id or reconciliation_id <= 0:
            raise ValidationError("Invalid reconciliation ID")
        
        try:
            # Get the reconciliation record
            record = ReconciliationRecord.query.get(reconciliation_id)
            if not record:
                raise RecordNotFoundError('ReconciliationRecord', reconciliation_id)
            
            if not record.is_anomaly:
                raise ReconciliationError(f"Reconciliation {reconciliation_id} is not flagged as anomalous")
            
            # Get workflow suggestions from anomaly manager
            anomaly_manager = self._get_anomaly_manager()
            suggestions = anomaly_manager.get_anomaly_workflow_suggestions(record)
            
            logger.info(f"Generated workflow suggestions for anomaly in reconciliation {reconciliation_id}")
            
            return suggestions
            
        except Exception as e:
            if isinstance(e, (ValidationError, RecordNotFoundError, ReconciliationError)):
                raise
            
            logger.error(f"Error getting workflow suggestions for reconciliation {reconciliation_id}: {str(e)}", exc_info=True)
            raise ReconciliationError(f"Failed to get workflow suggestions: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive anomaly statistics and metrics
        """
        logger.info("Generating anomaly statistics")
        
        try:
            # Get basic anomaly counts
            total_records = ReconciliationRecord.query.count()
            anomalous_records = ReconciliationRecord.query.filter_by(is_anomaly=True).count()
            
            # Get anomaly breakdown by type
            anomaly_types = db.session.query(
                ReconciliationRecord.anomaly_type,
                db.func.count(ReconciliationRecord.id).label('count')
            ).filter(ReconciliationRecord.is_anomaly == True).group_by(ReconciliationRecord.anomaly_type).all()
            
            # Get anomaly breakdown by severity
            anomaly_severities = db.session.query(
                ReconciliationRecord.anomaly_severity,
                db.func.count(ReconciliationRecord.id).label('count')
            ).filter(ReconciliationRecord.is_anomaly == True).group_by(ReconciliationRecord.anomaly_severity).all()
            
            # Get anomaly resolution statistics
            resolved_anomalies = ReconciliationRecord.query.filter(
                ReconciliationRecord.is_anomaly == True,
                ReconciliationRecord.anomaly_action.isnot(None)
            ).count()
            
            pending_anomalies = anomalous_records - resolved_anomalies
            
            # Calculate anomaly rate
            anomaly_rate = anomalous_records / total_records if total_records > 0 else 0
            resolution_rate = resolved_anomalies / anomalous_records if anomalous_records > 0 else 0
            
            # Get recent anomaly activity (last 7 days)
            seven_days_ago = datetime.now() - timedelta(days=7)
            recent_anomalies = ReconciliationRecord.query.filter(
                ReconciliationRecord.is_anomaly == True,
                ReconciliationRecord.anomaly_detected_at >= seven_days_ago
            ).count()
            
            recent_resolutions = ReconciliationRecord.query.filter(
                ReconciliationRecord.is_anomaly == True,
                ReconciliationRecord.anomaly_reviewed_at >= seven_days_ago
            ).count()
            
            # Compile statistics
            statistics = {
                'summary': {
                    'total_records': total_records,
                    'anomalous_records': anomalous_records,
                    'resolved_anomalies': resolved_anomalies,
                    'pending_anomalies': pending_anomalies,
                    'anomaly_rate': round(anomaly_rate, 4),
                    'resolution_rate': round(resolution_rate, 4)
                },
                'breakdown': {
                    'by_type': {row.anomaly_type: row.count for row in anomaly_types},
                    'by_severity': {row.anomaly_severity: row.count for row in anomaly_severities}
                },
                'recent_activity': {
                    'anomalies_detected_last_7_days': recent_anomalies,
                    'anomalies_resolved_last_7_days': recent_resolutions,
                    'net_anomaly_change': recent_anomalies - recent_resolutions
                },
                'performance_metrics': {
                    'anomalies_detected': self.performance_metrics['anomalies_detected'],
                    'anomalies_resolved': self.performance_metrics['anomalies_resolved'],
                    'human_supervision_interactions': self.performance_metrics['human_supervision_interactions']
                },
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Anomaly statistics generated. Total anomalies: {anomalous_records}, "
                       f"Resolution rate: {resolution_rate:.2%}")
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error generating anomaly statistics: {str(e)}", exc_info=True)
            raise ReconciliationError(f"Failed to generate anomaly statistics: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    def get_anomalous_reconciliations(self, limit: int = 50, offset: int = 0, 
                                    severity_filter: str = None, 
                                    status_filter: str = None) -> Dict[str, Any]:
        """
        Get paginated list of anomalous reconciliations with filtering options
        """
        if limit < 1 or limit > 100:
            raise ValidationError("Limit must be between 1 and 100")
        
        if offset < 0:
            raise ValidationError("Offset must be non-negative")
        
        logger.info(f"Getting anomalous reconciliations. Limit: {limit}, Offset: {offset}")
        
        try:
            # Build base query
            query = ReconciliationRecord.query.filter_by(is_anomaly=True)
            
            # Apply filters
            if severity_filter:
                query = query.filter_by(anomaly_severity=severity_filter)
            
            if status_filter:
                if status_filter == 'pending':
                    query = query.filter(ReconciliationRecord.anomaly_action.is_(None))
                elif status_filter == 'resolved':
                    query = query.filter(ReconciliationRecord.anomaly_action.isnot(None))
            
            # Get total count for pagination
            total_count = query.count()
            
            # Get paginated results
            records = query.order_by(ReconciliationRecord.anomaly_detected_at.desc()).offset(offset).limit(limit).all()
            
            # Convert to dictionary format
            anomalous_records = []
            for record in records:
                anomalous_records.append({
                    'id': record.id,
                    'bank_transaction_id': record.bank_transaction_id,
                    'company_entry_id': record.company_entry_id,
                    'match_score': record.match_score,
                    'anomaly_type': record.anomaly_type,
                    'anomaly_severity': record.anomaly_severity,
                    'anomaly_score': record.anomaly_score,
                    'anomaly_reason': record.anomaly_reason,
                    'anomaly_detected_at': record.anomaly_detected_at.isoformat() if record.anomaly_detected_at else None,
                    'anomaly_action': record.anomaly_action,
                    'anomaly_reviewed_at': record.anomaly_reviewed_at.isoformat() if record.anomaly_reviewed_at else None,
                    'status': 'resolved' if record.anomaly_action else 'pending',
                    'bank_transaction': record.bank_transaction.to_dict() if record.bank_transaction else None,
                    'company_entry': record.company_entry.to_dict() if record.company_entry else None
                })
            
            result = {
                'records': anomalous_records,
                'pagination': {
                    'total_count': total_count,
                    'limit': limit,
                    'offset': offset,
                    'has_more': offset + limit < total_count
                },
                'filters': {
                    'severity': severity_filter,
                    'status': status_filter
                }
            }
            
            logger.info(f"Retrieved {len(anomalous_records)} anomalous reconciliations (total: {total_count})")
            
            return result
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            
            logger.error(f"Error getting anomalous reconciliations: {str(e)}", exc_info=True)
            raise ReconciliationError(f"Failed to get anomalous reconciliations: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    @with_database_transaction
    def escalate_anomaly(self, reconciliation_id: int, user_id: int, escalation_reason: str, 
                        target_user_id: int = None) -> Dict[str, Any]:
        """
        Escalate an anomaly to a higher authority or specialist
        """
        if not reconciliation_id or reconciliation_id <= 0:
            raise ValidationError("Invalid reconciliation ID")
        
        if not user_id or user_id <= 0:
            raise ValidationError("Valid user ID is required for escalation")
        
        if not escalation_reason or len(escalation_reason.strip()) < 10:
            raise ValidationError("Escalation reason must be at least 10 characters long")
        
        logger.info(f"User {user_id} escalating anomaly in reconciliation {reconciliation_id}")
        
        try:
            # Get the reconciliation record
            record = ReconciliationRecord.query.get(reconciliation_id)
            if not record:
                raise RecordNotFoundError('ReconciliationRecord', reconciliation_id)
            
            if not record.is_anomaly:
                raise ReconciliationError(f"Reconciliation {reconciliation_id} is not flagged as anomalous")
            
            # Update anomaly severity and status
            if record.anomaly_severity != 'critical':
                record.anomaly_severity = 'critical'
            
            # Update escalation information
            record.anomaly_action = 'escalated'
            record.anomaly_justification = f"Escalated by user {user_id}: {escalation_reason}"
            record.anomaly_reviewed_by = user_id
            record.anomaly_reviewed_at = datetime.utcnow()
            
            # Log the escalation
            audit_logger.log_anomaly_operation('escalate', reconciliation_id, 'escalated', {
                'user_id': user_id,
                'target_user_id': target_user_id,
                'escalation_reason': escalation_reason,
                'anomaly_type': record.anomaly_type,
                'severity': record.anomaly_severity
            })
            
            # Update performance metrics
            self.performance_metrics['human_supervision_interactions'] += 1
            
            logger.info(f"Anomaly escalated successfully for reconciliation {reconciliation_id}")
            
            return {
                'success': True,
                'message': 'Anomaly escalated successfully',
                'reconciliation_id': reconciliation_id,
                'escalated_by': user_id,
                'escalated_at': record.anomaly_reviewed_at.isoformat(),
                'target_user_id': target_user_id
            }
            
        except Exception as e:
            if isinstance(e, (ValidationError, RecordNotFoundError, ReconciliationError)):
                raise
            
            logger.error(f"Error escalating anomaly for reconciliation {reconciliation_id}: {str(e)}", exc_info=True)
            raise ReconciliationError(f"Anomaly escalation failed: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    def bulk_anomaly_review(self, reconciliation_ids: List[int], user_id: int, action: str, 
                           justification: str = None) -> Dict[str, Any]:
        """
        Review multiple anomalies in bulk
        """
        if not reconciliation_ids:
            raise ValidationError("At least one reconciliation ID is required")
        
        if not user_id or user_id <= 0:
            raise ValidationError("Valid user ID is required for bulk review")
        
        if action not in ['confirmed', 'dismissed', 'escalated']:
            raise ValidationError("Action must be 'confirmed', 'dismissed', or 'escalated'")
        
        logger.info(f"User {user_id} performing bulk anomaly review on {len(reconciliation_ids)} records")
        
        try:
            processed_count = 0
            error_count = 0
            errors = []
            
            for rec_id in reconciliation_ids:
                try:
                    result = self.review_anomaly(rec_id, user_id, action, justification)
                    if result.get('success', False):
                        processed_count += 1
                    else:
                        error_count += 1
                        errors.append(f"Failed to review reconciliation {rec_id}")
                except Exception as e:
                    error_count += 1
                    errors.append(f"Error reviewing reconciliation {rec_id}: {str(e)}")
                    continue
            
            result = {
                'total_processed': len(reconciliation_ids),
                'processed_count': processed_count,
                'error_count': error_count,
                'errors': errors,
                'success_rate': processed_count / len(reconciliation_ids) if reconciliation_ids else 0,
                'action': action,
                'performed_by': user_id
            }
            
            logger.info(f"Bulk anomaly review completed: {processed_count} processed, {error_count} errors")
            
            return result
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            
            logger.error(f"Error in bulk anomaly review: {str(e)}", exc_info=True)
            raise ReconciliationError(f"Bulk anomaly review failed: {str(e)}")
