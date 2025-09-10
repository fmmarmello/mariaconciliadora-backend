"""
Context-aware matching system for financial reconciliation
Learns from historical patterns and user behavior to improve matching accuracy
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict, Counter
import json
import logging
from dataclasses import dataclass, field
from enum import Enum

from src.models.transaction import Transaction, ReconciliationRecord
from src.models.company_financial import CompanyFinancial
from src.models.user import db
from src.services.data_normalization import BrazilianDataNormalizer, NormalizationResult

logger = logging.getLogger(__name__)

class MatchingMode(Enum):
    """Matching behavior modes"""
    CONSERVATIVE = "conservative"  # High precision, lower recall
    BALANCED = "balanced"        # Balanced precision and recall
    AGGRESSIVE = "aggressive"    # Higher recall, lower precision
    LEARNING = "learning"        # Adaptive learning mode

class PatternType(Enum):
    """Types of patterns that can be learned"""
    DESCRIPTION = "description"
    AMOUNT = "amount"
    DATE = "date"
    ENTITY = "entity"
    TIME = "time"
    SUPPLIER = "supplier"
    USER_BEHAVIOR = "user_behavior"

@dataclass
class MatchingConfig:
    """Configuration for context-aware matching"""
    mode: MatchingMode = MatchingMode.BALANCED
    enable_learning: bool = True
    cache_update_interval_hours: int = 24
    min_pattern_frequency: float = 0.01
    min_pattern_occurrences: int = 2
    max_suggestions: int = 10
    suggestion_threshold: float = 0.6
    enable_user_specific_patterns: bool = True
    pattern_significance_threshold: float = 0.1
    auto_cache_update: bool = True
    debug_mode: bool = False

@dataclass
class PatternInfo:
    """Information about a learned pattern"""
    count: int
    frequency: float
    significance: str
    confidence: float = 0.0
    last_seen: datetime = field(default_factory=datetime.now)
    trend: str = "stable"

@dataclass
class MatchingSuggestion:
    """Matching suggestion with context information"""
    company_entry: CompanyFinancial
    confidence_score: float
    context_factors: Dict[str, float]
    explanation: str
    patterns_used: int
    predicted_missing_info: Dict[str, Any] = field(default_factory=dict)
    user_behavior_score: float = 0.0

@dataclass
class ContextualMatchResult:
    """Result of contextual matching analysis"""
    score: float
    base_score: float
    context_factors: Dict[str, float]
    total_bonus: float
    explanation: str
    patterns_used: int
    user_id: Optional[int] = None
    processing_time: float = 0.0

@dataclass
class PredictionResult:
    """Result of missing information prediction"""
    predicted_suppliers: List[str]
    predicted_payment_methods: List[str]
    predicted_categories: List[str]
    amount_variance_pattern: str
    confidence_score: float
    based_on_patterns: int

class ContextAwareMatcher:
    """
    Context-aware matching system that learns from historical patterns
    Advanced version with Phase 2 integration capabilities
    """
    
    def __init__(self, config: MatchingConfig = None):
        self.config = config or MatchingConfig()
        self.normalizer = BrazilianDataNormalizer()
        self.pattern_cache = {}
        self.user_patterns = defaultdict(dict)
        self.performance_metrics = {
            'total_patterns_analyzed': 0,
            'cache_updates': 0,
            'suggestions_generated': 0,
            'successful_predictions': 0,
            'average_processing_time': 0.0
        }
        self.last_cache_update = None
        self.learning_enabled = self.config.enable_learning
        
    def analyze_historical_patterns(self, user_id: int = None) -> Dict[str, Any]:
        """
        Analyze historical reconciliation patterns to learn matching rules
        Enhanced with Phase 2 pattern analysis
        """
        start_time = datetime.now()
        
        try:
            # Get confirmed reconciliations for pattern analysis
            query = ReconciliationRecord.query.filter_by(status='confirmed')
            if user_id and self.config.enable_user_specific_patterns:
                # Assuming user_id is stored in reconciliation records or related tables
                pass
            
            confirmed_matches = query.all()
            
            patterns = {
                'description_patterns': defaultdict(int),
                'amount_patterns': defaultdict(int),
                'date_patterns': defaultdict(int),
                'entity_patterns': defaultdict(int),
                'time_patterns': defaultdict(int),
                'supplier_patterns': defaultdict(int),
                'user_behavior_patterns': defaultdict(int)
            }
            
            for record in confirmed_matches:
                if not (record.bank_transaction and record.company_entry):
                    continue
                
                # Analyze description patterns
                bank_desc = self.normalizer.normalize_text(record.bank_transaction.description).normalized_text
                company_desc = self.normalizer.normalize_text(record.company_entry.description).normalized_text
                
                if bank_desc and company_desc:
                    # Store description mapping patterns
                    patterns['description_patterns'][f"{bank_desc}||{company_desc}"] += 1
                    
                    # Extract and store entity patterns
                    bank_entities = self.normalizer.normalize_text(record.bank_transaction.description).entities
                    company_entities = self.normalizer.normalize_text(record.company_entry.description).entities
                    
                    for entity_type in ['companies', 'payment_methods']:
                        for bank_entity in bank_entities.get(entity_type, []):
                            for company_entity in company_entities.get(entity_type, []):
                                patterns['entity_patterns'][f"{entity_type}:{bank_entity}||{company_entity}"] += 1
                
                # Analyze amount patterns
                bank_amount = abs(record.bank_transaction.amount)
                company_amount = abs(record.company_entry.amount)
                amount_diff = abs(bank_amount - company_amount)
                
                if amount_diff < 0.01:
                    patterns['amount_patterns']['exact'] += 1
                elif amount_diff / max(bank_amount, company_amount) < 0.05:
                    patterns['amount_patterns']['close'] += 1
                else:
                    patterns['amount_patterns']['variable'] += 1
                
                # Analyze date patterns
                if record.bank_transaction.date and record.company_entry.date:
                    date_diff = abs((record.bank_transaction.date - record.company_entry.date).days)
                    
                    if date_diff == 0:
                        patterns['date_patterns']['same_day'] += 1
                    elif date_diff <= 3:
                        patterns['date_patterns']['few_days'] += 1
                    elif date_diff <= 7:
                        patterns['date_patterns']['week'] += 1
                    else:
                        patterns['date_patterns']['extended'] += 1
                
                # Analyze time patterns (when reconciliations typically happen)
                if record.created_at:
                    hour = record.created_at.hour
                    patterns['time_patterns'][f"hour_{hour}"] += 1
                
                # Analyze supplier/customer patterns
                if record.bank_transaction.description:
                    supplier = self._extract_supplier_name(record.bank_transaction.description)
                    if supplier:
                        patterns['supplier_patterns'][supplier] += 1
            
            # Convert to probabilities and significant patterns
            analyzed_patterns = self._analyze_pattern_significance(patterns, len(confirmed_matches))
            
            # Cache the patterns
            self.pattern_cache = analyzed_patterns
            self.last_cache_update = datetime.now()
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['total_patterns_analyzed'] += len(confirmed_matches)
            self.performance_metrics['cache_updates'] += 1
            self._update_average_processing_time(processing_time)
            
            logger.info(f"Analyzed {len(confirmed_matches)} historical reconciliations, "
                       f"found {len(analyzed_patterns['description_patterns'])} description patterns, "
                       f"{len(analyzed_patterns['supplier_patterns'])} supplier patterns, "
                       f"processing time: {processing_time:.3f}s")
            
            return analyzed_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing historical patterns: {str(e)}")
            return {}
    
    def get_contextual_match_score(self, bank_transaction: Transaction, 
                                 company_entry: CompanyFinancial,
                                 base_score: float,
                                 user_id: int = None) -> ContextualMatchResult:
        """
        Calculate contextual match score based on historical patterns
        Enhanced with Phase 2 contextual analysis
        """
        start_time = datetime.now()
        
        try:
            # Get or update patterns
            if not self.pattern_cache or self._should_update_cache():
                self.analyze_historical_patterns(user_id)
            
            patterns = self.pattern_cache
            context_factors = {}
            
            # Description pattern bonus
            desc_bonus = self._calculate_description_pattern_bonus(
                bank_transaction.description, company_entry.description, patterns
            )
            context_factors['description_pattern_bonus'] = desc_bonus
            
            # Entity pattern bonus
            entity_bonus = self._calculate_entity_pattern_bonus(
                bank_transaction.description, company_entry.description, patterns
            )
            context_factors['entity_pattern_bonus'] = entity_bonus
            
            # Supplier consistency bonus
            supplier_bonus = self._calculate_supplier_consistency_bonus(
                bank_transaction, patterns
            )
            context_factors['supplier_consistency_bonus'] = supplier_bonus
            
            # Amount pattern likelihood
            amount_likelihood = self._calculate_amount_pattern_likelihood(
                bank_transaction.amount, company_entry.amount, patterns
            )
            context_factors['amount_pattern_likelihood'] = amount_likelihood
            
            # Date pattern likelihood
            date_likelihood = self._calculate_date_pattern_likelihood(
                bank_transaction.date, company_entry.date, patterns
            )
            context_factors['date_pattern_likelihood'] = date_likelihood
            
            # Time-based pattern bonus
            time_bonus = self._calculate_time_pattern_bonus(patterns)
            context_factors['time_pattern_bonus'] = time_bonus
            
            # User behavior pattern bonus (Phase 2 enhancement)
            user_behavior_bonus = 0.0
            if user_id and self.config.enable_user_specific_patterns:
                user_behavior_bonus = self._calculate_user_behavior_bonus(
                    user_id, bank_transaction, company_entry, patterns
                )
            context_factors['user_behavior_bonus'] = user_behavior_bonus
            
            # Apply mode-specific weights
            weights = self._get_mode_weights()
            total_bonus = sum([
                desc_bonus * weights['description'],
                entity_bonus * weights['entity'],
                supplier_bonus * weights['supplier'],
                amount_likelihood * weights['amount'],
                date_likelihood * weights['date'],
                time_bonus * weights['time'],
                user_behavior_bonus * weights['user_behavior']
            ])
            
            # Apply bonus to base score with mode-specific caps
            contextual_score = min(base_score + total_bonus, self._get_max_score_for_mode())
            
            # Generate explanation
            explanation = self._generate_contextual_explanation(context_factors, total_bonus)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ContextualMatchResult(
                score=contextual_score,
                base_score=base_score,
                context_factors=context_factors,
                total_bonus=total_bonus,
                explanation=explanation,
                patterns_used=len([f for f in context_factors.values() if f > 0]),
                user_id=user_id,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error calculating contextual match score: {str(e)}")
            return ContextualMatchResult(
                score=base_score,
                base_score=base_score,
                context_factors={},
                total_bonus=0.0,
                explanation=f"Error: {str(e)}",
                patterns_used=0,
                user_id=user_id,
                processing_time=0.0
            )
    
    def predict_missing_information(self, transaction: Transaction, 
                                 entry_type: str = 'company') -> PredictionResult:
        """
        Predict missing information based on historical patterns
        Enhanced with Phase 2 prediction capabilities
        """
        start_time = datetime.now()
        
        try:
            if not self.pattern_cache:
                self.analyze_historical_patterns()
            
            patterns = self.pattern_cache
            
            # Extract key information from available transaction
            if entry_type == 'company':
                # Predict company entry details based on bank transaction
                if transaction.description:
                    # Use enhanced normalization from Phase 2
                    normalized_result = self.normalizer.normalize_text(transaction.description)
                    entities = normalized_result.entities
                    
                    # Predict likely suppliers
                    predicted_suppliers = entities.get('company', [])[:3]
                    
                    # Predict payment methods
                    predicted_payment_methods = entities.get('payment_method', [])
                    
                    # Predict categories based on enhanced analysis
                    predicted_categories = self._predict_categories_from_context(transaction, patterns)
                    
                    # Predict amount variance pattern
                    amount_patterns = patterns.get('amount_patterns', {})
                    amount_variance_pattern = max(amount_patterns.items(), 
                                                key=lambda x: x[1])[0] if amount_patterns else 'unknown'
                    
                    # Calculate confidence score
                    confidence_score = self._calculate_prediction_confidence(
                        predicted_suppliers, predicted_payment_methods, predicted_categories, patterns
                    )
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    result = PredictionResult(
                        predicted_suppliers=predicted_suppliers,
                        predicted_payment_methods=predicted_payment_methods,
                        predicted_categories=predicted_categories,
                        amount_variance_pattern=amount_variance_pattern,
                        confidence_score=confidence_score,
                        based_on_patterns=len([p for p in patterns.values() if p])
                    )
                    
                    # Update performance metrics
                    self.performance_metrics['successful_predictions'] += 1
                    self._update_average_processing_time(processing_time)
                    
                    return result
            
            return PredictionResult(
                predicted_suppliers=[],
                predicted_payment_methods=[],
                predicted_categories=[],
                amount_variance_pattern='unknown',
                confidence_score=0.0,
                based_on_patterns=0
            )
            
        except Exception as e:
            logger.error(f"Error predicting missing information: {str(e)}")
            return PredictionResult(
                predicted_suppliers=[],
                predicted_payment_methods=[],
                predicted_categories=[],
                amount_variance_pattern='unknown',
                confidence_score=0.0,
                based_on_patterns=0
            )
    
    def get_matching_suggestions(self, bank_transaction: Transaction, 
                               available_entries: List[CompanyFinancial],
                               user_id: int = None) -> List[MatchingSuggestion]:
        """
        Get context-aware matching suggestions
        Enhanced with Phase 2 suggestion capabilities
        """
        start_time = datetime.now()
        
        try:
            suggestions = []
            
            # Extract key information from bank transaction using enhanced normalization
            normalized_result = self.normalizer.normalize_text(bank_transaction.description)
            bank_desc = normalized_result.normalized_text
            bank_entities = normalized_result.entities
            
            # Score each available entry
            for entry in available_entries:
                # Get base score using enhanced similarity calculation
                base_score = self.normalizer.calculate_similarity(
                    bank_transaction.description, entry.description
                )
                
                # Get contextual score
                contextual_result = self.get_contextual_match_score(
                    bank_transaction, entry, base_score, user_id
                )
                
                # Apply mode-specific threshold
                threshold = self._get_suggestion_threshold_for_mode()
                if contextual_result.score > threshold:
                    # Predict missing information for this suggestion
                    predicted_info = self.predict_missing_information(bank_transaction)
                    
                    suggestion = MatchingSuggestion(
                        company_entry=entry,
                        confidence_score=contextual_result.score,
                        context_factors=contextual_result.context_factors,
                        explanation=contextual_result.explanation,
                        patterns_used=contextual_result.patterns_used,
                        predicted_missing_info={
                            'suppliers': predicted_info.predicted_suppliers,
                            'payment_methods': predicted_info.predicted_payment_methods,
                            'categories': predicted_info.predicted_categories
                        },
                        user_behavior_score=contextual_result.context_factors.get('user_behavior_bonus', 0.0)
                    )
                    suggestions.append(suggestion)
            
            # Sort by confidence score
            suggestions.sort(key=lambda x: x.confidence_score, reverse=True)
            
            # Apply mode-specific limit
            max_suggestions = self._get_max_suggestions_for_mode()
            final_suggestions = suggestions[:max_suggestions]
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['suggestions_generated'] += len(final_suggestions)
            self._update_average_processing_time(processing_time)
            
            logger.debug(f"Generated {len(final_suggestions)} suggestions in {processing_time:.3f}s")
            
            return final_suggestions
            
        except Exception as e:
            logger.error(f"Error getting matching suggestions: {str(e)}")
            return []
    
    def _should_update_cache(self) -> bool:
        """Check if pattern cache should be updated"""
        if not self.last_cache_update:
            return True
        
        # Update cache every 24 hours or after 100 new reconciliations
        time_since_update = datetime.now() - self.last_cache_update
        if time_since_update > timedelta(hours=24):
            return True
        
        # Check for new reconciliations
        recent_count = ReconciliationRecord.query.filter(
            ReconciliationRecord.created_at > self.last_cache_update,
            ReconciliationRecord.status == 'confirmed'
        ).count()
        
        return recent_count > 100
    
    def _analyze_pattern_significance(self, patterns: Dict[str, Any], total_records: int) -> Dict[str, Any]:
        """Analyze pattern significance and convert to probabilities"""
        significant_patterns = {}
        
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict):
                significant_patterns[pattern_type] = {}
                
                for pattern, count in pattern_data.items():
                    # Calculate significance based on frequency
                    frequency = count / total_records if total_records > 0 else 0
                    
                    # Only keep significant patterns (appearing more than once and > 1%)
                    if count > 1 and frequency > 0.01:
                        significant_patterns[pattern_type][pattern] = {
                            'count': count,
                            'frequency': frequency,
                            'significance': 'high' if frequency > 0.1 else 'medium'
                        }
        
        return significant_patterns
    
    def _calculate_description_pattern_bonus(self, bank_desc: str, company_desc: str, 
                                         patterns: Dict[str, Any]) -> float:
        """Calculate bonus based on historical description patterns"""
        if not patterns.get('description_patterns'):
            return 0.0
        
        bank_norm = self.normalizer.normalize_text(bank_desc).normalized_text
        company_norm = self.normalizer.normalize_text(company_desc).normalized_text
        
        pattern_key = f"{bank_norm}||{company_norm}"
        pattern_info = patterns['description_patterns'].get(pattern_key)
        
        if pattern_info:
            # Bonus based on frequency of this pattern
            return min(pattern_info['frequency'] * 10, 0.3)  # Max 0.3 bonus
        
        return 0.0
    
    def _calculate_entity_pattern_bonus(self, bank_desc: str, company_desc: str,
                                      patterns: Dict[str, Any]) -> float:
        """Calculate bonus based on historical entity patterns"""
        if not patterns.get('entity_patterns'):
            return 0.0
        
        bank_entities = self.normalizer.normalize_text(bank_desc).entities
        company_entities = self.normalizer.normalize_text(company_desc).entities
        
        total_bonus = 0.0
        
        for entity_type in ['companies', 'payment_methods']:
            for bank_entity in bank_entities.get(entity_type, []):
                for company_entity in company_entities.get(entity_type, []):
                    pattern_key = f"{entity_type}:{bank_entity}||{company_entity}"
                    pattern_info = patterns['entity_patterns'].get(pattern_key)
                    
                    if pattern_info:
                        total_bonus += min(pattern_info['frequency'] * 5, 0.15)
        
        return min(total_bonus, 0.25)
    
    def _calculate_supplier_consistency_bonus(self, bank_transaction: Transaction,
                                            patterns: Dict[str, Any]) -> float:
        """Calculate bonus based on supplier consistency"""
        if not patterns.get('supplier_patterns'):
            return 0.0
        
        supplier = self._extract_supplier_name(bank_transaction.description)
        if not supplier:
            return 0.0
        
        supplier_frequency = patterns['supplier_patterns'].get(supplier, {}).get('frequency', 0)
        return min(supplier_frequency * 2, 0.2)
    
    def _calculate_amount_pattern_likelihood(self, bank_amount: float, company_amount: float,
                                           patterns: Dict[str, Any]) -> float:
        """Calculate likelihood based on historical amount patterns"""
        if not patterns.get('amount_patterns'):
            return 0.0
        
        amount_diff = abs(abs(bank_amount) - abs(company_amount))
        
        if amount_diff < 0.01:
            pattern_type = 'exact'
        elif amount_diff / max(abs(bank_amount), abs(company_amount)) < 0.05:
            pattern_type = 'close'
        else:
            pattern_type = 'variable'
        
        pattern_info = patterns['amount_patterns'].get(pattern_type, {})
        frequency = pattern_info.get('frequency', 0)
        
        return min(frequency, 0.2)
    
    def _calculate_date_pattern_likelihood(self, bank_date, company_date,
                                         patterns: Dict[str, Any]) -> float:
        """Calculate likelihood based on historical date patterns"""
        if not patterns.get('date_patterns') or not (bank_date and company_date):
            return 0.0
        
        date_diff = abs((bank_date - company_date).days)
        
        if date_diff == 0:
            pattern_type = 'same_day'
        elif date_diff <= 3:
            pattern_type = 'few_days'
        elif date_diff <= 7:
            pattern_type = 'week'
        else:
            pattern_type = 'extended'
        
        pattern_info = patterns['date_patterns'].get(pattern_type, {})
        frequency = pattern_info.get('frequency', 0)
        
        return min(frequency, 0.2)
    
    def _calculate_time_pattern_bonus(self, patterns: Dict[str, Any]) -> float:
        """Calculate bonus based on time patterns"""
        if not patterns.get('time_patterns'):
            return 0.0
        
        current_hour = datetime.now().hour
        hour_pattern = f"hour_{current_hour}"
        
        pattern_info = patterns['time_patterns'].get(hour_pattern, {})
        frequency = pattern_info.get('frequency', 0)
        
        return min(frequency, 0.1)
    
    def _extract_supplier_name(self, description: str) -> Optional[str]:
        """Extract supplier name from transaction description"""
        if not description:
            return None
        
        entities = self.normalizer.normalize_text(description).entities
        companies = entities.get('company', [])
        
        if companies:
            # Return the most frequent or first company
            return companies[0]
        
        # Try to extract from description patterns
        words = description.split()
        for i, word in enumerate(words):
            if word.upper() in ['LTDA', 'S.A', 'ME', 'EPP']:
                # Take up to 3 words before the company type
                start = max(0, i - 3)
                supplier = ' '.join(words[start:i + 1])
                return supplier
        
        return None
    
    def _generate_contextual_explanation(self, context_factors: Dict[str, float], 
                                      total_bonus: float) -> str:
        """Generate human-readable explanation for contextual score"""
        explanations = []
        
        for factor, value in context_factors.items():
            if value > 0.01:  # Only include significant factors
                factor_name = factor.replace('_', ' ').title()
                explanations.append(f"{factor_name}: +{value:.3f}")
        
        if explanations:
            base_text = "Contextual adjustments: " + "; ".join(explanations)
            if total_bonus > 0.1:
                base_text += f" (Total bonus: +{total_bonus:.3f})"
            return base_text
        
        return "No significant contextual patterns found"
    
    # Phase 2 Enhancement Methods
    
    def _get_mode_weights(self) -> Dict[str, float]:
        """Get pattern weights based on matching mode"""
        if self.config.mode == MatchingMode.CONSERVATIVE:
            return {
                'description': 0.35,
                'entity': 0.25,
                'supplier': 0.20,
                'amount': 0.15,
                'date': 0.15,
                'time': 0.05,
                'user_behavior': 0.10
            }
        elif self.config.mode == MatchingMode.AGGRESSIVE:
            return {
                'description': 0.25,
                'entity': 0.20,
                'supplier': 0.15,
                'amount': 0.10,
                'date': 0.10,
                'time': 0.05,
                'user_behavior': 0.05
            }
        elif self.config.mode == MatchingMode.LEARNING:
            return {
                'description': 0.30,
                'entity': 0.25,
                'supplier': 0.20,
                'amount': 0.10,
                'date': 0.10,
                'time': 0.05,
                'user_behavior': 0.20
            }
        else:  # BALANCED
            return {
                'description': 0.30,
                'entity': 0.25,
                'supplier': 0.20,
                'amount': 0.10,
                'date': 0.10,
                'time': 0.05,
                'user_behavior': 0.15
            }
    
    def _get_max_score_for_mode(self) -> float:
        """Get maximum allowed score based on matching mode"""
        mode_limits = {
            MatchingMode.CONSERVATIVE: 0.85,
            MatchingMode.BALANCED: 0.95,
            MatchingMode.AGGRESSIVE: 1.0,
            MatchingMode.LEARNING: 1.0
        }
        return mode_limits.get(self.config.mode, 0.95)
    
    def _get_suggestion_threshold_for_mode(self) -> float:
        """Get suggestion threshold based on matching mode"""
        mode_thresholds = {
            MatchingMode.CONSERVATIVE: 0.7,
            MatchingMode.BALANCED: 0.6,
            MatchingMode.AGGRESSIVE: 0.5,
            MatchingMode.LEARNING: 0.4
        }
        return mode_thresholds.get(self.config.mode, 0.6)
    
    def _get_max_suggestions_for_mode(self) -> int:
        """Get maximum number of suggestions based on matching mode"""
        mode_limits = {
            MatchingMode.CONSERVATIVE: 5,
            MatchingMode.BALANCED: 10,
            MatchingMode.AGGRESSIVE: 15,
            MatchingMode.LEARNING: 20
        }
        return mode_limits.get(self.config.mode, 10)
    
    def _calculate_user_behavior_bonus(self, user_id: int, bank_transaction: Transaction,
                                      company_entry: CompanyFinancial, patterns: Dict[str, Any]) -> float:
        """Calculate user behavior pattern bonus"""
        if not self.config.enable_user_specific_patterns:
            return 0.0
        
        # Get user-specific patterns
        user_patterns = self.user_patterns.get(user_id, {})
        if not user_patterns:
            return 0.0
        
        # Analyze user's historical behavior
        user_bonus = 0.0
        
        # Check if user has similar matching patterns
        if user_patterns.get('matching_style'):
            # Adjust bonus based on user's typical matching behavior
            style_bonus = user_patterns['matching_style'].get('bonus_factor', 0.0)
            user_bonus += style_bonus * 0.1
        
        # Check user's time-based patterns
        current_hour = datetime.now().hour
        if user_patterns.get('time_preferences'):
            time_pref = user_patterns['time_preferences'].get(f'hour_{current_hour}', 0.0)
            user_bonus += time_pref * 0.05
        
        return min(user_bonus, 0.15)
    
    def _predict_categories_from_context(self, transaction: Transaction, 
                                        patterns: Dict[str, Any]) -> List[str]:
        """Predict categories based on contextual analysis"""
        categories = []
        
        if transaction.description:
            # Use enhanced normalization to extract context
            normalized_result = self.normalizer.normalize_text(transaction.description)
            
            # Analyze entities for category clues
            entities = normalized_result.entities
            
            # Category prediction based on entities
            if entities.get('tax'):
                categories.extend(['taxes', 'government_fees'])
            if entities.get('payment_method'):
                if 'boleto' in entities['payment_method']:
                    categories.append('bill_payment')
                if 'pix' in entities['payment_method']:
                    categories.append('instant_payment')
            if entities.get('company'):
                categories.append('supplier_payment')
            
            # Amount-based category hints
            if transaction.amount:
                if abs(transaction.amount) < 100:
                    categories.append('small_expense')
                elif abs(transaction.amount) > 10000:
                    categories.append('large_payment')
        
        return list(set(categories))[:5]  # Return unique categories
    
    def _calculate_prediction_confidence(self, suppliers: List[str], payment_methods: List[str],
                                      categories: List[str], patterns: Dict[str, Any]) -> float:
        """Calculate confidence score for predictions"""
        confidence = 0.0
        
        # Base confidence from data availability
        if suppliers:
            confidence += 0.3
        if payment_methods:
            confidence += 0.2
        if categories:
            confidence += 0.2
        
        # Pattern-based confidence boost
        pattern_count = len([p for p in patterns.values() if p])
        if pattern_count > 10:
            confidence += 0.2
        elif pattern_count > 5:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time metric"""
        if self.performance_metrics['average_processing_time'] == 0:
            self.performance_metrics['average_processing_time'] = processing_time
        else:
            # Simple moving average
            alpha = 0.1
            current_avg = self.performance_metrics['average_processing_time']
            self.performance_metrics['average_processing_time'] = (
                alpha * processing_time + (1 - alpha) * current_avg
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the context-aware matcher"""
        return {
            **self.performance_metrics,
            'cache_status': {
                'last_update': self.last_cache_update.isoformat() if self.last_cache_update else None,
                'total_cached_patterns': sum(len(v) if isinstance(v, dict) else 0 for v in self.pattern_cache.values()),
                'cache_age_hours': ((datetime.now() - self.last_cache_update).total_seconds() / 3600) if self.last_cache_update else None
            },
            'config': {
                'mode': self.config.mode.value,
                'learning_enabled': self.learning_enabled,
                'auto_cache_update': self.config.auto_cache_update
            }
        }
    
    def update_config(self, new_config: MatchingConfig):
        """Update matcher configuration"""
        self.config = new_config
        self.learning_enabled = new_config.enable_learning
        logger.info(f"Updated context-aware matcher configuration: {new_config.mode.value}")
    
    def clear_cache(self):
        """Clear pattern cache and force relearning"""
        self.pattern_cache = {}
        self.user_patterns = defaultdict(dict)
        self.last_cache_update = None
        logger.info("Cleared context-aware matcher cache")
    
    def export_patterns(self, file_path: str):
        """Export learned patterns to file"""
        try:
            export_data = {
                'patterns': self.pattern_cache,
                'user_patterns': dict(self.user_patterns),
                'metrics': self.performance_metrics,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported patterns to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting patterns: {str(e)}")
            return False
    
    def import_patterns(self, file_path: str):
        """Import learned patterns from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            self.pattern_cache = import_data.get('patterns', {})
            self.user_patterns = defaultdict(dict, import_data.get('user_patterns', {}))
            self.performance_metrics.update(import_data.get('metrics', {}))
            
            logger.info(f"Imported patterns from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing patterns: {str(e)}")
            return False