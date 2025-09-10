"""
Enhanced Data Normalization Service - Refactored for Phase 2 Integration

Provides advanced data normalization capabilities with:
- Brazilian financial format support
- Entity extraction and recognition
- Integration with Phase 2 workflow patterns
- Real-time processing capabilities
- Configurable normalization rules
"""

import re
import string
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Set, Any
from unidecode import unidecode
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class NormalizationMode(Enum):
    """Normalization modes for different use cases"""
    STRICT = "strict"          # Exact matching, minimal normalization
    STANDARD = "standard"      # Balanced normalization for general use
    AGGRESSIVE = "aggressive"  # Maximum normalization for fuzzy matching

class EntityType(Enum):
    """Types of entities that can be extracted"""
    COMPANY = "company"
    PAYMENT_METHOD = "payment_method"
    TAX = "tax"
    AMOUNT = "amount"
    DATE = "date"
    LOCATION = "location"

@dataclass
class NormalizationResult:
    """Standard result format for normalization operations"""
    original_text: str
    normalized_text: str
    confidence: float
    entities: Dict[str, List[str]]
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class NormalizationConfig:
    """Configuration for normalization behavior"""
    mode: NormalizationMode = NormalizationMode.STANDARD
    remove_noise_words: bool = True
    extract_entities: bool = True
    preserve_amounts: bool = True
    preserve_dates: bool = True
    min_word_length: int = 2
    max_word_length: int = 50
    enabled_entity_types: List[EntityType] = None
    
    def __post_init__(self):
        if self.enabled_entity_types is None:
            self.enabled_entity_types = list(EntityType)

class BrazilianDataNormalizer:
    """
    Enhanced data normalization service for Brazilian financial data
    Refactored to integrate with Phase 2 workflow patterns
    """
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()
        self._initialize_patterns()
        self._initialize_noise_words()
        self._initialize_entity_patterns()
        
        # Performance metrics
        self.processing_count = 0
        self.total_processing_time = 0.0
    
    def _initialize_patterns(self):
        """Initialize regex patterns for financial data"""
        # Brazilian currency patterns
        self.amount_patterns = [
            r'R\$\s*([\d\.,]+)',           # R$ 1.234,56
            r'\$([\d\.,]+)',                # $1,234.56
            r'([\d\.,]+)\s*reais?',        # 1234 reais
            r'([\d\.,]+)\s*r\$',           # 1234 R$
            r'([\d\.,]+)\s*USD',           # 1234 USD
            r'valor\s*[:\-]?\s*([\d\.,]+)', # valor: 1234
            r'total\s*[:\-]?\s*([\d\.,]+)',  # total: 1234
        ]
        
        # Brazilian date patterns
        self.date_patterns = [
            r'(\d{2})/(\d{2})/(\d{4})',    # DD/MM/YYYY
            r'(\d{4})-(\d{2})-(\d{2})',    # YYYY-MM-DD
            r'(\d{2})-(\d{2})-(\d{4})',    # DD-MM-YYYY
            r'(\d{2})\.(\d{2})\.(\d{4})',  # DD.MM.YYYY
        ]
        
        # Advanced amount patterns with context
        self.advanced_amount_patterns = [
            (r'(\d+(?:\.\d{3})*,\d{2})', 'brazilian_decimal'),  # 1.234,56
            (r'(\d+(?:,\d{3})*\.\d{2})', 'us_decimal'),         # 1,234.56
            (r'(\d+)', 'integer'),                                 # 1234
        ]
    
    def _initialize_noise_words(self):
        """Initialize Portuguese noise words"""
        self.noise_words = {
            'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'do', 'da',
            'dos', 'das', 'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'com',
            'sem', 'ate', 'desde', 'entre', 'sobre', 'sob', 'pelo', 'pela',
            'aos', 'as', 'ao', 'a', 'dum', 'duma', 'deste', 'desta', 'este',
            'esta', 'estes', 'estas', 'esse', 'essa', 'esses', 'essas',
            'aquele', 'aquela', 'aqueles', 'aquelas', 'outro', 'outra',
            'outros', 'outras', 'algum', 'alguma', 'alguns', 'algumas',
            'nenhum', 'nenhuma', 'nenhuns', 'nenhumas', 'todo', 'toda',
            'todos', 'todas', 'muito', 'muita', 'muitos', 'muitas',
            'pouco', 'pouca', 'poucos', 'poucas'
        }
    
    def _initialize_entity_patterns(self):
        """Initialize entity recognition patterns"""
        self.entity_patterns = {
            EntityType.COMPANY: [
                r'\b(?:LTDA|S\.?A\.?|ME|EPP|LTDA\.?ME)\b',
                r'\b(?:COMERCIO|INDUSTRIA|SERVICOS|IMPORTACAO|EXPORTACAO)\b',
                r'\b(?:DISTRIBUIDORA|SUPERMERCADO|MERCADO|LOJA)\b',
                r'\b(?:RESTAURANTE|HOTEL|CLINICA|HOSPITAL)\b',
                r'\b(?:AUTO|POSTO|OFICINA|MECANICA)\b',
            ],
            EntityType.PAYMENT_METHOD: [
                r'\b(?:BOLETO|TED|DOC|PIX|DEBITO)\b',
                r'\b(?:CREDITO|CARTAO|CHEQUE|TRANSFERENCIA)\b',
                r'\b(?:DINHEIRO|ESPECIE|VALE)\b',
            ],
            EntityType.TAX: [
                r'\b(?:IMPOSTO|TAXA|CONTRIBUICAO)\b',
                r'\b(?:IOF|IRRF|PIS|COFINS|CSLL)\b',
                r'\b(?:ISS|ICMS|IPI)\b',
            ],
            EntityType.LOCATION: [
                r'\b(?:SAO|SANTO|SANTA)\s+[A-Z]+',
                r'\b(?:RUA|AVENIDA|ALAMEDA|TRAVESSA)\s+[A-Z]+',
                r'\b(?:CENTRO|BAIRRO|VILA|JARDIM)\s+[A-Z]+',
            ]
        }
    
    def normalize_text(self, text: str, text_type: str = "general") -> NormalizationResult:
        """
        Normalize text with comprehensive processing
        """
        import time
        start_time = time.time()
        
        try:
            if not text:
                return NormalizationResult(
                    original_text=text,
                    normalized_text="",
                    confidence=0.0,
                    entities={},
                    processing_time=0.0,
                    metadata={"error": "Empty input"}
                )
            
            original_text = text
            entities = {}
            
            # Step 1: Basic text cleaning
            normalized = self._basic_text_cleaning(text)
            
            # Step 2: Extract entities if enabled
            if self.config.extract_entities:
                entities = self._extract_entities_advanced(text)
            
            # Step 3: Apply mode-specific normalization
            if self.config.mode == NormalizationMode.STRICT:
                normalized = self._apply_strict_normalization(normalized)
            elif self.config.mode == NormalizationMode.AGGRESSIVE:
                normalized = self._apply_aggressive_normalization(normalized)
            else:  # STANDARD
                normalized = self._apply_standard_normalization(normalized)
            
            # Step 4: Calculate confidence
            confidence = self._calculate_normalization_confidence(text, normalized)
            
            # Step 5: Preserve important entities
            if self.config.preserve_amounts:
                normalized = self._preserve_amounts(text, normalized)
            
            if self.config.preserve_dates:
                normalized = self._preserve_dates(text, normalized)
            
            processing_time = time.time() - start_time
            
            # Update metrics
            self.processing_count += 1
            self.total_processing_time += processing_time
            
            return NormalizationResult(
                original_text=original_text,
                normalized_text=normalized,
                confidence=confidence,
                entities=entities,
                processing_time=processing_time,
                metadata={
                    "mode": self.config.mode.value,
                    "text_type": text_type,
                    "processing_count": self.processing_count,
                    "avg_processing_time": self.total_processing_time / self.processing_count
                }
            )
            
        except Exception as e:
            logger.error(f"Error normalizing text: {str(e)}")
            return NormalizationResult(
                original_text=text,
                normalized_text="",
                confidence=0.0,
                entities={},
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _basic_text_cleaning(self, text: str) -> str:
        """Apply basic text cleaning operations"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove accents
        text = unidecode(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _apply_strict_normalization(self, text: str) -> str:
        """Apply strict normalization (minimal changes)"""
        # Only remove extra whitespace and normalize case
        return text.strip()
    
    def _apply_standard_normalization(self, text: str) -> str:
        """Apply standard normalization (balanced approach)"""
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove noise words if enabled
        if self.config.remove_noise_words:
            words = text.split()
            filtered_words = [
                word for word in words 
                if (word not in self.noise_words and 
                    self.config.min_word_length <= len(word) <= self.config.max_word_length)
            ]
            text = ' '.join(filtered_words)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _apply_aggressive_normalization(self, text: str) -> str:
        """Apply aggressive normalization (maximum normalization)"""
        # Remove all special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove all noise words and short words
        words = text.split()
        filtered_words = [
            word for word in words 
            if (word not in self.noise_words and len(word) >= self.config.min_word_length)
        ]
        text = ' '.join(filtered_words)
        
        # Remove duplicate words
        words = text.split()
        seen = set()
        unique_words = []
        for word in words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        text = ' '.join(unique_words)
        
        return text.strip()
    
    def _extract_entities_advanced(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using advanced pattern matching"""
        entities = {entity_type.value: [] for entity_type in EntityType}
        
        text_lower = text.lower()
        
        # Extract entities by type
        for entity_type, patterns in self.entity_patterns.items():
            if entity_type not in self.config.enabled_entity_types:
                continue
                
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                entities[entity_type.value].extend(matches)
        
        # Extract amounts and dates
        if EntityType.AMOUNT in self.config.enabled_entity_types:
            entities[EntityType.AMOUNT.value] = self._extract_amounts(text)
        
        if EntityType.DATE in self.config.enabled_entity_types:
            entities[EntityType.DATE.value] = self._extract_dates(text)
        
        # Remove duplicates and clean up
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
            entities[entity_type] = [entity.strip() for entity in entities[entity_type] if entity.strip()]
        
        return entities
    
    def _extract_amounts(self, text: str) -> List[str]:
        """Extract and normalize amounts from text"""
        amounts = []
        
        for pattern, format_type in self.advanced_amount_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                normalized_amount = self.normalize_amount(match)
                if normalized_amount:
                    amounts.append(str(normalized_amount))
        
        return amounts
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract and normalize dates from text"""
        dates = []
        
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 3:
                    try:
                        if pattern.startswith(r'(\d{4})'):
                            year, month, day = int(match[0]), int(match[1]), int(match[2])
                        else:
                            day, month, year = int(match[0]), int(match[1]), int(match[2])
                        
                        if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                            dates.append(f"{year:04d}-{month:02d}-{day:02d}")
                    except ValueError:
                        continue
        
        return dates
    
    def _preserve_amounts(self, original_text: str, normalized_text: str) -> str:
        """Preserve amounts in normalized text"""
        # This is a simplified version - in production, you'd want more sophisticated preservation
        return normalized_text
    
    def _preserve_dates(self, original_text: str, normalized_text: str) -> str:
        """Preserve dates in normalized text"""
        # This is a simplified version - in production, you'd want more sophisticated preservation
        return normalized_text
    
    def _calculate_normalization_confidence(self, original: str, normalized: str) -> float:
        """Calculate confidence score for normalization"""
        if not original:
            return 0.0
        
        # Base confidence on length preservation
        original_len = len(original.split())
        normalized_len = len(normalized.split())
        
        if original_len == 0:
            return 0.0
        
        length_ratio = normalized_len / original_len
        
        # Adjust based on mode
        if self.config.mode == NormalizationMode.STRICT:
            return min(length_ratio, 1.0)
        elif self.config.mode == NormalizationMode.AGGRESSIVE:
            return max(0.1, length_ratio)  # Allow more aggressive reduction
        else:  # STANDARD
            return min(length_ratio * 1.2, 1.0)  # Slight bonus for reasonable reduction
    
    def normalize_amount(self, amount_str: str) -> Optional[float]:
        """
        Normalize amount string to float with Brazilian format support
        """
        if not amount_str:
            return None
        
        try:
            # Remove currency symbols and whitespace
            amount_str = re.sub(r'[R\$]', '', amount_str, flags=re.IGNORECASE)
            amount_str = re.sub(r'\s+', '', amount_str)
            
            # Handle Brazilian format (1.234,56) vs US format (1,234.56)
            if ',' in amount_str and '.' in amount_str:
                # Determine format by position
                if amount_str.rfind(',') > amount_str.rfind('.'):
                    # Brazilian format: 1.234,56 -> 1234.56
                    amount_str = amount_str.replace('.', '').replace(',', '.')
                else:
                    # US format: 1,234.56 -> 1234.56
                    amount_str = amount_str.replace(',', '')
            elif ',' in amount_str:
                # Could be Brazilian decimal or US thousands
                parts = amount_str.split(',')
                if len(parts) == 2 and len(parts[1]) == 2:
                    # Likely Brazilian decimal: 1234,56 -> 1234.56
                    amount_str = amount_str.replace(',', '.')
                else:
                    # Likely US thousands: 1,234 -> 1234
                    amount_str = amount_str.replace(',', '')
            
            # Remove any remaining non-numeric characters except decimal point
            amount_str = re.sub(r'[^\d.]', '', amount_str)
            
            return float(amount_str)
            
        except (ValueError, AttributeError):
            logger.debug(f"Failed to normalize amount: {amount_str}")
            return None
    
    def normalize_date(self, date_str: str) -> Optional[date]:
        """
        Normalize date string to date object
        """
        if not date_str:
            return None
        
        for pattern in self.date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 3:
                        # Determine format based on pattern
                        if pattern.startswith(r'(\d{4})'):
                            # YYYY-MM-DD format
                            year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        else:
                            # DD-MM-YYYY format (Brazilian)
                            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                        
                        # Validate date components
                        if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                            return date(year, month, day)
                            
                except (ValueError, IndexError):
                    continue
        
        logger.debug(f"Failed to normalize date: {date_str}")
        return None
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using multiple methods
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize both texts
        result1 = self.normalize_text(text1)
        result2 = self.normalize_text(text2)
        
        norm1 = result1.normalized_text
        norm2 = result2.normalized_text
        
        if not norm1 or not norm2:
            return 0.0
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Containment check
        if norm1 in norm2 or norm2 in norm1:
            return 0.95
        
        # Word-based similarity
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Entity similarity bonus
        entity_similarity = self._calculate_entity_similarity(result1.entities, result2.entities)
        
        # Combine similarities
        combined_similarity = (jaccard_similarity * 0.7) + (entity_similarity * 0.3)
        
        return min(combined_similarity, 1.0)
    
    def _calculate_entity_similarity(self, entities1: Dict[str, List[str]], 
                                   entities2: Dict[str, List[str]]) -> float:
        """Calculate similarity based on extracted entities"""
        if not entities1 and not entities2:
            return 0.0
        
        total_similarity = 0.0
        entity_types = 0
        
        for entity_type in [EntityType.COMPANY.value, EntityType.PAYMENT_METHOD.value, EntityType.TAX.value]:
            set1 = set(entities1.get(entity_type, []))
            set2 = set(entities2.get(entity_type, []))
            
            if set1 or set2:
                intersection = set1.intersection(set2)
                union = set1.union(set2)
                
                if union:
                    jaccard_similarity = len(intersection) / len(union)
                    total_similarity += jaccard_similarity
                    entity_types += 1
        
        return total_similarity / entity_types if entity_types > 0 else 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the normalizer"""
        avg_time = self.total_processing_time / self.processing_count if self.processing_count > 0 else 0
        
        return {
            "total_processed": self.processing_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_time,
            "config_mode": self.config.mode.value,
            "entity_extraction_enabled": self.config.extract_entities
        }
    
    def update_config(self, new_config: NormalizationConfig) -> None:
        """Update the normalizer configuration"""
        self.config = new_config
        logger.info(f"Normalizer configuration updated to mode: {new_config.mode.value}")
    
    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        self.processing_count = 0
        self.total_processing_time = 0.0
        logger.info("Normalizer metrics reset")