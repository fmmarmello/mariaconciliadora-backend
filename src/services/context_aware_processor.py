import re
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import logging
from itertools import combinations

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ContextAwareProcessor:
    """
    Context-aware text processor with intelligent disambiguation and pattern recognition
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the context-aware processor

        Args:
            config: Configuration dictionary for context processing
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize context knowledge bases
        self._initialize_context_bases()

        # Initialize pattern libraries
        self._initialize_patterns()

        # Processing cache and context memory
        self._processing_cache = {}
        self._context_memory = defaultdict(list)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'context_window_size': 5,
            'min_confidence_threshold': 0.6,
            'max_ambiguity_resolution_attempts': 3,
            'enable_multiword_recognition': True,
            'enable_entity_disambiguation': True,
            'enable_pattern_learning': True,
            'cache_enabled': True,
            'quality_assessment_enabled': True
        }

    def _initialize_context_bases(self):
        """Initialize context knowledge bases"""
        # Multi-word expressions in financial context
        self.multiword_expressions = {
            'banco_central': ['banco', 'central'],
            'conta_corrente': ['conta', 'corrente'],
            'conta_poupanca': ['conta', 'poupanca'],
            'cartao_credito': ['cartao', 'credito'],
            'cartao_debito': ['cartao', 'debito'],
            'transferencia_bancaria': ['transferencia', 'bancaria'],
            'pagamento_online': ['pagamento', 'online'],
            'saque_automatico': ['saque', 'automatico'],
            'deposito_identificado': ['deposito', 'identificado'],
            'cheque_compensado': ['cheque', 'compensado'],
            'limite_credito': ['limite', 'credito'],
            'saldo_disponivel': ['saldo', 'disponivel'],
            'juros_mensal': ['juros', 'mensal'],
            'taxa_administracao': ['taxa', 'administracao'],
            'fundo_investimento': ['fundo', 'investimento']
        }

        # Ambiguous terms and their context-based resolutions
        self.ambiguous_terms = {
            'conta': {
                'bancaria': 'conta_bancaria',
                'corrente': 'conta_corrente',
                'poupanca': 'conta_poupanca',
                'salario': 'conta_salario',
                'investimento': 'conta_investimento'
            },
            'cartao': {
                'credito': 'cartao_credito',
                'debito': 'cartao_debito',
                'presente': 'cartao_presente'
            },
            'transferencia': {
                'bancaria': 'transferencia_bancaria',
                'ted': 'transferencia_ted',
                'doc': 'transferencia_doc',
                'pix': 'transferencia_pix'
            },
            'pagamento': {
                'boleto': 'pagamento_boleto',
                'fatura': 'pagamento_fatura',
                'online': 'pagamento_online',
                'loja': 'pagamento_loja'
            }
        }

        # Financial entity patterns
        self.entity_patterns = {
            'monetary_amount': r'r\$\s*\d+[\.,]\d{2}',
            'account_number': r'\b\d{5,12}\b',
            'agency_number': r'ag\.?\s*\d{4}',
            'document_number': r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b',
            'transaction_code': r'aut\s*\d{6,12}',
            'date_pattern': r'\b\d{1,2}/\d{1,2}/\d{4}\b'
        }

        # Context indicators for different transaction types
        self.context_indicators = {
            'banking': ['banco', 'agencia', 'conta', 'saldo', 'transferencia'],
            'payment': ['pagamento', 'boleto', 'fatura', 'vencimento', 'valor'],
            'investment': ['investimento', 'fundo', 'acao', 'renda', 'fixa'],
            'credit': ['credito', 'limite', 'parcela', 'juros', 'divida'],
            'cash': ['saque', 'deposito', 'dinheiro', 'caixa', 'atm']
        }

    def _initialize_patterns(self):
        """Initialize pattern recognition libraries"""
        self.patterns = {
            # Noise patterns to remove
            'noise_patterns': [
                r'\b(www|http|https|com|br)\b',
                r'\b\d{10,}\b',  # Very long numbers
                r'\b[a-zA-Z]{20,}\b',  # Very long words
                r'[^\w\s]{3,}',  # Multiple special characters
            ],

            # Financial document patterns
            'document_patterns': [
                r'comprovante\s+de\s+pagamento',
                r'extrato\s+bancario',
                r'nota\s+fiscal',
                r'recibo\s+de\s+pagamento',
                r'boleto\s+bancario'
            ],

            # Transaction status patterns
            'status_patterns': [
                r'status\s*:\s*(\w+)',
                r'situacao\s*:\s*(\w+)',
                r'estado\s*:\s*(\w+)'
            ]
        }

    def process_with_context(self, text: str, context_history: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process text with context awareness

        Args:
            text: Input text to process
            context_history: Previous texts for context

        Returns:
            Dictionary with context-aware processing results
        """
        if not text or not isinstance(text, str):
            return self._create_empty_result()

        # Check cache
        cache_key = hash((text, str(context_history or [])))
        if self.config['cache_enabled'] and cache_key in self._processing_cache:
            return self._processing_cache[cache_key]

        try:
            result = {
                'original_text': text,
                'processed_text': text,
                'context_analysis': {},
                'disambiguated_terms': [],
                'multiword_expressions': [],
                'named_entities': [],
                'quality_score': 0.0,
                'confidence_score': 1.0,
                'processing_steps': []
            }

            # Step 1: Context analysis
            context_info = self._analyze_context(text, context_history)
            result['context_analysis'] = context_info
            result['processing_steps'].append('context_analysis')

            # Step 2: Multi-word expression recognition
            if self.config['enable_multiword_recognition']:
                multiword_info = self._recognize_multiword_expressions(text, context_info)
                result['multiword_expressions'] = multiword_info
                result['processing_steps'].append('multiword_recognition')

            # Step 3: Entity disambiguation
            if self.config['enable_entity_disambiguation']:
                disambiguation_info = self._disambiguate_entities(text, context_info)
                result['disambiguated_terms'] = disambiguation_info
                result['processing_steps'].append('entity_disambiguation')

            # Step 4: Named entity recognition
            entity_info = self._recognize_named_entities(text, context_info)
            result['named_entities'] = entity_info
            result['processing_steps'].append('named_entity_recognition')

            # Step 5: Pattern-based cleaning
            cleaned_text = self._apply_pattern_cleaning(text, context_info)
            result['processed_text'] = cleaned_text
            result['processing_steps'].append('pattern_cleaning')

            # Step 6: Quality assessment
            if self.config['quality_assessment_enabled']:
                quality_info = self._assess_processing_quality(result)
                result['quality_score'] = quality_info['overall_quality']
                result['confidence_score'] = quality_info['confidence']
                result['processing_steps'].append('quality_assessment')

            # Step 7: Update context memory
            self._update_context_memory(text, result)

            # Cache result
            if self.config['cache_enabled']:
                self._processing_cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Error in context-aware processing: {str(e)}")
            return self._create_error_result(text, str(e))

    def _analyze_context(self, text: str, context_history: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze context from text and history"""
        try:
            context_info = {
                'primary_domain': 'unknown',
                'confidence': 0.0,
                'contextual_terms': [],
                'temporal_indicators': [],
                'domain_indicators': []
            }

            # Analyze current text
            text_lower = text.lower()
            domain_scores = defaultdict(float)

            for domain, indicators in self.context_indicators.items():
                matches = sum(1 for indicator in indicators if indicator in text_lower)
                if matches > 0:
                    domain_scores[domain] = matches / len(indicators)

            if domain_scores:
                primary_domain = max(domain_scores.items(), key=lambda x: x[1])
                context_info['primary_domain'] = primary_domain[0]
                context_info['confidence'] = primary_domain[1]
                context_info['domain_indicators'] = [k for k, v in domain_scores.items() if v > 0.1]

            # Analyze context history
            if context_history:
                historical_terms = []
                for prev_text in context_history[-5:]:  # Last 5 texts
                    prev_lower = prev_text.lower()
                    for domain, indicators in self.context_indicators.items():
                        matches = [ind for ind in indicators if ind in prev_lower]
                        historical_terms.extend(matches)

                context_info['contextual_terms'] = list(set(historical_terms))

            # Extract temporal indicators
            temporal_patterns = [
                r'ontem|hoje|amanha',
                r'semana\s+passada|esta\s+semana|proxima\s+semana',
                r'mes\s+passado|este\s+mes|proximo\s+mes'
            ]

            for pattern in temporal_patterns:
                if re.search(pattern, text_lower):
                    context_info['temporal_indicators'].append(pattern)

            return context_info

        except Exception as e:
            self.logger.warning(f"Error analyzing context: {str(e)}")
            return {'primary_domain': 'unknown', 'confidence': 0.0}

    def _recognize_multiword_expressions(self, text: str, context_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recognize multi-word expressions in text"""
        try:
            multiword_info = []
            text_lower = text.lower()
            words = text_lower.split()

            # Check for known multi-word expressions
            for expression, components in self.multiword_expressions.items():
                # Check if all components are present
                if all(component in text_lower for component in components):
                    # Check if components appear in sequence or nearby
                    component_positions = []
                    for component in components:
                        positions = [i for i, word in enumerate(words) if component in word]
                        component_positions.append(positions)

                    if component_positions and all(positions for positions in component_positions):
                        # Find if components are close to each other
                        min_pos = min(min(pos) for pos in component_positions)
                        max_pos = max(max(pos) for pos in component_positions)

                        if max_pos - min_pos <= self.config['context_window_size']:
                            multiword_info.append({
                                'expression': expression,
                                'components': components,
                                'confidence': 0.8,
                                'context_relevance': context_info.get('confidence', 0.0)
                            })

            return multiword_info

        except Exception as e:
            self.logger.warning(f"Error recognizing multi-word expressions: {str(e)}")
            return []

    def _disambiguate_entities(self, text: str, context_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Disambiguate ambiguous terms based on context"""
        try:
            disambiguation_info = []
            text_lower = text.lower()
            words = text_lower.split()

            for ambiguous_term, resolutions in self.ambiguous_terms.items():
                if ambiguous_term in words:
                    term_index = words.index(ambiguous_term)

                    # Look for resolution terms in context window
                    window_start = max(0, term_index - self.config['context_window_size'])
                    window_end = min(len(words), term_index + self.config['context_window_size'] + 1)
                    context_window = words[window_start:window_end]

                    best_resolution = None
                    best_confidence = 0.0

                    for resolution_term, resolved_form in resolutions.items():
                        if resolution_term in context_window:
                            # Calculate confidence based on proximity
                            resolution_index = context_window.index(resolution_term)
                            distance = abs(resolution_index - (term_index - window_start))
                            confidence = max(0.5, 1.0 - (distance * 0.1))

                            if confidence > best_confidence:
                                best_resolution = resolved_form
                                best_confidence = confidence

                    if best_resolution and best_confidence >= self.config['min_confidence_threshold']:
                        disambiguation_info.append({
                            'original_term': ambiguous_term,
                            'disambiguated_term': best_resolution,
                            'context_term': resolution_term if 'resolution_term' in locals() else '',
                            'confidence': best_confidence,
                            'method': 'context_window'
                        })

            return disambiguation_info

        except Exception as e:
            self.logger.warning(f"Error disambiguating entities: {str(e)}")
            return []

    def _recognize_named_entities(self, text: str, context_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recognize named entities in financial context"""
        try:
            entities = []

            # Apply entity patterns
            for entity_type, pattern in self.entity_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    confidence = self._calculate_entity_confidence(match, entity_type, context_info)

                    if confidence >= self.config['min_confidence_threshold']:
                        entities.append({
                            'text': match,
                            'type': entity_type,
                            'confidence': confidence,
                            'context_relevance': context_info.get('confidence', 0.0)
                        })

            # Recognize bank names
            bank_patterns = [
                r'\b(itau|itaú|bradesco|santander|banco\s+brasil|caixa)\b',
                r'\b(nubank|inter|c6|original|btg)\b'
            ]

            for pattern in bank_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match,
                        'type': 'bank_name',
                        'confidence': 0.9,
                        'context_relevance': context_info.get('confidence', 0.0)
                    })

            return entities

        except Exception as e:
            self.logger.warning(f"Error recognizing named entities: {str(e)}")
            return []

    def _calculate_entity_confidence(self, entity_text: str, entity_type: str, context_info: Dict[str, Any]) -> float:
        """Calculate confidence score for entity recognition"""
        try:
            base_confidence = 0.7  # Base confidence

            # Adjust based on entity type
            type_multipliers = {
                'monetary_amount': 0.9,
                'account_number': 0.8,
                'agency_number': 0.8,
                'document_number': 0.95,
                'transaction_code': 0.85,
                'date_pattern': 0.8
            }

            confidence = base_confidence * type_multipliers.get(entity_type, 1.0)

            # Adjust based on context relevance
            context_conf = context_info.get('confidence', 0.0)
            if context_conf > 0.5:
                confidence *= 1.1

            return min(confidence, 1.0)

        except Exception:
            return 0.5

    def _apply_pattern_cleaning(self, text: str, context_info: Dict[str, Any]) -> str:
        """Apply pattern-based cleaning"""
        try:
            cleaned_text = text

            # Remove noise patterns
            for pattern in self.patterns['noise_patterns']:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

            # Clean up extra whitespace
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

            # Remove leading/trailing whitespace
            cleaned_text = cleaned_text.strip()

            return cleaned_text

        except Exception as e:
            self.logger.warning(f"Error in pattern cleaning: {str(e)}")
            return text

    def _assess_processing_quality(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality of processing results"""
        try:
            quality_metrics = {
                'entity_recognition_quality': 0.0,
                'disambiguation_quality': 0.0,
                'context_relevance': 0.0,
                'text_coherence': 0.0
            }

            # Entity recognition quality
            entities = result.get('named_entities', [])
            if entities:
                avg_confidence = sum(e['confidence'] for e in entities) / len(entities)
                quality_metrics['entity_recognition_quality'] = avg_confidence

            # Disambiguation quality
            disambiguations = result.get('disambiguated_terms', [])
            if disambiguations:
                avg_confidence = sum(d['confidence'] for d in disambiguations) / len(disambiguations)
                quality_metrics['disambiguation_quality'] = avg_confidence

            # Context relevance
            context_analysis = result.get('context_analysis', {})
            quality_metrics['context_relevance'] = context_analysis.get('confidence', 0.0)

            # Text coherence (simplified)
            processed_text = result.get('processed_text', '')
            if processed_text:
                # Check for reasonable length and word count
                word_count = len(processed_text.split())
                quality_metrics['text_coherence'] = min(word_count / 20, 1.0)  # Normalize to 0-1

            # Overall quality score
            valid_scores = [v for v in quality_metrics.values() if v > 0]
            overall_quality = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

            return {
                'overall_quality': overall_quality,
                'confidence': overall_quality,
                'metrics': quality_metrics
            }

        except Exception as e:
            self.logger.warning(f"Error assessing processing quality: {str(e)}")
            return {'overall_quality': 0.0, 'confidence': 0.0}

    def _update_context_memory(self, text: str, result: Dict[str, Any]):
        """Update context memory with processing results"""
        try:
            # Store recent processing results for context
            context_entry = {
                'text': text,
                'domain': result.get('context_analysis', {}).get('primary_domain', 'unknown'),
                'entities': [e['text'] for e in result.get('named_entities', [])],
                'timestamp': 'current'  # In real implementation, use actual timestamp
            }

            # Keep only recent entries
            self._context_memory['recent'].append(context_entry)
            if len(self._context_memory['recent']) > 10:
                self._context_memory['recent'].pop(0)

        except Exception as e:
            self.logger.warning(f"Error updating context memory: {str(e)}")

    def process_batch_with_context(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of texts with context awareness

        Args:
            texts: List of texts to process

        Returns:
            List of processing results
        """
        if not texts:
            return []

        try:
            results = []
            context_history = []

            for text in texts:
                result = self.process_with_context(text, context_history)
                results.append(result)

                # Update context history
                context_history.append(text)
                if len(context_history) > 5:  # Keep last 5 texts
                    context_history.pop(0)

            self.logger.info(f"Processed {len(texts)} texts with context awareness")
            return results

        except Exception as e:
            self.logger.error(f"Error in batch context processing: {str(e)}")
            return [self._create_error_result(text, str(e)) for text in texts]

    def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about context processing"""
        try:
            stats = {
                'cache_size': len(self._processing_cache),
                'context_memory_size': len(self._context_memory.get('recent', [])),
                'ambiguous_terms_count': len(self.ambiguous_terms),
                'multiword_expressions_count': len(self.multiword_expressions)
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error getting context statistics: {str(e)}")
            return {}

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result for invalid input"""
        return {
            'original_text': '',
            'processed_text': '',
            'context_analysis': {},
            'disambiguated_terms': [],
            'multiword_expressions': [],
            'named_entities': [],
            'quality_score': 0.0,
            'confidence_score': 0.0,
            'processing_steps': []
        }

    def _create_error_result(self, text: str, error: str) -> Dict[str, Any]:
        """Create error result"""
        result = self._create_empty_result()
        result['original_text'] = text
        result['error'] = error
        return result

    def clear_cache(self):
        """Clear processing cache and context memory"""
        self._processing_cache.clear()
        self._context_memory.clear()
        self.logger.info("Context-aware processing cache cleared")