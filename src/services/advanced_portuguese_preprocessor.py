import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import spacy
from collections import Counter
import logging

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class AdvancedPortuguesePreprocessor:
    """
    Advanced Portuguese text preprocessor with stemming, lemmatization, and enhanced NLP capabilities
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced Portuguese preprocessor

        Args:
            config: Configuration dictionary for preprocessing options
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize components
        self._initialize_components()

        # Load Portuguese language models
        self._load_language_models()

        # Initialize caches
        self._processed_cache = {}
        self._entity_cache = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'use_spacy': True,
            'use_nltk': True,
            'lemmatization': True,
            'advanced_stemming': True,
            'enhanced_stopwords': True,
            'accent_normalization': True,
            'encoding_handling': True,
            'cache_enabled': True,
            'batch_size': 100,
            'confidence_threshold': 0.7
        }

    def _initialize_components(self):
        """Initialize NLTK components"""
        try:
            # Download required NLTK data
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('rslp', quiet=True)

            # Initialize NLTK stemmer
            self.nltk_stemmer = RSLPStemmer()

            # Get Portuguese stopwords
            self.nltk_stopwords = set(stopwords.words('portuguese'))

            # Enhanced financial stopwords
            self.financial_stopwords = {
                'valor', 'total', 'saldo', 'conta', 'banco', 'agencia', 'numero',
                'data', 'hora', 'tipo', 'operacao', 'transacao', 'debito', 'credito',
                'pagamento', 'recebimento', 'transferencia', 'real', 'reais', 'rs',
                'r$', 'centavo', 'centavos', 'moeda', 'dinheiro', 'financeiro'
            }

            self.all_stopwords = self.nltk_stopwords.union(self.financial_stopwords)

            self.logger.info("NLTK components initialized successfully")

        except Exception as e:
            self.logger.warning(f"Failed to initialize NLTK components: {str(e)}")
            self.nltk_stemmer = None
            self.all_stopwords = set()

    def _load_language_models(self):
        """Load spaCy Portuguese language model"""
        try:
            if self.config['use_spacy']:
                self.nlp = spacy.load('pt_core_news_lg')
                self.logger.info("spaCy Portuguese model loaded successfully")
            else:
                self.nlp = None
        except Exception as e:
            self.logger.warning(f"Failed to load spaCy model: {str(e)}. Falling back to NLTK only.")
            self.nlp = None
            self.config['use_spacy'] = False

    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Advanced preprocessing of Portuguese text

        Args:
            text: Input text to preprocess

        Returns:
            Dictionary with preprocessing results and metadata
        """
        if not text or not isinstance(text, str):
            return self._create_empty_result()

        # Check cache
        cache_key = hash(text)
        if self.config['cache_enabled'] and cache_key in self._processed_cache:
            return self._processed_cache[cache_key]

        try:
            result = {
                'original_text': text,
                'processed_text': text,
                'tokens': [],
                'lemmas': [],
                'stems': [],
                'pos_tags': [],
                'entities': [],
                'confidence_score': 1.0,
                'processing_steps': [],
                'quality_metrics': {}
            }

            # Step 1: Text normalization
            normalized_text = self._normalize_text(text)
            result['processed_text'] = normalized_text
            result['processing_steps'].append('normalization')

            # Step 2: Tokenization and POS tagging
            if self.nlp:
                doc = self.nlp(normalized_text)
                tokens = [token.text for token in doc]
                lemmas = [token.lemma_ for token in doc]
                pos_tags = [token.pos_ for token in doc]
                entities = [(ent.text, ent.label_) for ent in doc.ents]

                result['tokens'] = tokens
                result['lemmas'] = lemmas
                result['pos_tags'] = pos_tags
                result['entities'] = entities
                result['processing_steps'].append('spacy_processing')

            # Step 3: Advanced stemming (fallback to NLTK)
            if not self.nlp or not result['lemmas']:
                nltk_tokens = nltk.word_tokenize(normalized_text)
                result['tokens'] = nltk_tokens

                if self.config['advanced_stemming'] and self.nltk_stemmer:
                    stems = [self.nltk_stemmer.stem(token) for token in nltk_tokens]
                    result['stems'] = stems
                    result['processing_steps'].append('nltk_stemming')

            # Step 4: Enhanced stopword filtering
            if self.config['enhanced_stopwords']:
                filtered_result = self._apply_enhanced_stopword_filtering(result)
                result.update(filtered_result)
                result['processing_steps'].append('enhanced_stopword_filtering')

            # Step 5: Quality assessment
            result['quality_metrics'] = self._assess_text_quality(result)

            # Step 6: Final text reconstruction
            result['processed_text'] = self._reconstruct_processed_text(result)

            # Cache result
            if self.config['cache_enabled']:
                self._processed_cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Error in advanced preprocessing: {str(e)}")
            return self._create_error_result(text, str(e))

    def _normalize_text(self, text: str) -> str:
        """Advanced text normalization"""
        try:
            # Handle encoding issues
            if self.config['encoding_handling']:
                text = self._fix_encoding_issues(text)

            # Accent normalization
            if self.config['accent_normalization']:
                text = self._normalize_accents(text)

            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)

            # Remove control characters
            text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

            return text.strip()

        except Exception as e:
            self.logger.warning(f"Error in text normalization: {str(e)}")
            return text

    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common Portuguese encoding issues"""
        # Common encoding fixes for Portuguese
        fixes = {
            'Ã£': 'ã', 'Ã¢': 'â', 'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í',
            'Ã³': 'ó', 'Ãº': 'ú', 'Ã§': 'ç', 'Ãµ': 'õ', 'Ãª': 'ê',
            'Ã ': 'à', 'Ãš': 'Ú', 'Ã‰': 'É', 'Ã“': 'Ó', 'Ã': 'Á'
        }

        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)

        return text

    def _normalize_accents(self, text: str) -> str:
        """Normalize accents using Unicode normalization"""
        try:
            # NFKD normalization separates accents
            normalized = unicodedata.normalize('NFKD', text)
            # Remove combining characters (accents)
            without_accents = ''.join(
                char for char in normalized
                if unicodedata.category(char) != 'Mn'
            )
            return without_accents
        except Exception:
            return text

    def _apply_enhanced_stopword_filtering(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply enhanced stopword filtering with context awareness"""
        try:
            filtered_tokens = []
            filtered_lemmas = []
            filtered_stems = []

            tokens = result.get('tokens', [])
            lemmas = result.get('lemmas', [])
            stems = result.get('stems', [])

            for i, token in enumerate(tokens):
                # Skip if token is a stopword
                if token.lower() in self.all_stopwords:
                    continue

                # Skip if token is too short
                if len(token) < 2:
                    continue

                # Skip if token is numeric
                if token.isdigit():
                    continue

                # Context-aware filtering
                if self._is_contextually_relevant(token, tokens, i):
                    filtered_tokens.append(token)
                    if lemmas and i < len(lemmas):
                        filtered_lemmas.append(lemmas[i])
                    if stems and i < len(stems):
                        filtered_stems.append(stems[i])

            return {
                'filtered_tokens': filtered_tokens,
                'filtered_lemmas': filtered_lemmas,
                'filtered_stems': filtered_stems
            }

        except Exception as e:
            self.logger.warning(f"Error in enhanced stopword filtering: {str(e)}")
            return {}

    def _is_contextually_relevant(self, token: str, tokens: List[str], position: int) -> bool:
        """Check if token is contextually relevant"""
        # Financial terms that should not be filtered
        financial_terms = {
            'banco', 'conta', 'agencia', 'numero', 'codigo', 'cpf', 'cnpj',
            'pix', 'ted', 'doc', 'boleto', 'fatura', 'nota', 'fiscal'
        }

        if token.lower() in financial_terms:
            return True

        # Check if token is part of a compound term
        if position > 0 and position < len(tokens) - 1:
            prev_token = tokens[position - 1].lower()
            next_token = tokens[position + 1].lower()

            # Financial compound terms
            if (prev_token in ['numero', 'codigo', 'agencia'] or
                next_token in ['banco', 'conta', 'agencia']):
                return True

        return False

    def _assess_text_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of processed text"""
        try:
            metrics = {
                'token_count': len(result.get('tokens', [])),
                'filtered_token_count': len(result.get('filtered_tokens', [])),
                'entity_count': len(result.get('entities', [])),
                'avg_token_length': 0.0,
                'lexical_diversity': 0.0,
                'has_financial_terms': False
            }

            tokens = result.get('filtered_tokens', result.get('tokens', []))
            if tokens:
                metrics['avg_token_length'] = sum(len(t) for t in tokens) / len(tokens)
                metrics['lexical_diversity'] = len(set(tokens)) / len(tokens)

            # Check for financial terms
            financial_indicators = ['banco', 'conta', 'valor', 'pagamento', 'transferencia']
            text_lower = result.get('processed_text', '').lower()
            metrics['has_financial_terms'] = any(term in text_lower for term in financial_indicators)

            return metrics

        except Exception as e:
            self.logger.warning(f"Error assessing text quality: {str(e)}")
            return {}

    def _reconstruct_processed_text(self, result: Dict[str, Any]) -> str:
        """Reconstruct processed text from filtered tokens"""
        try:
            # Use filtered tokens if available, otherwise use original tokens
            tokens = result.get('filtered_tokens', result.get('tokens', []))

            if not tokens:
                return result.get('processed_text', '')

            # Join tokens with spaces
            processed_text = ' '.join(tokens)

            # Final cleaning
            processed_text = re.sub(r'\s+', ' ', processed_text)
            return processed_text.strip()

        except Exception as e:
            self.logger.warning(f"Error reconstructing processed text: {str(e)}")
            return result.get('processed_text', '')

    def preprocess_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Preprocess a batch of texts

        Args:
            texts: List of input texts
            batch_size: Size of processing batches

        Returns:
            List of preprocessing results
        """
        if not texts:
            return []

        batch_size = batch_size or self.config['batch_size']

        try:
            results = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1} of {(len(texts) + batch_size - 1)//batch_size}")

                for text in batch:
                    result = self.preprocess_text(text)
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Error in batch preprocessing: {str(e)}")
            return [self._create_error_result(text, str(e)) for text in texts]

    def get_processing_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about preprocessing results

        Args:
            results: List of preprocessing results

        Returns:
            Dictionary with processing statistics
        """
        try:
            stats = {
                'total_texts': len(results),
                'successful_processing': 0,
                'average_confidence': 0.0,
                'average_token_count': 0.0,
                'average_quality_score': 0.0,
                'error_count': 0,
                'cache_hit_rate': 0.0
            }

            confidences = []
            token_counts = []
            quality_scores = []

            for result in results:
                if 'error' not in result:
                    stats['successful_processing'] += 1
                    confidences.append(result.get('confidence_score', 0.0))
                    token_counts.append(len(result.get('tokens', [])))

                    # Calculate quality score
                    quality = result.get('quality_metrics', {})
                    quality_score = (
                        (quality.get('lexical_diversity', 0) * 0.3) +
                        (min(quality.get('token_count', 0) / 20, 1.0) * 0.3) +
                        (1.0 if quality.get('has_financial_terms', False) else 0.0) * 0.4
                    )
                    quality_scores.append(quality_score)
                else:
                    stats['error_count'] += 1

            if confidences:
                stats['average_confidence'] = sum(confidences) / len(confidences)
            if token_counts:
                stats['average_token_count'] = sum(token_counts) / len(token_counts)
            if quality_scores:
                stats['average_quality_score'] = sum(quality_scores) / len(quality_scores)

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating processing statistics: {str(e)}")
            return {}

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result for invalid input"""
        return {
            'original_text': '',
            'processed_text': '',
            'tokens': [],
            'lemmas': [],
            'stems': [],
            'pos_tags': [],
            'entities': [],
            'confidence_score': 0.0,
            'processing_steps': [],
            'quality_metrics': {}
        }

    def _create_error_result(self, text: str, error: str) -> Dict[str, Any]:
        """Create error result"""
        result = self._create_empty_result()
        result['original_text'] = text
        result['error'] = error
        result['confidence_score'] = 0.0
        return result

    def clear_cache(self):
        """Clear processing cache"""
        self._processed_cache.clear()
        self._entity_cache.clear()
        self.logger.info("Processing cache cleared")