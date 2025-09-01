import re
import unicodedata
from typing import Dict, Any, List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

from src.utils.logging_config import get_logger
from src.services.preprocessing_pipeline import PreprocessingPipeline, ProcessingStep, PipelineConfig

logger = get_logger(__name__)


class PortugueseTextPreprocessor:
    """
    Portuguese text preprocessor for financial transaction descriptions
    Enhanced with advanced preprocessing pipeline integration
    """

    def __init__(self, use_advanced_pipeline: bool = True):
        """
        Initialize the Portuguese text preprocessor

        Args:
            use_advanced_pipeline: Whether to use the advanced preprocessing pipeline
        """
        self.logger = get_logger(__name__)
        self.use_advanced_pipeline = use_advanced_pipeline

        # Initialize NLTK resources (legacy)
        self._initialize_nltk()

        # Financial-specific patterns (legacy)
        self.financial_patterns = self._get_financial_patterns()

        # Common financial abbreviations and their expansions (legacy)
        self.abbreviations = self._get_abbreviations()

        # Initialize advanced pipeline if requested
        if self.use_advanced_pipeline:
            self._initialize_advanced_pipeline()
        else:
            self.advanced_pipeline = None

    def _initialize_nltk(self):
        """Initialize NLTK resources for Portuguese"""
        try:
            # Download required NLTK data
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('rslp', quiet=True)

            # Initialize Portuguese stemmer
            self.stemmer = RSLPStemmer()

            # Get Portuguese stopwords
            self.stopwords = set(stopwords.words('portuguese'))

            # Add financial-specific stopwords
            financial_stopwords = {
                'valor', 'total', 'saldo', 'conta', 'banco', 'agencia',
                'numero', 'data', 'hora', 'tipo', 'operacao', 'transacao',
                'debito', 'credito', 'pagamento', 'recebimento', 'transferencia'
            }
            self.stopwords.update(financial_stopwords)

            self.logger.info("NLTK resources initialized successfully")

        except Exception as e:
            self.logger.warning(f"Failed to initialize NLTK resources: {str(e)}")
            # Fallback initialization
            self.stemmer = None
            self.stopwords = set()

    def _initialize_advanced_pipeline(self):
        """Initialize the advanced preprocessing pipeline"""
        try:
            # Configure pipeline with all available steps
            pipeline_config = PipelineConfig(
                steps=[
                    ProcessingStep.ADVANCED_PORTUGUESE,
                    ProcessingStep.FINANCIAL_PROCESSING,
                    ProcessingStep.CONTEXT_AWARE,
                    ProcessingStep.QUALITY_ASSESSMENT
                ],
                batch_size=32,
                max_workers=4,
                enable_parallel_processing=True,
                error_handling="continue",
                quality_threshold=0.6,
                cache_enabled=True,
                performance_monitoring=True
            )

            self.advanced_pipeline = PreprocessingPipeline(pipeline_config)
            self.logger.info("Advanced preprocessing pipeline initialized successfully")

        except Exception as e:
            self.logger.warning(f"Failed to initialize advanced pipeline: {str(e)}. Falling back to legacy mode.")
            self.advanced_pipeline = None
            self.use_advanced_pipeline = False

    def _get_financial_patterns(self) -> Dict[str, str]:
        """Get financial-specific text patterns for normalization"""
        return {
            # Remove common prefixes
            r'^\s*(compra|pagamento|recebimento|transferencia|ted|doc|pix)\s+': '',
            r'^\s*(debito|credito)\s+em\s+': '',

            # Normalize bank names
            r'\b(itau|itaú)\b': 'itau',
            r'\b(bradesco|brad)\b': 'bradesco',
            r'\b(santander|sant)\b': 'santander',
            r'\b(bb|brasil)\b': 'banco_brasil',

            # Normalize transaction types
            r'\b(doc|ted|pix)\b': 'transferencia',
            r'\b(saque|retirada)\b': 'saque',
            r'\b(deposito|depósito)\b': 'deposito',

            # Remove monetary symbols and amounts
            r'r\$?\s*\d+[\.,]\d{2}': 'valor_monetario',
            r'\b\d+[\.,]\d{2}\b': 'valor',

            # Remove dates
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b': 'data',
            r'\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b': 'data',

            # Remove times
            r'\b\d{1,2}:\d{2}(:\d{2})?\b': 'hora',

            # Remove account/agency numbers
            r'\b\d{4,5}[-]?\d{1,2}\b': 'numero_conta',
            r'\bag\.?\s*\d{4}\b': 'agencia',

            # Remove extra whitespace
            r'\s+': ' ',
        }

    def _get_abbreviations(self) -> Dict[str, str]:
        """Get common financial abbreviations and their expansions"""
        return {
            'r$': 'reais',
            'rs': 'reais',
            'cx': 'caixa',
            'dep': 'deposito',
            'transf': 'transferencia',
            'pgto': 'pagamento',
            'rcbto': 'recebimento',
            'liq': 'liquidacao',
            'est': 'estorno',
            'dev': 'devolucao',
            'sld': 'saldo',
            'lim': 'limite',
            'disp': 'disponivel',
            'bloq': 'bloqueado',
            'canc': 'cancelado',
            'proc': 'processamento',
            'aut': 'autorizacao',
            'ref': 'referencia',
            'doc': 'documento',
            'ted': 'transferencia_eletronica_disponivel',
            'pix': 'pagamento_instantaneo',
        }

    def preprocess(self, text: str, config: Dict[str, Any]) -> str:
        """
        Preprocess Portuguese text with configurable options

        Args:
            text: Input text to preprocess
            config: Preprocessing configuration

        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""

        # Use advanced pipeline if available and requested
        if self.use_advanced_pipeline and self.advanced_pipeline:
            return self._preprocess_with_advanced_pipeline(text, config)

        # Fall back to legacy preprocessing
        return self._preprocess_legacy(text, config)

    def _preprocess_with_advanced_pipeline(self, text: str, config: Dict[str, Any]) -> str:
        """
        Preprocess text using the advanced pipeline

        Args:
            text: Input text
            config: Preprocessing configuration

        Returns:
            Preprocessed text
        """
        try:
            # Process through advanced pipeline
            result = self.advanced_pipeline.process_text(text)

            if result.success:
                # Extract the final processed text
                processed_text = result.processed_text

                # Apply legacy config options that might not be covered by pipeline
                processed_text = self._apply_legacy_config(processed_text, config)

                self.logger.debug(f"Advanced preprocessing completed for text: {text[:50]}...")
                return processed_text
            else:
                self.logger.warning(f"Advanced preprocessing failed: {result.error_message}")
                # Fall back to legacy preprocessing
                return self._preprocess_legacy(text, config)

        except Exception as e:
            self.logger.error(f"Error in advanced preprocessing: {str(e)}")
            # Fall back to legacy preprocessing
            return self._preprocess_legacy(text, config)

    def _preprocess_legacy(self, text: str, config: Dict[str, Any]) -> str:
        """
        Legacy preprocessing method (original implementation)

        Args:
            text: Input text to preprocess
            config: Preprocessing configuration

        Returns:
            Preprocessed text
        """
        try:
            # Convert to string and basic cleaning
            processed_text = str(text).strip()

            if not processed_text:
                return ""

            # Apply financial patterns
            processed_text = self._apply_financial_patterns(processed_text)

            # Expand abbreviations
            if config.get('expand_abbreviations', True):
                processed_text = self._expand_abbreviations(processed_text)

            # Convert to lowercase
            if config.get('lowercase', True):
                processed_text = processed_text.lower()

            # Remove accents
            if config.get('remove_accents', True):
                processed_text = self._remove_accents(processed_text)

            # Remove numbers
            if config.get('remove_numbers', False):
                processed_text = re.sub(r'\d+', '', processed_text)

            # Remove punctuation
            if config.get('remove_punctuation', True):
                processed_text = re.sub(r'[^\w\s]', ' ', processed_text)

            # Remove stopwords
            if config.get('stopwords', True):
                processed_text = self._remove_stopwords(processed_text)

            # Apply stemming
            if config.get('stemming', False):
                processed_text = self._apply_stemming(processed_text)

            # Final cleaning
            processed_text = self._final_cleaning(processed_text)

            return processed_text

        except Exception as e:
            self.logger.error(f"Error in legacy preprocessing '{text}': {str(e)}")
            return text  # Return original text as fallback

    def _apply_legacy_config(self, text: str, config: Dict[str, Any]) -> str:
        """
        Apply legacy configuration options to advanced pipeline results

        Args:
            text: Processed text from pipeline
            config: Legacy configuration

        Returns:
            Text with legacy config applied
        """
        try:
            processed_text = text

            # Apply config options that might not be handled by the pipeline
            if config.get('remove_numbers', False):
                processed_text = re.sub(r'\d+', '', processed_text)

            if config.get('stemming', False) and not self.use_advanced_pipeline:
                # Only apply legacy stemming if not using advanced pipeline
                processed_text = self._apply_stemming(processed_text)

            return processed_text

        except Exception as e:
            self.logger.warning(f"Error applying legacy config: {str(e)}")
            return text

    def _apply_financial_patterns(self, text: str) -> str:
        """Apply financial-specific text patterns"""
        for pattern, replacement in self.financial_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common financial abbreviations"""
        words = text.split()
        expanded_words = []

        for word in words:
            # Check for exact matches first
            if word.lower() in self.abbreviations:
                expanded_words.append(self.abbreviations[word.lower()])
            else:
                expanded_words.append(word)

        return ' '.join(expanded_words)

    def _remove_accents(self, text: str) -> str:
        """Remove accents from Portuguese text"""
        try:
            # Normalize to decomposed form (NFD)
            normalized = unicodedata.normalize('NFD', text)
            # Remove combining characters (accents)
            without_accents = ''.join(
                char for char in normalized
                if unicodedata.category(char) != 'Mn'
            )
            return without_accents
        except Exception:
            return text

    def _remove_stopwords(self, text: str) -> str:
        """Remove Portuguese stopwords"""
        if not self.stopwords:
            return text

        words = text.split()
        filtered_words = [
            word for word in words
            if word.lower() not in self.stopwords and len(word) > 1
        ]
        return ' '.join(filtered_words)

    def _apply_stemming(self, text: str) -> str:
        """Apply Portuguese stemming"""
        if not self.stemmer:
            return text

        words = text.split()
        stemmed_words = []

        for word in words:
            try:
                stemmed = self.stemmer.stem(word)
                stemmed_words.append(stemmed)
            except Exception:
                stemmed_words.append(word)

        return ' '.join(stemmed_words)

    def _final_cleaning(self, text: str) -> str:
        """Apply final cleaning steps"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Remove duplicate words (simple approach)
        words = text.split()
        unique_words = []
        seen = set()

        for word in words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)

        return ' '.join(unique_words)

    def preprocess_batch(self, texts: List[str], config: Dict[str, Any]) -> List[str]:
        """
        Preprocess a batch of texts

        Args:
            texts: List of input texts
            config: Preprocessing configuration

        Returns:
            List of preprocessed texts
        """
        try:
            processed_texts = []

            for text in texts:
                processed = self.preprocess(text, config)
                processed_texts.append(processed)

            self.logger.info(f"Processed {len(texts)} texts successfully")
            return processed_texts

        except Exception as e:
            self.logger.error(f"Error in batch preprocessing: {str(e)}")
            return texts  # Return original texts as fallback

    def get_preprocessing_stats(self, texts: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about text preprocessing

        Args:
            texts: List of input texts
            config: Preprocessing configuration

        Returns:
            Dictionary with preprocessing statistics
        """
        try:
            original_lengths = [len(text) for text in texts]
            processed_texts = self.preprocess_batch(texts, config)
            processed_lengths = [len(text) for text in processed_texts]

            stats = {
                'total_texts': len(texts),
                'avg_original_length': sum(original_lengths) / len(original_lengths),
                'avg_processed_length': sum(processed_lengths) / len(processed_lengths),
                'total_length_reduction': sum(original_lengths) - sum(processed_lengths),
                'avg_length_reduction': (sum(original_lengths) - sum(processed_lengths)) / len(texts),
                'empty_texts_after_processing': sum(1 for text in processed_texts if not text.strip())
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating preprocessing stats: {str(e)}")
            return {}

    def extract_financial_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract financial entities from text (basic implementation)

        Args:
            text: Input text

        Returns:
            Dictionary with extracted entities
        """
        entities = {
            'amounts': [],
            'dates': [],
            'banks': [],
            'transaction_types': []
        }

        try:
            # Extract monetary amounts
            amount_pattern = r'r\$?\s*(\d+[\.,]\d{2})'
            amounts = re.findall(amount_pattern, text, re.IGNORECASE)
            entities['amounts'] = amounts

            # Extract dates
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
            dates = re.findall(date_pattern, text)
            entities['dates'] = dates

            # Extract bank names
            banks = ['itau', 'bradesco', 'santander', 'banco_brasil', 'caixa']
            found_banks = [bank for bank in banks if bank in text.lower()]
            entities['banks'] = found_banks

            # Extract transaction types
            transaction_types = ['transferencia', 'saque', 'deposito', 'pagamento', 'recebimento']
            found_types = [ttype for ttype in transaction_types if ttype in text.lower()]
            entities['transaction_types'] = found_types

        except Exception as e:
            self.logger.error(f"Error extracting financial entities: {str(e)}")

        return entities

    def preprocess_with_advanced_features(self, text: str) -> Dict[str, Any]:
        """
        Preprocess text using advanced features and return comprehensive results

        Args:
            text: Input text to preprocess

        Returns:
            Dictionary with comprehensive preprocessing results
        """
        if not self.use_advanced_pipeline or not self.advanced_pipeline:
            # Fall back to legacy preprocessing with basic results
            processed_text = self.preprocess(text, {})
            return {
                'original_text': text,
                'processed_text': processed_text,
                'method': 'legacy',
                'quality_score': 0.5,
                'entities': self.extract_financial_entities(text)
            }

        try:
            # Use advanced pipeline
            result = self.advanced_pipeline.process_text(text)

            # Convert to expected format
            advanced_result = {
                'original_text': result.original_text,
                'processed_text': result.processed_text,
                'method': 'advanced',
                'quality_score': result.quality_metrics.get('overall_quality', 0.0),
                'processing_time': result.processing_time,
                'success': result.success,
                'intermediate_results': result.intermediate_results,
                'quality_metrics': result.quality_metrics
            }

            # Extract entities from intermediate results
            entities = {}
            for step_name, step_result in result.intermediate_results.items():
                if isinstance(step_result, dict):
                    if 'entities' in step_result:
                        entities.update({f"{step_name}_{k}": v for k, v in step_result['entities'].items()})
                    elif 'extracted_entities' in step_result:
                        entities.update(step_result['extracted_entities'])

            advanced_result['entities'] = entities

            return advanced_result

        except Exception as e:
            self.logger.error(f"Error in advanced preprocessing: {str(e)}")
            return self.preprocess_with_advanced_features.__wrapped__(self, text)

    def get_advanced_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from the advanced preprocessing system

        Returns:
            Dictionary with preprocessing metrics
        """
        if not self.use_advanced_pipeline or not self.advanced_pipeline:
            return {
                'method': 'legacy',
                'features': ['basic_preprocessing', 'financial_patterns', 'stopword_filtering']
            }

        try:
            pipeline_metrics = self.advanced_pipeline.get_pipeline_metrics()

            return {
                'method': 'advanced',
                'pipeline_metrics': pipeline_metrics,
                'features': [
                    'advanced_portuguese_processing',
                    'financial_text_processing',
                    'context_aware_processing',
                    'quality_assessment',
                    'parallel_processing',
                    'caching'
                ]
            }

        except Exception as e:
            self.logger.error(f"Error getting advanced metrics: {str(e)}")
            return {'method': 'error', 'error': str(e)}

    def enable_advanced_mode(self):
        """Enable advanced preprocessing mode"""
        if not self.use_advanced_pipeline:
            self.use_advanced_pipeline = True
            if not self.advanced_pipeline:
                self._initialize_advanced_pipeline()
        self.logger.info("Advanced preprocessing mode enabled")

    def disable_advanced_mode(self):
        """Disable advanced preprocessing mode"""
        self.use_advanced_pipeline = False
        self.logger.info("Advanced preprocessing mode disabled")

    def clear_advanced_cache(self):
        """Clear the advanced preprocessing cache"""
        if self.advanced_pipeline:
            self.advanced_pipeline.clear_cache()
            self.logger.info("Advanced preprocessing cache cleared")

    def optimize_advanced_pipeline(self, sample_texts: List[str]) -> Dict[str, Any]:
        """
        Optimize the advanced preprocessing pipeline

        Args:
            sample_texts: Sample texts for optimization

        Returns:
            Optimization results
        """
        if not self.advanced_pipeline:
            return {'error': 'Advanced pipeline not available'}

        try:
            return self.advanced_pipeline.optimize_pipeline(sample_texts)
        except Exception as e:
            self.logger.error(f"Error optimizing pipeline: {str(e)}")
            return {'error': str(e)}