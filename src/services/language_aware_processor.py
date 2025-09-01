import re
import nltk
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

from src.utils.logging_config import get_logger
from src.services.language_detector import LanguageDetector, DetectionResult
from src.services.multi_language_preprocessor import MultiLanguagePreprocessor, PreprocessingResult

logger = get_logger(__name__)


@dataclass
class LanguageProcessingConfig:
    """Configuration for language-aware processing"""
    language_code: str
    use_stemming: bool = True
    use_lemmatization: bool = True
    use_stopword_filtering: bool = True
    use_ner: bool = True
    use_translation: bool = False
    min_term_length: int = 2
    max_term_length: int = 50
    confidence_threshold: float = 0.7


@dataclass
class ProcessingResult:
    """Result of language-aware processing"""
    original_text: str
    processed_text: str
    detected_language: str
    language_confidence: float
    tokens: List[str]
    lemmas: List[str]
    stems: List[str]
    pos_tags: List[Tuple[str, str]]
    entities: List[Dict[str, Any]]
    filtered_terms: List[str]
    financial_terms: List[Dict[str, Any]]
    translations: Dict[str, List[str]]
    quality_metrics: Dict[str, float]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class LanguageAwareProcessor:
    """
    Intelligent language-aware processor with stemming, lemmatization,
    NER, and translation capabilities for financial text
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the language-aware processor

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize language detector and preprocessor
        self.language_detector = LanguageDetector()
        self.preprocessor = MultiLanguagePreprocessor()

        # Initialize language-specific processors
        self._language_processors = self._initialize_language_processors()

        # Initialize financial terminology mappings
        self._financial_mappings = self._initialize_financial_mappings()

        # Initialize translation capabilities
        self._translation_enabled = self.config.get('enable_translation', False)
        if self._translation_enabled:
            self._initialize_translation()

        # Processing cache
        self._processing_cache = {}

        # Performance tracking
        self._performance_stats = {
            'total_processed': 0,
            'language_distribution': {},
            'average_processing_time': 0.0,
            'cache_hit_rate': 0.0
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'cache_enabled': True,
            'enable_translation': False,
            'batch_size': 32,
            'timeout_seconds': 30.0,
            'confidence_threshold': 0.6
        }

    def _initialize_language_processors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize language-specific processing components"""
        processors = {}

        # Portuguese
        processors['pt'] = {
            'stemmer': self._initialize_portuguese_stemmer(),
            'lemmatizer': self._initialize_portuguese_lemmatizer(),
            'stopwords': self._get_portuguese_stopwords(),
            'pos_tagger': self._initialize_portuguese_pos_tagger(),
            'ner_model': None  # Would use spaCy pt_core_news_lg
        }

        # English
        processors['en'] = {
            'stemmer': self._initialize_english_stemmer(),
            'lemmatizer': self._initialize_english_lemmatizer(),
            'stopwords': self._get_english_stopwords(),
            'pos_tagger': self._initialize_english_pos_tagger(),
            'ner_model': None  # Would use spaCy en_core_web_sm
        }

        # Spanish
        processors['es'] = {
            'stemmer': self._initialize_spanish_stemmer(),
            'lemmatizer': self._initialize_spanish_lemmatizer(),
            'stopwords': self._get_spanish_stopwords(),
            'pos_tagger': self._initialize_spanish_pos_tagger(),
            'ner_model': None  # Would use spaCy es_core_news_sm
        }

        # French
        processors['fr'] = {
            'stemmer': self._initialize_french_stemmer(),
            'lemmatizer': self._initialize_french_lemmatizer(),
            'stopwords': self._get_french_stopwords(),
            'pos_tagger': self._initialize_french_pos_tagger(),
            'ner_model': None  # Would use spaCy fr_core_news_sm
        }

        # German
        processors['de'] = {
            'stemmer': self._initialize_german_stemmer(),
            'lemmatizer': self._initialize_german_lemmatizer(),
            'stopwords': self._get_german_stopwords(),
            'pos_tagger': self._initialize_german_pos_tagger(),
            'ner_model': None  # Would use spaCy de_core_news_sm
        }

        # Italian
        processors['it'] = {
            'stemmer': self._initialize_italian_stemmer(),
            'lemmatizer': self._initialize_italian_lemmatizer(),
            'stopwords': self._get_italian_stopwords(),
            'pos_tagger': self._initialize_italian_pos_tagger(),
            'ner_model': None  # Would use spaCy it_core_news_sm
        }

        return processors

    def _initialize_portuguese_stemmer(self):
        """Initialize Portuguese stemmer"""
        try:
            from nltk.stem import RSLPStemmer
            return RSLPStemmer()
        except ImportError:
            self.logger.warning("RSLPStemmer not available for Portuguese")
            return None

    def _initialize_portuguese_lemmatizer(self):
        """Initialize Portuguese lemmatizer"""
        try:
            # Would use spaCy for lemmatization
            return None  # Placeholder
        except ImportError:
            return None

    def _get_portuguese_stopwords(self) -> Set[str]:
        """Get Portuguese stopwords"""
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words('portuguese'))
        except ImportError:
            return {
                'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com',
                'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos',
                'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua'
            }

    def _initialize_portuguese_pos_tagger(self):
        """Initialize Portuguese POS tagger"""
        # Would use NLTK or spaCy for POS tagging
        return None

    def _initialize_english_stemmer(self):
        """Initialize English stemmer"""
        try:
            from nltk.stem import PorterStemmer
            return PorterStemmer()
        except ImportError:
            return None

    def _initialize_english_lemmatizer(self):
        """Initialize English lemmatizer"""
        try:
            from nltk.stem import WordNetLemmatizer
            return WordNetLemmatizer()
        except ImportError:
            return None

    def _get_english_stopwords(self) -> Set[str]:
        """Get English stopwords"""
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words('english'))
        except ImportError:
            return {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
            }

    def _initialize_english_pos_tagger(self):
        """Initialize English POS tagger"""
        try:
            import nltk
            return nltk.pos_tag
        except ImportError:
            return None

    def _initialize_spanish_stemmer(self):
        """Initialize Spanish stemmer"""
        try:
            from nltk.stem import SnowballStemmer
            return SnowballStemmer('spanish')
        except ImportError:
            return None

    def _initialize_spanish_lemmatizer(self):
        """Initialize Spanish lemmatizer"""
        return None  # Would use spaCy

    def _get_spanish_stopwords(self) -> Set[str]:
        """Get Spanish stopwords"""
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words('spanish'))
        except ImportError:
            return {
                'el', 'la', 'los', 'las', 'de', 'del', 'y', 'o', 'pero', 'en',
                'un', 'una', 'unos', 'unas', 'es', 'son', 'era', 'eran', 'ser',
                'estar', 'haber', 'tener', 'hacer', 'ir', 'ver', 'dar', 'saber'
            }

    def _initialize_spanish_pos_tagger(self):
        """Initialize Spanish POS tagger"""
        return None

    def _initialize_french_stemmer(self):
        """Initialize French stemmer"""
        try:
            from nltk.stem import SnowballStemmer
            return SnowballStemmer('french')
        except ImportError:
            return None

    def _initialize_french_lemmatizer(self):
        """Initialize French lemmatizer"""
        return None

    def _get_french_stopwords(self) -> Set[str]:
        """Get French stopwords"""
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words('french'))
        except ImportError:
            return {
                'le', 'la', 'les', 'de', 'du', 'des', 'et', 'à', 'un', 'une',
                'dans', 'sur', 'avec', 'pour', 'par', 'il', 'elle', 'nous',
                'vous', 'ils', 'elles', 'ce', 'cette', 'ces', 'son', 'sa'
            }

    def _initialize_french_pos_tagger(self):
        """Initialize French POS tagger"""
        return None

    def _initialize_german_stemmer(self):
        """Initialize German stemmer"""
        try:
            from nltk.stem import SnowballStemmer
            return SnowballStemmer('german')
        except ImportError:
            return None

    def _initialize_german_lemmatizer(self):
        """Initialize German lemmatizer"""
        return None

    def _get_german_stopwords(self) -> Set[str]:
        """Get German stopwords"""
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words('german'))
        except ImportError:
            return {
                'der', 'die', 'das', 'den', 'dem', 'des', 'und', 'oder', 'aber',
                'in', 'auf', 'mit', 'für', 'von', 'zu', 'ist', 'sind', 'war',
                'waren', 'sein', 'haben', 'hat', 'hatte', 'ich', 'du', 'er'
            }

    def _initialize_german_pos_tagger(self):
        """Initialize German POS tagger"""
        return None

    def _initialize_italian_stemmer(self):
        """Initialize Italian stemmer"""
        try:
            from nltk.stem import SnowballStemmer
            return SnowballStemmer('italian')
        except ImportError:
            return None

    def _initialize_italian_lemmatizer(self):
        """Initialize Italian lemmatizer"""
        return None

    def _get_italian_stopwords(self) -> Set[str]:
        """Get Italian stopwords"""
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words('italian'))
        except ImportError:
            return {
                'il', 'lo', 'la', 'i', 'gli', 'le', 'di', 'a', 'da', 'in', 'con',
                'su', 'per', 'tra', 'fra', 'e', 'o', 'ma', 'se', 'come', 'che',
                'chi', 'cui', 'quanto', 'quanta', 'quanti', 'quante', 'questo'
            }

    def _initialize_italian_pos_tagger(self):
        """Initialize Italian POS tagger"""
        return None

    def _initialize_financial_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize cross-language financial terminology mappings"""
        return {
            'bank': {
                'pt': ['banco', 'bancos'],
                'en': ['bank', 'banks'],
                'es': ['banco', 'bancos'],
                'fr': ['banque', 'banques'],
                'de': ['bank', 'banken'],
                'it': ['banca', 'banche']
            },
            'account': {
                'pt': ['conta', 'contas'],
                'en': ['account', 'accounts'],
                'es': ['cuenta', 'cuentas'],
                'fr': ['compte', 'comptes'],
                'de': ['konto', 'konten'],
                'it': ['conto', 'conti']
            },
            'transfer': {
                'pt': ['transferência', 'transferências'],
                'en': ['transfer', 'transfers'],
                'es': ['transferencia', 'transferencias'],
                'fr': ['transfert', 'transferts'],
                'de': ['überweisung', 'überweisungen'],
                'it': ['trasferimento', 'trasferimenti']
            },
            'payment': {
                'pt': ['pagamento', 'pagamentos'],
                'en': ['payment', 'payments'],
                'es': ['pago', 'pagos'],
                'fr': ['paiement', 'paiements'],
                'de': ['zahlung', 'zahlungen'],
                'it': ['pagamento', 'pagamenti']
            },
            'credit': {
                'pt': ['crédito', 'créditos'],
                'en': ['credit', 'credits'],
                'es': ['crédito', 'créditos'],
                'fr': ['crédit', 'crédits'],
                'de': ['kredit', 'kredite'],
                'it': ['credito', 'crediti']
            },
            'debit': {
                'pt': ['débito', 'débitos'],
                'en': ['debit', 'debits'],
                'es': ['débito', 'débitos'],
                'fr': ['débit', 'débits'],
                'de': ['debit', 'debite'],
                'it': ['debito', 'debiti']
            },
            'balance': {
                'pt': ['saldo', 'saldos'],
                'en': ['balance', 'balances'],
                'es': ['saldo', 'saldos'],
                'fr': ['solde', 'soldes'],
                'de': ['saldo', 'saldi'],
                'it': ['saldo', 'saldi']
            },
            'transaction': {
                'pt': ['transação', 'transações'],
                'en': ['transaction', 'transactions'],
                'es': ['transacción', 'transacciones'],
                'fr': ['transaction', 'transactions'],
                'de': ['transaktion', 'transaktionen'],
                'it': ['transazione', 'transazioni']
            }
        }

    def _initialize_translation(self):
        """Initialize translation capabilities"""
        try:
            from googletrans import Translator
            self.translator = Translator()
            self.logger.info("Translation capabilities initialized")
        except ImportError:
            self.logger.warning("Translation not available")
            self.translator = None

    def process_text(self, text: str, detected_language: Optional[str] = None) -> ProcessingResult:
        """
        Process text with language-aware analysis

        Args:
            text: Input text to process
            detected_language: Pre-detected language (optional)

        Returns:
            ProcessingResult with comprehensive language-aware analysis
        """
        import time
        start_time = time.time()

        if not text or not isinstance(text, str):
            return self._create_empty_result()

        # Check cache
        cache_key = hash((text, detected_language))
        if self.config['cache_enabled'] and cache_key in self._processing_cache:
            return self._processing_cache[cache_key]

        try:
            result = ProcessingResult(
                original_text=text,
                processed_text=text,
                detected_language='unknown',
                language_confidence=0.0,
                tokens=[],
                lemmas=[],
                stems=[],
                pos_tags=[],
                entities=[],
                filtered_terms=[],
                financial_terms=[],
                translations={},
                quality_metrics={},
                processing_time=0.0,
                success=True
            )

            # Step 1: Language detection
            if detected_language:
                result.detected_language = detected_language
                result.language_confidence = 1.0
            else:
                detection_result = self.language_detector.detect_language(text)
                result.detected_language = detection_result.consensus_language
                result.language_confidence = detection_result.consensus_confidence

            # Step 2: Preprocessing
            preprocessing_result = self.preprocessor.preprocess_text(text, result.detected_language)
            result.processed_text = preprocessing_result.processed_text

            # Step 3: Tokenization and linguistic analysis
            tokens, lemmas, stems, pos_tags = self._perform_linguistic_analysis(
                result.processed_text, result.detected_language
            )
            result.tokens = tokens
            result.lemmas = lemmas
            result.stems = stems
            result.pos_tags = pos_tags

            # Step 4: Named Entity Recognition
            entities = self._perform_ner(result.processed_text, result.detected_language)
            result.entities = entities

            # Step 5: Stopword filtering
            filtered_terms = self._apply_stopword_filtering(
                result.tokens, result.detected_language
            )
            result.filtered_terms = filtered_terms

            # Step 6: Financial term identification
            financial_terms = self._identify_financial_terms(
                result.tokens, result.detected_language
            )
            result.financial_terms = financial_terms

            # Step 7: Translation (if enabled)
            if self._translation_enabled and self.translator:
                translations = self._perform_translation(
                    result.financial_terms, result.detected_language
                )
                result.translations = translations

            # Step 8: Quality assessment
            result.quality_metrics = self._assess_processing_quality(result)

            # Calculate processing time
            result.processing_time = time.time() - start_time

            # Update performance stats
            self._update_performance_stats(result)

            # Cache result
            if self.config['cache_enabled']:
                self._processing_cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Error in language-aware processing: {str(e)}")
            return self._create_error_result(text, str(e))

    def _perform_linguistic_analysis(self, text: str, language: str) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str]]]:
        """Perform linguistic analysis (tokenization, lemmatization, stemming, POS tagging)"""
        try:
            # Get language processor
            processor = self._language_processors.get(language, {})

            # Tokenization
            tokens = self._tokenize_text(text, language)

            # Lemmatization
            lemmas = []
            if processor.get('lemmatizer'):
                lemmas = [processor['lemmatizer'].lemmatize(token) for token in tokens]
            else:
                lemmas = tokens.copy()  # Fallback to original tokens

            # Stemming
            stems = []
            if processor.get('stemmer'):
                stems = [processor['stemmer'].stem(token) for token in tokens]
            else:
                stems = tokens.copy()  # Fallback to original tokens

            # POS tagging
            pos_tags = []
            if processor.get('pos_tagger') and language == 'en':
                pos_tags = processor['pos_tagger'](tokens)
            else:
                # Fallback: assign unknown POS tags
                pos_tags = [(token, 'UNK') for token in tokens]

            return tokens, lemmas, stems, pos_tags

        except Exception as e:
            self.logger.warning(f"Error in linguistic analysis: {str(e)}")
            return [], [], [], []

    def _tokenize_text(self, text: str, language: str) -> List[str]:
        """Tokenize text based on language"""
        try:
            # Use NLTK for tokenization
            if language in ['en', 'pt']:
                from nltk.tokenize import word_tokenize
                return word_tokenize(text)
            else:
                # Simple whitespace tokenization for other languages
                return text.split()

        except ImportError:
            # Fallback tokenization
            return re.findall(r'\b\w+\b', text)

    def _perform_ner(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Perform Named Entity Recognition"""
        try:
            entities = []

            # Use spaCy if available
            if language in ['pt', 'en', 'es', 'fr', 'de', 'it']:
                try:
                    import spacy
                    model_name = f"{language}_core_news_sm"
                    if language == 'en':
                        model_name = "en_core_web_sm"
                    elif language == 'pt':
                        model_name = "pt_core_news_lg"

                    nlp = spacy.load(model_name)
                    doc = nlp(text)

                    for ent in doc.ents:
                        entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': 0.9
                        })

                except (ImportError, OSError):
                    # Fallback: pattern-based NER
                    entities = self._pattern_based_ner(text, language)

            return entities

        except Exception as e:
            self.logger.warning(f"Error in NER: {str(e)}")
            return []

    def _pattern_based_ner(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Pattern-based Named Entity Recognition as fallback"""
        entities = []

        # Financial entity patterns
        patterns = {
            'MONEY': r'\b\d{1,3}(?:\.\d{3})*,\d{2}\b',  # Currency amounts
            'PERCENT': r'\b\d+(?:\.\d+)?%\b',  # Percentages
            'DATE': r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # Dates
            'ACCOUNT': r'\b\d{5,12}\b'  # Account numbers
        }

        for label, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': label,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.7
                })

        return entities

    def _apply_stopword_filtering(self, tokens: List[str], language: str) -> List[str]:
        """Apply language-specific stopword filtering"""
        try:
            processor = self._language_processors.get(language, {})
            stopwords = processor.get('stopwords', set())

            if not stopwords:
                return tokens

            # Filter out stopwords and short tokens
            filtered = []
            for token in tokens:
                token_lower = token.lower()
                if (len(token) >= 2 and
                    token_lower not in stopwords and
                    not token.isdigit()):
                    filtered.append(token)

            return filtered

        except Exception as e:
            self.logger.warning(f"Error in stopword filtering: {str(e)}")
            return tokens

    def _identify_financial_terms(self, tokens: List[str], language: str) -> List[Dict[str, Any]]:
        """Identify financial terms in the text"""
        try:
            financial_terms = []

            # Convert tokens to lowercase for matching
            token_set = set(token.lower() for token in tokens)

            # Check against financial mappings
            for concept, translations in self._financial_mappings.items():
                if language in translations:
                    for term in translations[language]:
                        if term in token_set:
                            financial_terms.append({
                                'term': term,
                                'concept': concept,
                                'language': language,
                                'confidence': 0.9
                            })

            # Additional financial terms specific to each language
            additional_terms = self._get_additional_financial_terms(language)
            for term in additional_terms:
                if term in token_set:
                    financial_terms.append({
                        'term': term,
                        'concept': 'financial',
                        'language': language,
                        'confidence': 0.8
                    })

            return financial_terms

        except Exception as e:
            self.logger.warning(f"Error identifying financial terms: {str(e)}")
            return []

    def _get_additional_financial_terms(self, language: str) -> List[str]:
        """Get additional financial terms for specific languages"""
        terms = {
            'pt': [
                'banco', 'conta', 'valor', 'pagamento', 'transferência', 'crédito',
                'débito', 'saldo', 'agência', 'número', 'cpf', 'cnpj', 'pix',
                'boleto', 'fatura', 'real', 'reais', 'r$', 'centavo', 'moeda'
            ],
            'en': [
                'bank', 'account', 'value', 'payment', 'transfer', 'credit',
                'debit', 'balance', 'agency', 'number', 'tax', 'id', 'invoice',
                'bill', 'currency', 'dollar', 'financial', 'transaction'
            ],
            'es': [
                'banco', 'cuenta', 'valor', 'pago', 'transferencia', 'crédito',
                'débito', 'saldo', 'agencia', 'número', 'dni', 'cif', 'factura',
                'moneda', 'dinero', 'financiero', 'euro', 'pesos'
            ],
            'fr': [
                'banque', 'compte', 'valeur', 'paiement', 'transfert', 'crédit',
                'débit', 'solde', 'agence', 'numéro', 'siret', 'facture',
                'monnaie', 'argent', 'financier', 'euro'
            ],
            'de': [
                'bank', 'konto', 'wert', 'zahlung', 'überweisung', 'kredit',
                'debit', 'saldo', 'filiale', 'nummer', 'steuer', 'rechnung',
                'währung', 'geld', 'finanziell', 'euro'
            ],
            'it': [
                'banca', 'conto', 'valore', 'pagamento', 'trasferimento', 'credito',
                'debito', 'saldo', 'agenzia', 'numero', 'iva', 'fattura',
                'moneta', 'denaro', 'finanziario', 'euro'
            ]
        }

        return terms.get(language, [])

    def _perform_translation(self, financial_terms: List[Dict[str, Any]], source_language: str) -> Dict[str, List[str]]:
        """Perform translation of financial terms"""
        try:
            translations = {}

            if not self.translator:
                return translations

            # Translate to English if not already in English
            target_languages = ['en'] if source_language != 'en' else ['pt', 'es', 'fr', 'de', 'it']

            for term_info in financial_terms:
                term = term_info['term']
                translations[term] = []

                for target_lang in target_languages:
                    try:
                        translation = self.translator.translate(term, src=source_language, dest=target_lang)
                        if translation and translation.text:
                            translations[term].append(f"{target_lang}:{translation.text}")
                    except Exception as e:
                        self.logger.warning(f"Translation error for {term}: {str(e)}")

            return translations

        except Exception as e:
            self.logger.warning(f"Error in translation: {str(e)}")
            return {}

    def _assess_processing_quality(self, result: ProcessingResult) -> Dict[str, float]:
        """Assess the quality of language-aware processing"""
        try:
            metrics = {
                'token_count': len(result.tokens),
                'filtered_token_count': len(result.filtered_terms),
                'entity_count': len(result.entities),
                'financial_term_count': len(result.financial_terms),
                'language_confidence': result.language_confidence,
                'lemmatization_ratio': len(result.lemmas) / len(result.tokens) if result.tokens else 0.0,
                'stemming_ratio': len(result.stems) / len(result.tokens) if result.tokens else 0.0
            }

            # Calculate overall quality score
            base_score = 0.5
            confidence_bonus = result.language_confidence * 0.2
            entity_bonus = min(metrics['entity_count'] * 0.05, 0.15)
            financial_bonus = min(metrics['financial_term_count'] * 0.1, 0.15)

            metrics['overall_quality'] = min(base_score + confidence_bonus + entity_bonus + financial_bonus, 1.0)

            return metrics

        except Exception as e:
            self.logger.warning(f"Error assessing processing quality: {str(e)}")
            return {'overall_quality': 0.5}

    def _update_performance_stats(self, result: ProcessingResult):
        """Update performance statistics"""
        self._performance_stats['total_processed'] += 1

        # Update language distribution
        lang = result.detected_language
        if lang not in self._performance_stats['language_distribution']:
            self._performance_stats['language_distribution'][lang] = 0
        self._performance_stats['language_distribution'][lang] += 1

        # Update average processing time
        current_avg = self._performance_stats['average_processing_time']
        self._performance_stats['average_processing_time'] = (
            (current_avg * (self._performance_stats['total_processed'] - 1)) +
            result.processing_time
        ) / self._performance_stats['total_processed']

    def process_batch(self, texts: List[str], languages: Optional[List[str]] = None) -> List[ProcessingResult]:
        """
        Process a batch of texts with language-aware analysis

        Args:
            texts: List of texts to process
            languages: Optional list of pre-detected languages

        Returns:
            List of ProcessingResult objects
        """
        if not texts:
            return []

        try:
            results = []

            for i, text in enumerate(texts):
                detected_lang = languages[i] if languages and i < len(languages) else None
                result = self.process_text(text, detected_lang)
                results.append(result)

            self.logger.info(f"Processed {len(texts)} texts with language-aware analysis")
            return results

        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            return [self._create_error_result(text, str(e)) for text in texts]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self._performance_stats.copy()

    def clear_cache(self):
        """Clear processing cache"""
        self._processing_cache.clear()
        self.logger.info("Language-aware processing cache cleared")

    def _create_empty_result(self) -> ProcessingResult:
        """Create empty processing result"""
        return ProcessingResult(
            original_text='',
            processed_text='',
            detected_language='unknown',
            language_confidence=0.0,
            tokens=[],
            lemmas=[],
            stems=[],
            pos_tags=[],
            entities=[],
            filtered_terms=[],
            financial_terms=[],
            translations={},
            quality_metrics={'overall_quality': 0.0},
            processing_time=0.0,
            success=False,
            error_message='Empty or invalid input'
        )

    def _create_error_result(self, text: str, error: str) -> ProcessingResult:
        """Create error processing result"""
        return ProcessingResult(
            original_text=text,
            processed_text=text,
            detected_language='unknown',
            language_confidence=0.0,
            tokens=[],
            lemmas=[],
            stems=[],
            pos_tags=[],
            entities=[],
            filtered_terms=[],
            financial_terms=[],
            translations={},
            quality_metrics={'overall_quality': 0.0},
            processing_time=0.0,
            success=False,
            error_message=error
        )