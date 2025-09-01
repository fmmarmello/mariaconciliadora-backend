import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import logging

from src.utils.logging_config import get_logger
from src.services.language_detector import LanguageDetector, DetectionResult

logger = get_logger(__name__)


@dataclass
class LanguageConfig:
    """Configuration for language-specific processing"""
    language_code: str
    currency_symbols: List[str]
    date_formats: List[str]
    number_formats: Dict[str, str]
    encoding_preferences: List[str]
    stopwords: Set[str]
    stemmer_available: bool
    lemmatizer_available: bool


@dataclass
class PreprocessingResult:
    """Result of multi-language preprocessing"""
    original_text: str
    processed_text: str
    detected_language: str
    language_confidence: float
    normalization_steps: List[str]
    extracted_entities: Dict[str, List[Dict[str, Any]]]
    quality_metrics: Dict[str, float]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class MultiLanguagePreprocessor:
    """
    Multi-language text preprocessor with language-specific normalization
    and entity extraction capabilities
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-language preprocessor

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize language detector
        self.language_detector = LanguageDetector()

        # Initialize language configurations
        self._language_configs = self._initialize_language_configs()

        # Initialize pattern libraries
        self._patterns = self._initialize_patterns()

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
            'normalize_currency': True,
            'normalize_dates': True,
            'normalize_numbers': True,
            'handle_encoding': True,
            'extract_entities': True,
            'confidence_threshold': 0.6,
            'batch_size': 50,
            'timeout_seconds': 30.0
        }

    def _initialize_language_configs(self) -> Dict[str, LanguageConfig]:
        """Initialize configurations for supported languages"""
        configs = {}

        # Portuguese (Brazil)
        configs['pt'] = LanguageConfig(
            language_code='pt',
            currency_symbols=['R$', 'BRL', 'reais', 'real'],
            date_formats=[
                r'(\d{1,2})/(\d{1,2})/(\d{4})',  # DD/MM/YYYY
                r'(\d{1,2})/(\d{1,2})/(\d{2})',   # DD/MM/YY
                r'(\d{4})-(\d{1,2})-(\d{1,2})',   # YYYY-MM-DD
                r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})'  # DD de Janeiro de YYYY
            ],
            number_formats={
                'decimal_separator': ',',
                'thousands_separator': '.',
                'currency_position': 'before'  # R$ 1.234,56
            },
            encoding_preferences=['utf-8', 'latin1', 'cp1252'],
            stopwords={
                'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com',
                'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos',
                'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua'
            },
            stemmer_available=True,
            lemmatizer_available=True
        )

        # English
        configs['en'] = LanguageConfig(
            language_code='en',
            currency_symbols=['$', 'USD', 'dollars', 'dollar'],
            date_formats=[
                r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
                r'(\d{4})-(\d{1,2})-(\d{1,2})',   # YYYY-MM-DD
                r'(\d{1,2})\s+(\w+)\s+(\d{4})',   # DD Month YYYY
                r'(\w+)\s+(\d{1,2}),?\s+(\d{4})'  # Month DD, YYYY
            ],
            number_formats={
                'decimal_separator': '.',
                'thousands_separator': ',',
                'currency_position': 'before'  # $1,234.56
            },
            encoding_preferences=['utf-8', 'ascii', 'latin1'],
            stopwords={
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
            },
            stemmer_available=True,
            lemmatizer_available=True
        )

        # Spanish
        configs['es'] = LanguageConfig(
            language_code='es',
            currency_symbols=['€', 'EUR', 'euros', 'euro', '$', 'USD'],
            date_formats=[
                r'(\d{1,2})/(\d{1,2})/(\d{4})',  # DD/MM/YYYY
                r'(\d{4})-(\d{1,2})-(\d{1,2})',   # YYYY-MM-DD
                r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})'  # DD de enero de YYYY
            ],
            number_formats={
                'decimal_separator': ',',
                'thousands_separator': '.',
                'currency_position': 'after'  # 1.234,56 €
            },
            encoding_preferences=['utf-8', 'latin1', 'iso-8859-1'],
            stopwords={
                'el', 'la', 'los', 'las', 'de', 'del', 'y', 'o', 'pero', 'en',
                'un', 'una', 'unos', 'unas', 'es', 'son', 'era', 'eran', 'ser',
                'estar', 'haber', 'tener', 'hacer', 'ir', 'ver', 'dar', 'saber'
            },
            stemmer_available=True,
            lemmatizer_available=True
        )

        # French
        configs['fr'] = LanguageConfig(
            language_code='fr',
            currency_symbols=['€', 'EUR', 'euros', 'euro'],
            date_formats=[
                r'(\d{1,2})/(\d{1,2})/(\d{4})',  # DD/MM/YYYY
                r'(\d{4})-(\d{1,2})-(\d{1,2})',   # YYYY-MM-DD
                r'(\d{1,2})\s+(\w+)\s+(\d{4})'    # DD janvier YYYY
            ],
            number_formats={
                'decimal_separator': ',',
                'thousands_separator': ' ',
                'currency_position': 'after'  # 1 234,56 €
            },
            encoding_preferences=['utf-8', 'latin1', 'iso-8859-1'],
            stopwords={
                'le', 'la', 'les', 'de', 'du', 'des', 'et', 'à', 'un', 'une',
                'dans', 'sur', 'avec', 'pour', 'par', 'il', 'elle', 'nous',
                'vous', 'ils', 'elles', 'ce', 'cette', 'ces', 'son', 'sa'
            },
            stemmer_available=True,
            lemmatizer_available=True
        )

        # German
        configs['de'] = LanguageConfig(
            language_code='de',
            currency_symbols=['€', 'EUR', 'euro', 'euros'],
            date_formats=[
                r'(\d{1,2})\.(\d{1,2})\.(\d{4})',  # DD.MM.YYYY
                r'(\d{4})-(\d{1,2})-(\d{1,2})',     # YYYY-MM-DD
                r'(\d{1,2})\.\s+(\w+)\s+(\d{4})'    # DD. Januar YYYY
            ],
            number_formats={
                'decimal_separator': ',',
                'thousands_separator': '.',
                'currency_position': 'before'  # 1.234,56 €
            },
            encoding_preferences=['utf-8', 'latin1', 'iso-8859-1'],
            stopwords={
                'der', 'die', 'das', 'den', 'dem', 'des', 'und', 'oder', 'aber',
                'in', 'auf', 'mit', 'für', 'von', 'zu', 'ist', 'sind', 'war',
                'waren', 'sein', 'haben', 'hat', 'hatte', 'ich', 'du', 'er'
            },
            stemmer_available=True,
            lemmatizer_available=True
        )

        # Italian
        configs['it'] = LanguageConfig(
            language_code='it',
            currency_symbols=['€', 'EUR', 'euro', 'euros'],
            date_formats=[
                r'(\d{1,2})/(\d{1,2})/(\d{4})',  # DD/MM/YYYY
                r'(\d{4})-(\d{1,2})-(\d{1,2})',   # YYYY-MM-DD
                r'(\d{1,2})\s+(\w+)\s+(\d{4})'    # DD gennaio YYYY
            ],
            number_formats={
                'decimal_separator': ',',
                'thousands_separator': '.',
                'currency_position': 'after'  # 1.234,56 €
            },
            encoding_preferences=['utf-8', 'latin1', 'iso-8859-1'],
            stopwords={
                'il', 'lo', 'la', 'i', 'gli', 'le', 'di', 'a', 'da', 'in', 'con',
                'su', 'per', 'tra', 'fra', 'e', 'o', 'ma', 'se', 'come', 'che',
                'chi', 'cui', 'quanto', 'quanta', 'quanti', 'quante', 'questo'
            },
            stemmer_available=True,
            lemmatizer_available=True
        )

        return configs

    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize regex patterns for entity extraction"""
        return {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?[\d\s\-\(\)]{10,})',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))*)?',
            'account_number': r'\b\d{5,12}\b',
            'agency_number': r'\b\d{4}\b',
            'tax_id': r'\b\d{2,3}\.\d{3}\.\d{3}/\d{4}-\d{2}\b',
            'cpf': r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b',
            'cnpj': r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b'
        }

    def preprocess_text(self, text: str, detected_language: Optional[str] = None) -> PreprocessingResult:
        """
        Preprocess text with multi-language support

        Args:
            text: Input text to preprocess
            detected_language: Pre-detected language (optional)

        Returns:
            PreprocessingResult with processed text and metadata
        """
        import time
        start_time = time.time()

        if not text or not isinstance(text, str):
            return self._create_empty_result()

        # Check cache
        cache_key = hash((text, detected_language))
        if self.config['cache_enabled'] and cache_key in self._processing_cache:
            cached_result = self._processing_cache[cache_key]
            self._performance_stats['cache_hit_rate'] = (
                self._performance_stats['cache_hit_rate'] + 0.1
            ) % 1.0
            return cached_result

        try:
            result = PreprocessingResult(
                original_text=text,
                processed_text=text,
                detected_language='unknown',
                language_confidence=0.0,
                normalization_steps=[],
                extracted_entities={},
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

            result.normalization_steps.append('language_detection')

            # Step 2: Get language configuration
            lang_config = self._language_configs.get(result.detected_language)
            if not lang_config:
                # Fallback to Portuguese for unknown languages
                lang_config = self._language_configs.get('pt')
                result.normalization_steps.append('fallback_to_portuguese')

            # Step 3: Character encoding detection and conversion
            if self.config['handle_encoding']:
                processed_text = self._handle_encoding(text, lang_config)
                result.processed_text = processed_text
                result.normalization_steps.append('encoding_handling')

            # Step 4: Text normalization
            normalized_text = self._normalize_text(result.processed_text, lang_config)
            result.processed_text = normalized_text
            result.normalization_steps.append('text_normalization')

            # Step 5: Currency symbol recognition and normalization
            if self.config['normalize_currency']:
                currency_result = self._normalize_currency(result.processed_text, lang_config)
                result.processed_text = currency_result['text']
                result.extracted_entities['currency'] = currency_result['entities']
                result.normalization_steps.append('currency_normalization')

            # Step 6: Date format handling
            if self.config['normalize_dates']:
                date_result = self._normalize_dates(result.processed_text, lang_config)
                result.processed_text = date_result['text']
                result.extracted_entities['dates'] = date_result['entities']
                result.normalization_steps.append('date_normalization')

            # Step 7: Number format parsing
            if self.config['normalize_numbers']:
                number_result = self._normalize_numbers(result.processed_text, lang_config)
                result.processed_text = number_result['text']
                result.extracted_entities['numbers'] = number_result['entities']
                result.normalization_steps.append('number_normalization')

            # Step 8: Entity extraction
            if self.config['extract_entities']:
                entities = self._extract_entities(result.processed_text, lang_config)
                result.extracted_entities.update(entities)
                result.normalization_steps.append('entity_extraction')

            # Step 9: Quality assessment
            result.quality_metrics = self._assess_quality(result)

            # Step 10: Final text reconstruction
            result.processed_text = self._reconstruct_text(result)

            # Calculate processing time
            result.processing_time = time.time() - start_time

            # Update performance stats
            self._update_performance_stats(result)

            # Cache result
            if self.config['cache_enabled']:
                self._processing_cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Error in multi-language preprocessing: {str(e)}")
            return self._create_error_result(text, str(e))

    def _handle_encoding(self, text: str, lang_config: LanguageConfig) -> str:
        """Handle character encoding detection and conversion"""
        try:
            # Try to detect encoding issues
            for encoding in lang_config.encoding_preferences:
                try:
                    # Test if text can be properly decoded/encoded
                    test_bytes = text.encode(encoding, errors='ignore')
                    decoded_text = test_bytes.decode(encoding, errors='ignore')

                    if len(decoded_text) > len(text) * 0.8:  # If we recovered most characters
                        return decoded_text
                except (UnicodeDecodeError, UnicodeEncodeError, LookupError):
                    continue

            # If no encoding worked well, try to fix common issues
            return self._fix_common_encoding_issues(text, lang_config.language_code)

        except Exception as e:
            self.logger.warning(f"Error handling encoding: {str(e)}")
            return text

    def _fix_common_encoding_issues(self, text: str, language: str) -> str:
        """Fix common encoding issues for specific languages"""
        try:
            # Language-specific encoding fixes
            if language == 'pt':
                # Common Portuguese encoding issues
                fixes = {
                    'Ã£': 'ã', 'Ã¢': 'â', 'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í',
                    'Ã³': 'ó', 'Ãº': 'ú', 'Ã§': 'ç', 'Ãµ': 'õ', 'Ãª': 'ê',
                    'Ã ': 'à', 'Ãš': 'Ú', 'Ã‰': 'É', 'Ã“': 'Ó', 'Ã': 'Á'
                }
            elif language == 'es':
                # Common Spanish encoding issues
                fixes = {
                    'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
                    'Ã±': 'ñ', 'Ã ': 'à', 'Ãš': 'Ú', 'Ã‰': 'É', 'Ã“': 'Ó'
                }
            elif language == 'fr':
                # Common French encoding issues
                fixes = {
                    'Ã©': 'é', 'Ã¨': 'è', 'Ã¢': 'â', 'Ã®': 'î', 'Ã´': 'ô',
                    'Ã»': 'û', 'Ã§': 'ç', 'Ã ': 'à', 'Ãš': 'Ú', 'Ã‰': 'É'
                }
            else:
                fixes = {}

            for wrong, correct in fixes.items():
                text = text.replace(wrong, correct)

            return text

        except Exception:
            return text

    def _normalize_text(self, text: str, lang_config: LanguageConfig) -> str:
        """Language-specific text normalization"""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)

            # Remove control characters
            text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

            # Language-specific normalization
            if lang_config.language_code == 'pt':
                # Portuguese-specific normalization
                text = self._normalize_portuguese_text(text)
            elif lang_config.language_code == 'es':
                # Spanish-specific normalization
                text = self._normalize_spanish_text(text)
            elif lang_config.language_code == 'fr':
                # French-specific normalization
                text = self._normalize_french_text(text)
            elif lang_config.language_code == 'de':
                # German-specific normalization
                text = self._normalize_german_text(text)
            elif lang_config.language_code == 'it':
                # Italian-specific normalization
                text = self._normalize_italian_text(text)

            # General normalization
            text = text.strip()

            return text

        except Exception as e:
            self.logger.warning(f"Error in text normalization: {str(e)}")
            return text

    def _normalize_portuguese_text(self, text: str) -> str:
        """Portuguese-specific text normalization"""
        # Normalize accents
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')

        # Fix common abbreviations
        abbreviations = {
            'sr ': 'senhor ',
            'sra ': 'senhora ',
            'dr ': 'doutor ',
            'dra ': 'doutora ',
            'av ': 'avenida ',
            'r ': 'rua '
        }

        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)

        return text

    def _normalize_spanish_text(self, text: str) -> str:
        """Spanish-specific text normalization"""
        # Normalize accents
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')

        # Fix common abbreviations
        abbreviations = {
            'sr ': 'señor ',
            'sra ': 'señora ',
            'dr ': 'doctor ',
            'dra ': 'doctora ',
            'av ': 'avenida ',
            'c/ ': 'calle '
        }

        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)

        return text

    def _normalize_french_text(self, text: str) -> str:
        """French-specific text normalization"""
        # Normalize accents
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')

        # Fix common abbreviations
        abbreviations = {
            'm ': 'monsieur ',
            'mme ': 'madame ',
            'dr ': 'docteur ',
            'av ': 'avenue ',
            'rue ': 'rue '
        }

        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)

        return text

    def _normalize_german_text(self, text: str) -> str:
        """German-specific text normalization"""
        # Normalize umlauts and sharp s
        text = text.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
        text = text.replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue')

        # Fix common abbreviations
        abbreviations = {
            'hr ': 'herr ',
            'fr ': 'frau ',
            'dr ': 'doktor ',
            'str ': 'strasse '
        }

        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)

        return text

    def _normalize_italian_text(self, text: str) -> str:
        """Italian-specific text normalization"""
        # Normalize accents
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')

        # Fix common abbreviations
        abbreviations = {
            'sig ': 'signore ',
            'sig.ra ': 'signora ',
            'dr ': 'dottore ',
            'dra ': 'dottoressa ',
            'via ': 'via '
        }

        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)

        return text

    def _normalize_currency(self, text: str, lang_config: LanguageConfig) -> Dict[str, Any]:
        """Normalize currency symbols and extract currency information"""
        try:
            entities = []
            processed_text = text

            # Find currency mentions
            for symbol in lang_config.currency_symbols:
                # Create pattern for currency symbol
                if lang_config.number_formats['currency_position'] == 'before':
                    pattern = r'\b' + re.escape(symbol) + r'\s*([\d\.,]+)'
                else:
                    pattern = r'([\d\.,]+)\s*' + re.escape(symbol) + r'\b'

                matches = re.findall(pattern, processed_text, re.IGNORECASE)
                for match in matches:
                    # Standardize the amount
                    standardized_amount = self._standardize_amount(match, lang_config)

                    if standardized_amount:
                        entities.append({
                            'original': f"{symbol} {match}" if lang_config.number_formats['currency_position'] == 'before' else f"{match} {symbol}",
                            'amount': standardized_amount,
                            'currency': self._get_currency_code(symbol),
                            'confidence': 0.9
                        })

                        # Replace in text with standardized format
                        replacement = f"{self._get_currency_code(symbol)} {standardized_amount}"
                        if lang_config.number_formats['currency_position'] == 'before':
                            processed_text = re.sub(
                                r'\b' + re.escape(symbol) + r'\s*' + re.escape(match) + r'\b',
                                replacement,
                                processed_text,
                                flags=re.IGNORECASE
                            )
                        else:
                            processed_text = re.sub(
                                r'\b' + re.escape(match) + r'\s*' + re.escape(symbol) + r'\b',
                                replacement,
                                processed_text,
                                flags=re.IGNORECASE
                            )

            return {
                'text': processed_text,
                'entities': entities
            }

        except Exception as e:
            self.logger.warning(f"Error normalizing currency: {str(e)}")
            return {'text': text, 'entities': []}

    def _standardize_amount(self, amount: str, lang_config: LanguageConfig) -> Optional[str]:
        """Standardize amount format"""
        try:
            # Remove separators
            decimal_sep = lang_config.number_formats['decimal_separator']
            thousands_sep = lang_config.number_formats['thousands_separator']

            # Handle thousands separator
            if thousands_sep and thousands_sep in amount:
                amount = amount.replace(thousands_sep, '')

            # Handle decimal separator
            if decimal_sep != '.':
                amount = amount.replace(decimal_sep, '.')

            # Convert to float to validate
            float(amount)

            return amount

        except (ValueError, AttributeError):
            return None

    def _get_currency_code(self, symbol: str) -> str:
        """Get ISO currency code from symbol"""
        currency_map = {
            'R$': 'BRL', '$': 'USD', '€': 'EUR',
            'reais': 'BRL', 'real': 'BRL',
            'dollars': 'USD', 'dollar': 'USD',
            'euros': 'EUR', 'euro': 'EUR'
        }
        return currency_map.get(symbol.lower(), symbol.upper())

    def _normalize_dates(self, text: str, lang_config: LanguageConfig) -> Dict[str, Any]:
        """Normalize date formats and extract date information"""
        try:
            entities = []
            processed_text = text

            for pattern in lang_config.date_formats:
                matches = re.findall(pattern, processed_text, re.IGNORECASE)
                for match in matches:
                    parsed_date = self._parse_date(match, lang_config.language_code)

                    if parsed_date:
                        entities.append({
                            'original': '/'.join(match) if isinstance(match, tuple) else match,
                            'parsed': parsed_date.isoformat(),
                            'confidence': 0.85
                        })

                        # Replace with ISO format
                        replacement = parsed_date.strftime('%Y-%m-%d')
                        if isinstance(match, tuple):
                            # Replace the original format
                            original_pattern = '/'.join(match)
                            processed_text = processed_text.replace(original_pattern, replacement)
                        else:
                            processed_text = processed_text.replace(match, replacement)

            return {
                'text': processed_text,
                'entities': entities
            }

        except Exception as e:
            self.logger.warning(f"Error normalizing dates: {str(e)}")
            return {'text': text, 'entities': []}

    def _parse_date(self, date_match: Tuple[str, ...], language: str) -> Optional[datetime]:
        """Parse date from regex match"""
        try:
            if len(date_match) == 3:
                part1, part2, part3 = date_match

                # Determine date format based on language and pattern
                if language == 'en':
                    # Assume MM/DD/YYYY for English
                    if len(part3) == 4:  # YYYY
                        month, day, year = int(part1), int(part2), int(part3)
                    else:  # Assume DD/MM/YYYY
                        day, month, year = int(part1), int(part2), int(part3)
                else:
                    # Assume DD/MM/YYYY for other languages
                    day, month, year = int(part1), int(part2), int(part3)

                # Handle 2-digit years
                if year < 100:
                    year = 2000 + year if year < 50 else 1900 + year

                return datetime(year, month, day).date()

            return None

        except (ValueError, IndexError):
            return None

    def _normalize_numbers(self, text: str, lang_config: LanguageConfig) -> Dict[str, Any]:
        """Normalize number formats"""
        try:
            entities = []
            processed_text = text

            # Pattern for numbers with language-specific formatting
            decimal_sep = lang_config.number_formats['decimal_separator']
            thousands_sep = lang_config.number_formats['thousands_separator']

            # Create pattern for numbers
            if thousands_sep:
                number_pattern = r'\b\d{1,3}(?:' + re.escape(thousands_sep) + r'\d{3})*(?:' + re.escape(decimal_sep) + r'\d+)?\b'
            else:
                number_pattern = r'\b\d+(?:' + re.escape(decimal_sep) + r'\d+)?\b'

            matches = re.findall(number_pattern, processed_text)
            for match in matches:
                # Standardize the number
                standardized = self._standardize_number(match, lang_config)

                if standardized:
                    entities.append({
                        'original': match,
                        'standardized': standardized,
                        'confidence': 0.95
                    })

                    # Replace in text
                    processed_text = processed_text.replace(match, standardized)

            return {
                'text': processed_text,
                'entities': entities
            }

        except Exception as e:
            self.logger.warning(f"Error normalizing numbers: {str(e)}")
            return {'text': text, 'entities': []}

    def _standardize_number(self, number: str, lang_config: LanguageConfig) -> Optional[str]:
        """Standardize number format to English convention"""
        try:
            # Remove separators
            decimal_sep = lang_config.number_formats['decimal_separator']
            thousands_sep = lang_config.number_formats['thousands_separator']

            # Handle thousands separator
            if thousands_sep and thousands_sep in number:
                number = number.replace(thousands_sep, '')

            # Handle decimal separator
            if decimal_sep != '.':
                number = number.replace(decimal_sep, '.')

            # Validate as number
            float(number)

            return number

        except (ValueError, AttributeError):
            return None

    def _extract_entities(self, text: str, lang_config: LanguageConfig) -> Dict[str, List[Dict[str, Any]]]:
        """Extract various entities from text"""
        try:
            entities = {}

            # Extract emails
            emails = re.findall(self._patterns['email'], text)
            if emails:
                entities['emails'] = [{'value': email, 'confidence': 0.95} for email in emails]

            # Extract phones
            phones = re.findall(self._patterns['phone'], text)
            if phones:
                entities['phones'] = [{'value': phone.strip(), 'confidence': 0.9} for phone in phones]

            # Extract URLs
            urls = re.findall(self._patterns['url'], text)
            if urls:
                entities['urls'] = [{'value': url, 'confidence': 0.95} for url in urls]

            # Extract account numbers (Brazilian context)
            if lang_config.language_code == 'pt':
                accounts = re.findall(self._patterns['account_number'], text)
                if accounts:
                    entities['account_numbers'] = [
                        {'value': acc, 'confidence': 0.8} for acc in accounts
                        if 5 <= len(acc) <= 12  # Valid account number length
                    ]

                # Extract CPF/CNPJ
                cpfs = re.findall(self._patterns['cpf'], text)
                if cpfs:
                    entities['cpf'] = [{'value': cpf, 'confidence': 0.9} for cpf in cpfs]

                cnpjs = re.findall(self._patterns['cnpj'], text)
                if cnpjs:
                    entities['cnpj'] = [{'value': cnpj, 'confidence': 0.9} for cnpj in cnpjs]

            return entities

        except Exception as e:
            self.logger.warning(f"Error extracting entities: {str(e)}")
            return {}

    def _assess_quality(self, result: PreprocessingResult) -> Dict[str, float]:
        """Assess the quality of preprocessing results"""
        try:
            metrics = {
                'text_length_ratio': len(result.processed_text) / len(result.original_text) if result.original_text else 1.0,
                'entity_count': sum(len(entities) for entities in result.extracted_entities.values()),
                'language_confidence': result.language_confidence,
                'processing_steps_count': len(result.normalization_steps)
            }

            # Calculate overall quality score
            base_score = 0.5
            confidence_bonus = result.language_confidence * 0.3
            entity_bonus = min(metrics['entity_count'] * 0.05, 0.2)  # Max 0.2 for entities

            metrics['overall_quality'] = min(base_score + confidence_bonus + entity_bonus, 1.0)

            return metrics

        except Exception as e:
            self.logger.warning(f"Error assessing quality: {str(e)}")
            return {'overall_quality': 0.5}

    def _reconstruct_text(self, result: PreprocessingResult) -> str:
        """Reconstruct final processed text"""
        try:
            # Basic cleaning
            text = result.processed_text
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

            return text

        except Exception as e:
            self.logger.warning(f"Error reconstructing text: {str(e)}")
            return result.processed_text

    def _update_performance_stats(self, result: PreprocessingResult):
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

    def preprocess_batch(self, texts: List[str], languages: Optional[List[str]] = None) -> List[PreprocessingResult]:
        """
        Preprocess a batch of texts

        Args:
            texts: List of texts to preprocess
            languages: Optional list of pre-detected languages

        Returns:
            List of PreprocessingResult objects
        """
        if not texts:
            return []

        try:
            results = []

            for i, text in enumerate(texts):
                detected_lang = languages[i] if languages and i < len(languages) else None
                result = self.preprocess_text(text, detected_lang)
                results.append(result)

            self.logger.info(f"Processed {len(texts)} texts with multi-language preprocessing")
            return results

        except Exception as e:
            self.logger.error(f"Error in batch preprocessing: {str(e)}")
            return [self._create_error_result(text, str(e)) for text in texts]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self._performance_stats.copy()

    def clear_cache(self):
        """Clear processing cache"""
        self._processing_cache.clear()
        self.logger.info("Multi-language preprocessing cache cleared")

    def _create_empty_result(self) -> PreprocessingResult:
        """Create empty preprocessing result"""
        return PreprocessingResult(
            original_text='',
            processed_text='',
            detected_language='unknown',
            language_confidence=0.0,
            normalization_steps=[],
            extracted_entities={},
            quality_metrics={'overall_quality': 0.0},
            processing_time=0.0,
            success=False,
            error_message='Empty or invalid input'
        )

    def _create_error_result(self, text: str, error: str) -> PreprocessingResult:
        """Create error preprocessing result"""
        return PreprocessingResult(
            original_text=text,
            processed_text=text,
            detected_language='unknown',
            language_confidence=0.0,
            normalization_steps=[],
            extracted_entities={},
            quality_metrics={'overall_quality': 0.0},
            processing_time=0.0,
            success=False,
            error_message=error
        )