import os
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import hashlib

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DetectionMethod(Enum):
    """Available language detection methods"""
    FASTTEXT = "fasttext"
    LANGDETECT = "langdetect"
    LANGID = "langid"
    POLYGLOT = "polyglot"
    ENSEMBLE = "ensemble"


@dataclass
class LanguageDetection:
    """Language detection result"""
    language: str
    confidence: float
    method: str
    alternatives: List[Tuple[str, float]]
    processing_time: float
    text_sample: str
    text_length: int


@dataclass
class DetectionResult:
    """Complete detection result with metadata"""
    primary_detection: LanguageDetection
    all_detections: List[LanguageDetection]
    consensus_language: str
    consensus_confidence: float
    detection_methods: List[str]
    fallback_used: bool
    error_message: Optional[str] = None


class LanguageDetector:
    """
    Advanced language detection service with multiple detection methods
    and financial domain specialization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the language detector

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize detection methods
        self._detection_methods = {}
        self._initialize_detection_methods()

        # Financial language patterns for domain-specific detection
        self._financial_patterns = self._initialize_financial_patterns()

        # Language confidence thresholds
        self._confidence_thresholds = self._initialize_confidence_thresholds()

        # Cache for detection results
        self._detection_cache = {}

        # Performance tracking
        self._performance_stats = {
            'total_detections': 0,
            'cache_hits': 0,
            'method_usage': {},
            'average_confidence': 0.0,
            'error_count': 0
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'primary_method': DetectionMethod.FASTTEXT.value,
            'fallback_methods': [DetectionMethod.LANGDETECT.value, DetectionMethod.LANGID.value],
            'ensemble_voting': True,
            'confidence_threshold': 0.7,
            'cache_enabled': True,
            'financial_domain_boost': True,
            'min_text_length': 10,
            'max_text_length': 10000,
            'timeout_seconds': 5.0,
            'batch_size': 100
        }

    def _initialize_detection_methods(self):
        """Initialize available detection methods"""
        try:
            # FastText - Primary method
            if self.config.get('use_fasttext', True):
                try:
                    import fasttext
                    # Try to load pre-trained model
                    model_path = self._get_fasttext_model_path()
                    if os.path.exists(model_path):
                        self._detection_methods[DetectionMethod.FASTTEXT] = fasttext.load_model(model_path)
                        self.logger.info("FastText model loaded successfully")
                    else:
                        self.logger.warning("FastText model not found, FastText detection disabled")
                except ImportError:
                    self.logger.warning("FastText not available, falling back to other methods")
                except Exception as e:
                    self.logger.warning(f"Error loading FastText model: {str(e)}")

            # LangDetect
            if self.config.get('use_langdetect', True):
                try:
                    from langdetect import detect, detect_langs, LangDetectError
                    self._detection_methods[DetectionMethod.LANGDETECT] = {
                        'detect': detect,
                        'detect_langs': detect_langs,
                        'error': LangDetectError
                    }
                    self.logger.info("LangDetect initialized successfully")
                except ImportError:
                    self.logger.warning("LangDetect not available")

            # LangID
            if self.config.get('use_langid', True):
                try:
                    import langid
                    self._detection_methods[DetectionMethod.LANGID] = langid
                    self.logger.info("LangID initialized successfully")
                except ImportError:
                    self.logger.warning("LangID not available")

            # Polyglot (limited to supported languages)
            if self.config.get('use_polyglot', False):
                try:
                    from polyglot.detect import Detector
                    self._detection_methods[DetectionMethod.POLYGLOT] = Detector
                    self.logger.info("Polyglot initialized successfully")
                except ImportError:
                    self.logger.warning("Polyglot not available")

        except Exception as e:
            self.logger.error(f"Error initializing detection methods: {str(e)}")

    def _get_fasttext_model_path(self) -> str:
        """Get path to FastText language identification model"""
        # Use lid.176.bin (176 languages) from FastText
        model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'fasttext')
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, 'lid.176.bin')

    def _initialize_financial_patterns(self) -> Dict[str, List[str]]:
        """Initialize financial domain language patterns"""
        return {
            'pt': [
                'banco', 'conta', 'valor', 'pagamento', 'transferência', 'crédito', 'débito',
                'saldo', 'agência', 'número', 'cpf', 'cnpj', 'pix', 'boleto', 'fatura',
                'real', 'reais', 'r$', 'centavo', 'moeda', 'dinheiro', 'financeiro'
            ],
            'en': [
                'bank', 'account', 'value', 'payment', 'transfer', 'credit', 'debit',
                'balance', 'agency', 'number', 'tax', 'id', 'invoice', 'bill', 'currency',
                'dollar', 'financial', 'transaction', 'deposit', 'withdrawal'
            ],
            'es': [
                'banco', 'cuenta', 'valor', 'pago', 'transferencia', 'crédito', 'débito',
                'saldo', 'agencia', 'número', 'dni', 'cif', 'factura', 'moneda', 'dinero',
                'financiero', 'euro', 'pesos', 'transacción', 'depósito', 'retiro'
            ],
            'fr': [
                'banque', 'compte', 'valeur', 'paiement', 'transfert', 'crédit', 'débit',
                'solde', 'agence', 'numéro', 'siret', 'facture', 'monnaie', 'argent',
                'financier', 'euro', 'transaction', 'dépôt', 'retrait'
            ],
            'de': [
                'bank', 'konto', 'wert', 'zahlung', 'überweisung', 'kredit', 'debit',
                'saldo', 'filiale', 'nummer', 'steuer', 'rechnung', 'währung', 'geld',
                'finanziell', 'euro', 'transaktion', 'einzahlung', 'auszahlung'
            ],
            'it': [
                'banca', 'conto', 'valore', 'pagamento', 'trasferimento', 'credito', 'debito',
                'saldo', 'agenzia', 'numero', 'iva', 'fattura', 'moneta', 'denaro',
                'finanziario', 'euro', 'transazione', 'deposito', 'prelievo'
            ]
        }

    def _initialize_confidence_thresholds(self) -> Dict[str, float]:
        """Initialize confidence thresholds for different languages"""
        return {
            'pt': 0.8,  # Portuguese - higher threshold due to financial domain focus
            'en': 0.7,  # English - standard threshold
            'es': 0.7,  # Spanish
            'fr': 0.7,  # French
            'de': 0.7,  # German
            'it': 0.7,  # Italian
            'default': 0.6  # Default for other languages
        }

    def detect_language(self, text: str) -> DetectionResult:
        """
        Detect language of input text using multiple methods

        Args:
            text: Input text to analyze

        Returns:
            DetectionResult with comprehensive language detection information
        """
        if not text or not isinstance(text, str):
            return self._create_empty_result()

        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if self.config['cache_enabled'] and cache_key in self._detection_cache:
            self._performance_stats['cache_hits'] += 1
            return self._detection_cache[cache_key]

        start_time = time.time()

        try:
            # Validate text
            if len(text.strip()) < self.config['min_text_length']:
                return self._create_short_text_result(text)

            # Run detection methods
            all_detections = []
            detection_methods_used = []

            # Primary method
            primary_detection = self._run_primary_detection(text)
            if primary_detection:
                all_detections.append(primary_detection)
                detection_methods_used.append(primary_detection.method)

            # Fallback methods
            fallback_detections = self._run_fallback_detections(text)
            all_detections.extend(fallback_detections)
            detection_methods_used.extend([d.method for d in fallback_detections])

            # Ensemble detection if enabled
            if self.config['ensemble_voting'] and len(all_detections) > 1:
                ensemble_detection = self._run_ensemble_detection(all_detections)
                if ensemble_detection:
                    all_detections.insert(0, ensemble_detection)
                    detection_methods_used.insert(0, DetectionMethod.ENSEMBLE.value)

            # Determine consensus
            consensus_result = self._determine_consensus(all_detections)

            # Apply financial domain boost if enabled
            if self.config['financial_domain_boost']:
                consensus_result = self._apply_financial_boost(text, consensus_result)

            # Create final result
            result = DetectionResult(
                primary_detection=all_detections[0] if all_detections else self._create_unknown_detection(text),
                all_detections=all_detections,
                consensus_language=consensus_result['language'],
                consensus_confidence=consensus_result['confidence'],
                detection_methods=detection_methods_used,
                fallback_used=len(detection_methods_used) > 1,
                error_message=None
            )

            # Cache result
            if self.config['cache_enabled']:
                self._detection_cache[cache_key] = result

            # Update performance stats
            self._update_performance_stats(result, time.time() - start_time)

            return result

        except Exception as e:
            self.logger.error(f"Error in language detection: {str(e)}")
            self._performance_stats['error_count'] += 1
            return self._create_error_result(text, str(e))

    def _run_primary_detection(self, text: str) -> Optional[LanguageDetection]:
        """Run primary detection method"""
        try:
            method = DetectionMethod(self.config['primary_method'])

            if method == DetectionMethod.FASTTEXT and method in self._detection_methods:
                return self._detect_with_fasttext(text)
            elif method == DetectionMethod.LANGDETECT and method in self._detection_methods:
                return self._detect_with_langdetect(text)
            elif method == DetectionMethod.LANGID and method in self._detection_methods:
                return self._detect_with_langid(text)

        except Exception as e:
            self.logger.warning(f"Error in primary detection: {str(e)}")

        return None

    def _run_fallback_detections(self, text: str) -> List[LanguageDetection]:
        """Run fallback detection methods"""
        detections = []

        for method_name in self.config['fallback_methods']:
            try:
                method = DetectionMethod(method_name)

                if method == DetectionMethod.FASTTEXT and method in self._detection_methods:
                    detection = self._detect_with_fasttext(text)
                elif method == DetectionMethod.LANGDETECT and method in self._detection_methods:
                    detection = self._detect_with_langdetect(text)
                elif method == DetectionMethod.LANGID and method in self._detection_methods:
                    detection = self._detect_with_langid(text)
                elif method == DetectionMethod.POLYGLOT and method in self._detection_methods:
                    detection = self._detect_with_polyglot(text)
                else:
                    continue

                if detection:
                    detections.append(detection)

            except Exception as e:
                self.logger.warning(f"Error in fallback detection {method_name}: {str(e)}")

        return detections

    def _detect_with_fasttext(self, text: str) -> Optional[LanguageDetection]:
        """Detect language using FastText"""
        try:
            start_time = time.time()
            model = self._detection_methods[DetectionMethod.FASTTEXT]

            # Preprocess text for FastText
            clean_text = self._preprocess_for_fasttext(text)

            # Get predictions
            predictions = model.predict(clean_text, k=3)  # Top 3 predictions

            if predictions and len(predictions) >= 2:
                labels, probabilities = predictions

                # Extract language codes and convert to ISO format
                primary_lang = self._fasttext_label_to_iso(labels[0])
                primary_conf = float(probabilities[0])

                alternatives = []
                for i in range(1, min(3, len(labels))):
                    alt_lang = self._fasttext_label_to_iso(labels[i])
                    alt_conf = float(probabilities[i])
                    alternatives.append((alt_lang, alt_conf))

                processing_time = time.time() - start_time

                return LanguageDetection(
                    language=primary_lang,
                    confidence=primary_conf,
                    method=DetectionMethod.FASTTEXT.value,
                    alternatives=alternatives,
                    processing_time=processing_time,
                    text_sample=text[:100],
                    text_length=len(text)
                )

        except Exception as e:
            self.logger.warning(f"FastText detection error: {str(e)}")

        return None

    def _detect_with_langdetect(self, text: str) -> Optional[LanguageDetection]:
        """Detect language using LangDetect"""
        try:
            start_time = time.time()
            langdetect = self._detection_methods[DetectionMethod.LANGDETECT]

            # Get language probabilities
            lang_probs = langdetect['detect_langs'](text)

            if lang_probs:
                primary = lang_probs[0]
                primary_lang = primary.lang
                primary_conf = primary.prob

                alternatives = [(lp.lang, lp.prob) for lp in lang_probs[1:3]]

                processing_time = time.time() - start_time

                return LanguageDetection(
                    language=primary_lang,
                    confidence=primary_conf,
                    method=DetectionMethod.LANGDETECT.value,
                    alternatives=alternatives,
                    processing_time=processing_time,
                    text_sample=text[:100],
                    text_length=len(text)
                )

        except Exception as e:
            self.logger.warning(f"LangDetect error: {str(e)}")

        return None

    def _detect_with_langid(self, text: str) -> Optional[LanguageDetection]:
        """Detect language using LangID"""
        try:
            start_time = time.time()
            langid = self._detection_methods[DetectionMethod.LANGID]

            # Get language prediction
            lang, conf = langid.classify(text)

            processing_time = time.time() - start_time

            return LanguageDetection(
                language=lang,
                confidence=conf,
                method=DetectionMethod.LANGID.value,
                alternatives=[],  # LangID doesn't provide alternatives
                processing_time=processing_time,
                text_sample=text[:100],
                text_length=len(text)
            )

        except Exception as e:
            self.logger.warning(f"LangID error: {str(e)}")

        return None

    def _detect_with_polyglot(self, text: str) -> Optional[LanguageDetection]:
        """Detect language using Polyglot"""
        try:
            start_time = time.time()
            Detector = self._detection_methods[DetectionMethod.POLYGLOT]

            detector = Detector(text, quiet=True)

            if detector.languages:
                primary = detector.languages[0]
                primary_lang = primary.code
                primary_conf = primary.confidence / 100.0  # Convert to 0-1 scale

                alternatives = [(lang.code, lang.confidence / 100.0)
                              for lang in detector.languages[1:3]]

                processing_time = time.time() - start_time

                return LanguageDetection(
                    language=primary_lang,
                    confidence=primary_conf,
                    method=DetectionMethod.POLYGLOT.value,
                    alternatives=alternatives,
                    processing_time=processing_time,
                    text_sample=text[:100],
                    text_length=len(text)
                )

        except Exception as e:
            self.logger.warning(f"Polyglot error: {str(e)}")

        return None

    def _run_ensemble_detection(self, detections: List[LanguageDetection]) -> Optional[LanguageDetection]:
        """Run ensemble detection using voting"""
        try:
            if not detections:
                return None

            # Collect votes for each language
            votes = {}
            total_confidence = 0

            for detection in detections:
                lang = detection.language
                conf = detection.confidence

                if lang not in votes:
                    votes[lang] = {'count': 0, 'total_conf': 0.0}

                votes[lang]['count'] += 1
                votes[lang]['total_conf'] += conf
                total_confidence += conf

            # Find language with most votes, break ties by confidence
            best_lang = None
            best_score = 0

            for lang, vote_data in votes.items():
                # Score = vote_count + (average_confidence * weight)
                avg_conf = vote_data['total_conf'] / vote_data['count']
                score = vote_data['count'] + (avg_conf * 0.5)

                if score > best_score:
                    best_score = score
                    best_lang = lang

            if best_lang:
                ensemble_conf = votes[best_lang]['total_conf'] / votes[best_lang]['count']

                return LanguageDetection(
                    language=best_lang,
                    confidence=min(ensemble_conf + 0.1, 1.0),  # Boost confidence for ensemble
                    method=DetectionMethod.ENSEMBLE.value,
                    alternatives=sorted(
                        [(l, votes[l]['total_conf'] / votes[l]['count'])
                         for l in votes if l != best_lang],
                        key=lambda x: x[1],
                        reverse=True
                    )[:2],
                    processing_time=sum(d.processing_time for d in detections),
                    text_sample=detections[0].text_sample,
                    text_length=detections[0].text_length
                )

        except Exception as e:
            self.logger.warning(f"Ensemble detection error: {str(e)}")

        return None

    def _determine_consensus(self, detections: List[LanguageDetection]) -> Dict[str, Any]:
        """Determine consensus language from multiple detections"""
        if not detections:
            return {'language': 'unknown', 'confidence': 0.0}

        # Use primary detection if confidence is high enough
        primary = detections[0]
        threshold = self._confidence_thresholds.get(
            primary.language,
            self._confidence_thresholds['default']
        )

        if primary.confidence >= threshold:
            return {
                'language': primary.language,
                'confidence': primary.confidence
            }

        # Find most common language among all detections
        lang_counts = {}
        lang_confidences = {}

        for detection in detections:
            lang = detection.language
            conf = detection.confidence

            if lang not in lang_counts:
                lang_counts[lang] = 0
                lang_confidences[lang] = []

            lang_counts[lang] += 1
            lang_confidences[lang].append(conf)

        # Get language with highest count, then highest average confidence
        consensus_lang = max(
            lang_counts.keys(),
            key=lambda l: (lang_counts[l], sum(lang_confidences[l]) / len(lang_confidences[l]))
        )

        avg_conf = sum(lang_confidences[consensus_lang]) / len(lang_confidences[consensus_lang])

        return {
            'language': consensus_lang,
            'confidence': avg_conf
        }

    def _apply_financial_boost(self, text: str, consensus_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply financial domain boost to detection result"""
        try:
            text_lower = text.lower()
            boosted_result = consensus_result.copy()

            # Check for financial terms in each supported language
            for lang, patterns in self._financial_patterns.items():
                if lang == consensus_result['language']:
                    continue

                # Count financial terms for this language
                term_count = sum(1 for pattern in patterns if pattern in text_lower)

                if term_count >= 2:  # At least 2 financial terms
                    # Calculate boost factor based on term density
                    term_density = term_count / len(text.split())
                    boost_factor = min(term_density * 2.0, 0.3)  # Max 30% boost

                    # Only boost if this language wasn't already detected with high confidence
                    if consensus_result['confidence'] < 0.8:
                        boosted_result['language'] = lang
                        boosted_result['confidence'] = min(
                            consensus_result['confidence'] + boost_factor,
                            0.95  # Cap at 95%
                        )
                        self.logger.info(f"Applied financial boost: {consensus_result['language']} -> {lang}")
                        break

            return boosted_result

        except Exception as e:
            self.logger.warning(f"Error applying financial boost: {str(e)}")
            return consensus_result

    def _preprocess_for_fasttext(self, text: str) -> str:
        """Preprocess text for FastText input"""
        # FastText works best with clean, normalized text
        import re

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.,!?\-]', '', text)

        # Normalize case (FastText is case-sensitive)
        text = text.lower()

        return text.strip()

    def _fasttext_label_to_iso(self, label: str) -> str:
        """Convert FastText label to ISO language code"""
        # FastText labels are in format __label__LANG
        if label.startswith('__label__'):
            lang_code = label[9:]  # Remove __label__ prefix

            # Handle common mappings
            mappings = {
                'zh': 'zh-cn',  # Chinese
                'ar': 'ar',     # Arabic
                'hi': 'hi',     # Hindi
                'bn': 'bn',     # Bengali
                'pt': 'pt',     # Portuguese
                'es': 'es',     # Spanish
                'fr': 'fr',     # French
                'de': 'de',     # German
                'it': 'it',     # Italian
                'ja': 'ja',     # Japanese
                'ko': 'ko',     # Korean
                'ru': 'ru',     # Russian
                'tr': 'tr',     # Turkish
                'pl': 'pl',     # Polish
                'nl': 'nl',     # Dutch
                'sv': 'sv',     # Swedish
                'da': 'da',     # Danish
                'no': 'no',     # Norwegian
                'fi': 'fi',     # Finnish
                'cs': 'cs',     # Czech
                'hu': 'hu',     # Hungarian
                'ro': 'ro',     # Romanian
                'sk': 'sk',     # Slovak
                'sl': 'sl',     # Slovenian
                'hr': 'hr',     # Croatian
                'sr': 'sr',     # Serbian
                'mk': 'mk',     # Macedonian
                'bg': 'bg',     # Bulgarian
                'uk': 'uk',     # Ukrainian
                'el': 'el',     # Greek
                'he': 'he',     # Hebrew
                'th': 'th',     # Thai
                'vi': 'vi',     # Vietnamese
                'id': 'id',     # Indonesian
                'ms': 'ms',     # Malay
                'tl': 'tl',     # Tagalog
                'sw': 'sw',     # Swahili
                'am': 'am',     # Amharic
                'ne': 'ne',     # Nepali
                'si': 'si',     # Sinhala
                'my': 'my',     # Burmese
                'km': 'km',     # Khmer
                'lo': 'lo',     # Lao
                'ka': 'ka',     # Georgian
                'hy': 'hy',     # Armenian
                'az': 'az',     # Azerbaijani
                'kk': 'kk',     # Kazakh
                'uz': 'uz',     # Uzbek
                'ky': 'ky',     # Kyrgyz
                'tg': 'tg',     # Tajik
                'tk': 'tk',     # Turkmen
                'mn': 'mn',     # Mongolian
                'bo': 'bo',     # Tibetan
                'dz': 'dz',     # Dzongkha
                'ug': 'ug',     # Uyghur
                'si': 'si',     # Sinhala (duplicate)
                'pa': 'pa',     # Punjabi
                'gu': 'gu',     # Gujarati
                'or': 'or',     # Oriya
                'ta': 'ta',     # Tamil
                'te': 'te',     # Telugu
                'kn': 'kn',     # Kannada
                'ml': 'ml',     # Malayalam
                'sd': 'sd',     # Sindhi
                'ur': 'ur',     # Urdu
                'fa': 'fa',     # Persian
                'ps': 'ps',     # Pashto
                'ku': 'ku',     # Kurdish
                'sd': 'sd',     # Sindhi (duplicate)
                'yo': 'yo',     # Yoruba
                'ig': 'ig',     # Igbo
                'ha': 'ha',     # Hausa
                'rw': 'rw',     # Kinyarwanda
                'mg': 'mg',     # Malagasy
                'st': 'st',     # Sesotho
                'tn': 'tn',     # Tswana
                'xh': 'xh',     # Xhosa
                'zu': 'zu',     # Zulu
                'af': 'af',     # Afrikaans
                'sq': 'sq',     # Albanian
                'bs': 'bs',     # Bosnian
                'et': 'et',     # Estonian
                'lv': 'lv',     # Latvian
                'lt': 'lt',     # Lithuanian
                'mt': 'mt',     # Maltese
                'is': 'is',     # Icelandic
                'ga': 'ga',     # Irish
                'cy': 'cy',     # Welsh
                'br': 'br',     # Breton
                'co': 'co',     # Corsican
                'eo': 'eo',     # Esperanto
                'vo': 'vo',     # Volapük
                'jv': 'jv',     # Javanese
                'su': 'su',     # Sundanese
                'ceb': 'ceb',   # Cebuano
                'ht': 'ht',     # Haitian Creole
                'hmn': 'hmn',   # Hmong
                'ny': 'ny',     # Chichewa
                'sm': 'sm',     # Samoan
                'to': 'to',     # Tongan
                'mi': 'mi',     # Maori
                'haw': 'haw',   # Hawaiian
                'ty': 'ty',     # Tahitian
                'so': 'so',     # Somali
                'ti': 'ti',     # Tigrinya
                'om': 'om',     # Oromo
                'ss': 'ss',     # Swati
                'nr': 'nr',     # Ndebele
                've': 've',     # Venda
                'ts': 'ts',     # Tsonga
                'sn': 'sn',     # Shona
                'ee': 'ee',     # Ewe
                'ak': 'ak',     # Akan
                'tw': 'tw',     # Twi
                'bm': 'bm',     # Bambara
                'ff': 'ff',     # Fulah
                'wo': 'wo',     # Wolof
                'ln': 'ln',     # Lingala
                'kg': 'kg',     # Kongo
                'lu': 'lu',     # Luba-Katanga
                'fy': 'fy',     # Western Frisian
                'lb': 'lb',     # Luxembourgish
                'sc': 'sc',     # Sardinian
                'ast': 'ast',   # Asturian
                'gl': 'gl',     # Galician
                'an': 'an',     # Aragonese
                'oc': 'oc',     # Occitan
                'be': 'be',     # Belarusian
                'cv': 'cv',     # Chuvash
                'tt': 'tt',     # Tatar
                'ba': 'ba',     # Bashkir
                'ch': 'ch',     # Chamorro
                'mh': 'mh',     # Marshallese
                'bi': 'bi',     # Bislama
                'gil': 'gil',   # Kiribati
                'tpi': 'tpi',   # Tok Pisin
                'niu': 'niu',   # Niuean
                'tet': 'tet',   # Tetum
                'sg': 'sg',     # Sango
                'ln': 'ln',     # Lingala (duplicate)
                'ki': 'ki',     # Kikuyu
                'lg': 'lg',     # Ganda
                'na': 'na',     # Nauru
                'rn': 'rn',     # Kirundi
                'chr': 'chr',   # Cherokee
                'iu': 'iu',     # Inuktitut
                'ik': 'ik',     # Inupiaq
                'kl': 'kl',     # Kalaallisut
                'azj': 'az',    # North Azerbaijani
                'azb': 'az',    # South Azerbaijani
                'gan': 'zh',    # Gan Chinese
                'hak': 'zh',    # Hakka Chinese
                'hsn': 'zh',    # Xiang Chinese
                'lzh': 'zh',    # Literary Chinese
                'nan': 'zh',    # Min Nan Chinese
                'wuu': 'zh',    # Wu Chinese
                'yue': 'zh',    # Yue Chinese
                'cmn': 'zh',    # Mandarin Chinese
                'dtp': 'ms',    # Central Dusun
                'kzj': 'ms',    # Coastal Kadazan
                'bjn': 'ms',    # Banjar
                'btg': 'ms',    # Gagnoa Bété
                'bvu': 'ms',    # Bukit Malay
                'coa': 'ms',    # Cocos Islands Malay
                'dup': 'ms',    # Duano
                'hji': 'ms',    # Haji
                'ind': 'id',    # Indonesian
                'zlm': 'ms',    # Malay
                'zsm': 'ms',    # Standard Malay
                'map': 'ms',    # Austronesian
                'poz': 'ms',    # Malayo-Polynesian
                'pqw': 'ms',    # Northwest Sumatra Barisan Malay
                'bew': 'ms',    # Betawi
                'jak': 'ms',    # Jakun
                'jax': 'ms',    # Jambi Malay
                'kvb': 'ms',    # Kubu
                'kvr': 'ms',    # Kerinci
                'ljp': 'ms',    # Lampung Api
                'mad': 'ms',    # Madurese
                'mak': 'ms',    # Makasar
                'min': 'ms',    # Minangkabau
                'mui': 'ms',    # Musi
                'orn': 'ms',    # Orang Kanaq
                'ors': 'ms',    # Orang Seletar
                'pel': 'pel',   # Pekal
                'slm': 'ms',    # Salam
                'svr': 'ms',    # Savara
                'vkt': 'ms',    # Tenggarong Kutai Malay
                'xmm': 'ms',    # Manado Malay
                'ace': 'ace',   # Acehnese
                'ban': 'ban',   # Balinese
                'btk': 'ms',    # Batak
                'btx': 'ms',    # Batak Karo
                'bug': 'bug',   # Buginese
                'bvb': 'ms',    # Bubat
                'bve': 'ms',    # Berau Malay
                'bvu': 'ms',    # Bukit Malay (duplicate)
                'coa': 'ms',    # Cocos Islands Malay (duplicate)
                'dak': 'dak',   # Dakota
                'day': 'ms',    # Land Dayak
                'djk': 'ms',    # Eastern Maroon Creole
                'dup': 'ms',    # Duano (duplicate)
                'flu': 'ms',    # Fula
                'gor': 'gor',   # Gorontalo
                'hji': 'ms',    # Haji (duplicate)
                'iba': 'iba',   # Iban
                'ind': 'id',    # Indonesian (duplicate)
                'jav': 'jv',    # Javanese (duplicate)
                'kge': 'ms',    # Komering
                'kvr': 'ms',    # Kerinci (duplicate)
                'lce': 'ms',    # Loncong
                'lcf': 'ms',    # Lubu
                'liw': 'ms',    # Col
                'mad': 'ms',    # Madurese (duplicate)
                'mak': 'ms',    # Makasar (duplicate)
                'mfa': 'ms',    # Pattani Malay
                'mfb': 'ms',    # Bangka
                'min': 'ms',    # Minangkabau (duplicate)
                'mly': 'ms',    # Malay
                'mnc': 'mnc',   # Manchu
                'mqy': 'ms',    # Manggarai
                'msa': 'ms',    # Malay
                'mui': 'ms',    # Musi (duplicate)
                'nia': 'nia',   # Nias
                'nij': 'ms',    # Ngaju
                'niu': 'niu',   # Niuean (duplicate)
                'njm': 'ms',    # Angami
                'nli': 'ms',    # Grangali
                'orn': 'ms',    # Orang Kanaq (duplicate)
                'ors': 'ms',    # Orang Seletar (duplicate)
                'osa': 'osa',   # Osage
                'pag': 'pag',   # Pangasinan
                'pam': 'pam',   # Pampanga
                'pau': 'pau',   # Palauan
                'pcc': 'ms',    # Bouyei
                'pdc': 'de',    # Pennsylvania German
                'pdt': 'de',    # Plautdietsch
                'pel': 'pel',   # Pekal (duplicate)
                'pfa': 'ms',    # Páez
                'pnb': 'pa',    # Western Punjabi
                'ppl': 'ms',    # Pipil
                'prf': 'de',    # Parthian
                'rkt': 'bn',    # Rangpuri
                'rnl': 'ms',    # Ranglong
                'sda': 'ms',    # Toraja-Sa'dan
                'sea': 'ms',    # Semai
                'slm': 'ms',    # Salam (duplicate)
                'sml': 'ms',    # Central Sama
                'sun': 'su',    # Sundanese (duplicate)
                'svr': 'ms',    # Savara (duplicate)
                'swv': 'ms',    # Shekhawati
                'syl': 'bn',    # Sylheti
                'tby': 'ms',    # Tabaru
                'tdt': 'ms',    # Tetun Dili
                'tem': 'ms',    # Temuan
                'tet': 'tet',   # Tetum (duplicate)
                'tgl': 'tl',    # Tagalog (duplicate)
                'tgy': 'ms',    # Tagbanwa
                'tkl': 'tkl',   # Tokelau
                'tlb': 'ms',    # Tobelo
                'tmr': 'ms',    # Jewish Babylonian Aramaic
                'tpi': 'tpi',   # Tok Pisin (duplicate)
                'trp': 'ms',    # Kok Borok
                'tsg': 'ms',    # Tausug
                'txy': 'ms',    # Tanosy Malagasy
                'vgt': 'ms',    # Vlaamse Gebarentaal
                'vkt': 'ms',    # Tenggarong Kutai Malay (duplicate)
                'war': 'war',   # Waray
                'wls': 'ms',    # Wallisian
                'xmm': 'ms',    # Manado Malay (duplicate)
                'xmm': 'ms',    # Manado Malay (duplicate)
                'xsb': 'ms',    # Sambal
                'yap': 'yap',   # Yapese
                'ydd': 'yi',    # Eastern Yiddish
                'yih': 'yi',    # Western Yiddish
                'yua': 'yua',   # Yucateco
                'yue': 'zh',    # Yue Chinese (duplicate)
                'zha': 'za',    # Zhuang
                'zho': 'zh',    # Chinese
                'zlm': 'ms',    # Malay (duplicate)
                'zsm': 'ms',    # Standard Malay (duplicate)
                'zul': 'zu',    # Zulu (duplicate)
                'zza': 'zza',   # Zaza
            }

            return mappings.get(lang_code, lang_code)

        return label

    def _update_performance_stats(self, result: DetectionResult, processing_time: float):
        """Update performance statistics"""
        self._performance_stats['total_detections'] += 1

        # Update average confidence
        current_avg = self._performance_stats['average_confidence']
        self._performance_stats['average_confidence'] = (
            (current_avg * (self._performance_stats['total_detections'] - 1)) +
            result.consensus_confidence
        ) / self._performance_stats['total_detections']

        # Update method usage
        for method in result.detection_methods:
            if method not in self._performance_stats['method_usage']:
                self._performance_stats['method_usage'][method] = 0
            self._performance_stats['method_usage'][method] += 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self._performance_stats.copy()

    def clear_cache(self):
        """Clear detection cache"""
        self._detection_cache.clear()
        self.logger.info("Language detection cache cleared")

    def _create_empty_result(self) -> DetectionResult:
        """Create empty detection result"""
        return DetectionResult(
            primary_detection=LanguageDetection(
                language='unknown',
                confidence=0.0,
                method='none',
                alternatives=[],
                processing_time=0.0,
                text_sample='',
                text_length=0
            ),
            all_detections=[],
            consensus_language='unknown',
            consensus_confidence=0.0,
            detection_methods=[],
            fallback_used=False,
            error_message='Empty or invalid input'
        )

    def _create_short_text_result(self, text: str) -> DetectionResult:
        """Create result for short text"""
        return DetectionResult(
            primary_detection=LanguageDetection(
                language='unknown',
                confidence=0.0,
                method='validation',
                alternatives=[],
                processing_time=0.0,
                text_sample=text[:100],
                text_length=len(text)
            ),
            all_detections=[],
            consensus_language='unknown',
            consensus_confidence=0.0,
            detection_methods=['validation'],
            fallback_used=False,
            error_message=f'Text too short for reliable detection (min {self.config["min_text_length"]} chars)'
        )

    def _create_unknown_detection(self, text: str) -> LanguageDetection:
        """Create unknown language detection"""
        return LanguageDetection(
            language='unknown',
            confidence=0.0,
            method='unknown',
            alternatives=[],
            processing_time=0.0,
            text_sample=text[:100],
            text_length=len(text)
        )

    def _create_error_result(self, text: str, error: str) -> DetectionResult:
        """Create error detection result"""
        return DetectionResult(
            primary_detection=self._create_unknown_detection(text),
            all_detections=[],
            consensus_language='unknown',
            consensus_confidence=0.0,
            detection_methods=[],
            fallback_used=False,
            error_message=error
        )

    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        """
        Detect languages for a batch of texts

        Args:
            texts: List of texts to analyze

        Returns:
            List of DetectionResult objects
        """
        if not texts:
            return []

        try:
            results = []
            for text in texts:
                result = self.detect_language(text)
                results.append(result)

            self.logger.info(f"Processed language detection for {len(texts)} texts")
            return results

        except Exception as e:
            self.logger.error(f"Error in batch language detection: {str(e)}")
            return [self._create_error_result(text, str(e)) for text in texts]