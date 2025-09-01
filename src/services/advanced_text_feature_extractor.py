import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from collections import Counter, defaultdict
import unicodedata

# ML and NLP libraries
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import torch

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from src.services.portuguese_preprocessor import PortugueseTextPreprocessor

logger = get_logger(__name__)


class AdvancedTextFeatureExtractor:
    """
    Advanced text feature extractor with Portuguese language specialization
    Provides comprehensive text processing, feature extraction, and quality assessment
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AdvancedTextFeatureExtractor

        Args:
            config: Configuration dictionary for text feature extraction
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize core components
        self._initialize_components()

        # Initialize specialized Portuguese components
        self._initialize_portuguese_components()

        # Initialize ML models
        self._initialize_ml_models()

        # Feature storage and caching
        self.feature_cache = {}
        self.quality_metrics = {}
        self.extraction_stats = defaultdict(int)

        self.logger.info("AdvancedTextFeatureExtractor initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for advanced text feature extraction"""
        return {
            'portuguese_processing': {
                'use_advanced_preprocessor': True,
                'stemming_enabled': False,  # Use lemmatization instead
                'financial_terminology_detection': True,
                'abbreviation_expansion': True,
                'accent_normalization': True
            },
            'embedding_models': {
                'sentence_transformer': 'all-MiniLM-L6-v2',
                'financial_domain_model': 'paraphrase-multilingual-MiniLM-L12-v2',
                'max_length': 512,
                'batch_size': 32
            },
            'text_features': {
                'tfidf_enabled': True,
                'count_vectorizer_enabled': True,
                'topic_modeling_enabled': True,
                'semantic_features_enabled': True,
                'linguistic_features_enabled': True,
                'financial_features_enabled': True
            },
            'topic_modeling': {
                'method': 'lda',  # 'lda' or 'nmf'
                'n_topics': 10,
                'max_features': 1000,
                'random_state': 42
            },
            'quality_assessment': {
                'text_quality_scoring': True,
                'outlier_detection': True,
                'diversity_analysis': True,
                'semantic_coherence_check': True
            },
            'performance': {
                'cache_enabled': True,
                'parallel_processing': True,
                'batch_processing': True,
                'memory_optimization': True
            }
        }

    def _initialize_components(self):
        """Initialize core text processing components"""
        try:
            # Portuguese text preprocessor
            self.portuguese_preprocessor = PortugueseTextPreprocessor(
                use_advanced_pipeline=self.config['portuguese_processing']['use_advanced_preprocessor']
            )

            # Financial terminology patterns
            self.financial_patterns = self._get_financial_patterns()

            # Portuguese linguistic patterns
            self.linguistic_patterns = self._get_linguistic_patterns()

            self.logger.info("Core components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing core components: {str(e)}")
            raise ValidationError(f"Failed to initialize AdvancedTextFeatureExtractor: {str(e)}")

    def _initialize_portuguese_components(self):
        """Initialize Portuguese-specific processing components"""
        try:
            # Portuguese financial terminology
            self.financial_terms = self._get_portuguese_financial_terms()

            # Portuguese linguistic resources
            self.portuguese_stopwords = self._get_portuguese_stopwords()

            # Financial entity patterns
            self.entity_patterns = self._get_entity_patterns()

            self.logger.info("Portuguese components initialized successfully")

        except Exception as e:
            self.logger.warning(f"Error initializing Portuguese components: {str(e)}")

    def _initialize_ml_models(self):
        """Initialize machine learning models for text processing"""
        try:
            # Sentence transformer for embeddings
            self.sentence_transformer = SentenceTransformer(
                self.config['embedding_models']['sentence_transformer']
            )

            # Financial domain model (if different)
            if self.config['embedding_models']['financial_domain_model'] != self.config['embedding_models']['sentence_transformer']:
                self.financial_transformer = SentenceTransformer(
                    self.config['embedding_models']['financial_domain_model']
                )
            else:
                self.financial_transformer = self.sentence_transformer

            # Topic modeling components
            if self.config['text_features']['topic_modeling_enabled']:
                self._initialize_topic_model()

            # Text quality assessment model
            if self.config['quality_assessment']['text_quality_scoring']:
                self.quality_model = self._initialize_quality_model()

            self.logger.info("ML models initialized successfully")

        except Exception as e:
            self.logger.warning(f"Error initializing ML models: {str(e)}. Using basic processing.")

    def _initialize_topic_model(self):
        """Initialize topic modeling components"""
        try:
            if self.config['topic_modeling']['method'] == 'lda':
                self.topic_model = LatentDirichletAllocation(
                    n_components=self.config['topic_modeling']['n_topics'],
                    random_state=self.config['topic_modeling']['random_state'],
                    max_iter=10
                )
            else:  # nmf
                self.topic_model = NMF(
                    n_components=self.config['topic_modeling']['n_topics'],
                    random_state=self.config['topic_modeling']['random_state'],
                    max_iter=200
                )

            # Vectorizers for topic modeling
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config['topic_modeling']['max_features'],
                stop_words=self.portuguese_stopwords
            )

            self.count_vectorizer = CountVectorizer(
                max_features=self.config['topic_modeling']['max_features'],
                stop_words=self.portuguese_stopwords
            )

        except Exception as e:
            self.logger.warning(f"Error initializing topic model: {str(e)}")
            self.topic_model = None

    def _initialize_quality_model(self):
        """Initialize text quality assessment model"""
        try:
            # Simple quality model based on heuristics
            # In production, this could be a trained ML model
            self.quality_model = {
                'min_length': 3,
                'max_length': 500,
                'min_words': 1,
                'max_words': 100,
                'financial_term_bonus': 0.1,
                'diversity_bonus': 0.05
            }
            return self.quality_model

        except Exception as e:
            self.logger.warning(f"Error initializing quality model: {str(e)}")
            return None

    def _get_financial_patterns(self) -> Dict[str, str]:
        """Get financial-specific text patterns"""
        return {
            # Amount patterns
            r'r\$?\s*\d+[\.,]\d{2}': 'MONETARY_VALUE',
            r'\b\d+[\.,]\d{2}\b': 'AMOUNT',

            # Transaction types
            r'\b(transferencia|ted|doc|pix)\b': 'TRANSFER',
            r'\b(pagamento|pgto)\b': 'PAYMENT',
            r'\b(deposito|depósito)\b': 'DEPOSIT',
            r'\b(saque|retirada)\b': 'WITHDRAWAL',

            # Financial institutions
            r'\b(itau|itaú)\b': 'BANK_ITAU',
            r'\b(bradesco|brad)\b': 'BANK_BRADESCO',
            r'\b(santander|sant)\b': 'BANK_SANTANDER',

            # Account information
            r'\bag\.?\s*\d{4}': 'AGENCY',
            r'\b\d{4,7}[-]?\d{1,2}': 'ACCOUNT',

            # Financial terms
            r'\b(saldo|limite|credito|debito)\b': 'FINANCIAL_TERM',
            r'\b(juros|taxa|tarifa)\b': 'FINANCIAL_COST'
        }

    def _get_linguistic_patterns(self) -> Dict[str, Any]:
        """Get Portuguese linguistic patterns"""
        return {
            'sentence_enders': r'[.!?]+',
            'word_separators': r'\s+',
            'punctuation': r'[^\w\s]',
            'numbers': r'\d+',
            'uppercase_ratio': 0.3,  # Threshold for uppercase detection
            'special_chars': r'[@#$%^&*()_+={}\[\]|\\:;"\'<>,.?/~`]'
        }

    def _get_portuguese_financial_terms(self) -> List[str]:
        """Get comprehensive list of Portuguese financial terms"""
        return [
            # Transaction types
            'transferencia', 'pagamento', 'deposito', 'saque', 'credito', 'debito',
            'ted', 'doc', 'pix', 'boleto', 'cheque', 'talao',

            # Financial institutions
            'banco', 'agencia', 'conta', 'saldo', 'limite', 'juros', 'taxa',
            'tarifa', 'custo', 'rendimento', 'investimento', 'aplicacao',

            # Account operations
            'extrato', 'movimentacao', 'lancamento', 'conciliacao', 'reconciliacao',
            'balanco', 'balancete', 'demonstrativo',

            # Financial products
            'emprestimo', 'financiamento', 'consorcio', 'seguro', 'previdencia',
            'poupanca', 'cdb', 'lci', 'lca', 'fundos',

            # Business terms
            'cliente', 'fornecedor', 'empresa', 'cnpj', 'cpf', 'nota', 'fiscal',
            'contrato', 'negocio', 'transacao', 'operacao',

            # Banking terms
            'caixa', 'guiche', 'atendente', 'gerente', 'diretor', 'contador',
            'auditor', 'compliance', 'regulador', 'banco_central'
        ]

    def _get_portuguese_stopwords(self) -> List[str]:
        """Get Portuguese stopwords"""
        return [
            'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com',
            'nao', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos',
            'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'a', 'seu', 'sua',
            'ou', 'ser', 'quando', 'muito', 'ha', 'nos', 'ja', 'esta', 'eu',
            'tambem', 'so', 'pelo', 'pela', 'ate', 'isso', 'ela', 'entre',
            'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem',
            'nas', 'me', 'esse', 'eles', 'estao', 'voce', 'tinha', 'foram',
            'essa', 'num', 'nem', 'suas', 'meu', 'as', 'minha', 'tem', 'numa',
            'pelos', 'elas', 'havia', 'seja', 'qual', 'sera', 'nos', 'tenho',
            'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse', 'dele',
            'tu', 'te', 'voces', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua',
            'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas'
        ]

    def _get_entity_patterns(self) -> Dict[str, str]:
        """Get patterns for entity extraction"""
        return {
            'monetary_values': r'r\$?\s*\d+[\.,]\d{2}|\b\d+[\.,]\d{2}\s*(reais|real)',
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'account_numbers': r'\b\d{4,7}[-]?\d{1,2}',
            'agency_codes': r'\bag\.?\s*\d{4}',
            'document_numbers': r'\b\d{11}|\b\d{14}',  # CPF/CNPJ
            'transaction_codes': r'\b\d{6,12}'  # Transaction IDs
        }

    def extract_text_features(self, texts: List[str],
                            feature_types: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive text features from a list of texts

        Args:
            texts: List of input texts
            feature_types: Types of features to extract (optional)

        Returns:
            Dictionary of extracted features
        """
        try:
            if not texts:
                return {}

            self.logger.info(f"Extracting text features for {len(texts)} texts")

            # Default feature types
            if feature_types is None:
                feature_types = ['embeddings', 'tfidf', 'linguistic', 'financial', 'semantic']

            features_dict = {}

            # Preprocess texts
            processed_texts = self._preprocess_texts_batch(texts)

            # Extract different types of features
            for feature_type in feature_types:
                if feature_type == 'embeddings':
                    features_dict['embeddings'] = self._extract_embeddings(processed_texts)
                elif feature_type == 'tfidf':
                    features_dict['tfidf'] = self._extract_tfidf_features(processed_texts)
                elif feature_type == 'linguistic':
                    features_dict['linguistic'] = self._extract_linguistic_features(processed_texts)
                elif feature_type == 'financial':
                    features_dict['financial'] = self._extract_financial_features(processed_texts)
                elif feature_type == 'semantic':
                    features_dict['semantic'] = self._extract_semantic_features(processed_texts)
                elif feature_type == 'topics':
                    features_dict['topics'] = self._extract_topic_features(processed_texts)

            # Update extraction statistics
            self._update_extraction_stats(features_dict)

            return features_dict

        except Exception as e:
            self.logger.error(f"Error extracting text features: {str(e)}")
            return {}

    def _preprocess_texts_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts using advanced Portuguese processing"""
        try:
            processed_texts = []

            for text in texts:
                if not text or not isinstance(text, str):
                    processed_texts.append("")
                    continue

                # Use advanced Portuguese preprocessor
                if self.config['portuguese_processing']['use_advanced_preprocessor']:
                    result = self.portuguese_preprocessor.preprocess_with_advanced_features(text)
                    processed_text = result.get('processed_text', text)
                else:
                    # Basic preprocessing
                    processed_text = self._basic_preprocess_text(text)

                processed_texts.append(processed_text)

            self.logger.debug(f"Preprocessed {len(texts)} texts")
            return processed_texts

        except Exception as e:
            self.logger.warning(f"Error in batch preprocessing: {str(e)}. Using original texts.")
            return texts

    def _basic_preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        try:
            # Convert to lowercase
            text = text.lower()

            # Remove accents
            text = ''.join(
                char for char in unicodedata.normalize('NFD', text)
                if unicodedata.category(char) != 'Mn'
            )

            # Remove special characters and extra whitespace
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)

            return text.strip()

        except Exception:
            return text

    def _extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract sentence embeddings"""
        try:
            if not texts:
                return np.array([])

            # Use financial domain model for financial texts
            model = self.financial_transformer

            embeddings = model.encode(
                texts,
                batch_size=self.config['embedding_models']['batch_size'],
                show_progress_bar=False,
                convert_to_numpy=True
            )

            self.logger.debug(f"Extracted embeddings with shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error extracting embeddings: {str(e)}")
            return np.zeros((len(texts), self.sentence_transformer.get_sentence_embedding_dimension()))

    def _extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Extract TF-IDF features"""
        try:
            if not texts or not self.config['text_features']['tfidf_enabled']:
                return np.array([])

            # Filter out empty texts
            valid_texts = [text for text in texts if text.strip()]

            if not valid_texts:
                return np.zeros((len(texts), 100))  # Return zeros for empty input

            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words=self.portuguese_stopwords,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )

            # Fit and transform
            tfidf_features = vectorizer.fit_transform(valid_texts).toarray()

            # Handle case where we have fewer valid texts
            if len(valid_texts) < len(texts):
                full_features = np.zeros((len(texts), tfidf_features.shape[1]))
                valid_indices = [i for i, text in enumerate(texts) if text.strip()]
                full_features[valid_indices] = tfidf_features
                return full_features

            return tfidf_features

        except Exception as e:
            self.logger.error(f"Error extracting TF-IDF features: {str(e)}")
            return np.zeros((len(texts), 100))

    def _extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract linguistic features"""
        try:
            features = []

            for text in texts:
                if not text:
                    features.append([0] * 15)  # 15 linguistic features
                    continue

                # Basic text statistics
                text_length = len(text)
                word_count = len(text.split())
                sentence_count = len(re.findall(self.linguistic_patterns['sentence_enders'], text)) or 1
                avg_word_length = text_length / word_count if word_count > 0 else 0
                avg_sentence_length = word_count / sentence_count

                # Character analysis
                uppercase_count = sum(1 for c in text if c.isupper())
                uppercase_ratio = uppercase_count / text_length if text_length > 0 else 0
                digit_count = sum(1 for c in text if c.isdigit())
                digit_ratio = digit_count / text_length if text_length > 0 else 0
                special_char_count = len(re.findall(self.linguistic_patterns['special_chars'], text))
                special_char_ratio = special_char_count / text_length if text_length > 0 else 0

                # Word analysis
                words = text.split()
                unique_words = set(words)
                unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0

                # Long words (more than 6 characters)
                long_words = [word for word in words if len(word) > 6]
                long_word_ratio = len(long_words) / word_count if word_count > 0 else 0

                # Stopword ratio
                stopwords_in_text = [word for word in words if word in self.portuguese_stopwords]
                stopword_ratio = len(stopwords_in_text) / word_count if word_count > 0 else 0

                # Punctuation analysis
                punctuation_count = len(re.findall(self.linguistic_patterns['punctuation'], text))
                punctuation_ratio = punctuation_count / text_length if text_length > 0 else 0

                feature_vector = [
                    text_length, word_count, sentence_count, avg_word_length, avg_sentence_length,
                    uppercase_ratio, digit_ratio, special_char_ratio, unique_word_ratio,
                    long_word_ratio, stopword_ratio, punctuation_ratio, uppercase_count,
                    digit_count, special_char_count
                ]

                features.append(feature_vector)

            return np.array(features)

        except Exception as e:
            self.logger.error(f"Error extracting linguistic features: {str(e)}")
            return np.zeros((len(texts), 15))

    def _extract_financial_features(self, texts: List[str]) -> np.ndarray:
        """Extract financial-specific features"""
        try:
            features = []

            for text in texts:
                if not text:
                    features.append([0] * 20)  # 20 financial features
                    continue

                text_lower = text.lower()

                # Count financial terms
                financial_term_counts = [text_lower.count(term) for term in self.financial_terms[:10]]  # Top 10 terms

                # Entity extraction counts
                monetary_count = len(re.findall(self.entity_patterns['monetary_values'], text))
                date_count = len(re.findall(self.entity_patterns['dates'], text))
                account_count = len(re.findall(self.entity_patterns['account_numbers'], text))
                agency_count = len(re.findall(self.entity_patterns['agency_codes'], text))
                document_count = len(re.findall(self.entity_patterns['document_numbers'], text))

                # Pattern matching features
                transfer_count = len(re.findall(r'\btransferencia\b', text_lower))
                payment_count = len(re.findall(r'\bpagamento\b', text_lower))
                deposit_count = len(re.findall(r'\bdeposito\b', text_lower))
                withdrawal_count = len(re.findall(r'\bsaque\b', text_lower))

                # Bank mention features
                bank_itau = 1 if 'itau' in text_lower else 0
                bank_bradesco = 1 if 'bradesco' in text_lower else 0
                bank_santander = 1 if 'santander' in text_lower else 0

                # Financial operation indicators
                has_credit = 1 if 'credito' in text_lower else 0
                has_debit = 1 if 'debito' in text_lower else 0
                has_balance = 1 if 'saldo' in text_lower else 0
                has_limit = 1 if 'limite' in text_lower else 0

                feature_vector = financial_term_counts + [
                    monetary_count, date_count, account_count, agency_count, document_count,
                    transfer_count, payment_count, deposit_count, withdrawal_count,
                    bank_itau, bank_bradesco, bank_santander,
                    has_credit, has_debit, has_balance, has_limit
                ]

                features.append(feature_vector)

            return np.array(features)

        except Exception as e:
            self.logger.error(f"Error extracting financial features: {str(e)}")
            return np.zeros((len(texts), 20))

    def _extract_semantic_features(self, texts: List[str]) -> np.ndarray:
        """Extract semantic features"""
        try:
            if not texts:
                return np.array([])

            # For now, use embeddings as semantic features
            # In production, this could include more sophisticated semantic analysis
            embeddings = self._extract_embeddings(texts)

            # Additional semantic features could include:
            # - Semantic similarity scores
            # - Topic coherence
            # - Semantic diversity measures

            return embeddings

        except Exception as e:
            self.logger.error(f"Error extracting semantic features: {str(e)}")
            return np.zeros((len(texts), self.sentence_transformer.get_sentence_embedding_dimension()))

    def _extract_topic_features(self, texts: List[str]) -> np.ndarray:
        """Extract topic modeling features"""
        try:
            if not self.topic_model or not texts:
                return np.array([])

            # Filter out empty texts
            valid_texts = [text for text in texts if text.strip()]

            if not valid_texts:
                return np.zeros((len(texts), self.config['topic_modeling']['n_topics']))

            # Vectorize texts
            if self.config['topic_modeling']['method'] == 'lda':
                text_vectors = self.count_vectorizer.fit_transform(valid_texts)
            else:
                text_vectors = self.tfidf_vectorizer.fit_transform(valid_texts)

            # Extract topics
            topic_features = self.topic_model.fit_transform(text_vectors)

            # Handle case where we have fewer valid texts
            if len(valid_texts) < len(texts):
                full_features = np.zeros((len(texts), topic_features.shape[1]))
                valid_indices = [i for i, text in enumerate(texts) if text.strip()]
                full_features[valid_indices] = topic_features
                return full_features

            return topic_features

        except Exception as e:
            self.logger.error(f"Error extracting topic features: {str(e)}")
            return np.zeros((len(texts), self.config['topic_modeling']['n_topics']))

    def assess_text_quality(self, texts: List[str]) -> Dict[str, Any]:
        """
        Assess quality of input texts

        Args:
            texts: List of texts to assess

        Returns:
            Dictionary with quality assessment results
        """
        try:
            quality_scores = []
            quality_details = []

            for text in texts:
                if not text or not isinstance(text, str):
                    quality_scores.append(0.0)
                    quality_details.append({'score': 0.0, 'issues': ['empty_text']})
                    continue

                score = 1.0
                issues = []

                # Length checks
                text_length = len(text)
                word_count = len(text.split())

                if text_length < self.quality_model['min_length']:
                    score -= 0.3
                    issues.append('too_short')
                elif text_length > self.quality_model['max_length']:
                    score -= 0.2
                    issues.append('too_long')

                if word_count < self.quality_model['min_words']:
                    score -= 0.2
                    issues.append('too_few_words')
                elif word_count > self.quality_model['max_words']:
                    score -= 0.1
                    issues.append('too_many_words')

                # Financial terminology bonus
                financial_term_count = sum(1 for term in self.financial_terms if term in text.lower())
                if financial_term_count > 0:
                    score += min(financial_term_count * self.quality_model['financial_term_bonus'], 0.2)

                # Diversity bonus
                words = text.split()
                unique_ratio = len(set(words)) / len(words) if words else 0
                if unique_ratio > 0.7:
                    score += self.quality_model['diversity_bonus']

                # Ensure score is between 0 and 1
                score = max(0.0, min(1.0, score))

                quality_scores.append(score)
                quality_details.append({
                    'score': score,
                    'issues': issues,
                    'length': text_length,
                    'word_count': word_count,
                    'financial_terms': financial_term_count,
                    'unique_ratio': unique_ratio
                })

            # Overall statistics
            avg_score = np.mean(quality_scores)
            quality_distribution = {
                'excellent': sum(1 for s in quality_scores if s >= 0.8),
                'good': sum(1 for s in quality_scores if 0.6 <= s < 0.8),
                'poor': sum(1 for s in quality_scores if s < 0.6)
            }

            return {
                'quality_scores': quality_scores,
                'quality_details': quality_details,
                'average_score': avg_score,
                'quality_distribution': quality_distribution,
                'texts_assessed': len(texts)
            }

        except Exception as e:
            self.logger.error(f"Error assessing text quality: {str(e)}")
            return {
                'quality_scores': [0.5] * len(texts),
                'average_score': 0.5,
                'error': str(e)
            }

    def _update_extraction_stats(self, features_dict: Dict[str, np.ndarray]):
        """Update extraction statistics"""
        try:
            for feature_type, features in features_dict.items():
                if hasattr(features, 'shape'):
                    self.extraction_stats[f'{feature_type}_extractions'] += 1
                    self.extraction_stats[f'{feature_type}_total_features'] += features.shape[1] if len(features.shape) > 1 else 1

        except Exception as e:
            self.logger.warning(f"Error updating extraction stats: {str(e)}")

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return dict(self.extraction_stats)

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics"""
        return self.quality_metrics.copy()

    def clear_cache(self):
        """Clear feature extraction cache"""
        self.feature_cache.clear()
        self.logger.info("Feature extraction cache cleared")

    def save_extractor(self, filepath: str):
        """Save the text feature extractor state"""
        try:
            import joblib

            save_dict = {
                'config': self.config,
                'extraction_stats': dict(self.extraction_stats),
                'quality_metrics': self.quality_metrics
            }

            joblib.dump(save_dict, filepath)
            self.logger.info(f"AdvancedTextFeatureExtractor saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving AdvancedTextFeatureExtractor: {str(e)}")

    def load_extractor(self, filepath: str):
        """Load the text feature extractor state"""
        try:
            import joblib

            save_dict = joblib.load(filepath)

            self.config = save_dict['config']
            self.extraction_stats = defaultdict(int, save_dict['extraction_stats'])
            self.quality_metrics = save_dict['quality_metrics']

            # Reinitialize components
            self._initialize_components()
            self._initialize_portuguese_components()
            self._initialize_ml_models()

            self.logger.info(f"AdvancedTextFeatureExtractor loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading AdvancedTextFeatureExtractor: {str(e)}")
            raise ValidationError(f"Failed to load AdvancedTextFeatureExtractor: {str(e)}")