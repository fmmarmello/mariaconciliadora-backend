import re
import random
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import nltk
from nltk.corpus import wordnet
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from src.services.portuguese_preprocessor import PortugueseTextPreprocessor
from .data_augmentation_pipeline import AugmentationStrategy

logger = get_logger(__name__)


class TextAugmentationEngine(AugmentationStrategy):
    """
    Advanced text augmentation engine with multiple strategies:
    - Synonym replacement using WordNet and domain-specific dictionaries
    - Back-translation through multiple languages
    - Paraphrasing using transformer models
    - Financial terminology preservation
    - Context-aware augmentation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text augmentation engine

        Args:
            config: Configuration for text augmentation
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize components
        self._initialize_components()

        # Financial terminology dictionary
        self.financial_terms = self._load_financial_terms()

        # Domain-specific synonym dictionary
        self.domain_synonyms = self._load_domain_synonyms()

        # Quality tracking
        self.augmentation_quality = defaultdict(float)

        self.logger.info("TextAugmentationEngine initialized")

    def _initialize_components(self):
        """Initialize text augmentation components"""
        try:
            # Download NLTK data if needed
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
                nltk.download('omw-1.4')

            # Initialize Portuguese text preprocessor
            self.text_preprocessor = PortugueseTextPreprocessor(use_advanced_pipeline=True)

            # Initialize back-translation models
            if 'back_translation' in self.config['strategies']:
                self._initialize_back_translation_models()

            # Initialize paraphrasing model
            if 'paraphrasing' in self.config['strategies']:
                self._initialize_paraphrasing_model()

            self.logger.info("Text augmentation components initialized")

        except Exception as e:
            self.logger.error(f"Error initializing text augmentation components: {str(e)}")
            raise ValidationError(f"Failed to initialize TextAugmentationEngine: {str(e)}")

    def _initialize_back_translation_models(self):
        """Initialize models for back-translation"""
        try:
            self.translation_models = {}

            # Portuguese to English
            self.translation_models['pt_en'] = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-pt-en",
                device=0 if torch.cuda.is_available() else -1
            )

            # English to Portuguese
            self.translation_models['en_pt'] = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-en-pt",
                device=0 if torch.cuda.is_available() else -1
            )

            # Add other language pairs as needed
            for lang in self.config.get('back_translation_config', {}).get('languages', []):
                if lang != 'en':
                    try:
                        model_name = f"Helsinki-NLP/opus-mt-pt-{lang}"
                        self.translation_models[f'pt_{lang}'] = pipeline(
                            "translation",
                            model=model_name,
                            device=0 if torch.cuda.is_available() else -1
                        )

                        model_name = f"Helsinki-NLP/opus-mt-{lang}-pt"
                        self.translation_models[f'{lang}_pt'] = pipeline(
                            "translation",
                            model=model_name,
                            device=0 if torch.cuda.is_available() else -1
                        )
                    except Exception as e:
                        self.logger.warning(f"Could not load translation model for {lang}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error initializing back-translation models: {str(e)}")

    def _initialize_paraphrasing_model(self):
        """Initialize paraphrasing model"""
        try:
            model_name = self.config.get('paraphrasing_config', {}).get('model_name', 'tuner007/pegasus_paraphrase')

            self.paraphrasing_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.paraphrasing_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.paraphrasing_model = self.paraphrasing_model.cuda()

            self.logger.info(f"Paraphrasing model loaded: {model_name}")

        except Exception as e:
            self.logger.error(f"Error initializing paraphrasing model: {str(e)}")
            self.paraphrasing_model = None

    def _load_financial_terms(self) -> Dict[str, List[str]]:
        """Load financial terminology dictionary"""
        return {
            'pagamento': ['pagamento', 'pago', 'quitado', 'liquidado'],
            'transferência': ['transferência', 'transferido', 'movimentação'],
            'débito': ['débito', 'debito', 'saída', 'despesa'],
            'crédito': ['crédito', 'credito', 'entrada', 'receita'],
            'saldo': ['saldo', 'balance', 'total'],
            'conta': ['conta', 'account', 'carteira'],
            'banco': ['banco', 'bank', 'instituição financeira'],
            'valor': ['valor', 'value', 'montante', 'quantia'],
            'data': ['data', 'date', 'dia'],
            'taxa': ['taxa', 'fee', 'juros', 'rate'],
            'boleto': ['boleto', 'bill', 'fatura'],
            'pix': ['pix', 'transferência instantânea', 'pagamento instantâneo'],
            'ted': ['ted', 'transferência bancária', 'tef'],
            'doc': ['doc', 'transferência documental']
        }

    def _load_domain_synonyms(self) -> Dict[str, List[str]]:
        """Load domain-specific synonym dictionary"""
        return {
            'mercado': ['mercado', 'supermercado', 'minimercado', 'mercearia'],
            'restaurante': ['restaurante', 'lanchonete', 'bar', 'cafeteria'],
            'posto': ['posto', 'posto de gasolina', 'combustível'],
            'farmácia': ['farmácia', 'drogaria', 'farmacia'],
            'shopping': ['shopping', 'centro comercial', 'mall'],
            'padaria': ['padaria', 'panificadora', 'bakery'],
            'academia': ['academia', 'ginásio', 'fitness'],
            'hospital': ['hospital', 'clínica', 'médico'],
            'escola': ['escola', 'colégio', 'educação'],
            'universidade': ['universidade', 'faculdade', 'ensino superior']
        }

    def augment(self, text: str, config: Dict[str, Any] = None) -> List[str]:
        """
        Apply text augmentation to a single text

        Args:
            text: Input text to augment
            config: Augmentation configuration

        Returns:
            List of augmented texts
        """
        if not text or not isinstance(text, str):
            return [text]

        try:
            augmented_texts = [text]  # Always include original

            # Apply different augmentation strategies
            strategies = config.get('strategies', self.config.get('strategies', []))

            for strategy in strategies:
                if strategy == 'synonym_replacement':
                    synonym_augmented = self._apply_synonym_replacement(text)
                    if synonym_augmented != text:
                        augmented_texts.append(synonym_augmented)

                elif strategy == 'back_translation':
                    back_translated = self._apply_back_translation(text)
                    if back_translated and back_translated != text:
                        augmented_texts.append(back_translated)

                elif strategy == 'paraphrasing':
                    paraphrased = self._apply_paraphrasing(text)
                    if paraphrased and paraphrased != text:
                        augmented_texts.extend(paraphrased)

            # Remove duplicates while preserving order
            seen = set()
            unique_augmented = []
            for aug_text in augmented_texts:
                if aug_text not in seen:
                    seen.add(aug_text)
                    unique_augmented.append(aug_text)

            return unique_augmented[:5]  # Limit to 5 variations

        except Exception as e:
            self.logger.error(f"Error augmenting text: {str(e)}")
            return [text]

    def augment_batch(self, texts: List[str], config: Dict[str, Any] = None) -> List[List[str]]:
        """
        Apply text augmentation to a batch of texts

        Args:
            texts: List of input texts
            config: Augmentation configuration

        Returns:
            List of lists containing augmented texts for each input
        """
        try:
            self.logger.info(f"Augmenting batch of {len(texts)} texts")

            augmented_batch = []
            for text in texts:
                augmented = self.augment(text, config)
                augmented_batch.append(augmented)

            self.logger.info("Batch text augmentation completed")
            return augmented_batch

        except Exception as e:
            self.logger.error(f"Error in batch text augmentation: {str(e)}")
            return [[text] for text in texts]

    def _apply_synonym_replacement(self, text: str) -> str:
        """Apply synonym replacement augmentation"""
        try:
            words = text.split()
            augmented_words = []

            replacement_rate = self.config.get('synonym_config', {}).get('replacement_rate', 0.3)

            for word in words:
                if random.random() < replacement_rate:
                    synonym = self._get_synonym(word.lower())
                    if synonym:
                        augmented_words.append(synonym)
                    else:
                        augmented_words.append(word)
                else:
                    augmented_words.append(word)

            return ' '.join(augmented_words)

        except Exception as e:
            self.logger.error(f"Error in synonym replacement: {str(e)}")
            return text

    def _get_synonym(self, word: str) -> Optional[str]:
        """Get synonym for a word using multiple sources"""
        try:
            # Check financial terms first
            if word in self.financial_terms:
                synonyms = self.financial_terms[word]
                return random.choice(synonyms) if synonyms else None

            # Check domain-specific synonyms
            if word in self.domain_synonyms:
                synonyms = self.domain_synonyms[word]
                return random.choice(synonyms) if synonyms else None

            # Use WordNet for general synonyms
            if self.config.get('synonym_config', {}).get('use_wordnet', True):
                synonyms = []
                for syn in wordnet.synsets(word, lang='por'):
                    for lemma in syn.lemmas(lang='por'):
                        if lemma.name() != word:
                            synonyms.append(lemma.name())

                if synonyms:
                    return random.choice(synonyms)

            return None

        except Exception as e:
            self.logger.error(f"Error getting synonym for {word}: {str(e)}")
            return None

    def _apply_back_translation(self, text: str) -> Optional[str]:
        """Apply back-translation augmentation"""
        try:
            if not hasattr(self, 'translation_models'):
                return None

            # Choose random intermediate language
            languages = self.config.get('back_translation_config', {}).get('languages', ['en'])
            intermediate_lang = random.choice(languages)

            # Translate to intermediate language
            if f'pt_{intermediate_lang}' in self.translation_models:
                translated = self.translation_models[f'pt_{intermediate_lang}'](
                    text, max_length=512
                )[0]['translation_text']

                # Translate back to Portuguese
                if f'{intermediate_lang}_pt' in self.translation_models:
                    back_translated = self.translation_models[f'{intermediate_lang}_pt'](
                        translated, max_length=512
                    )[0]['translation_text']

                    return back_translated

            return None

        except Exception as e:
            self.logger.error(f"Error in back-translation: {str(e)}")
            return None

    def _apply_paraphrasing(self, text: str) -> Optional[List[str]]:
        """Apply paraphrasing augmentation"""
        try:
            if not self.paraphrasing_model or not self.paraphrasing_tokenizer:
                return None

            # Prepare input
            inputs = self.paraphrasing_tokenizer(
                [text],
                truncation=True,
                padding=True,
                max_length=60,
                return_tensors="pt"
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate paraphrases
            config = self.config.get('paraphrasing_config', {})
            outputs = self.paraphrasing_model.generate(
                **inputs,
                max_length=config.get('max_length', 60),
                num_beams=config.get('num_beams', 5),
                num_return_sequences=config.get('num_return_sequences', 1),
                early_stopping=True
            )

            # Decode outputs
            paraphrases = []
            for output in outputs:
                paraphrase = self.paraphrasing_tokenizer.decode(
                    output, skip_special_tokens=True
                )
                if paraphrase and paraphrase != text:
                    paraphrases.append(paraphrase)

            return paraphrases[:3] if paraphrases else None  # Limit to 3 paraphrases

        except Exception as e:
            self.logger.error(f"Error in paraphrasing: {str(e)}")
            return None

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics for text augmentation"""
        return {
            'augmentation_quality': dict(self.augmentation_quality),
            'strategies_used': self.config.get('strategies', []),
            'financial_terms_coverage': len(self.financial_terms),
            'domain_synonyms_coverage': len(self.domain_synonyms)
        }

    def add_financial_term(self, term: str, synonyms: List[str]):
        """Add custom financial term with synonyms"""
        self.financial_terms[term.lower()] = [s.lower() for s in synonyms]

    def add_domain_synonym(self, term: str, synonyms: List[str]):
        """Add domain-specific synonym"""
        self.domain_synonyms[term.lower()] = [s.lower() for s in synonyms]