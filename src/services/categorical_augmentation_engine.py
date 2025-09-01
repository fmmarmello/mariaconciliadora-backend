import random
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Local imports
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError
from .data_augmentation_pipeline import AugmentationStrategy

logger = get_logger(__name__)


class CategoricalAugmentationEngine(AugmentationStrategy):
    """
    Categorical augmentation engine with:
    - Label-preserving transformations
    - Similar category mapping
    - Rare category handling
    - Business rule compliance
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the categorical augmentation engine

        Args:
            config: Configuration for categorical augmentation
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Category mappings and relationships
        self.category_mappings = {}
        self.similarity_matrix = {}
        self.business_rules = {}

        # Quality tracking
        self.augmentation_quality = {}

        # Initialize with default mappings
        self._initialize_default_mappings()

        self.logger.info("CategoricalAugmentationEngine initialized")

    def _initialize_default_mappings(self):
        """Initialize default category mappings for financial data"""
        try:
            # Transaction categories
            self.category_mappings = {
                'alimentação': ['restaurante', 'lanchonete', 'alimentacao', 'comida', 'refeicao'],
                'transporte': ['uber', 'taxi', 'onibus', 'metro', 'transporte', 'combustivel'],
                'saúde': ['farmacia', 'hospital', 'medico', 'saude', 'consulta'],
                'educação': ['escola', 'universidade', 'curso', 'livraria', 'educacao'],
                'lazer': ['cinema', 'teatro', 'show', 'lazer', 'entretenimento'],
                'compras': ['shopping', 'loja', 'compras', 'varejo'],
                'serviços': ['conta', 'luz', 'agua', 'telefone', 'internet', 'servicos'],
                'transferência': ['pix', 'ted', 'transferencia', 'pagamento'],
                'investimento': ['acoes', 'fundos', 'investimento', 'renda_fixa'],
                'imposto': ['imposto', 'taxa', 'multa', 'juros']
            }

            # Business rules for category preservation
            self.business_rules = {
                'bank_transfer': ['transferência', 'pix', 'ted'],
                'utility_bill': ['serviços', 'conta', 'luz', 'agua'],
                'salary': ['salário', 'recebimento', 'provento'],
                'investment': ['investimento', 'renda', 'dividendo']
            }

            self.logger.info("Default category mappings initialized")

        except Exception as e:
            self.logger.error(f"Error initializing default mappings: {str(e)}")

    def augment(self, data: Union[str, List[str]], config: Dict[str, Any] = None) -> List[str]:
        """
        Apply categorical augmentation to data

        Args:
            data: Input categorical data (single value or list)
            config: Augmentation configuration

        Returns:
            List of augmented categories
        """
        try:
            if isinstance(data, str):
                return self._augment_single_category(data, config)
            elif isinstance(data, list):
                return self._augment_category_list(data, config)
            else:
                return [str(data)]

        except Exception as e:
            self.logger.error(f"Error in categorical augmentation: {str(e)}")
            return [str(data)] if isinstance(data, str) else [str(x) for x in data]

    def augment_categorical(self, categories: List[str], config: Dict[str, Any] = None) -> List[List[str]]:
        """
        Apply categorical augmentation to a list of categories

        Args:
            categories: List of input categories
            config: Augmentation configuration

        Returns:
            List of lists containing augmented categories for each input
        """
        try:
            augmented_batch = []

            for category in categories:
                augmented = self.augment(category, config)
                augmented_batch.append(augmented)

            return augmented_batch

        except Exception as e:
            self.logger.error(f"Error in categorical batch augmentation: {str(e)}")
            return [[cat] for cat in categories]

    def _augment_single_category(self, category: str, config: Dict[str, Any] = None) -> List[str]:
        """Augment a single category"""
        try:
            if not category or not isinstance(category, str):
                return [category]

            augmented_categories = [category]  # Always include original

            # Apply different augmentation strategies
            strategies = config.get('strategies', self.config.get('strategies', []))

            for strategy in strategies:
                if strategy == 'label_preservation':
                    preserved = self._apply_label_preservation(category)
                    if preserved:
                        augmented_categories.extend(preserved)

                elif strategy == 'similar_category_mapping':
                    similar = self._apply_similar_category_mapping(category)
                    if similar:
                        augmented_categories.extend(similar)

            # Remove duplicates while preserving order
            seen = set()
            unique_augmented = []
            for aug_cat in augmented_categories:
                if aug_cat not in seen:
                    seen.add(aug_cat)
                    unique_augmented.append(aug_cat)

            return unique_augmented[:4]  # Limit to 4 variations

        except Exception as e:
            self.logger.error(f"Error augmenting single category: {str(e)}")
            return [category]

    def _augment_category_list(self, categories: List[str], config: Dict[str, Any] = None) -> List[str]:
        """Augment a list of categories"""
        try:
            all_augmented = []

            for category in categories:
                augmented = self._augment_single_category(category, config)
                all_augmented.extend(augmented)

            # Remove duplicates while preserving some variety
            unique_augmented = list(set(all_augmented))

            # Ensure we have at least the original categories
            for cat in categories:
                if cat not in unique_augmented:
                    unique_augmented.append(cat)

            return unique_augmented

        except Exception as e:
            self.logger.error(f"Error augmenting category list: {str(e)}")
            return categories

    def _apply_label_preservation(self, category: str) -> Optional[List[str]]:
        """Apply label preservation strategy"""
        try:
            preserved_categories = []

            # Check business rules first
            for rule_name, rule_categories in self.business_rules.items():
                if any(cat.lower() in category.lower() for cat in rule_categories):
                    # Apply rule-based preservation
                    preserved_categories.extend(rule_categories[:2])  # Limit to 2
                    break

            # Apply general preservation based on mappings
            category_lower = category.lower()
            for main_cat, variations in self.category_mappings.items():
                if category_lower in variations or main_cat in category_lower:
                    # Add some variations from the same group
                    preserved_categories.extend(variations[:3])
                    break

            return preserved_categories if preserved_categories else None

        except Exception as e:
            self.logger.error(f"Error in label preservation: {str(e)}")
            return None

    def _apply_similar_category_mapping(self, category: str) -> Optional[List[str]]:
        """Apply similar category mapping strategy"""
        try:
            similar_categories = []
            threshold = self.config.get('mapping_config', {}).get('similarity_threshold', 0.7)

            category_lower = category.lower()

            # Find similar categories based on string similarity
            for main_cat, variations in self.category_mappings.items():
                # Check similarity with main category
                similarity = self._calculate_string_similarity(category_lower, main_cat)
                if similarity >= threshold:
                    similar_categories.extend(variations[:2])
                    continue

                # Check similarity with variations
                for variation in variations:
                    similarity = self._calculate_string_similarity(category_lower, variation)
                    if similarity >= threshold:
                        similar_categories.append(main_cat)
                        similar_categories.extend([v for v in variations if v != variation][:2])
                        break

            return list(set(similar_categories)) if similar_categories else None

        except Exception as e:
            self.logger.error(f"Error in similar category mapping: {str(e)}")
            return None

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity"""
        try:
            # Simple Jaccard similarity based on character trigrams
            def get_trigrams(s):
                return set(s[i:i+3] for i in range(len(s)-2))

            trigrams1 = get_trigrams(str1.lower())
            trigrams2 = get_trigrams(str2.lower())

            if not trigrams1 or not trigrams2:
                return 0.0

            intersection = len(trigrams1 & trigrams2)
            union = len(trigrams1 | trigrams2)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics for categorical augmentation"""
        return {
            'augmentation_quality': self.augmentation_quality,
            'strategies_used': self.config.get('strategies', []),
            'category_mappings_count': len(self.category_mappings),
            'business_rules_count': len(self.business_rules),
            'rare_category_handling': self._calculate_rare_category_coverage()
        }

    def _calculate_rare_category_coverage(self) -> float:
        """Calculate coverage for rare categories"""
        try:
            # Placeholder implementation
            # In a real system, this would analyze actual category distributions
            return 0.75
        except Exception:
            return 0.0

    def add_category_mapping(self, main_category: str, variations: List[str]):
        """Add custom category mapping"""
        self.category_mappings[main_category.lower()] = [v.lower() for v in variations]

    def add_business_rule(self, rule_name: str, categories: List[str]):
        """Add business rule for category preservation"""
        self.business_rules[rule_name] = [c.lower() for c in categories]

    def handle_rare_categories(self, categories: List[str], min_frequency: int = 5) -> Dict[str, str]:
        """
        Handle rare categories by mapping them to similar common categories

        Args:
            categories: List of all categories
            min_frequency: Minimum frequency to be considered common

        Returns:
            Mapping from rare to common categories
        """
        try:
            # Count category frequencies
            category_counts = Counter(categories)

            # Identify rare and common categories
            rare_categories = {cat: count for cat, count in category_counts.items() if count < min_frequency}
            common_categories = {cat: count for cat, count in category_counts.items() if count >= min_frequency}

            # Create mapping for rare categories
            rare_mapping = {}

            for rare_cat in rare_categories:
                # Find most similar common category
                best_match = None
                best_similarity = 0.0

                for common_cat in common_categories:
                    similarity = self._calculate_string_similarity(rare_cat, common_cat)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = common_cat

                if best_match and best_similarity > 0.5:
                    rare_mapping[rare_cat] = best_match

            return rare_mapping

        except Exception as e:
            self.logger.error(f"Error handling rare categories: {str(e)}")
            return {}

    def generate_category_variations(self, base_category: str, n_variations: int = 3) -> List[str]:
        """
        Generate realistic variations of a category

        Args:
            base_category: Base category to vary
            n_variations: Number of variations to generate

        Returns:
            List of category variations
        """
        try:
            variations = [base_category]

            # Get variations from mappings
            base_lower = base_category.lower()
            for main_cat, cats in self.category_mappings.items():
                if base_lower in cats or main_cat == base_lower:
                    variations.extend(cats)
                    break

            # Generate additional variations through augmentation
            augmented = self._augment_single_category(base_category)
            variations.extend(augmented)

            # Remove duplicates and limit
            unique_variations = list(set(variations))
            return unique_variations[:n_variations + 1]  # +1 for original

        except Exception as e:
            self.logger.error(f"Error generating category variations: {str(e)}")
            return [base_category]