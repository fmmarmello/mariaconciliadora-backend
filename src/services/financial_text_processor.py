import re
import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
import logging

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FinancialTextProcessor:
    """
    Domain-specific text processor for Brazilian financial transaction data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the financial text processor

        Args:
            config: Configuration dictionary for financial processing
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize financial knowledge bases
        self._initialize_knowledge_bases()

        # Initialize patterns
        self._initialize_patterns()

        # Processing cache
        self._processing_cache = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'extract_monetary_values': True,
            'normalize_bank_names': True,
            'standardize_transaction_types': True,
            'extract_dates': True,
            'expand_abbreviations': True,
            'categorize_transactions': True,
            'confidence_threshold': 0.8,
            'cache_enabled': True
        }

    def _initialize_knowledge_bases(self):
        """Initialize financial knowledge bases"""
        # Brazilian bank names and codes
        self.banks = {
            # Major banks
            'itau': {'name': 'Itaú Unibanco', 'code': '341', 'aliases': ['itau', 'itaú', '341']},
            'bradesco': {'name': 'Bradesco', 'code': '237', 'aliases': ['bradesco', 'brad', '237']},
            'santander': {'name': 'Santander', 'code': '033', 'aliases': ['santander', 'sant', '033']},
            'banco_brasil': {'name': 'Banco do Brasil', 'code': '001', 'aliases': ['banco_brasil', 'bb', 'brasil', '001']},
            'caixa': {'name': 'Caixa Econômica Federal', 'code': '104', 'aliases': ['caixa', 'cef', '104']},
            'nubank': {'name': 'Nubank', 'code': '260', 'aliases': ['nubank', '260']},
            'inter': {'name': 'Banco Inter', 'code': '077', 'aliases': ['inter', '077']},
            'c6': {'name': 'C6 Bank', 'code': '336', 'aliases': ['c6', '336']},
            'original': {'name': 'Banco Original', 'code': '212', 'aliases': ['original', '212']},
            'btg': {'name': 'BTG Pactual', 'code': '208', 'aliases': ['btg', 'pactual', '208']},

            # Regional banks
            'banrisul': {'name': 'Banrisul', 'code': '041', 'aliases': ['banrisul', '041']},
            'banpara': {'name': 'Banpará', 'code': '037', 'aliases': ['banpara', '037']},
            'sicoob': {'name': 'Sicoob', 'code': '756', 'aliases': ['sicoob', '756']},
            'sicredi': {'name': 'Sicredi', 'code': '748', 'aliases': ['sicredi', '748']},
        }

        # Transaction types and their categories
        self.transaction_types = {
            # Payment types
            'pix': {'category': 'transferencia', 'description': 'Pagamento Instantâneo'},
            'ted': {'category': 'transferencia', 'description': 'Transferência Eletrônica Disponível'},
            'doc': {'category': 'transferencia', 'description': 'Documento de Ordem de Crédito'},
            'boleto': {'category': 'pagamento', 'description': 'Boleto Bancário'},
            'debito': {'category': 'pagamento', 'description': 'Débito'},
            'credito': {'category': 'pagamento', 'description': 'Crédito'},

            # Cash operations
            'saque': {'category': 'saque', 'description': 'Saque'},
            'deposito': {'category': 'deposito', 'description': 'Depósito'},
            'deposito_cheque': {'category': 'deposito', 'description': 'Depósito de Cheque'},

            # Service charges
            'tarifa': {'category': 'tarifa', 'description': 'Tarifa'},
            'juros': {'category': 'juros', 'description': 'Juros'},
            'multa': {'category': 'multa', 'description': 'Multa'},
            'iof': {'category': 'imposto', 'description': 'Imposto sobre Operações Financeiras'},

            # Investment operations
            'aplicacao': {'category': 'investimento', 'description': 'Aplicação'},
            'resgate': {'category': 'investimento', 'description': 'Resgate'},
            'dividendo': {'category': 'investimento', 'description': 'Dividendo'},
        }

        # Financial abbreviations
        self.abbreviations = {
            'r$': 'BRL',
            'rs': 'BRL',
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
            'cpf': 'cadastro_pessoa_fisica',
            'cnpj': 'cadastro_nacional_pessoa_juridica',
            'fgts': 'fundo_garantia_tempo_servico',
            'inss': 'instituto_nacional_seguro_social',
        }

    def _initialize_patterns(self):
        """Initialize regex patterns for financial text processing"""
        self.patterns = {
            # Monetary values - comprehensive patterns
            'monetary_values': [
                r'r\$\s*(\d{1,3}(?:\.\d{3})*,\d{2})',  # R$ 1.234,56
                r'r\$\s*(\d+,\d{2})',                   # R$ 123,45
                r'(\d{1,3}(?:\.\d{3})*,\d{2})\s*reais', # 1.234,56 reais
                r'(\d+,\d{2})\s*rs',                    # 123,45 rs
                r'valor\s*:\s*r\$\s*(\d{1,3}(?:\.\d{3})*,\d{2})',  # valor: R$ 1.234,56
            ],

            # Account numbers and agency
            'account_numbers': [
                r'ag\.?\s*(\d{4})',                    # ag. 1234
                r'agencia\s*(\d{4})',                  # agencia 1234
                r'cta\.?\s*(\d{5,12})',                # cta. 12345
                r'conta\s*(\d{5,12})',                 # conta 12345
                r'número\s*(\d{5,12})',                # número 12345
            ],

            # Document numbers
            'document_numbers': [
                r'cpf\s*(\d{3}\.\d{3}\.\d{3}-\d{2})',  # CPF 123.456.789-01
                r'cnpj\s*(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})',  # CNPJ 12.345.678/0001-90
            ],

            # Dates - multiple formats
            'dates': [
                r'(\d{1,2})/(\d{1,2})/(\d{4})',       # DD/MM/YYYY
                r'(\d{1,2})/(\d{1,2})/(\d{2})',        # DD/MM/YY
                r'(\d{4})-(\d{1,2})-(\d{1,2})',        # YYYY-MM-DD
                r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})',  # DD de Janeiro de YYYY
            ],

            # Transaction references
            'transaction_refs': [
                r'aut\s*(\d{6,12})',                   # aut 123456
                r'nsu\s*(\d{6,15})',                   # NSU 123456789
                r'ref\s*(\d{6,12})',                   # ref 123456
            ]
        }

    def process_financial_text(self, text: str) -> Dict[str, Any]:
        """
        Process financial text and extract structured information

        Args:
            text: Input financial text

        Returns:
            Dictionary with processed information
        """
        if not text or not isinstance(text, str):
            return self._create_empty_result()

        # Check cache
        cache_key = hash(text)
        if self.config['cache_enabled'] and cache_key in self._processing_cache:
            return self._processing_cache[cache_key]

        try:
            result = {
                'original_text': text,
                'normalized_text': text,
                'extracted_entities': {},
                'confidence_scores': {},
                'processing_steps': []
            }

            # Step 1: Normalize bank names
            if self.config['normalize_bank_names']:
                normalized_text, bank_info = self._normalize_bank_names(text)
                result['normalized_text'] = normalized_text
                result['extracted_entities']['banks'] = bank_info
                result['processing_steps'].append('bank_normalization')

            # Step 2: Extract monetary values
            if self.config['extract_monetary_values']:
                monetary_info = self._extract_monetary_values(result['normalized_text'])
                result['extracted_entities']['monetary_values'] = monetary_info
                result['processing_steps'].append('monetary_extraction')

            # Step 3: Standardize transaction types
            if self.config['standardize_transaction_types']:
                transaction_info = self._standardize_transaction_types(result['normalized_text'])
                result['extracted_entities']['transaction_types'] = transaction_info
                result['processing_steps'].append('transaction_standardization')

            # Step 4: Extract dates
            if self.config['extract_dates']:
                date_info = self._extract_dates(result['normalized_text'])
                result['extracted_entities']['dates'] = date_info
                result['processing_steps'].append('date_extraction')

            # Step 5: Expand abbreviations
            if self.config['expand_abbreviations']:
                expanded_text, abbreviation_info = self._expand_abbreviations(result['normalized_text'])
                result['normalized_text'] = expanded_text
                result['extracted_entities']['abbreviations'] = abbreviation_info
                result['processing_steps'].append('abbreviation_expansion')

            # Step 6: Categorize transaction
            if self.config['categorize_transactions']:
                category_info = self._categorize_transaction(result)
                result['extracted_entities']['categories'] = category_info
                result['processing_steps'].append('transaction_categorization')

            # Step 7: Calculate confidence scores
            result['confidence_scores'] = self._calculate_confidence_scores(result)

            # Cache result
            if self.config['cache_enabled']:
                self._processing_cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Error processing financial text: {str(e)}")
            return self._create_error_result(text, str(e))

    def _normalize_bank_names(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Normalize bank names in text"""
        normalized_text = text
        bank_info = []

        try:
            text_lower = text.lower()

            for bank_key, bank_data in self.banks.items():
                for alias in bank_data['aliases']:
                    if alias in text_lower:
                        # Replace alias with standardized name
                        normalized_text = re.sub(
                            r'\b' + re.escape(alias) + r'\b',
                            bank_data['name'],
                            normalized_text,
                            flags=re.IGNORECASE
                        )

                        bank_info.append({
                            'original': alias,
                            'normalized': bank_data['name'],
                            'code': bank_data['code'],
                            'confidence': 0.9
                        })
                        break

            return normalized_text, bank_info

        except Exception as e:
            self.logger.warning(f"Error normalizing bank names: {str(e)}")
            return text, []

    def _extract_monetary_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract monetary values from text"""
        monetary_values = []

        try:
            for pattern in self.patterns['monetary_values']:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Clean and standardize the value
                    clean_value = self._standardize_monetary_value(match)

                    if clean_value:
                        monetary_values.append({
                            'original': match,
                            'standardized': clean_value,
                            'currency': 'BRL',
                            'confidence': 0.95
                        })

            return monetary_values

        except Exception as e:
            self.logger.warning(f"Error extracting monetary values: {str(e)}")
            return []

    def _standardize_monetary_value(self, value: str) -> Optional[str]:
        """Standardize monetary value format"""
        try:
            # Remove currency symbols and extra spaces
            clean_value = re.sub(r'[r$\s]', '', value.lower())

            # Handle Brazilian number format (1.234,56)
            if ',' in clean_value:
                # Split into integer and decimal parts
                parts = clean_value.split(',')
                if len(parts) == 2:
                    integer_part = parts[0].replace('.', '')
                    decimal_part = parts[1][:2]  # Take only first 2 decimal digits
                    return f"{integer_part}.{decimal_part}"

            return None

        except Exception:
            return None

    def _standardize_transaction_types(self, text: str) -> List[Dict[str, Any]]:
        """Standardize transaction types"""
        transaction_info = []

        try:
            text_lower = text.lower()

            for tx_type, tx_data in self.transaction_types.items():
                if tx_type in text_lower:
                    transaction_info.append({
                        'original': tx_type,
                        'standardized': tx_data['category'],
                        'description': tx_data['description'],
                        'confidence': 0.85
                    })

            return transaction_info

        except Exception as e:
            self.logger.warning(f"Error standardizing transaction types: {str(e)}")
            return []

    def _extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract and normalize dates from text"""
        dates = []

        try:
            for pattern in self.patterns['dates']:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    parsed_date = self._parse_date(match)

                    if parsed_date:
                        dates.append({
                            'original': '/'.join(match) if isinstance(match, tuple) else match,
                            'parsed': parsed_date.isoformat(),
                            'confidence': 0.9
                        })

            return dates

        except Exception as e:
            self.logger.warning(f"Error extracting dates: {str(e)}")
            return []

    def _parse_date(self, date_match: Tuple[str, ...]) -> Optional[datetime.date]:
        """Parse date from regex match"""
        try:
            if len(date_match) == 3:
                day, month, year = date_match

                # Handle 2-digit years
                if len(year) == 2:
                    year = f"20{year}" if int(year) < 50 else f"19{year}"

                return datetime.date(int(year), int(month), int(day))

            return None

        except (ValueError, IndexError):
            return None

    def _expand_abbreviations(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Expand financial abbreviations"""
        expanded_text = text
        abbreviation_info = []

        try:
            words = text.split()
            expanded_words = []

            for word in words:
                word_lower = word.lower()
                if word_lower in self.abbreviations:
                    expanded = self.abbreviations[word_lower]
                    expanded_words.append(expanded)
                    abbreviation_info.append({
                        'original': word,
                        'expanded': expanded,
                        'confidence': 0.95
                    })
                else:
                    expanded_words.append(word)

            expanded_text = ' '.join(expanded_words)
            return expanded_text, abbreviation_info

        except Exception as e:
            self.logger.warning(f"Error expanding abbreviations: {str(e)}")
            return text, []

    def _categorize_transaction(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize transaction based on extracted information"""
        try:
            entities = result.get('extracted_entities', {})
            text = result.get('normalized_text', '').lower()

            category_info = {
                'primary_category': 'unknown',
                'subcategories': [],
                'confidence': 0.0,
                'indicators': []
            }

            # Analyze transaction types
            tx_types = entities.get('transaction_types', [])
            if tx_types:
                primary_tx = tx_types[0]  # Take the first one
                category_info['primary_category'] = primary_tx['standardized']
                category_info['confidence'] = primary_tx['confidence']
                category_info['indicators'].append(f"transaction_type:{primary_tx['original']}")

            # Analyze bank information
            banks = entities.get('banks', [])
            if banks:
                category_info['subcategories'].append('bank_operation')
                category_info['indicators'].append(f"bank:{banks[0]['normalized']}")

            # Analyze monetary values
            monetary = entities.get('monetary_values', [])
            if monetary:
                category_info['subcategories'].append('financial_transaction')
                category_info['indicators'].append('has_amount')

            # Rule-based categorization for common patterns
            if 'pix' in text:
                category_info['primary_category'] = 'transferencia'
                category_info['subcategories'].append('instant_payment')
                category_info['confidence'] = max(category_info['confidence'], 0.9)
            elif 'boleto' in text:
                category_info['primary_category'] = 'pagamento'
                category_info['subcategories'].append('bill_payment')
                category_info['confidence'] = max(category_info['confidence'], 0.85)
            elif 'saque' in text:
                category_info['primary_category'] = 'saque'
                category_info['subcategories'].append('cash_withdrawal')
                category_info['confidence'] = max(category_info['confidence'], 0.9)

            return category_info

        except Exception as e:
            self.logger.warning(f"Error categorizing transaction: {str(e)}")
            return {'primary_category': 'unknown', 'confidence': 0.0}

    def _calculate_confidence_scores(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for extracted information"""
        try:
            entities = result.get('extracted_entities', {})
            scores = {}

            # Bank confidence
            banks = entities.get('banks', [])
            scores['bank_confidence'] = sum(b['confidence'] for b in banks) / len(banks) if banks else 0.0

            # Monetary confidence
            monetary = entities.get('monetary_values', [])
            scores['monetary_confidence'] = sum(m['confidence'] for m in monetary) / len(monetary) if monetary else 0.0

            # Transaction confidence
            tx_types = entities.get('transaction_types', [])
            scores['transaction_confidence'] = sum(t['confidence'] for t in tx_types) / len(tx_types) if tx_types else 0.0

            # Overall confidence
            valid_scores = [s for s in scores.values() if s > 0]
            scores['overall_confidence'] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

            return scores

        except Exception as e:
            self.logger.warning(f"Error calculating confidence scores: {str(e)}")
            return {'overall_confidence': 0.0}

    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of financial texts

        Args:
            texts: List of financial texts

        Returns:
            List of processing results
        """
        if not texts:
            return []

        try:
            results = []
            for text in texts:
                result = self.process_financial_text(text)
                results.append(result)

            self.logger.info(f"Processed {len(texts)} financial texts")
            return results

        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            return [self._create_error_result(text, str(e)) for text in texts]

    def get_processing_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about financial text processing

        Args:
            results: List of processing results

        Returns:
            Dictionary with processing statistics
        """
        try:
            stats = {
                'total_texts': len(results),
                'banks_found': 0,
                'monetary_values_found': 0,
                'transaction_types_found': 0,
                'dates_found': 0,
                'average_confidence': 0.0,
                'processing_errors': 0
            }

            confidences = []

            for result in results:
                if 'error' in result:
                    stats['processing_errors'] += 1
                    continue

                entities = result.get('extracted_entities', {})

                stats['banks_found'] += len(entities.get('banks', []))
                stats['monetary_values_found'] += len(entities.get('monetary_values', []))
                stats['transaction_types_found'] += len(entities.get('transaction_types', []))
                stats['dates_found'] += len(entities.get('dates', []))

                confidence_scores = result.get('confidence_scores', {})
                overall_conf = confidence_scores.get('overall_confidence', 0.0)
                if overall_conf > 0:
                    confidences.append(overall_conf)

            if confidences:
                stats['average_confidence'] = sum(confidences) / len(confidences)

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating processing statistics: {str(e)}")
            return {}

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result for invalid input"""
        return {
            'original_text': '',
            'normalized_text': '',
            'extracted_entities': {},
            'confidence_scores': {},
            'processing_steps': []
        }

    def _create_error_result(self, text: str, error: str) -> Dict[str, Any]:
        """Create error result"""
        result = self._create_empty_result()
        result['original_text'] = text
        result['error'] = error
        return result

    def clear_cache(self):
        """Clear processing cache"""
        self._processing_cache.clear()
        self.logger.info("Financial processing cache cleared")