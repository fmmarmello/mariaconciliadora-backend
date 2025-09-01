import re
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import logging

from src.utils.logging_config import get_logger
from src.services.language_detector import LanguageDetector, DetectionResult
from src.services.multi_language_preprocessor import MultiLanguagePreprocessor, PreprocessingResult

logger = get_logger(__name__)


@dataclass
class CurrencyInfo:
    """Currency information"""
    code: str
    name: str
    symbol: str
    countries: List[str]
    subunit: str
    subunit_ratio: int


@dataclass
class BankInfo:
    """International bank information"""
    name: str
    code: str
    country: str
    aliases: List[str]
    swift_code: Optional[str] = None
    regulatory_body: Optional[str] = None


@dataclass
class TransactionPattern:
    """Cross-border transaction pattern"""
    pattern_type: str
    description: str
    risk_level: str
    indicators: List[str]
    jurisdictions: List[str]


@dataclass
class ComplianceInfo:
    """Regulatory compliance information"""
    jurisdiction: str
    requirements: List[str]
    restrictions: List[str]
    reporting_thresholds: Dict[str, float]
    sanctions_lists: List[str]


@dataclass
class FinancialProcessingResult:
    """Result of international financial processing"""
    original_text: str
    processed_text: str
    detected_language: str
    currencies: List[Dict[str, Any]]
    banks: List[Dict[str, Any]]
    transaction_patterns: List[Dict[str, Any]]
    compliance_flags: List[Dict[str, Any]]
    tax_indicators: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    jurisdictions: List[str]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class InternationalFinancialProcessor:
    """
    International financial processor for global financial document analysis
    with multi-currency support, compliance checking, and cross-border analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the international financial processor

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize language detector and preprocessor
        self.language_detector = LanguageDetector()
        self.preprocessor = MultiLanguagePreprocessor()

        # Initialize financial knowledge bases
        self._currencies = self._initialize_currencies()
        self._banks = self._initialize_banks()
        self._transaction_patterns = self._initialize_transaction_patterns()
        self._compliance_rules = self._initialize_compliance_rules()
        self._tax_terms = self._initialize_tax_terms()

        # Processing cache
        self._processing_cache = {}

        # Performance tracking
        self._performance_stats = {
            'total_processed': 0,
            'currencies_found': 0,
            'banks_found': 0,
            'compliance_flags': 0,
            'risk_assessments': 0,
            'average_processing_time': 0.0
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'cache_enabled': True,
            'enable_compliance_checking': True,
            'enable_risk_assessment': True,
            'enable_tax_analysis': True,
            'batch_size': 32,
            'timeout_seconds': 30.0,
            'risk_threshold': 0.7,
            'compliance_threshold': 0.8
        }

    def _initialize_currencies(self) -> Dict[str, CurrencyInfo]:
        """Initialize currency information database"""
        currencies = {}

        # Major world currencies
        currencies['USD'] = CurrencyInfo(
            code='USD',
            name='US Dollar',
            symbol='$',
            countries=['US', 'EC', 'PR', 'VI'],
            subunit='cent',
            subunit_ratio=100
        )

        currencies['EUR'] = CurrencyInfo(
            code='EUR',
            name='Euro',
            symbol='€',
            countries=['AT', 'BE', 'CY', 'EE', 'FI', 'FR', 'DE', 'GR', 'IE', 'IT',
                      'LV', 'LT', 'LU', 'MT', 'NL', 'PT', 'SK', 'SI', 'ES'],
            subunit='cent',
            subunit_ratio=100
        )

        currencies['BRL'] = CurrencyInfo(
            code='BRL',
            name='Brazilian Real',
            symbol='R$',
            countries=['BR'],
            subunit='centavo',
            subunit_ratio=100
        )

        currencies['GBP'] = CurrencyInfo(
            code='GBP',
            name='British Pound',
            symbol='£',
            countries=['GB', 'GG', 'IM', 'JE'],
            subunit='pence',
            subunit_ratio=100
        )

        currencies['JPY'] = CurrencyInfo(
            code='JPY',
            name='Japanese Yen',
            symbol='¥',
            countries=['JP'],
            subunit='sen',
            subunit_ratio=100
        )

        currencies['CHF'] = CurrencyInfo(
            code='CHF',
            name='Swiss Franc',
            symbol='CHF',
            countries=['CH', 'LI'],
            subunit='rappen',
            subunit_ratio=100
        )

        currencies['CAD'] = CurrencyInfo(
            code='CAD',
            name='Canadian Dollar',
            symbol='C$',
            countries=['CA'],
            subunit='cent',
            subunit_ratio=100
        )

        currencies['AUD'] = CurrencyInfo(
            code='AUD',
            name='Australian Dollar',
            symbol='A$',
            countries=['AU', 'CX', 'CC', 'HM', 'KI', 'NR', 'NF', 'TV'],
            subunit='cent',
            subunit_ratio=100
        )

        currencies['CNY'] = CurrencyInfo(
            code='CNY',
            name='Chinese Yuan',
            symbol='¥',
            countries=['CN'],
            subunit='fen',
            subunit_ratio=100
        )

        currencies['INR'] = CurrencyInfo(
            code='INR',
            name='Indian Rupee',
            symbol='₹',
            countries=['IN'],
            subunit='paisa',
            subunit_ratio=100
        )

        currencies['MXN'] = CurrencyInfo(
            code='MXN',
            name='Mexican Peso',
            symbol='$',
            countries=['MX'],
            subunit='centavo',
            subunit_ratio=100
        )

        currencies['ARS'] = CurrencyInfo(
            code='ARS',
            name='Argentine Peso',
            symbol='$',
            countries=['AR'],
            subunit='centavo',
            subunit_ratio=100
        )

        currencies['CLP'] = CurrencyInfo(
            code='CLP',
            name='Chilean Peso',
            symbol='$',
            countries=['CL'],
            subunit='peso',
            subunit_ratio=100
        )

        currencies['COP'] = CurrencyInfo(
            code='COP',
            name='Colombian Peso',
            symbol='$',
            countries=['CO'],
            subunit='centavo',
            subunit_ratio=100
        )

        currencies['PEN'] = CurrencyInfo(
            code='PEN',
            name='Peruvian Sol',
            symbol='S/',
            countries=['PE'],
            subunit='céntimo',
            subunit_ratio=100
        )

        return currencies

    def _initialize_banks(self) -> Dict[str, BankInfo]:
        """Initialize international bank information database"""
        banks = {}

        # Brazilian banks
        banks['itau'] = BankInfo(
            name='Itaú Unibanco',
            code='341',
            country='BR',
            aliases=['itau', 'itaú', '341', 'itau unibanco'],
            swift_code='ITAU',
            regulatory_body='BACEN'
        )

        banks['bradesco'] = BankInfo(
            name='Bradesco',
            code='237',
            country='BR',
            aliases=['bradesco', '237', 'banco bradesco'],
            swift_code='BBDE',
            regulatory_body='BACEN'
        )

        banks['santander'] = BankInfo(
            name='Santander Brasil',
            code='033',
            country='BR',
            aliases=['santander', '033', 'banco santander'],
            swift_code='BSCH',
            regulatory_body='BACEN'
        )

        banks['banco_brasil'] = BankInfo(
            name='Banco do Brasil',
            code='001',
            country='BR',
            aliases=['banco_brasil', 'bb', 'brasil', '001', 'banco do brasil'],
            swift_code='BRAS',
            regulatory_body='BACEN'
        )

        banks['caixa'] = BankInfo(
            name='Caixa Econômica Federal',
            code='104',
            country='BR',
            aliases=['caixa', 'cef', '104', 'caixa economica'],
            swift_code='CEFE',
            regulatory_body='BACEN'
        )

        # International banks
        banks['jpmorgan'] = BankInfo(
            name='JPMorgan Chase',
            code='',
            country='US',
            aliases=['jpmorgan', 'jp morgan', 'chase', 'jpm'],
            swift_code='CHAS',
            regulatory_body='FED'
        )

        banks['citibank'] = BankInfo(
            name='Citibank',
            code='',
            country='US',
            aliases=['citibank', 'citi', 'citigroup'],
            swift_code='CITI',
            regulatory_body='FED'
        )

        banks['hsbc'] = BankInfo(
            name='HSBC',
            code='',
            country='GB',
            aliases=['hsbc', 'hongkong shanghai'],
            swift_code='MIDL',
            regulatory_body='FCA'
        )

        banks['deutsche_bank'] = BankInfo(
            name='Deutsche Bank',
            code='',
            country='DE',
            aliases=['deutsche bank', 'deutsche', 'db'],
            swift_code='DEUT',
            regulatory_body='BaFin'
        )

        banks['bnp_paribas'] = BankInfo(
            name='BNP Paribas',
            code='',
            country='FR',
            aliases=['bnp paribas', 'bnp', 'paribas'],
            swift_code='BNPA',
            regulatory_body='ACPR'
        )

        banks['ubs'] = BankInfo(
            name='UBS',
            code='',
            country='CH',
            aliases=['ubs', 'union bank of switzerland'],
            swift_code='UBSW',
            regulatory_body='FINMA'
        )

        banks['barclays'] = BankInfo(
            name='Barclays',
            code='',
            country='GB',
            aliases=['barclays', 'barclay'],
            swift_code='BARC',
            regulatory_body='FCA'
        )

        banks['credit_suisse'] = BankInfo(
            name='Credit Suisse',
            code='',
            country='CH',
            aliases=['credit suisse', 'credit', 'suisse'],
            swift_code='CRES',
            regulatory_body='FINMA'
        )

        return banks

    def _initialize_transaction_patterns(self) -> List[TransactionPattern]:
        """Initialize cross-border transaction patterns"""
        patterns = []

        # High-risk patterns
        patterns.append(TransactionPattern(
            pattern_type='round_dollar_transfers',
            description='Large round dollar transfers',
            risk_level='high',
            indicators=['transfer', '100000', '500000', '1000000'],
            jurisdictions=['ALL']
        ))

        patterns.append(TransactionPattern(
            pattern_type='frequent_small_transfers',
            description='Frequent small transfers to same destination',
            risk_level='medium',
            indicators=['transfer', 'pix', 'multiple', 'same recipient'],
            jurisdictions=['BR', 'US', 'EU']
        ))

        patterns.append(TransactionPattern(
            pattern_type='sanctions_evasion',
            description='Transactions involving sanctioned entities',
            risk_level='critical',
            indicators=['iran', 'north korea', 'cuba', 'syria', 'venezuela'],
            jurisdictions=['ALL']
        ))

        patterns.append(TransactionPattern(
            pattern_type='tax_haven_transfers',
            description='Transfers to known tax havens',
            risk_level='high',
            indicators=['cayman', 'bermuda', 'panama', 'switzerland', 'luxembourg'],
            jurisdictions=['ALL']
        ))

        # Medium-risk patterns
        patterns.append(TransactionPattern(
            pattern_type='international_wire_transfers',
            description='International wire transfers above threshold',
            risk_level='medium',
            indicators=['swift', 'wire transfer', 'international'],
            jurisdictions=['ALL']
        ))

        patterns.append(TransactionPattern(
            pattern_type='cash_deposits',
            description='Large cash deposits',
            risk_level='medium',
            indicators=['cash deposit', 'dinheiro vivo', 'efectivo'],
            jurisdictions=['BR', 'US', 'EU']
        ))

        # Low-risk patterns
        patterns.append(TransactionPattern(
            pattern_type='domestic_transfers',
            description='Domestic transfers within same country',
            risk_level='low',
            indicators=['domestic', 'internal', 'same country'],
            jurisdictions=['ALL']
        ))

        return patterns

    def _initialize_compliance_rules(self) -> Dict[str, ComplianceInfo]:
        """Initialize regulatory compliance rules"""
        compliance = {}

        # Brazil
        compliance['BR'] = ComplianceInfo(
            jurisdiction='Brazil',
            requirements=[
                'BACEN registration for international transfers',
                'CPF/CNPJ validation',
                'Anti-money laundering compliance',
                'Foreign exchange regulations'
            ],
            restrictions=[
                'No transactions with sanctioned countries',
                'Foreign exchange limits for individuals',
                'Reporting requirements for large transactions'
            ],
            reporting_thresholds={
                'international_transfer': 10000.0,
                'foreign_exchange': 30000.0,
                'suspicious_activity': 50000.0
            },
            sanctions_lists=['OFAC', 'EU Sanctions', 'UN Sanctions']
        )

        # United States
        compliance['US'] = ComplianceInfo(
            jurisdiction='United States',
            requirements=[
                'OFAC compliance',
                'BSA/AML compliance',
                'FinCEN reporting',
                'CTF compliance'
            ],
            restrictions=[
                'No transactions with OFAC sanctioned entities',
                'Enhanced due diligence for high-risk customers',
                'Transaction monitoring requirements'
            ],
            reporting_thresholds={
                'cash_transaction': 10000.0,
                'international_transfer': 10000.0,
                'suspicious_activity': 5000.0
            },
            sanctions_lists=['OFAC', 'EU Sanctions', 'UN Sanctions']
        )

        # European Union
        compliance['EU'] = ComplianceInfo(
            jurisdiction='European Union',
            requirements=[
                'AML Directive compliance',
                'GDPR compliance',
                'MiFID II compliance',
                'PSD2 compliance'
            ],
            restrictions=[
                'No transactions with EU sanctioned entities',
                'Enhanced customer due diligence',
                'Transaction reporting requirements'
            ],
            reporting_thresholds={
                'suspicious_activity': 15000.0,
                'international_transfer': 10000.0,
                'cash_transaction': 10000.0
            },
            sanctions_lists=['EU Sanctions', 'OFAC', 'UN Sanctions']
        )

        return compliance

    def _initialize_tax_terms(self) -> Dict[str, List[str]]:
        """Initialize tax-related terms in multiple languages"""
        return {
            'pt': [
                'imposto', 'tributo', 'taxa', 'alíquota', 'dedução', 'crédito fiscal',
                'declaração', 'irpf', 'irpj', 'csll', 'pis', 'cofins', 'iss', 'icms',
                'ipi', 'iof', 'fgts', 'inss', 'previdência', 'sonegação', 'evasão'
            ],
            'en': [
                'tax', 'taxes', 'duty', 'rate', 'deduction', 'tax credit',
                'declaration', 'income tax', 'corporate tax', 'social security',
                'value added tax', 'sales tax', 'excise tax', 'withholding tax',
                'tax evasion', 'tax avoidance', 'irs', 'hmrc', 'irs audit'
            ],
            'es': [
                'impuesto', 'tributo', 'tasa', 'alícuota', 'deducción', 'crédito fiscal',
                'declaración', 'irpf', 'iva', 'impuesto sociedades', 'seguridad social',
                'impuesto ventas', 'retención', 'evasión fiscal', 'fraude fiscal'
            ],
            'fr': [
                'impôt', 'taxe', 'contribution', 'déduction', 'crédit d\'impôt',
                'déclaration', 'ir', 'tva', 'impôt sociétés', 'sécurité sociale',
                'retenue', 'évasion fiscale', 'fraude fiscale'
            ],
            'de': [
                'steuer', 'abgabe', 'steuersatz', 'abzug', 'steuergutschrift',
                'erklärung', 'einkommensteuer', 'körperschaftsteuer', 'umsatzsteuer',
                'sozialversicherung', 'quellensteuer', 'steuerhinterziehung'
            ],
            'it': [
                'tassa', 'imposta', 'aliquota', 'deduzione', 'credito d\'imposta',
                'dichiarazione', 'irpef', 'iva', 'imposta società', 'previdenza sociale',
                'ritenuta', 'evasione fiscale', 'frode fiscale'
            ]
        }

    def process_financial_text(self, text: str, detected_language: Optional[str] = None) -> FinancialProcessingResult:
        """
        Process financial text with international compliance and risk analysis

        Args:
            text: Input financial text
            detected_language: Pre-detected language (optional)

        Returns:
            FinancialProcessingResult with comprehensive international analysis
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
            result = FinancialProcessingResult(
                original_text=text,
                processed_text=text,
                detected_language='unknown',
                currencies=[],
                banks=[],
                transaction_patterns=[],
                compliance_flags=[],
                tax_indicators=[],
                risk_assessment={},
                jurisdictions=[],
                processing_time=0.0,
                success=True
            )

            # Step 1: Language detection and preprocessing
            if detected_language:
                result.detected_language = detected_language
            else:
                detection_result = self.language_detector.detect_language(text)
                result.detected_language = detection_result.consensus_language

            preprocessing_result = self.preprocessor.preprocess_text(text, result.detected_language)
            result.processed_text = preprocessing_result.processed_text

            # Step 2: Currency recognition and conversion
            currencies = self._extract_currencies(result.processed_text, result.detected_language)
            result.currencies = currencies

            # Step 3: Bank name recognition
            banks = self._extract_banks(result.processed_text, result.detected_language)
            result.banks = banks

            # Step 4: Transaction pattern recognition
            patterns = self._identify_transaction_patterns(result.processed_text, result.detected_language)
            result.transaction_patterns = patterns

            # Step 5: Tax term recognition
            if self.config['enable_tax_analysis']:
                tax_indicators = self._extract_tax_indicators(result.processed_text, result.detected_language)
                result.tax_indicators = tax_indicators

            # Step 6: Jurisdiction identification
            jurisdictions = self._identify_jurisdictions(result)
            result.jurisdictions = jurisdictions

            # Step 7: Compliance checking
            if self.config['enable_compliance_checking']:
                compliance_flags = self._check_compliance(result)
                result.compliance_flags = compliance_flags

            # Step 8: Risk assessment
            if self.config['enable_risk_assessment']:
                risk_assessment = self._assess_risk(result)
                result.risk_assessment = risk_assessment

            # Calculate processing time
            result.processing_time = time.time() - start_time

            # Update performance stats
            self._update_performance_stats(result)

            # Cache result
            if self.config['cache_enabled']:
                self._processing_cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Error in international financial processing: {str(e)}")
            return self._create_error_result(text, str(e))

    def _extract_currencies(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extract and analyze currency information"""
        try:
            currencies = []
            text_lower = text.lower()

            # Check for currency symbols and codes
            for currency_code, currency_info in self._currencies.items():
                # Check for currency symbol
                if currency_info.symbol in text:
                    currencies.append({
                        'currency': currency_code,
                        'name': currency_info.name,
                        'symbol': currency_info.symbol,
                        'countries': currency_info.countries,
                        'confidence': 0.9
                    })

                # Check for currency code
                if currency_code.lower() in text_lower:
                    currencies.append({
                        'currency': currency_code,
                        'name': currency_info.name,
                        'code': currency_code,
                        'countries': currency_info.countries,
                        'confidence': 0.95
                    })

                # Check for currency name
                if currency_info.name.lower() in text_lower:
                    currencies.append({
                        'currency': currency_code,
                        'name': currency_info.name,
                        'countries': currency_info.countries,
                        'confidence': 0.85
                    })

            # Remove duplicates
            seen = set()
            unique_currencies = []
            for currency in currencies:
                key = currency['currency']
                if key not in seen:
                    seen.add(key)
                    unique_currencies.append(currency)

            return unique_currencies

        except Exception as e:
            self.logger.warning(f"Error extracting currencies: {str(e)}")
            return []

    def _extract_banks(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extract and analyze bank information"""
        try:
            banks = []
            text_lower = text.lower()

            for bank_key, bank_info in self._banks.items():
                found = False

                # Check aliases
                for alias in bank_info.aliases:
                    if alias.lower() in text_lower:
                        banks.append({
                            'name': bank_info.name,
                            'code': bank_info.code,
                            'country': bank_info.country,
                            'swift_code': bank_info.swift_code,
                            'regulatory_body': bank_info.regulatory_body,
                            'matched_alias': alias,
                            'confidence': 0.9
                        })
                        found = True
                        break

                if found:
                    continue

                # Check for SWIFT code
                if bank_info.swift_code and bank_info.swift_code.lower() in text_lower:
                    banks.append({
                        'name': bank_info.name,
                        'code': bank_info.code,
                        'country': bank_info.country,
                        'swift_code': bank_info.swift_code,
                        'regulatory_body': bank_info.regulatory_body,
                        'matched_swift': bank_info.swift_code,
                        'confidence': 0.95
                    })

            return banks

        except Exception as e:
            self.logger.warning(f"Error extracting banks: {str(e)}")
            return []

    def _identify_transaction_patterns(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Identify cross-border transaction patterns"""
        try:
            patterns = []
            text_lower = text.lower()

            for pattern in self._transaction_patterns:
                match_count = 0
                matched_indicators = []

                for indicator in pattern.indicators:
                    if indicator.lower() in text_lower:
                        match_count += 1
                        matched_indicators.append(indicator)

                if match_count >= 2:  # At least 2 indicators match
                    confidence = min(match_count / len(pattern.indicators), 1.0)
                    patterns.append({
                        'pattern_type': pattern.pattern_type,
                        'description': pattern.description,
                        'risk_level': pattern.risk_level,
                        'matched_indicators': matched_indicators,
                        'jurisdictions': pattern.jurisdictions,
                        'confidence': confidence
                    })

            return patterns

        except Exception as e:
            self.logger.warning(f"Error identifying transaction patterns: {str(e)}")
            return []

    def _extract_tax_indicators(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extract tax-related indicators"""
        try:
            tax_indicators = []
            text_lower = text.lower()

            tax_terms = self._tax_terms.get(language, [])
            for term in tax_terms:
                if term in text_lower:
                    tax_indicators.append({
                        'term': term,
                        'language': language,
                        'confidence': 0.9
                    })

            return tax_indicators

        except Exception as e:
            self.logger.warning(f"Error extracting tax indicators: {str(e)}")
            return []

    def _identify_jurisdictions(self, result: FinancialProcessingResult) -> List[str]:
        """Identify relevant jurisdictions based on extracted information"""
        try:
            jurisdictions = set()

            # From currencies
            for currency in result.currencies:
                jurisdictions.update(currency.get('countries', []))

            # From banks
            for bank in result.banks:
                jurisdictions.add(bank.get('country', ''))

            # From transaction patterns
            for pattern in result.transaction_patterns:
                jurisdictions.update(pattern.get('jurisdictions', []))

            # Remove empty strings and 'ALL'
            jurisdictions = {j for j in jurisdictions if j and j != 'ALL'}

            return list(jurisdictions)

        except Exception as e:
            self.logger.warning(f"Error identifying jurisdictions: {str(e)}")
            return []

    def _check_compliance(self, result: FinancialProcessingResult) -> List[Dict[str, Any]]:
        """Check regulatory compliance requirements"""
        try:
            compliance_flags = []

            for jurisdiction in result.jurisdictions:
                if jurisdiction in self._compliance_rules:
                    rules = self._compliance_rules[jurisdiction]

                    # Check for sanctions violations
                    sanctions_violations = self._check_sanctions_compliance(result, rules)
                    compliance_flags.extend(sanctions_violations)

                    # Check reporting thresholds
                    threshold_violations = self._check_reporting_thresholds(result, rules)
                    compliance_flags.extend(threshold_violations)

            return compliance_flags

        except Exception as e:
            self.logger.warning(f"Error checking compliance: {str(e)}")
            return []

    def _check_sanctions_compliance(self, result: FinancialProcessingResult, rules: ComplianceInfo) -> List[Dict[str, Any]]:
        """Check for sanctions compliance violations"""
        try:
            violations = []
            text_lower = result.processed_text.lower()

            for sanctions_list in rules.sanctions_lists:
                # Simplified sanctions checking - in real implementation,
                # this would query actual sanctions databases
                if 'iran' in text_lower or 'north korea' in text_lower:
                    violations.append({
                        'type': 'sanctions_violation',
                        'list': sanctions_list,
                        'severity': 'critical',
                        'description': f'Potential sanctions violation in {sanctions_list}',
                        'confidence': 0.8
                    })

            return violations

        except Exception as e:
            self.logger.warning(f"Error checking sanctions compliance: {str(e)}")
            return []

    def _check_reporting_thresholds(self, result: FinancialProcessingResult, rules: ComplianceInfo) -> List[Dict[str, Any]]:
        """Check for reporting threshold violations"""
        try:
            violations = []

            # Extract monetary amounts (simplified)
            amounts = self._extract_monetary_amounts(result.processed_text)

            for amount_info in amounts:
                amount = amount_info.get('amount', 0.0)
                currency = amount_info.get('currency', 'USD')

                # Convert to USD for comparison (simplified conversion)
                usd_amount = self._convert_to_usd(amount, currency)

                # Check against thresholds
                for threshold_type, threshold_value in rules.reporting_thresholds.items():
                    if usd_amount >= threshold_value:
                        violations.append({
                            'type': 'reporting_threshold',
                            'threshold_type': threshold_type,
                            'threshold_value': threshold_value,
                            'transaction_amount': usd_amount,
                            'currency': currency,
                            'jurisdiction': rules.jurisdiction,
                            'confidence': 0.9
                        })

            return violations

        except Exception as e:
            self.logger.warning(f"Error checking reporting thresholds: {str(e)}")
            return []

    def _extract_monetary_amounts(self, text: str) -> List[Dict[str, Any]]:
        """Extract monetary amounts from text"""
        try:
            amounts = []

            # Simple regex patterns for amounts
            patterns = [
                r'\$[\d,]+\.?\d*',  # $1,234.56
                r'€[\d,]+\.?\d*',   # €1.234,56
                r'R\$[\d,]+\.?\d*', # R$1.234,56
                r'£[\d,]+\.?\d*',   # £1,234.56
                r'[\d,]+\.?\d*\s*(?:USD|EUR|BRL|GBP)',  # 1234.56 USD
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    # Extract numeric value
                    numeric_match = re.search(r'[\d,]+\.?\d*', match)
                    if numeric_match:
                        amount_str = numeric_match.group()
                        amount = float(amount_str.replace(',', ''))

                        # Determine currency
                        currency = 'USD'  # default
                        if '$' in match or 'USD' in match.upper():
                            currency = 'USD'
                        elif '€' in match or 'EUR' in match.upper():
                            currency = 'EUR'
                        elif 'R$' in match or 'BRL' in match.upper():
                            currency = 'BRL'
                        elif '£' in match or 'GBP' in match.upper():
                            currency = 'GBP'

                        amounts.append({
                            'amount': amount,
                            'currency': currency,
                            'original': match
                        })

            return amounts

        except Exception as e:
            self.logger.warning(f"Error extracting monetary amounts: {str(e)}")
            return []

    def _convert_to_usd(self, amount: float, currency: str) -> float:
        """Convert amount to USD (simplified conversion rates)"""
        # Simplified conversion rates (in real implementation, use current rates)
        rates = {
            'USD': 1.0,
            'EUR': 1.1,
            'BRL': 0.2,
            'GBP': 1.3,
            'JPY': 0.007,
            'CAD': 0.75,
            'AUD': 0.68
        }

        return amount * rates.get(currency, 1.0)

    def _assess_risk(self, result: FinancialProcessingResult) -> Dict[str, Any]:
        """Assess overall risk of the financial transaction"""
        try:
            risk_score = 0.0
            risk_factors = []

            # Risk from transaction patterns
            for pattern in result.transaction_patterns:
                risk_level = pattern.get('risk_level', 'low')
                confidence = pattern.get('confidence', 0.0)

                if risk_level == 'critical':
                    risk_score += confidence * 1.0
                    risk_factors.append(f"Critical pattern: {pattern.get('description', '')}")
                elif risk_level == 'high':
                    risk_score += confidence * 0.7
                    risk_factors.append(f"High-risk pattern: {pattern.get('description', '')}")
                elif risk_level == 'medium':
                    risk_score += confidence * 0.4
                    risk_factors.append(f"Medium-risk pattern: {pattern.get('description', '')}")

            # Risk from compliance flags
            for flag in result.compliance_flags:
                severity = flag.get('severity', 'low')
                confidence = flag.get('confidence', 0.0)

                if severity == 'critical':
                    risk_score += confidence * 1.0
                    risk_factors.append(f"Critical compliance issue: {flag.get('description', '')}")
                elif severity == 'high':
                    risk_score += confidence * 0.8
                    risk_factors.append(f"High compliance issue: {flag.get('description', '')}")
                elif severity == 'medium':
                    risk_score += confidence * 0.5
                    risk_factors.append(f"Medium compliance issue: {flag.get('description', '')}")

            # Risk from multiple jurisdictions
            if len(result.jurisdictions) > 2:
                risk_score += 0.3
                risk_factors.append("Multiple jurisdictions involved")

            # Risk from tax indicators
            if result.tax_indicators:
                risk_score += 0.2
                risk_factors.append("Tax-related terms detected")

            # Normalize risk score
            risk_score = min(risk_score, 1.0)

            # Determine risk level
            if risk_score >= 0.8:
                risk_level = 'critical'
            elif risk_score >= 0.6:
                risk_level = 'high'
            elif risk_score >= 0.4:
                risk_level = 'medium'
            elif risk_score >= 0.2:
                risk_level = 'low'
            else:
                risk_level = 'minimal'

            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'recommendations': self._generate_risk_recommendations(risk_level, risk_factors)
            }

        except Exception as e:
            self.logger.warning(f"Error assessing risk: {str(e)}")
            return {
                'risk_score': 0.0,
                'risk_level': 'unknown',
                'risk_factors': [],
                'recommendations': []
            }

    def _generate_risk_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []

        if risk_level in ['critical', 'high']:
            recommendations.extend([
                "Enhanced due diligence required",
                "Transaction monitoring recommended",
                "Consider additional verification steps",
                "Consult compliance officer"
            ])

        if risk_level == 'medium':
            recommendations.extend([
                "Additional verification recommended",
                "Document transaction purpose",
                "Monitor for suspicious patterns"
            ])

        if any('sanctions' in factor.lower() for factor in risk_factors):
            recommendations.append("Immediate sanctions screening required")

        if any('threshold' in factor.lower() for factor in risk_factors):
            recommendations.append("Regulatory reporting may be required")

        return recommendations

    def _update_performance_stats(self, result: FinancialProcessingResult):
        """Update performance statistics"""
        self._performance_stats['total_processed'] += 1
        self._performance_stats['currencies_found'] += len(result.currencies)
        self._performance_stats['banks_found'] += len(result.banks)
        self._performance_stats['compliance_flags'] += len(result.compliance_flags)

        if result.risk_assessment:
            self._performance_stats['risk_assessments'] += 1

        # Update average processing time
        current_avg = self._performance_stats['average_processing_time']
        self._performance_stats['average_processing_time'] = (
            (current_avg * (self._performance_stats['total_processed'] - 1)) +
            result.processing_time
        ) / self._performance_stats['total_processed']

    def process_batch(self, texts: List[str], languages: Optional[List[str]] = None) -> List[FinancialProcessingResult]:
        """
        Process a batch of financial texts

        Args:
            texts: List of financial texts
            languages: Optional list of pre-detected languages

        Returns:
            List of FinancialProcessingResult objects
        """
        if not texts:
            return []

        try:
            results = []

            for i, text in enumerate(texts):
                detected_lang = languages[i] if languages and i < len(languages) else None
                result = self.process_financial_text(text, detected_lang)
                results.append(result)

            self.logger.info(f"Processed {len(texts)} texts with international financial analysis")
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
        self.logger.info("International financial processing cache cleared")

    def _create_empty_result(self) -> FinancialProcessingResult:
        """Create empty financial processing result"""
        return FinancialProcessingResult(
            original_text='',
            processed_text='',
            detected_language='unknown',
            currencies=[],
            banks=[],
            transaction_patterns=[],
            compliance_flags=[],
            tax_indicators=[],
            risk_assessment={},
            jurisdictions=[],
            processing_time=0.0,
            success=False,
            error_message='Empty or invalid input'
        )

    def _create_error_result(self, text: str, error: str) -> FinancialProcessingResult:
        """Create error financial processing result"""
        return FinancialProcessingResult(
            original_text=text,
            processed_text=text,
            detected_language='unknown',
            currencies=[],
            banks=[],
            transaction_patterns=[],
            compliance_flags=[],
            tax_indicators=[],
            risk_assessment={},
            jurisdictions=[],
            processing_time=0.0,
            success=False,
            error_message=error
        )