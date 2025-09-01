import ofxparse
from datetime import datetime
from typing import List, Dict, Any
import re
import os
from src.utils.logging_config import get_logger
from src.utils.exceptions import (
    FileProcessingError, FileNotFoundError, InvalidFileFormatError,
    FileCorruptedError, ValidationError, InsufficientDataError
)
from src.utils.error_handler import handle_service_errors, with_timeout, recovery_manager
from src.utils.validators import validate_file_upload, transaction_validator
from src.utils.advanced_validation_engine import advanced_validation_engine
from src.utils.cross_field_validation_engine import cross_field_validation_engine
from src.utils.business_logic_validator import business_logic_validator
from src.utils.financial_business_rules import financial_business_rules
from src.utils.temporal_validation_engine import temporal_validation_engine
from src.utils.referential_integrity_validator import referential_integrity_validator
from src.services.data_completeness_analyzer import DataCompletenessAnalyzer
from src.services.missing_data_handler import MissingDataHandler

# Initialize logger
logger = get_logger(__name__)

class OFXProcessor:
    """
    Processador de arquivos OFX para diferentes bancos brasileiros
    """
    
    SUPPORTED_BANKS = {
        'caixa': 'CAIXA ECONÔMICA FEDERAL',
        'sicoob': 'SICOOB',
        'nubank': 'NUBANK',
        'itau': 'ITAÚ',
        'bradesco': 'BRADESCO',
        'santander': 'SANTANDER',
        'bb': 'BANCO DO BRASIL'
    }
    
    def __init__(self):
        self.bank_patterns = {
            'caixa': ['caixa', 'cef', 'economica', '104'],
            'sicoob': ['sicoob', 'sicob', '756'],
            'nubank': ['nubank', 'nu bank', 'nu pagamentos', '260'],
            'itau': ['itau', 'itaú', 'banco itau', '341'],
            'bradesco': ['bradesco', '237'],
            'santander': ['santander', '033'],
            'bb': ['banco do brasil', '001']
        }

        # Initialize data quality services
        self.completeness_analyzer = DataCompletenessAnalyzer()
        self.missing_data_handler = MissingDataHandler()
        self.data_quality_enabled = True
    
    def identify_bank(self, ofx_content: str) -> str:
        """
        Identifica o banco baseado no conteúdo do arquivo OFX
        """
        ofx_content_lower = ofx_content.lower()
        
        for bank_key, patterns in self.bank_patterns.items():
            for pattern in patterns:
                if pattern in ofx_content_lower:
                    return bank_key
        
        return 'unknown'
    
    @handle_service_errors('ofx_processor')
    @with_timeout(60)  # 60 second timeout for OFX processing
    def parse_ofx_file(self, file_path: str) -> Dict[str, Any]:
        """
        Processa um arquivo OFX e retorna os dados estruturados
        """
        logger.info(f"Starting OFX file parsing: {file_path}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(os.path.basename(file_path))
        
        # Validate file format
        filename = os.path.basename(file_path)
        validation_result = validate_file_upload(file_path, filename, 'ofx')
        if not validation_result.is_valid:
            raise InvalidFileFormatError(filename, ['ofx', 'qfx'])
        
        # Read file content with encoding fallback
        ofx_content = self._read_file_with_encoding_fallback(file_path)
        
        # Validate OFX content structure
        if not self._validate_ofx_structure(ofx_content):
            raise FileCorruptedError(filename)
        
        # Identifica o banco
        bank_name = self.identify_bank(ofx_content)
        logger.info(f"Bank identified: {bank_name}")
        
        # Parse do OFX with error handling
        try:
            ofx = self._parse_ofx_with_recovery(file_path)
        except Exception as e:
            logger.error(f"Failed to parse OFX file: {str(e)}")
            raise FileCorruptedError(filename)
        
        result = {
            'bank_name': bank_name,
            'account_info': {},
            'transactions': [],
            'summary': {
                'total_transactions': 0,
                'total_credits': 0,
                'total_debits': 0,
                'balance': None
            }
        }
        
        # Processa informações da conta
        if hasattr(ofx, 'account'):
            account = ofx.account
            result['account_info'] = {
                'account_id': getattr(account, 'account_id', ''),
                'routing_number': getattr(account, 'routing_number', ''),
                'account_type': getattr(account, 'account_type', ''),
                'bank_id': getattr(account, 'bank_id', '')
            }
            
            # Processa transações
            if hasattr(account, 'statement') and hasattr(account.statement, 'transactions'):
                transactions = []
                total_credits = 0
                total_debits = 0
                
                for transaction in account.statement.transactions:
                    amount = float(transaction.amount)
                    transaction_type = 'credit' if amount > 0 else 'debit'
                    
                    if transaction_type == 'credit':
                        total_credits += amount
                    else:
                        total_debits += abs(amount)
                    
                    transaction_data = {
                        'transaction_id': getattr(transaction, 'id', ''),
                        'date': transaction.date.date() if hasattr(transaction, 'date') else None,
                        'amount': amount,
                        'description': self._clean_description(getattr(transaction, 'memo', '') or getattr(transaction, 'payee', '')),
                        'transaction_type': transaction_type,
                        'balance': getattr(transaction, 'balance', None)
                    }
                    
                    transactions.append(transaction_data)
                
                result['transactions'] = transactions
                result['summary']['total_transactions'] = len(transactions)
                result['summary']['total_credits'] = total_credits
                result['summary']['total_debits'] = total_debits
                
                # Tenta obter o saldo final
                if hasattr(account.statement, 'balance'):
                    result['summary']['balance'] = float(account.statement.balance)
        
        # Validate parsed data
        self._validate_parsed_data(result)

        # Perform data quality analysis if enabled
        if self.data_quality_enabled and result['transactions']:
            try:
                logger.info("Performing data quality analysis on parsed transactions")
                quality_result = self._analyze_data_quality(result['transactions'])
                result['data_quality'] = quality_result
            except Exception as e:
                logger.warning(f"Data quality analysis failed: {str(e)}")
                result['data_quality'] = {'error': str(e)}

        logger.info(f"OFX parsing completed. Bank: {bank_name}, Transactions: {result['summary']['total_transactions']}")
        return result
    
    def _read_file_with_encoding_fallback(self, file_path: str) -> str:
        """Read file with encoding fallback mechanism."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                logger.debug(f"Successfully read file with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                logger.debug(f"Failed to read with {encoding} encoding, trying next")
                continue
        
        raise FileCorruptedError(os.path.basename(file_path))
    
    def _validate_ofx_structure(self, content: str) -> bool:
        """Validate basic OFX file structure."""
        content_upper = content.upper()
        
        # Check for OFX headers
        has_ofx_header = any(header in content_upper for header in [
            'OFXHEADER:', '<OFX>', 'DATA:OFXSGML', 'VERSION:'
        ])
        
        if not has_ofx_header:
            return False
        
        # Check for required sections
        required_sections = ['<BANKMSGSRSV1>', '<STMTRS>', '<BANKTRANLIST>']
        has_required_sections = any(section in content_upper for section in required_sections)
        
        return has_required_sections
    
    def _parse_ofx_with_recovery(self, file_path: str):
        """Parse OFX with recovery mechanisms."""
        def primary_parse():
            with open(file_path, 'rb') as file:
                return ofxparse.OfxParser.parse(file)
        
        def fallback_parse():
            # Try with different parsing options or preprocessing
            logger.warning("Primary OFX parsing failed, attempting recovery")
            
            # Read and preprocess the file
            content = self._read_file_with_encoding_fallback(file_path)
            
            # Fix common OFX issues
            content = self._fix_common_ofx_issues(content)
            
            # Write to temporary file and parse
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            try:
                with open(temp_path, 'rb') as file:
                    return ofxparse.OfxParser.parse(file)
            finally:
                os.unlink(temp_path)
        
        return recovery_manager.fallback_on_error(primary_parse, fallback_parse)
    
    def _fix_common_ofx_issues(self, content: str) -> str:
        """Fix common OFX file issues."""
        # Remove BOM if present
        if content.startswith('\ufeff'):
            content = content[1:]
        
        # Fix missing closing tags
        if '<OFX>' in content.upper() and '</OFX>' not in content.upper():
            content += '\n</OFX>'
        
        # Fix malformed dates
        content = re.sub(r'<DTPOSTED>(\d{8})(\d{6})?', r'<DTPOSTED>\1\2', content)
        
        # Remove invalid characters
        content = re.sub(r'[^\x20-\x7E\n\r\t]', '', content)
        
        return content
    
    def _validate_parsed_data(self, result: Dict[str, Any]):
        """Validate parsed OFX data."""
        if not result.get('transactions'):
            raise InsufficientDataError('OFX parsing', 1, 0)
        
        if result['summary']['total_transactions'] == 0:
            raise InsufficientDataError('OFX parsing', 1, 0)
        
        # Validate account info
        account_info = result.get('account_info', {})
        if not account_info.get('account_id'):
            logger.warning("OFX file missing account ID")
    
    def _clean_description(self, description: str) -> str:
        """
        Limpa e padroniza a descrição da transação
        """
        if not description:
            return 'Transação sem descrição'
        
        # Remove espaços extras e caracteres especiais desnecessários
        description = re.sub(r'\s+', ' ', description.strip())
        
        # Remove códigos de transação comuns no início
        description = re.sub(r'^(TED|DOC|PIX|TRANSF|SAQUE|DEPOSITO|COMPRA)\s*-?\s*', '', description, flags=re.IGNORECASE)
        
        return description
    
    @handle_service_errors('ofx_processor')
    def validate_transactions(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Valida e limpa os dados das transações usando múltiplos engines de validação
        """
        logger.info(f"Validating {len(transactions)} transactions with comprehensive validation engines")

        if not transactions:
            raise InsufficientDataError('transaction validation', 1, 0)

        # Initialize comprehensive validation context
        validation_context = {
            'schema_name': 'transaction',
            'rule_group': 'financial_transaction',
            'source': 'ofx_processor',
            'validation_profile': 'comprehensive'
        }

        # Track validation results from all engines
        all_valid_transactions = []
        all_invalid_transactions = []
        all_warnings = []
        validation_summary = {
            'engines_used': [],
            'total_validation_time_ms': 0,
            'engine_results': {}
        }

        start_time = datetime.now()

        # 1. Advanced Validation Engine (existing)
        try:
            logger.debug("Running Advanced Validation Engine")
            advanced_result = advanced_validation_engine.validate_bulk(
                transactions,
                profile='transaction',
                context=validation_context,
                parallel=True
            )
            validation_summary['engines_used'].append('advanced_validation_engine')
            validation_summary['engine_results']['advanced_validation_engine'] = {
                'errors': len(advanced_result.errors),
                'warnings': len(advanced_result.warnings),
                'duration_ms': advanced_result.validation_duration
            }
        except Exception as e:
            logger.warning(f"Advanced validation engine error: {str(e)}")
            advanced_result = None

        # 2. Cross-Field Validation Engine
        try:
            logger.debug("Running Cross-Field Validation Engine")
            cross_field_results = []
            for i, transaction in enumerate(transactions):
                context_with_index = {**validation_context, 'transaction_index': i}
                result = cross_field_validation_engine.validate(
                    transaction, rule_group='financial_transaction', context=context_with_index
                )
                cross_field_results.append(result)

            validation_summary['engines_used'].append('cross_field_validation_engine')
            validation_summary['engine_results']['cross_field_validation_engine'] = {
                'errors': sum(len(r.errors) for r in cross_field_results),
                'warnings': sum(len(r.warnings) for r in cross_field_results),
                'duration_ms': sum(r.validation_duration for r in cross_field_results)
            }
        except Exception as e:
            logger.warning(f"Cross-field validation engine error: {str(e)}")
            cross_field_results = None

        # 3. Business Logic Validator
        try:
            logger.debug("Running Business Logic Validator")
            business_logic_results = []
            for i, transaction in enumerate(transactions):
                context_with_index = {**validation_context, 'transaction_index': i}
                result = business_logic_validator.validate(
                    transaction, rule_group='transaction_validation', context=context_with_index
                )
                business_logic_results.append(result)

            validation_summary['engines_used'].append('business_logic_validator')
            validation_summary['engine_results']['business_logic_validator'] = {
                'errors': sum(len(r.errors) for r in business_logic_results),
                'warnings': sum(len(r.warnings) for r in business_logic_results),
                'duration_ms': sum(r.validation_duration for r in business_logic_results)
            }
        except Exception as e:
            logger.warning(f"Business logic validator error: {str(e)}")
            business_logic_results = None

        # 4. Financial Business Rules
        try:
            logger.debug("Running Financial Business Rules")
            financial_results = []
            for i, transaction in enumerate(transactions):
                context_with_index = {**validation_context, 'transaction_index': i}
                result = financial_business_rules.validate(
                    transaction, rule_group='transaction_processing', context=context_with_index
                )
                financial_results.append(result)

            validation_summary['engines_used'].append('financial_business_rules')
            validation_summary['engine_results']['financial_business_rules'] = {
                'errors': sum(len(r.errors) for r in financial_results),
                'warnings': sum(len(r.warnings) for r in financial_results),
                'duration_ms': sum(r.validation_duration for r in financial_results)
            }
        except Exception as e:
            logger.warning(f"Financial business rules error: {str(e)}")
            financial_results = None

        # 5. Temporal Validation Engine
        try:
            logger.debug("Running Temporal Validation Engine")
            temporal_results = []
            for i, transaction in enumerate(transactions):
                context_with_index = {**validation_context, 'transaction_index': i}
                result = temporal_validation_engine.validate(
                    transaction, rule_group='date_validation', context=context_with_index
                )
                temporal_results.append(result)

            validation_summary['engines_used'].append('temporal_validation_engine')
            validation_summary['engine_results']['temporal_validation_engine'] = {
                'errors': sum(len(r.errors) for r in temporal_results),
                'warnings': sum(len(r.warnings) for r in temporal_results),
                'duration_ms': sum(r.validation_duration for r in temporal_results)
            }
        except Exception as e:
            logger.warning(f"Temporal validation engine error: {str(e)}")
            temporal_results = None

        # 6. Referential Integrity Validator
        try:
            logger.debug("Running Referential Integrity Validator")
            referential_results = []
            for i, transaction in enumerate(transactions):
                context_with_index = {**validation_context, 'transaction_index': i}
                result = referential_integrity_validator.validate(
                    transaction, rule_group='foreign_key_validation', context=context_with_index
                )
                referential_results.append(result)

            validation_summary['engines_used'].append('referential_integrity_validator')
            validation_summary['engine_results']['referential_integrity_validator'] = {
                'errors': sum(len(r.errors) for r in referential_results),
                'warnings': sum(len(r.warnings) for r in referential_results),
                'duration_ms': sum(r.validation_duration for r in referential_results)
            }
        except Exception as e:
            logger.warning(f"Referential integrity validator error: {str(e)}")
            referential_results = None

        # Process combined validation results
        for i, transaction in enumerate(transactions):
            transaction_errors = []
            transaction_warnings = []

            # Collect errors and warnings from all engines
            if advanced_result:
                transaction_key = f"[{i}]"
                for field_name, field_result in advanced_result.field_results.items():
                    if field_name.startswith(transaction_key):
                        if field_result.errors:
                            transaction_errors.extend([f"Advanced: {err}" for err in field_result.errors])
                        if field_result.warnings:
                            transaction_warnings.extend([f"Advanced: {warn}" for warn in field_result.warnings])

            if cross_field_results and i < len(cross_field_results):
                result = cross_field_results[i]
                transaction_errors.extend([f"Cross-field: {err}" for err in result.errors])
                transaction_warnings.extend([f"Cross-field: {warn}" for warn in result.warnings])

            if business_logic_results and i < len(business_logic_results):
                result = business_logic_results[i]
                transaction_errors.extend([f"Business Logic: {err}" for err in result.errors])
                transaction_warnings.extend([f"Business Logic: {warn}" for warn in result.warnings])

            if financial_results and i < len(financial_results):
                result = financial_results[i]
                transaction_errors.extend([f"Financial: {err}" for err in result.errors])
                transaction_warnings.extend([f"Financial: {warn}" for warn in result.warnings])

            if temporal_results and i < len(temporal_results):
                result = temporal_results[i]
                transaction_errors.extend([f"Temporal: {err}" for err in result.errors])
                transaction_warnings.extend([f"Temporal: {warn}" for warn in result.warnings])

            if referential_results and i < len(referential_results):
                result = referential_results[i]
                transaction_errors.extend([f"Referential: {err}" for err in result.errors])
                transaction_warnings.extend([f"Referential: {warn}" for warn in result.warnings])

            if transaction_errors:
                all_invalid_transactions.append({
                    'index': i,
                    'transaction': transaction,
                    'errors': transaction_errors,
                    'warnings': transaction_warnings
                })
            else:
                # Clean and normalize transaction data
                cleaned_transaction = self._clean_transaction_data(transaction)
                all_valid_transactions.append(cleaned_transaction)

                # Collect warnings even for valid transactions
                if transaction_warnings:
                    all_warnings.append({
                        'index': i,
                        'transaction': transaction,
                        'warnings': transaction_warnings
                    })

        # Calculate total validation time
        total_duration = (datetime.now() - start_time).total_seconds() * 1000
        validation_summary['total_validation_time_ms'] = total_duration

        # Log comprehensive validation summary
        logger.info(f"Comprehensive validation completed in {total_duration:.2f}ms")
        logger.info(f"Valid: {len(all_valid_transactions)}, Invalid: {len(all_invalid_transactions)}, Warnings: {len(all_warnings)}")
        logger.info(f"Engines used: {validation_summary['engines_used']}")

        if all_invalid_transactions:
            logger.warning(f"Invalid transactions: {len(all_invalid_transactions)}")

        if not all_valid_transactions:
            raise InsufficientDataError('comprehensive transaction validation', 1, 0)

        return {
            'valid_transactions': all_valid_transactions,
            'invalid_transactions': all_invalid_transactions,
            'warnings': all_warnings,
            'validation_summary': validation_summary,
            'validation_duration_ms': total_duration
        }
    
    def _clean_transaction_data(self, transaction: Dict) -> Dict:
        """Clean and normalize transaction data."""
        cleaned = transaction.copy()
        
        # Ensure description is not empty
        if not cleaned.get('description') or cleaned['description'].strip() == '':
            cleaned['description'] = 'Transação sem descrição'
        
        # Clean description
        cleaned['description'] = self._clean_description(cleaned['description'])
        
        # Ensure amount is float
        if isinstance(cleaned.get('amount'), str):
            try:
                cleaned['amount'] = float(cleaned['amount'].replace(',', '.'))
            except ValueError:
                cleaned['amount'] = 0.0
        
        # Ensure date is properly formatted
        if isinstance(cleaned.get('date'), str):
            try:
                cleaned['date'] = datetime.fromisoformat(cleaned['date']).date()
            except ValueError:
                pass  # Keep original if conversion fails
        
        return cleaned
    
    @handle_service_errors('ofx_processor')
    def detect_duplicates(self, transactions: List[Dict]) -> List[int]:
        """
        Detecta possíveis transações duplicadas
        """
        logger.info(f"Detecting duplicates in {len(transactions)} transactions")
        
        if not transactions:
            return []
        
        duplicates = []
        seen = set()
        
        try:
            for i, transaction in enumerate(transactions):
                # Validate transaction has required fields for duplicate detection
                if not all(key in transaction for key in ['date', 'amount', 'description']):
                    logger.warning(f"Transaction {i} missing required fields for duplicate detection")
                    continue
                
                # Cria uma chave única baseada em data, valor e descrição
                key = self._create_duplicate_key(transaction)
                
                if key in seen:
                    duplicates.append(i)
                    logger.debug(f"Duplicate found at index {i}: {transaction}")
                else:
                    seen.add(key)
            
            logger.info(f"Duplicate detection completed. Found {len(duplicates)} duplicates")
            return duplicates
            
        except Exception as e:
            logger.error(f"Error during duplicate detection: {str(e)}")
            raise FileProcessingError(f"Duplicate detection failed: {str(e)}")
    
    def _create_duplicate_key(self, transaction: Dict) -> tuple:
        """Create a unique key for duplicate detection."""
        date_key = transaction.get('date')
        if hasattr(date_key, 'isoformat'):
            date_key = date_key.isoformat()
        elif isinstance(date_key, str):
            date_key = date_key
        else:
            date_key = str(date_key)
        
        amount_key = float(transaction.get('amount', 0))
        description_key = str(transaction.get('description', '')).lower().strip()
        
        return (date_key, amount_key, description_key)

    def _analyze_data_quality(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Analyze data quality of parsed transactions and provide recommendations

        Args:
            transactions: List of parsed transaction dictionaries

        Returns:
            Dictionary with data quality analysis results
        """
        try:
            logger.info(f"Analyzing data quality for {len(transactions)} transactions")

            # Convert to DataFrame for analysis
            df = pd.DataFrame(transactions)

            # Perform completeness analysis
            completeness_report = self.completeness_analyzer.generate_completeness_report(df)

            # Get imputation recommendations
            recommendations = self.missing_data_handler.get_imputation_recommendations(df)

            # Check for critical data quality issues
            critical_issues = self._identify_critical_issues(completeness_report, df)

            # Auto-apply imputation if confidence is high
            imputation_applied = False
            imputed_data = None

            if self._should_auto_impute(completeness_report):
                try:
                    logger.info("Auto-applying imputation for data quality improvement")
                    imputation_result = self.missing_data_handler.analyze_and_impute(df)
                    imputed_data = imputation_result.imputed_data
                    imputation_applied = True

                    logger.info(f"Auto-imputation applied: {imputation_result.imputation_count} values imputed")
                except Exception as e:
                    logger.warning(f"Auto-imputation failed: {str(e)}")

            quality_result = {
                'completeness_analysis': completeness_report,
                'recommendations': recommendations,
                'critical_issues': critical_issues,
                'auto_imputation_applied': imputation_applied,
                'data_quality_score': self._calculate_data_quality_score(completeness_report),
                'processing_timestamp': datetime.now().isoformat()
            }

            if imputation_applied and imputed_data is not None:
                quality_result['imputed_transactions'] = imputed_data.to_dict('records')

            logger.info(f"Data quality analysis completed. Score: {quality_result['data_quality_score']:.3f}")
            return quality_result

        except Exception as e:
            logger.error(f"Error in data quality analysis: {str(e)}")
            return {
                'error': str(e),
                'completeness_analysis': None,
                'recommendations': [],
                'critical_issues': [{'type': 'analysis_error', 'message': str(e)}],
                'data_quality_score': 0.0
            }

    def _identify_critical_issues(self, completeness_report: Dict[str, Any], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify critical data quality issues that need immediate attention

        Args:
            completeness_report: Completeness analysis results
            df: DataFrame with transaction data

        Returns:
            List of critical issues
        """
        critical_issues = []

        try:
            # Check critical field completeness
            critical_fields = ['date', 'amount', 'description']
            for field in critical_fields:
                if field in completeness_report.get('field_completeness', {}):
                    completeness = completeness_report['field_completeness'][field]['completeness_score']
                    if completeness < 0.8:  # Less than 80% complete
                        critical_issues.append({
                            'type': 'critical_field_incomplete',
                            'field': field,
                            'completeness': completeness,
                            'severity': 'high',
                            'message': f"Critical field '{field}' has low completeness ({completeness:.1%})"
                        })

            # Check for high missing data ratio
            overall_completeness = completeness_report.get('dataset_completeness', 1.0)
            if overall_completeness < 0.7:
                critical_issues.append({
                    'type': 'high_missing_data',
                    'overall_completeness': overall_completeness,
                    'severity': 'high',
                    'message': f"Dataset has high missing data ratio ({(1-overall_completeness):.1%} missing)"
                })

            # Check for potential data corruption
            if len(df) > 0:
                # Check for transactions with missing both amount and description
                missing_both = df[(df['amount'].isnull()) & (df['description'].isnull())]
                if len(missing_both) > len(df) * 0.1:  # More than 10%
                    critical_issues.append({
                        'type': 'potential_corruption',
                        'affected_records': len(missing_both),
                        'percentage': len(missing_both) / len(df),
                        'severity': 'medium',
                        'message': f"High number of records missing both amount and description ({len(missing_both)} records)"
                    })

        except Exception as e:
            logger.warning(f"Error identifying critical issues: {str(e)}")

        return critical_issues

    def _should_auto_impute(self, completeness_report: Dict[str, Any]) -> bool:
        """
        Determine if auto-imputation should be applied based on completeness analysis

        Args:
            completeness_report: Completeness analysis results

        Returns:
            Boolean indicating if auto-imputation should be applied
        """
        try:
            overall_completeness = completeness_report.get('dataset_completeness', 1.0)

            # Auto-impute if:
            # 1. Overall completeness is above 60% (not too much missing data)
            # 2. No critical fields are severely incomplete (<50%)
            # 3. There are some missing values but not catastrophic

            if overall_completeness < 0.6:
                return False  # Too much missing data

            if overall_completeness > 0.95:
                return False  # Very complete data, no need for imputation

            # Check critical fields
            critical_fields = ['date', 'amount', 'description']
            for field in critical_fields:
                if field in completeness_report.get('field_completeness', {}):
                    field_completeness = completeness_report['field_completeness'][field]['completeness_score']
                    if field_completeness < 0.5:  # Critical field too incomplete
                        return False

            return True

        except Exception as e:
            logger.warning(f"Error determining auto-imputation: {str(e)}")
            return False

    def _calculate_data_quality_score(self, completeness_report: Dict[str, Any]) -> float:
        """
        Calculate an overall data quality score

        Args:
            completeness_report: Completeness analysis results

        Returns:
            Data quality score between 0 and 1
        """
        try:
            base_score = completeness_report.get('dataset_completeness', 0.0)

            # Adjust for critical fields
            critical_fields = ['date', 'amount', 'description']
            critical_penalty = 0.0

            for field in critical_fields:
                if field in completeness_report.get('field_completeness', {}):
                    field_score = completeness_report['field_completeness'][field]['completeness_score']
                    if field_score < 0.8:
                        critical_penalty += (0.8 - field_score) * 0.2  # 20% penalty per critical field

            quality_score = max(0.0, base_score - critical_penalty)

            return round(quality_score, 3)

        except Exception as e:
            logger.warning(f"Error calculating data quality score: {str(e)}")
            return 0.0

