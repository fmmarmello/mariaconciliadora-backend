import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import re
import os
from src.services.duplicate_detection_service import DuplicateDetectionService
import hashlib
from src.utils.logging_config import get_logger
from src.utils.exceptions import (
    FileProcessingError, FileNotFoundError, InvalidFileFormatError,
    FileCorruptedError, ValidationError, InsufficientDataError
)
from src.utils.error_handler import handle_service_errors, with_timeout, recovery_manager
from src.utils.validators import validate_file_upload, company_financial_validator
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

class XLSXProcessor:
    """
    Processador de arquivos XLSX para despesas e receitas empresariais
    """
    
    def __init__(self):
        self.supported_columns = {
            'date': ['data', 'date', 'dia'],
            'description': ['descricao', 'description', 'histórico', 'historico'],
            'amount': ['valor', 'amount', 'value'],
            'category': ['categoria', 'category'],
            'cost_center': ['centro de custo', 'cost center'],
            'department': ['departamento', 'department'],
            'project': ['projeto', 'project'],
            'transaction_type': ['tipo', 'type', 'transaction type'],
            'observations': ['observações', 'observations', 'obs'],
            'monthly_report_value': ['valor para relat mensal', 'valor para relatório mensal', 'monthly report value']
        }
        self.duplicate_service = DuplicateDetectionService()

        # Initialize data quality services
        self.completeness_analyzer = DataCompletenessAnalyzer()
        self.missing_data_handler = MissingDataHandler()
        self.data_quality_enabled = True
    
    @handle_service_errors('xlsx_processor')
    @with_timeout(120)  # 2 minute timeout for XLSX processing
    def parse_xlsx_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Processa um arquivo XLSX e retorna os dados estruturados
        """
        logger.info(f"Starting XLSX file parsing: {file_path}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(os.path.basename(file_path))
        
        # Validate file format
        filename = os.path.basename(file_path)
        validation_result = validate_file_upload(file_path, filename, 'xlsx')
        if not validation_result.is_valid:
            raise InvalidFileFormatError(filename, ['xlsx'])
        
        try:
            # Read XLSX file with error handling
            df = self._read_xlsx_with_recovery(file_path)
            logger.info(f"XLSX file loaded successfully. Rows: {len(df)}, Columns: {len(df.columns)}")
            
            # Validate file structure
            self._validate_xlsx_structure(df, filename)
            
            # Normalize column names
            original_columns = df.columns.tolist()
            df.columns = [self._normalize_column_name(col) for col in df.columns]
            logger.info(f"Column normalization completed. Original: {original_columns}, Normalized: {df.columns.tolist()}")

            # Debug: Log what columns we found
            found_columns = {}
            for standard_col, variations in self.supported_columns.items():
                for norm_col in df.columns:
                    for variation in variations:
                        if variation in norm_col or norm_col in variation:
                            if standard_col not in found_columns:
                                found_columns[standard_col] = []
                            found_columns[standard_col].append(norm_col)
                            break

            logger.info(f"Column mapping results: {found_columns}")
            
            # Process rows with validation
            financial_data = self._process_xlsx_rows(df)
            
            # Validate processed data
            self._validate_processed_data(financial_data, filename)

            # Perform data quality analysis if enabled
            if self.data_quality_enabled and financial_data:
                try:
                    logger.info("Performing data quality analysis on processed XLSX data")
                    quality_result = self._analyze_data_quality(financial_data)
                    # Return both data and quality analysis
                    return {
                        'financial_data': financial_data,
                        'data_quality': quality_result
                    }
                except Exception as e:
                    logger.warning(f"Data quality analysis failed: {str(e)}")
                    return {
                        'financial_data': financial_data,
                        'data_quality': {'error': str(e)}
                    }

            logger.info(f"XLSX parsing completed successfully. Processed: {len(financial_data)} entries")
            return financial_data
            
        except pd.errors.EmptyDataError:
            raise FileCorruptedError(filename)
        except (pd.errors.ClosedFileError, ValueError) as e:
            # Handle various pandas Excel reading errors
            if "Excel file format cannot be determined" in str(e) or "Unsupported format" in str(e):
                raise FileCorruptedError(filename)
            else:
                raise FileCorruptedError(filename)
        except Exception as e:
            if isinstance(e, (FileProcessingError, ValidationError)):
                raise
            logger.error(f"Unexpected error parsing XLSX file: {str(e)}", exc_info=True)
            raise FileProcessingError(f"Erro ao processar arquivo XLSX: {str(e)}", filename)
    
    def _read_xlsx_with_recovery(self, file_path: str) -> pd.DataFrame:
        """Read XLSX file with recovery mechanisms."""
        def primary_read():
            return pd.read_excel(file_path)
        
        def fallback_read():
            logger.warning("Primary XLSX reading failed, attempting recovery")
            # Try with different engines
            engines = ['openpyxl', 'xlrd']
            
            for engine in engines:
                try:
                    logger.debug(f"Trying to read XLSX with {engine} engine")
                    return pd.read_excel(file_path, engine=engine)
                except Exception as e:
                    logger.debug(f"Failed with {engine}: {str(e)}")
                    continue
            
            # If all engines fail, try reading as CSV (in case it's misnamed)
            try:
                logger.debug("Attempting to read as CSV")
                return pd.read_csv(file_path)
            except Exception:
                raise FileCorruptedError(os.path.basename(file_path))
        
        return recovery_manager.fallback_on_error(primary_read, fallback_read)
    
    def _validate_xlsx_structure(self, df: pd.DataFrame, filename: str):
        """Validate XLSX file structure."""
        if df.empty:
            raise InsufficientDataError('XLSX parsing', 1, 0)
        
        if len(df.columns) < 2:
            raise ValidationError(
                f"XLSX file must have at least 2 columns, found {len(df.columns)}",
                user_message="Arquivo deve conter pelo menos 2 colunas (data e descrição)."
            )
        
        # Check for required columns (flexible matching)
        required_columns = ['data', 'valor', 'description']
        normalized_columns = [self._normalize_column_name(col) for col in df.columns]
        
        missing_columns = []
        for req_col in required_columns:
            if not any(req_col in norm_col or norm_col in req_col for norm_col in normalized_columns):
                missing_columns.append(req_col)
        
        if missing_columns:
            logger.warning(f"Missing recommended columns: {missing_columns}")
            # Don't raise error, just warn - file might still be processable
    
    def _process_xlsx_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process XLSX rows with validation and error handling."""
        financial_data = []
        skipped_balance_entries = 0
        processing_errors = []
        
        for index, row in df.iterrows():
            try:
                entry = self._process_single_row(row, index)
                
                # Skip balance entries
                if entry and 'saldo' in entry.get('description', '').lower():
                    skipped_balance_entries += 1
                    continue
                
                if entry:
                    financial_data.append(entry)
                    
            except Exception as e:
                error_msg = f"Error processing row {index + 1}: {str(e)}"
                logger.warning(error_msg)
                processing_errors.append({
                    'row': index + 1,
                    'error': str(e),
                    'data': row.to_dict()
                })
        
        # Log processing summary
        logger.info(f"Row processing completed. Valid: {len(financial_data)}, "
                   f"Skipped balance: {skipped_balance_entries}, Errors: {len(processing_errors)}")
        
        if processing_errors:
            logger.warning(f"Processing errors: {processing_errors[:5]}")  # Log first 5 errors
        
        return financial_data
    
    def _process_single_row(self, row: pd.Series, index: int) -> Dict[str, Any]:
        """Process a single XLSX row with comprehensive validation using all engines."""
        try:
            entry = {
                'date': self._parse_date(row.get('data')),
                'description': self._parse_description(row.get('description', '')),
                'amount': self._parse_amount(row.get('valor')),
                'category': str(row.get('tipo', '')),
                'cost_center': str(row.get('cost_center', '')),
                'department': str(row.get('department', '')),
                'project': str(row.get('project', '')),
                'transaction_type': self._determine_transaction_type(row),
                'observations': str(row.get('observations', '')),
                'monthly_report_value': self._parse_amount(row.get('monthly_report_value'))
            }

            # Comprehensive validation context
            validation_context = {
                'schema_name': 'company_financial',
                'rule_group': 'company_financial',
                'source': 'xlsx_processor',
                'row_index': index + 1,
                'validation_profile': 'comprehensive'
            }

            # Track validation results from all engines
            all_errors = []
            all_warnings = []

            # 1. Advanced Validation Engine (existing)
            try:
                advanced_result = advanced_validation_engine.validate(
                    entry,
                    profile='company_financial',
                    context=validation_context
                )
                all_errors.extend([f"Advanced: {err}" for err in advanced_result.errors])
                all_warnings.extend([f"Advanced: {warn}" for warn in advanced_result.warnings])
            except Exception as e:
                logger.warning(f"Advanced validation engine error for row {index + 1}: {str(e)}")
                all_warnings.append(f"Advanced validation engine failed: {str(e)}")

            # 2. Cross-Field Validation Engine
            try:
                cross_field_result = cross_field_validation_engine.validate(
                    entry, rule_group='financial_transaction', context=validation_context
                )
                all_errors.extend([f"Cross-field: {err}" for err in cross_field_result.errors])
                all_warnings.extend([f"Cross-field: {warn}" for warn in cross_field_result.warnings])
            except Exception as e:
                logger.warning(f"Cross-field validation engine error for row {index + 1}: {str(e)}")
                all_warnings.append(f"Cross-field validation engine failed: {str(e)}")

            # 3. Business Logic Validator
            try:
                business_logic_result = business_logic_validator.validate(
                    entry, rule_group='transaction_validation', context=validation_context
                )
                all_errors.extend([f"Business Logic: {err}" for err in business_logic_result.errors])
                all_warnings.extend([f"Business Logic: {warn}" for warn in business_logic_result.warnings])
            except Exception as e:
                logger.warning(f"Business logic validator error for row {index + 1}: {str(e)}")
                all_warnings.append(f"Business logic validator failed: {str(e)}")

            # 4. Financial Business Rules
            try:
                financial_result = financial_business_rules.validate(
                    entry, rule_group='transaction_processing', context=validation_context
                )
                all_errors.extend([f"Financial: {err}" for err in financial_result.errors])
                all_warnings.extend([f"Financial: {warn}" for warn in financial_result.warnings])
            except Exception as e:
                logger.warning(f"Financial business rules error for row {index + 1}: {str(e)}")
                all_warnings.append(f"Financial business rules failed: {str(e)}")

            # 5. Temporal Validation Engine
            try:
                temporal_result = temporal_validation_engine.validate(
                    entry, rule_group='date_validation', context=validation_context
                )
                all_errors.extend([f"Temporal: {err}" for err in temporal_result.errors])
                all_warnings.extend([f"Temporal: {warn}" for warn in temporal_result.warnings])
            except Exception as e:
                logger.warning(f"Temporal validation engine error for row {index + 1}: {str(e)}")
                all_warnings.append(f"Temporal validation engine failed: {str(e)}")

            # 6. Referential Integrity Validator
            try:
                referential_result = referential_integrity_validator.validate(
                    entry, rule_group='foreign_key_validation', context=validation_context
                )
                all_errors.extend([f"Referential: {err}" for err in referential_result.errors])
                all_warnings.extend([f"Referential: {warn}" for warn in referential_result.warnings])
            except Exception as e:
                logger.warning(f"Referential integrity validator error for row {index + 1}: {str(e)}")
                all_warnings.append(f"Referential integrity validator failed: {str(e)}")

            # Check if entry is valid (no critical errors)
            if all_errors:
                logger.debug(f"Row {index + 1} comprehensive validation failed: {all_errors}")
                # Return None for invalid entries - they will be skipped
                return None

            # Log warnings if any
            if all_warnings:
                logger.debug(f"Row {index + 1} validation warnings: {all_warnings}")

            return entry

        except Exception as e:
            logger.debug(f"Error processing row {index + 1}: {str(e)}")
            return None
    
    def _parse_description(self, description: Any) -> str:
        """Parse and validate description field."""
        if pd.isna(description) or description is None:
            return ''
        
        desc_str = str(description).strip()
        
        # Remove excessive whitespace
        desc_str = re.sub(r'\s+', ' ', desc_str)
        
        # Limit length
        if len(desc_str) > 500:
            desc_str = desc_str[:500]
            logger.debug("Description truncated to 500 characters")
        
        return desc_str
    
    def _validate_processed_data(self, financial_data: List[Dict[str, Any]], filename: str):
        """Validate processed financial data."""
        if not financial_data:
            raise InsufficientDataError('XLSX processing', 1, 0)
        
        # Check data quality
        valid_entries = 0
        for entry in financial_data:
            if (entry.get('date') and
                entry.get('description') and
                entry.get('amount') is not None):
                valid_entries += 1
        
        if valid_entries == 0:
            raise InsufficientDataError('XLSX processing', 1, 0)
        
        # Warn if data quality is low
        quality_ratio = valid_entries / len(financial_data)
        if quality_ratio < 0.5:
            logger.warning(f"Low data quality in {filename}: {quality_ratio:.1%} valid entries")
    
    def _normalize_column_name(self, column_name: str) -> str:
        """Normaliza nomes de colunas para padrão consistente"""
        column_name = str(column_name).strip().lower()
        # Remove acentos e caracteres especiais
        column_name = re.sub(r'[^\w\s]', '', column_name)
        return column_name
    
    def _parse_date(self, date_value) -> datetime:
        """Parse de datas em vários formatos com tratamento robusto de erros"""
        if pd.isna(date_value) or date_value is None:
            return None
        
        # If already a date object, return it
        if hasattr(date_value, 'date'):
            return date_value.date()
        
        # Try multiple date formats
        date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y',
            '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d'
        ]
        
        date_str = str(date_value).strip()
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        
        # Try pandas parsing as last resort
        try:
            return pd.to_datetime(date_value, dayfirst=True).date()
        except Exception:
            logger.debug(f"Failed to parse date: {date_value}")
            return None
    
    def _parse_amount(self, amount_value) -> float:
        """Parse de valores monetários com tratamento robusto de erros"""
        if pd.isna(amount_value) or amount_value is None:
            return 0.0
        
        # If already a number, return it
        if isinstance(amount_value, (int, float)):
            return float(amount_value)
        
        # Clean string value
        amount_str = str(amount_value).strip()
        
        if not amount_str:
            return 0.0
        
        try:
            # Remove currency symbols and spaces
            amount_str = re.sub(r'[R$\s]', '', amount_str)
            
            # Handle Brazilian decimal format (comma as decimal separator)
            if ',' in amount_str and '.' in amount_str:
                # Format like 1.234.567,89
                amount_str = amount_str.replace('.', '').replace(',', '.')
            elif ',' in amount_str:
                # Check if comma is thousands separator or decimal separator
                comma_parts = amount_str.split(',')
                if len(comma_parts) == 2 and len(comma_parts[1]) <= 2:
                    # Decimal separator
                    amount_str = amount_str.replace(',', '.')
                else:
                    # Thousands separator
                    amount_str = amount_str.replace(',', '')
            
            # Handle parentheses for negative numbers
            if amount_str.startswith('(') and amount_str.endswith(')'):
                amount_str = '-' + amount_str[1:-1]
            
            return float(amount_str)
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse amount '{amount_value}': {str(e)}")
            return 0.0
    
    def _determine_transaction_type(self, row) -> str:
        """Determina se é despesa ou receita"""
        transaction_type = row.get('tipo', '').lower()
        #TODO adicionar IA para interpretar tipos
        if transaction_type in ['despesa', 'expense', 'débito', 'debit', 'retirada sócio',  'impostos / tributos', 'tarifas bancárias', 'juros / multa', 'seguro', 'emprestimo']:
            return 'expense'
        elif transaction_type in ['reembolso','receita', 'income', 'crédito', 'credit','credito']:
            return 'income'
        else:
            # Determina pelo valor (negativo = despesa, positivo = receita)
            amount = self._parse_amount(row.get('valor'))
            return 'expense' if amount < 0 else 'income'
    
    @handle_service_errors('xlsx_processor')
    def detect_duplicates(self, entries: List[Dict]) -> List[int]:
        """
        Detecta possíveis entradas duplicadas
        Returns a list of indices of duplicate entries
        """
        logger.info(f"Detecting duplicates in {len(entries)} XLSX entries")
        
        if not entries:
            return []
        
        duplicates = []
        seen = set()
        db_duplicates = 0
        file_duplicates = 0
        processing_errors = 0
        
        try:
            for i, entry in enumerate(entries):
                try:
                    # Validate entry has required fields for duplicate detection
                    if not self._has_required_fields_for_duplicate_check(entry):
                        logger.debug(f"Entry {i} missing required fields for duplicate detection")
                        processing_errors += 1
                        continue
                    
                    # Check database duplicates with error handling
                    try:
                        is_duplicate = self.duplicate_service.check_financial_entry_duplicate(
                            entry.get('date'),
                            entry.get('amount'),
                            entry.get('description')
                        )
                        
                        if is_duplicate:
                            duplicates.append(i)
                            db_duplicates += 1
                            continue
                            
                    except Exception as e:
                        logger.warning(f"Error checking database duplicate for entry {i}: {str(e)}")
                        processing_errors += 1
                        # Continue with file-level duplicate check
                    
                    # Check file-level duplicates
                    key = self._create_duplicate_key(entry)
                    
                    if key in seen:
                        duplicates.append(i)
                        file_duplicates += 1
                    else:
                        seen.add(key)
                        
                except Exception as e:
                    logger.warning(f"Error processing entry {i} for duplicate detection: {str(e)}")
                    processing_errors += 1
                    continue
            
            logger.info(f"Duplicate detection completed. Total duplicates: {len(duplicates)} "
                       f"(DB: {db_duplicates}, File: {file_duplicates}, Errors: {processing_errors})")
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Critical error during duplicate detection: {str(e)}")
            raise FileProcessingError(f"Duplicate detection failed: {str(e)}")
    
    def _has_required_fields_for_duplicate_check(self, entry: Dict) -> bool:
        """Check if entry has required fields for duplicate detection."""
        required_fields = ['date', 'amount', 'description']
        return all(
            field in entry and entry[field] is not None
            for field in required_fields
        )
    
    def _create_duplicate_key(self, entry: Dict) -> tuple:
        """Create a unique key for duplicate detection."""
        date_key = entry.get('date')
        if hasattr(date_key, 'isoformat'):
            date_key = date_key.isoformat()
        elif isinstance(date_key, str):
            date_key = date_key
        else:
            date_key = str(date_key)

        amount_key = float(entry.get('amount', 0))
        description_key = str(entry.get('description', '')).lower().strip()

        return (date_key, amount_key, description_key)

    @handle_service_errors('xlsx_processor')
    @with_timeout(60)  # 1 minute timeout for analysis
    def analyze_xlsx_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze XLSX file structure and content to determine type and provide detailed analysis
        """
        logger.info(f"Starting XLSX file analysis: {file_path}")

        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(os.path.basename(file_path))

        # Validate file format
        filename = os.path.basename(file_path)
        validation_result = validate_file_upload(file_path, filename, 'xlsx')
        if not validation_result.is_valid:
            raise InvalidFileFormatError(filename, ['xlsx'])

        try:
            # Read XLSX file
            df = self._read_xlsx_with_recovery(file_path)
            logger.info(f"XLSX file loaded successfully. Rows: {len(df)}, Columns: {len(df.columns)}")

            # Analyze file structure
            analysis_result = self._analyze_file_structure(df, filename)

            # Determine file type
            file_type = self._determine_file_type(df, analysis_result)

            # Generate detailed analysis
            detailed_analysis = self._generate_detailed_analysis(df, analysis_result, file_type)

            logger.info(f"XLSX analysis completed successfully. Type: {file_type}")

            # Convert pandas/numpy types to Python native types for JSON serialization
            def convert_to_native_types(obj):
                if isinstance(obj, dict):
                    return {key: convert_to_native_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native_types(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy types
                    return obj.item()
                elif hasattr(obj, 'isoformat'):  # datetime objects
                    return obj.isoformat()
                else:
                    return obj

            return {
                'file_type': file_type,
                'analysis': convert_to_native_types(detailed_analysis),
                'structure': convert_to_native_types(analysis_result),
                'summary': {
                    'total_rows': int(len(df)),
                    'total_columns': int(len(df.columns)),
                    'filename': filename,
                    'file_size': int(os.path.getsize(file_path))
                }
            }

        except Exception as e:
            if isinstance(e, (FileProcessingError, ValidationError)):
                raise
            logger.error(f"Unexpected error analyzing XLSX file: {str(e)}", exc_info=True)
            raise FileProcessingError(f"Erro ao analisar arquivo XLSX: {str(e)}", filename)

    def _analyze_file_structure(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Analyze the structure of the XLSX file."""
        structure = {
            'columns': [],
            'data_types': {},
            'missing_values': {},
            'sample_data': {},
            'patterns': {}
        }

        # Analyze each column
        for col in df.columns:
            col_name = str(col).strip()
            structure['columns'].append(col_name)

            # Determine data type
            sample_values = df[col].dropna().head(10).tolist()
            data_type = self._infer_column_type(sample_values)
            structure['data_types'][col_name] = data_type

            # Count missing values
            missing_count = df[col].isna().sum()
            structure['missing_values'][col_name] = missing_count

            # Get sample data
            structure['sample_data'][col_name] = [str(val) for val in sample_values[:5]]

            # Detect patterns
            structure['patterns'][col_name] = self._detect_column_patterns(sample_values, data_type)

        return structure

    def _infer_column_type(self, sample_values: List) -> str:
        """Infer the data type of a column based on sample values."""
        if not sample_values:
            return 'empty'

        # Check for dates
        date_count = 0
        for val in sample_values:
            if pd.isna(val):
                continue
            try:
                pd.to_datetime(str(val))
                date_count += 1
            except:
                pass

        if date_count > len(sample_values) * 0.5:
            return 'date'

        # Check for numeric values
        numeric_count = 0
        for val in sample_values:
            if pd.isna(val):
                continue
            try:
                float(str(val).replace(',', '').replace('.', '').replace('R$', '').replace(' ', ''))
                numeric_count += 1
            except:
                pass

        if numeric_count > len(sample_values) * 0.5:
            return 'numeric'

        # Check for boolean values
        bool_keywords = ['sim', 'não', 'yes', 'no', 'true', 'false', '1', '0']
        bool_count = 0
        for val in sample_values:
            if pd.isna(val):
                continue
            if str(val).lower().strip() in bool_keywords:
                bool_count += 1

        if bool_count > len(sample_values) * 0.5:
            return 'boolean'

        return 'text'

    def _detect_column_patterns(self, sample_values: List, data_type: str) -> Dict[str, Any]:
        """Detect patterns in column data."""
        patterns = {
            'has_currency_symbol': False,
            'has_percentage': False,
            'has_parentheses': False,
            'common_prefixes': [],
            'common_suffixes': [],
            'unique_values_ratio': 0
        }

        if not sample_values:
            return patterns

        # Convert to strings for analysis
        str_values = [str(val) for val in sample_values if not pd.isna(val)]

        if not str_values:
            return patterns

        # Check for currency symbols
        currency_symbols = ['R$', '$', '€', '£', '¥']
        for val in str_values:
            for symbol in currency_symbols:
                if symbol in val:
                    patterns['has_currency_symbol'] = True
                    break
            if patterns['has_currency_symbol']:
                break

        # Check for percentages
        if any('%' in val for val in str_values):
            patterns['has_percentage'] = True

        # Check for parentheses (often used for negative numbers)
        if any('(' in val and ')' in val for val in str_values):
            patterns['has_parentheses'] = True

        # Calculate unique values ratio
        unique_values = set(str(val).lower().strip() for val in str_values)
        patterns['unique_values_ratio'] = len(unique_values) / len(str_values)

        # Detect common prefixes/suffixes for text columns
        if data_type == 'text' and len(str_values) > 3:
            prefixes = []
            suffixes = []

            for val in str_values:
                if len(val) > 3:
                    prefixes.append(val[:3])
                    suffixes.append(val[-3:])

            if prefixes:
                from collections import Counter
                prefix_counts = Counter(prefixes)
                common_prefixes = [prefix for prefix, count in prefix_counts.items() if count > 1]
                patterns['common_prefixes'] = common_prefixes[:3]

            if suffixes:
                suffix_counts = Counter(suffixes)
                common_suffixes = [suffix for suffix, count in suffix_counts.items() if count > 1]
                patterns['common_suffixes'] = common_suffixes[:3]

        return patterns

    def _determine_file_type(self, df: pd.DataFrame, structure: Dict[str, Any]) -> str:
        """Determine if the file is Bank Statement or Company Financial Data."""
        # Bank statement indicators
        bank_indicators = [
            'saldo', 'balance', 'extrato', 'statement', 'banco', 'bank',
            'agencia', 'agency', 'conta', 'account', 'transacao', 'transaction',
            'credito', 'debito', 'credit', 'debit', 'transferencia', 'transfer'
        ]

        # Company financial data indicators
        company_indicators = [
            'categoria', 'category', 'centro de custo', 'cost center',
            'departamento', 'department', 'projeto', 'project',
            'tipo', 'type', 'observacoes', 'observations', 'relatorio', 'report'
        ]

        # Analyze column names
        column_names = [col.lower() for col in structure['columns']]
        bank_score = 0
        company_score = 0

        for col in column_names:
            for indicator in bank_indicators:
                if indicator in col:
                    bank_score += 1
                    break

            for indicator in company_indicators:
                if indicator in col:
                    company_score += 1
                    break

        # Analyze data patterns
        numeric_columns = [col for col, dtype in structure['data_types'].items() if dtype == 'numeric']
        date_columns = [col for col, dtype in structure['data_types'].items() if dtype == 'date']

        # Bank statements typically have balance columns and transaction amounts
        balance_indicators = ['saldo', 'balance']
        balance_columns = [col for col in column_names if any(ind in col for ind in balance_indicators)]

        # Company data often has more categorical columns
        categorical_indicators = ['categoria', 'tipo', 'departamento', 'projeto']
        categorical_columns = [col for col in column_names if any(ind in col for ind in categorical_indicators)]

        # Additional scoring based on structure
        if balance_columns:
            bank_score += 2
        if categorical_columns:
            company_score += 2

        # Check for transaction patterns (mixed positive/negative amounts)
        if numeric_columns:
            sample_numeric_data = []
            for col in numeric_columns[:3]:  # Check first 3 numeric columns
                values = df[col].dropna()
                if len(values) > 10:
                    sample_numeric_data.extend(values.head(20).tolist())

            if sample_numeric_data:
                positive_count = 0
                negative_count = 0

                for val in sample_numeric_data:
                    try:
                        # Convert to float for comparison
                        num_val = float(val) if not isinstance(val, (int, float)) else val
                        if num_val > 0:
                            positive_count += 1
                        elif num_val < 0:
                            negative_count += 1
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        continue

                # If we have both positive and negative values, likely bank statement
                if positive_count > 0 and negative_count > 0:
                    bank_score += 1

        # Final decision
        if bank_score > company_score:
            return 'Bank Statement'
        elif company_score > bank_score:
            return 'Company Financial Data'
        else:
            # If scores are equal, check for specific patterns
            if balance_columns and len(numeric_columns) >= 2:
                return 'Bank Statement'
            elif categorical_columns and len(date_columns) >= 1:
                return 'Company Financial Data'
            else:
                return 'Unknown'

    def _generate_detailed_analysis(self, df: pd.DataFrame, structure: Dict[str, Any], file_type: str) -> Dict[str, Any]:
        """Generate detailed analysis based on file type."""
        analysis = {
            'decision_factors': [],
            'column_analysis': {},
            'data_quality': {},
            'recommendations': []
        }

        # Decision factors
        if file_type == 'Bank Statement':
            analysis['decision_factors'] = [
                'Presence of balance/transaction columns',
                'Mixed positive and negative amounts',
                'Bank-related terminology in column names'
            ]
        elif file_type == 'Company Financial Data':
            analysis['decision_factors'] = [
                'Presence of category/cost center columns',
                'Structured categorization fields',
                'Company financial terminology'
            ]
        else:
            analysis['decision_factors'] = [
                'Unclear data structure',
                'Mixed or ambiguous column types',
                'Insufficient distinguishing features'
            ]

        # Column analysis
        for col, dtype in structure['data_types'].items():
            col_analysis = {
                'type': dtype,
                'missing_percentage': (structure['missing_values'][col] / len(df)) * 100,
                'patterns': structure['patterns'][col]
            }
            analysis['column_analysis'][col] = col_analysis

        # Data quality assessment
        total_missing = sum(structure['missing_values'].values())
        missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100

        analysis['data_quality'] = {
            'overall_completeness': 100 - missing_percentage,
            'columns_with_missing_data': len([col for col, count in structure['missing_values'].items() if count > 0]),
            'total_missing_values': total_missing
        }

        # Recommendations
        if missing_percentage > 20:
            analysis['recommendations'].append('High percentage of missing data - consider data cleaning')

        if len(structure['columns']) < 3:
            analysis['recommendations'].append('Limited number of columns - may need additional data fields')

        if file_type == 'Unknown':
            analysis['recommendations'].append('File type could not be determined - manual review recommended')

        return analysis

    def _analyze_data_quality(self, financial_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze data quality of processed XLSX financial data

        Args:
            financial_data: List of processed financial data dictionaries

        Returns:
            Dictionary with data quality analysis results
        """
        try:
            logger.info(f"Analyzing data quality for {len(financial_data)} XLSX entries")

            # Convert to DataFrame for analysis
            df = pd.DataFrame(financial_data)

            # Perform completeness analysis
            completeness_report = self.completeness_analyzer.generate_completeness_report(df)

            # Get imputation recommendations
            recommendations = self.missing_data_handler.get_imputation_recommendations(df)

            # Check for critical data quality issues specific to financial data
            critical_issues = self._identify_critical_financial_issues(completeness_report, df)

            # Auto-apply imputation if confidence is high
            imputation_applied = False
            imputed_data = None

            if self._should_auto_impute_financial_data(completeness_report):
                try:
                    logger.info("Auto-applying imputation for financial data quality improvement")
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
                'data_quality_score': self._calculate_financial_data_quality_score(completeness_report),
                'processing_timestamp': datetime.now().isoformat()
            }

            if imputation_applied and imputed_data is not None:
                quality_result['imputed_financial_data'] = imputed_data.to_dict('records')

            logger.info(f"Financial data quality analysis completed. Score: {quality_result['data_quality_score']:.3f}")
            return quality_result

        except Exception as e:
            logger.error(f"Error in financial data quality analysis: {str(e)}")
            return {
                'error': str(e),
                'completeness_analysis': None,
                'recommendations': [],
                'critical_issues': [{'type': 'analysis_error', 'message': str(e)}],
                'data_quality_score': 0.0
            }

    def _identify_critical_financial_issues(self, completeness_report: Dict[str, Any], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify critical data quality issues specific to financial data

        Args:
            completeness_report: Completeness analysis results
            df: DataFrame with financial data

        Returns:
            List of critical financial issues
        """
        critical_issues = []

        try:
            # Check critical financial fields
            critical_fields = ['date', 'amount', 'description']
            for field in critical_fields:
                if field in completeness_report.get('field_completeness', {}):
                    completeness = completeness_report['field_completeness'][field]['completeness_score']
                    if completeness < 0.9:  # Less than 90% complete for critical financial fields
                        critical_issues.append({
                            'type': 'critical_financial_field_incomplete',
                            'field': field,
                            'completeness': completeness,
                            'severity': 'high',
                            'message': f"Critical financial field '{field}' has low completeness ({completeness:.1%})"
                        })

            # Check for amount anomalies
            if 'amount' in df.columns:
                amounts = df['amount'].dropna()
                if len(amounts) > 0:
                    # Check for extreme outliers
                    q1, q3 = amounts.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr

                    extreme_outliers = amounts[(amounts < lower_bound) | (amounts > upper_bound)]
                    if len(extreme_outliers) > len(amounts) * 0.05:  # More than 5%
                        critical_issues.append({
                            'type': 'amount_anomalies',
                            'affected_records': len(extreme_outliers),
                            'percentage': len(extreme_outliers) / len(amounts),
                            'severity': 'medium',
                            'message': f"High number of extreme amount outliers detected ({len(extreme_outliers)} records)"
                        })

            # Check for missing transaction types
            if 'transaction_type' in df.columns:
                missing_types = df['transaction_type'].isnull().sum()
                if missing_types > len(df) * 0.1:  # More than 10%
                    critical_issues.append({
                        'type': 'missing_transaction_types',
                        'affected_records': missing_types,
                        'percentage': missing_types / len(df),
                        'severity': 'medium',
                        'message': f"High number of missing transaction types ({missing_types} records)"
                    })

            # Check for data consistency issues
            if len(df) > 0:
                # Check for transactions with zero amount
                if 'amount' in df.columns:
                    zero_amounts = (df['amount'].fillna(0) == 0).sum()
                    if zero_amounts > len(df) * 0.05:  # More than 5%
                        critical_issues.append({
                            'type': 'zero_amount_transactions',
                            'affected_records': zero_amounts,
                            'percentage': zero_amounts / len(df),
                            'severity': 'low',
                            'message': f"High number of zero-amount transactions ({zero_amounts} records)"
                        })

        except Exception as e:
            logger.warning(f"Error identifying critical financial issues: {str(e)}")

        return critical_issues

    def _should_auto_impute_financial_data(self, completeness_report: Dict[str, Any]) -> bool:
        """
        Determine if auto-imputation should be applied for financial data

        Args:
            completeness_report: Completeness analysis results

        Returns:
            Boolean indicating if auto-imputation should be applied
        """
        try:
            overall_completeness = completeness_report.get('dataset_completeness', 1.0)

            # For financial data, be more conservative with auto-imputation
            if overall_completeness < 0.75:
                return False  # Too much missing data for financial records

            if overall_completeness > 0.95:
                return False  # Very complete data, no need for imputation

            # Check critical financial fields
            critical_fields = ['date', 'amount', 'description']
            for field in critical_fields:
                if field in completeness_report.get('field_completeness', {}):
                    field_completeness = completeness_report['field_completeness'][field]['completeness_score']
                    if field_completeness < 0.8:  # Critical field too incomplete
                        return False

            return True

        except Exception as e:
            logger.warning(f"Error determining auto-imputation for financial data: {str(e)}")
            return False

    def _calculate_financial_data_quality_score(self, completeness_report: Dict[str, Any]) -> float:
        """
        Calculate data quality score specifically for financial data

        Args:
            completeness_report: Completeness analysis results

        Returns:
            Financial data quality score between 0 and 1
        """
        try:
            base_score = completeness_report.get('dataset_completeness', 0.0)

            # Apply financial-specific adjustments
            financial_penalty = 0.0

            # Higher penalty for missing critical financial fields
            critical_fields = ['date', 'amount', 'description']
            for field in critical_fields:
                if field in completeness_report.get('field_completeness', {}):
                    field_score = completeness_report['field_completeness'][field]['completeness_score']
                    if field_score < 0.9:
                        financial_penalty += (0.9 - field_score) * 0.3  # 30% penalty per critical field

            # Penalty for missing transaction types
            if 'transaction_type' in completeness_report.get('field_completeness', {}):
                type_score = completeness_report['field_completeness']['transaction_type']['completeness_score']
                if type_score < 0.8:
                    financial_penalty += (0.8 - type_score) * 0.1  # 10% penalty

            quality_score = max(0.0, base_score - financial_penalty)

            return round(quality_score, 3)

        except Exception as e:
            logger.warning(f"Error calculating financial data quality score: {str(e)}")
            return 0.0