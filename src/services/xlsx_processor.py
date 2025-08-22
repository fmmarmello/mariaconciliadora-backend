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
            logger.debug(f"Column normalization completed. Original: {original_columns}")
            
            # Process rows with validation
            financial_data = self._process_xlsx_rows(df)
            
            # Validate processed data
            self._validate_processed_data(financial_data, filename)
            
            logger.info(f"XLSX parsing completed successfully. Processed: {len(financial_data)} entries")
            return financial_data
            
        except pd.errors.EmptyDataError:
            raise FileCorruptedError(filename)
        except pd.errors.ExcelFileError:
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
        """Process a single XLSX row with validation."""
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
            
            # Validate the entry using company financial validator
            validation_result = company_financial_validator.validate_entry(entry)
            
            if not validation_result.is_valid:
                logger.debug(f"Row {index + 1} validation failed: {validation_result.errors}")
                # Return None for invalid entries - they will be skipped
                return None
            
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