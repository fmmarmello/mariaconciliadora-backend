import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import re
import unicodedata
import os
from src.services.duplicate_detection_service import DuplicateDetectionService
import hashlib
from src.services.ai_service import AIService
from src.utils.logging_config import get_logger
from src.utils.exceptions import (
    FileProcessingError, FileNotFoundError, InvalidFileFormatError,
    FileCorruptedError, ValidationError, InsufficientDataError
)
from src.utils.error_handler import handle_service_errors, with_timeout, recovery_manager
from src.utils.validators import validate_file_upload, company_financial_validator
from src.constants.financial import normalize_company_financial_category

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
        # Canonical targets expected downstream in row processing
        self._canonical_targets = {
            'date': 'data',
            'description': 'description',
            'amount': 'valor',
            'transaction_type': 'tipo',
            'category': 'category',
            'cost_center': 'cost_center',
            'department': 'department',
            'project': 'project',
            'observations': 'observations',
            'monthly_report_value': 'monthly_report_value'
        }
    
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
            # Read ALL sheets with error handling and merge the valid ones
            sheets = self._read_all_sheets_with_recovery(file_path)
            logger.info(f"XLSX workbook loaded. Sheets: {list(sheets.keys())}")

            df = self._select_and_merge_sheets(sheets, filename)
            logger.info(f"Merged DataFrame. Rows: {len(df)}, Columns: {len(df.columns)}")
            
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

            # Optional: Use AI to infer transaction type as assist (sign still prevails)
            try:
                if financial_data:
                    ai_service = AIService()
                    financial_data = ai_service.infer_transaction_type_batch(financial_data)
                    # Apply AI inference only for zero-amount lines (sign-first precedence)
                    for entry in financial_data:
                        try:
                            amt = float(entry.get('amount') or 0)
                        except Exception:
                            amt = 0.0
                        if amt == 0 and entry.get('transaction_type_ai'):
                            entry['transaction_type'] = entry['transaction_type_ai']
            except Exception as ai_err:
                logger.warning(f"AI type inference skipped due to error: {str(ai_err)}")
            
            # Validate processed data
            self._validate_processed_data(financial_data, filename)
            
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

    def _read_all_sheets_with_recovery(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Read all sheets from an XLSX workbook with recovery mechanisms."""
        def primary_read_all():
            return pd.read_excel(file_path, sheet_name=None)

        def fallback_read_all():
            logger.warning("Primary XLSX multi-sheet reading failed, attempting recovery")
            engines = ['openpyxl', 'xlrd']
            for engine in engines:
                try:
                    logger.debug(f"Trying to read all sheets with {engine} engine")
                    return pd.read_excel(file_path, sheet_name=None, engine=engine)
                except Exception as e:
                    logger.debug(f"Failed with {engine}: {str(e)}")
                    continue
            # If all engines fail, escalate
            raise FileCorruptedError(os.path.basename(file_path))

        return recovery_manager.fallback_on_error(primary_read_all, fallback_read_all)

    def _build_canonical_rename_map(self, df: pd.DataFrame) -> Dict[str, str]:
        """Create a rename map from incoming columns to canonical names based on synonyms."""
        rename_map: Dict[str, str] = {}
        normalized_cols = {col: self._normalize_column_name(col) for col in df.columns}

        for original, norm_col in normalized_cols.items():
            mapped = None
            # Try to match against each supported standard and its variations
            for standard, variations in self.supported_columns.items():
                variations_norm = [self._normalize_column_name(v) for v in variations]
                target = self._canonical_targets.get(standard, standard)
                # Also consider the canonical target as a variation
                candidates = set(variations_norm + [self._normalize_column_name(target)])
                if any(v in norm_col or norm_col in v for v in candidates):
                    mapped = target
                    break

            # Fallback to normalized column name if not mapped
            rename_map[original] = mapped if mapped else norm_col

        # Resolve collisions by keeping first occurrence; suffix others
        seen: Dict[str, int] = {}
        final_map: Dict[str, str] = {}
        for orig, target in rename_map.items():
            if target in seen:
                seen[target] += 1
                final_map[orig] = f"{target}_{seen[target]}"
            else:
                seen[target] = 0
                final_map[orig] = target

        return final_map

    def _select_and_merge_sheets(self, sheets: Dict[str, pd.DataFrame], filename: str) -> pd.DataFrame:
        """Select sheets that contain expected columns and merge them into a single DataFrame."""
        valid_frames: List[pd.DataFrame] = []
        included_sheets: List[str] = []
        skipped_sheets: List[str] = []

        required_keys = ['data', 'valor', 'description']  # canonical expected

        for sheet_name, df in sheets.items():
            try:
                if df is None or df.empty or len(df.columns) < 2:
                    skipped_sheets.append(sheet_name)
                    continue

                # Rename columns to canonical using synonyms
                rename_map = self._build_canonical_rename_map(df)
                df_renamed = df.rename(columns=rename_map)
                # Normalize canonical names (idempotent)
                df_renamed.columns = [self._normalize_column_name(c) for c in df_renamed.columns]

                # Determine if sheet contains required columns
                cols = set(df_renamed.columns)
                if all(key in cols for key in required_keys):
                    valid_frames.append(df_renamed)
                    included_sheets.append(sheet_name)
                else:
                    skipped_sheets.append(sheet_name)
            except Exception as e:
                logger.warning(f"Error processing sheet '{sheet_name}': {str(e)}")
                skipped_sheets.append(sheet_name)

        logger.info(f"Included sheets: {included_sheets}; Skipped sheets: {skipped_sheets}")

        if valid_frames:
            return pd.concat(valid_frames, ignore_index=True, sort=False)

        # Fallback: use first non-empty sheet with normalized columns (maintains backward compatibility)
        for sheet_name, df in sheets.items():
            if df is not None and not df.empty:
                rename_map = self._build_canonical_rename_map(df)
                df_renamed = df.rename(columns=rename_map)
                df_renamed.columns = [self._normalize_column_name(c) for c in df_renamed.columns]
                logger.warning(
                    f"No sheets matched required columns in {filename}. Falling back to first non-empty sheet: {sheet_name}"
                )
                return df_renamed

        # If everything is empty
        raise InsufficientDataError('XLSX parsing', 1, 0)
    
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
            # Compute type diagnostics
            raw_tipo = row.get('tipo', '')
            mapped_label = self._map_label_to_type(raw_tipo)
            amt_num = self._parse_amount(row.get('valor'))
            sign_type = 'expense' if amt_num < 0 else ('income' if amt_num > 0 else '')
            final_type = self._determine_transaction_type(row)
            type_conflict = bool(sign_type and mapped_label and sign_type != mapped_label)

            entry = {
                'date': self._parse_date(row.get('data')),
                'description': self._parse_description(row.get('description', '')),
                'amount': self._parse_amount(row.get('valor')),
                'category': normalize_company_financial_category(row.get('tipo', '')),
                'cost_center': str(row.get('cost_center', '')),
                'department': str(row.get('department', '')),
                'project': str(row.get('project', '')),
                'transaction_type': final_type,
                'observations': str(row.get('observations', '')),
                'monthly_report_value': self._parse_amount(row.get('monthly_report_value')),
                # Diagnostics for UI/reporting
                'transaction_type_label': mapped_label if mapped_label else None,
                'transaction_type_conflict': type_conflict,
                'tipo_raw': str(raw_tipo) if raw_tipo is not None else ''
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
        # Lowercase
        column_name = str(column_name).strip().lower()
        # Remove diacritics (accents)
        column_name = ''.join(
            ch for ch in unicodedata.normalize('NFD', column_name)
            if unicodedata.category(ch) != 'Mn'
        )
        # Remove non-word characters except whitespace
        column_name = re.sub(r'[^\w\s]', '', column_name)
        # Collapse inner whitespace
        column_name = re.sub(r'\s+', ' ', column_name).strip()
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
        """Determina se é despesa ou receita, priorizando o sinal do valor."""
        # 1) Ground truth pelo sinal do valor
        amount = self._parse_amount(row.get('valor'))
        try:
            if amount < 0:
                return 'expense'
            if amount > 0:
                return 'income'
        except Exception:
            pass

        # 2) Fallback pelo conteúdo do campo 'tipo'
        mapped = self._map_label_to_type(str(row.get('tipo', '')))
        if mapped:
            return mapped

        # 3) Fallback padrão
        return 'expense'

    def _map_label_to_type(self, label: str) -> str:
        """Mapeia o texto do campo 'tipo' para 'income'/'expense'. Retorna '' se indeterminado."""
        lbl = str(label or '').strip().lower()
        # Normaliza acentos
        lbl = ''.join(ch for ch in unicodedata.normalize('NFD', lbl) if unicodedata.category(ch) != 'Mn')
        expense_terms = {
            'despesa', 'expense', 'debito', 'retirada socio', 'impostos / tributos', 'impostos', 'tributos',
            'tarifas bancarias', 'juros / multa', 'juros', 'multa', 'seguro', 'emprestimo', 'pagamento', 'compra',
            'fatura', 'taxa', 'tarifa', 'saque'
        }
        income_terms = {
            'reembolso', 'receita', 'income', 'credito', 'deposito', 'aporte', 'venda', 'vendas', 'recebimento', 'salario'
        }
        if lbl in expense_terms:
            return 'expense'
        if lbl in income_terms:
            return 'income'
        return ''
    
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
        Analyze XLSX workbook structure and content to determine type and provide detailed analysis.
        - Inspects all sheets
        - Picks a primary sheet for top-level fields (structure/analysis/summary)
        - Returns per-sheet details under 'sheets'
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
            # Read ALL sheets in the workbook
            sheets = self._read_all_sheets_with_recovery(file_path)
            logger.info(f"XLSX workbook loaded for analysis. Sheets: {list(sheets.keys())}")

            # Build per-sheet analysis
            per_sheet_details = []
            required_keys = ['data', 'valor', 'description']

            def prepare_df_for_sheet(df):
                rename_map = self._build_canonical_rename_map(df)
                df2 = df.rename(columns=rename_map)
                df2.columns = [self._normalize_column_name(c) for c in df2.columns]
                return df2

            for sheet_name, raw_df in sheets.items():
                if raw_df is None or raw_df.empty:
                    per_sheet_details.append({
                        'sheet_name': sheet_name,
                        'rows': 0,
                        'columns': 0,
                        'skipped': True,
                        'reason': 'empty'
                    })
                    continue
                try:
                    df_sheet = prepare_df_for_sheet(raw_df)
                    structure = self._analyze_file_structure(df_sheet, filename)
                    ftype = self._determine_file_type(df_sheet, structure)
                    analysis = self._generate_detailed_analysis(df_sheet, structure, ftype)
                    per_sheet_details.append({
                        'sheet_name': sheet_name,
                        'rows': int(len(df_sheet)),
                        'columns': int(len(df_sheet.columns)),
                        'file_type': ftype,
                        'structure': structure,
                        'analysis': analysis,
                        'has_required_columns': all(k in set(df_sheet.columns) for k in required_keys)
                    })
                except Exception as e:
                    logger.warning(f"Error analyzing sheet '{sheet_name}': {str(e)}")
                    per_sheet_details.append({
                        'sheet_name': sheet_name,
                        'rows': int(len(raw_df)) if raw_df is not None else 0,
                        'columns': int(len(raw_df.columns)) if raw_df is not None else 0,
                        'error': str(e)
                    })

            # Choose primary sheet: prefer sheet with required columns and most rows; else largest sheet
            primary = None
            candidates = [s for s in per_sheet_details if s.get('has_required_columns')]
            if candidates:
                primary = max(candidates, key=lambda s: s['rows'])
            else:
                non_empty = [s for s in per_sheet_details if s.get('rows', 0) > 0]
                if non_empty:
                    primary = max(non_empty, key=lambda s: s['rows'])

            if not primary:
                raise InsufficientDataError('XLSX analysis', 1, 0)

            # Top-level fields use the primary sheet for backward compatibility
            file_type = primary.get('file_type', 'Unknown')
            analysis_result = primary.get('structure', {})
            detailed_analysis = primary.get('analysis', {})

            logger.info(f"XLSX analysis completed. Primary sheet: {primary['sheet_name']} Type: {file_type}")

            # Convert pandas/numpy types to Python native types for JSON serialization
            def convert_to_native_types(obj):
                if isinstance(obj, dict):
                    return {key: convert_to_native_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native_types(item) for item in obj]
                elif hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                else:
                    return obj

            response = {
                'file_type': file_type,
                'analysis': convert_to_native_types(detailed_analysis),
                'structure': convert_to_native_types(analysis_result),
                'summary': {
                    'total_rows': int(primary.get('rows', 0)),
                    'total_columns': int(primary.get('columns', 0)),
                    'filename': filename,
                    'file_size': int(os.path.getsize(file_path)),
                    'primary_sheet': primary.get('sheet_name')
                },
                'sheets': convert_to_native_types(per_sheet_details)
            }

            return response

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
