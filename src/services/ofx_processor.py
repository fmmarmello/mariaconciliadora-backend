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
                        # Preserve full datetime as 'timestamp' (ISO 8601) and keep 'date' for DB
                        'timestamp': transaction.date.isoformat() if hasattr(transaction, 'date') and getattr(transaction, 'date') else None,
                        'date': transaction.date.date() if hasattr(transaction, 'date') and getattr(transaction, 'date') else None,
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
        
        # Remove códigos de transação comuns no início, somente quando forem tokens inteiros
        # Evita remover prefixos parciais (ex.: não remover "TRANSF" dentro de "TRANSFERENCIA")
        pattern = r'^(?:TED|DOC|PIX|TRANSF|SAQUE|DEPOSITO|DEPÓSITO|COMPRA)\b(?:\s*[-:–—]\s*)?'
        description = re.sub(pattern, '', description, flags=re.IGNORECASE)
        
        return description
    
    @handle_service_errors('ofx_processor')
    def validate_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """
        Valida e limpa os dados das transações
        """
        logger.info(f"Validating {len(transactions)} transactions")
        
        if not transactions:
            raise InsufficientDataError('transaction validation', 1, 0)
        
        valid_transactions = []
        invalid_transactions = []
        
        for i, transaction in enumerate(transactions):
            try:
                # Use the transaction validator
                validation_result = transaction_validator.validate_transaction(transaction)
                
                if validation_result.is_valid:
                    # Clean and normalize transaction data
                    cleaned_transaction = self._clean_transaction_data(transaction)
                    valid_transactions.append(cleaned_transaction)
                else:
                    logger.warning(f"Transaction {i+1} validation failed: {validation_result.errors}")
                    invalid_transactions.append({
                        'index': i,
                        'transaction': transaction,
                        'errors': validation_result.errors
                    })
                    
            except Exception as e:
                logger.error(f"Error validating transaction {i+1}: {str(e)}")
                invalid_transactions.append({
                    'index': i,
                    'transaction': transaction,
                    'errors': [str(e)]
                })
        
        # Log validation summary
        logger.info(f"Transaction validation completed. Valid: {len(valid_transactions)}, Invalid: {len(invalid_transactions)}")
        
        if invalid_transactions:
            logger.warning(f"Invalid transactions details: {invalid_transactions[:5]}")  # Log first 5
        
        if not valid_transactions:
            raise InsufficientDataError('transaction validation', 1, 0)
        
        return valid_transactions
    
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

