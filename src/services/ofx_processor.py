import ofxparse
from datetime import datetime
from typing import List, Dict, Any
import re

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
    
    def parse_ofx_file(self, file_path: str) -> Dict[str, Any]:
        """
        Processa um arquivo OFX e retorna os dados estruturados
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                ofx_content = file.read()
        except UnicodeDecodeError:
            # Tenta com encoding latin-1 se UTF-8 falhar
            with open(file_path, 'r', encoding='latin-1') as file:
                ofx_content = file.read()
        
        # Identifica o banco
        bank_name = self.identify_bank(ofx_content)
        
        # Parse do OFX
        with open(file_path, 'rb') as file:
            ofx = ofxparse.OfxParser.parse(file)
        
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
        
        return result
    
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
    
    def validate_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """
        Valida e limpa os dados das transações
        """
        valid_transactions = []
        
        for transaction in transactions:
            # Validações básicas
            if not transaction.get('date'):
                continue
            
            if transaction.get('amount') is None or transaction.get('amount') == 0:
                continue
            
            if not transaction.get('description'):
                transaction['description'] = 'Transação sem descrição'
            
            valid_transactions.append(transaction)
        
        return valid_transactions
    
    def detect_duplicates(self, transactions: List[Dict]) -> List[int]:
        """
        Detecta possíveis transações duplicadas
        """
        duplicates = []
        seen = set()
        
        for i, transaction in enumerate(transactions):
            # Cria uma chave única baseada em data, valor e descrição
            key = (
                transaction.get('date'),
                transaction.get('amount'),
                transaction.get('description', '').lower().strip()
            )
            
            if key in seen:
                duplicates.append(i)
            else:
                seen.add(key)
        
        return duplicates

