import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import re

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
    
    def parse_xlsx_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Processa um arquivo XLSX e retorna os dados estruturados
        """
        try:
            # Lê o arquivo XLSX
            df = pd.read_excel(file_path)
            
            # Normaliza os nomes das colunas
            df.columns = [self._normalize_column_name(col) for col in df.columns]
            
            # Converte para lista de dicionários
            financial_data = []
            for _, row in df.iterrows():
                entry = {
                    'date': self._parse_date(row.get('date')),
                    'description': str(row.get('description', '')),
                    'amount': self._parse_amount(row.get('amount')),
                    'category': str(row.get('category', '')),
                    'cost_center': str(row.get('cost_center', '')),
                    'department': str(row.get('department', '')),
                    'project': str(row.get('project', '')),
                    'transaction_type': self._determine_transaction_type(row),
                    'observations': str(row.get('observations', '')),
                    'monthly_report_value': self._parse_amount(row.get('monthly_report_value'))
                }
                if 'saldo' in entry['description'].lower():
                    continue
                financial_data.append(entry)
            
            return financial_data
            
        except Exception as e:
            raise Exception(f"Erro ao processar arquivo XLSX: {str(e)}")
    
    def _normalize_column_name(self, column_name: str) -> str:
        """Normaliza nomes de colunas para padrão consistente"""
        column_name = str(column_name).strip().lower()
        # Remove acentos e caracteres especiais
        column_name = re.sub(r'[^\w\s]', '', column_name)
        return column_name
    
    def _parse_date(self, date_value) -> datetime:
        """Parse de datas em vários formatos"""
        if pd.isna(date_value):
            return None
        try:
            return pd.to_datetime(date_value)
        except:
            return None
    
    def _parse_amount(self, amount_value) -> float:
        """Parse de valores monetários"""
        if pd.isna(amount_value):
            return 0.0
        try:
            return float(amount_value)
        except:
            return 0.0
    
    def _determine_transaction_type(self, row) -> str:
        """Determina se é despesa ou receita"""
        transaction_type = row.get('transaction_type', '').lower()
        if transaction_type in ['despesa', 'expense', 'débito', 'debit', 'retirada sócio',  'impostos / tributos', 'tarifas bancárias', 'juros / multa', 'seguro', 'emprestimo']:
            return 'expense'
        elif transaction_type in ['reembolso','receita', 'income', 'crédito', 'credit','credito']:
            return 'income'
        else:
            # Determina pelo valor (negativo = despesa, positivo = receita)
            amount = self._parse_amount(row.get('amount'))
            return 'expense' if amount < 0 else 'income'